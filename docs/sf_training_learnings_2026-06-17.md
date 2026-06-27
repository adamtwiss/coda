# SF training learnings for Coda (2026-06-17)

Deep comparison of Stockfish's NNUE **training** recipe against Coda's, to
find portable, actionable training improvements. Grounded in: vondele's
`optimize/optimize.py` (github.com/vondele/nettest — SF's training lead's
recipe-optimization harness), the official nnue-pytorch wiki
(`official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html`),
`docs/sfnnv13_architecture_review_2026-05-23.md`,
`docs/sf_recipe_experiments_2026-05-22.md`,
`docs/eval_blindspot_training_fix_2026-06-17.md`, `experiments.md`, and the
Bullet fork `coda_v9_768_threats.rs`.

## TL;DR

**Coda has already ported AND tested almost the entire SF NNUE training
recipe** — WDL ramp (5× H0), lambda-jitter (+7.9 with retune), SWA (+5 done
right), multistage (S4 regressed), fen-skip, batch-scaling, factoriser,
mse-power. The Bullet fork has CLI flags for all of them and SPRT verdicts in
`experiments.md`. So the high-value learnings left are **not** "port another
flag." They are:

1. **Learned eval calibration** (in/out-scaling + in/out-offset) — the one SF
   training mechanism never ported; Coda hardcodes a single `EVAL_SCALE=400`.
2. **`ft_optimize`** — feature-transformer optimization/reordering; absent from
   Coda's pipeline.
3. **The targeted-data blindspot loop** — convert *measured* deployment loss
   classes into eval fixes (this is the highest-leverage training item, and it
   is exactly what the current re-wire + SF-labelled datagen flywheel is doing).
4. **The meta-process** — SF auto-tunes the recipe directly against **nElo**
   (nevergrad/TBPSA), never loss; Coda sweeps one axis at a time by hand.

Most raw flag-ports are **spent levers**. The recipe is essentially fully
ported and mostly tested.

## Coda's architecture + recipe (verified against code)

- **Architecture (v9, prod `549C20A5`):**
  `(768×N PSQ + 66,864 threats → FT=1024)×2 → CReLU → pairwise_mul → 512×2 →
  L1=16 → L2=32 → 1×8 buckets`. N=10 (Reckless kb layout).
- **FT width = 1024 = SF's "L1"** — SF names the Feature Transformer "L1"; it
  is the same 1024 width as ours. **No FT-width gap with SF.** Coda's L1=16 vs
  SF's first dense hidden (32) is the one real width gap; L1=32 is structurally
  done but NPS-taxed + regressed at S800 (#1509/#1506).
- **Quantization:** QA=255 (i16 accumulator), QB=64 (i8 L1). Matches SF. Parity.
- **Loss:** `output.sigmoid().power_error(target, 3.0)` — **mse-power 3.0** in
  prod (validated > 2.4 at STC *and* LTC, #2032/#2033). SF docs use 2.6;
  **Coda is already past SF's exponent.** Settled.
- **WDL:** fixed blend, 0.20 in prod `549C20A5` (0.15 in the S200 control). SF
  uses a *ramp* (λ_score 0.85 → 0.75).
- **LR:** linear warmup (~0.0001→0.001 over ~30 SB) → cosine to
  `final_lr ≈ 2.43e-6`. SF auto-tunes lr+gamma via nevergrad.
- **Eval calibration:** single hardcoded constant — `eval_scale=400.0`
  (trainer) + `EVAL_SCALE=400` (`nnue.rs:165`). SF learns **four** params
  (in/out scaling + in/out offset).
- **Data:** linrock T80 min-v2.v6 binpacks, filter `ply≥16 & !in_check &
  |score|≤10000 & quiet`, interleaved. Same binpacks SF/vondele use. No
  self-play in the loop.

## Ranked learnings (by expected Elo-per-effort)

### Tier 1 — genuinely unexplored, worth a probe

**0. Soft early-ply filtering — IMPLEMENTED 2026-06-17, pending S200 probe.**
SF separates *two* ply/skip mechanisms that Coda had conflated (correction to
the original framing of this doc):
- **`random_fen_skipping`** — *uniform* game-**decorrelation** ("stirring"), so
  you don't over-read one game's correlated chain. SF runs it *high*
  (`--random-fen-skipping=10` ⇒ skip ≈ 10/11 ≈ **91%**). Coda's
  `--fen-skip-prob` is the same mechanism at **0.5** — and that 0.5 is a
  **decoder-throughput floor, not a data choice**: interleaving already shuffles,
  and the Bullet binpack decoder is single-threaded in places, so skip >0.5
  makes it CPU-bound and starves the GPU. (A prior parallelization attempt —
  bullet branch `feature/parallel-binpack-reader` — just moved the bottleneck to
  the next single-threaded stage, so the *global* skip is genuinely capped until
  that's fixed. Parked, not abandoned.)
- **`soft_early_fen_skipping`** — a *ply-dependent accept curve*
  (`training_data_loader.cpp`, piecewise-linear through control points) applied
  *separately* from the random skip, to stop early-opening positions being
  over-**represented** (they recur across games). **Coda had nothing equivalent**
  — only the hard `ply>=16` cut + the uniform skip.

Implemented (bullet `feature/soft-early-ply`). Initial probe used a crude single
linear ramp (`--soft-early-ply 28 --soft-early-ply-floor 0.2/0.1`, OB SPRTs
2073/2074). Then ported SF's **actual** mechanism (commit b4a1c08).

**SF's REAL tuned values** (from `vondele/nettest/threats.yaml` + nnue-pytorch
`data_loader/`, fetched 2026-06-18 — supersedes earlier guesses in this doc):
- `early-fen-skipping: 18` (HARD cut, skip ply≤18; nnue-pytorch *default* is −1
  = off, but the threats run uses 18). Coda's hard cut is ply≥16.
- `soft-early-fen_skipping: 32` ⇒ **peak ply = 32** (accept reaches 1.0 there).
  Our 28 was below this. The soft curve is **5-point piecewise-LINEAR**
  `interpolate_ply` through (x1,y1)..(x4,y4)+(32,1.0).
- **Two stage curves** (matches the multi-stage ~3000-SB prod recipe):
  - *warmup*: `(0,.01)(14,.20)(18.5,.50)(29.5,.80)(32,1.0)` → after hard-cut-18,
    ~0.51 accept @ply19 → 1.0 @32.
  - *advanced*: `(0,.025)(22,.05)(25.5,.20)(29.5,.80)(32,1.0)` → brutal: ~5% @22,
    20% @25.5. Late training nearly drops sub-30-ply.
- **Piece-count rebalancer** (`pc-y0..y4 = -0.20,0.45,1.0,0.95,0.75`): a *Hermite*
  target spline at piece counts 0/8/16/24/32 + an **adaptive importance-resampler**
  (alpha=(1−0.975)/min_ratio). Targets peak ~16 pieces, suppresses deep endgames
  (≤8) and mildly the opening (32). **Coda had NO analog** — biggest new lever.

Both ported to bullet (commit b4a1c08): `--ply-x1..y4` (default = SF warmup),
`--soft-early-ply` = peak; `--pc-y0..y4` enables the rebalancer (default OFF).
Hard ply≥16 cut and `--fen-skip-prob` untouched. **Effort: done. Prior: med-high**
— ply filtering is a *known* 30-40 Elo-sensitive axis. **NB CPU cost (Adam
2026-06-18):** stacking soft-ply + pc + wld on top of fenskip=0.5 will make the
single-threaded decoder CPU-bound fast; fine for S200 probes, but **drop
`--fen-skip-prob` if we adopt this** beyond probes. (We have the full SF-sized
dataset on GPU4; constraint is SB/compute + decoder throughput, not data qty.)

**Other threats.yaml borrowables** (beyond ply/pc): `pow-exp 2.435` (MSE power —
LOWER than our 3.0), learned eval calibration `in/out-scaling 300/350`,
`in/out-offset 300/300` (vs single EVAL_SCALE=400 — see §1), WDL `start/end-lambda`
ramp (warmup 1.0→0.75, advanced 0.74 const + jitter), `qp-asymmetry 0.23`,
`wld_filtered` + `simple_eval_skipping` (extra skip stages Coda lacks).

**1. Learned eval calibration: in/out-scaling + in/out-offset (4 params) vs a
single hardcoded EVAL_SCALE=400.**
- *SF:* the eval→win-prob sigmoid uses scaling (~410 in docs); `optimize.py`
  exposes in-scaling, out-scaling, in-offset, out-offset as four separately
  nElo-tuned params. Offsets/asymmetric scaling correct systematic eval bias
  and the win/loss asymmetry.
- *Coda:* one constant; no offset, no separate in/out scaling.
- *Why it might matter:* Coda's loss analysis shows wins gradual (92%), losses
  stepped/sudden (`project_coda_winloss_asymmetry`) — exactly what an
  **out-offset** models. This is the one SF training mechanism with a real
  prior that has never been ported or tested.
- *Actionability:* needs a Bullet patch (~30-50 LoC in the loss path + plumb
  the inference-side offset into the `.nnue` header & `nnue.rs`). **Effort:
  medium. Prior: low-med** (T80 data is already LC0-WP-calibrated, so the bias
  may be small). **Validate:** S200 paired probe → S800 vs prod at `[-1.5,1.5]`.
  Start with **out-offset alone** (cheapest, targets the asymmetry directly).

**2. `ft_optimize` — feature-transformer optimization / reordering.**
- *SF:* a post-training `ft_optimize` pass (`--ft_optimize_count=100000`)
  reorders/prunes/optimizes FT weights.
- *Coda:* none (no `ft_optimize` anywhere in the Bullet fork); quantizes
  straight from the checkpoint.
- *Why it might matter:* the row-reorder variant improves FT cache residency.
  Coda is bandwidth-starved under concurrency
  (`project_coda_bandwidth_starvation_under_concurrency`); the threat-accumulator
  apply is ~31% of cycles vs SF's ~5.5%
  (`project_threat_index_microopts_neutral`). FT-row reordering is squarely in
  Coda's biggest measured NPS lever.
- *Actionability:* Bullet patch + inference support. **Effort: medium-high.
  Prior: low direct Elo, compounds as fleet throughput + ARM.** **Validate:**
  NPS bench on a non-memory-bound host (NOT Hercules —
  `feedback_hercules_bench_overstates_bandwidth_opts`), then `[-2,1]`.

**3. Targeted-data blindspot loop.**
- *SF:* strength flywheel is data-driven; recipe + data optimized against nElo.
- *Coda:* the methodology exists (`eval_blindspot_training_fix_2026-06-17.md`)
  and is **in flight now** (re-wire pass + SF-labelled Coda-vs-SF datagen).
- *Why it might matter:* ~60% of live blindspots persist at depth (static-eval,
  not search); the forward-bishop-sortie class is ~70% threat-feature over-credit
  and **systematic across all v9 nets** — it will NOT fall out of the
  train-bigger flywheel. The fix is mining the position type, SF-labelling, and
  mixing it into the next run.
- *Actionability:* uses existing infra. **Effort: medium. Prior: med, and
  unusually well-targeted** (attacks a measured deployment loss class).
  **Validate:** inner loop `testdata/overrate.epd` static-eval ranks; outer
  loop SPRT vs prod. **Highest-leverage training item.**

### Tier 2 — already-ported knob, re-run in an untested regime

**4. WDL ramp at S800 / diverse stage-1 data (NOT at S200 again).** 5× S200
probes H0; direction *narrowed not closed*. Untested: ramp at S400/S800, ramp +
fen-skip, ramp with diverse stage-1 data, endpoint 0.18, cosine ramp shape.
Flags already exist. **Caveat:** Coda's T80 data is LC0-WP-calibrated, so
λ=0.15-0.20 may already match SF's effective 0.4 — a real reason it may not
transfer (hence Tier 2). **Effort: low. Prior: low.** Validate at S800.

**5. Clean lambda-jitter σ-sweep.** Jitter ported (PR #473); σ=0.1/d=0.9 banked
+7.9 **but conflated with a retune** (#1505). SF's own values (σ=0.01/d=0.999)
tested *worse* (#1522 H0 −3.8). A clean σ=0 vs σ=0.1 same-trunk paired probe
was never run; jitter **rescued the 2×-data regression by +28** (#1533 — best
evidence it's a real regularizer for Coda's under-iteration regime). **Effort:
low-med. Prior: low-med.** Validate clean σ=0 vs σ=0.1 + finish the decay sweep.

### Tier 3 — confirmed parity or spent (do NOT re-test)

- **mse-power 2.6** — Coda is at 3.0, validated better at STC+LTC. SF's is
  *below* Coda. Settled.
- **WDL ramp at S200** — 5× H0; that regime is exhausted.
- **Fixed WDL 0.24** — `canonical-w24` H0 −5.6 at S200.
- **Multistage warm-restart** — 4-stage attempt regressed (S4 −13/−32,
  #1494/#1499); this configuration is dead (the light multistage+SWA tail in v6
  prod is the salvageable form).
- **SWA window tuning** — done: +5.0 with a ≤10% late window; ≥25% destructive
  (#1501/#1514/#1515). Already in v6 prod.
- **Batch 4×+LR 2×** — H0 −28 (under-trained). Spent.
- **2× raw data** — −34.6 at S800. Not a guaranteed lever at Coda's scale.
- **PSQT outputs, dual SqrCReLU+CReLU** — both H0-regress at S800 (#2036 −27.9,
  #2037 −12.5), not rescuable by recipe (`project_psqt_dual_regress_s800`).
- **Factoriser-drop-late** — training-speed only, not Elo.

## The meta-learning (most important, least flag-shaped)

SF **optimizes the training recipe directly against nElo (SPRT), via
nevergrad/TBPSA** — lr, gamma, pow-exp, WDL endpoints, the four scaling/offset
params, qp-asymmetry, all auto-tuned against *measured strength*, never loss
(which `feedback_loss_is_not_strength` repeatedly warns is wrong — e.g.
#2036/#2037 ranked the *regressing* nets best by loss). Coda hand-sets
eval_scale=400, picks WDL/mse-power by one-axis-at-a-time paired probes, and
reads loss curves.

**Portable version for Coda:** Coda already has the paired-probe + SPSA-on-trunk
machinery. The missing piece is a **small auto-search over the 3-4 most
uncertain training hyperparameters** (eval-scale/offset, WDL endpoint, jitter σ)
evaluated by S200 paired probes, instead of manual sweeps. This is process, not
code, and it is where SF's training advantage compounds.

## Recommendation

Two to fund: **(#3)** the targeted-data blindspot loop (already in flight; tied
to a measured loss class) and **(#1)** the out-offset eval calibration (cheapest
slice of the one unported SF mechanism; targets Coda's measured win/loss
asymmetry). Both are S200-probe-cheap to falsify before committing GPU-hours.

---

## Source-verified threats.yaml values + fen-skip semantics (2026-06-27)

Re-read SF's recipe from source (`vondele/nettest/threats.yaml` @ `b5023a3`, and
the nnue-pytorch dataloader `data_loader/cpp/training_data_loader.cpp`) to pin
the *exact* semantics — a casual read of the YAML got the skip direction
backwards.

**Verified threats.yaml values (advanced stage):**

| key | value | meaning |
|-----|-------|---------|
| `random-fen-skipping` | 10 | keep 1/(N+1) = **1/11 ≈ 9.1%** of positions (NOT "every 10th") |
| `early-fen-skipping` | 18 | hard cut: skip `ply <= 18` (keep ply ≥ 19) |
| `soft-early-fen-skipping` | 32 | probabilistic accept ramp for ply 19–31, full keep ≥ 32 |
| `pow-exp` | 2.435 | loss exponent |
| `qp-asymmetry` | 0.23 | over-estimation penalty |
| `in-scaling` / `in-offset` | 300 / 300 | sigmoid on NETWORK output (loss) |
| `out-scaling` / `out-offset` | 350 / 300 | sigmoid on DATA score (loss) |
| `start-lambda` / `end-lambda` | 0.74 / 0.74 | constant **WDL = 0.26** |

**`random-fen-skipping` — exact code** (`training_data_loader.cpp`):
```cpp
skip_prob = double(N) / (N + 1);                  // N=10 -> 10/11 = 0.909
random_skip_threshold = skip_prob * ~0ULL;
if (random_fen_skipping && prng() < random_skip_threshold) return true; // SKIP
```
So N=10 **skips ~90.9%, keeps ~9.1%**. Its purpose is intra-game decorrelation
(consecutive positions in a game are highly correlated; sparse sampling breaks
that), plus broad epoch coverage.

**How Coda achieves the same goal — differently, and we keep it that way:**
- **Shuffle buffer** (size-adjustable) + **multi-file interleave**
  (`--data-order interleave`, measured very effective — +40 #1712). These
  decorrelate by *reordering* while keeping **all** decoded positions.
- We deliberately do **NOT** copy SF's keep-9%: Bullet decodes positions on CPU,
  so skipping ~90% means decoding ~11× the positions to feed the GPU the same
  batch rate → **CPU-decode becomes the bottleneck** (would need a Bullet
  redesign). Our shuffle+interleave is strictly more *decode-efficient* (no
  discarded decodes) and reaches the same decorrelation end. **This is not a
  recipe gap to close** — it is a different valid solution to the same problem.
  We also run a light `--fen-skip-prob 0.5` (keep 50%) for coverage, not as the
  primary decorrelator.
- The ply cuts DO map: our hard `ply >= 16` ≈ SF `> 18` (SF one ply stricter;
  the S200 ply18 probe tested moving toward it); our `--soft-early-ply 28
  --soft-early-ply-floor 0.25` ≈ SF soft-32 + its 4-point ply ramp.

**in/out scaling + offset — what they are, and our take (we do none of it).**
These parameterize the antisymmetric double-sigmoid that maps eval →
win-probability *inside the loss*: `in-*` on the network output, `out-*` on the
data/target score. `offset` shifts where the sigmoid is centred (offset 300
concentrates loss sensitivity around ±300cp — moderate, decisive-ish advantages
— and flattens it near 0 and at the extremes); `scaling` sets the slope. Bullet
currently applies a plain sigmoid (no in/out split, no offset).

Assessment: a **second-order loss-calibration refinement, not a fundamental** —
Coda already measures SF-class on eval quality (Spearman ~0.853) without it, so
**low priority in isolation**. Two things keep it from zero: (a) the
**out-offset** specifically reshapes *where* eval accuracy is pushed, which
plausibly touches our measured win/loss asymmetry (this doc's rec #1); (b) these
four params are **co-tuned by SF against nElo alongside pow-exp / qp-asymmetry /
WDL**, so bolting one onto our current loss in isolation may undersell or misfire
— same coupling lesson as qp-asymmetry over-correcting at our exponent (#2308 /
the exponent study). If pursued: implement in the Bullet fork and test the
*coherent* SF loss block together, not this one slice alone.
