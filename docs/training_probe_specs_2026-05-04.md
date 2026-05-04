# Training Probe Specs — 2026-05-04

Companion to `training_patterns_2026-04-30.md` and `training_methodology_cross_engine_2026-05-01.md`.

Each probe spec'd as: **hypothesis / baseline / probe / read / failure-mode-to-avoid / cost**.

All probes share the canonical V9 control recipe (per
`memory/project_canonical_v9_control_recipe.md`):

```
--kb-layout reckless --hidden-activation crelu --factoriser
--warmup 30 --wdl 0.15 --superbatches 200
--seed 42 --threads 8 --save-rate 200
```

(FT activation = SCReLU, hidden chain CReLU, FT width = 768, threats
on, 16 buckets via reckless layout, factor folded at save.)

Probes only differ from control by the single knob being tested.

---

## Already removed from queue

- **Fixed-WDL bracket (P1/P2/P12 in earlier draft)** — w0/w5/w10/w15/w20
  SB200 nets exist on disk (Apr 17–21); cohort already used as seed-
  variance substrate. WDL fixed-value space is sampled.
- **A1 length curve** — pushed back: SB200/400/800 adds rigour to known
  ~20 Elo/doubling result, not new direction. SB1200 alone is the only
  truly-untested length point; not blocking 75h GPU on it.
- **A7 warmup re-test** — sweep at SB200 already showed warm30 best;
  re-validating under fixed seed adds rigour, not direction.

---

## A2 — Hobbes-shape WDL v1 (moderate ramp + tail)

| | |
|---|---|
| **Hypothesis** | Late-training WDL > 0.15 (linear ramp + constant tail per Hobbes pattern h-11 → h-32 generation series) outperforms constant 0.15 at SB200. Tests whether the staged-WDL pattern that worked across 30+ Hobbes iterations transfers to Coda v9 architecture. |
| **Baseline** | Canonical control SB200, same host, same seed. |
| **Probe** | Same recipe + `--wdl 0.05 --wdl-end 0.20 --wdl-tail 0.30 --wdl-tail-from 150`. Stage 1 = SB150 linear 0.05→0.20, stage 2 = SB50 constant 0.30. Conservative: bracketed below Viri's `basilisk` regression zone (≥ 0.4 finetune lost 5-19 Elo). |
| **Read** | SPRT control.nnue vs probe.nnue, ~10K games. Elo only. |
| **Failure mode** | Mid-training trajectory divergence between control and probe (different RNG draws hit different positions). Mitigated by same-host sequential + fixed seed + Bullet loader patch. |
| **Cost** | 2× SB200 ≈ 24h paired on one host. Bullet patch already pushed (`3aa0022`). |

## A3 — Hobbes-shape WDL v2 (aggressive ramp + tail) — brackets A2

| | |
|---|---|
| **Hypothesis** | A more aggressive ramp (0.05→0.30) + higher tail (0.50) reveals whether the gain is from "any late high WDL" or "moderate level". If A3 ≫ A2 → high-WDL tail is the lever; if A3 < A2 → moderate is the sweet spot. Brackets A2. |
| **Baseline** | **Shares A2's control** (same recipe, same seed) — only valid IF Day-0 calibration shows fixed-seed cross-host variance < 5 Elo. Otherwise: re-train an A3-specific control on the A3 host. |
| **Probe** | `--wdl 0.05 --wdl-end 0.30 --wdl-tail 0.50 --wdl-tail-from 150` |
| **Read** | SPRT vs A2 control (same host) or vs fresh control (different host). |
| **Failure mode** | Cross-host control reuse contaminates Δ if Day-0 says variance > 5 Elo. If unsure, re-train control on probe host. |
| **Cost** | 1× SB200 ≈ 12h (re-uses A2 control if cross-host OK), or 2× SB200 ≈ 24h (paired). |

## A4 — FT=512 vs FT=768 (cache/NPS lever) — phased

**Phased because the FT-width change touches both Bullet (training-side
shape) and Coda (inference-side accumulator allocation + SIMD kernel
shapes). Don't pay inference engineering cost without first checking
the training-loss curve.**

### A4 Phase 1 — training-loss screen (no inference work)

| | |
|---|---|
| **Hypothesis** | FT=512 trains to within ~5% loss of FT=768 at SB200 with all other knobs equal. If loss gap > 10%, drop without inference work. |
| **Baseline** | Canonical control SB200, FT=768 (= Day-0 calibration triple control). |
| **Probe** | SB200 with `--ft-size 512` (Bullet patch needed; currently hardcoded `let ft_size = 768;` at line 61), all other knobs identical. |
| **Read** | Final-SB training loss + plot of loss curve vs control. NO bench, NO SPRT — Coda inference doesn't load FT=512 yet. |
| **Failure mode** | Tree-shape NPS artefact — irrelevant for Phase 1 (no NPS read). |
| **Cost** | 1× SB200 ≈ 12h + Bullet `--ft-size` patch (~30 min, but needs to propagate `ft_size` through all `Shape` constructions in the example). |

### A4 Phase 2 — full inference + SPRT (only if Phase 1 promotes)

| | |
|---|---|
| **Hypothesis** | FT=512 trades < 10 Elo for measurable NPS gain (cache residency: 49 MB threat-weight matrix → 33 MB at FT=512, closer to L3 fit). |
| **Baseline** | Canonical control SB200, FT=768. |
| **Probe** | SB200 with FT=512, both nets at SAME training stage. |
| **Read** | (1) NPS via `make && ./coda bench` on EACH net at SB200 — both at SAME training stage to avoid tree-shape artefact. (2) Elo via SPRT, ~10K games. |
| **Failure mode** | **Tree-shape NPS artefact** — undertrained nets produce wider trees, inflating NPS. We saw this with the compact threat encoder (587K vs 578K real, NOT the SB10 +38% claim). Both Phase 2 nets MUST be at the agreed strength-test length, not earlier. Smoke tests at SB10 only confirm the FT=512 build works. |
| **Cost** | Phase 2 inference work: ~3-5 days engineer time (parameterize accumulator, SIMD kernels, finny table). Only undertaken if Phase 1 loss curve looks promising. |

## A5 — Reckless non-linear output buckets

| | |
|---|---|
| **Hypothesis** | Reckless's endgame-weighted bucket layout (LAYOUT[33] = 9/4/4/3/3/3/3/2 piece counts per bucket) better matches Coda's eval landscape than uniform `(popcount-2)/4` buckets. Endgame eval is more discontinuous → more output-head capacity helps; opening eval is smoother → less capacity sufficient. |
| **Baseline** | Canonical control SB200 (uniform `MaterialCount<8>` buckets). |
| **Probe** | SB200 with new `RecklessBuckets` output struct (per agent spec from `~/chess/engines/Reckless/src/nnue.rs:83-92`). Bullet patch + Coda inference patch (`src/nnue.rs:266-269` table-lookup). |
| **Read** | SPRT, ~10K games. |
| **Failure mode** | Search thresholds (RFP/futility/SEE) calibrated to current eval scale — bucketing change can shift EVAL_SCALE. Per `feedback_net_swap_needs_full_sweep_retune.md`, plan a 77-param full-sweep retune before deploy if probe lands positive. |
| **Cost** | Bullet patch + Coda nnue.rs patch (~1h total, both small) + 2× SB200 paired (24h). |

## A6 — Final LR up: 4.86e-6 (2× current)

| | |
|---|---|
| **Hypothesis** | Coda's `final_lr = 2.43e-6` was validated downward (lower regressed) but never tested upward. Hobbes h-8+ lands at 8.1e-6 (3.3×) on a smaller architecture. Possibly v9 sits to the LEFT of optimum. |
| **Baseline** | Canonical control SB200 (`final_lr = 2.43e-6` Bullet default, no flag). |
| **Probe** | SB200 with `--final-lr 4.86e-6` (existing flag, no patch needed). |
| **Read** | SPRT, ~10K games. |
| **Failure mode** | If positive, immediately try 7.3e-6 (3×) before declaring 4.86e-6 the optimum — could be on the slope, not at peak. |
| **Cost** | 2× SB200 paired (24h). No patch. |

## A8 — Sibilant: L1 norm of L0 OUTPUT activations

| | |
|---|---|
| **Hypothesis** | Adding `α * mean(\|L0_out\|)` to loss (α small, ~1e-4) regularises FT activations, encouraging sparser representations. Distinct from `--l1-decay` (L1 on weights) which Coda has tested. Viridithas's `sibilant` reported +5.9 Elo LTC. |
| **Baseline** | Canonical control SB200 (no activation regularisation). |
| **Probe** | SB200 with `--sibilant <α>` (Bullet patch needed: add a loss-graph term computing L1 of L0_out and add it to the objective). |
| **Read** | SPRT, ~10K games. |
| **Failure mode** | (a) α too large → L0_out collapses to zero, kills signal. Bracket α at 1e-5/1e-4/1e-3 if first run is flat or H0. (b) Tree-shape NPS artefact NOT applicable since arch is unchanged — bench-NPS read directly comparable. |
| **Cost** | Bullet patch (moderate — needs loss-graph addition mirroring how `l1_decay` is wired) + 2× SB200 paired (24h). |

## A9 — Dual-activation FT (Hobbes h-40 / Alexandria DUAL_ACTIVATION) — phased

### A9 Phase 1 — training-loss screen

| | |
|---|---|
| **Hypothesis** | Concatenating two activations on the FT output (CReLU + custom-SCReLU) gives L1 access to two complementary nonlinearities. Hobbes h-40 reported +9.5 Elo over h-39 (single SCReLU FT); Alexandria runs `DUAL_ACTIVATION=true` in production. Two-engine confirmation. |
| **Baseline** | Canonical control SB200. |
| **Probe** | SB200 with dual-activation FT (Bullet patch: apply both activations and concatenate at FT output, so L1 input dim doubles: 384 → 768 per perspective). NO inference work. |
| **Read** | Training-loss curve. If end-of-cosine loss < control by ≥ 5%: promote to Phase 2. |
| **Failure mode** | Dual-act FT is a different mechanism from V8 "dual L1" (which was parallel hidden paths inside L1). V8 was abandoned, not rejected — don't treat as a prior. Hobbes h-40 + Alexandria are independent confirmations. |
| **Cost** | Bullet patch (substantial — concat two activated tensors into doubled-width L1 input) + 1× SB200 ≈ 12h. |

### A9 Phase 2 — full inference + SPRT (only if Phase 1 promotes)

Coda inference SIMD path needs both activations + concatenation before the L1 matmul. Real engineering (~2-3 days). Net format header bit for "dual-activation FT". Converter flag. Defer until Phase 1 justifies the investment.

## ~~A10 — Arasan FT weight clip ±0.99~~ — DROPPED (already in canonical control)

`coda_v9_768_threats.rs:482-497` unconditionally applies
`AdamWParams { max_weight: 0.99, min_weight: -0.99 }` to `l0w` (and
`l0f` when factoriser is on). This is **identical to Arasan**.

Bullet default is ±1.98 (`acyclib/src/trainer/optimiser/adam.rs:39`), so
Coda's 0.99 setting is genuinely tighter than default — it's not a
no-op. The cross-engine doc N3 listing this as a "missing Arasan
setting" was wrong. Coda already has it.

**Implication for the rest of the Arasan stack (N3):** of the four
unusual Arasan settings, two are already in Coda (factoriser, ±0.99
FT clip), one is in the doc-validated WDL bracket already swept
(WDL=0.0), and one is genuinely untested upward (LR endpoint;
covered by A6/A11). The Arasan stack is mostly already digested.

## A11 — Arasan-shape final LR (1% of initial)

| | |
|---|---|
| **Hypothesis** | Arasan uses `final_lr = INITIAL_LR * 0.01` (0.001 → 1e-5, 100× decay). Coda uses `0.001 * 0.3^5 = 2.43e-6` (~412× decay). Arasan's endpoint is ~4× HIGHER than Coda's. Tests the same direction as A6 but at a more extreme value. |
| **Baseline** | Canonical control SB200. |
| **Probe** | SB200 with `--final-lr 1e-5`. No patch needed. |
| **Read** | SPRT, ~10K games. |
| **Failure mode** | A11 is essentially a more aggressive A6. Run A6 first; only run A11 if A6 (`final_lr = 4.86e-6`) is positive — A11 brackets up further. If A6 is flat/negative, A11 likely also is. |
| **Cost** | 2× SB200 paired (24h). No patch. Conditional on A6 being positive. |

---

## Day 1 fire order, conditional on Day-0 outcome

### If Day-0 cross-host variance < 5 Elo (shared-control protocol)

```
Hour 0:   Day-0 control net is the shared baseline for all probes
Hour 0:   gpu3 → A2 probe SB200 (12h)
          gpu4 → A6 probe SB200 (12h)  — no patch needed
          gpu5 → A8 probe SB200 (12h)  — needs sibilant Bullet patch first
Hour 12:  SPRT all three vs Day-0 control
          gpu3 → A3 probe SB200 (12h)
          gpu4 → A4 Phase 1 probe SB200 FT=512 (12h)  — needs FT-size patch
          gpu5 → A11 if A6 positive (12h), else A5 Reckless buckets (12h, needs patch)
```

### If Day-0 cross-host variance 5-15 Elo (same-host paired)

```
Hour 0:   gpu3 → control + A2 paired SB200 (24h)
          gpu4 → control + A6 paired SB200 (24h)
          gpu5 → control + A3 paired SB200 (24h)
Hour 24:  SPRT each pair on its own host
          Repeat with next probes...
```

## Bullet patches needed before fire

| Probe | Patch | Effort |
|---|---|---|
| A2/A3 | Already pushed (`3aa0022`) | done |
| A6/A11 | None (existing `--final-lr`) | done |
| ~~A10~~ | DROPPED — already in canonical control | n/a |
| A4 Phase 1 | Add `--ft-size <N>` flag, propagate through Shape constructions | ~30 min |
| A8 | Add `--sibilant <α>` flag → loss-graph L1 term on L0_out | ~1-2h |
| A5 | Add `RecklessBuckets` struct + `--output-bucket-layout` flag | ~30 min |
| A9 Phase 1 | Concat two activations at FT output, double L1 input dim | ~2-3h |

A10 + A4 Phase 1 + A8 patches will batch into one Bullet commit during
the Day-0 calibration window so all three are ready when training
finishes.
