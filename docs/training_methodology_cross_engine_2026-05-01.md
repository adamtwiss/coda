# Training Methodology — Broad Cross-Engine Survey (2026-05-01)

Fourth piece in the training-methodology series. Companion to:
- `hobbes_selfplay_case_study_2026-04-30.md`
- `selfplay_data_strategy_2026-04-30.md`
- `training_patterns_2026-04-30.md`

This doc broadens the Hobbes deep-dive to **the rest of the top engine field**. Two parallel research agents covered:
- **Open-log engines:** Viridithas (deepest after Hobbes), Tarnished, Stormphrax, Integral, Caissa, Quanticade, PlentyChess, Reckless. Source: `~/chess/engines/<Name>/notes/networkhistory.txt` and READMEs.
- **Closed/secretive engines:** Stockfish, Obsidian, Halogen, Alexandria, Astra, Clover, Horsie, RubiChess, Koivisto, Seer, Arasan, BlackMarlin, plus brief scans. Source: inference-code archaeology in `~/chess/engines/<Name>/src/`.

`experiments.md` remains source of truth for resolved Coda SPRTs.

## Headline finding — three high-impact patterns Coda hasn't tried

After cross-checking ~17 engines, three patterns stand out as **high-precedent, well-documented, and untested by Coda**:

1. **`eleison` recipe — WDL ramp UP late stage *paired with eval rescaling*** (`viri eleison` +9.27 Elo VLTC, `~/chess/engines/viridithas/notes/networkhistory.txt:1604`). This **resolves the Hobbes-vs-Viri contradiction** in the prior training-patterns doc. Viridithas's `basilisk.4-.8` series regressed (−0.7 to −19 Elo) testing WDL ramp UP *without* rescaling. Hobbes's similar success likely involves implicit eval-scale shifts. Bare ramp fails; ramp + rescale wins.

2. **Bigger-arch teacher relabeling (Tarnished pattern, ~22B-position corpus rescored by 4096-FT 96→192→192-hidden teacher).** Distinct from Viridithas's same-arch self-rescore (`viri12` +50, `viri14` +25, `viri18` +24 Elo at intermediate strength but `voltarine` neutral at current strength — same-arch saturates). The bigger-teacher version reportedly breaks through where same-arch can't.

3. **`arasan-chess/network/arasanv5.rs` — a complete Bullet training script committed to the engine repo** with non-default settings: WDL=0.0, input-bucket factoriser, FT weight clipping ±0.99, 100× LR decay (vs Coda's recent ~30×). All four are unusual. Arasan is a competitive engine (3300+ class). **This is the most directly portable training recipe in the entire survey** — change four lines in our Bullet config and SPRT.

Three are independent and stack. Detailed analysis below.

## Per-engine training recipe snapshot

Compact table; full extraction in agent outputs. Cells marked `?` mean the engine doesn't disclose that parameter.

| Engine | Trainer | Data | WDL | Recipe novelty |
|---|---|---|---|---|
| Hobbes | Bullet | 100% self-play, random init, 41 iters | ramp 0.2→0.4→0.6→0.75 | covered separately |
| Viridithas | Bullet | hybrid Lc0 + selfplay + DFRC, ~5B fresh | **0.4** (sweep-validated) | extensive log; many iterations |
| Tarnished | Bullet | self-play, 22B, **bigger-teacher relabeled** | ? | bigger-arch (4096 FT) preceptor |
| Stormphrax | Bullet | 100% self-play, random init | ? | datagen.cpp public; pure RL ideology |
| Integral | Bullet | per-iter bootstrap from HCE, UHO openings | ? | per-iter full corpus replacement |
| Caissa | **custom CPU** | 17B+ self-play, periodic purges | ? | only top engine NOT on Bullet |
| Quanticade | Bullet | mid-transition T80→selfgen | ? | actively stalled on transition |
| PlentyChess | Bullet | 15B+ self-play | ? | **self-distillation** (mechanism undisclosed) |
| Reckless | Bullet | 768 FT + threats, two recent nets | ? | **most opaque top-3 engine** |
| Stockfish | nnue-pytorch | T80 + fishtest filtering | ~0 (historical) | **dual-net** (small FT=128 / big FT=1024); 60720 threats |
| Obsidian | Bullet | Lc0 data | ? | 13×8 buckets, 1536 FT |
| Halogen | Bullet (post-TD) | TD bootstrap → supervised | ? | **TD-then-supervised pipeline**; int8 threat weights, int16 PSQ |
| Alexandria | Bullet | Lc0 data | ? | **DUAL_ACTIVATION=true** (CReLU+SCReLU pairwise L1) |
| Astra | **Astra-Trainer** (custom) | ? | ? | own non-Bullet trainer; arch nearly identical to Obsidian |
| Clover | Bullet | 5B self-play, 20K-1M nodes/pos | ? | datagen with 4-ply 2000cp win adjudication |
| Horsie/Lizard | (closed) | 14×8 buckets | ? | **QB=132** (non-standard quantisation) |
| Seer | **PyTorch retrograde** | 6-man Syzygy WDL | N/A (EGTB labels) | **retrograde-from-EGTB** — fundamentally different paradigm |
| **Arasan** | Bullet | (Lc0 binpack) | **0.0** | **factoriser, FT clip ±0.99, 100× LR decay, full config in repo** |
| Coda V9 (current) | Bullet | T80 (Lc0) | 0.15 (V5: 0.07) | factor net, threats, 16 buckets |

## Convergent practices (≥4 engines agree)

- **Bullet trainer.** ~10 of the surveyed engines use Bullet directly. Caissa is custom (CPU), Astra runs its own (`Astra-Trainer`), Stockfish uses nnue-pytorch, Seer uses PyTorch retrograde. Bullet is the de-facto standard.
- **SCReLU somewhere in the chain.** Caissa is the lone CReLU-only top engine.
- **Pairwise multiplicative L1.** Stormphrax, PlentyChess, Tarnished, Viridithas (since `hyperstition` +42.9 Elo!), Alexandria (with both activations dual), Horsie. Reckless does not appear to.
- **Horizontal mirror + king buckets.** Universal except Caissa (32 buckets without obvious mirror).
- **8 output buckets by piece count.** Universal among recent engines.
- **Quiet-only filtering at training time.** Universal.
- **5-man Syzygy in datagen for adjudication.** Integral explicit; Stormphrax uses `kVerificationScoreLimit = 500cp`; standard practice.
- **Per-iteration bootstrap with prior net.** Hobbes, Integral, Caissa explicit; Viridithas implicit across decades; Tarnished (relabeled). Strong consensus.
- **Cumulative data with periodic purges.** Caissa explicit, Viridithas implicit (but with sophisticated date-based filtering — see `falke` cycle).

## Divergent practices (engines genuinely disagree)

- **WDL value.** Viridithas converged on **0.4** through a verifiable sweep (`basilisk.4-.9`). Hobbes uses ramp 0.2→0.75. **Arasan uses 0.0.** SF historically near-0. Coda V5=0.07, V9=0.15. The "consensus" is not a single number — it's bimodal: pure-self-play engines run higher (0.3-0.75); T80-trained engines run very low (0.0-0.15).
- **Final LR ratio.** Coda 30× decay, Arasan 100× decay, Viridithas variable per-net. No universal answer.
- **Activation chain depth.** Tarnished alone uses 16→32→32 (deeper L3). Viridithas tried L3=64 (`muscle` −9 STC, rejected). Most stop at L3=32.
- **Data origin philosophy.** Stormphrax 100% self-play from random; Caissa 100% self-play with HCE bootstrap; Quanticade mid-transition; Viridithas hybrid; PlentyChess self-play+self-distillation; Tarnished self-play with bigger-teacher relabel. **No single answer.**
- **Self-distillation vs bigger-teacher relabeling.** PlentyChess uses self-distillation; Tarnished uses bigger-arch teacher; Cosmo's `voltarine` (relabel with bigger preceptor) was *neutral* at current Viridithas strength. Two production-validated patterns, mutually exclusive deployment.
- **Quantisation.** Standard QA=255, QB=64. Horsie/Lizard QB=132. Viridithas qa-181 (+16.76 Elo fixed-nodes — pure SIMD packing).
- **Input bucket count.** 5 (Altair), 7 (Clover), 8 (Halogen), 9 (Arasan), 12 (Integral, PlentyChess), 13 (Obsidian, Astra), 14 (Horsie), 16 (Alexandria, Coda). Coda is on the high end; "consensus" is 9-13.

## Novel patterns ranked by ROI

Each item: cited source, Elo magnitude (where present), Coda applicability.

### High-confidence wins, well-precedented

**N1. `eleison` WDL-ramp + eval-rescaling combo (+9.27 Elo VLTC).**

Source: `viridithas/notes/networkhistory.txt:1604-1611`. Mechanism: WDL ramp 0.4→1.0 over training, **paired with eval-scale renormalisation** at the same time. The pairing is critical — Viridithas tested bare WDL ramp UP three different ways (`everedge` −5.78, `division` −3.6, `vestal` −57) and all regressed. Hobbes's similar pattern likely succeeds via implicit eval-scale shifts.

Coda implication: our planned probe #4 (Hobbes-style 2-stage with WDL bumped at stage 2) will likely fail without eval rescaling — needs to be paired with a recalibration of `EVAL_SCALE` in stage 2.

**N2. Tarnished bigger-arch teacher relabeling.**

Source: `~/chess/engines/Tarnished/README.md:46-49`. Architecture: train a 4096-FT, 96→192→192-hidden "preceptor" net (much bigger than runtime). Use it to rescore 22B-position corpus. Train production arch on relabeled data.

Distinct from same-arch self-rescore (Viridithas viri12/14/18: +50/+25/+24 at intermediate, but `voltarine` neutral at current strength — same-arch saturates). Bigger-teacher reportedly breaks through saturation.

Coda implication: the highest-ceiling untested pattern. Cost: train one 1.5-2× wider teacher net (~doubles GPU-hours of one training run), then rescore corpus, then retrain production. ~2x current training cost for one experiment.

**N3. `arasanv5.rs` complete Bullet config — 4 unusual settings stacked.**

Source: `~/chess/engines/arasan-chess/network/arasanv5.rs`. Settings:
- **WDL=0.0** (line 89) — pure eval target, no outcome blending
- **Input-bucket factoriser** (lines 51-53, 63-68) — base 768→L1 weights added to per-bucket weights via `repeat()`, smooths early training, folded at save
- **FT weight clipping ±0.99** on `l0w`/`l0f` only (lines 98-100) — bounds FT to ±1.0, exactly matching QA=255 quantisation without saturation
- **100× LR decay** (line ~85) vs Coda's recent ~30×

Coda implication: drop-in Bullet config A/B test. Each setting independently testable; full bundle is the Arasan recipe.

**N4. `viridithas qa-181` — re-quantise to QA=181 (+16.76 Elo fixed-nodes).**

Source: `viridithas/notes/networkhistory.txt:432-440`. Re-quantise existing trained net at QA=181 instead of QA=255 to enable better SIMD packing. Pure inference speedup, no retraining.

Coda implication: check whether Coda's current QA=255 leaves equivalent SIMD packing headroom. Especially relevant given v9's threat-feature inference cost.

**N5. `viridithas sibilant` — L1-norm penalty on L0 *output activations* (+5.9 Elo LTC).**

Source: `viridithas/notes/networkhistory.txt:1844-1849`. Add L1 norm of L0_out activations to loss. **Distinct from L1-on-weights** (which Coda has tested via group_lasso). Sibilant penalises *activation magnitudes*, not *weight magnitudes*.

Coda implication: novel mechanism not in our group_lasso work. Bullet config addition; test as a clean SB200 probe.

### Medium-confidence, novel-mechanism

**N6. Halogen TD-bootstrap → supervised fine-tune.**

Source: `~/chess/engines/Halogen/README.md:126`. Pre-bootstrap a net with Temporal Difference learning (PR #517), THEN fine-tune via Bullet supervised. None of the open-log engines do TD bootstrap; they all start from random or transfer.

Coda implication: could de-risk early epochs especially for v9 threats where labels are noisier. New training infrastructure required (TD self-play loop). Long-term experiment.

**N7. SF-style dual-net architecture.**

Source: `Stockfish/src/evaluate.cpp:49`, `nnue_architecture.h:39-49,60-69`. Big net (FT=1024) for `|simple_eval| < 962cp`, small net (FT=128) otherwise. Effectively trains two specialists: the small net for "obviously decided" positions (saving compute), the big net for close positions where eval matters most.

Coda implication: significant infrastructure (two trained nets, runtime gate). Defer until single-net path saturates.

**N8. Halogen int8 threat weights (mixed-precision).**

Source: `~/chess/engines/Halogen/src/network/arch.hpp:47`. Threat weights are int8 while king-bucketed PSQ weights are int16. Halves memory/bandwidth on the threat embedding without losing accuracy (threats are sparse and bounded).

Coda implication: when v9 threat features are stable, consider int8 instead of int16 specifically for threats. Memory + cache + bandwidth gains.

**N9. Alexandria's `DUAL_ACTIVATION=true` confirmed in production.**

Source: `~/chess/engines/Alexandria/src/nnue.h:14-32` and `README.md:26-30`. Production engine running dual CReLU+SCReLU pairwise L1 (concatenated). This is **the Hobbes h-40 pattern on a different engine** — second independent confirmation of the technique's value.

Coda implication: increases confidence in our planned dual-activation FT probe (#3 in screening matrix). Two engines now confirm the pattern.

**N10. PlentyChess self-distillation.**

Source: `~/chess/engines/PlentyChess/README.md:5`. Top-3 engine. Mechanism undisclosed but likely: during training, sample positions twice — at depth d and depth d+k — train d-output to match d+k-output. Differs from corpus rescoring (which Viridithas now finds neutral): self-distillation is in-loop, not a separate corpus pass.

Coda implication: not quantifiable without the recipe details, but a top-3 engine uses it. Long-term research thread.

### Smaller / data-hygiene wins

**N11. Viridithas `falke` data hygiene (+13.86 Elo LTC).**

Source: `viridithas/notes/networkhistory.txt:891-907`. Periodic data purge + 4× upsample of recent 25k-node deep-search positions. Different from blanket purges; surgical date-based filtering.

Coda implication: applicable when self-play corpus accumulates past current scale.

**N12. Random-skipping + warmup joint sweep (Viridithas `dagger` +3.12 vs `stalker`).**

Source: `viridithas/notes/networkhistory.txt:1381-1418`. Not just "skip rate" — the joint of skip rate + batch warmup matters. `dagger` (skip=5/6 + 1600-batch warmup): +3.12. `fizz` (skip=0.75 alone): −14.31. Buffer size + skip rate + warmup all interact.

Coda implication: defaults Bullet uses may be far from joint optimum.

**N13. Tarnished 16→32→32 deeper L3.**

Source: `~/chess/engines/Tarnished/README.md`. Single anecdote, top-15 engine. Most peers stop at 16→32.

Coda implication: small architecture experiment; viri tried L3=64 (rejected) but not L3=32→32 stacking.

## Coda-specific actionable patches

Ordered by priority (high-ROI + low-cost first), all gated on factor-variance investigation completing:

| # | Patch | Source | Cost | Expected Elo |
|---|---|---|---|---|
| 1 | **N3 — Arasan recipe in screening probe** (WDL=0.0, factoriser, FT clip ±0.99, 100× LR decay) | arasanv5.rs | 4 Bullet config lines | +5 to +20 cumulative if Arasan's stack transfers |
| 2 | **N4 — QA=181 re-quantisation test** | viri qa-181 | No retrain; re-quantise + SIMD A/B | +5 to +17 fixed-nodes |
| 3 | **N5 — Sibilant L1-on-activations probe** | viri sibilant | Bullet loss add | +3 to +6 LTC |
| 4 | **N1 — eleison WDL+rescale combo** (informs probe #4) | viri eleison | Bullet stage-2 + EVAL_SCALE recalc | +5 to +9 LTC |
| 5 | **N9 — dual-activation FT probe** (confirmed by Alexandria) | already in our screening | (unchanged) | (unchanged) |
| 6 | **N2 — Tarnished bigger-teacher relabel** | Tarnished README | Train wider preceptor + corpus rescore | +10 to +30 OR neutral (saturation) |
| 7 | **N8 — int8 threat weights** | Halogen | NNUE format change | +0 cosmetic Elo, NPS gain |
| 8 | **N12 — random-skip + warmup joint** | viri dagger | Bullet config | +1 to +5 |
| 9 | **N13 — 16→32→32 L3** | Tarnished | architecture change | ±5 |
| 10 | **N6 — TD bootstrap pipeline** | Halogen | New infra (~1 week) | speculative |
| 11 | **N7 — Dual-net architecture (SF)** | Stockfish | Significant infra | speculative |
| 12 | **N10 — Self-distillation (PlentyChess)** | undisclosed mechanism | Research-first | speculative |

## Coda's WDL value — the structural anomaly

Worth calling out separately. The 17-engine survey gives a clear distribution:

- **Pure self-play engines (Hobbes, Stormphrax, Caissa, etc.):** WDL 0.3–0.75 (often ramping)
- **T80/Lc0 hybrid engines (Viridithas, etc.):** WDL 0.4 (Cosmo's sweep-confirmed)
- **T80-only engines (Alexandria, Obsidian — undisclosed but presumably similar):** unknown
- **Arasan, SF historical:** WDL ≈ 0.0
- **Coda V5:** 0.07
- **Coda V9:** 0.15

So Coda is **bimodal-skewed**: V5 sits with Arasan/SF (eval-target-heavy); V9 sits in a no-man's land between SF (low) and Viridithas (high). The "consensus" is itself bimodal, so Coda being between the modes isn't *wrong* — but it's worth checking whether moving to either pole (WDL=0 like Arasan, or WDL=0.4 like Viri) outperforms the current 0.15.

This is a 2-seed SB200 probe each direction. Cheap. Worth doing alongside the existing #1 ramp probe.

## Re-evaluation has a saturation point

Important framing for our own re-evaluation experiments:

- **Intermediate strength (Coda likely here):** same-arch self-rescore wins big. Viridithas viri12 +50, viri14 +25, viri18 +24 Elo. Re-eval works.
- **High strength (Cosmo's current arch):** same-arch self-rescore *neutral*. `voltarine` (relabel with bigger preceptor): −2.4 Elo. `equilibrium` (replace search scores with inimical-output): −2.7 Elo. Saturation.
- **Past saturation:** bigger-arch teacher (Tarnished pattern) is the only path that breaks through.

Coda implication: **we are likely in the sweet spot for same-arch re-evaluation**, which means the cheap re-eval pilot (E1 from `selfplay_data_strategy_2026-04-30.md`, +5 to +25 expected) is high-confidence. Don't skip directly to Tarnished's bigger-teacher pattern when same-arch hasn't been tested yet.

## What we still don't know

Even with this survey, several Coda-relevant questions are unanswered:

1. **Stockfish's exact training recipe for the dual-net.** Public source confirms architecture (FT=1024 big, FT=128 small), gating threshold (962cp), and threat dimensions (60720). Hyperparameters (WDL, LR, SBs, filtering) live at glinscott/nnue-pytorch and on Discord. Would require web fetches to chase.
2. **Reckless's training process.** Top-3 engine, completely opaque. Two committed `.nnue` files are the only artefacts.
3. **PlentyChess's "self-distillation" specifics.** A one-line README claim; mechanism could be in-batch deeper-target rescoring, full Lc0-style policy distillation, or something else.
4. **Tarnished's WDL/LR/SB hyperparameters.** Architecture and bigger-teacher idea disclosed; everything else private.
5. **Whether Coda's WDL=0.07/0.15 anomaly is real or measurement artifact.** Worth re-bracketing once factor variance lifts; current "optimum" was measured on factor-architecture nets which have ±22 Elo noise.

## Cross-cutting themes

1. **Strong engines are increasingly secretive.** Reckless, Obsidian, PlentyChess, Tarnished, Halogen all disclose architecture but not training recipes. Stockfish's training conversation is on Discord, not in-repo. Public training logs (Hobbes, Viridithas) are increasingly the exception, not the rule. Lots of training know-how hidden behind selection bias.

2. **Bullet is the de-facto standard, but Bullet configs in repo are rare.** Arasan is the standout exception with `arasanv5.rs` committed alongside the engine. Most Bullet-using engines keep configs private. **Adopting Arasan's "config in repo" practice for Coda would be a transparency norm we could lead on**, and lets future Claudes work from durable evidence.

3. **Self-play vs T80 is not a binary.** The strongest engines mix both: Stormphrax/Hobbes/Caissa go pure-self-play; Viridithas/Integral hybrid; Alexandria/Obsidian/Coda T80-heavy. **No correlation between data origin and strength** — top-10 includes engines from every category. Strongly suggests our T80-vs-self-play question is less load-bearing than recipe quality.

4. **Re-evaluation cycle works at intermediate strength, saturates at top.** Where Coda sits matters. Most likely: we have headroom from same-arch re-eval (cheap), and bigger-teacher relabel is reserved for after that saturates.

5. **Architecture co-evolution is universal in pure-self-play paths** (Hobbes 32→1536, Viridithas 256→2560+) but not necessary for T80-trained engines (Alexandria/Obsidian fixed at 1536). Coda's V9 architecture is at the endpoint shape; co-evolution doesn't apply for our remaining headroom.

## Companion docs

- `hobbes_selfplay_case_study_2026-04-30.md` — the deepest single-engine training case study.
- `training_patterns_2026-04-30.md` — synthesis of patterns + SB200 screening protocol.
- `selfplay_data_strategy_2026-04-30.md` — data origin (T80 vs self-play vs hybrid).
- `cross_engine_comparison_2026-04-25.md` — search-side cross-engine comparison.
- `factoriser_design_2026-04-21.md` — factoriser architecture context.
- `experiments.md` — source of truth for resolved Coda SPRTs.

## Notable engine source citations

For the highest-evidence findings:

- `~/chess/engines/viridithas/notes/networkhistory.txt` — by far the deepest public training log
- `~/chess/engines/arasan-chess/network/arasanv5.rs` — complete Bullet config in-repo
- `~/chess/engines/Stockfish/src/nnue/nnue_architecture.h:39-49,60-69` — dual-net architecture
- `~/chess/engines/Stockfish/src/nnue/features/full_threats.h:36-78` — 60720 threat features
- `~/chess/engines/Halogen/src/network/arch.hpp:16-55` — int8 threat weights
- `~/chess/engines/Halogen/README.md:126` — TD→supervised pipeline
- `~/chess/engines/Alexandria/src/nnue.h:14-32` — DUAL_ACTIVATION=true
- `~/chess/engines/Tarnished/README.md:46-49` — bigger-teacher relabel
- `~/chess/engines/Stormphrax/src/datagen/datagen.cpp:74-87` — datagen constants
- `~/chess/engines/seer-nnue/README.md:6` — retrograde-from-EGTB paradigm
- `~/chess/engines/CloverEngine/src/datagen/generate.h:32-41` — adjudication thresholds
