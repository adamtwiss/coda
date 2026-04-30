# Hobbes — Self-Play Case Study (2026-04-30)

Companion to `selfplay_data_strategy_2026-04-30.md`. Hobbes is a strong counter-example to the prior doc's conclusion that "no engine in the survey reports beating LC0/T80 by replacing it" — Hobbes is **100% self-play, started from random weights, reached CCRL Blitz #19 at 3715 Elo**. Source: cloned `kelseyde/hobbes-chess-engine` to `~/chess/engines/Hobbes/`, full network_history.txt (482 lines, 41 documented iterations) is the gold standard public log.

## What changes from the prior doc

Hobbes invalidates the strong claim "pure self-play replacement is universally negative" but **confirms** the weaker claim "single-iteration self-play replacement at parity scale loses". The two are compatible: Hobbes succeeded by running 41+ iterations over many months and growing the architecture in lockstep with data quality. Single-iteration replacement is still bad; **multi-iteration progressive bootstrap is what works**. My prior doc didn't have a clean existence proof for the latter; Hobbes provides one.

## Architecture progression — Hobbes ends where Coda v9 starts

| Iter | Architecture | Dataset | Notes |
|---|---|---|---|
| h-1 (random) | 768→32 | 33M | random init |
| h-2 | 768→64 | 59M | |
| h-5 | 768→128 | 115M | architecture jump |
| h-6 | 768→256 | 210M | |
| h-9 | 768hm→256 | 310M | added horizontal mirror |
| h-12 | 768hm→512 | 900M | |
| h-13 | 768hm→768 | 1.0B | |
| h-15 | 768hm→1024 | 1.2B | |
| h-16 | 768x4hm→1024 | 1.7B | 4 input buckets |
| h-19 | 768x6hm→1024 | 2.1B | 6 buckets |
| h-31 | 768x10hm→1280 | 12.5B (h6-h27) | 10 buckets |
| h-33 | 768x16hm→1280 | 14B std + 3B DFRC | 16 buckets |
| h-39 | (768x16hm→**1280pw**)x2→(16→32→1)x8 | 23B std + 5B DFRC | first pairwise + hidden |
| h-41 | **(768x16hm→1536pw)x2→(16x2→32→1)x8** | 23B+5B | **identical to Coda v9 architecture (minus threats)** |

**Key implication for Coda.** Hobbes's endpoint architecture is the same shape as Coda v9 (1536 pairwise FT, 16x2→32→1 hidden chain, 16 buckets, horizontal mirror). The architecture isn't the bottleneck; data and iteration count are. Coda has threat features Hobbes doesn't, which is an architectural advantage we keep on top.

## What Hobbes does that Coda's V5→V9 experiment didn't

Eight specific things:

1. **WDL signal ramp.** Hobbes started at `wdl: constant(1)` for h-1 — *the entire training signal is W/D/L of the game outcome, eval is ignored*. Reduced to 0.75 (h-3), 0.5 (h-4), 0.4 (h-7), then a schedule `100sb constant(0.2), 700sb linear(0.2, 0.4)` from h-11 onward. Late nets reach `k(0.6)` (h-33+) or `k(0.75)` (h-38+).

   **Why this matters:** WDL signal is *ground truth*, doesn't depend on eval quality. When the eval is poor (early iterations), full WDL bypasses the teacher-bias problem. As eval improves, weight shifts to the eval signal. **Coda's V5→V9 experiment likely used a fixed WDL weight tuned for high-quality T80 evals**, which is wrong for V5-quality evals.

2. **Architecture co-evolution.** Hobbes grew the FT output dim from 32 → 1536 across 41 iterations. **A small net trained on weak evals can't overfit the noise**; a large net can. Coda jumped from V5 architecture to V9 architecture in one iteration — V9 has the capacity to learn V5's eval blindspots in detail rather than averaging over them. This compounds with the WDL=fixed problem.

3. **Cumulative data, not latest-only.** Hobbes consistently writes "source: hobbes 6 → N-1" — using games from many recent generations, not just the latest net. The dataset accumulates, doesn't churn. h-31's data spans h6-h27 (21 generations of nets contributed). Fights diversity collapse.

4. **Two-stage training (since h-29).** Stage 1: `d1, 800sb, lr cosine(0.001 → 0.0000081), wdl 100sb(0.2) → 700sb linear(0.2, 0.4)`. Stage 2: `d2, 200sb, lr constant(0.00000081), wdl k(0.6)`. Where d1 is the broad cumulative corpus and d2 is just the recent slice. **Bootstrap + finetune pattern.** Coda doesn't use two-stage.

5. **Opening filter on absolute-eval bound.** `tools/datagen.rs:51-53`: after random opening, if `abs(eval) > 1000cp`, **regenerate the opening**. Throws away wildly imbalanced starts that wouldn't produce informative game data. Coda's datagen has random openings but no eval-based filter.

6. **DFRC (Chess960) blend.** ~80/20 standard/DFRC since h-32. Forces the net to evaluate non-standard piece configurations, which Hobbes credits with breaking through specific late-iteration plateaus (h-32: +159 Elo *on DFRC* with 22B-position retrain). Coda doesn't generate DFRC data.

7. **High-node-budget data subset (since h-37).** Notation like "16b std (2b 20k), 5b dfrc" means 2B positions out of 16B were generated at 20k soft nodes vs the default 5k. **Quality-vs-quantity tradeoff applied to a subset, not the whole corpus.**

8. **Fen skipping (since h-35).** `viriformat (fen skipping: 0.5)` — load 50% of FENs per epoch. Dataset is large enough that random subsampling per epoch acts as additional regularization without losing coverage. h-37 uses 0.5; h-35 used 0.75 (skip more aggressive).

## Hobbes's datagen architecture is *not* what Coda has

Hobbes's `tools/datagen.rs` is **71 lines** — only generates random opening FENs. It doesn't actually play games. The game-playing is driven externally by **OpenBench's `genfens` + datagen UCI command**, which calls Hobbes for opening generation and search but orchestrates the games itself.

This matches PlentyChess's pattern (`PlentyChess/src/datagen.cpp` is similarly minimal). Coda's `src/datagen.rs` is a full self-contained game-loop — different infrastructure model.

**Implication:** if Coda goes meaningful self-play, the OpenBench `genfens` route is the strongly-precedented infrastructure. Coda's existing in-process datagen is fine for one-shot experiments but probably not the tool for a 40-iteration multi-month bootstrap.

## What "vs Calvin" tells us about iteration economics

Hobbes-N's right column compares to a fixed external baseline (Calvin, the Java engine the same author wrote). The gap closes monotonically:

- h-1 (random init): −1199 Elo vs Calvin
- h-7 (256-FT, ~5.5B data): −183
- h-13 (768-FT, 1B): −85
- h-19 (1024-FT, 6 buckets, 2.1B): −42
- h-30+ : column drops out (Hobbes presumably surpassed Calvin somewhere around h-25–28)

**~6-7 doublings of architecture × data closed a 1200-Elo gap.** That's the actual cost curve of from-scratch self-play. Coda starting from current v9 quality skips perhaps the first 15-20 iterations of Hobbes's curve (we have a non-trivial eval); the question is what the curve looks like from iteration ~20.

## Re-reading Coda's V5→V9 −20 result with Hobbes context

The V5→V9 experiment was structurally **two iterations of architecture growth in one step**:
- Architecture jump (V5 → V9): equivalent to Hobbes h-15 → h-39 (3 architecture growth steps + pairwise + hidden chain)
- Data quality jump (T80 → V5 self-play): roughly equivalent to walking back to Hobbes h-13's eval quality

Doing both at once explains why a "single iteration" looked so bad. Hobbes never did anything that radical in a single step. **The cleaner same-arch experiment (V9 self-play to retrain V9) would land much more positively** — by analogy with the per-iteration Hobbes deltas of +5 to +25 Elo when matched on architecture.

## Updated recommendations

The prior doc's Recipe-C re-eval pilot and Recipe-B wider-teacher relabel still stand as the cheap near-term options. Hobbes adds two patterns Coda hasn't considered:

### New Recipe-E: Hobbes-style WDL-ramped + 2-stage continuation training

If/when Coda runs *any* self-play (mixed or replaced), adopt the Hobbes schedule:
- WDL=1.0 (or 0.75) for the first ~10% of training; ramp to eval-weight by 50%; settle around 0.4-0.6 for the tail.
- Two-stage: 800sb on broad data, then 200sb on recent slice at constant low LR.
- DFRC opening blend at 20%.
- Opening eval-bound filter (regenerate if `|eval| > 1000`).
- High-node subset (10% of data at 4× the soft-node budget).

These are configuration changes to the existing pipeline, not new infrastructure. Test them as **a Coda self-play recipe** for the V9-self-play-mixed-into-T80 pilot (E2 in the prior doc).

### New Recipe-F: from-current-V9 progressive bootstrap (the maximalist path)

If Coda commits long-term to self-play, follow Hobbes's iteration recipe **from the current V9 net, not from random**:
- Iteration 1: generate 1B games with V9, fixed-node 5k soft / 1M hard, 8 random plies + eval-bound filter. Cumulate with T80 (don't replace). Train V9' with Hobbes-style two-stage + WDL ramp.
- Iteration 2: generate 1B with V9'. Mix with T80 + iter1. Train V9''.
- ...

Hobbes's per-iteration delta in the late-stage architecture-stable regime is **+5 to +25 Elo per iteration** (h-31 to h-41 deltas). Even at the low end, 5-10 iterations is +25 to +250 Elo. The high end is what got Hobbes to top-20.

**Cost:** each iteration is ~200-1000 GPU-hours of training plus ~1000-5000 CPU-hours of game generation. 5 iterations ≈ 2-3 months of focused compute.

**Requires:** OpenBench `genfens` + datagen UCI integration on the Coda side (Hobbes already has it; we'd need to add it). Stable factor architecture (the variance issue is currently blocking this).

## Key delta from the prior doc

- "No engine in the survey reports beating LC0/T80 by replacing it in a single step" → still true.
- "Pure self-play replacement is universally negative" → false; Hobbes refutes it across 40+ iterations.
- The actionable difference: **Recipe-F is now on the table** as a credible long-term path. Not for now, but for after the variance investigation lands and we've banked the +30-50 Elo of structural fixes in `cross_engine_comparison_2026-04-25.md`.

## Open questions

1. **Did Hobbes ever import T80 data?** The README says "trained entirely on data generated from self-play". Reading the network_history I find no T80 references. Pure RL throughout.
2. **What's Hobbes's eval scale at each iteration?** Their `scale: 400` is constant from h-1 to h-41 — same as Coda's `EVAL_SCALE=400`. So the scale calibration isn't iteration-specific; the *data* is.
3. **What's the per-iteration compute?** Not documented numerically. From dataset growth (~1B per iteration in late stages) and standard genfens throughput, probably 500-2000 CPU-hours of generation + 100-500 GPU-hours of training per iteration. Confirmable by asking the author.
4. **Does Hobbes ever revert?** Network history shows monotonic SPRT-vs-prior wins with occasional "non-reg stopped" entries (h-34) — they kept going past a non-positive iteration when the architecture was new. So they tolerate flat iterations during architecture growth but not during stable-arch refinement.

## Sources

- `~/chess/engines/Hobbes/network_history.txt` — full 41-iteration log
- `~/chess/engines/Hobbes/README.md` — architecture + strength summary
- `~/chess/engines/Hobbes/src/tools/datagen.rs` — opening-only datagen confirms OpenBench-driven game-play
- Mattbench OpenBench instance (`https://chess.n9x.co/`) — test logs linked from each network row
- CCRL Blitz #19, 3715 Elo — README headline figure; CCRL list at `https://computerchess.org.uk/404/`

## Companion docs

- `selfplay_data_strategy_2026-04-30.md` — prior strategy doc; this doc adds the "multi-iteration bootstrap is viable" data point.
- `cross_engine_comparison_2026-04-25.md` — Hobbes wasn't in the original 17-engine survey; should be added to the next survey.
- `factoriser_design_2026-04-21.md` — Hobbes doesn't use factoriser; their progressive bootstrap doesn't need it.
- `experiments.md` — source of truth for resolved Coda SPRTs.
