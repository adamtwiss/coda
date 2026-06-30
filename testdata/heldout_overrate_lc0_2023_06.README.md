# heldout_overrate_lc0_2023_06.tsv

Held-out eval-overrate test set: positions where Coda's NNUE static eval is
known to sit far from ground truth, used to measure whether the blindspot
corrective fine-tune actually reduces eval error on the population it targets.

Format: `fen<TAB>lc0_stm_cp` — LC0 800-node-MCTS score, side-to-move POV (cp).

## Provenance

- **Source**: `/training/sf/test80-2023-06-jun-2tb7p.min-v2.v6.binpack`
  (June **2023** T80 — **held out** from the Jan–Jun **2024** training data, so
  no candidate position was seen during training).
- **Filter = the training harvest's** (docs/blindspot_data_generation.md):
  1. Coda static far from LC0 oracle: calibrated `|coda − lc0| ≥ 150cp`.
  2. SF static materially closer: calibrated `(coda_err − sf_err) ≥ 80cp`
     — rules out "only deep search resolves this" (a tactic), isolating a
     Coda-specific learned eval error. SF eval is **static** (`evalfile`,
     no search), matching the harvest arbitrator. **No search anywhere** —
     truth is the binpack's LC0 MCTS score.
  3. Quiet-only, in-check skipped, oracle band `|lc0| ≤ 600cp`.
- **Game-deduped**: ≤ 1 position per game (eval-dist `game_id`, derived from
  ply-continuation breaks). Same-game positions share the blind spot, so
  several from one game would correlate the set and inflate the signal.
  All 7,847 positions come from 7,847 distinct games.

## Why not the old EPD

The retired `coda_overrate_gauntlet.epd` was adjudicated by Stockfish **searched
to depth 24** (dynamic), whereas the harvest arbitrates on SF **static** eval.
Search resolves tactics, so the EPD oracle kept deep-tactical positions the
static corrective data cannot fix — contaminating the set and producing a
misleading (reversed) verdict. Removed 2026-06-30.

## N

7,847 positions / 7,847 distinct games. Truth mean −39cp, median −5cp, full
±600 band; STM balance 4039 b / 3808 w.

## Usage

```
python3 scripts/eval_compare_nets.py \
    --net ctrl=<t80_only>.nnue --net mix=<t80_plus_blindspot>.nnue \
    --tsv heldout=testdata/heldout_overrate_lc0_2023_06.tsv
```
Mix should show lower `mean |eval−truth|` than ctrl (same bake length).
