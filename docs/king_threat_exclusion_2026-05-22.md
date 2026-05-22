# King-Threat Exclusion — Candidate Net Experiment

**Date:** 2026-05-22
**Status:** Research finding, not yet tested. Park for a future net-recipe iteration.
**Source trigger:** Viridithas PR #425 (https://github.com/cosmobobak/viridithas/pull/425)
landed threat inputs using the **SFNNv12 layout**, which "excludes all threats
involving a king" — both king-as-attacker and king-as-victim.

## Current state of Coda's threats

`src/threats.rs:249-256` `PIECE_INTERACTION_MAP`:

```
            P   N   B   R   Q   K
Pawn:       0   1   -1  2   -1  -1
Knight:     0   1   2   3   4   -1
Bishop:     0   1   2   3   -1  -1
Rook:       0   1   2   3   -1  -1
Queen:      0   1   2   3   4   -1
King:       0   1   2   3   -1  -1
```

- **King as victim (last column):** all `-1`. **Already excluded.** Matches SF/Viridithas.
- **King as attacker (last row):** `0,1,2,3` for P/N/B/R. **Coda still tracks "king attacks {P,N,B,R}".** SFNNv12 excludes this row too.

So Coda has 4 piece-pair-types worth of king-as-attacker features that the consensus
SF-derived layout has decided are net-not-worth-keeping. Total Coda threat features
at startup: 66,864 (printed at engine init).

## Why SF/Viridithas excludes king-as-attacker (best guesses)

1. **Sparse signal.** King attacks at most 8 squares; most positions have zero
   active king-attacker threats firing. Sparse features → weak gradient → noise
   at training time.
2. **Redundant with HalfKA king-square encoding.** Every position already encodes
   king-on-X with 16 mirror buckets; king-attacker patterns are largely derivable
   from FT-encoded king-square + enemy-piece positions, so the explicit features
   add little incremental signal but full incremental cost.
3. **NPS tax.** Each tracked feature costs incremental-update work in the
   accumulator and Finny delta path. If marginal contribution is low,
   removing it returns NPS for free.

## Expected magnitude

- **Feature-count reduction:** 4 of 27 non-excluded ordered piece-pair types →
  rough ~15% by pair count, but king-attacker rows are the smallest fanout in
  the matrix (king attacks ≤ 8 squares vs queen's ~27), so the actual
  feature-count drop is probably 8-12% of the 66,864 total.
- **NPS gain:** small-to-moderate — proportional to incremental-update cost
  share, plausibly 2-6% NPS on AVX2/AVX-512.
- **Elo:** uncertain. Direction range likely **−2 to +5**. SF/Viridithas
  convergence suggests at-worst-neutral. If feature was net-noise, removing
  it banks training signal-to-noise improvement; if it carried real signal,
  removing hurts.

## Cost to test

Not a recipe-only flag flip. The 66,864 feature-count is baked into the
net file header (extended_kb layout encodes shape) and is read at load time.
A king-attacker-exclusion experiment requires:

1. Edit `PIECE_INTERACTION_MAP` last row to `[-1, -1, -1, -1, -1, -1]`.
2. Re-derive threat-feature offset tables (handled by `init()` in `threats.rs`).
3. Re-build Bullet trainer with matching feature count.
4. Train S200 fresh from scratch under canonical paired-probe recipe.
5. SPRT vs current baby-prod via mini-prod paired probe at `[-5, 5]`.
6. If positive, plan a fresh S800 production net.

Total wall-clock: ~3-4h S200 train on 5070 Ti + ~6-12h SPRT.

## Where this sits in the priority stack

- Lower-EV than chasing the **+15 Elo from more-data doubling** experiment
  (prod-2024 result + 30B intermediate probe).
- Higher-EV than further ply-filter bracketing (ply-16 confirmed peak).
- Worth grouping with the next net-architecture iteration — e.g. if we test
  a v10 architectural change (different L1/L2 widths, dual activation, etc.),
  bundle the king-attacker-exclusion change into the same baseline-reset.
- **Do NOT do this as a standalone change on the current prod recipe** unless
  fleet is genuinely idle — it requires a full re-train + paired-probe and
  the expected magnitude doesn't justify the GPU hours when bigger known
  levers (data scaling) are queued.

## Reference numbers from Viridithas #425

For calibrating expectations from external evidence:

| TC                        | Elo gain          | Games  |
|---------------------------|-------------------|--------|
| 25 knode fixed (eval-only)| +27.43 ±7.91      | 3,046  |
| 8+0.08 STC                | +2.98 ±2.09       | 30,040 |
| 40+0.4 LTC                | +5.38 ±2.97       | 12,730 |

NPS cost: 13% slower on AVX-512. The headline +30 collapses to +3-5 at
wall-clock TCs because the NPS tax eats most of the eval-quality gain.

That +27 → +3 collapse pattern is the relevant baseline for thinking about
our own king-attacker-exclusion: the eval-only delta from removing king
features will be small (we're not adding features, we're trimming them),
and the NPS-rebate is the more interesting half. Don't oversize the
expectation.
