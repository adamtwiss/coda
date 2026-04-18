# Coda Blunder Regression Suite

Built 2026-04-18 from 400 recent lichess blitz games (11370 moves analysed,
323 SF-flagged blunders, 116 candidates after filtering, 22 confirmed eval-gap
positions).

## Construction

Pipeline (see `/tmp/coda_blunders/build_epd.py`):

1. Start from all Coda moves SF classified as blunders (cp_loss >= 100)
   from actual lichess games (user account coda_bot).
2. Filter to positions where Coda wasn't already lost/winning
   (|sf_cp_before| <= 500).
3. Run fresh Coda-v5 (/home/adam/code/coda/net-v5-768pw-consensus-w7-e800s800.nnue) at movetime 3000ms on each candidate.
4. Categorize by what fresh Coda plays:
   - Same as game move → **eval_gap** (this suite)
   - SF's best → **state_pollution** (F1/TM territory, not this suite)
   - Different other move → **different_wrong** (excluded for now)
5. For each eval_gap position, re-query SF at depth 22 to get a
   reliable `bm`.
6. Emit as EPD with `bm <sf_best>; am <played>; ...`.

## Category counts for this build

- state_pollution: 70
- different_wrong: 24
- eval_gap: 22

Final EPD size: 22

## Important: state-pollution dominates the blunder mix

Of 116 filtered candidates:
- **70 state_pollution** (60%) — fresh Coda plays the RIGHT move, meaning the
  blunder came from TT/history carried across moves during the game. These are
  the target of the F1 fix (branch `fix/ponder-wait-loop`, merged to main).
- **24 different_wrong** (21%) — fresh Coda plays a different-but-also-wrong
  move. Eval weakness in a different dimension; excluded from this suite.
- **22 eval_gap** (19%) — fresh Coda plays the SAME bad move. This is what
  only a better NNUE can fix. **This suite targets these.**

If you see a losing game on lichess, the 60/20/20 split is a rough prior on
what class it falls into.

## Baselines (2026-04-18, 3s per position, Threads=1 default)

- v5 production (`net-v5-768pw-consensus-w7-e800s800.nnue`): **10/22 (45.5%)**
- v9 SB200 draft (`net-v9-768th16x32-w15-e200s200-xray-fixed.nnue`): **14/22 (63.6%)**

Note: v5 "passes" 10/22 on a suite constructed from v5 failures because
categorization ran at Threads=4 (SMP nondeterminism) while scoring runs at
Threads=1 by default. Positions where Coda's choice is Threads-sensitive
show as "inconsistent with categorization". Despite this, v9 SB200 is
clearly ahead, which is the signal we want.

## How to use

Score with Coda's built-in EPD runner (scores `bm` + `am` since commit e88981c):

```
./coda epd testdata/coda_blunders.epd -t 3000 -n <your_net>.nnue
```

A position passes if engine plays the SF-preferred move AND does not play
the historical blunder move. Both conditions must hold.

For fair comparison across architectures, always use the same `-t` (time)
budget and Threads setting (default 1).

## Purpose

Track eval-quality improvement across Coda net architectures. Each net
version should improve pass-rate on this suite. Position types represent
real failure modes observed in production play on lichess.

## Regeneration

When you've played more lichess games and have more blunder data:

```
# 1. Re-fetch recent games and re-run SF analysis
# (see /tmp/coda_lichess/analyze_blunders.py)
# 2. Regenerate suite
python3 /tmp/coda_blunders/build_epd.py
# 3. Review /tmp/coda_blunders/build_epd.log
```

The suite size will grow (or shrink if F1/v9 fixes existing categories).
