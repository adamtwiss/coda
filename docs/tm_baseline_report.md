# TM Baseline Report (2026-04-09)

10 games per TC against Stockfish, local cutechess-cli.

## Summary: Coda vs SF

| TC | Coda Usage | SF Usage | Coda Open | SF Open | Coda Mid | SF Mid | Coda @40 | SF @40 |
|----|-----------|---------|-----------|---------|----------|--------|----------|--------|
| 3+2 | 86% | 89% | 10.3s | 6.9s | 5.0s | 6.5s | 31s | 32s |
| 3+0 | **89%** | 74% | **7.3s** | 3.7s | 3.6s | 3.5s | **25s** | 58s |
| 5+0 | **90%** | 82% | **12.7s** | 6.5s | 5.6s | 6.1s | **42s** | 78s |
| 1+1 | 92% | 92% | 3.6s | 2.5s | 2.3s | 2.6s | 7s | 11s |

## Key Issues

### 1. Opening Overspend (all TCs)
We spend 50-100% more time than SF on moves 1-10. These moves are
often semi-forced (EPD start positions are ~move 3-5) and don't
benefit from deep search as much as middlegame positions.

- 3+2: Coda 10.3s vs SF 6.9s (49% more)
- 3+0: Coda 7.3s vs SF 3.7s (97% more)  
- 5+0: Coda 12.7s vs SF 6.5s (95% more)

### 2. No-Increment Over-Aggression (3+0, 5+0)
At 3+0 we use 89% of time (25s left at move 40).
SF uses 74% (58s left). Without increment, we're risking time trouble
while SF maintains a comfortable buffer.

### 3. Middlegame Deficit (3+2, 5+0)
SF spends more per middlegame move than us despite spending LESS
in the opening. The time saved in opening goes directly to middlegame
depth.

- 3+2: SF 6.5s vs Coda 5.0s (SF +30%)
- 5+0: SF 6.1s vs Coda 5.6s (SF +9%)

### 4. 1+1 is Well Calibrated
Our best TC match. Nearly identical to SF on all metrics.
The short base time naturally prevents opening overspend.

## Targets for Redesign

| TC | Current Usage | Target Usage | Current @40 | Target @40 |
|----|--------------|-------------|-------------|-----------|
| 3+2 | 86% | 90% | 31s | <27s |
| 3+0 | 89% | 77% | 25s | <54s |
| 5+0 | 90% | 80% | 42s | <75s |
| 1+1 | 92% | 90% | 7s | <9s |

## Root Cause

The base allocation formula `time_left / 25 + inc * 0.8` doesn't
account for game phase. It gives the same per-move budget at move 5
(opening) as at move 25 (critical middlegame). Combined with the
TT being cold at game start (no history, no TT entries), the engine
spends heavily on the first few searched moves.

Phase-based allocation (Reckless pattern) would naturally fix this:
less at low move numbers, more as the game progresses.
