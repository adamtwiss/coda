# T80 Training Data Analysis (2026-04-16)

Analysis of `test80-2024-01-jan-2tb7p.min-v2.v6.binpack` (5M positions sampled).
Tool: `coda inspect-binpack`.

## Key Findings

1. **Opening positions are included and scored.** Ply 0-15 = 14.3% of data, uniformly
   distributed (~0.9% per ply). Our `ply >= 16` filter discards all of these.

2. **Score distribution is bimodal.** 43.6% of positions have |score| < 100 (roughly equal),
   and 18.3% have |score| >= 32001 (mate scores). The high average |score| of 7850 is
   caused by the 25% mate/near-mate positions.

3. **18.9% of moves are captures.** 6.1% of positions are in check. 0.2% are promotions.
   The remaining ~75% are quiet, non-tactical positions.

4. **Piece count is well-distributed.** Bucket 0 (32 pieces/opening) has 12.6% of data.
   All 8 output buckets get reasonable coverage.

## Ply Distribution

| Ply range | Count | % | Notes |
|-----------|-------|---|-------|
| 0-7 | ~358K | 7.2% | Early opening (filtered by ply >= 16) |
| 8-15 | ~357K | 7.1% | Mid opening (filtered by ply >= 16) |
| 16-31 | ~704K | 14.1% | Early middlegame |
| 32-59 | ~1.13M | 22.5% | Middlegame to endgame |
| **Total < 16** | **715K** | **14.3%** | **Discarded by current filter** |

Distribution is nearly uniform per ply (~0.85-0.90% each), slowly decreasing as
games end. Data quality is consistent across all plies.

## Score Distribution

| |Score| range | Count | % | Notes |
|---------------|-------|---|-------|
| 0-99 | 2,179,814 | **43.6%** | Roughly equal positions |
| 100-499 | 1,115,446 | **22.3%** | Slight advantage |
| 500-999 | 239,941 | 4.8% | Clear advantage (~1 pawn) |
| 1000-1999 | 106,033 | 2.1% | Major advantage |
| 2000-4999 | 51,525 | 1.0% | Winning |
| 5000-9999 | 24,499 | 0.5% | Decisive |
| 10000-19999 | 38,780 | 0.8% | Near-mate |
| 20000-29999 | 329,686 | **6.6%** | Likely decisive/tablebase |
| >= 32001 | 914,276 | **18.3%** | **Mate scores** |

T80 uses LC0 win-probability scoring, not centipawn. Scores above ~10000 represent
near-certain outcomes. The 18.3% mate scores (32001+) provide no useful gradient for
NNUE training and should be filtered.

Our current filter: `score.unsigned_abs() <= 10000` removes 25.7% of data (the
mate/near-mate positions). Raising to 20000 would keep the 6.6% decisive-but-playable
positions while still removing mate scores.

## Position Type Distribution

| Type | Count | % | Notes |
|------|-------|---|-------|
| Quiet (non-capture, non-check) | ~3.75M | ~75% | What our filter keeps |
| Captures | 946,218 | 18.9% | Filtered out (score reflects post-capture state) |
| In check | 306,579 | 6.1% | Filtered out (forced response, noisy score) |
| Promotions | 10,330 | 0.2% | Filtered out (tactical) |

## Output Bucket Coverage (Piece Count)

| Bucket | Piece count | % of data | Notes |
|--------|------------|-----------|-------|
| 0 | 29-32 | 22.1% | Opening/early. Loses data from ply filter. |
| 1 | 25-28 | 13.5% | Early middlegame |
| 2 | 21-24 | 11.8% | Middlegame |
| 3 | 17-20 | 11.8% | Late middlegame |
| 4 | 13-16 | 13.4% | Endgame transition |
| 5 | 9-12 | 12.2% | Endgame |
| 6 | 5-8 | 9.3% | Late endgame |
| 7 | 3-4 | 5.9% | Tablebase territory |

All buckets get reasonable coverage. Bucket 0 is well-populated (22%) so the
ply >= 16 filter is the primary cause of bucket 0 undertraining, not data scarcity.

## Current Filter Impact

Our Bullet training filter:
```rust
entry.ply >= 16                    // removes 14.3%
&& !entry.pos.is_checked(stm)     // removes ~6.1%
&& entry.score.unsigned_abs() <= 10000  // removes ~25.7%
&& entry.mv.mtype() == MoveType::Normal // removes promotions (~0.2%)
&& dest_square_empty               // removes captures (~18.9%)
```

Estimated positions kept: **~50-55%** of raw data (filters overlap — e.g., many
mate-score positions are also captures or checks).

## Recommended Experiments

1. **Remove ply >= 16**: Opening positions are legitimate and well-scored.
   Bucket 0 will get proper training data. Expected: +5-15 Elo.

2. **Keep captures/checks filtered**: These positions have scores that reflect
   post-tactical resolution, not the current position. This is likely where
   the +48 Elo v5 filtering gain came from.

3. **Raise score threshold to 20000**: Keep decisive-but-playable positions
   (6.6% of data) while still removing mate scores (18.3%). Small impact.
