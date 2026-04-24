# v9 NPS Profile (2026-04-17)

`perf record -F 999 --call-graph=fp` on `./coda bench 16` with
`net-v9-768th16x32-w15-e200s200-xray-fixed.nnue` embedded. 9,628 samples over
4.79M nodes at ~480K NPS. Release build with `-Cforce-frame-pointers=yes`
and `debuginfo=1`.

## Top self-time functions

| Rank | % | Function | Notes |
|---|---|---|---|
| 1 | 23.6% | `NNUENet::forward_with_l1_pairwise_inner` | NNUE L1 matmul. SIMD-optimised already. |
| 2 | 20.2% | `SearchInfo::eval` | Wrapper; includes threat accumulator add/materialize. |
| 3 | **10.1%** | **`push_threats_for_piece`** | **X-ray delta generation. Primary target.** |
| 4 | 7.1% | `NNUEAccumulator::refresh_accumulator` | PSQ refresh on KB change. Reckless 10-KB would reduce frequency. |
| 5 | 5.0% | `negamax` | Core search. |
| 6 | 4.0% | `NNUEAccumulator::materialize` | Lazy accumulator. |
| 7 | 2.7% | `MovePicker::next` | |
| 8 | 2.0% | `Board::attackers_to` | |
| 9 | 1.9% | `TT::probe` | |
| 10 | 1.9% | `piece_attacks_occ` | Magic lookups, partly from threat code. |

## Within `push_threats_for_piece` — where the 10.1% goes

Annotated disassembly shows the hot instructions cluster around the
**Section 2b 8-direction raycast** (`for &(df, dr, ortho) in DIRS.iter()`
at threats.rs line ~584):

| % (self) | Instruction / source proximity |
|---|---|
| 9.60% | Section 2b outer loop entry (file/rank bounds reload) |
| 4.54% | Inner ray-walk bounds check `(0..8).contains(&f) && (0..8).contains(&r)` |
| 4.40% | Inner ray-walk `cmp $0x7, %esi` |
| 3.81% | Direction delta load from `DIRS` array |
| 3.60% | `ATTACK_TABLE` indirect load (inside a `piece_attacks_occ` call) |

**Interpretation**: Section 2b alone accounts for roughly half of
`push_threats_for_piece`'s 10.1% of total time — **~5% of CPU time on 2b
alone**. The 8-direction × up-to-7-step scalar ray walks are the bottleneck.
They run on essentially every move (captured piece, moved piece from, moved
piece to → 2-3 calls per move each doing 8 direction walks).

## Primary optimisation candidates

**1. Section 2b early-out (highest ROI)**

Before the 8-direction loop, quick bitboard test: are there any sliders on
any empty-board ray from `square`? If zero sliders could possibly x-ray to
this square, skip the whole loop.

```rust
let empty_ray_mask = rook_attacks(square, 0) | bishop_attacks(square, 0);
let potential_sliders = (pieces_bb[ROOK] | pieces_bb[BISHOP] | pieces_bb[QUEEN]) & empty_ray_mask;
if potential_sliders == 0 { return; }
```

Cheap first cull. Skips section 2b entirely in endgames / positions where
sliders aren't aligned with the changed square. Estimated recovery: 2-4% of
NPS (~6-12 Elo at 100 Elo/doubling).

**2. Section 2b batched raycast**

Replace the scalar 8-direction walks with a bitboard-based "find first two
pieces in each direction" using magic bitboards on `square`:

```rust
// For each direction, rook_attacks(sq, occ) gives squares until first blocker.
// Y is first blocker. Compute rook_attacks(sq, occ ^ Y) to find what's past Y.
```

This replaces ~64 scalar iterations with ~8 bitboard ops per direction group.
Estimated recovery: 2-3% of NPS additional.

**3. Reckless 10-KB layout (orthogonal gain, reduces refresh frequency)**

Planned as T2 experiment. Reduces `refresh_accumulator` calls by ~30-50%
because king moves within ranks 4-8 (lumped into one bucket) don't trigger a
refresh. 7.1% → ~4-5% of CPU time. Estimated recovery: ~2-3% of NPS.

**4. Cache direct-target bitboards across sections 1 / 1b / 2 / 2b**

Section 1 computes `my_attacks & occ`. Section 1b recomputes via
`piece_attacks_occ`. Sections 2 and 2b do similar work on the other side of
the square. Threading a pre-computed cache through would remove ~1-2% of
NPS overhead.

## Estimated recoverable NPS

Stacking the first three items: ~7-10% of NPS → ~20-30 Elo at bullet TC.
Comparable to a successful architecture change, and independent of all the
other gap-closing work.

## Profile notes

- `forward_with_l1_pairwise_inner` at 23.6% is already SIMD-optimised. Const
  generic dimensions were tried and didn't help. L2/L3 layers are small
  enough that further vectorisation has limited headroom.
- `refresh_accumulator` at 7.1% is worth further study after the KB work.
- `piece_attacks_occ` at 1.87% is distributed across many callers; not a
  single optimisation target.
- The 20.2% for `SearchInfo::eval` is partly a framing artefact — it
  includes dispatching to threat + PSQ accumulators and summing them. Most
  of the "own" time is in child functions.
