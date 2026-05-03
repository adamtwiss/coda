# Byteboard Splat Port — Scoping (2026-05-03)

Companion to `docs/nps_structural_findings_2026-05-01.md` and the
just-merged cache-hygiene leg (#921 H1 +6.77 Elo). This doc scopes
the next NPS lever identified after the threat-delta-count
diagnostic: porting Reckless's "byteboard splat" SIMD threat-delta
enumeration.

## Context

After the cache-hygiene leg landed (#921 +6.77 Elo) and the v9-hot
+ v7 memset fixes shipped (`81d494e`), eval-bench L1 misses are
down 304 M → 98 M (-68%). The remaining cache gap to Reckless is
12× (from 38× originally).

Per-callsite decomposition shows the two biggest remaining hotspots:
- `forward_with_l1_pairwise_inner` — actual VPDPBUSD + L2 dequant
  work, hard to reduce further on the matmul side
- **Threat-delta enumeration + apply** — Reckless's batched SIMD
  beats Coda's per-piece scalar pattern on cache locality

The threat-delta-count diagnostic (`bench --features profile-threats`)
showed Coda applies ~10.6 deltas per push call (12.96 M total deltas
in a 966720-node bench). Likely similar volume in Reckless. The
cache benefit is from access *order*, not call count.

## What Reckless does differently

**Coda (current):**
- `push_threats_for_piece(piece, sq, ...)` — scalar enumeration
  over magic bitboard lookups for each piece change
- Emits `RawThreatDelta` entries to `board.threat_deltas: Vec`
- Apply path: `apply_threat_deltas` loops over deltas, computes
  `threat_index()`, applies weight rows one at a time

**Reckless:**
- `push_threats_on_change(piece, sq, add)` — SIMD batched
  enumeration over the entire mailbox in one pass
- Uses a static `RAY_PERMUTATIONS[64][64]` table to permute the
  mailbox into ray-aligned order (`__m512i` of bytes)
- `closest_on_rays`, `attackers_along_rays`, `sliders_along_rays` —
  derive all attackers/attacked along all 8 rays in batched SIMD
- `splat_threats` / `splat_xray_threats` emit `ThreatDelta` records
  via `_mm512_maskz_compress_epi8` into the delta ArrayVec
- Apply path is structurally identical to Coda's — load parent,
  apply add/sub weight rows, store

The cache benefit comes from:
1. **Single SIMD pass over the board** vs Coda's per-piece magic
   lookups (much less branching, much less scalar bitboard ops)
2. **Ray-grouped delta emission** — consecutive deltas in the
   ArrayVec correspond to spatially adjacent threat features. Even
   if the apply path itself is structurally similar, accessing
   weight rows in this order is more cache-friendly than Coda's
   scattered per-piece ordering
3. **No `Vec<RawThreatDelta>` heap allocation** — Reckless uses
   `ArrayVec<ThreatDelta, 80>` inline in the accumulator

## Source files to port

| Reckless file | LoC | What it provides |
|---|---:|---|
| `src/nnue/accumulator/threats/vectorized.rs` | 191 | Static tables (`RAY_PERMUTATIONS`, `RAY_ATTACKERS_MASK`, `RAY_SLIDERS_MASK`, `PIECE_TO_BIT_TABLE`) + dispatch wrappers |
| `src/nnue/accumulator/threats/vectorized/avx512.rs` | 198 | AVX-512 SIMD primitives + `splat_threats` / `splat_xray_threats` / `splat_xray_threats2` |
| `src/nnue/accumulator/threats/vectorized/avx2.rs` | 178 | AVX-2 fallback (8 helper fns mirroring avx512.rs) |
| `src/board.rs` `mailbox_vector_*` | ~10 | Tiny — load mailbox as `__m256i` / `__m512i` |
| **Total** | **~580 LoC** | + tests |

Plus integration into Coda's threat path (replace
`push_threats_for_piece`).

## Effort estimate

| Phase | Scope | Effort | Expected gain |
|---|---|---|---:|
| A | AVX-512 path: tables + ray helpers + splat fns + integration | 2 days | +2-3 Elo on AVX-512 hosts (Reckless's `42399e79` headline +4.33 STC; we'd see less because our other code is already faster) |
| B | AVX-2 fallback for non-AVX-512 hosts | 1.5 days | +1-2 Elo on AVX-2-only hosts (Reckless's `45d9cc5a` +2.27 STC) |
| C | NEON port for ARM | 1 day | Variable; defer |
| D | Correctness verification (parity vs scalar) + SPRT | 0.5 day | — |

**Realistic total: 4-5 days for a deployable Phase A+B+D.**

## Gotchas / risks

1. **Threat-index encoding compatibility.** Coda's `threat_index`
   has a "Bullet semi-exclusion" check that Reckless's lacks. The
   byteboard splat emits raw `(piece, from, attacked, to)` tuples;
   the encoding into a feature index happens in
   `apply_threat_deltas`. As long as Coda's existing `threat_index`
   is plumbed in correctly, this should be transparent. Need to
   verify with parity tests against the scalar path.

2. **Mailbox encoding.** Reckless's `Piece` enum has a specific
   ordering (`Piece::None` = 12, etc.) that the
   `RAY_ATTACKERS_MASK`/`RAY_SLIDERS_MASK`/`PIECE_TO_BIT_TABLE`
   tables depend on. Coda's `mailbox` uses different encoding
   (`u8`, with NO_PIECE = 14). The static tables would need
   re-deriving or Coda's mailbox would need a re-encoding pass.

3. **AVX-512 VBMI2 dependency.** Reckless's avx512.rs uses
   `_mm512_permutexvar_epi8`, `_mm512_maskz_compress_epi8`,
   `_mm512_test_epi8_mask` — all VBMI2. So this is a Sapphire
   Rapids+ / Zen 4+ feature path. AVX-512 hosts without VBMI2
   (Cascade Lake-X) would need a different specialisation. We
   already use VBMI2 elsewhere (commit `946763e` find_nnz scan)
   so this is not new.

4. **Reckless's threat encoding excludes king-mirror crossing.**
   Their `push_threats_on_move` doesn't handle king-file crossings
   that change `mirrored`. Coda's enumeration does this implicitly
   via the king-bucket Finny path. The byteboard splat is for
   non-king-crossing moves; we'd keep our existing scalar path as
   the king-crossing fallback.

5. **AccDataStack interaction.** Our just-landed AccDataStack
   refactor changed how the threat accumulator is stored
   (separate `Box<[i16]>` per stream). The byteboard splat writes
   `ThreatDelta` records to a Vec — independent of accumulator
   layout — so no conflict.

## Decision points

Before starting Phase A, we should decide:

1. **Mailbox re-encoding vs table re-derivation.** Re-encoding
   the mailbox to match Reckless's piece IDs is a small one-time
   change but touches every place that reads mailbox values
   (movegen, eval). Re-deriving the static tables in Coda's piece
   IDs is a one-time table-generation script. **Recommendation:
   re-derive tables — keeps Coda's existing mailbox conventions.**

2. **AVX-512 only first, or AVX-2 same commit?** Reckless's
   commits split them (`42399e79` AVX-512 first, `45d9cc5a` AVX-2
   later). For Coda, AVX-2 hosts (Hercules, Atlas) showed the
   biggest gain from the cache-hygiene leg — they'd benefit most
   from AVX-2 byteboard splat. **Recommendation: AVX-512 first
   for development simplicity, then AVX-2 follow-up. Don't ship
   AVX-512 alone — would shift fleet results away from older
   hosts.**

3. **SPRT bounds.** Per Reckless's commit messages, headline
   gains were +4.33 + +2.27 = +6.6 Elo combined. Adjusting for
   Coda being ahead on related cache work (lower upside left),
   estimate +2-4 Elo. SPRT bounds `[0, 5]` (structural NPS class).

## Recommendation

This is a 4-5 day port, not a single-session task. It's the
clearest remaining cache-side NPS lever after the just-merged
hygiene leg. Go/no-go question:

- **Go:** start Phase A (AVX-512 splat + tables + integration).
  Iterate to passing parity tests, then add AVX-2 fallback, then
  SPRT.
- **No-go:** park byteboard splat, focus on something else (e.g.
  Step C-full inline-barrier removal, more memset waste in non-hot
  paths, hot-feature frontload). The +2-4 Elo from this is real
  but not the cheapest remaining lever.

Other levers in the queue (rough ROI ordering):
- **Hot-feature frontloading** — rank features by activation count,
  permute the 49 MB threat matrix at load. ~100 LoC, no SIMD work.
  Targets the same cache axis as byteboard splat. Half-day.
- **Step C-full** — drop `#[target_feature]` barriers from outer
  SIMD functions. Targets the 1.79× instruction-count gap (the
  axis the cache work doesn't touch). 2-3 days, multi-binary
  build complications.
- **Push-path delta-row count instrumentation** — done; revealed
  threat-side dominance. No follow-up except byteboard splat.

Lowest-hanging fruit ordering: hot-feature frontloading first
(half day), then byteboard splat (4-5 days), then Step C-full
when ready for the bigger investment.
