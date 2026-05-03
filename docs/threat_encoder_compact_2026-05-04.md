# Threat encoder compact + importance reorder — design (2026-05-04)

**Goal:** validate Zeus's NPS-from-cache-residency hypothesis for the threat
matrix via a single SB50 K=1 train. NPS is deterministic; K=1 is sufficient
for the NPS question. Elo measurement (K=3 SB200) only proceeds if NPS gains
materialize.

## Two changes bundled into one encoder revision

1. **Phantom cleanup** — the current encoder allocates address-space slots
   that no input tuple `(a_cp, a_sq, v_cp, v_sq, mir, pov)` can produce.
   Brute-force probe found 6,717 unreachable indices (10% of features). Two
   sources:
   - **Semi-excluded pair over-allocation.** Same-piece pairs (N→N, B→B,
     R→R, Q→Q regardless of color, plus P→P enemy-color) are semi-excluded
     in `PiecePair::base()` — the half where `a_sq < v_sq` returns negative.
     But `init_threats` allocates `count = sum_over_a_sq(popcount(attacks))`
     slots, NOT half that. Fix: tighten count for semi-excluded channels.
   - **Pawn-victim on rank 1/8.** A pawn cannot exist on rank 1 or 8.
     Current encoder allocates address space for `(any_attacker_at_*,
     victim=*P, v_sq ∈ {a1..h1, a8..h8})`. Fix: drop those slots from
     pawn-victim channels.

2. **Importance reorder at channel granularity.** The 144 `(attacker_cp,
   victim_cp)` channels currently allocate addresses in `attacker_cp × victim_pt`
   order. Reorder so hot channels (highest total hits on T80 1M sample)
   land at low addresses. This concentrates the hot ~2.57 MB working set
   in cache-friendly low-address territory.

   Within-channel `(a_sq, v_sq)` ordering preserved (existing
   `PIECE_OFFSET_LOOKUP` × `ATTACK_INDEX_LOOKUP` scheme). Reordering at
   slot granularity within a channel adds complexity without changing the
   cache story meaningfully — the top-K hot features cluster in few hot
   channels (wK→wP=2.6M, wQ→bN=971K, etc.); putting those channels first
   captures most of the win.

Combined matrix shrink target: 51 MB → ~46 MB (~10% reduction). Combined
with reorder: top 5% (~2.57 MB) lands at addresses 0..2.57 MB, cleanly L2-
resident on all fleet hosts.

## Channel ordering source

Use total hits per `(attacker_cp, victim_cp)` pair from
`/tmp/threat_hits_t80_1m.csv` aggregated to channel level. Validated stable
across data sources — top 5% feature-set has 86.5% overlap between T80 and
self-play sample (rank shuffles within, set membership stable).

Tabulate channel hits, sort desc, that's the new order.

## Lockstep mechanism

Bullet's training enumeration MUST produce the same indices as Coda's
inference. Three layers of defence:

1. **`training_flags` bit 1 = `compact_encoding`** (added at v10). New bit
   in the existing v10 byte at `src/nnue.rs:2251`. No version bump. Old
   Coda binaries reading new nets fail at the threat-weight size check
   (compact has fewer rows) — explicit failure, not silent corruption.
2. **`num_threat_features`** in the .nnue header (already present) — a
   second cross-check that catches encoder size mismatches even if
   `training_flags` was misread.
3. **Fuzzer** — extended to test both classic and compact encoders. Must
   pass before any retrain.

`convert-bullet` gets a `--compact-encoding` CLI override following the
`--xray-trained` / `--kb-layout` precedent (read flag from quantised.bin,
allow CLI override, refuse on mismatch unless `--load-anyway`).

## File layout

### Coda (`/home/adam/code/coda/`)

- `src/threats.rs`:
  - Existing `init_threats()`, `threat_index()`, `enumerate_threats()`,
    `threat_index_bullet_ref()` stay as the **classic** path.
  - Add `init_threats_compact()` builds parallel `COMPACT_*` static
    tables. Always called at startup (~110 KB extra static data,
    negligible).
  - Add `threat_index_compact()` — same algorithm, dispatches to
    `COMPACT_*` tables.
  - Add `enumerate_threats_compact()` — same enumeration loop, calls
    `threat_index_compact()`.
  - Add `num_threat_features_compact()`.
- `src/nnue.rs`: `NNUENet` gets a `compact_encoding: bool` field set from
  `training_flags` bit 1. Threat operations (in `threats::push_threats_*`,
  `apply_threat_deltas`) branch on this flag.
- `src/threat_accum.rs` / search call sites: pass the flag through to
  enumeration.
- `src/bullet_convert.rs`: read flag from `quantised.bin`, write
  `training_flags` bit 1, add `--compact-encoding` CLI flag.
- `src/main.rs` (`convert-bullet`): add `--compact-encoding` arg.
- Fuzzer (`run_fuzz_threats`): extend to test both encoders.

### Bullet (`~/code/bullet/`)

- `examples/coda_v9_768_threats.rs`: add `--compact-encoding` CLI flag.
  Pass to threat-enumeration code.
- Bullet's threats module (mirror of Coda's): add compact encoder. Must
  produce **bit-identical indices** to Coda's `threat_index_compact()`.
- Save flag into `quantised.bin` header.

## Implementation order

Each step verifiable before proceeding.

1. ✅ **Coda: `init_threats_compact` + `threat_index_compact`** —
   committed `06ba067` (skeleton) + `af39e7b` (channel reorder).
   Bench unchanged at 966720; phantom cleanup deferred.
2. ✅ **Coda: `enumerate_threats` dispatches** — both encoders use the
   shared loop. Runtime dispatch via `is_compact_encoding()` flag. No
   separate `enumerate_threats_compact` needed.
3. ✅ **Coda: fuzzer extension** — `dispatch_classic_vs_compact_real_positions`
   in `src/threats.rs` (commit `471a019`) covers six varied
   positions × both POVs, asserts same feature counts and consistent
   classic→compact mapping with injectivity. Plus
   `compact_encoder_permutes_classic` for canonical-state encoder coverage.
4. ✅ **Coda: NNUE load reads `training_flags` bit 1** — `NNUENet`
   gains `compact_encoding`, validate_compat() refuses size-mismatched
   nets, search.rs::load_nnue activates compact dispatch (commit
   `cd9ca02`).
5. ✅ **Bullet: mirror the encoder** — Bullet branch
   `feature/threat-encoder-compact`, commit `756137f`.
   `ChessBucketsWithThreats::new_with_xray_compact()`, dual OnceLock
   singletons, CHANNEL_ORDER pidx-form translated from Coda's table.
   Cannot build/test on Hercules (CUDA absent); GPU-host smoke test
   queued.
6. ✅ **convert-bullet: read flag, set `training_flags` bit 1.**
   `--compact-encoding` CLI flag, propagated through `convert_v7`
   (commit `874c7c7`). Default false; existing nets convert unchanged.
7. ✅ **Smoke test (Hercules-side)**: prod net loads + benches at
   966720 identically with new dispatcher (compact_encoding=false
   path).
8. ✅ **SB10 smoke train on GPU host** — compact-encoded SB10 loaded
   and benched cleanly (encoder roundtrip works end-to-end). Bench
   showed 4.9M nodes / 812K nps vs prod-classic's 966K nodes / 588K
   nps. **NPS difference (+38%) was training-stage artefact, not
   encoder.** Confirmed by the permute-net diagnostic below.
9. ✅ **`coda permute-net-compact`** (commit `31ec0e8`) — diagnostic
   tool: row-permutes classic prod's threat weights into compact
   layout, producing a compact-flagged net with bit-identical
   accumulator outputs. Bench node count exactly matches classic
   (966720 == 966720), proving the permutation preserves inference
   semantics. **NPS: classic 587K vs permuted 578K (3-run medians) —
   encoder reorder alone is ~0% within run-to-run noise.**
10. ❌ **SB50 train + NPS comparison cancelled.** Permute-net result
    rules out an encoder-cost win at this hardware/workload. The
    cache-residency hypothesis was that hot channels at low addresses
    would reduce L3 spill, but: (a) channel reorder doesn't change
    matrix size (51 MB → 51 MB) so the spill envelope is unchanged;
    (b) threat lookups are pseudo-random per position, so CPU
    prefetchers don't benefit from contiguous hot blocks; (c) LRU
    keeps frequently-accessed rows resident regardless of address.

## Result — channel reorder alone delivers ~0% NPS

Negative result on the channel-reorder portion of the compact encoder.
The encoder permutation is correct (bit-identical search trees confirm
this), but the cache-residency lever lives in matrix shrink, not
address reordering.

Lessons:

- **The 38% lift in the SB10 vs prod bench was tree-shape artefact.**
  Under-trained eval → 5× wider trees → more QS / shallow nodes / more
  incremental updates → cheaper per-node NNUE work → higher NPS.
  Generalises: **never read NPS across nets at different training
  stages**; matched-stage comparison or row-permutation is required.
- **Channel reorder doesn't help random-access matrices.** Threat
  lookups are pseudo-random by position; address ordering doesn't
  change cache hit rate after warm-up. Prefetcher-friendly contiguous
  layouts only help linear scans.
- **Cache residency requires shrink, not rearrangement.** The 51 MB
  threat matrix exceeds L3 on most fleet hosts; that's the spill
  problem. Reorder doesn't change the working-set size; only fewer
  rows or smaller rows do.
- **Row-permute trick is a generalisable diagnostic.** Any
  "permutation only" encoder change (channel order, feature reorder,
  bucket reorder) can be A/B'd without retraining via this technique.
  Tool kept for future use.

## Where the cache lever actually lives

Three threads with real shrink leverage, ranked by leverage:

1. **Group-lasso L1 sparsity** — already in flight. Targets ~30% row
   reduction (51 MB → ~36 MB). Crosses 32 MB L3 boundary on most
   fleet hosts. Real cache-residency effect plausible.
2. **Hidden size 768 → 512** (Zeus's idea 3) — 33% cut by hidden
   shrink (51 MB → 34 MB). Tabled until group-lasso resolves.
3. **Phantom cleanup** (the deferred half of this thread) — 5%
   matrix shrink (51 → ~48 MB), unlikely to cross any cache boundary.
   Skip unless a dependent feature requires the index-space reduction.

## Known risks

- **Silent encoder drift between Bullet and Coda.** Same failure class as
  C8 x-ray (~110 Elo). Fuzzer is the canonical defence — must pass before
  any retrain.
- **Importance ranking shift after retrain.** Ranking is from current
  prod's hit pattern. New net with compacted encoder may have a slightly
  different hot-feature distribution. Acceptable for first probe; if
  ranks drift materially, a second iteration on the new net's hits can
  refine.
- **Within-channel ordering is unchanged.** Hot rows within a hot channel
  may not be at the channel's lowest addresses. The cache-residency
  benefit is at the channel-cluster granularity, not slot-perfect. Good
  enough for first probe.

## Out of scope (deferred)

- Within-channel reorder (hot v_sq within a channel at low offset).
- Encoder hash field for paranoid validation (requires v11 bump).
- Group-lasso interaction. The +11.22 Elo SPRT was K=1 — possibly seed
  noise. Don't condition the compact-encoder design on group-lasso
  assumptions.
- Hidden size 768 → 512 (idea 3). Tabled until idea 1 + 2 land.
