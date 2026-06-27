# NNUE Inference Deep Audit — 2026-06-26

Read-only structural audit of the NNUE **inference** path (correctness + NPS,
NOT training). Scope: `src/nnue.rs`, `src/sparse_l1.rs`, `src/setwise.rs`,
`src/nnue_simd.rs`. Threat-feature *generation* (`threats.rs`/`threat_accum.rs`)
is owned by another agent and excluded; this audit covers how the
accumulator/forward path *consumes* threats.

---

## Headline

**Correctness verdict: PASS.** The integer inference path (FT accumulate →
pairwise pack → int8 L1 matmul) is bit-identical across AVX2 / AVX-512 /
AVX-VNNI / AVX-512-VNNI / NEON / scalar, and that parity is *enforced at
runtime* (`CODA_VERIFY_NNUE`) and *at compile time* (debug-assert reference
compares). The one genuinely non-bit-identical area is the **float L2→L3→output
tail** (FMA vs scalar mul-add, ISA-dependent reduction order) — sub-centipawn,
known, accepted; it is not a search-tree hazard.

**aarch64/SMP: SAFE.** NNUE holds no shared mutable atomics. Accumulator stacks,
Finny tables, and threat accumulators are all per-thread; net weights are
immutable after load. The only atomics in the path are profiling counters and
load-time feature-flag `AtomicBool`s (both correctly `Relaxed`). No
Acquire/Release obligation is unmet because there is no reader-publish handoff
here.

**Biggest remaining NPS opportunity:** there is no large structural win left in
the *inference kernels* — they are at or ahead of the reference set (native
VPDPBUSD vs Reckless/Obsidian's emulated path). The L1=32 column-major VNNI
kernel that was the last open kernel lever (`l1_32_vnni_kernel_handoff`) is now
**implemented and wired** (`DenseAvx512VnniL1_32` / `DenseAvxVnniL1_32`,
`select_l1_kernel` nnue.rs:2503/2512). The remaining levers are (a) the AVX2
pairwise-pack permute removal mirroring the already-landed AVX-512 no-permute
pack (small, bit-identical, **worth one `[-2,1]` SPRT**), and (b) the
strategic L2→int conversion (§H) which is a *retrain*, not an inference-only
change. The dominant NPS gap vs SF is the **threat accumulator** (~31% of cycles
vs SF ~5.5%), which is deliberate eval richness (X-ray = +187 Elo for ~11% NPS,
SPRT #2014), not waste — and out of this audit's scope.

Net: the inference path is **tight and SF-parity**. This audit found no
correctness defect and no large free-NPS structural lever. Findings below are
ranked; most are "confirmed-correct, no action" with a couple of small,
optional speed probes.

---

## Findings (ranked: correctness before NPS)

### C1 — Float L2/L3/output tail is not bit-identical across ISA (correctness-adjacent, LOW, accepted)

- **Where:** `src/nnue.rs` float L2 path (~3596–3677), CReLU/SCReLU activation
  (~3679–3707), output dot (~3708–3742); `dot_fmadd_avx2_x32` /
  `dot_fmadd_avx512` / NEON FMA vs scalar mul-add.
- **Issue:** The integer pipeline is bit-identical, but the L2→L3→output stage
  uses fused multiply-add and ISA-dependent horizontal-reduction order. Two
  hosts (AVX2 vs AVX-512 vs scalar) can produce sub-cp-different raw outputs.
  After `EVAL_SCALE`/rounding these almost always collapse to the same cp, but
  in principle a boundary case can flip one cp → a different node-count on
  different hardware.
- **Why it's accepted:** This is the same class SF lives with; it is below the
  TT static-eval quantization granularity (13-bit, ±4095cp) and the search is
  robust to ±1cp. `CODA_VERIFY_NNUE` compares incremental-vs-recompute on the
  *same* ISA (catches accumulator bugs), not cross-ISA float order, so this is
  invisible to that harness by design.
- **Fix (only if ever a problem):** the structural cure is §H — make L2 integer
  (int8/int16 matmul) so the whole forward is bit-identical end to end. That is
  a **retrain + quantization-config change**, not an inference-only edit.
  - **Test plan:** N/A as a standalone change (no action recommended now). If
    §H is pursued, it is a net change → `[-1.5, 1.5]` net-vs-net SPRT plus
    `CODA_VERIFY_NNUE` 0-mismatch gate.

### C2 — Integer forward parity is enforced, not assumed (CONFIRMED CORRECT, no action)

- **Where:** runtime `CODA_VERIFY_NNUE` harness (`src/search.rs:1171–1189`,
  `force_recompute` at nnue.rs:~4849); compile-time debug reference compare in
  `forward_with_l1_pairwise_inner` (nnue.rs:~3429–3456, DenseAvx2 vs row-major
  dense reference).
- **Finding:** The saturating-add threat-combine
  (`_mm256_adds_epi16`/`_mm512_adds_epi16`) provably matches the scalar
  i32-add-then-clamp(0,255) (the #1948 latent-correctness fix). Pairwise pack,
  L1 matmul, and accumulate are all covered. This is the strongest part of the
  path. **No action.**

### C3 — Finny-table diff logic is correct (CONFIRMED CORRECT, no action)

- **Where:** `refresh_accumulator` (nnue.rs:~5432–5534), `finny_batch_apply`
  dispatcher (~5633–5669), Finny cache key
  `bucket*(NUM_PIECE_TYPES*64) + pi*64 + ps`.
- **Finding:** `halfka_index_with` (nnue.rs:305–333) takes king square ONLY via
  `king_bucket[ks]` / `king_mirror[ks]`. The Finny cache is keyed per
  (perspective, bucket, mirror), so two king squares that map to the same
  bucket+mirror are genuinely interchangeable for the cached feature rows — the
  diff (entry.piece_bbs vs board bitboards → add_rows/sub_rows) is sound. The
  scalar arm of `finny_batch_apply` is *unconditionally present* (not inside a
  cfg-else that aarch64 could swallow into a silent no-op). **No action.**

### C4 — Lazy materialization gating is correct; pruned nodes do not pay (CONFIRMED CORRECT, no action)

- **Where:** `eval` (search.rs:1158) → `evaluate_nnue` → `materialize`
  (nnue.rs:~5293–5428); `materialize_tt_barrier` (search.rs:1221–1228).
- **Finding:** Materialization is demand-driven: a node that is pruned before
  calling `eval` never forces accumulator realization — push only records
  `DirtyPiece`. The multi-ply walk-back refresh (`replay_ancestor` for
  WHITE/BLACK) is implemented, with a full Finny refresh as the fallback. The
  TT-barrier path (`has_unmaterialized_psq_barrier`) correctly forces
  materialization only when a PSQ barrier is outstanding. This is the SF lazy
  pattern, correctly realized. **No action.**

### N1 — AVX2 pairwise-pack still uses packus+permute; AVX-512 no-permute path already landed (NPS, LOW, OPTIONAL)

- **Where:** AVX2 `simd_pairwise_pack_impl<HAS_THREAT>` (nnue.rs:~866–930) uses
  `_mm256_packus_epi16` + `_mm256_permute4x64_epi64(packed, 0xD8)` to undo
  lane-crossing; the AVX-512 sibling `simd512_pairwise_pack_impl`
  (nnue.rs:~1418–1487) already uses `_mm512_cvtusepi16_epi8` (no cross-lane
  fixup) and that no-permute form measured **neutral** on the AVX-512 fleet.
- **Issue:** The AVX2 pack pays one `permute4x64` per pack iteration to fix
  `packus` lane interleave. There is no AVX2 single-instruction equivalent of
  `cvtusepi16_epi8`, but the permute can sometimes be hoisted/amortized, or the
  pack reorganized so the permute is absorbed into the downstream weight layout
  (pre-permuted weights, paid once at load). This is the AVX2 mirror of the
  AVX-512 no-permute win.
- **Severity:** LOW. The AVX-512 version of this exact change was neutral, so
  expectation here is small. Worth a single cheap probe given it's the only
  remaining bit-identical kernel lever and the AVX2 fleet is the OB measurement
  frame.
- **Fix:** pre-permute the L1 weight columns at load so the pack output lands in
  natural order without the per-iteration `permute4x64`; or batch two pack
  iterations to share one permute. Bit-identical.
  - **Test plan:** PURE SPEED, node-count UNCHANGED. `make && ./coda bench` must
    match HEAD exactly; `CODA_VERIFY_NNUE=1 ./coda bench 8` → 0 mismatches; then
    `[-2, 1]` non-regression SPRT (STC). Do NOT expect a node delta — if bench
    nodes move, the pack diverged and the change is wrong.

### N2 — L1=32 VNNI column-major kernels now present (CONFIRMED DONE, no action)

- **Where:** `select_l1_kernel` (nnue.rs:2500–2514) now dispatches
  `DenseAvx512VnniL1_32` (l1==32, AVX-512-VNNI) and `DenseAvxVnniL1_32`
  (l1==32, AVX-VNNI) above the row-major fallback.
- **Finding:** The open lever in `l1_32_vnni_kernel_handoff_2026-06-15.md`
  (write the column-major VPDPBUSD kernel for 32 neurons so L1=32 stops
  re-scanning input per neuron) is **implemented and wired**. Column-major
  `Dense*` arms are correctly guarded with `!bucketed_hidden` (they read
  `l1_weights_sparse` with total-neuron stride and take no bucket offset; only
  sound for unbucketed nets — documented at nnue.rs:2442–2448). **No action**;
  this finding records that the handoff is closed.

### N3 — Threat accumulator is the NPS gap, but it is eval richness, not waste (OUT OF SCOPE, noted)

- The ~31%-of-cycles threat cost vs SF ~5.5% is the dominant single/contended
  NPS gap. It is validated eval richness (X-ray +187 Elo / ~11% NPS, SPRT
  #2014; recapture-combine double-inc port SPRT #2015). The *generation* side
  is another agent's scope; the *consumption* side (saturating-add combine into
  the FT accumulator, pairwise pack of the threat half) is correct (C2). No
  inference-side waste identified.

---

## What was checked and found clean (no finding)

- **Atomics / aarch64 memory ordering:** only Relaxed profiling counters +
  load-time feature-flag AtomicBools. Per-thread accumulator/Finny/threat
  stacks; immutable shared weights. No reader-publish handoff → no
  Acquire/Release gap. SAFE.
- **int8 quantization saturation:** pairwise pack clamps to [0,255] (u8) and the
  VPMADDUBSW (u8×i8→i16) / VPDPBUSD paths are dimensioned so the i16/i32
  accumulators do not overflow at QA=255/QB=64. Legacy v7 `simd_screlu_pack`
  pushes u8 up to 255 (the classic VPMADDUBSW i16-saturation concern) but v7 is
  non-prod; prod v9 pairwise pack is within range.
- **SCReLU ×0.8 + EVAL_SCALE=400 calibration chain:** consistent across the
  SIMD and scalar activation arms; the scale correction is applied once.
- **Kernel dispatch coverage:** every `L1Kernel` enum arm has a forward match
  arm and a scalar fallback exists for all CPU families (no cfg-swallowed
  else).

---

## Summary of recommended actions

| ID | Type | Action | Test plan |
|----|------|--------|-----------|
| N1 | NPS, optional | AVX2 pack permute removal (mirror AVX-512 no-permute) | bench-identical + `CODA_VERIFY_NNUE` 0-mismatch + `[-2,1]` STC |
| C1 | correctness, deferred | float L2→int (§H) — only via retrain | `[-1.5,1.5]` net-vs-net + verify, if §H pursued |
| C2–C4, N2, N3 | — | confirmed correct / done — no action | — |

**Bottom line:** the NNUE inference path is tight, heavily pre-audited, and at
parity with the stronger reference engines. No correctness defect found; no
large free-NPS structural lever remains. The only actionable inference-only item
is the small, optional, bit-identical AVX2 pack-permute probe (N1).
