# AVX2-host NPS gap audit — 2026-07-03

**Question (Adam):** on AVX-512 hosts Coda's NPS is now on par with Stockfish;
on AVX2-only hosts we're still ~15-20% slower. Are the AVX2 fallback paths
missing tricks? And is the gap partly memory/cache — the model is large and
the active working set is big?

Three parallel investigations: (A) audit of Coda's own AVX2 vs AVX-512 kernel
variants, (B) a cross-engine catalog of AVX2 NNUE tricks from
Stockfish/Reckless/Obsidian/Berserk/PlentyChess/Alexandria, (C) a memory/cache
working-set analysis. Summary of each, then the merged ranked plan.

---

## Verdict in one paragraph

The AVX2 fallbacks are **not structurally neglected** — every hot kernel has a
tuned AVX2 variant with the same fused/one-pass/const-tiling architecture as
the AVX-512 path, and AVX-VNNI hosts (Alder Lake+, Zen 4 "AVX2" parts) already
get fused 256-bit dot products. A large share of the 15-20% gap on true
AVX2-only silicon is hardware-inherent (no VNNI: the L1 matmul costs 3
instructions where AVX-512 pays 1; half register width everywhere). The
**memory hypothesis is directionally refuted** by existing same-host data:
Coda moved *less* DRAM per node than SF on the AVX2 box (102 vs 115 B/node,
`docs/coda_vs_sf_speed_2026-06-14.md`), SF's resident matrices are as big or
bigger than ours (~111 MB vs ~81 MB — SF master now has threat features too),
and the measured gap was **instructions/node-shaped**, not miss-shaped.
However, all three investigations surfaced real, mostly-cheap items: two
kernel-level consensus tricks we genuinely lack (mulhi pairwise pack +
load-time packus permutation; maddubs-pair fusion), one true tiling asymmetry
(threat-apply REGS=8), several free memory-locality wins (hugepages +
alignment for the weight matrices), and one **invalidated prior conclusion**
(the sparse-L1 rejection compared against a scalar NNZ harvest — the
consensus fused movemask+LUT form was never tested).

---

## A. Coda's own AVX2 paths — what's already right

Confirmed compliant with top-engine consensus (no action):

- **Fused out-of-place add/sub accumulator update** (6/6 engines do this):
  `simd_acc_fused_avx2` `nnue.rs:391`, REGS=12 const-tiled — matches SF's
  deliberate 12-of-16-YMM budget exactly.
- **Column-major broadcast-input L1=32 kernel** (6/6): `sparse_l1.rs:245`,
  same 4-byte-interleave layout as SF/Obsidian; no horizontal adds until the
  scalar output.
- **AVX-VNNI (256-bit VPDPBUSD) dispatched ahead of plain AVX2**
  (`nnue.rs:2298-2302`) — Alder/Raptor-Lake and Zen 4 already get fused dots.
- **int8 threat weights widened via `cvtepi8_epi16` at update time** (3/3
  threat engines): `threats.rs:1982+`.
- **MaybeUninit scratch everywhere hot** (the 4 KB/eval memset was banked
  earlier); **compile-time-foldable dispatch** on native builds (the
  target_feature inline-barrier fix landed in the Step-C work).
- **No structural deficiency in the PSQ fused update or Finny apply** —
  AVX-512's advantage there is pure width.

Known-dead ends (do NOT revisit without new evidence): software prefetch in
apply loops (SPRT H0 #719/#721); the *old scalar-harvest* sparse L1 (but see
item 7 — the fused form is a different animal); merged-maddubs without a
quantization-headroom check; REGS=24 threat apply on AVX-512 (spilled, −4.4%).

## B. Consensus tricks Coda is missing (cross-engine catalog)

1. **mulhi pairwise pack with clamp elision** (5/5 pairwise engines: SF,
   Reckless, Obsidian, PlentyChess, Alexandria). Coda's pack
   (`nnue.rs:848-877`) fully clamps both operands, then `mullo`+`srli`.
   Consensus: full clamp on the first operand only, `min`-only on the second,
   `slli(16-FT_SHIFT)` + `mulhi` (folds the shift, preserves sign so `packus`
   zero-saturates negative products). Saves 2 `vpmaxsw` per 32 outputs.
   SF pre-scales FT weights ×2 at load to keep the shifted value out of the
   sign bit (`nnue_feature_transformer.h:288-297`); Reckless does the same
   shape with `MUL_HI_SHIFT=0` on AVX2.
2. **Load-time packus lane permutation** (4/6: SF, Obsidian, Berserk,
   PlentyChess; Reckless also pays the runtime permute, so we're in OK
   company). Coda issues `_mm256_permute4x64_epi64(packed, 0xD8)` per packed
   vector (`nnue.rs:867,873`) — 32 port-5 lane-crossing shuffles per eval.
   Permuting FT weights/biases once at load ({0,2,1,3} in 128-bit blocks)
   makes packus output land in natural order. Must be applied consistently:
   pack path, column-major L1 chunk order (`sparse_l1.rs:36-68`), the
   scalar debug check (`nnue.rs:3291-3319`), and the threat accumulator's
   contribution to the same pack. NOTE: the analogous AVX-512 change
   measured neutral on Zen 5 — this is an AVX2-host-specific candidate.
3. **maddubs-pair fusion ("double_dpbusd", 4/6: Reckless, Berserk,
   PlentyChess, Alexandria)**: sum two `maddubs` results with `add_epi16`
   *before* one shared `madd(ones)` — halves the madd count in the VNNI
   emulation (3 uops → ~2.5 per chunk). **Precondition**: quantization
   headroom. Alexandria's static_assert is the safety bound:
   `max|w| × (FT_QUANT²>>FT_SHIFT) × 4 ≤ 32767` → with our FT term = 127,
   requires max|l1_weight| ≤ 64. Our L1 weights are QB=64-scaled i16 clamped
   to i8 at load (`nnue.rs:2733-2735`) — nominal scale says ≤64 but training
   clipping determines the real max. **Gate the kernel on a load-time
   max|w| scan** (fall back to the current kernel if violated) rather than
   assuming.
4. **Threat-apply AVX2 tiling REGS 8→12** (`threats.rs:1997`): the one real
   tiling asymmetry vs both our own PSQ kernel (REGS=12) and SF's AVX2
   budget (NumRegs=12). h=1024 currently takes 8 outer passes, each
   re-walking the delta index lists; 12 regs → 5 passes + const-4 tail.
   Widening loads fold, so 12 accs + temps fits the 16-YMM file. Threat
   apply is inside the largest cycle block (~20-30% incl. replay).
5. **Residual per-call `is_x86_feature_detected!`** in `apply_threat_indices`
   (`threats.rs:1936-1950`) and `add_weight_rows` (`threats.rs:2181-2196`) —
   two failed AVX-512 atomic checks per call on AVX2 hosts, before falling
   into the (compile-time-true) AVX2 arm. Hoist to cached flags (the
   `NNUENet` bools already exist).

## C. Memory/cache findings

Working set (from struct definitions, prod E6C62000 kb10 net):
threat weights **65.3 MiB** (66864×1024 i8) + PSQ weights **15.0 MiB**
(10×768×1024 i16) ≈ 81 MiB resident; per-thread mutable ~7.4 MiB; a
materialized node touches ~28-32 KiB (~450-500 cache lines), dominated by
~20 random 1 KiB threat rows. BUT: caches absorb ~99% of that even on the
AVX2 host (102 B/node net DRAM), and SF's equivalent set is bigger. The gap
is not "our model is too big" — it's per-node instruction count on
non-VNNI silicon, plus the non-SIMD structural items already documented in
`docs/coda_vs_sf_speed_2026-06-14.md` (threat-index table chase, chain-break
rebuild rate).

Genuinely unexploited locality items (all bit-identical, cheap):

6. **Hugepage-back the two weight matrices.** TT already has the full 3-tier
   hugetlb/THP allocator (`tt.rs:115-240`); the 65 MiB threat matrix is a
   plain `Vec<i8>` (`nnue.rs:2624`) and PSQ an `AlignedVec` on 4 KiB pages.
   ~20 random rows/node over ~16,700 4-KiB pages is real STLB pressure on
   older cores (1-2K entry STLB); 2 MiB pages collapse the matrix to ~40
   pages. Reuses proven TT code. Cheap A/B first: prototype
   `madvise(MADV_HUGEPAGE)` on the two allocations and compare
   dTLB-load-misses/node on an AVX2 host.
7. **64-B-align `threat_weights`** (`Vec<i8>` → `AlignedVec<i8>`,
   `nnue.rs:2624`): unaligned base makes every 16-line row straddle 17 lines
   (+6% lines on the largest miss source). One line. Free rider on #6.
8. **Sparse L1 re-test with the consensus fused-NNZ form.** The 1.8-2.4×
   "sparse loses" measurement (`sparse_l1.rs:1-17`) compared against a
   scalar separate-pass `find_nnz_chunks` (`nnue.rs:986-1001`). All six
   reference engines fuse NNZ extraction into the activation while the
   packed vector is in-register (cmpgt+movemask + 256-entry `[u16;8]` LUT
   branch-free decompress) — including Reckless on our exact architecture
   shape. Re-measure actual pairwise-output density for the prod net first;
   if it's genuinely ~58-60% dense, sparse still loses and this dies quickly.
9. **Threat delta add/sub cancellation** — instrumentation already counts
   net-zero add/sub pairs as wasted bandwidth (`threats.rs:157-161`, "SF
   cancels these"); each cancelled pair saves 2 KiB of random streaming.
10. **Generation-time threat-row prefetch** (make_move /
    `store_threat_deltas`, `nnue.rs:4648`): lazy materialization gives ~1.24
    plies of lead time the apply-entry prefetch can't use. The FT-gather
    analog was a no-op on Zen 5 (32 MB L3) — this is an AVX2-host-only
    experiment; measure there or don't bother.
11. **Hot-feature frontloading is NOT implemented** (contrary to a stale
    task-list entry): loader reads threat rows in file order
    (`nnue.rs:2620-2633`); the permutation was scoped
    (`docs/byteboard_splat_scoping_2026-05-03.md:161-171`) and held. Tooling
    (activation counters, `threat_accum.rs:1368-1449`) exists. Second-order
    after hugepages.
12. **Footprint trims** (~1-2 MiB/thread): the threat accumulator state is
    held twice (`NNUEAccumulator.threat` AccDataStack `nnue.rs:4552` AND
    `SearchInfo::threat_stack` `search.rs:917`) — audit whether the former's
    replay path is dead in search; Finny slab allocates 64 slots where kb10
    uses 40; `ThreatEntry.values` hard-sized at 1024 regardless of net width.
    Matters mainly at high thread counts.

---

## Merged ranked plan

Tier 1 — trivial, bit-identical, do as one branch (verify NPS on an AVX2
host, `[-2,1]` SPRT):
- (5) hoist per-call ISA detection in threat apply/refresh
- (7) 64-B-align threat_weights
- (4) threat-apply REGS 8→12 (verify no spill with perf annotate)
- (6) hugepage the weight matrices via the TT allocator (with the madvise
  A/B measurement first if convenient)

Tier 2 — kernel changes, each bit-exact-verifiable against the scalar path,
separate branches:
- (1)+(2) mulhi pairwise pack + load-time packus permutation (do together —
  the weight-layout change serves both)
- (3) maddubs-pair fusion gated on a load-time max|w| ≤ 64 scan

Tier 3 — measure-first experiments:
- (8) fused-NNZ sparse L1: measure prod-net pairwise-output density first
- (9) delta cancellation; (10) gen-time prefetch (AVX2 host only)
- (11) hot-feature frontloading; (12) footprint trims

Measurement discipline for all of it: per the standing policy, primary
metrics are **insns/eval and cache-refs/eval** (plus dTLB-misses/node for
Tier-1 item 6), measured on an AVX2-only host with the OB worker stopped,
with an SF bench on the same box as the control. The decision matrix:
insns/node equal + misses higher ⇒ memory-bound (Tier-1 locality items
matter most); insns/node higher + misses similar ⇒ compute-bound (Tier-2
kernel items matter most). Existing Hercules data says the latter.
