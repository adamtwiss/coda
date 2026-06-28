# NNUE Inference Profile — v8s3 net (2026-06-28)

Fresh profiling of the inference hot path on the **v8s3 prod-beater net**
(`multi-v8-l132-s3-v3-swa.nnue`, FT=1024 / L1=32, v9 inference format).
Host: **AMD Ryzen 7 9700X** (Zen 5, AVX-512 + AVX-VNNI), single-thread,
**OB worker stopped** for clean numbers. `coda bench 13`, ~1.84M NPS.

**Companion / supersedes-context:** the authoritative Coda-vs-SF speed analysis
is `docs/coda_vs_sf_speed_2026-06-14.md` (read it first). This doc re-confirms
its findings on v8s3 and adds AVX-512-host detail + a tested (negative) prefetch
experiment. Correctness of the threat path: `docs/threat_pipeline_deepaudit_2026-06-26.md`.

## Important: SF has the SAME architecture (FT1024 + threats)

**Stockfish has threat features too** — same FT1024 + threats arch as Coda.
SF does NOT do *x-ray* threats; Coda does. So the speed gap is **implementation
efficiency, not "SF lacks threats."** Per `coda_vs_sf_speed_2026-06-14.md`:
Coda's threat accumulator is **~31% of cycles vs SF's ~5.5%** for the same
feature class. Two drivers:
- **X-ray threats (Coda-only):** ~11% single-thread / ~14% contended NPS cost,
  but only ~25% of the total gap. **Kept** — SPRT #2014 = +187 Elo for ~3-5
  deployment Elo of NPS. Not a drop candidate.
- **The threat-accumulator APPLY design (the real, untouched lever):** Coda
  replays deltas from an ancestor and re-derives indices; SF builds a
  dirty-threat list once inside `do_move` and applies once. This is the bulk
  (~75-85%) of the gap and is independent of x-ray.

## Headline: memory-bound, and AVX-512 is already saturated

`perf stat` (depth 13, v8s3):

| metric | value | read |
|---|---|---|
| IPC | 2.20 | OK |
| **L1-dcache load-miss** | **21.95%** | high — memory-bound |
| cache-miss (→LLC/RAM) | 5.20% of refs (279M) | the expensive ones |
| branch-miss | 2.69% | fine |
| NPS (1T) | 1.84M | — |

Dispatch log confirms **AVX-512 + VNNI active** ("VPDPBUSD int8 matmul"). The
hot kernels already run 512-bit with maximal register blocking (REGS=24). **A
code-survey pass claimed two AVX-512 gaps — both are already closed and do NOT
apply to v8s3** (verified vs live profile): v8s3 uses the fast column-major
`dense_l1_avx512_vnni_l1_32` (9.2% self), and `finny_batch_apply_avx512` /
`simd_acc_fused_avx512` are already REGS=24 (nnue.rs:517/5632), not REGS=8. So
**"add more AVX-512" buys little here** — the inference kernels are saturated.

## Hot functions (self %, cycles)

| self% | function | role |
|---|---|---|
| 12.8 | search::negamax | search |
| **11.5** | **threats::apply_threat_indices** | threat weight apply |
| 9.2 | sparse_l1::dense_l1_avx512_vnni_l1_32 | L1=32 matmul (AVX-512 VNNI ✓) |
| 8.2 | nnue::forward_with_l1_pairwise_threats | FT pack + L1/L2 orchestration |
| 6.6 | nnue::simd_acc_fused_avx512 | FT accumulator update (1024-wide) |
| 6.5 | movepicker::next_slow | move ordering |
| **6.3** | **threats::push_threats_for_piece** | threat feature enumeration (scalar) |
| **5.2** | **threat_accum::update_dual** | threat delta orchestration |
| 3.4 | see::see_ge | — |

**Threat subsystem ≈ 23% of cycles** (apply_threat_indices + push_threats +
update_dual) — consistent with the 06-14 finding that the threat accumulator,
*as Coda implements it*, is the central inefficiency vs SF (~5.5%). The fix is
making Coda's apply SF-cheap, not removing threats.

## Cache/RAM-miss attribution

**LLC / RAM misses** (the DRAM traffic that actually costs):
| % of RAM misses | function | prefetch today? |
|---|---|---|
| 36.3 | apply_threat_indices | ✅ yes (4-deep + next-row) |
| 22.7 | simd_acc_fused_avx512 (FT apply) | ❌ none |
| 11.1 | finny_batch_apply (king-bucket refresh) | ❌ none |

## Tested experiment — FT-gather software prefetch: NO-OP (do not ship)

Hypothesis: add next-row `_mm_prefetch(T0)` to the two FT gathers that lack it
(`simd_acc_fused_avx512`, `finny_batch_apply_avx512`), mirroring the threat
path. Implemented on `zeus/ft-gather-prefetch` (bench-identical: node count
4880208 both sides).

**Result on Zen 5: measured no-op.** L1-dcache-miss rate **21.82% identical**
both binaries; LLC misses marginally *worse* (5.00% vs 4.90%); NPS within noise
(branch 1839k vs main 1836k mean, +0.2%). The Zen 5 HW prefetcher + 32MB L3
already cover these gathers, and the rows are consumed almost immediately
(no lead time). **Abandoned — not worth fleet time** (zero miss-rate movement;
it could only help on weaker-prefetcher AVX-512 hosts, but the local signal is
nil and slightly negative on LLC). Branch left unpushed.

## Spike — threat-apply cost decomposition: redesign is NO-GO

Before committing to the big "SF-style apply" refactor, decomposed the threat
cost by `perf annotate` (symbolized, v8s3). Result: **the dominant threat slice
is irreducible weight-streaming bandwidth, not avoidable overhead.**

- `apply_threat_indices` (11.7% self, 36% of RAM misses) is **~100%
  weight-streaming**: 27% on `vpmovsxbw` (i8→i16 widening load of the scattered
  threat weight row — the cache-missing gather), 13%+13% on `vpaddw`/`vpsubw`
  (accumulate). No index derivation in it; already optimal AVX-512 REGS=24.
- The SF-style "dirty-list-once-in-make_move, apply-once" redesign attacks
  **replay + re-derivation** — but replay is only ~1.24 plies (deepaudit), and
  the derivation slice (`update_dual` ~5.5%: `PiecePair::base`, `threat_index`,
  attack-index chase) is exactly what item E tested **NEUTRAL twice**.
- So the redesign would chase a ~5.5% slice already shown not to bank, and leave
  the dominant 11.7% streaming untouched. **NO-GO** — saved a large
  correctness-critical refactor.

**The real threat lever is eval-coupled, not a speed refactor.** Bytes streamed
= deltas/node × accumulator width (~10 × 1024 i8). Cutting it means fewer deltas
(drop x-ray = rejected +187 Elo; or a leaner threat set) or a **narrower
threat-accumulator width** (apply cost scales linearly with width — a training
experiment, SPRT eval-cost vs NPS gain). Owned by the training/eval workstream.

## Revised levers (pure-speed)

1. ~~Threat-apply redesign~~ — **NO-GO** (spike above).
2. **Movegen-side bundle: D (direct-write movegen, kill ~514B `MoveList`
   by-value copies) + F (fixed-array undo stack) + G-hoists (lift
   per-node-constant attack sets out of the per-quiet scoring loop).** The best
   remaining *pure-speed, bit-identical* lever. Bundle + measure once + `[-2,1]`
   SPRT. Prior: modest/possibly-neutral (4 prior micro-opts were), but it bounds
   the recoverable pure-speed headroom.
3. **Narrow-threat-accumulator** training experiment — the genuine big threat
   lever; eval-coupled, training workstream.
4. **H — int8 L2/L3/output pipeline** (drop f32 dequant for folded-scale
   VPDPBUSD). Structural (requant/retrain). Deferred.
5. **Not B** (check-info cache) — high-risk/low-value, sibling C already neutral.

Already tested DEAD/neutral (don't re-attempt): L1 sparse-input (loses
1.8-2.4×), threat-index micro-opts (neutral), check-square cache C (neutral),
FT-gather prefetch (no-op, above).

## Method notes
- `strip = true` in `[profile.release]` blanks perf symbols — rebuild with
  `CARGO_PROFILE_RELEASE_STRIP=false make RUSTFLAGS="-Ctarget-cpu=native -g
  -Cforce-frame-pointers=yes"` for symbolized `perf record --call-graph fp`.
- Driver: `scripts/profile_inference.sh <net> [depth]`.
- All numbers on v8s3 with the local OB worker stopped (single-engine, full NPS).

## HEAD-TO-HEAD vs Stockfish on this box (2026-06-28) — the "30% gap" is STALE

Ran Coda (v8s3) and the OB Stockfish binary (`Stockfish-E556FA69`) under `perf
stat` on the **same** Zen 5 box, worker stopped. This is the apples-to-apples
the June-14 doc did on *old* hardware (Hercules AVX2) + *old* net; redone here:

| regime | Coda | SF | gap |
|---|---|---|---|
| single-thread cycles/node | 2981 | 2912 | Coda **+2.4%** |
| single-thread insns/node | 6587 | 6173 | Coda +6.7% |
| single-thread IPC | 2.21 | 2.12 | Coda higher |
| single-thread NPS | ~1.85M | ~1.97M | SF +6.5% |
| **16× contended agg NPS** | **12.2M** | **8.7M** | **Coda ahead** |

**Coda is SF-class on current hardware + net.** The "SF ~44% faster
single-thread / ~69-80% contended" premise in `coda_vs_sf_speed_2026-06-14.md`
is **no longer true** — closed by the L1=32 AVX-512 VNNI kernels, FT retile,
SIMD-latent fixes, and the v8s3 net since June. Single-thread is ~even
(+2.4% cycles/node); **contended (the bots' regime) Coda is ahead**, because
Coda uses less memory traffic/node so it scales better under SMT bandwidth
pressure. (Contended SF may also be hash-thrashing — the binary is tagged
`1C000000` ≈ 469MB hash × 16 — so don't over-read the +40%; the point is it is
NOT 30-69% slower.)

**Consequence:** there is no 30% NPS gap to find. The +6.7% single-thread
instruction overhead (checkers/pinned ~2.4% measured, the f32 L2/output tail,
threat-gen scalar) is distributed single-digit slices that Coda's higher IPC +
better contended scaling already offset — none worth a refactor. **Pure-speed
vs SF is effectively done.** The real ~100-Elo Coda↔SF gap is eval/search
*quality* per node, not NPS — redirect speed effort there.

Caveats: bench positions, one box (Zen 5 / AVX-512 — matches the codabot/coda_bot
deploy hosts); cross-engine node-counting differs slightly so cycles/node is
approximate but standard. A pure-AVX2 host (no AVX-512) could still favor SF more.
