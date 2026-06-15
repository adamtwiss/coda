# Handoff: column-major VPDPBUSD kernel for L1=32 (AVX-512 VNNI)

**For:** a Claude on Zeus (Zen5 = AVX-512 + VNNI). Hercules is AVX2-only so this
work can only be *measured* on a VNNI host.

**TL;DR:** L1=32 leaves VNNI efficiency on the table because we never wrote a
*column-major* VPDPBUSD kernel for 32 neurons. Prod L1=16 has one
(`dense_l1_avx512_vnni`); L1=32 falls back to the **row-major** per-neuron VNNI
path. Writing the column-major L1=32 kernel cuts a real chunk of the ~10% L1=32
NPS tax on deployment (Zen5) — directly informs the L1=32 go/no-go (tune #2017).

---

## The finding (confirmed from code)

The fused int8 dot is **`VPDPBUSD`** (1 instruction; AVX2 needs 2:
`VPMADDUBSW`+`VPMADDWD`). Dispatch is `select_l1_kernel` (src/nnue.rs:2449-2472):

| Config | On VNNI hardware | Kernel | Layout |
|---|---|---|---|
| **L1=16** (prod) | `DenseAvx512Vnni` (nnue.rs:2452) | column-major | all 16 neurons in 1 ZMM, **input loaded once per chunk**, 1 VPDPBUSD/chunk |
| **L1=32** | `RowMajorAvx512Vnni` (nnue.rs:2455) | **row-major** | per-neuron `simd512_l1_int8_dot_vnni`, **re-scans input per neuron** (the "16× cache-line touches/chunk" problem the column-major kernel exists to kill) |
| L1=32 (AVX2 only) | `DenseAvx2L1_32` (nnue.rs:2464) | column-major | 4 YMM accumulators (8 neurons each), VPMADDUBSW+VPMADDWD |

So L1=32 *uses* VNNI, just in the inefficient layout. Missing kernel, not
intrinsic cost.

## The task

Write `dense_l1_avx512_vnni_l1_32` (column-major, 32 neurons) and wire it in.

**It's a merge of two existing templates:**
- `dense_l1_avx2_l1_32` (src/sparse_l1.rs:232) — already the column-major L1=32
  *structure* (input-chunk-major weights `l1_weights_sparse`, 4 YMM accumulators
  of 8 neurons), but with AVX2 VPMADDUBSW+VPMADDWD.
- `dense_l1_avx512_vnni` (src/sparse_l1.rs:427) — already the VPDPBUSD *kernel*
  for L1=16: 4 interleaved ZMM accumulators (a0..a3, to break the VPDPBUSD
  dep-chain), `_mm512_dpbusd_epi32(acc, _mm512_set1_epi32(chunk), w)` per chunk.
  16 neurons = 16 i32 = 1 ZMM wide.

**Combine:** take `dense_l1_avx2_l1_32`'s column-major outer loop, emit VPDPBUSD
instead of maddubs/madd. 32 neurons = 32 i32 = **2 ZMM wide**. Keep dep-chain
interleaving (e.g. 2 ZMM × a few interleaved accumulators; 32 ZMM regs
available, budget freely). Weights are input-chunk-major
(`l1_weights_sparse`) — same layout `dense_l1_avx2_l1_32` consumes, so no
repack needed.

**Wire-in:**
1. Add `L1Kernel::DenseAvx512VnniL1_32` to the enum (nnue.rs:~2407).
2. In `select_l1_kernel`, add ABOVE the row-major fallback (before nnue.rs:2455):
   ```rust
   if has_avx512_vnni && col_ok && l1 == 32 && pw.is_multiple_of(4) {
       return L1Kernel::DenseAvx512VnniL1_32;
   }
   ```
3. Add the match arm in the forward dispatch (nnue.rs:~3255, next to
   `DenseAvx512Vnni` at 3257).
4. `#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]` on the new fn.

## Verification (bit-identical — this is a pure kernel swap)

1. **Correctness**: `CODA_VERIFY_NNUE=1 ./coda bench 8 -n nets/multi-v6-l132-s5-swa.nnue`
   → expect `0/N mismatches`. Also the in-tree debug reference compare fires under
   `#[cfg(debug_assertions)]` (nnue.rs:~3347) — run a debug build over a few
   positions.
2. **Bench node count unchanged** vs the current build (it's bit-identical):
   `make && ./coda bench` should be identical to HEAD's number.
3. **NPS win (the point)** — Zen5 only: with the L1=32 net, compare the new
   kernel vs the row-major fallback. Quick A/B: gate the new kernel behind an
   env flag (or temporarily force `RowMajorAvx512Vnni` for the baseline) and
   bench both. Expect the L1=32 tax (~10%, see
   `memory/project_l1_32_nps_tax.md`) to shrink by whatever the column-major
   layout saves on this width.

**Test net:** `nets/multi-v6-l132-s5-swa.nnue` (L1=32, FT1024, OB SHA `035195DB`).
Fetch via `coda fetch-net` or it's in the repo's `nets/`.

## Why this matters now

- Tune **#2017** (main, L1=32 net 035195DB, 1500-iter core, STC) is live to test
  whether a tune closes the L1=32 model's −3.62 vs prod (#1988). The L1=32
  *eval* question and the L1=32 *speed tax* are the two halves of the go/no-go.
  This kernel attacks the speed half — and part of the "tax" is just this
  missing kernel, so the real L1=32 deployment cost is lower than the Hercules
  AVX2 numbers suggest.
- Deployment (lichess codabot) runs a **Ryzen 9700X (Zen5, VNNI)**, so this is a
  real deployment win, not just a benchmark curio.

## Broader state (context)

- **Eval is near-ceiling** (Coda Spearman 0.853 vs LC0 — #2 behind only SF 0.861,
  ahead of Reckless 0.836). So **speed is the high-leverage lever**, not eval.
- The SF gap (~44% single / ~93% contended on Hercules AVX2) is dominated by the
  **threat accumulator** (Coda ~31% of cycles vs SF ~5.5%). See
  `docs/threat_accumulator_findings_2026-06-15.md` for the full decomposition.
- Most of that threat cost is **eval richness we've validated as worth it**
  (X-ray = +187 Elo for ~11% NPS, SPRT #2014). The bit-identical recovery is
  small: `recapture-combine` (double_inc_update port, SPRT #2015, ~+2% contended)
  was the last big one; this L1=32 kernel + the FT-prefetch/contention thread
  (#1994) are the remaining free-speed levers.
- This L1 work is **VNNI-only** and **L1=32-only**; prod (L1=16) already runs the
  optimal column-major VPDPBUSD on Zen5.
