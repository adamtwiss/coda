# Cache-layout optimization scoping — 2026-05-04 overnight

**Triggered by:** the cap-shrink #941 incident (-5.4 Elo from incidental L1d
cache-set collision on a bench-identical change). Insight: if random reshuffles
move us ±5 Elo, deliberate targeting probably has more headroom than we'd been
crediting after the +16 Elo cache leg "exhaustion" framing. Adam's
counter-evidence: Zeus banked +17 Elo in 48h on cache hygiene.

This doc inventories candidates, prioritises by L1d-miss attribution data, and
proposes specific probes ranked by leverage × tractability.

## Methodology

`perf stat -d -d -d` (3 iters each) and `perf record -F 997` on `coda-main bench 13`
on Hercules with ob-worker stopped. Two runs:

1. **Cycle attribution** (`perf record -F 997 -g`) — where time is spent
2. **L1d miss attribution** (`perf record -F 997 -e L1-dcache-load-misses`) — where
   memory stalls happen

Both runs identical workload (bench 13, 966720 nodes). Differences in attribution
between cycles and L1d misses point at memory-stall-dominated functions where
cycle-perf attribution under-counts the real cost.

## Headline finding: cycle ≠ L1d miss share

Top functions, both rankings:

| Function | L1d miss % | Cycle % | Miss/cycle ratio |
|---|---:|---:|---:|
| `ThreatStack::ensure_computed` | **28.18%** | ~0.7% | **~40×** |
| `forward_with_l1_pairwise_inner` | 22.81% | 24.15% | 1× |
| `nnue::finny_batch_apply` | **10.12%** | <1% | ~10× |
| `simd_acc_fused_avx2` | 8.55% | <1% | ~8× |
| `negamax` | 6.73% | 4.78% | 1.4× |
| `MovePicker::next` | 2.41% | 2.68% | 0.9× |
| `threat_accum::refresh` | 1.46% | 1.33% | 1.1× |
| `materialize` | 1.40% | 4.07% | 0.34× |
| `refresh_accumulator` | 1.37% | 7.73% | 0.18× |
| `Board::attackers_to` | 1.18% | 1.84% | 0.6× |

The miss/cycle ratio shows **what's memory-bound vs compute-bound**:
- **>5× ratio** = memory-stall-dominated. Layout work has high leverage.
- **~1× ratio** = balanced. Compute and memory both matter.
- **<1× ratio** = compute-dominated. Layout work has low leverage; SIMD/algorithm work matters.

**ThreatStack::ensure_computed is the single biggest L1d miss producer in the
engine, and it's almost pure memory stalls.** This was invisible in the cycle
profile (perf cycle attribution lands stalls on whatever instruction the OOO
engine has at retirement).

History tables don't appear in either ranking — `corrected_eval` 0.92%,
`MovePicker::next` 2.41%. **History layout work is not the top lever**, despite
my earlier framing. Withdraw that suggestion.

## Candidate 1: `ThreatStack` hot/cold split

### Why this is likely the top lever

**Working set:** 256 plies × 3648 bytes per ThreatEntry = **912 KB** total stack.
Bigger than L2 on most x86 CPUs (256-512 KB), fits L3.

**`ThreatEntry` layout today** (`src/threat_accum.rs:64-78`):

```rust
#[repr(C, align(64))]
pub struct ThreatEntry {
    pub values: [[i16; 768]; 2],  // 3072 bytes — only read in update/refresh
    pub accurate: [bool; 2],      // 2 bytes — read on every can_update step
    pub delta: DeltaVec,          // ~520 bytes — overflowed flag read in can_update
    pub mv: Move,                 // 2 bytes — read in can_update
    pub moved_pt: u8,             // 1 byte — read in can_update
    pub moved_color: u8,          // 1 byte — read in can_update
}
```

Total: **3648 bytes per entry, align(64)** (verified via sizeof probe).

**Hot loop: `can_update` (line 213)** walks BACKWARD through `self.stack` looking
for the most-recent accurate ancestor. Each iteration touches:
- `stack[i+1].mv`           (offset ~3584 within entry)
- `stack[i+1].delta.overflowed()`  (offset ~3520)
- `stack[i+1].moved_pt`     (offset ~3586)
- `stack[i+1].moved_color`  (offset ~3587)
- `stack[i].accurate[pov]`  (offset ~3072)

These are ALL near the END of the 3648-byte entry (after `values` and `delta.data`).

**Backwards walk = pessimal prefetch:**
- Stride: 3648 bytes (>> hardware prefetch range)
- Direction: backwards (HW prefetchers don't anticipate)
- Working set: 14 plies × 3648 = 51 KB → blows L1d (32 KB)

Result: **5-14 forced L1d misses per `ensure_computed` call** for the can_update
walk alone. At 1M `ensure_computed` calls per bench (1 per node), that's 5-14M
misses just from this walk pattern.

### The fix

Split metadata from values:

```rust
// Hot — walked linearly in can_update; small enough to stay L1d-resident
#[repr(C)]
struct ThreatMeta {
    accurate: [bool; 2],
    delta_overflowed: bool,
    moved_pt: u8,
    moved_color: u8,
    _pad: u8,
    mv: u16,
    delta_len: u32,
}  // 12 bytes per entry, 4-byte align

// Cold — only touched in update()/refresh() body, same access pattern as today
#[repr(C, align(64))]
struct ThreatValues {
    values: [[i16; 768]; 2],
    delta_data: [RawThreatDelta; 128],
}  // 3584 bytes, 64-byte align

pub struct ThreatStack {
    metas: Box<[ThreatMeta; 256]>,    // 3 KB — fits in L1d entirely
    values: Box<[ThreatValues; 256]>, // 896 KB — same as today
    index: usize,
    hidden_size: usize,
    pub active: bool,
}
```

### Predicted impact

| Property | Today | Split | Δ |
|---|---:|---:|---|
| Total stack memory | 912 KB | 899 KB | ~flat |
| can_update stride | 3648 B | 12 B | **300×** |
| can_update working set (14 plies) | 51 KB | 168 B | **300×** |
| Cache lines touched per walk | ~14 | ~3 (sequential) | **5×** |
| Estimated L1d misses saved per ensure_computed | 5-14 | 0-1 | most of the walk |
| Estimated cycles saved (12 cyc/L1d→L2 miss) | 0 | 60-150 | 60-150 |
| Estimated NPS improvement | — | 0.5-1.5% | — |
| Predicted Elo at 100 Elo/NPS-doubling | — | +1-3 Elo | — |

**Risk factors:**
- Two separate Box allocations — each tracked separately, separate cache lines for control state. Likely fine.
- Borrow patterns: `update()` calls `self.stack.split_at_mut(ply)` to get prev/curr disjointly; this needs to be reproduced for `values` separately. Should be straightforward.
- Layout regression risk: incidental cache-set shifts (lessons from cap-shrink). Mitigation: 3× perf-stat NPS measurement + L1d miss rate before SPRT.

**Tractability: HIGH.** Estimated 2-3 hour implementation + microbench + SPRT.
Self-contained refactor of `src/threat_accum.rs` only. No public API change
outside the file.

## Candidate 2: deeper prefetch in `apply_threat_deltas`

### Current code (`src/threats.rs:1648-1667`)

```rust
// Prefetch weight rows for upcoming deltas (hide L3 latency)
for &idx in adds.iter().take(4) { _mm_prefetch(...row[idx]..., T0); }
for &idx in subs.iter().take(4) { _mm_prefetch(...row[idx]..., T0); }
```

Only the first 4 add and 4 sub rows are prefetched. Typical delta count is
20-60 per call (refresh_stats histogram). The remaining 12-50+ rows hit cold
on the AVX path, since each row is 1.5 KB and the threat_weights matrix is
~102 MB (66864 features × 768 i16).

### Variants to probe

A. **Prefetch all adds + subs** (no take(4) cap). Risk: prefetch instructions
   themselves saturate the LFB; L1d eviction of the accumulator state.
B. **Pipeline prefetch:** prefetch chunk N+1 while applying chunk N inside the
   AVX loop. Requires structural change to apply_deltas_avx512.
C. **Use T1/T2 hint** for the deeper rows (further-ahead prefetch into L2 or L3
   instead of L1).

### Predicted impact

simd_acc_fused_avx2 was 8.55% L1d misses; apply_threat_deltas is structurally
similar. If we cut weight-row miss rate by 50%, that's ~4% absolute reduction in
total L1d misses → 1-2% NPS gain → 1-2 Elo. But there's a real risk of LFB
saturation regression.

**Tractability: MEDIUM.** Variant A is 1-line; Variant B is moderate refactor.
Worth a 1-hour A/B microbench probe before SPRT.

## Candidate 3: `ThreatEntry.values` null-move copy elision

### Current code (`src/threat_accum.rs:262`)

For null-move plies (or plies with no deltas), `update()` does:

```rust
curr[0].values[p][..h].copy_from_slice(&prev[ply - 1].values[p][..h]);
```

This is a **1.5 KB memcpy per perspective per null-move ply**. Null moves fire
~30K times per bench; with 2 perspectives = 60K copies × 1.5 KB = 90 MB of
copy traffic per bench.

### The fix

Replace value-copying with a "borrow from prev" indirection:

```rust
enum ThreatValuesRef {
    Owned(ThreatValues),
    Reflect(usize),  // ply this entry mirrors
}
```

When `values()` is queried, follow the indirection chain to find the actual
buffer. Replay-applied entries store directly; null-move plies store the
indirection.

### Predicted impact

90 MB of avoided memcpy traffic per bench. At ~50 GB/s memcpy bandwidth that's
~1.8 ms saved out of ~4500 ms total → ~0.04% NPS. **Disappointingly small** —
the copies overlap with prefetch and aren't the critical path.

**Tractability: LOW** (significant refactor of the values-access invariants).
Combined with low payoff: **deprioritise** unless we revisit later.

## Candidate 4: alignment fences on hot stack scratch

Stack-allocated scratch arrays in hot inference functions:

| Function | Scratch | Size |
|---|---|---:|
| `apply_threat_deltas` (`threats.rs:1622-29`) | `[usize; 128]` × 2 | 2 KB |
| `ThreatStack::refresh` (`threat_accum.rs:168-71`) | `[usize; 256]` | 2 KB |
| `ThreatStack::update` (`threat_accum.rs:271-78`) | `[RawThreatDelta; 128]` | 512 B |
| `refresh_accumulator` (`nnue.rs:4541-44, 4572-79`) | `[usize; 32]` × 3 | 384 B |

Currently these are stack-allocated via `MaybeUninit`. Their position in the
stack frame is **incidental** — exactly the case that bit us in #941.

### The fix

Wrap each in an alignment-forcing struct:

```rust
#[repr(C, align(64))]
struct AlignedScratch128([usize; 128]);
let mut scratch_storage = MaybeUninit::<AlignedScratch128>::uninit();
```

This guarantees each scratch buffer starts on a 64-byte boundary, removing
incidental cache-set collisions with adjacent stack frames. Doesn't fix
"different stack frame layout shifts the buffer onto a colliding set" — only
fixes "buffer starts mid-cache-line so loads cross lines unnecessarily."

### Predicted impact

Small. Maybe 0.1-0.3% NPS, more variance reduction than mean improvement.
Functions already use SIMD-friendly access; cross-line splits are infrequent
relative to total loads.

**Tractability: HIGH.** 5 minutes per call site. **Ship as a defensive
measure** alongside Candidate 1 if that lands — costs nothing to do them
together and removes a future cap-shrink-style accident class.

## Candidate 5: thread-local heap scratch buffers (instead of stack)

For the same hot scratch buffers as Candidate 4, move them off the stack
entirely into a per-thread `RefCell<Box<[usize; 128]>>` or similar. Removes
all stack-frame layout sensitivity for these buffers.

### Predicted impact

Negligible mean improvement, but **fully removes the cap-shrink-style accident
class** for these specific buffers. Probably overkill if Candidate 4 lands —
alignment fence is sufficient.

**Tractability: MEDIUM.** TLS access has a few-cycle overhead per call; need
to verify it doesn't regress NPS by itself. **Defer** unless Candidate 4
proves insufficient.

## Candidate 6: history table layout (DEPRIORITISED)

Original highlighted candidate. After perf data:
- `MovePicker::next` is 2.41% L1d misses, 2.68% cycles → balanced, not memory-bound
- `corrected_eval` (history-read-heavy) is 0.92% L1d misses, 0.75% cycles
- Even if we halved miss rate from history layout work, total NPS impact is
  bounded by ~1% → ~1 Elo

**Conclusion: not worth investigating further until Candidates 1-2 are exhausted.**
Note in passing for posterity, but no probe planned.

## Candidate 7: `Board` struct layout

Inspected `src/board.rs:30-56`. Hot fields are already at the top:
- `pieces[6]` (48 B = 1 cache line)
- `colors[2]` (16 B)
- `mailbox[64]` (64 B = 1 cache line)
- side_to_move / castling / ep / clocks (~10 B)
- hash + pawn_hash + key arrays (~64 B)

Board is ~250 B of stable hot fields = 4 cache lines, well-laid-out.
`undo_stack` and `threat_deltas` (Vec heap pointers) are at the end — cold
metadata. **Already good. No probe planned.**

## Recommended probe order

1. **Candidate 1: ThreatStack hot/cold split** — 2-3 hours, predicted +1-3 Elo.
   Do this first.
2. **Candidate 4: alignment fences on hot scratch** — 30 min, ship alongside #1.
3. **Candidate 2: deeper prefetch in apply_threat_deltas** — 1-hour microbench
   probe to decide A/B/C variants, then SPRT.

If 1+2+4 land, expect **+2-5 Elo cumulative** with high confidence the
mechanism is real and locally measurable. Combined with Zeus's recent +17 from
the broader cache-hygiene leg, the "cache layout still has runway" hypothesis
gets concrete validation.

## Late-night correction: perf annotate refines the picture

After writing the above, ran `perf annotate -i perf-l1d.data --symbol
ensure_computed`. Top miss-attributed instructions inside the function:

| Sample % | Instruction | Read source |
|---:|---|---|
| 11.03% | `vpsubw %ymm9, %ymm0, %ymm0` | reg-only (waiting on prior load) |
| 8.11%  | `vpsubw %ymm9, %ymm5, %ymm5` | reg-only |
| 5.59%  | `vpmovsxbw 0x70(%rdi,%rcx,1), %ymm10` | **threat_weights[idx*h+offset+0x70]** |
| 4.93%  | `vpmovsxbw 0x30(%rdi,%rcx,1), %ymm10` | **threat_weights[idx*h+offset+0x30]** |
| 2.77%  | `vpmovsxbw (%rdi,%rcx,1), %ymm9` | **threat_weights[idx*h+offset]** |

These are inside the inlined `apply_deltas_avx2` body. The `vpmovsxbw` loads
are reading scattered i8 weight rows from `threat_weights` (the 102 MB
threat-feature weight matrix). The vpsubw stalls are waiting on those loads.

**Revised mechanism breakdown for `ensure_computed`'s 28.18% L1d miss share:**
- ~22-25 percentage points: scattered weight-row reads from the 102 MB
  threat_weights matrix, inside the AVX-2 delta-apply inner loop
- ~2-4 percentage points: the can_update backwards metadata walk (Candidate 1
  target)
- ~1-2 percentage points: misc (the null-move copy_from_slice, perspective
  setup)

### What this means for Candidate 1

The original prediction of "+1-3 Elo from ThreatStack split" was based on
modeling 14 metadata cache misses per call. The annotate data shows
metadata-walk misses are a small fraction of ensure_computed's stalls.

**Revised Candidate 1 prediction: +0.3-1 Elo, not +1-3.** Still worth doing
(it's a clean refactor with measurable mechanism), but it's not the homerun
I'd assumed when I wrote the candidate above.

### What this elevates

**Candidate 2 (deeper prefetch in apply_threat_deltas) is now the top
inference-side leverage.** The dominant cache-miss source in the engine is
scattered-weight-row reads. Currently we prefetch 4 adds + 4 subs out of
typical 20-60 total. Increasing prefetch coverage (variants A/B/C) directly
attacks the dominant miss class.

**Even bigger lever: shrinking the 102 MB threat_weights matrix itself.**
This is the training-side path already in flight as task #184 (compact
threat encoder + importance reorder + group lasso). At the inference layer
we can't make the matrix smaller, but training-side reduction would
proportionally reduce all miss rates here.

A speculative inference-side angle: **reorder weight rows by activation
frequency** so hot rows cluster in the same L3 region. Top-K most-active
features account for X% of all accesses. If we sort weights by frequency
post-training (in `convert-bullet`), hot rows land contiguously and L3
warm-set locality improves. This is a one-shot permutation; net behavior
unchanged. Worth scoping after Candidates 1+2.

### Revised probe order

1. **Candidate 2 first** (deeper prefetch in apply_threat_deltas) — A/B/C
   1-hour microbench, biggest single-source-of-misses target. Expected
   +0.5-2 Elo.
2. **Candidate 1** (ThreatStack split) — refactor for clarity + small win.
   Expected +0.3-1 Elo.
3. **Candidate 4** (alignment fences) — defensive, ship alongside.
4. **New candidate (post-discussion):** weight-row activation-frequency
   reorder via convert-bullet. Predicted leverage TBD; needs a data
   gathering pass first (which features are hottest in real games?).

## Validation protocol (per probe)

For each branch, before SPRT:

1. `make && ./coda bench` — 3 iters, confirm node count matches main
2. `perf stat -d -d -d ./coda bench 13` — 3 iters, compare NPS + L1d miss rate
3. **Reject early** if L1d miss rate goes UP without a clear mechanism
   explanation (signs of an incidental layout collision like #941)
4. **Proceed to SPRT** if NPS improves >0.5% with corresponding L1d miss
   reduction

Bounds: `[0, 3]` for "does this help" probes (default per CLAUDE.md). If
mechanism is purely defensive (alignment fence with no expected mean improvement),
use `[-3, 3]` to confirm non-regression.

## Open questions for morning discussion

- Worth implementing the ThreatStack split as one branch, or split into "metas
  separate" + "values separate" two-step? Single branch is simpler; two-step
  isolates which sub-change delivers.
- Is there an even bigger lever in the threat_weights matrix layout (102 MB)?
  Importance reorder + group-lasso (already in flight as #184) attacks this
  from training side. Inference-side reordering would be a different angle.
- The miss/cycle ratio diagnostic is now in our toolkit — should we run it on
  every NPS-touching branch's "post" build to build a longitudinal record?
