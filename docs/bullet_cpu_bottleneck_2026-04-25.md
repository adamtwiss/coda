# Bullet trainer CPU bottleneck — diagnosis 2026-04-25

Hercules-side code-read of the bullet fork (no GPU host repro yet).
GPU host symptoms Adam reported:

- GPU util 75% on factor (no lasso), 55-66% on lasso (no factor).
- One CPU thread doing all the real work; other cores idle.
- Factor SB200 ≈ 7h, ~570K pos/sec; was ~1M+ pos/sec without factor.
- Group-lasso SB200 ≈ 13h. SB800 lasso ≈ 52h.
- Old GPU-host CPUs with many cores but weak single-thread.

## Where the per-batch CPU work lands

Per-batch path in `crates/acyclib/src/trainer.rs`:

1. `batch_on_device.load_into_graph(&mut graph)` — swap host buffers
   into graph node tensors (cheap pointer swap).
2. `step()` — `zero_grads / forward / backward` then
   `optimiser.update()`. All async kernel launches.
3. `receiver.recv()` — get next host batch from loader thread.
4. `batch_on_device.load_new_data(&next_batch)` — H2D copies.
   - **Calls `device.synchronise()` after every H2D**
     (`dataloader.rs:175,237`) — this is one full sync per batch.
5. `graph.get_output_value()` — read scalar loss from GPU.
   - Implicit GPU→CPU readback ⇒ another sync per batch.

So **2 explicit syncs per batch** in the steady state.

## Why those syncs hurt

`crates/bullet_cuda_backend/src/device.rs:107` sets
`CU_CTX_SCHED_BLOCKING_SYNC` via `set_blocking_synchronize()`. That
makes the calling CPU thread *sleep* on every sync (driver waits via
OS-level blocking). On the GPU-host CPUs (older Xeons, weak
single-thread, high context-switch latency) that costs ~30-100 µs
per sync, blocking the whole training thread.

CUDA default (no call) is `CU_CTX_SCHED_AUTO`, which selects spin if
active CPU threads ≤ GPU count (the typical case). Spin path
costs ~1-3 µs per sync — same kernel work, but the CPU returns
control immediately when GPU finishes.

With ~6000-batch superbatches and 2 sleep-syncs per batch, that's
12K sleep/wake round-trips per superbatch. At 50 µs each that's only
~10 minutes per SB — *but* it hides any kernel-launch-pipeline
overlap, so the real wall-clock impact is bigger than the raw
sleep cost.

## Why factor and lasso amplify it

**Factor** doubles the active feature count per position
(`max_active * 2`). GPU work per batch grows ~50-80% (more
sparse-to-dense bandwidth, larger FT matmul, larger backward
gradient reduction). With `set_blocking_synchronize` masking
launch-pipeline overlap, the longer GPU work + per-batch sync gap
both grow. GPU climbs from idle-baseline → 75%; gap (25%) is the
sync window not closing in time.

**Group-lasso** adds *no extra GPU work that should matter* — the
`GroupLassoKernel` runs on GPU (CUDA `kernels.cu:301`, HIP
`optimiser.cu:91`), launching one block per row, 256 threads per
block. For threat l0w (66864 × 768) that's 66864 small blocks of
~3 ops/thread. Should be sub-millisecond per call. **Yet wall-clock
is 13h vs 7h for factor (no lasso).** That's the part I can't
account for from code-read alone — see "what's left to confirm"
below. The lasso path *also* has the same sync cost as factor, but
that doesn't explain a 1.85× slowdown vs factor.

The CPU fallback path (`acyclib/src/device/cpu/base.rs:283-379`)
DOES have a serial Rust loop for both AdamW and the per-row
group-lasso reduction. If for some reason the CUDA path isn't
selected (build mode, feature flag), training silently falls back
to CPU AdamW. **Worth checking:** is the example actually being
built with `--features cuda`? See "what's left to confirm".

## Cheapest fixes, ranked

### 1. Drop `set_blocking_synchronize()` (1-line change)

```diff
--- a/crates/bullet_cuda_backend/src/device.rs
+++ b/crates/bullet_cuda_backend/src/device.rs
@@ -104,7 +104,6 @@
     fn new(id: Self::IdType) -> Result<Self, Self::DeviceError> {
         let ctx = CudaContext::new(id).map_err(CudaError::Driver)?;
-        ctx.set_blocking_synchronize().map_err(CudaError::Driver)?;
         let stream = ctx.default_stream();
```

Default mode (`CU_CTX_SCHED_AUTO` → SPIN for 1-GPU) replaces sleep
with poll. CPU thread pegs one core at 100% but sync wakeup latency
drops 10-50× and kernel-launch pipeline can run ahead of compute.

**Expected**: 20-40% wall-clock reduction on factor and lasso (the
slow-CPU+blocking-sync interaction). HIP backend doesn't have a
matching call — it's CUDA-specific.

**Risk**: one CPU core pinned at 100% during training. On a
many-core box that's free anyway.

**Test plan (GPU 3 if it stays idle)**: branch
`feature/no-blocking-sync` off `feature/decouple-l1-lr` (where
group-lasso lives). Run SB10 baseline vs no-blocking on lasso and
factor configs; compare pos/sec. Doesn't need a full SB200.

### 2. Async loss readback (small change)

`get_output_value()` is called every batch but only used for
`running_loss` accumulation and the periodic logger. The logger
fires on `curr_batch % schedule.log_rate == 0` — typically every
50-200 batches.

Change: use a CUDA event to mark when the loss is ready, only sync
+ readback on logger ticks. Would skip ~199/200 of the implicit
syncs.

Trickier than fix 1 (needs an event API on top of `get_output_value`),
but stacks with fix 1 cleanly.

### 3. Double-buffer next batch H2D via `copystream`

`device.rs:109` already creates a `copystream`. `load_new_data`
issues H2D copies but uses the *default* stream (line 192-198 with
`load_sparse_nonblocking`). Then `synchronise()` waits on BOTH
streams (line 124). If H2D goes on `copystream` and we record an
event the forward pass waits on, the H2D overlaps compute of the
PREVIOUS batch.

This is the largest potential win after fix 1, but requires more
plumbing (event recording, stream-on-stream waits in the graph
forward).

### 4. Multi-thread loader chunking (already done)

`loader.rs:189-249` already chunks batch prep across `threads`
worker threads. So the position-decode → sparse-indices step is
parallel. Not a lever; documenting that we already did this.

## What's left to confirm (need GPU host)

1. **Is the build using CUDA?** Run on the GPU host:
   ```bash
   cd ~/code/bullet
   cargo build --release --example coda_v9_768_threats --features cuda 2>&1 | grep -i "compiling bullet_cuda_backend\|warning:" | head
   ```
   If `bullet_cuda_backend` doesn't appear, training is on CPU and
   that's the entire bottleneck. The `--features cuda` flag
   matters; without it, the example may build but fall back.

2. **Profile a 50-batch run with `nsys`** to see whether the
   training loop is launch-bound (CPU-side gaps between kernels) or
   GPU-bound (kernels back-to-back, gaps in CPU). If launch-bound:
   fixes 1 + 2 stack. If GPU-bound: there's a kernel inefficiency.

3. **Lasso vs factor wall-clock asymmetry (13h vs 7h)** isn't
   explained by anything in the code I read — the CUDA lasso kernel
   is small and async. Two candidates:
   - Lasso forces a fallback to the CPU AdamW path (check via build
     log + GPU util mid-run).
   - The 13h figure was from earlier when group-lasso also disabled
     other optimisations; check the SB200 lasso run's actual
     launch profile.

## Triage recommendation

Go fix 1 first — one-line change, GPU 3 SB10 microbench will
confirm or refute in <30 min. If it lands a 20-40% wall-clock win,
that immediately makes SB800 lasso (52h → ~35-40h) viable for the
weekend, and gets the SB800 factor down from ~30-40h to ~25-30h.

Fix 2 (async loss readback) is the next compounding lever, ~30 min
of code change.

Fixes 3 and 4 are bigger refactors; defer until 1+2 land.
