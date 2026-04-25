# Group-lasso SB200 probe — GPU-host runbook

Written 2026-04-24 by Hercules. For GPU-host Claude (Atlas/Titan) to execute.

## What this tests

Group-lasso (per-row L1) regularization on v9's l0w. Unlike plain L1, which
leaves a long tail of tiny non-zero weights per feature, group-lasso zeroes
ENTIRE input-feature columns. That's what we need to compact the 49 MB
threat matrix for L3 cache residency — a zero column can be elided at load
time; a weight with 0.1 % of its outputs zeroed cannot.

Prior L1 experiment (`--l1-decay 1e-3`) finished at SB150 (decouple branch):
produced weight-level sparsity but row-level sparsity only crept from the
L2-only baseline of 8.4 % to ~10-15 % — not enough to compact. SB150 SPRT
vs prior-best net finished −25 ±13, ambiguous on Elo damage.

Group-lasso should produce row-level sparsity directly. SB200 is the first
probe — minimum length needed to see whether the mechanism fires and how
many feature rows zero out.

## Environment

- Bullet fork: `adamtwiss/bullet` branch `feature/decouple-l1-lr`
- Commit: `12d11c4` "Add group-lasso (per-row L1) regularization"
- Training example: `examples/coda_v9_768_threats.rs`
- Data: `/workspace/data` (GPU hosts) per
  `memory/project_training_data_paths.md`
- Flag name discipline: always `--dataset-dir`, always read source if
  unsure (per `memory/project_bullet_training_invocation.md`)

## Setup on the GPU host

```bash
cd ~/code/bullet
git fetch origin
git checkout feature/decouple-l1-lr
git pull
cargo build --release --example coda_v9_768_threats
```

The CUDA kernel `GroupLassoKernel` will compile as part of the CUDA
build. If nvcc errors out, that's a real bug — flag it, don't paper over.

## Baseline comparison — use existing nets, don't retrain

We have plenty of SB200 v9 nets to compare against; same-seed retraining
isn't possible across different machines and isn't standard practice.

Primary baseline: `nets/net-v9-768th16x32-reckless-w15-e200s200-warm30.nnue`

This is the current warm30 best for SB200 runs. Match its training config
exactly, EXCEPT for the group-lasso flag, so any sparsity/Elo difference
is attributable to group-lasso alone.

Secondary baselines with L1 only (already trained on decouple branch,
useful for isolating element-wise-L1 vs group-L1 effect):
  - `nets/net-v9-768th16x32-reckless-w15-e200s200-warm30-l1e-6.nnue`
  - `nets/net-v9-768th16x32-reckless-w15-e200s200-warm30-l1e-7.nnue`

## Probe #1 — SB200 warm30 + `--group-l1-decay 1e-2`

Match warm30 reference config exactly, add one flag:

```bash
cd ~/code/bullet
./target/release/examples/coda_v9_768_threats \
    --dataset-dir /workspace/data \
    --kb-layout reckless \
    --hidden-activation crelu \
    --warmup 30 \
    --wdl 0.15 \
    --superbatches 200 \
    --group-l1-decay 1e-2
```

Overhead of group-lasso is one small reduction kernel per optimiser step
on l0w only — wall-clock should match the plain warm30 SB200 reference.

Banner should include:
```
Group-L1 decay on l0w: 1.0e-2 (row_size=768)
```

If it doesn't, the flag isn't plumbed through — stop and report.

## After training: convert with `coda convert-bullet`

Conversion is done by `coda`, not by `bullet`. Flags must match the
training config exactly — mismatched flags produce a broken-but-
superficially-valid `.nnue` that loads, benches, and then H0s in SPRT.

For this warm30 v9 config:

```bash
cd ~/code/coda
./target/release/coda convert-bullet \
  -i <path/to/quantised-s200.bin> \
  -o <path/to/group-lasso-sb200.nnue> \
  --pairwise --screlu \
  --hidden 16 --hidden2 32 --int8l1 \
  --threats 66864 \
  --kb-layout reckless \
  --hl-crelu
```

`--hl-crelu` is load-bearing: the warm30 training uses
`--hidden-activation crelu`, which sets the hidden-layer activation to
CReLU. Without `--hl-crelu` at convert time, header bit 5 stays cleared
and the engine auto-configures `HiddenActivation=screlu` on load —
silent 50-150 Elo eval regression (see
`memory/feedback_verify_net_flag_bits_before_sprt.md`).

Verify conversion with `coda bench -n <net>` and confirm:
- load line shows `Loaded 66864 threat features`
- initialisation line shows `Threat features initialised: 66864 total`
- bit 5 (`hl_crelu`) is set on the header — use
  `coda patch-net -i <net>` with no flags to dump the current flags.

If any of these are missing or wrong, do NOT use the net — flag it and
debug.

## Sparsity measurement

Measure row-level sparsity on Hercules (Hercules has the latest
`measure-net-sparsity` subcommand):

```bash
cd ~/code/coda
./target/release/coda measure-net-sparsity -i <net.nnue>
```

Record: row_sparsity (fraction of all-zero 768-wide columns),
weight_sparsity (fraction of individual zero weights).

## Decision tree after sparsity measurement

Let `row_sparsity` = fraction of 768-wide columns that are all-zero.

- **row_sparsity ≥ 40 %**: group-lasso is biting. SPRT the new net vs
  `net-v9-768th16x32-reckless-w15-e200s200-warm30.nnue` at `[-5, 5]`.
  If Elo cost < 20 Elo at 40 % row sparsity, the L3 compaction case is
  strong — queue a clean SB800 run.

- **row_sparsity 10-40 %**: mechanism is working but under-powered.
  Next probe: `--group-l1-decay 3e-2` at SB200.

- **row_sparsity < 10 %**: either (a) `1e-2` is too weak for our scale,
  or (b) the implementation isn't routing the proximal step to l0w.
  Confirm the banner line fired; retry at `3e-2` before concluding the
  implementation is broken.

## Reporting back

Post in chat:
- SB200 train wall-time and final loss
- row_sparsity and weight_sparsity from measure-net-sparsity
- Banner text (to confirm group-lasso actually enabled)
- Net filename so Hercules can grab it for the SPRT

That's all Hercules needs to decide on the next probe and whether to
parallel a group-lasso L2/L3 warm-30 fleet sweep.

## 2026-04-25 update — Probe #1 results

Probe #1 at `--group-l1-decay 1e-2` produced:
- Threat row sparsity: **13.48 %** (9,010 / 66,864)
- PSQ row sparsity: 15.56 % (1,195 / 7,680)
- Final loss: 0.003608 (clean monotonic curve)
- Train wall-time: 8h 19m

Net file: `nets/net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-1e2.nnue`
SHA8: `573854EF`

**Surprising SPRT signal (test #753 vs C8fix dense kb10 baseline):**
+11.22 Elo at 1100 games on `[-5, 5]` bounds — sparse net is STRONGER
than dense, not just non-regressing. Likely mix of L1 regularization
combating SB200 overfitting + noise-feature removal.

**Cache-residency status:** 51.3 MB → 44.4 MB. Still well above
the 32 MB L3 target — no cache step-function yet. NPS bench showed
-3 to -9 % vs dense, so sparse-matmul overhead currently dominates
savings at 13.5 %. Need ~35 % threat sparsity for cache benefit.

## Probes #2 and #3 — parallel SB200 at 3e-2 and 5e-2

GPU-host workflow: two GPU machines available, run in parallel.

Probe #2 (GPU A): `--group-l1-decay 3e-2`
Probe #3 (GPU B): `--group-l1-decay 5e-2`

```bash
# On each GPU host, identical command except for the decay value:
cd ~/code/bullet
git fetch origin && git checkout feature/decouple-l1-lr && git pull
cargo build --release --example coda_v9_768_threats

# GPU A
./target/release/examples/coda_v9_768_threats \
    --dataset-dir /workspace/data \
    --kb-layout reckless \
    --hidden-activation crelu \
    --warmup 30 \
    --wdl 0.15 \
    --superbatches 200 \
    --group-l1-decay 3e-2

# GPU B
./target/release/examples/coda_v9_768_threats \
    --dataset-dir /workspace/data \
    --kb-layout reckless \
    --hidden-activation crelu \
    --warmup 30 \
    --wdl 0.15 \
    --superbatches 200 \
    --group-l1-decay 5e-2
```

**Flag-value note:** valid `--kb-layout` values per
`bullet/examples/coda_v9_768_threats.rs:217` are `uniform | consensus |
reckless` — there is no `kb10` flag value. The "kb10" naming in net
filenames refers to the produced bucket count or post-hoc convention,
not the training-flag value. Always pass `reckless` to match the
original probe #1 recipe and the warm30 reference family.

## Decision tree after probes #2 and #3

Let `s_3e2` = threat row sparsity at 3e-2, `s_5e2` = at 5e-2.

| s_3e2 | s_5e2 | Action |
|---|---|---|
| ≥ 35 % | (any) | 3e-2 is the SB800 production decay. Train at SB800. |
| 25-35 % | ≥ 35 % | 5e-2 is SB800 production candidate IF its SPRT vs C8fix is still positive or neutral. Else fall back to 3e-2 SB800. |
| 25-35 % | 25-35 % | Both decays in working range. Pick whichever has better SPRT vs C8fix. SB800 at the winner. |
| < 25 % | < 25 % | Both too weak. Probe #4 at 1e-1 sequentially before committing to SB800. |
| (any) | training divergent / loss explosion | 5e-2 too aggressive; 3e-2 is production decay. |

## Probe #2/#3 SPRT plan

For each new sparse net, repeat the same comparison:
- Dev: new sparse net
- Base: C8fix dense kb10 (`1836917B`)
- Bounds: `[-5, 5]` (non-regression)
- Both nets benched on same trunk binary, explicit `--dev-network` and
  `--base-network` per `feedback_netvsnet_sprt_explicit_benches.md`

If +11 result from probe #1 holds across higher decays, the
"L1-as-regularizer" reading is validated. If higher decays start
losing Elo, we've found the Elo-vs-sparsity Pareto frontier.
