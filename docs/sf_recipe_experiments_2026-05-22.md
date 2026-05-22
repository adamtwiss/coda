# SF threats.yaml Recipe Experiments — Training Plans (Tiers 1-3)

**Date:** 2026-05-22
**Source trigger:** Analysis of `vondele/nettest/threats.yaml` (SF's training
recipe for threat-input nets). See conversation notes for the full readthrough.

This doc lays out runnable training plans for 7 experiments derived from the
SF recipe, ordered by cost. Each is a candidate for **S200 paired probe vs
baby-prod via the mini-prod branch** unless otherwise noted.

**Common setup for ALL experiments:**
- **Train on a single GPU host** (currently any 5070 Ti / equivalent in
  the fleet). Each S200 train ≈ 3-4h wall-clock.
- **Compare via mini-prod branch paired-probe**: upload the candidate net,
  SPRT vs baby-prod (`61115E7F`, `cal-day0-factor-w15-warm30-hlcrelu-s200`)
  using `ob_submit.py mini-prod` with `--dev-network <CANDIDATE_SHA>
  --base-network 61115E7F`.
- **SPRT bounds**: `[-5, 5]` for net comparisons where magnitude matters
  (paired-probe net direction is often signed but not always small).
- **Dataset**: `/workspace/data` on GPU hosts (per memory
  `project_training_data_paths`).
- **Canonical recipe form (mini-prod-like S200)**:
  ```bash
  cargo run --release --features cuda --example coda_v9_768_threats -- \
    --dataset-dir /workspace/data \
    --superbatches 200 \
    --wdl 0.15 \
    --warmup 30 \
    --kb-layout reckless \
    --hidden-activation crelu \
    --factoriser \
    --seed 42 \
    --save-rate 200
  ```
  All experiments below diff from this canonical form by one (or
  bundled) axes.

---

## Tier 1 — Cheapest experiments to fire first

These need at most a tiny code change (4× batch only) and unlock the rest.

### 1A. 90% FEN skipping (`--fen-skip-prob 0.9`)

**Hypothesis**: At 90% skip, each epoch samples a fresh ~10% of the
filtered position pool. The decorrelation between epochs acts as data
augmentation. Prior 50% skip test was −0.5 retuned (essentially neutral)
— 90% is a qualitatively different regime (SF uses this).

**Recipe** (canonical except for the one flag):
```bash
cargo run --release --features cuda --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 200 \
  --wdl 0.15 \
  --warmup 30 \
  --kb-layout reckless \
  --hidden-activation crelu \
  --factoriser \
  --seed 42 \
  --save-rate 200 \
  --fen-skip-prob 0.9 \
  --net-id coda-v9-fenskip-0.9
```

**Decision criteria**:
- H1 at SPRT vs baby-prod → bracket further (test 0.7, 0.8 to find the
  peak) AND queue a fresh S800 prod-replacement candidate with the
  best skip value.
- H0 at SPRT → conclude "skipping doesn't help at our data scale."
  Don't chase further skip variants.

**Wall-clock**: ~10h train (2-3× slower than canonical due to decoder
bottleneck — fen-skip 0.9 means decoding 10× positions per one kept,
and binpack's chain-compressed format makes seek-skip impossible) +
6h SPRT = **~16h total**. 1A is the slowest Tier 1 experiment despite
being a flag flip.

**Mitigation candidates if scaling to S800**:
- Offline pre-filter binpacks (one-time CPU pass) to produce
  pre-skipped binpacks. Eliminates runtime overhead.
- Test at intermediate rates first: a 0.7 or 0.75 probe (decode
  ~3.3× per kept) is 1.5-2× slower vs 2-3× at 0.9, and is already
  qualitatively in the "fresh sampling" regime. Cheaper directional
  signal before committing to the full 0.9.

**Why SF can afford 0.9 cheaply**: their GPU work per batch (L1=1024
matrix ops) is ~64× ours (L1=16), so they're GPU-bound and the 10×
decode overhead is masked. Our smaller architecture is closer to
data-loader-bound at baseline.

---

### 1B. WDL warmup 0→0.15 (ramp at start, keep current endpoint)

**Hypothesis**: A short WDL warmup phase lets the eval signal stabilize
before WDL is blended in. Isolates "does warmup help" from "is 0.24 a
better endpoint." Standard pattern in SF + Hobbes + others.

**Recipe**:
```bash
cargo run --release --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 200 \
  --wdl 0.0 \
  --wdl-end 0.15 \
  --wdl-tail 0.15 \
  --wdl-tail-from 6 \
  --warmup 30 \
  --kb-layout reckless \
  --ob-layout material \
  --hidden-activation crelu \
  --factoriser \
  --seed 42 \
  --threads 8 \
  --save-rate 200 \
  --net-id coda-v9-wdl-warmup
```

(`--wdl-tail-from 6` = ramp ends at SB 6, so warmup is 6/200 ≈ 3% of
training. Matches SF's stage-1 warmup proportion.)

**Decision criteria**:
- H1 → confirms warmup helps at our scale. Queue 2A (WDL endpoint at 0.24)
  to test the second axis.
- H0 → drop the WDL-warmup direction.

**Wall-clock**: same as 1A.

---

### 1C. 4× batch size + 2× peak LR

**Hypothesis**: Larger batch + LR scaled by √batch (sub-linear) recovers
training quality while gaining 50% wall-clock. If H1 / neutral, all
subsequent experiments benefit from the speedup.

**Cost note**: Patch available at `feature/batch-size-flag` on
`adamtwiss/bullet` (commit e19fa54). Adds `--batch-size` /
`--batches-per-superbatch` flags; default preserves the 100M
positions/SB invariant when batch is overridden. Merge to bullet main
before running.

**Recipe** (after batch-size flag added):
```bash
cargo run --release --features cuda --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 200 \
  --wdl 0.15 \
  --warmup 30 \
  --kb-layout reckless \
  --hidden-activation crelu \
  --factoriser \
  --seed 42 \
  --save-rate 200 \
  --batch-size 65536 \
  --lr 0.002 \
  --net-id coda-v9-batch4x-lr2x
```

**Decision criteria**:
- H1 / non-regression → lock 4× batch as the new baseline; all future
  experiments inherit 50% wall-clock speedup.
- H0 → try `--lr 0.0014` (√batch scaling). If still H0 → revert, stay
  at 16k batch.
- Note: `--final-lr` may also need scaling proportionally if cosine
  endpoint sensitivity matters at the new batch.

**Wall-clock**: ~2.5h train (50% faster) + 6h SPRT = ~9h total.

---

## Tier 2 — Conditional on Tier 1 results

These are flag flips with no code change but should wait for Tier 1
signal before firing.

### 2A. WDL flat endpoint at 0.24 (no warmup)

**Hypothesis**: SF's endpoint (0.24) is better than ours (0.15) for our
data scale. Tests the endpoint axis independently of warmup.

**Recipe**:
```bash
cargo run --release --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 200 \
  --wdl 0.24 \
  --warmup 30 \
  --kb-layout reckless \
  --ob-layout material \
  --hidden-activation crelu \
  --factoriser \
  --seed 42 \
  --threads 8 \
  --save-rate 200 \
  --net-id coda-v9-wdl-0.24
```

**Run only if 1B (WDL warmup at 0.15) was H1 or neutral.** If 1B was
H0-negative, the WDL signal at our scale is fine at 0.15 — don't chase
the SF endpoint.

---

### 2B. WDL warmup 0→0.25 then flat 0.24 (full SF translation)

**Hypothesis**: Bundling warmup + SF endpoint = best WDL configuration
if both axes help independently.

**Recipe**:
```bash
cargo run --release --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 200 \
  --wdl 0.0 \
  --wdl-end 0.25 \
  --wdl-tail 0.24 \
  --wdl-tail-from 6 \
  --warmup 30 \
  --kb-layout reckless \
  --ob-layout material \
  --hidden-activation crelu \
  --factoriser \
  --seed 42 \
  --threads 8 \
  --save-rate 200 \
  --net-id coda-v9-wdl-sf-full
```

**Run only if BOTH 1B and 2A landed positive.** Bundle the winners.

---

## Tier 3 — Bullet code changes required

These require new mechanisms not currently in Bullet. Each is ~20-50 LoC.

### 3A. SWA (Stochastic Weight Averaging)

**Hypothesis**: Averaging the last 10-20% of training checkpoints
smooths the final weights, reducing noise from the end of cosine
decay. Standard ML technique; typical magnitude +1-3 Elo.

**Implementation sketch** (against `adamtwiss/bullet` fork):
1. Add `swa_start_sb: Option<usize>` to TrainingSchedule or as a CLI flag.
2. After each SB ≥ swa_start_sb, accumulate `swa_weights += current_weights`
   and `swa_count += 1`.
3. At end of training, save BOTH the final checkpoint and the SWA
   average (`swa_weights / swa_count`) as separate output dirs.
4. Convert both to .nnue via `coda convert-bullet`, SPRT each separately
   to identify the SWA contribution.

**Estimated LoC**: ~50 in `bullet_lib/src/value/trainer.rs` + a new
CLI flag in `coda_v9_768_threats.rs`. The "running average of model
state_dict" pattern is standard PyTorch — Bullet should have an analog
in its checkpoint-save path.

**Recipe** (after implementation, S200 example):
```bash
cargo run --release --example coda_v9_768_threats -- \
  [...canonical flags...] \
  --swa-start-sb 160 \
  --net-id coda-v9-swa
```

**SPRT setup**: two paired-probes vs baby-prod — one with the final
checkpoint, one with the SWA-averaged checkpoint. The delta isolates
the SWA effect.

**Decision criteria**: H1 on the SWA-averaged net (and not on the
final-checkpoint net) → SWA bankable. Lock as part of standard recipe.

**Risk**: at our smaller architecture (L1=16) the noise floor may be
low enough that averaging doesn't recover much. Magnitude prior is
+0-3 Elo.

---

### 3B. Jitter (WDL lambda noise injection)

**Hypothesis**: Random per-sample and per-batch perturbations of the
WDL blend (lambda) act as regularization, preventing overfitting to a
specific eval/WDL ratio. SF uses
`jitter-lambda-sample: 0.003, jitter-lambda-batch: 0.010,
jitter-decay-lambda-batch: 0.999`.

**Implementation sketch**:
1. In Bullet's WDL scheduler or loss-computation path, add Gaussian
   noise to the `lambda` value per-sample and per-batch.
2. Per-batch jitter decays its standard deviation by `decay` factor
   each step (~exp(-step/1000) for decay=0.999).
3. CLI flags: `--jitter-lambda-sample`, `--jitter-lambda-batch`,
   `--jitter-decay-lambda-batch`.

**Estimated LoC**: ~20 in the loss function path + CLI plumbing.

**Recipe** (after implementation, S200 example with SF values):
```bash
cargo run --release --example coda_v9_768_threats -- \
  [...canonical flags...] \
  --jitter-lambda-sample 0.003 \
  --jitter-lambda-batch 0.010 \
  --jitter-decay-lambda-batch 0.999 \
  --net-id coda-v9-jitter
```

**Decision criteria**: H1 → bankable. H0 → likely SF-recipe-specific,
abandon.

**Risk**: more speculative than SWA. SF's specific jitter values are
calibrated for their loss formulation; ours uses the same Bullet loss
so should transfer, but magnitudes may need rescaling. Magnitude prior
is +0-3 Elo.

**Sequence**: run **after** SWA so the two regularization mechanisms
are isolated. Both bundle-tested only if both pass independently.

---

## Recommended parallelism

The fleet has multiple GPU hosts (Hercules / Titan / Atlas / others).
If 3+ are available, fire **1A + 1B + 1C in parallel** — three
orthogonal probes, results in by end of day.

| Phase | Experiments | Duration | Hosts needed |
|---|---|---|---|
| 1 (now) | 1A (~16h), 1B (~10h), 1C (~9h) | ~16h wall-clock (bounded by 1A) | 3 |
| 2 (Tier 1 settled) | 2A or 2B conditional | ~10h | 1 |
| 3 (Tier 1/2 settled) | SWA implementation | dev time | 0 (code) |
| 4 (SWA built) | 3A | ~10h | 1 |
| 5 (3A settled) | Jitter implementation | dev time | 0 (code) |
| 6 (jitter built) | 3B | ~10h | 1 |

If only 1 host is available, run Tier 1 sequentially (1C first since
it unlocks speedup for the rest), then Tier 2 conditional, then
implement + test Tier 3 in order.

## Bookkeeping

Each new net should be uploaded to OB via `ob_upload_net.py` immediately
after `coda convert-bullet`. Record the SHA8 here as you fire each
experiment:

| Experiment | Net SHA8 | Bullet checkpoint dir | SPRT ID | Result |
|---|---|---|---|---|
| 1A fenskip-0.9 | TBD | `checkpoints/coda-v9-fenskip-0.9-200/` | TBD | TBD |
| 1B wdl-warmup | TBD | `checkpoints/coda-v9-wdl-warmup-200/` | TBD | TBD |
| 1C batch4x-lr2x | TBD | `checkpoints/coda-v9-batch4x-lr2x-200/` | TBD | TBD |
| 2A wdl-0.24 | TBD | `checkpoints/coda-v9-wdl-0.24-200/` | TBD | TBD |
| 2B wdl-sf-full | TBD | `checkpoints/coda-v9-wdl-sf-full-200/` | TBD | TBD |
| 3A SWA | TBD | `checkpoints/coda-v9-swa-200/` | TBD | TBD |
| 3B Jitter | TBD | `checkpoints/coda-v9-jitter-200/` | TBD | TBD |

Log each SPRT result to `experiments.md` as it resolves, with the
recipe diff vs canonical clearly stated.

## Open questions / risks

- **Reproducibility**: all recipes use `--seed 42` which fixes the data
  shuffle. Each experiment differs from baby-prod by exactly one axis
  (modulo bundles in Tier 2/3). This is the paired-probe protocol.
- **Recipe drift**: confirm the current `bullet/examples/coda_v9_768_threats.rs`
  flag defaults match what baby-prod was trained with. If they diverge
  silently, all paired-probes are muddied.
- **Compounding**: if multiple Tier 1 experiments pass, they should be
  bundled and re-tested as a single S200 candidate — but only one
  bundle per round to keep variable isolation.
- **S800 promotion**: Tier 3 results only matter if S200 wins scale.
  A clean Tier 1+2 winning combination should next be trained at S800
  and SPRT'd vs current prod (1EF1C3E5) on main, not mini-prod.
