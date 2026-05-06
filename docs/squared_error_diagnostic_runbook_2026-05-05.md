# Squared-error Diagnostic Runbook (2026-05-05)

## Purpose

Test whether the post-rebase `power_error(target, 2.5)` decomposition
(`Abs → Power` with autograd that hits `log(0)` when |output−target| ≈ 0)
is the source of the residual ~528 Elo SB200 regression after the init-line
fix. Branch `experiment/squared-error-diagnostic` (commit `c07f5f3` on
`bullet`) swaps the loss to `squared_error` (pointwise `diff * diff`, no
log-of-zero hazard) — everything else identical to canonical SB200 recipe.

Decision rule:

- **Net lands within ±50 Elo of anchor** → `power_error` decomposition
  is the second bug. Proceed to re-implement the fused kernel or
  sanitise the decomposition, then re-train at SB200 with
  `power_error(target, 2.5)` restored.
- **Net still hundreds of Elo behind anchor** → diagnosis points
  elsewhere. Next move is SB1 weight/gradient-norm instrumentation.

## Length: SB100 is fine

The 400+ Elo gap is wide enough that the ~30 Elo SB200→SB100 undertrain
penalty doesn't muddy the signal. SB100 is ~1.5h on a 5070-class host vs
~2.5h for SB200. Use SB100 for the diagnostic to free the GPU sooner;
upgrade to SB200 only if SB100 result is borderline.

## On the GPU host

```bash
cd ~/code/bullet
git fetch origin
git checkout experiment/squared-error-diagnostic
git pull
# expect: HEAD = c07f5f3 "DIAGNOSTIC: swap power_error(2.5) → squared_error..."
cargo build --release --features cuda --example coda_v9_768_threats
```

**CRITICAL: `--features cuda` is mandatory on post-rebase Bullet.** The
upstream backend rewrite (commit `2ff6d54`) made the CUDA backend
opt-in via cargo feature flag. Without `--features cuda` the build
succeeds but the trainer panics at runtime trying to allocate device
tensors. Same flag required on `cargo run --release` if invoking
directly without a separate build step.

## Training command (SB100 variant)

Canonical recipe with `--seed=42`, `--factoriser`, `--hl-crelu` (via
`--hidden-activation crelu`), `--kb-layout reckless`, `--ob-layout
material`, `--wdl 0.15`. Output paths set so it doesn't collide with the
gpu1/gpu1-v2 runs.

```bash
cd ~/code/bullet
cargo run --release --features cuda --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 100 \
  --wdl 0.15 \
  --warmup 15 \
  --kb-layout reckless \
  --ob-layout material \
  --hidden-activation crelu \
  --factoriser \
  --seed 42 \
  --threads 8 \
  --save-rate 100 \
  --output-dir checkpoints \
  --net-id coda-v9-768t-squarederr-sb100
```

(For SB200 instead, change `--superbatches 200`, `--warmup 30`,
`--save-rate 200`, and `--net-id coda-v9-768t-squarederr-sb200`. Same
command otherwise.)

`--warmup 15` keeps warmup at 15% of total (matches anchor's ratio).
The "warmup not proportional" memory finding was at SB200; we don't
have data at SB100 either way. 15% is the safe default for shorter runs.

`--save-rate` set to total SBs so we only checkpoint at the end —
disk-cheap and we only care about the final net.

Watch for any panic mentioning `MAX_THREAT_ACTIVE` overflow (would
invalidate the experiment). Watch for `NaN`/`Inf` in loss output (would
also be diagnostic).

## After training: convert

On the GPU host (or wherever the .bin lands):

```bash
cd ~/code/coda
./target/release/coda convert-bullet \
  -i ~/code/bullet/checkpoints/coda-v9-768t-squarederr-sb100-100/quantised.bin \
  -o nets/cal-day0-gpu1-v3-squarederr-sb100.nnue \
  --pairwise --screlu \
  --hidden 16 --hidden2 32 --int8l1 \
  --threats 66864 \
  --kb-layout reckless \
  --hl-crelu
```

(Adjust the input path / SB number if SB200 instead.)

## After convert: validate loads + bench + RR

```bash
cd ~/code/coda

# 1. Net loads + threat features + flag bits
./coda check-net --nnue nets/cal-day0-gpu1-v3-squarederr-sb100.nnue

# 2. Bench (fixed depth) + tree-shape data
./coda bench 13 --nnue nets/cal-day0-gpu1-v3-squarederr-sb100.nnue

# 3. Triangular RR vs anchor + gpu1-v2-fix on Hercules
#    (kill OB worker first if needed: ./ob-worker.sh stop)
cutechess-cli \
  -engine name=anchor   cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/net-v9-768th16x32-kb10-w15-e200s200-crelu-C8fix-factor.nnue \
  -engine name=v2-init  cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/cal-day0-gpu1-v2.nnue \
  -engine name=v3-sqerr cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/cal-day0-gpu1-v3-squarederr-sb100.nnue \
  -each option.Hash=64 tc=0/10+0.1 \
  -rounds 300 -concurrency 16 \
  -openings file=/home/adam/code/gochess/testdata/noob_3moves.epd format=epd order=random \
  -pgnout sqerr_diagnostic.pgn -recover -ratinginterval 20 \
  -draw movenumber=20 movecount=10 score=10 \
  -resign movecount=3 score=500 twosided=true \
  2>&1 | tee sqerr_diagnostic.log
```

900 games at 10+0.1, ~10-12 min on Hercules.

## Reading the result

Anchor's 600-game pairwise gap to gpu1-v2 was −528 Elo (±25). The
diagnostic SB100 result will be one of:

- **v3-sqerr within ±50 Elo of anchor** → `power_error` decomp
  confirmed. Likely outcome if the agent's hypothesis is right.
- **v3-sqerr at ~−500 Elo (matches v2-init)** → diagnosis wrong, the
  bug is somewhere else in the rebase.
- **v3-sqerr in between** (e.g. ~−250 Elo) → there's MORE THAN ONE
  remaining bug; squared_error fixed part of it. Next probe needed.

The SB100 vs SB200 difference (~30 Elo) is small enough that the
decision rule still applies cleanly at SB100.

## Post-diagnosis paths

If `power_error` is confirmed:

1. **Restore fused kernel.** The pre-rebase fused `|x|^p` + derivative
   kernel lived in `crates/bullet_hip_backend/kernels/base/power_error.cu`
   (deleted in upstream PR #488). Port it to the new backend's kernel
   conventions. Bank the +3-5 Elo Viridithas-style trade.
2. **OR sanitise the decomposition.** Add `clamp(input, eps, ∞)` before
   `log(input)` in the `Power` op autograd to prevent log(0). Cheaper
   patch, may have its own subtle issues.

After fix, re-train an SB200 with `power_error(target, 2.5)` restored
on a fresh branch (`fix/power-error-fused` or similar), validate
against anchor, and merge to `main` (post-rebase trunk; old bullet on `pre-rebase`).

If `power_error` is NOT the bug, the next diagnostic is to instrument
SB1 weight/gradient norms (a separate runbook to be written).
