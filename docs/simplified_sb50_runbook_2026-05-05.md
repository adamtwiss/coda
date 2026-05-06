# Simplified SB50 Diagnostic Runbook (2026-05-05)

## Purpose

Strip every "+5-10 Elo finesse" from the training recipe to see if the
bare-bones SB50 converges in line with anchor's expected SB50 result. This
isolates whether the residual ~500 Elo regression lives in the complex
features (factoriser fold-in, warmup schedule, WDL blending, hl-crelu) or
in the core training pipeline (sparse threat encoder, FT layer init,
optimizer math).

Each "feature" we strip costs ~5-15 Elo when correctly implemented. If the
stripped SB50 still lands ~400+ Elo behind anchor, the bug is in core
training. If it lands within ~50-100 Elo of anchor (roughly = penalty for
SB50-vs-SB200 + the stripped features), the bugs are in the layered
features and we'd then re-add them one at a time to bisect.

## Stripped features

- `--factoriser` REMOVED → no l0f shared PSQ weight, no fold-in at save
- `--warmup 0` → no warmup phase, pure cosine from SB 1
- `--wdl 0` → score regression only, no game-outcome blend
- `--hidden-activation` defaulted to **screlu** (drops hl-crelu, ~4 Elo)
- everything else same as canonical recipe

Anchor (current production net.txt) was trained WITH all these enabled.
Expected SB50 result if pipeline is healthy: **~70-120 Elo behind anchor**
(rough sum: −50 SB length + −10 factoriser + −10 warmup + −10 WDL + −4 hl-crelu).

## On the GPU host (post-rebase Bullet)

```bash
cd ~/code/bullet
git fetch origin
git checkout main
git pull   # post-rebase trunk; old bullet preserved on `pre-rebase` branch
cargo build --release --features cuda --example coda_v9_768_threats
```

**CRITICAL**: `--features cuda` is mandatory on post-rebase Bullet.
Without it the binary builds but the trainer panics at runtime.

## Training command

```bash
cd ~/code/bullet
cargo run --release --features cuda --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 50 \
  --wdl 0 \
  --warmup 0 \
  --kb-layout reckless \
  --ob-layout material \
  --seed 42 \
  --threads 8 \
  --save-rate 50 \
  --output-dir checkpoints \
  --net-id coda-v9-768t-stripped-sb50
```

Note no `--factoriser` and no `--hidden-activation` — both default to off /
screlu respectively.

Expected wall-clock: ~30-45 min on a 5070, ~50-75 min on a 4090.

Watch for any panic mentioning `MAX_THREAT_ACTIVE` (would invalidate the
experiment) or NaN/Inf in loss values.

## Convert (after training)

```bash
cd ~/code/coda
./target/release/coda convert-bullet \
  -i ~/code/bullet/checkpoints/coda-v9-768t-stripped-sb50-50/quantised.bin \
  -o nets/cal-day0-gpu1-v4-stripped-sb50.nnue \
  --pairwise --screlu \
  --hidden 16 --hidden2 32 --int8l1 \
  --threats 66864 \
  --kb-layout reckless
  # NOTE: no --hl-crelu (training used default screlu hidden)
```

If `--hidden-activation crelu` was OMITTED from training, the saved
hidden activation is screlu → convert without `--hl-crelu` flag.

## Validate

```bash
cd /home/adam/code/coda

./coda check-net --nnue nets/cal-day0-gpu1-v4-stripped-sb50.nnue
# Expect 8/8 pass (squared_error showed even bare-bones power_error path
# can give correct endgame buckets — only v2's factoriser+power_error combo
# corrupted them)

./coda bench 13 --nnue nets/cal-day0-gpu1-v4-stripped-sb50.nnue
```

## RR

Triangular comparing bare-bones SB50 against anchor and the previously-
banked broken-init / fixed-init / squared-error nets:

```bash
cd /home/adam/code/coda
cutechess-cli \
  -engine name=anchor      cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/net-v9-768th16x32-kb10-w15-e200s200-crelu-C8fix-factor.nnue \
  -engine name=v4-stripped cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/cal-day0-gpu1-v4-stripped-sb50.nnue \
  -engine name=v2-init     cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/cal-day0-gpu1-v2.nnue \
  -engine name=v3-sqerr    cmd=./coda dir=$PWD proto=uci option.NNUEFile=$PWD/nets/cal-day0-gpu1-v3.nnue \
  -each option.Hash=64 tc=0/10+0.1 \
  -rounds 200 -concurrency 16 \
  -openings file=/home/adam/code/gochess/testdata/noob_3moves.epd format=epd order=random \
  -pgnout v4_stripped_rr.pgn -recover -ratinginterval 20 \
  -draw movenumber=20 movecount=10 score=10 \
  -resign movecount=3 score=500 twosided=true \
  2>&1 | tee v4_stripped_rr.log
```

## Decision matrix

After the RR resolves to ±25 Elo bars (~600 games / engine):

| v4-stripped Elo gap to anchor | Diagnosis | Next probe |
|---:|:---|:---|
| **−50 to −150** | Pipeline is healthy at the bare-bones level. Bug is in the layered features (factoriser interaction with new backend, hl-crelu, or warmup/WDL combination). Bisect by re-adding features one at a time. | SB50 with each feature added back individually |
| **−150 to −300** | Pipeline has a moderate-magnitude bug in core training. May still be a feature interaction but not the dominant cause. | Probe encoder (drop x-ray threats: `--xray 0`) |
| **−400 to −600** | Pipeline core training is broken. Bug is in something common to all SB50 trainings: chess_threats encoder, sparse-input handling, FT init, or optimizer math. | Drop x-ray threats AND drop kb-layout (use uniform); compare to anchor pre-rebase commit retrain |

## Companion: parallel SB50 probes

If multiple GPUs available, fire in parallel for richer signal:

- **Probe X (HEAD, factoriser ON, hl-crelu ON, all features)** — replicates v2-init at SB50 for length-only baseline
- **Probe Y (anchor commit `8628c3b`, full anchor recipe)** — replicates anchor at SB50; if it lands at anchor strength, post-rebase code class is the bug; if it lags, environment changed
- **Probe Z (HEAD, `--xray 0`)** — strips x-ray threats from encoder; tests sparse-encoder-specific bug

Probes X and Y are easy parallel runs. Z requires no other code changes,
just a CLI flag. All three are SB50.
