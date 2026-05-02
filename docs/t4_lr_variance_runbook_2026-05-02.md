# T4 — Final-LR variance probe runbook (2026-05-02)

**Purpose:** test whether raising final LR reduces single-replica variance.
**Not measured:** strength vs prod, optimal LR. Those are separate experiments.
**Outcome question:** are 3 replicas at high-LR clustered tighter than T1's gpu3/4/5 hl-screlu cohort (currently ±20 Elo single-replica spread)?

## Background

- Current Coda V9 final LR: **2.43e-7** (set explicitly via `--final-lr 2.43e-7`)
- This is 10× lower than Bullet's example reference (2.4e-6) and 33× lower than Hobbes h-8+ cosine endpoint (8.1e-6).
- Lower than 2.43e-7 has been tested and regressed — floor known.
- Above 2.43e-7 has **not** been tested at this architecture.
- T1 ruled out hl-crelu activation as variance source (3 hl-screlu replicas still spread ±20 Elo). Factor previously ruled out. LR is the leading remaining suspect.

## LR options

Pick one before firing. Both are within "tested elsewhere" range — neither risks runaway non-convergence.

| Multiplier | Final LR | Reference points | Risk profile |
|---|---:|---|---|
| **10×** | **2.43e-6** | = Bullet examples reference; ≈ Hobbes h-1..h-7 cosine endpoint (2.7e-6) | Conservative. This is literally the Bullet *default* (remove the `--final-lr` override and you get this value). Should converge cleanly. |
| **30×** | **7.3e-6** | ≈ Hobbes h-8+ cosine endpoint (8.1e-6) | Aggressive but known-safe at the Hobbes-architecture level. Larger gap from current = stronger consistency signal if LR is the lever. Some risk of late-stage convergence wiggle on V9's threat-feature tail. |

**Recommendation:** start with **30×** (7.3e-6). The variance-test signal is amplified by larger gap from current; if 30× shows tightened clustering we definitively implicate LR. If 30× regresses on absolute strength but tightens variance, we've still confirmed the mechanism and can iterate to find the optimum at lower mult later.

If you want to be more conservative and pick 10×: still informative but smaller signal-to-noise ratio if the variance reduction is partial.

## Bullet command — 3 sequential SB200 trains

Same recipe as T1 (hl-screlu, non-factor) except `--final-lr`:

```bash
# 30× option (recommended): 7.3e-6
cd /home/adam/code/bullet
for SEED_TAG in s1 s2 s3; do
  cargo run --release --example coda_v9_768_threats -- \
    --data-dir /workspace/data \
    --superbatches 200 \
    --wdl 0.15 \
    --lr 0.001 \
    --warmup 30 \
    --final-lr 7.3e-6 \
    --hidden-activation screlu \
    --xray 1 \
    --save-rate 50 \
    2>&1 | tee /workspace/logs/v9-lr30x-${SEED_TAG}.log

  # Move final checkpoint with seed-specific name
  mv /workspace/quantised-s200.bin /workspace/quantised-lr30x-${SEED_TAG}.bin
done

# 10× option: --final-lr 2.43e-6  (or omit --final-lr entirely; that's Bullet's default)
```

Seeds differ between runs because Bullet derives them from `SystemTime::now()` per data-loader init (`bullet_lib/src/value/loader/rng.rs:7`). Sequential launches → different launch times → different seeds.

**Wall-clock:** SB200 ≈ 7-10h per replica, ~24-30h sequential for 3 replicas. Fits today's "I'm out, training is fine" envelope.

**Save rate 50** (4 checkpoints over SB200) avoids disk-fill from save-rate-25 default.

## Convert to .nnue

For each `.bin`, on the GPU host:

```bash
cd /home/adam/code/coda
for SEED_TAG in s1 s2 s3; do
  ./coda convert-bullet \
    -input /workspace/quantised-lr30x-${SEED_TAG}.bin \
    -output nets/net-v9-768th16x32-kb10-w15-e200s200-hlscrelu-nonfactor-lr30x-${SEED_TAG}.nnue \
    -screlu \
    -hidden 16 \
    -hidden2 32 \
    -int8l1 \
    -kb-layout reckless \
    -kb-count 10 \
    -threats
done
```

(Verify `convert-bullet` flags match training config — see `feedback_convert_bullet_invocation`. Wrong flags silently corrupt the net.)

## SPRT — variance assessment

**Three cross-replica pairs at [-3, 3] / 10+0.1 / 1T / Hash=64**, same protocol as T1 (#915/916/917):

```bash
# Upload first
for n in nets/net-v9-768th16x32-kb10-w15-e200s200-hlscrelu-nonfactor-lr30x-s*.nnue; do
  OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_upload_net.py "$n"
done

# Bench each locally to get per-side bench numbers (CRITICAL — see feedback_net_vs_net_sprt_per_side_bench)
for n in nets/net-v9-768th16x32-kb10-w15-e200s200-hlscrelu-nonfactor-lr30x-s*.nnue; do
  echo "=== $(basename $n) ==="
  ./coda --nnue "$n" bench 2>&1 | grep -E "^[0-9]+ nodes|First-move cut|EBF"
done

# Submit 3 cross-replica pairs (replace SHA + bench placeholders):
OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_submit.py main BENCH_S1 \
  --base-bench BENCH_S2 \
  --dev-network SHA_S1 \
  --base-network SHA_S2 \
  --bounds '[-3, 3]'
# ... and similarly for s1-vs-s3, s2-vs-s3
```

## Outcome decode

Compare to T1 hl-screlu cohort (current LR):
- **#915 gpu4-vs-gpu5: −4.34 ±9.56 / 1360 games** (still resolving, currently tight)
- **#916 gpu3-vs-gpu5: +20.13 ±12.65 / 812 games**
- **#917 gpu3-vs-gpu4: +20.36 ±11.98 / 820 games**

T1 spread: gpu3 outlier ~+20 above gpu4≈gpu5. Spread ≈ ±20.

**Decision rule:**
- **All 3 high-LR pairs cluster within ±5 each:** LR is the variance lever. Queue follow-up: optimal-LR experiment (4-point bracket from training_patterns_2026-04-30.md:118).
- **One outlier replica (~±15-20):** mixed evidence. LR may help partially. Queue at SB50 with more seeds (4-6) at SB50 to triage cheaper before committing to next SB200.
- **All 3 pairs ±15-20:** LR isn't the lever. Reframe — focus shifts to other suspects (MAX_THREAT_ACTIVE bump, v9 architecture inherent variance) or accept variance as structural.

## Future iterations: SB50/SB80 default

Per Adam (2026-05-02): future variance probes should default to SB50 or SB80 (~2-3h per replica vs SB200's 7-10h). SB200 is reserved for promotion candidates. This run is SB200 because today is "I'm out, training is fine" — the 24h wall-clock isn't a productivity bottleneck. Subsequent variance experiments → SB50 ×3 first, promote on positive signal.

## Files

- This runbook
- `docs/seed_variance_investigation_2026-05-01.md` — full hypothesis trail
- `docs/training_patterns_2026-04-30.md` §LR — cross-engine LR reference points
- `feedback_net_vs_net_sprt_per_side_bench.md` — bench gotcha
- `feedback_convert_bullet_invocation.md` — convert-bullet flag verification
