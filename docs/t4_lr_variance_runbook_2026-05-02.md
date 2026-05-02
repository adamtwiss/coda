# T4 — Final-LR variance probe runbook (2026-05-02)

**Purpose:** test whether raising final LR reduces single-replica variance.
**Not measured:** strength vs prod, optimal LR. Those are separate experiments.
**Outcome question:** are 3 replicas at high-LR clustered tighter than T1's gpu3/4/5 hl-screlu cohort (currently ±20 Elo single-replica spread)?

## Background

- Current Coda V9 final LR: **2.43e-6** (Bullet default, `initial_lr * 0.3^5` with `initial_lr=0.001`). Recent models, including T1's gpu3/4/5 hl-screlu replicas, do not pass `--final-lr` and use this default.
- Lower than 2.43e-7 (10× below current) has been tested and regressed — floor known there.
- Above 2.43e-6 has **not** been tested at this architecture.
- Reference points: Coda V5 was 5e-6 (≈ 2× current); Hobbes h-8+ cosine endpoint is 8.1e-6 (≈ 3.3× current).
- T1 ruled out hl-crelu activation as variance source (3 hl-screlu replicas still spread ±20 Elo). Factor previously ruled out. LR is the leading remaining suspect.

## LR options

Pick one before firing. All deliver `--final-lr <value>` to the Bullet config.

| Multiplier | Final LR | Reference points | Risk profile |
|---|---:|---:|---|
| **2×** | **4.86e-6** | ≈ Coda V5 (5e-6) — known-safe at our previous architecture | Most conservative bump. Smaller signal if LR is the lever. |
| **3×** (recommended) | **7.3e-6** | ≈ Hobbes h-8+ cosine endpoint (8.1e-6); +9.6 Elo *strength* in Hobbes's smaller arch | Within tested envelope at smaller architectures. Meaningful 3× gap from current = clear consistency signal if LR is the lever. |
| **10×** | **2.43e-5** | 3× higher than any engine reference; untested territory | Larger signal but runs the risk that "non-convergence" looks like "variance reduced" if late SBs fail to settle. |

**Recommendation:** start with **3× (7.3e-6)**. Within tested envelope, clean signal-to-noise on the variance question, and cheap follow-up if positive (queue 10× to see if more is better).

If 3× shows same ±20 spread, LR isn't the lever — save the 10× / 30× experiments and move to other suspects.

## Bullet command — 3 sequential SB200 trains

Same recipe as T1 (hl-screlu, non-factor) except `--final-lr`:

```bash
# 3× option (recommended): 7.3e-6
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
    2>&1 | tee /workspace/logs/v9-lr3x-${SEED_TAG}.log

  # Move final checkpoint with seed-specific name
  mv /workspace/quantised-s200.bin /workspace/quantised-lr3x-${SEED_TAG}.bin
done

# Other options: --final-lr 4.86e-6 (2×, conservative) or --final-lr 2.43e-5 (10×, aggressive)
# Note: T1 baseline used Bullet default (no --final-lr) = 2.43e-6.
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
    -input /workspace/quantised-lr3x-${SEED_TAG}.bin \
    -output nets/net-v9-768th16x32-kb10-w15-e200s200-hlscrelu-nonfactor-lr3x-${SEED_TAG}.nnue \
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
for n in nets/net-v9-768th16x32-kb10-w15-e200s200-hlscrelu-nonfactor-lr3x-s*.nnue; do
  OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_upload_net.py "$n"
done

# Bench each locally to get per-side bench numbers (CRITICAL — see feedback_net_vs_net_sprt_per_side_bench)
for n in nets/net-v9-768th16x32-kb10-w15-e200s200-hlscrelu-nonfactor-lr3x-s*.nnue; do
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
- **All 3 high-LR pairs cluster within ±5 each:** LR is the variance lever. Queue follow-up: optimal-LR experiment (4-point upward bracket from `2.43e-6` per `training_patterns_2026-04-30.md` Probe 5).
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
