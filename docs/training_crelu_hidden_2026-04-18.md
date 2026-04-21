# Training instructions: clipped-ReLU L1/L2 + warm30 experiment (2026-04-18)

## Status as of 2026-04-21 — RESOLVED ✓

Both experiments resolved positively, landed as compounding production
changes:

1. **warm30 — H1 at +4.44 Elo (#496)**, merged. Peak of the warmup
   curve at e200 (warm40 and warm50 both H0 vs warm30). Full curve
   captured in `training_warmup_curve_2026-04-19.md`.
2. **CReLU hidden — now the v9 production activation.** Training
   supported via `--hidden-activation crelu` flag on Bullet's
   coda_v9_768_threats example. Produced the current v9 prod net
   `net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue` (DAA4C54E) which
   landed **#582 +15.2 Elo H1** — biggest single net upgrade this
   cycle. Coda inference auto-configures from header bit 5 (landed
   as `nnue/hl-crelu-file-marker` merge).

Both experiments became production. No outstanding follow-ups from
this doc.

---

Two training experiments for idle GPUs.

## Experiment 1: warm30 (one GPU)

Continuation of the warmup-length investigation. Current data:
- warm5: -6.7 Elo vs warm20 (SPRT #475, flat-trending-H0)
- warm20: baseline
- warm30: **to test**

Slope so far says longer warmup helps. warm30 tests whether the
improvement continues past 20 SBs or plateaus.

**Config change:** none to source. Just pass `--warmup 30` on the CLI.

**Command** (assuming GPU host layout matches `~/code/bullet/examples/coda_v9_768_threats.rs`):

```bash
cd ~/code/bullet
cargo run --release --example coda_v9_768_threats -- \
    --dataset /workspace/data/T80_test-80-2024-01-d9-12B.binpack \
    --superbatches 200 \
    --warmup 30 \
    --wdl 0.15 \
    --kb-layout reckless \
    --save-rate 50
```

Adjust dataset path to match host (some hosts use `/training/data/`).

**Output:**
- Save as `coda/nets/net-v9-768th16x32-kb10-w15-e200s200-warm30.nnue`
  (via `coda convert-bullet -input quantised-s200.bin -output net-v9-768th16x32-kb10-w15-e200s200-warm30.nnue -screlu -hidden 16 -hidden2 32 -int8l1`)

**Validation:** SPRT vs `net-v9-768th16x32-kb10-w15-e200s200` (hash
BE5849B6) on `feature/threat-inputs`, bounds `[-3, 3]`.

## Experiment 2: clipped-ReLU L1/L2 (one GPU)

Test whether Reckless's clipped-ReLU on hidden layers works for v9.
Reckless uses clamp(0, 1) on L1/L2 (instead of our SCReLU which is
clamp(0, 1)²). Simpler activation, cheaper inference, potentially
trains more stably.

Prior note (in conversation, saved for reference):
- CReLU on FT: clamp [0, 255], integer, before pairwise. Both Reckless
  and Coda do this. ✓
- SCReLU on hidden layers: clamp then square. Viridithas found +50%
  effective network size vs CReLU. Coda currently uses this.
- Clipped ReLU on hidden layers: just clamp [0, 1] in float. Simple.
  Reckless uses this.

Reckless is #2 on CCRL despite not using SCReLU. Plausible that threat
features + their setup make the activation matter less, OR clipped
ReLU trains more stably at narrow hidden widths (16→32).

**Config change:** one line in `~/code/bullet/examples/coda_v9_768_threats.rs`:

```diff
-            let hl2 = l1.forward(hl1).screlu();
-            let hl3 = l2.forward(hl2).screlu();
+            let hl2 = l1.forward(hl1).crelu();
+            let hl3 = l2.forward(hl2).crelu();
```

Lines 191-192 in the current file. Save as a variant (either a new
example file or a git branch).

**Command** (same as warm30 but default warmup=20):

```bash
cd ~/code/bullet
cargo run --release --example coda_v9_768_threats -- \
    --dataset /workspace/data/T80_test-80-2024-01-d9-12B.binpack \
    --superbatches 200 \
    --warmup 20 \
    --wdl 0.15 \
    --kb-layout reckless \
    --save-rate 50
```

**Output:**
- Save as `coda/nets/net-v9-768th16x32-kb10-w15-e200s200-creluHL.nnue`
  (converter command same as warm30; Coda will auto-detect SCReLU vs
  CReLU hidden via UCI option `HiddenActivation` at load time — no
  converter change needed yet).

**Validation:**
- Adam will load with `setoption name HiddenActivation value crelu`
  then SPRT vs `net-v9-768th16x32-kb10-w15-e200s200` baseline,
  bounds `[-3, 3]`.
- EVAL_SCALE is self-tuning via SPSA; no pre-flight RMS calibration
  needed.

## Coda-side inference work (Hercules handles)

Adding `HiddenActivation` UCI option to Coda. When set to "crelu":
- Skip the `v * v` step in the L1 and L2 activation paths.
- The CReLU-only code path already exists as part of the dual_l1
  branch (nnue.rs:2331, 2487). Just needs to be reachable without
  requiring dual_l1 net format.

Expected merge within a few hours. GPU training can start immediately
— inference will be ready before the net finishes training.
