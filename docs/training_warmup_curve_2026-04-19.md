# Training: complete the warmup-length curve (2026-04-19)

Following warm5 (−7 Elo), warm20 (baseline), warm30 (+4.44 Elo @ #496)
at e200. Two more data points needed to see shape, not chase Elo.

## RESOLVED (2026-04-21)

warm40 and warm50 both trained at e200 and SPRT'd. **warm30 was the
strongest**; warm40 and warm50 each regressed vs warm30 at SB200.
Shape: positive from 5→30, peak at 30, mild regression beyond.

**Conclusion:** don't warmup past ~15% of total training length.
For e200 that's warm30. For e400/e800 production training, anchor
at warm30 unless absolute-count follow-up (below) says otherwise.
The "does longer training need longer warmup" question is deferred
to e400/e800 production runs.

Original design/motivation preserved below for reference.

---

## Why these matter (original)

- warm30 is positive. Question: does it monotonically improve with
  longer warmup, plateau, or reverse?
- At e200, warm50 (25% warmup) leaves only 150 SB of cosine-decay
  tail. Our `v9_low_lr_tail_critical` finding says the tail buys
  ~100 Elo; shortening it 12% might start to hurt.
- Clean one-variable curve at e200 is the methodology we want — not
  production candidates.

Expected outcomes:
- All positive (monotone): warmup length genuinely matters; longer
  is better even at the cost of tail. Probably test at e800 too.
- warm40 positive, warm50 flat-or-negative: peak is ~warm30-40.
  That's where to anchor e800 experiments.
- Both regress: warm30 was the peak; don't go further.

## Experiments

### Experiment A: warm40 at e200

```bash
cd ~/code/bullet
cargo run --release --example coda_v9_768_threats -- \
    --dataset /workspace/data/T80_test-80-2024-01-d9-12B.binpack \
    --superbatches 200 \
    --warmup 40 \
    --wdl 0.15 \
    --kb-layout reckless \
    --save-rate 50
```

Output: `coda/nets/net-v9-768th16x32-kb10-w15-e200s200-warm40.nnue`
(via `coda convert-bullet -input quantised-s200.bin -output <above> -screlu -hidden 16 -hidden2 32 -int8l1`)

SPRT vs `net-v9-768th16x32-kb10-w15-e200s200` (hash BE5849B6) on
`feature/threat-inputs`, bounds `[-3, 3]`.

### Experiment B: warm50 at e200

Same command with `--warmup 50`.

Output: `net-v9-768th16x32-kb10-w15-e200s200-warm50.nnue`.

Same SPRT setup.

## Non-goal

These are not production candidates. If positive, we'll use the
shape information to choose which warmup length to pair with
creluHL at e400/e800 training, not promote these specific nets.

## Follow-up: ratio vs absolute-count

Once the e200 curve is filled in, a separate question remains:
does warmup length matter as **absolute SB count** or as
**fraction of total training**?

Test design (one data point):
- If warm30/e200 ≈ warm30/e400 (same count, different fraction) →
  absolute count dominates.
- If warm30/e200 ≈ warm60/e400 (same fraction, different count) →
  fraction dominates.

Cheapest version: use an already-trained e200-warm30 as reference,
train one new e400-warm30 net. That's one extra training run for
the answer.

Deferred until the warm40/warm50 data lands — might change what
"equivalent" warmup means for the e800 production run.
