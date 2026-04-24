# Move Ordering Quality Tracking

Track first-move cut rate and avg cutoff position across net versions.
These are key indicators of eval quality for search efficiency.

Run: `coda bench 13 --nnue <net>` and record the "Move ordering" line.

## Baseline (2026-04-16)

| Net | First-move cut | Avg cutoff pos | Avg pos² | Total nodes | Notes |
|-----|---------------|----------------|----------|-------------|-------|
| v5 production (consensus) | **82.1%** | 1.55 | 9.2 | 994,607 | Target quality |
| v9 s200 | 72.7% | 1.95 | 17.8 | 1,331,483 | |
| v9 s400 (pre-tune) | 72.4% | 2.08 | 22.5 | 2,058,785 | s400 worse than s200! |
| v9 s400 (with escape bonus) | 71.6% | 2.04 | 20.4 | 1,998,575 | |
| v9 s400 (tune r3) | 72.4% | 2.12 | 24.9 | 1,637,590 | Tune traded pos for fewer nodes |

## Key Observations

1. **v5 is at 82%, v9 at 71-72%.** The 10pp gap is significant — v9 tries ~30% more
   moves before finding the cutoff. This costs nodes and Elo.

2. **s200 → s400 went in the wrong direction** (72.7% → 72.4%). More training didn't
   improve move ordering. Possible explanation: longer training refined position scoring
   but the extra threat weight updates added noise to move discrimination.

3. **The v9 accumulator is half the width of v5** (768 vs 1536). Fewer dimensions =
   less ability to discriminate between similar moves. Threats add positional knowledge
   but not per-move discrimination.

4. **Track this for EVERY new net.** If s800/x-ray/consensus-bucket nets don't improve
   first-move cut, the narrow accumulator may be a structural limitation.

## What to watch for

- **Improving**: first-move cut trending toward 75%+ = threat features learning useful
  move discrimination patterns. Training quality is improving.
- **Flat at 71-72%**: the narrow accumulator can't distinguish moves well enough.
  Consider wider v9 (1024 + threats) or different training approaches.
- **Declining**: something is wrong — more training is adding noise not signal.
