# Move Ordering Analysis and Improvement Plan

## Current Baseline (2026-04-12, post-tune2 + selfplay discovery)

```
Move ordering (e800-filtered-lowestlr, tuned):  avg cutoff pos 1.48, avg pos² 7.3, first-move 81.7%
EBF: 1.84, Total nodes: 1,993,680
```

Improved from 75% → 81.7% through history magnitude fix, killer removal, SPSA tuning,
and better NNUE models. Stockfish achieves 85-90% — gap closing fast.

### Impact of Different NNUE Models on Ordering

| Model | First-move % | Avg pos² | Notes |
|-------|-------------|----------|-------|
| 768pw selfplay s120 | **81.7%** | **7.3** | Best ever — selfplay data matches our search |
| e800-filtered-lowestlr (production) | 80.8% | ~8 | Production net, SPSA-tuned |
| T80 baseline s120 | 78.8% | 12.6 | Same arch, T80 data |
| MSE loss s120 | 81.6% | 6.6 | T80, MSE loss |
| Power-2.5 loss s120 | 80.8% | 6.7 | T80, power loss |
| v7 1024h16x32s | 52.7% | 11.0 | Dead hidden layers — training issue, not inference |

## Completed Experiments

### Merged
| Feature | Elo | Date | OB # | Notes |
|---------|-----|------|------|-------|
| Quiet check bonus | +6.0 | Apr 6 | | Checking squares bitboard, +10K bonus |
| History magnitude fix | +31.6 | Apr 6 | | 27× too small — fixed all history tables |
| Capture history fix | +5.4 | Apr 6 | | Unconditional malus + linear bonus |
| LMP BASE 9→5 | +3.5 | Apr 6 | | Working history enables better LMP |
| TT PV flag | +4.5 | Apr 7 | | Sticky PV bit, LMR -1 for PV positions |
| Cont-hist malus + tune | +6.5 | Apr 7 | | TT cutoff penalizes opponent's quiet (needed retune) |
| LMR simplify + tune | +6.3 | Apr 8 | | Remove failing/alpha_raised/unstable (needed retune) |
| NMP capture R flip + tune | +3.5 | Apr 9 | #195 | Flip r-=1 to r+=1 (needed retune) |
| Killer/counter-move removal + tune | +3.77 | Apr 9 | #215 | Pure history ordering. 53 lines removed. |
| SEE piece values update | +1.7 | Apr 10 | #281 | N=420, B=420, R=640, Q=1200 (consensus) |
| SPSA tune #280 (e800 lowestlr net) | +23.7 | Apr 10 | #282 | Full 48-param retune for new net |
| SPSA tune #283 (refinement) | +3.5 | Apr 12 | #287 | SEE values + new tunables calibration |

### Resolved — Not Merging
| Feature | Attempts | Result | OB # | Notes |
|---------|----------|--------|------|-------|
| Cont hist equal weights (1x/1x) | 2 | -1.1 to -6.3 | | 3x/3x genuinely better |
| Root history table | 3 | -0.8 to -7.9 | | Root ordering already good |
| Piece-type threat escape bonus | 1 | -2.6 | | 4D history handles threats |
| Low-ply history bonus | 1 | -2.9 | #209 | 8× at ply 0 doesn't help us |
| Good/bad quiet split + tune | 1 | -3.6 | #216 | SF-only feature. May revisit with new nets. |
| Corr-cont-2ply + tune | 1 | -1.1 | #213 | Dead flat at 28K games. May revisit. |
| Histprune no-improving + tune | 1 | -3.4 | #220 | Faded from early positive |
| Futility-full + tune | 1 | -5.7 | #222 | History→lmrDepth. Didn't help. |

### Revisit Candidates
Features that failed previously but conditions have changed (new nets, new SEE values, better SPSA calibration):

| Feature | Previous Result | Why Revisit |
|---------|----------------|-------------|
| Dynamic capture SEE threshold | -0.5 to -11.6 (5 attempts) | SEE values and SEE_MATERIAL_SCALE completely changed. Worth one more try. |
| Good/bad quiet split | -3.6 (#216) | Selfplay net has much better move ordering (pos² 7.3 vs 12.6). Threshold may work now. |
| Corr-cont-2ply | -1.1 (#213) | Eval improvements may make 2-ply correction signal cleaner. |
| Pawn history initialisation | Untested | PlentyChess pattern: -1000 pessimistic prior. |
| Capture scoring MVV×16 → lower | Failed 5× | With SEE_MATERIAL_SCALE at 159 and new values, ratios changed. |

## LMP Interaction (resolved)

LMP_BASE was pushed to 9 by SPSA (nearly disabling LMP). Two fixes resolved this:
1. **History magnitude fix** — working history means late moves are genuinely bad. LMP BASE dropped to 5.
2. **Retune methodology** — LMP BASE now converges to 6-8 depending on the feature branch, which is healthy.

## Key Lessons

1. **History magnitude is everything** — 27× fix improved all move ordering metrics and cascaded through every pruning feature.
2. **Retune-on-branch discovers hidden value** — features that change tree shape need parameter recalibration to show their Elo. 4/4 retune candidates turned from H0 to positive.
3. **Killer removal works** — +3.77 Elo from removing killers/counters. History tables (post magnitude fix) fully replace them.
4. **Move ordering and pruning are coupled** — better ordering enables tighter pruning (LMP, futility), which enables deeper search, which enables better ordering. Virtuous cycle.
5. **Better eval = better history** — selfplay net's pos² of 7.3 (vs 12.6 baseline) shows that consistent eval produces cleaner history signal. As nets improve, history-dependent features gain more.
