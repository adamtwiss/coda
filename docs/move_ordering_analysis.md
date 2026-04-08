# Move Ordering Analysis and Improvement Plan

## Current Baseline (2026-04-08, post-history-fix + tt-pv + malus)

```
Move ordering:  avg cutoff pos 1.92, avg pos² 16.6, first-move 78.5%
EBF: 1.87, Total nodes: 1,780,721
```

Improved from 75% → 78.5% with quiet check bonus, history magnitude fix, tt-pv flag.
Stockfish achieves 85-90% first-move cutoff rate — gap is closing.

### Impact of Different NNUE Models on Ordering

| Model | First-move % | Avg pos | Avg pos² | Notes |
|-------|-------------|---------|----------|-------|
| 768pw (production) | **75.0%** | 1.91 | 16.6 | Pre-history-fix baseline |
| 1024s-13f (blunders) | **76.0%** | 1.88 | 17.6 | Best first-move — diverse training helps |
| 1024s | 71.6% | 1.84 | 13.6 | Lower first-move but tight distribution |
| 1024c (oldest) | 72.6% | 2.34 | 30.7 | Worst late-miss tail |
| v7 1024h16x32s | **52.7%** | 2.09 | 11.0 | Broken — eval inconsistent between iterations |

## Completed Experiments

### Merged
| Feature | Elo | Date | Notes |
|---------|-----|------|-------|
| Quiet check bonus | +6.0 | Apr 6 | Checking squares bitboard, +10K bonus |
| History magnitude fix | +31.6 | Apr 6 | 27× too small — fixed all history tables |
| Capture history fix | +5.4 | Apr 6 | Unconditional malus + linear bonus |
| LMP BASE 9→5 | +3.5 | Apr 6 | Working history enables better LMP |
| TT PV flag | +4.5 | Apr 7 | Sticky PV bit, LMR -1 for PV positions |
| Cont-hist malus + tune | +6.5 | Apr 7 | TT cutoff penalizes opponent's quiet (needed retune) |
| LMR simplify + tune | +6.3 | Apr 8 | Remove failing/alpha_raised/unstable (needed retune) |

### Permanently Dropped
| Feature | Attempts | Result | Notes |
|---------|----------|--------|-------|
| Dynamic capture SEE threshold | 5 | -0.5 to -11.6 | Captures are opponent-dependent |
| Cont hist equal weights (1x/1x) | 2 | -1.1 to -6.3 | 3x/3x genuinely better |
| Root history table | 3 | -0.8 to -7.9 | Root ordering already good |
| Piece-type threat escape bonus | 1 | -2.6 | 4D history handles threats |

### Testing (Apr 8)
| Feature | Status | Notes |
|---------|--------|-------|
| Good/bad quiet split | Branch ready | SF-only feature. Avg cutoff 1.92→1.82, nodes -4.4% |
| NMP capture R flip + tune | SPRT running | +4.1 at 5.7K games, trending H1 |
| Corr-cont-2ply + tune | SPRT running | Early, most divergent tune |
| SE ply gate + tune | SPRT running | Early |
| Histprune no-improving + tune | SPRT running | Calmest tune, least likely |
| Futility-full + tune | SPRT running | Flat at -0.7 |

## Remaining Ideas (prioritised)

### High Priority
1. **Good/bad quiet split** — implemented, ready to test. SF pattern: defer quiets with score <= -14000 after bad captures. Avg cutoff position improved 5.2%. Only SF has this.

### Medium Priority
2. **Killer/counter-move removal** — SF removed both, relies on history alone. With our history 27× stronger + 4D threat-aware, killers may be dead weight. Simplification + retune candidate.
3. **Low-ply history bonus** (SF: 8× at ply 0, decaying) — cheap, directly improves root ordering.

### Low Priority
4. **Pawn history initialisation** (PlentyChess: -1000 pessimistic prior) — untested.
5. **Capture scoring MVV×16 → lower** — with working captHist, ratio may need adjustment. But capture ordering has failed to improve 5× already.

## LMP Interaction (resolved)

LMP_BASE was pushed to 9 by SPSA (nearly disabling LMP). Two fixes resolved this:
1. **History magnitude fix** — working history means late moves are genuinely bad. LMP BASE dropped to 5.
2. **Retune methodology** — LMP BASE now converges to 6-8 depending on the feature branch, which is healthy.

## Key Lessons

1. **History magnitude is everything** — 27× fix improved all move ordering metrics and cascaded through every pruning feature.
2. **Retune-on-branch discovers hidden value** — features that change tree shape need parameter recalibration to show their Elo. 4/4 retune candidates turned from H0 to positive.
3. **Only SF has good/bad quiet split** — this is not a consensus feature. Test but manage expectations.
4. **Move ordering and pruning are coupled** — better ordering enables tighter pruning (LMP, futility), which enables deeper search, which enables better ordering. Virtuous cycle.
