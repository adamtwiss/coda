# Search Feature Review Findings (2026-04-02 to 2026-04-04)

Systematic comparison of Coda's search implementation against top engines
(Stockfish, Viridithas, Obsidian, Alexandria, Berserk).

## Retry After SPSA Round 2

Ideas that failed on the old parameter baseline but may work after SPSA retuning:

### High Priority Retry
- **SE ply gate** (ply < 2*rootDepth) — pure safety, consensus. Bench unchanged on old main.
- **SE negative extensions** (-3 when ttScore>=beta, -2 at cutNode, 0 otherwise) — H0 at -10 but tested pre-SPSA.
- **Dynamic SEE threshold for captures** (-score/32) — H0 at -27 but consensus across all engines.
- **Node-based time management** — concept proven, implementation failed twice. Needs better params.
- **Correction history proportional gravity** — every engine uses it, we have linear. Untested on new base.
- **Correction history no depth gate** — no engine gates by depth. Untested on new base.
- **Futility reimplementation** (lmr_depth gate, history in depth, bestScore update) — H0 pre-SPSA.

### Medium Priority Retry  
- **LMR remove failing heuristic** — H0 at -4.1 but unique to Coda. Revisit after SPSA.
- **LMR remove alpha_raised_count** — flat/negative. SPSA may have already compensated.
- **LMR remove unstable** — H0 at -3.0. Correction history serves same purpose.
- **NMP no dampening** — every engine returns raw score. H0 at -1.9 pre-SPSA.
- **NMP capture R flip** — H0 at -7. May interact with SPSA'd NMP params.
- **History pruning full reimplementation** — H0. Needs search stack moved_piece first.
- **Futility TT/killer exemptions** — H0 at -2.3. Correctness but hurts.
- **SMP copy history to helpers** — can't test single-threaded, merge if SMP testing available.

### Skip (conclusively wrong for our engine)
- LMR nonpawn side fix (H0 -3.0)
- Hindsight sign fix (H0 -10, reverted)  
- SE bundled fix (H0 -29)
- Cutnode LMR +2 (H0 -15.7)
- Asp depth clamp to depth-2 (H0 -4.3)

## Feature Analysis Summaries

### NMP (ablation: +18, expected +100)
- **depth-r fix merged (+11)**: was depth-1-r, every engine uses depth-r
- **Depth gate >= 3 merged (+0.7)**: was >= 4, consensus >= 2-3  
- **SPSA moved**: EVAL_DIV 200→164, BASE_R 4→3, DEPTH_DIV 3→4, VERIFY 12→13
- Still unique: dampening (score*2+beta)/3, capture R-=1. Consider removing post-SPSA r2.

### LMR (ablation: +262)
- C=1.30 is most aggressive of any engine (others 2.1-3.1)
- 5 unique adjustments: failing, alpha_raised, unstable, opp_non_pawn, enemy_pawn_escape
- Missing: TTPV flag (5/5 consensus), multi-ply cont hist in stat score, killer/counter reduction
- SPSA moved LMR_HIST_DIV 5000→5489 (less history influence)
- C value now SPSA-tunable for round 2

### Futility (ablation: +6, expected +15-30)
- Missing: history adjustment to lmr_depth, lmr_depth as sole gate, bestScore update, skip_quiets
- Our depth<=8 raw gate is unique — every engine uses lmr_depth only
- Margin 90+100*d is tighter than SF (42+120*d) but SPSA moved to 103+109*d

### SE (ablation: +16, expected +30-50)
- No ply gate (every engine has one)
- No double/triple extensions
- Depth gate 8 (consensus 5-6, SPSA moved to 9)
- Flat -1 negative ext (consensus: -3/-2/0 by condition)

### History Pruning (ablation: -17 to disable)
- Missing ply-2 cont hist (but piece lookup is buggy — needs search stack fix)
- !improving and !unstable guards unique (no engine has these)
- Threshold 1500 well-calibrated (SPSA moved to 1558)
- Depth 3 → SPSA moved to 2

### Move Ordering
- Capture scoring matches Obsidian/Alexandria (victim*16 + captHist)
- Missing: dynamic SEE threshold for good/bad split (-score/32)
- Missing: quiet check bonus (SF +16384, Viridithas +10000) — tested before at -11 Elo
- Cont hist weights 3x/3x/1x/1x unusual (consensus: equal weights, control via update)

### QS (ablation: +111)
- TT stand-pat refinement merged (+6.5) — was missing, every engine has it
- Missing: two-stage futility (eval+margin then SEE verify)
- Missing: move count cutoff (Obsidian: break after 3)
- Missing: negative SEE threshold (Obsidian -32, Viridithas -141)

### Time Management
- Node-based TM implemented but failed (-10, -31). Needs better params/implementation.
- Score stability only looks 1 depth back (SF uses 4-iter rolling avg)
- Missing forced move detection
- BM stability too conservative (should extend more on change)
- Movestogo fix merged (correctness)

### Aspiration Windows
- Delta growth x1.5 fastest of any engine (consensus x1.33)
- Fail-low beta contraction unique (3:5 weighted, consensus: midpoint)
- Fail-high alpha contraction rare (only us and SF)
- No depth reset on fail-low (Alexandria/Berserk reset)
- Always reduce depth on fail-high (should skip for winning scores)

### Correction History
- Linear gravity (every engine uses proportional)
- Entry clamp 32000 (every engine uses 1024)
- 1-ply continuation (consensus: 2-ply)
- Depth >= 3 gate (no engine has this)
- 6 sources is fine (Viridithas also has 5+)

### Lazy SMP
- Helpers discard results (3/4 engines use voting)
- Helpers start with zeroed history (Alexandria copies from main)
- thread_id % 2 depth diversity (no engine does explicit offsets)

## Infrastructure Built
- OpenBench at ob.atwiss.com (9 machines, ~126 threads)
- SPSA tuning via OpenBench (32 tunable parameters)
- 16 parameters tuned in round 1 (+31.48 Elo), 26 in round 2 (+25 Elo), round 4 in progress
- ob_status.py, ob_submit.py, ob_stop.py scripts
- Runtime PEXT detection (+20% NPS AMD)
- clap argument parsing

## Current Status (2026-04-05)

### Merged
| Change | Elo | Notes |
|--------|-----|-------|
| SEE quiet pruning fix (pre-MakeMove, lmrDepth², use see_ge) | +11.4 | Structural fix, 3 issues found |
| Correction history proportional gravity | +3.9 | Consensus |
| Futility pruning fix (pre-MakeMove, lmrDepth gate) | +2.6 | Correctness |
| Correction history no depth gate | ~0 | Consensus, 1-line |
| NMP no dampening | ~0 | Consensus, simplification |
| SMP copy history to helpers | +5.9 | Non-regression confirmed |
| Pondering fix (ponder move output, TM skip, ponderhit reset) | — | Functional fix |
| Embedded net fix (UCI mode wasn't using fat binary net) | — | Critical bug |

### Rejected (H0)
| Change | Elo | Notes |
|--------|-----|-------|
| SE ply gate | -4.6 | 4566 games, doesn't suit our engine |
| SE negative extensions | -10.5 | |
| Dynamic SEE threshold for captures | -26.7 | |
| Asp delta x1.33 | -6.7 | Our x1.5 works better |
| Asp skip winning depth reduction | -8.0 | |
| Asp fail-low midpoint | +0.7 | Flat, 4216 games |
| Asp fail-high alpha contraction | +0.8 | Flat, 3006 games |
| LMR remove alpha_raised | -0.2 | |
| LMR remove unstable | -3.0 | |
| LMR remove failing | -4.1 | |
| History pruning full reimplementation | -6.7 | |
| History pruning cont2 stack | -8.4 | Threshold wrong for 3 signals |
| Futility TT/killer exemptions | -2.3 | |
| NMP capture R flip | -7.1 | |
| Node-based TM v1/v2 | -31/-10 | Needs major rework |

### Not Yet Tried (prioritized)

**High priority:**
1. Correction history entry clamp 32000 → 1024 (every engine uses 1024)
2. Correction history 2-ply continuation (consensus, search stack now available)
3. LMR TTPV flag (5/5 consensus, reduce less for TT PV moves)
4. LMR multi-ply cont hist in stat score (consensus)
5. QS two-stage futility (eval+margin then SEE verify)

**Medium priority:**
6. QS move count cutoff (Obsidian: break after 3)
7. QS negative SEE threshold (Obsidian -32, Viridithas -141)
8. TM score stability rolling average (SF uses 4-iter)
9. TM forced move detection
10. Cont hist weights (our 3x/3x/1x/1x unusual, consensus equal)

**Low priority / risky:**
11. SE double/triple extensions (SE ply gate H0'd, SE may not suit us)
12. LMR killer/counter reduction (our unique adjustments cover similar ground)
13. Lazy SMP voting (needs multi-threaded testing)
14. Node-based TM (concept good, needs major rework)
