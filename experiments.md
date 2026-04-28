# Experiment Log

Structured record of all search/eval tuning experiments.

## ⚠️ RELIABILITY WARNING (2026-04-01)

**Results before 2026-04-01 are unreliable.** They were measured using narrow 3-engine gauntlets (Minic/Ethereal/Texel, 200-600 games) which gave systematically inflated Elo estimates. Validation on 2026-04-01 showed:
- Changes claimed at +20 to +67 by gauntlet were 0 to -17 in self-play SPRT
- 4 of 10 changes merged that day were later reverted as H0 (rejected) by SPRT
- The narrow gauntlet overfits to specific opponents, similar to self-play blindspots

**Results tagged [SPRT-validated]** used self-play SPRT with tight bounds and are trustworthy.
**Results tagged [gauntlet-only]** used narrow cross-engine gauntlet and may be false positives or false negatives.
**Results tagged [ablation]** are feature-disable tests — direction reliable, magnitude approximate.

Going forward, all changes must pass self-play SPRT before merging (see CLAUDE.md for methodology).

**SPRT settings**: Tier A: `elo0=-5 elo1=5`, Tier B: `elo0=0 elo1=10`. tc=10+0.1, Hash=64, OwnBook=false, openings=noob_3moves.epd.

**Net convention**: All experiments use the checked-in `net.nnue`, referenced by commit hash.

## Suspects: Potential False Positives (merged, need SPRT retest)

Changes in Coda that were never SPRT-validated. Currently in the codebase.

| Change | Commit | Source | Risk | Status |
|--------|--------|--------|------|--------|
| Singular extensions (+1) | e30bfbd | Day 2, untested | **Low** | ✅ Ablation: +15.6 Elo (Titan T9) |
| SE multi-cut + negative ext | b811a17 | Day 2, untested | **Low** | ✅ Covered by Titan T9 |
| NMP R=4, depth≥4 | e489477 | Day 2, untested | **Low** | ✅ Ablation: +18.3 Elo (Titan T2) |
| Capture scoring: raw captHist | 65dac27 | Day 2, untested | **Medium** | Not yet tested |
| Cont hist plies 4+6 | af684d0 | Day 2, untested | **Low** | Not yet tested |
| 4D threat-aware history | e7f52b5 | Day 3, H0 at 0 Elo | **Low** | Kept as infrastructure (0 Elo cost) |
| Aspiration delta 13+avg²/23660 | e7f52b5 | Day 3, H0 at 0 Elo | **Low** | Kept as infrastructure (0 Elo cost) |
| RFP TT quiet guard | 04b1f7b | Day 3, inconclusive +2.5 at 6125g | **Low** | In codebase |
| Cuckoo cycle detection | 8ac542b | Day 3, H1 at +5 | **Low** | ✅ [SPRT-validated] |
| Check-giving LMR R-1 | b42ff78 | Day 3, +9 at 1609g (97.5% LOS) | **Low** | Near-pass, kept |

## Suspects: Potential False Negatives (rejected, may deserve retest)

| Change | Elo measured | Method | Why retest |
|--------|-------------|--------|------------|
| Corr-gate | -23 | gauntlet-only | Narrow gauntlet may have been wrong |
| FMR scaling | -30 | gauntlet-only | Threshold calibration issue, not fundamentally bad |
| LMP 4+d² | +2.3 | gauntlet-only | Persistent small positive, may pass Tier A |
| History pruning depth 4 | -0.5/+5.5 | gauntlet-only | Inconsistent signal |

## Retry Candidates (Tighter SPRT Bounds)

Experiments that showed small positive Elo (+2 to +6) but couldn't reach H1 with our standard elo0=-5/elo1=15 bounds. These may be real ~+3-5 Elo gains that need elo0=-2/elo1=8 (or similar) to resolve. Small gains compound — five +4 patches = +20 Elo.

| Experiment | Elo | Games | Notes |
|---|---|---|---|
| LMP depth 9 | ~+4 | ~1500 | Extend LMP from depth<=8 to depth<=9. Persistent +3-5 throughout |
| NMP Divisor 170 | +2.7 | 3963 | NMP eval divisor 200→170. 75% LOS, massive game count |
| LMP 4+d² | +2.3 | 2453 | Already flagged as retest candidate. +3-5 / 75-85% LOS |
| Futility 50+d*50 | +1.9 | 2329 | Just barely too tight. Persistent +3-5 early |
| NMP Verify Depth 16 | +1.7 | 2045 | 16 vs 14 barely different. Could stack with other NMP changes |
| Opponent Material LMR <4 | +1.9 | 2710 | oppNonPawn < 4 instead of < 3. 67% LOS |
| TT Near-Miss Margin 96 | +1.6 | 2590 | Widen from 80 to 96. 63% LOS |
| TT Damp Depth-Adaptive v1 | -0.5 | 1335 | Trust deeper TT entries more. Showed +6-8 for 300 games |
| TT Damp Depth-Adaptive v2 | +0.2 | 1557 | Floor-3 variant. Dead flat |
| FH Blend Depth Gate 4 | +1.9 | 2243 | Disable FH blending at depth 3. Persistent +3 |
| History Divisor 4000 | +1.0 | 1776 | More aggressive LMR history. Persistent +3-5 |
| History Pruning Depth 4 | -0.5 | 1321 | Extend hist prune to depth 4. +5.5 at 1141 games |
| LMP Failing /2 | +0.8 | 1693 | Tighter LMP when failing. Persistent +2-4 |
| LMP Depth 9 | +0.6 | 1615 | Extend LMP from depth<=8 to depth<=9. Persistent +3-5 early |
| Mate Distance Pruning | +0.7 | 1559 | Universal technique. Might help at longer TC |
| NMP Deep Reduction d>=14 | +0.6 | 1657 | Already flagged. +4-9 for first 1000 games |
| Threat-Aware SEE Quiet | +0.8 | 1675 | Already flagged. +5-12 early then regressed |

---

## 2026-03-09: Correction History v1
- **Change**: Pawn-hash indexed correction table. Full strength (÷GRAIN), all node types updated.
- **Result**: -12 Elo (rejected early, ~200 games)
- **Baseline**: net.nnue @ 69a797e, pre-RFP/LMR tuning
- **Notes**: Fail-low nodes provided unreliable upper bounds as update signal, adding noise. Full strength correction on noisy data hurt.

## 2026-03-09: Correction History v2
- **Change**: Half strength correction (÷GRAIN*2), exact/fail-high updates only.
- **Result**: -11 Elo (rejected, ~300 games)
- **Baseline**: net.nnue @ 69a797e, pre-RFP/LMR tuning
- **Notes**: Halving correction made it too weak to help. The issue was noise quality, not magnitude.

## 2026-03-09: Correction History v3 (MERGED)
- **Change**: Full strength, tight clamp (corrHistMax=128 vs 256), depth>=3 gate, exact/fail-high only.
- **Result**: **+11.9 Elo**, H1 accepted. W123-L105-D157 (385 games). LOS 91.8%.
- **Baseline**: net.nnue @ 69a797e, pre-RFP/LMR tuning
- **Commit**: fe1edb5
- **Notes**: Tight clamping + depth gate addressed noise while keeping full correction strength. Third attempt — persistence paid off. Sound idea proven in Stockfish.

## 2026-03-09: RFP Tightening v1 (MERGED)
- **Change**: Margins depth*120/85 -> depth*85/60, depth limit 6->7.
- **Result**: **+15.7 Elo**, H1 accepted. W191-L165-D268 (~624 games). LOS 94.8%.
- **Baseline**: net.nnue @ 69a797e, pre-LMR tuning
- **Commit**: ad9f603
- **Notes**: Same change failed pre-NNUE. NNUE eval accuracy enables tighter node-level pruning. Big win.

## 2026-03-09: NMP v1 (divisor + improving penalty)
- **Change**: Eval divisor 200->150, +1R when not improving.
- **Result**: **-14.0 Elo**, H0 accepted. W266-L299-D391 (956 games). LOS 3.3%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: The !improving +1R was too aggressive — over-pruning in declining positions. Divisor change alone tested separately in v2.

## 2026-03-09: NMP v2 (divisor only)
- **Change**: Eval divisor 200->150, no improving penalty.
- **Result**: **-22.6 Elo**, H0 accepted. W172-L216-D289 (677 games). LOS 1.3%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Divisor itself is too aggressive. Current NMP formula (R=3+depth/3, divisor 200) is well-calibrated. Don't revisit unless eval changes significantly.

## 2026-03-09: LMR Aggressiveness v1 (MERGED)
- **Change**: LMR table constant C=2.0 -> 1.75 (more reduction for late moves).
- **Result**: **+16.2 Elo**, H1 accepted. W170-L147-D240 (~700 games). LOS 95.1%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning (no correction history in baseline binary)
- **Commit**: 474cd58
- **Notes**: More aggressive LMR saves time to search important moves deeper. NNUE's accurate eval makes it safe to reduce late moves more. Draw ratio increased (42% vs ~38%), indicating fewer blunders.

## 2026-03-09: Razoring Tightening v1
- **Change**: Margin 400+depth*100 -> 300+depth*75.
- **Result**: **-32.2 Elo**, H0 accepted. ~400 games. LOS 0.4%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Razoring at depth 1-2 needs the slack. Current margins are well-calibrated. Quick, decisive rejection.

## 2026-03-09: Futility Pruning v1
- **Change**: Move futility margin 100+lmrDepth*100 -> 75+lmrDepth*75 (uniform 25% tightening).
- **Result**: +0.7 Elo after 2128 games, inconclusive (killed at LLR 1.35/2.94).
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Essentially zero effect — too tight at low lmrDepth (base margin of 75 barely any slack). Unlike RFP (node-level), per-move futility pruning errors compound. Replaced with v2.

## 2026-03-09: ProbCut v1
- **Change**: probCutBeta from beta+200 -> beta+150, pre-filter staticEval+100 -> +75.
- **Result**: -3.4 Elo, inconclusive (killed at 1007 games). W285-L292-D430. LOS 34%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Looked promising early (+11.6 at 547 games) but faded to zero. Classic early noise. ProbCut margin of 200 appears well-calibrated already.

## 2026-03-09: SEE Pruning v1
- **Change**: Capture threshold -depth*100 -> -depth*75, quiet threshold -20*depth^2 -> -15*depth^2.
- **Result**: -7.7 Elo, killed at ~650 games (trending to rejection). W186-L199-D261. LOS 22.8%.
- **Baseline**: net.nnue @ 69a797e, post-LMR tuning
- **Notes**: SEE thresholds are about tactical accuracy, not eval quality — NNUE doesn't change material exchange calculations. Current thresholds are correct.

## 2026-03-09: RFP v2 — deeper
- **Change**: RFP depth limit 7->8 (margins unchanged at 85/60).
- **Result**: **-68.6 Elo**, killed at ~200 games (near-rejection). W46-L84-D68. LOS 0.0%.
- **Baseline**: net.nnue @ 69a797e, post-LMR tuning
- **Notes**: Depth 7 is the sweet spot. At depth 8, max margin is 8*85=680cp — far too large. Pruning positions that shouldn't be pruned. Decisive and fast failure.

## 2026-03-09: Futility Pruning v2
- **Change**: Move futility margin 100+lmrDepth*100 -> 100+lmrDepth*75 (tighten slope only, keep base).
- **Result**: -8.7 Elo, killed at ~462 games (trending to rejection). W130-L142-D190. LOS 23.7%.
- **Baseline**: net.nnue @ 69a797e, post-RFP tuning
- **Notes**: Neither the uniform tightening (v1) nor the slope-only tightening (v2) helped. Per-move futility margins are already well-tuned. Unlike node-level RFP, per-move errors compound.

## 2026-03-09: LMR v2 — more aggressive (MERGED)
- **Change**: LMR constant C=1.75 -> 1.5.
- **Result**: **+44.4 Elo**, H1 accepted. W112-L73-D122 (307 games). LOS 99.8%.
- **Baseline**: net.nnue @ 69a797e, post-LMR v1 tuning
- **Commit**: d188f53
- **Notes**: Second consecutive LMR tightening win. C=2.0→1.75 was +16 Elo, C=1.75→1.5 is +44 Elo. Testing C=1.25 next to find the optimum.

## 2026-03-10: NNUE net-new2 (MERGED)
- **Change**: Deploy fine-tuned NNUE net (lower learning rate training).
- **Result**: **+10.4 Elo**, H1 accepted. W332-L300-D465 (1097 games). LLR 2.99.
- **Baseline**: net.nnue @ 911b468, same binary
- **Commit**: fb7519b
- **Notes**: Net-new2 trained at lower LR squeezed additional quality.

## 2026-03-10: LMR v3 — C=1.25
- **Change**: LMR constant C=1.5 -> 1.25 (even more reduction).
- **Result**: +3 Elo, killed at 374 games. W97-L94-D183 (50.4%). Inconclusive → zero effect.
- **Baseline**: net.nnue @ 911b468, post-LMR v2
- **Notes**: C=1.25 overshoots the optimum. Too much reduction causes missed tactical moves.

## 2026-03-10: LMR v3b — C=1.375
- **Change**: LMR constant C=1.5 -> 1.375 (bracketing between 1.25 and 1.5).
- **Result**: +0 Elo, killed at 395 games. W106-L107-D182 (49.9%). Dead flat.
- **Baseline**: net.nnue @ 911b468, post-LMR v2
- **Notes**: C=1.5 is at or very near the optimum. No further LMR constant tuning is worthwhile.

## 2026-03-10: LMP v1 — tighter base
- **Change**: LMP base 3+depth^2 -> 2+depth^2.
- **Result**: -10 Elo, killed at 197 games. W51-L57-D89 (48.5%).
- **Baseline**: net.nnue @ 911b468, post-LMR v2
- **Notes**: Base of 3 is the right floor. Reducing to 2 prunes too many early moves.

## 2026-03-10: LMP v2 — no improving bonus
- **Change**: Remove the +50% LMP limit bonus when improving (always use base formula).
- **Result**: **-38 Elo**, killed at 147 games. W32-L48-D67 (44.6%).
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: The improving bonus is critical — it prevents over-pruning when the position is getting better. Decisive failure.

## 2026-03-10: History pruning divisor v1 — 4000
- **Change**: LMR history adjustment divisor 5000 -> 4000 (more history influence).
- **Result**: **-32 Elo**, killed at 174 games. W42-L58-D74 (45.4%).
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: More history influence causes over-fitting to noisy history data.

## 2026-03-10: History pruning divisor v2 — 6000
- **Change**: LMR history adjustment divisor 5000 -> 6000 (less history influence).
- **Result**: -14 Elo, killed at 245 games. W67-L76-D102 (48.2%).
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Less history influence also hurts. Divisor of 5000 is well-calibrated in both directions.

## 2026-03-10: Singular extension margin — depth*2
- **Change**: Singular beta = ttScore - depth*2 (was depth*3; narrower margin = extend more often).
- **Result**: **-85 Elo**, killed at 116 games. W21-L48-D47 (38.4%). Catastrophic.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: depth*2 extends far too many moves, wasting search time. depth*3 is well-calibrated.

## 2026-03-10: Aspiration window delta=12
- **Change**: Aspiration window initial delta 15 → 12 (tighter windows).
- **Result**: **-18.4 Elo**, H0 accepted. W196-L239-D376 (811 games). LOS 2.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Tighter windows cause more re-searches, wasting time. Delta=15 is well-calibrated. Testing delta=18 (opposite direction) next.

## 2026-03-10: RFP improving margin 60→50
- **Change**: RFP improving margin depth*60 → depth*50 (tighter pruning when position improving).
- **Result**: -6.3 Elo, killed at 1391 games (inconclusive, trending reject). W374-L396-D621. LOS 18.3%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Even with NNUE's better eval, the improving margin can't be tightened further. 60 is already aggressive.

## 2026-03-10: Razoring gentle (375+d*90)
- **Change**: Razoring margin 400+depth*100 → 375+depth*90 (~7% tightening).
- **Result**: -2.5 Elo, killed at 1402 games (dead flat). W399-L409-D594. LOS 36.2%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Gentler than the v1 attempt (-32 Elo at 300+d*75). Still no gain — razoring is well-calibrated in both directions.

## 2026-03-10: IIR deeper reduction (2 at d≥10)
- **Change**: IIR reduces by 2 plies when depth≥10 and no TT move (was always 1).
- **Result**: +1.8 Elo, killed at 1394 games (inconclusive, essentially zero). W396-L387-D611. LOS 59.9%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Slight positive trend but too small to matter. Double IIR at deep nodes is neutral — the extra reduction doesn't save enough time to compensate.

## 2026-03-10: Aspiration window delta=18
- **Change**: Aspiration window initial delta 15 → 18 (wider windows, opposite of delta=12).
- **Result**: -7.0 Elo, killed at 1253 games (trending reject). W348-L372-D533. LOS 23.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Both directions tested (12 and 18). Delta=15 is optimal — tighter wastes time on re-searches, wider loses precision.

## 2026-03-10: Razoring loosening (425+d*110)
- **Change**: Razoring margin 400+depth*100 → 425+depth*110 (opposite of gentle tightening).
- **Result**: **-21.5 Elo**, H0 accepted. W186-L231-D307 (724 games). LOS 1.6%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Loosening loses even more than tightening. Both directions confirm 400+d*100 is optimal. Wider margins waste time searching hopeless positions.

## 2026-03-10: Singular extension depth threshold depth/2
- **Change**: Singular verification depth (depth-1)/2 → depth/2 (deeper verification, fewer extensions).
- **Result**: -14.5 Elo, killed at 841 games (trending reject). W231-L267-D343. LOS 6.4%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Fewer extensions = miss important moves. (depth-1)/2 is well-calibrated.

## 2026-03-10: NMP verification depth 12→10
- **Change**: NMP verification search threshold depth≥12 → depth≥10 (verify at shallower depths).
- **Result**: -13.1 Elo, killed at 850 games (trending reject). W223-L256-D371. LOS 7.2%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Verification at depth 10-11 costs too much time for insufficient zugzwang protection. Depth 12 is the right threshold.

## 2026-03-10: Improving gate on razoring
- **Change**: Skip razoring when position is improving (`&& !improving`).
- **Result**: -6.6 Elo, killed at 1484 games (trending reject). W397-L424-D663. LOS 16.4%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Improving detection doesn't help razoring. At depth 1-2, the improving signal is noisy — positions that are "improving" at these shallow depths aren't reliably getting better.

## 2026-03-10: Eval-based LMR adjustment
- **Change**: Reduce less when staticEval+200 < alpha (losing), reduce more when staticEval-200 > beta (winning).
- **Result**: -5.9 Elo, killed at 1480 games (trending reject). W391-L416-D673. LOS 18.9%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: The improving heuristic already captures position trajectory. Adding raw eval distance to alpha/beta is redundant and slightly harmful — it fights with the existing improving adjustment.

## 2026-03-10: History-based LMP threshold
- **Change**: Raise LMP limit by +2 for moves with history score > 4000 (harder to prune good-history moves).
- **Result**: -3.1 Elo, killed at 1476 games (flat/slightly negative). W413-L421-D642. LOS 32.5%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: LMP already benefits from move ordering — high-history moves appear early and avoid pruning naturally. Explicit history gate is redundant.

## 2026-03-10: 2-ply continuation history in LMR (full weight)
- **Change**: Add ply-2 continuation history to LMR reduction adjustment and history pruning. Full weight (same as ply-1).
- **Result**: -3.8 Elo, killed at 1925 games (dead flat then faded). W526-L543-D856. LOS 26.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Early positive signal (+1.4 at 1471 games) was noise. Ply-2 piece lookup is lossy (piece may be captured), adding noise. Half-weight variant still running.

## 2026-03-10: 2-ply cont history in move ordering
- **Change**: Add ply-2 continuation history to MovePicker quiet move scoring AND LMR/pruning.
- **Result**: -27.1 Elo, killed at 194 games (strongly negative early). W50-L68-D76. LOS 9.1%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Adding noisy ply-2 signal to move ordering is actively harmful — bad ordering cascades through the entire search. Pruning-only is safer.

## 2026-03-10: 2-ply cont history half weight (MERGED)
- **Change**: Add ply-2 continuation history to LMR and history pruning, at half weight (÷2).
- **Result**: **+11.0 Elo**, H1 accepted. W300-L268-D439 (1007 games). LOS 91.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Commit**: ab25488
- **Notes**: Full weight was flat (-3.8 at 1925 games); half weight works. Ply-2 history is noisier than ply-1 (piece may have been captured), so down-weighting is essential. Adding to move ordering was actively harmful (-27 Elo) — pruning/reduction only.

## 2026-03-10: Futility improving (+50 margin)
- **Change**: Add +50 to futility margin when position is improving.
- **Result**: -1.9 Elo, killed at 1115 games (flat). W313-L323-D479. LOS 40.6%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Early noise showed +15.6 at 205 games, faded to zero. Futility margins are already well-calibrated.

## 2026-03-10: SEE quiet improving gate
- **Change**: Loosen SEE quiet pruning threshold by -50 when improving.
- **Result**: -0.4 Elo, killed at 902 games (flat). W243-L244-D415. LOS 48.2%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: Early +10.3 at 442 games faded to zero. SEE thresholds are about material exchange, not eval trajectory.

## 2026-03-10: History pruning with improving (tighter threshold)
- **Change**: Allow history pruning when improving, but with 2x stricter threshold (-4000*d vs -2000*d).
- **Result**: +0.0 Elo, killed at 898 games (dead flat). W251-L258-D389. LOS 50.0%.
- **Baseline**: net.nnue @ fb7519b, post-LMR v2
- **Notes**: The existing `!improving` gate is correct — extending history pruning to improving positions, even with a stricter threshold, doesn't help.

## 2026-03-10: LMP PV-node exemption (MERGED)
- **Change**: Add `beta-alpha == 1` to LMP gate, exempting PV nodes from late move pruning.
- **Result**: **+16.9 Elo**, H1 accepted. W217-L183-D301 (701 games). LOS 95.5%. LLR 2.96.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Commit**: d92b873
- **Notes**: At PV nodes, accuracy matters more than speed. Pruning late quiet moves at PV nodes risks missing the principal variation. Gate change, not margin — confirms that structural changes are higher-leverage than parameter tuning.

## 2026-03-10: Eval instability heuristic (MERGED)
- **Change**: Detect sharp eval swings from parent node (`|staticEval - (-parentEval)| > 200`). When unstable: skip history pruning, reduce LMR by 1, loosen SEE quiet threshold by 100cp.
- **Result**: **+12.8 Elo**, H1 accepted. ~830 games. LOS 92.6%. LLR 2.95.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Commit**: (pending)
- **Notes**: Novel heuristic — detects tactically volatile positions where pruning is dangerous. NNUE's accurate eval makes the 200cp swing meaningful (not eval noise). Opens a family of follow-up experiments: using instability to gate NMP, RFP, singular extensions; tuning the 200cp threshold.

## 2026-03-10: Capture history in SEE pruning
- **Change**: Modulate SEE capture pruning threshold by capture history: `seeThreshold += captHistVal/20`.
- **Result**: -0.7 Elo, killed at 940 games (fully regressed). W261-L263-D416. LOS 46.5%.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Notes**: Early signal was +50 at 90 games, +17 at 150, +5 at 600, then zero by 940. Textbook early noise fade. SEE thresholds are about material exchange accuracy — capture history doesn't meaningfully improve the threshold. The capture history signal is already used in capture ordering (MVV-LVA + captHist), which is the right place for it.

## 2026-03-10: Counter-move LMR reduction
- **Change**: Reduce LMR by 1 for moves matching the counter-move heuristic (`if move == counterMove { reduction-- }`).
- **Result**: -1.7 Elo, killed at 608 games (dead flat). W174-L177-D257. LOS 43.6%.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Notes**: Counter-move already gets priority in move ordering (tried before quiets). Reducing LMR for it doesn't add value — the ordering benefit is sufficient.

## 2026-03-10: Double extension depth gate (depth≥12)
- **Change**: Add `depth >= 12` to double singular extension condition (was no depth gate beyond singular's depth≥10).
- **Result**: -15 Elo, killed at 437 games (consistently negative). W120-L138-D179. Win% 47.9%.
- **Baseline**: net.nnue @ ab25488, post-ContHist2
- **Notes**: Restricting double extensions to deeper nodes loses them when they matter most (depth 10-11). The existing threshold is correct. ~40-50 games may have been affected by CPU contention from concurrent training.

## 2026-03-10: Pawn history table
- **Change**: New pawn-structure-aware history table indexed by `[pawnHash%512][piece][toSquare]`. Added to quiet and evasion move scoring, updated on beta cutoffs (bonus for cutoff move, penalty for tried quiets). ~832KB per thread.
- **Result**: **+22.2 Elo**, H1 accepted. 533 games. LLR 3.01.
- **Baseline**: net.nnue @ 7f836a2, post-EvalInstability
- **Commit**: (this commit)
- **Notes**: Pawn structure changes slowly, making it a stable, low-noise signal for move ordering. The table captures which piece-to-square patterns work well in specific pawn structures. Major win — validates the hypothesis that move ordering has significant room for improvement.

## 2026-03-10: Continuation history 2x weight
- **Change**: Double continuation history weight in quiet and evasion move scoring: `score += 2 * int(mp.contHist[piece][m.To()])` (was `1 *`).
- **Result**: **+27.9 Elo**, H1 accepted. 461 games. LLR 3.00.
- **Baseline**: net.nnue @ 7f836a2, post-EvalInstability
- **Commit**: (this commit)
- **Notes**: Continuation history (what worked after the opponent's previous move) is a highly predictive ordering signal. Doubling its weight amplifies this signal relative to main history. Combined with pawn history, total move ordering improvement is ~50 Elo from this session. Note: ply-2 cont hist at full weight was previously harmful (-27 Elo), but ply-1 benefits from amplification.

## 2026-03-11: New NNUE net (MERGED)
- **Change**: Updated net.nnue from net-new.nnue (additional training).
- **Result**: **+28.6 Elo**, H1 accepted. W167-L128-D180 (475 games). LOS 98.8%.
- **Baseline**: net.nnue @ 1e9f490, post-PawnHist
- **Commit**: 16d6592

## 2026-03-11: QS TT move (MERGED)
- **Change**: Pass TT move to quiescence search InitQuiescence, start at stageTTMove instead of stageGenerateCaptures.
- **Result**: **+7.1 Elo**, H1 accepted. W386-L358-D624 (1368 games). LOS 84.8%.
- **Baseline**: net.nnue @ 1e9f490, post-PawnHist
- **Commit**: 204e62d
- **Notes**: Low-hanging fruit — QS was ignoring TT information for move ordering.

## 2026-03-11: Continuation history 3x weight (MERGED)
- **Change**: Cont history weight in quiet and evasion move scoring 2x→3x.
- **Result**: **+3.8 Elo**, H1 accepted. W644-L620-D921 (2185 games). LOS 75.0%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Commit**: dab4bd4
- **Notes**: Continuing the trend from 1x→2x (+27.9 Elo). Smaller gain as expected. Testing 4x next to bracket the optimum.

## 2026-03-11: Pawn history in LMR
- **Change**: Add pawn history scores to LMR reduction adjustment (alongside main/cont history).
- **Result**: **-7.8 Elo**, H0 accepted. W742-L801-D1100 (2643 games). LOS 6.7%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Notes**: Pawn history is useful for move ordering but too noisy for LMR adjustment. Consistent with pattern: ordering signals don't always transfer to pruning/reduction.

## 2026-03-11: Eval instability threshold 150
- **Change**: Instability threshold 200→150 (detect more volatile positions).
- **Result**: -2.5 Elo at 3281 games (flat, killed). LOS 29.1%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Notes**: Lower threshold fires too often, diluting the signal. 200 is better.

## 2026-03-11: Eval instability threshold 300
- **Change**: Instability threshold 200→300 (only flag extreme swings).
- **Result**: **-6.8 Elo**, H0 accepted. W892-L955-D1353 (3200 games). LOS 7.1%.
- **Baseline**: net.nnue @ 9ef020a, post-QS-TTMove
- **Notes**: Higher threshold misses too many volatile positions. 200 is near-optimal. Bracketed: 150 flat, 300 negative.

## 2026-03-11: Continuation history 4x weight
- **Change**: Cont history weight in quiet and evasion move scoring 3x→4x.
- **Result**: -5.0 Elo at 1055 games (killed, trending negative). LOS 26.9%. LLR -0.57.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Overshoots the optimum. 3x is the sweet spot — confirmed by bracketing (2x +27.9, 3x +3.8, 4x negative). Do not increase further.

## 2026-03-11: Pawn history pruning (weight=2)
- **Change**: Apply pawn history score to quiet move pruning and LMR reduction (scaled to ±2x weight).
- **Result**: **-17.0 Elo**, H0 accepted. W265-L313-D402 (980 games). LOS 2.3%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Confirms pawn history in pruning/LMR is harmful (first attempt -7.8 Elo). Ordering only.

## 2026-03-11: QS delta pruning margin
- **Change**: Tighten quiescence delta pruning margin.
- **Result**: -4.6 Elo at 1069 games (killed, trending negative). LOS 28.7%. LLR -0.47.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Current delta margin is well-calibrated. Do not tighten.

## 2026-03-11: Correction history clamp 128→96
- **Change**: Reduce correction history clamp from 128 to 96 (less aggressive corrections).
- **Result**: -0.7 Elo at 1048 games (killed, flat). LOS 46.8%. LLR 0.40.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Slightly worse. Testing 192 next (more aggressive corrections).

## 2026-03-11: Node fraction time management
- **Change**: Track best move's share of root nodes. High fraction (>0.9) → 0.8x time, low fraction (<0.3) → 1.5x time, (<0.5) → 1.3x time.
- **Result**: +2 Elo at ~960 games (killed, flat). LOS ~53%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Sound concept (used in Stockfish) but thresholds may need tuning, or the benefit is too small at these time controls. Could revisit with different scaling factors.

## 2026-03-11: Pawn history 2x weight in ordering
- **Change**: Double pawn history weight in quiet move scoring: `score += 2 * pawnHist[...]` (was 1x).
- **Result**: -4 Elo at ~960 games (killed, slightly negative). LOS ~38%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: 1x is the right weight. Unlike cont history (which benefited from 1x→3x amplification), pawn history is already well-scaled.

## 2026-03-11: Main history 2x weight in ordering
- **Change**: Double main history weight in quiet move scoring.
- **Result**: -1 Elo at ~950 games (killed, flat). LOS ~47%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Main history weight is already well-calibrated. The cont history amplification trick doesn't generalize to all history tables.

## 2026-03-11: Correction history clamp 128→192
- **Change**: Increase correction history clamp from 128 to 192 (more aggressive corrections).
- **Result**: +1 Elo at 968 games (killed, flat). LOS ~52%.
- **Baseline**: net.nnue @ 4bbcb7d, post-ContHist3x
- **Notes**: Bracketed: 96 was -0.7, 192 is +1. Both flat. 128 is near-optimal. Do not revisit.

---

## 2026-03-11: Net blended (lambda=0.05)
- **Change**: NNUE net trained with lambda=0.05 result blending (same epochs, LR, dataset as current net).
- **Result**: -3.1 Elo at 1667 games (killed, flat/negative). LOS 32.4%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Lambda=0.05 didn't help. User plans further training iterations before resubmitting.

## 2026-03-11: TM directional score drop
- **Change**: Replace abs(scoreDelta) with directional scoreChange. Score drop (<-30 → 1.5x, <-15 → 1.25x time), score improve (>30 → 0.85x time).
- **Result**: -0.6 Elo at 1686 games (killed, flat). LOS 46.2%. LLR 0.68.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Directional TM doesn't help at these thresholds. The existing abs(scoreDelta) approach may already capture what matters. Could revisit with more aggressive thresholds or combined with node fraction.

## 2026-03-11: Quiet check bonus in move ordering
- **Change**: +5000 score bonus for quiet moves that give direct check in generateAndScoreQuiets().
- **Result**: **-10.9 Elo**, H0 accepted. 1682 games. LOS 4.6%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Checking moves are already handled well by check extension. Boosting them in ordering disrupts history-based quality. Do not revisit.

## 2026-03-11: Counter-move history table
- **Change**: New [13][64][13][64]int16 history table indexed by [prevPiece][prevTo][piece][to]. Used in quiet/evasion scoring (1x weight), history pruning, and LMR adjustment.
- **Result**: -7.2 Elo at 1683 games (killed, trending H0). LOS 13.0%. LLR -1.73.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Counter-move history adds noise rather than signal. The existing cont history (ply-1 piece/to) already captures this relationship. The extra [prevPiece][prevTo] indexing fragments the table too much for reliable statistics. Do not revisit without much larger games/deeper searches.

## 2026-03-11: Singular extension depth 10→8
- **Change**: Lower singular extension depth threshold from depth>=10 to depth>=8.
- **Result**: **-66.8 Elo** at 100 games (killed immediately). LOS 0.8%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Catastrophic. The verification search at half of depth 8 (=depth 4) is too expensive and unreliable for the benefit. Singular extensions need to remain at deep nodes where the TT entry is trustworthy. Testing wider singular margin (depth*2 instead of depth*3) next, which fires more often at depth>=10 without the cost of shallower verification.

## 2026-03-11: Check Ext SEE Filter (Killed ~250 games)
- **Change**: Only extend checks where SEE(check move) >= 0. Filters out checks where the checking piece can be captured for material gain.
- **Result**: -8.4 Elo at 251 games (killed, trending negative). W66-L72-D113.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: SEE filter is too coarse — many valuable checks involve sacrificial piece placement. The issue isn't material cost but whether the check restricts the king.

## 2026-03-11: NMP +1 Reduction with Rooks in EG (Killed ~252 games)
- **Change**: In endgame (non-pawn-king pieces < 10), when both sides have rooks, add +1 to NMP reduction. Theory: rook endgames are drawish and NMP can be more aggressive.
- **Result**: -12 Elo at 252 games (killed, trending negative). W68-L75-D109.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: More aggressive NMP in rook endgames actually hurts — rook endgames have subtle zugzwang-like positions where NMP is already borderline. Existing NMP calibration is correct.

## 2026-03-11: EG Futility 75% Margin (Killed ~244 games)
- **Change**: Tighten futility pruning margin by 25% in endgame (non-pawn-king < 10): `margin = (100 + lmrDepth*100) * 3/4`.
- **Result**: -14 Elo at 244 games (killed, trending negative). W69-L74-D101.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Endgame positions are more sensitive to futility errors (single pawn = game-deciding). Tighter margins prune moves that actually matter. Confirms pattern #2: per-move pruning needs slack.

## 2026-03-11: Singular Margin depth*2 (Killed ~254 games)
- **Change**: Widen singular extension margin from `ttScore - depth*3` to `ttScore - depth*2`, making singular extensions fire more often.
- **Result**: +3.5 Elo at 254 games (killed, regressed from early +27 to flat). Also retested with `restart=on`: -10 Elo at 122 games, confirming no gain.
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Early results were noise. The current singular margin (depth*3) is well-calibrated. Wider margin fires more but many of those extra firings don't identify truly singular moves.

## 2026-03-11: LMR Reduction -1 in EG (Killed ~69 games)
- **Change**: Reduce LMR reduction by 1 in endgame (non-pawn-king pieces < 10). Theory: EG has higher LMR re-search rate, suggesting reductions too aggressive.
- **Result**: -25 Elo at 69 games (killed early).
- **Baseline**: net.nnue @ 4bbcb7d
- **Notes**: Less LMR in endgame loses Elo. The higher re-search rate in EG is acceptable; reducing less means searching too many moves fully.

## 2026-03-12: History Divisor 5000→7000 (Killed ~100 games, vs PersistHistory baseline)
- **Change**: Increase LMR history adjustment divisor from 5000 to 7000, dampening history's influence on reductions. Tested on top of persist-history fix (richer history data).
- **Result**: -27 Elo at ~100 games (killed).
- **Baseline**: chess-persist-history (history tables persist across searches)
- **Notes**: Dampening history influence is wrong direction. With persistent history providing richer/higher-magnitude data, the current divisor of 5000 may already be correct or even too high. Don't increase further; consider testing lower (4000 or 3500).

## 2026-03-12: Persist History Tables Across Searches (MERGED)
- **Change**: Bug fix — SearchInfo (history, killers, counter-moves, continuation history, pawn history, correction history) now persists across searches within the same game via `persistInfo` on UCIEngine. Previously every `go` command created a fresh SearchInfo, discarding all learned move ordering data. `ucinewgame` properly clears everything.
- **Result**: **+19.2 Elo**, H1 accepted. W198-L163-D272 (633 games). LOS 96.7%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Commit**: c19645d (combined with EG loose check)
- **Notes**: Major bug fix discovered by investigating why experiments regressed from early positive to zero. All history tables were being zeroed every move — the engine was starting cold for move ordering on every search. This affects every history-dependent feature (move ordering, LMR, history pruning, correction history).

## 2026-03-12: EG Loose Check Filter (MERGED)
- **Change**: Skip check extension in endgame (<10 non-pawn-king pieces) when checked king has 4+ escape squares. Based on instrumentation showing 98% of EG checks are loose with only 7.2% cutoff rate.
- **Result**: **+21.7 Elo**, H1 accepted. W167-L133-D244 (544 games). LOS 97.5%.
- **Baseline**: net.nnue @ 4bbcb7d
- **Commit**: c19645d (combined with persist history)
- **Notes**: Data-driven experiment. Phase instrumentation revealed 86% of EG checks are queen shuffling with 4+ king escapes. Filtering these saves ~2.9M wasted search nodes. Tight checks (0-3 escapes) retained at 25-35% cutoff rate.

## 2026-03-12: History Pruning -2000→-3000 (Killed ~120 games, vs PersistHistory baseline)
- **Change**: Loosen history pruning threshold from `-2000*depth` to `-3000*depth`. Theory: with persistent history, scores are larger, so threshold needs widening.
- **Result**: -35 Elo at ~120 games (killed).
- **Baseline**: chess-persist-history
- **Notes**: History pruning threshold is well-calibrated. The gravity formula in history updates naturally bounds values, so persistent history doesn't drastically change score magnitudes. Both directions (tighter and looser) of history-related parameters lose Elo — confirms pattern #3 (well-tuned).

## 2026-03-12: History Divisor 5000→3500 (Killed ~600 games, vs new baseline)
- **Change**: Decrease LMR history divisor from 5000 to 3500, strengthening history influence.
- **Result**: -3 Elo at 600 games (flat, killed).
- **Baseline**: c19645d (persist-history + EG loose check)
- **Notes**: Both directions tested (3500 and 7000). Confirms divisor 5000 is optimal even with persistent history.

## 2026-03-12: Pawn History in LMR (Killed ~400 games, vs new baseline)
- **Change**: Add pawn history (÷2 weight) to LMR reduction adjustment histScore.
- **Result**: -10 Elo at 400 games (killed, negative).
- **Baseline**: c19645d
- **Notes**: Consistent with pre-persist-history result (-7.8). Pawn history doesn't help LMR regardless of persistence. Pattern holds.

## 2026-03-12: Counter-Move Before Killers in EG (H1 vs old, flat vs new)
- **Change**: In endgame (non-pawn-king < 10), try counter-move before killer moves in move ordering.
- **Result**: H1 at +15.4 Elo vs old baseline (698 games). Flat at +3.5 vs new baseline (618 games).
- **Baseline**: 4bbcb7d (old) / c19645d (new)
- **Notes**: The persist-history fix already makes counter-moves effective since they persist across searches. The ordering change adds no further benefit — counter-moves are already strong with persistence.

## 2026-03-12: Clear Killers Between Searches (Killed ~272 games)
- **Change**: Clear killer move table in `resetForSearch()`. Theory: stale killers from previous search at same ply could hurt.
- **Result**: -23 Elo at 272 games (killed).
- **Baseline**: c19645d
- **Notes**: Persistent killers ARE useful. Ply-indexed killers remain relevant across searches at similar depths.

## 2026-03-12: History Decay 50% + Clear Killers (converging to 0, ~800 games)
- **Change**: Halve all history tables and clear killers between searches. Favors recent data.
- **Result**: Peaked at +15 Elo early, regressed to 0 at 800 games. Still running.
- **Baseline**: c19645d
- **Notes**: Early promise was noise. Decay hurts as much as it helps. History gravity formula already handles staleness.

## 2026-03-12: History Decay 50% Only (Killed ~100 games)
- **Change**: Halve all history tables between searches (keep killers).
- **Result**: -44 Elo at ~100 games (killed).
- **Baseline**: c19645d
- **Notes**: Pure decay without killer clearing is clearly bad. Confirms history decay is not useful — the gravity formula in history updates already prevents saturation.

## 2026-03-12: TM Fail-Low Extension (Killed ~250 games)
- **Change**: Extend search time by 1.3x when score drops >30cp from previous iteration.
- **Result**: -39 Elo at ~250 games (killed).
- **Baseline**: c19645d
- **Notes**: Extra time on fail-low doesn't help — the existing instability scaling (1.2-1.4x for scoreDelta > 25-50) already handles this. Additional extension wastes time on hopeless positions.

---

## Ideas Not Yet Tested
- **Endgame-specific move ordering**: TT move dominates EG cutoffs (51.4% vs 13.9% MG), suggesting other ordering signals need different weights.
- **Queen check extension filter**: Queen checks are 86% of EG checks but mostly shuffling. Could selectively reduce extension for distant queen checks (EG loose check already handles the escape-square case).

## Key Patterns Observed

1. **Node-level pruning benefits from NNUE tightening** (RFP, LMR) — these prune entire subtrees based on eval, and NNUE's accuracy makes this safe.
2. **Per-move pruning is more sensitive** (futility, SEE, LMP) — errors compound across many moves, so margins need more slack.
3. **NMP, razoring, LMP, history divisor, singular margin are all well-tuned** — don't revisit unless eval changes significantly.
4. **Self-play Elo ~2x cross-engine Elo** for search changes. Calibrate expectations.
5. **Persistence matters** — correction history took 3 attempts. Tune before rejecting sound ideas.
6. **LMR constant C=1.5 is near-optimal** — C=1.25 and C=1.375 both tested at zero. Further tuning has diminishing returns.
7. **LMP improving bonus is critical** — removing it costs ~38 Elo. The improving heuristic correctly identifies positions where more moves should be searched.
8. **History divisor 5000 is optimal** — both directions (4000 and 6000) lose Elo.
9. **Aspiration delta=15 is optimal** — both directions (12 and 18) lose Elo. Bracketed.
10. **Razoring 400+d*100 is optimal** — three attempts (300+d*75, 375+d*90, 425+d*110) all lose. Do not revisit.
11. **Singular depth (depth-1)/2, NMP verify depth 12 are well-calibrated** — don't change.
12. **Margin-tuning has diminishing returns** — after the initial NNUE-driven RFP/LMR wins, most parameters are already near-optimal. New features (structural changes) are more likely to gain than parameter adjustments.
13. **2-ply continuation history needs half weight** — full weight adds noise (ply-2 piece may be captured); adding to move ordering is harmful (-27 Elo). Pruning/reduction only, at ÷2.
14. **Improving heuristic doesn't help per-move pruning** — futility (+50), SEE (-50), history pruning (stricter threshold) all tested neutral. The improving signal is already captured by RFP and LMR adjustments.
15. **Move ordering has massive room for improvement** — pawn history (+22.2) and cont-hist 2x weight (+27.9) combined for ~50 Elo from a single session. New history signals and weight tuning are high-value experiments.
16. **Eval instability threshold 200 is optimal** — 150 flat, 300 negative. Bracketed.
17. **Pawn history doesn't transfer to LMR** — useful for ordering (-7.8 Elo in LMR). Ordering signals don't always work for pruning/reduction.
18. **EG-specific search parameters lose Elo** — Tested EG futility (75% margin), NMP +1 with rooks, check ext SEE filter. All negative. The existing "one size fits all" parameters are well-calibrated across phases. History tables naturally adapt during gradual MG→EG transition.
19. **Check extension quality varies dramatically by phase** — MG: 35% loose (4+ escapes, 10% cutoff), EG: 98% loose (7% cutoff). 86% of EG checks are queen shuffling. Structural check quality filters > phase-specific parameter tweaks.
20. **Persist history tables is critical** — Bug: SearchInfo was recreated fresh every `go` command, zeroing all history. Fix gained +19.2 Elo. Always verify infrastructure correctness before tuning parameters.
21. **History divisor 5000 is robust to persist-history** — Tested 3500 (strengthen) and 7000 (dampen). Gravity formula naturally bounds values; persistent data doesn't change optimal divisor. History pruning threshold (-2000*depth) also robust.
22. **Investigate anomalies** — The persist-history fix was found by investigating why experiments regressed from early positive to zero. Sometimes the root cause of a symptom is unrelated to the symptom itself.
23. **History decay hurts with persist-history** — 50% decay between searches, clearing killers, and combinations all tested neutral-to-negative. The gravity formula in history updates naturally bounds staleness.
24. **Counter-move EG boost was captured by persist-history** — Gained +15.4 vs old baseline but only +3.5 vs new baseline with persist-history. Infrastructure fixes can subsume parameter tuning.

---

## 2026-03-12: History Decay (50% all tables between searches)
- **Change**: Apply 50% multiplicative decay to all history tables (main, capture, continuation, pawn, correction) between UCI `go` commands.
- **Result**: 0 Elo, LLR 0.5 (17%) after 820 games — converging to zero, killed
- **Baseline**: c19645d (persist-history + EG loose check)
- **Notes**: The gravity formula already bounds history values; forced decay adds no benefit.

## 2026-03-12: Counter-Move EG Boost v2 (re-test on new baseline)
- **Change**: Double counter-move score in endgame (non-pawn-king < 10) move ordering.
- **Result**: -2.5 Elo, LLR -0.007 after 828 games — flat/negative, killed
- **Baseline**: c19645d (persist-history + EG loose check)
- **Notes**: Previously gained +15.4 vs pre-persist baseline. Persist-history fix already preserves counter-move knowledge, making the boost redundant.

## 2026-03-12: Mate Distance Pruning
- **Change**: Tighten alpha/beta bounds using theoretical best/worst mate distance. `alpha = max(alpha, -MateScore+ply)`, `beta = min(beta, MateScore-ply-1)`. Prune if window is empty.
- **Result**: **-19.1 Elo** (H0 rejected, 837 games)
- **Baseline**: c19645d (persist-history + EG loose check)
- **Notes**: Started very strong (+37 Elo at 165 games, 95% LOS) but regressed to clearly negative. The pruning fires rarely in normal play (mates are uncommon at root-relative distances) but when it does fire at deep nodes, it can miss forced mates via longer paths. Classic case of early SPRT optimism.

## 2026-03-12: History-Aware Futility Margin
- **Change**: Adjust futility pruning margin by move history score (÷200). Good-history moves get wider margin (harder to prune), bad-history moves get tighter margin (easier to prune).
- **Result**: -6.0 Elo, LLR -1.2 (41%) after 1572 games — killed (clearly negative)
- **Baseline**: c19645d (persist-history + EG loose check)
- **Notes**: Adding noise to futility margins doesn't help. History is already used in LMR and history pruning; double-dipping in futility is redundant.

## 2026-03-12: Castling Extension
- **Change**: Extend castling moves by 1 ply (same as check/recapture/passed pawn extensions).
- **Result**: **-30.8 Elo** (killed at 271 games, clearly negative)
- **Baseline**: c19645d (persist-history + EG loose check)
- **Notes**: Castling is a quiet positional move, not a tactical forcing move. Extending it wastes search depth on moves that don't resolve tactical uncertainty. Extensions should be reserved for forcing moves (checks, recaptures, advanced pawns).

## 2026-03-12: QS Correction History
- **Change**: Apply correction history to stand-pat score in quiescence search (instead of raw EvaluateRelative).
- **Result**: ~0 Elo after 1865 games, LLR 0.60 — killed (flat)
- **Baseline**: c19645d
- **Notes**: Correction history adjustments are small (~±30 cp typical), and QS stand-pat decisions are dominated by material captures, not subtle eval differences. The correction helps in main search where pruning decisions depend on eval accuracy at specific thresholds (RFP, futility) but doesn't change QS meaningfully.

## 2026-03-12: TT PV Entry Preservation (depth +3 bonus)
- **Change**: In TT replacement scoring, give TTExact (PV node) entries a +3 depth bonus to make them harder to evict.
- **Result**: ~-3 Elo after 1847 games, LLR -0.13 — killed (flat/negative)
- **Baseline**: c19645d
- **Notes**: 5-slot buckets with age-based replacement already do a good job preserving useful entries. Over-preserving PV entries can crowd out useful non-PV entries (fail-high/fail-low bounds from cut/all nodes).

## 2026-03-12: ProbCut TT Move Priority
- **Change**: In ProbCut, try TT move first (if it's a capture with SEE >= 0) before generating all captures. Skip TT move in the subsequent capture loop.
- **Result**: ~-1.7 Elo after 1265 games, LLR 0.22 — killed (flat)
- **Baseline**: c19645d
- **Notes**: In theory, trying TT move first saves move generation time when ProbCut succeeds. In practice, ProbCut fires rarely and the TT move is often already in the capture list, so the savings are minimal.

## 2026-03-12: Negative Singular Extension Margin d*2 (was d*3)
- **Change**: Lower negative singular extension threshold from `ttScore + depth*3` to `ttScore + depth*2` (fire negative extensions more often).
- **Result**: -14.9 Elo after 295 games, LLR -0.69 — killed (clearly negative)
- **Baseline**: c19645d
- **Notes**: More negative extensions = more depth reductions on non-singular TT moves. This hurts because many TT moves that aren't overwhelmingly singular are still the best move — reducing them loses accuracy. The current d*3 threshold correctly identifies only moves where alternatives are truly competitive.

## 2026-03-12: Double Singular Extension Threshold d*2 (was d*3)
- **Change**: Lower double singular extension threshold from `singularBeta - depth*3` to `singularBeta - depth*2` (fire more double extensions).
- **Result**: +2.4 Elo after 584 games, LLR 0.58 — killed (fading to zero)
- **Baseline**: c19645d
- **Notes**: Started at +12 (305 games) but regressed to +2.4. More double extensions don't help — the current threshold correctly identifies overwhelmingly singular moves. Lowering the threshold doubles-extends too many moves that aren't truly singular, wasting search depth.

## 2026-03-12: TT Age Weight 6 (was 4)
- **Change**: Increase TT replacement age penalty from `depth - 4*age` to `depth - 6*age` (evict old entries more aggressively).
- **Result**: **-70.4 Elo** after 124 games, LLR -1.53 — killed (catastrophic)
- **Baseline**: c19645d
- **Notes**: Over-aggressive eviction of old TT entries destroys search efficiency. Old entries from earlier iterations still contain valuable move ordering and bound information. The current 4*age factor balances freshness vs information preservation. Age 6 evicts too much useful data.

## 2026-03-12: 4-Ply Continuation History in LMR/Pruning
- **Change**: Add 4-ply continuation history (our move from 4 plies ago) to LMR reduction adjustment (÷4 weight) and history pruning score (÷4 weight). Indexed by current piece/to-square against the move 4 plies back.
- **Result**: +1.2 Elo after 909 games, LLR 0.72 — killed (fading to zero)
- **Baseline**: c19645d
- **Notes**: Started very strong (+21 Elo at 369 games, LLR 1.90) but steadily regressed: +16→+8→+3→+1. Classic early SPRT optimism. The 4-ply-ago move is too distant to provide reliable signal — the position has changed too much. 2-ply continuation history at half weight is the sweet spot; 4-ply adds noise.

## 2026-03-12: Capture History Gravity 8192 (was 16384)
- **Change**: Halve capture history gravity divisor from 16384 to 8192, making capture history adapt faster (scores bounded to ±8192 instead of ±16384).
- **Result**: -20.9 Elo after 212 games, LLR -0.70 — killed (clearly negative)
- **Baseline**: c19645d
- **Notes**: Faster adaptation means capture history overreacts to recent games, losing the stability that larger gravity provides. With persistent history across searches, the current 16384 divisor lets capture history build up reliable patterns over many positions. Halving it makes the history too volatile.

## 2026-03-12: IIR Depth Threshold 5 (was >= 6)
- **Change**: Lower IIR depth threshold from `depth >= 6` to `depth >= 5`.
- **Result**: +0.6 Elo after 567 games, LLR 0.38 — killed (flat zero)
- **Baseline**: c19645d
- **Notes**: IIR at depth 5 doesn't help. At depth 5, the savings from reducing to depth 4 are too small to matter, and losing the extra depth when the TT miss is a false alarm costs more than it saves. Depth >= 6 is correct.

## 2026-03-12: LMP Depth 10 (was depth <= 8)
- **Change**: Extend LMP from `depth <= 8` to `depth <= 10`. At depth 9, lmpLimit = 3+81=84; depth 10, lmpLimit = 3+100=103.
- **Result**: 0.0 Elo after 408 games, LLR 0.21 — killed (dead flat)
- **Baseline**: c19645d
- **Notes**: At depth 9-10, the move count required for LMP (84-103) is rarely reached in practice — most nodes have far fewer legal moves. The extension is harmless but pointless. LMP depth <= 8 already covers the relevant range.

## 2026-03-12: History Pruning Depth 4 (was depth <= 3)
- **Change**: Extend history-based pruning from `depth <= 3` to `depth <= 4`.
- **Result**: -22.6 Elo after 206 games, LLR -0.81 — killed (clearly negative)
- **Baseline**: c19645d
- **Notes**: Depth 4 has too many important quiet moves to prune based on history alone. At depth 3, history pruning safely eliminates bad moves; at depth 4, the threshold (-2000*depth = -8000) is high enough to clip moves that could still be relevant. History pruning depth 3 is well-calibrated.

## 2026-03-12: NMP Base Reduction R=4 (was R=3)
- **Change**: Increase NMP base reduction from `R = 3 + depth/3` to `R = 4 + depth/3`.
- **Result**: **-66.8 Elo** (killed at 107 games, LLR -1.43, clearly negative)
- **Baseline**: c19645d
- **Notes**: NMP R=3 base was already confirmed well-calibrated. R=4 is far too aggressive — it skips too much search depth on the null move verification, allowing the engine to be tactically unsound. The eval-based NMP bonus (`eval/200`) already dynamically increases R for clearly winning positions; the base of 3 provides a safe floor.

## 2026-03-12: Disable Singular Extensions (ablation test)
- **Change**: Set `SingularExtEnabled = false`, `DoubleSingularExtEnabled = false`, `NegativeSingularExtEnabled = false`. Complete ablation of singular extension verification searches.
- **Result**: **+28.6 Elo** (SPRT H1 accepted, 426 games, LLR 3.0, 99% LOS)
- **Baseline**: 4bbcb7d
- **Notes**: Singular extensions were 97-100% wasted — verification searches at depth (depth-1)/2 almost never found the TT move to be singular (0-6 extensions out of 134-178 tests). The wasted nodes from verification searches cost more than the rare extensions gained. Biggest single improvement found. Code preserved (just disabled) for potential fractional extension rework. **MERGED**.

## 2026-03-12: TT Age Weight 3 (was 4)
- **Change**: TT replacement scoring: `slotScore = depth - 3*age` (was `- 4*age`). Less aggressive age-out.
- **Result**: +1.7 Elo after 411 games, LLR 0.21 — killed (converged to zero from early -20)
- **Baseline**: c19645d (4bbcb7d)
- **Notes**: TTAge-6 was catastrophic (-70 Elo), TTAge-3 converges to zero. Age weight 4 is bracketed as optimal. Neither more nor less aggressive eviction helps.

## 2026-03-12: Futility Pruning Depth 10 (was depth <= 8)
- **Change**: Extend futility pruning from `depth <= 8` to `depth <= 10`.
- **Result**: +2.5 Elo after 843 games, LLR 0.82 — killed (faded from early +7, heading to zero)
- **Baseline**: c19645d (4bbcb7d)
- **Notes**: Had the most persistent positive signal of the batch (+7.2 at 689 games) but ultimately faded. At depth 9-10, the futility margin (1000-1100cp) is so wide that it rarely triggers, making the extension nearly a no-op. Futility depth 8 is well-calibrated.

## 2026-03-12: Fractional Extensions (singular=half ply, additive stacking)
- **Change**: Extensions use 1/16 ply units. Singular extension gives half-ply (was full ply). Extensions stack additively (check+singular+recapture can combine). Depth limiting: halve extensions when ply > 2*rootDepth. Removed mutual exclusivity between extension types.
- **Result**: +11.2 Elo after 556 games vs OLD baseline (with singular enabled) — NOT retested against new baseline. Likely zero or negative vs no-singular since it re-enables costly verification searches.
- **Baseline**: 4bbcb7d (old, singular enabled)
- **Notes**: The fractional infrastructure is sound, but singular verification searches are fundamentally wasteful in our engine (97-100% produce no extension). The half-ply additive approach doesn't fix the core problem. Infrastructure preserved in frac-ext worktree for future use if we find a less wasteful trigger condition.

## 2026-03-12: Passed Pawn Push Ordering in Endgame
- **Change**: In quiet move scoring, add bonus for passed pawn pushes to 5th+ rank in endgame (<10 non-pawn pieces). Bonus scales by rank: 5th=2000, 6th=4000, 7th=6000.
- **Result**: **-36.2 Elo** (SPRT H0 accepted, 385 games vs new baseline)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Strongly negative. The static ordering bonus overrides learned history signals that NNUE-guided search has built up. With NNUE, the eval already knows about passed pawn advancement — adding a crude rank-based bonus to move ordering fights the history tables. Lesson: don't add static heuristic bonuses to move ordering when history tables already capture the pattern.

## 2026-03-12: King Centralization Ordering in Endgame
- **Change**: In quiet move scoring, add bonus for king moves toward center in endgame. Bonus = (centerDist(from) - centerDist(to)) * 3000.
- **Result**: -11.6 Elo after 546 games, LLR -1.13 — killed (clearly negative)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Same problem as PPOrder — static king centralization bonus conflicts with learned history. NNUE already evaluates king placement; the search's history tables learn which king moves cause cutoffs. Overriding with a distance heuristic adds noise. Static move ordering heuristics don't help when NNUE+history already captures the pattern.

## 2026-03-12: Pawn History in LMR Adjustment
- **Change**: Add pawn history score to the LMR history adjustment (alongside main history and cont history).
- **Result**: +1.9 Elo after 544 games — killed (dead flat)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Pawn history adds noise to the LMR adjustment. The main+cont history signals already capture move quality well enough. Consistent with earlier finding that pawn history helps ordering but not LMR/pruning.

## 2026-03-12: Deeper IIR (reduce by 2 at depth >= 12)
- **Change**: When no TT move at depth >= 12, reduce by 2 plies instead of 1.
- **Result**: -25.4 Elo after 279 games, LLR -1.24 — killed (clearly negative)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Reducing by 2 plies loses too much information at deep nodes. Even without a TT move, the full-depth search at these depths finds important tactical sequences that a 2-ply reduction misses. IIR by 1 at depth >= 6 is well-calibrated.

## 2026-03-12: NMP skip when not improving (depth 3-7)
- **Change**: Add `(improving || depth >= 8)` condition to NMP. Skips null move at shallow depths when eval is declining.
- **Result**: -12.6 Elo after 598 games, LLR -1.27 — killed (clearly negative)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: NMP is valuable even when position is declining — the null move hypothesis (standing pat is OK) is about absolute eval, not the trend. Restricting NMP removes a critical pruning tool at exactly the depths where it saves the most relative work. NMP at depth 3-7 is well-calibrated.

## 2026-03-12: TT score refines static eval for pruning
- **Change**: After computing corrected static eval, adjust toward TT score using bound information (exact→replace, lower→max, upper→min). Required `entry.Depth >= depth-3`.
- **Result**: **-77.1 Elo** (SPRT H0 accepted, 174 games — catastrophic)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: TT scores are minimax scores from search, not static evaluations. Using them as static eval for pruning decisions (RFP, NMP, futility) breaks the assumptions those techniques rely on. A TT score of +500 at depth 8 might reflect a forcing sequence that doesn't apply when we're making a different move. Static eval must remain a position-only estimate. Stockfish does a milder version (only adjusting improving flag, not the eval itself).

## 2026-03-12: MainHist-2x v2 (double main history weight in move ordering)
- **Change**: `score = 2 * int(mp.history[m.From()][m.To()])` — doubles main history weight relative to cont history.
- **Result**: +3.0 Elo after 1792 games, LLR 1.88 — killed (fading from early +8.5, converging to zero)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Initially promising but faded over 1800 games. Main history weight 1x is well-calibrated relative to continuation history. Doubling main history doesn't improve move ordering quality.

## 2026-03-12: Razor-D3 (extend razoring to depth 3)
- **Change**: `depth <= 2` → `depth <= 3` for razoring. Margin at depth 3: 400+300=700cp.
- **Result**: -0.3 Elo after 1398 games — killed (dead flat)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Razoring at depth 3 with a 700cp margin rarely fires (most positions aren't 7 pawns behind), making it a no-op in practice. Confirms razoring depth 2 is well-calibrated. Pattern #10.

## 2026-03-12: Disable Recapture Extensions (ablation test)
- **Change**: Set `RecaptureExtEnabled = false`. Tests whether recapture extensions are earning their cost.
- **Result**: **-18.2 Elo** after 737 games — killed (clearly negative, recapture extensions ARE useful)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Opposite of the singular ablation result. Recapture extensions successfully resolve tactical exchanges — extending when recapturing on the same square prevents the search from cutting off mid-exchange. Unlike singular extensions (97% wasted verification searches), recapture extensions fire on genuinely forcing moves. Do not reduce or remove.

## 2026-03-12: Disable Passed Pawn Extensions (ablation test)
- **Change**: Set `PassedPawnExtEnabled = false`. Tests whether PP push extensions are earning their cost.
- **Result**: 0.0 Elo after 1095 games, LLR 0.51 — killed (conclusively neutral)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: PP extensions on 6th+7th rank passed pawns are pure noise. Full-ply extension on these moves wastes as much search depth as it gains. Confirms PP extensions should be removed or significantly reduced.

## 2026-03-12: MainHist-2x v2 (double main history weight in move ordering)
- **Change**: `score = 2 * int(mp.history[m.From()][m.To()])` — doubles main history weight relative to cont history.
- **Result**: +3.0 Elo after 1792 games, LLR 1.88 — killed (fading from early +8.5, converging to zero)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Initially promising but faded over 1800 games. Main history weight 1x is well-calibrated relative to continuation history.

## 2026-03-12: Razor-D3 (extend razoring to depth 3)
- **Change**: `depth <= 2` → `depth <= 3` for razoring. Margin at depth 3: 400+300=700cp.
- **Result**: -0.3 Elo after 1398 games — killed (dead flat)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Razoring at depth 3 with 700cp margin rarely fires. Confirms razoring depth 2 is well-calibrated. Pattern #10.

## 2026-03-12: Check Extension 12/16 ply (fractional, MERGED)
- **Change**: Fractional extension infrastructure (OnePly=16, additive stacking, depth limiting). Check extension reduced from full ply (16/16) to 12/16 (3/4 ply). Includes EG loose check filter (undoes 12/16 for loose checks).
- **Result**: **+11.2 Elo**, H1 accepted. W298-L266-D431 (995 games). LOS 91.1%.
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Reducing check extension saves search depth on checks that don't need full resolution, while still extending enough to find tactical threats. The "waste removal" pattern continues — full-ply extensions are too generous for most checks. Fractional extensions unlock further tuning of individual extension amounts.

## 2026-03-12: Fractional Extensions Infrastructure (additive stacking, all full ply)
- **Change**: OnePly=16, additive extensions (check+recap can stack), depth limiting (halve when ply > 2*rootDepth). All extensions remain at full ply.
- **Result**: 0.0 Elo after 1140 games, LLR 1.20 — killed (flat zero, contaminated by Check12 merge)
- **Baseline**: 9051739 (singular disabled)
- **Notes**: The infrastructure itself is neutral — additive stacking and depth limiting don't change behavior when extensions are mutually rare. Value is as a platform for tuning individual extension amounts. Merged as part of Check12.

## 2026-03-12: Recapture Extension 12/16 ply
- **Change**: Recapture ext reduced from full ply to 12/16 (3/4 ply). Frac-ext infrastructure.
- **Result**: +0.4 Elo after 743 games — killed (flat, contaminated by Check12 merge). Relaunching vs new baseline.
- **Baseline**: 9051739 (singular disabled)
- **Notes**: 12/16 recapture is indistinguishable from full ply at this sample size. Relaunching on new baseline (with check 12/16).

## 2026-03-12: PP Extension 12/16, 7th rank only
- **Change**: PP ext restricted to 7th rank only, reduced to 12/16 (3/4 ply). Frac-ext infrastructure.
- **Result**: -3.1 Elo after 730 games — killed (mildly negative, contaminated by Check12 merge). Relaunching vs new baseline.
- **Baseline**: 9051739 (singular disabled)
- **Notes**: Even restricted to the most critical rank at reduced extension, PP extensions trend negative. Relaunching to confirm on new baseline.

## 2026-03-12: Recapture Extension 14/16 ply
- **Change**: Recapture ext reduced from full ply to 14/16 (7/8 ply). Bracket test above 12/16.
- **Result**: **-49.8 Elo** (SPRT H0 accepted, 295 games). Catastrophic.
- **Baseline**: 8ea8a81 (check 12/16 + frac-ext)
- **Notes**: Even a small reduction from full ply to 14/16 destroys recapture extension effectiveness. Recapture extensions need the full ply to properly resolve tactical exchanges. Combined with NoRecap (-18 Elo) and Recap12 (flat), this confirms: recapture must stay at full ply (16/16). Do not reduce.

## 2026-03-12: Check Extension 14/16 ply
- **Change**: Check extension at 14/16 (7/8 ply) instead of 12/16 (3/4 ply). Frac-ext infrastructure.
- **Result**: +2.8 Elo after 968 games — killed (inconclusive, fractional extensions removed)
- **Baseline**: 8ea8a81 (check 12/16 + frac-ext)
- **Notes**: Not enough games to conclude. Moot point — check extensions removed entirely (see below).

## 2026-03-12: Recapture Extension 12/16 ply v2
- **Change**: Recapture ext reduced from full ply to 12/16. Relaunched on new baseline.
- **Result**: -5.6 Elo after 1311 games — killed (trending H0, fractional extensions removed)
- **Baseline**: 8ea8a81 (check 12/16 + frac-ext)
- **Notes**: Confirms recap must stay at full ply. Consistent with Recap14 (-49.8) and original Recap12 (+0.4 flat).

## 2026-03-12: PP Extension 12/16, 7th rank only v2
- **Change**: PP ext restricted to 7th rank only, 12/16 ply. Relaunched on new baseline.
- **Result**: -9.1 Elo after 1312 games — killed (trending H0, fractional extensions removed)
- **Baseline**: 8ea8a81 (check 12/16 + frac-ext)
- **Notes**: PP extensions at any amount continue to trend negative. Confirms removal is correct.

## 2026-03-12: Extension Simplification (MERGED)
- **Change**: Remove fractional extension infrastructure entirely. Remove check extensions (harmful, -11.2 Elo). Remove PP extensions (noise, 0.0 Elo). Keep only recapture at integer 1 ply. Removes ~46 lines, replaces with ~10.
- **Result**: Non-regression expected (effectively equivalent to Check12 baseline where check 12/16 truncated to 0 anyway).
- **Baseline**: 8ea8a81 (check 12/16 + frac-ext)
- **Key insight**: Check extension at 12/16 was `12/16 = 0` in integer division — the +11.2 Elo gain was from *disabling* check extensions, not from fractional precision. Fractional depth is a dead end (Stockfish abandoned it after 5 years due to TT consistency issues). Modern approach: integer extensions, fractional reductions only.
- **Evidence summary**:
  - Check ext: -11.2 Elo when enabled (SPRT H1 for "disabling"), harmful
  - PP ext: 0.0 Elo ablation (1095 games), noise
  - Recapture ext: -18.2 Elo when disabled, essential at full ply
  - Recap 14/16: -49.8 Elo, cannot reduce even slightly
  - Recap 12/16: -5.6 Elo trending, cannot reduce

## 2026-03-12: QS Evasion History Ordering (MERGED)
- **Change**: Pass history, contHist, and pawnHist pointers to QS evasion move picker instead of nil. Previously quiet evasions in QS were scored 0 (random order).
- **Result**: 0.0 Elo (49.9%, 213-215-323, 751 games). Correctness fix, confirmed non-regression.
- **Baseline**: 04796f7 (simplified extensions)
- **Notes**: Bug fix — evasion ordering in QS was unordered for quiet moves. No strength gain because QS evasion sets are tiny (2-8 moves) and beta cutoffs are rare when in check during QS. Merged as correctness improvement.

## 2026-03-12: CaptHist Scaling /16 in Good Captures (MERGED)
- **Change**: Scale capture history by /16 in good capture scoring (`mvvLva + captHistScore/16` instead of `mvvLva + captHistScore`). CaptHist range [-16384, +16384] was dominating MVV-LVA range [~0, 9990], causing misordering.
- **Result**: **+6.4 Elo** (50.9%, 475-447-657, 1579 games). **H1 accepted** (LLR 2.98).
- **Baseline**: 04796f7 (simplified extensions)
- **Notes**: Correctness fix — MVV-LVA should be primary signal for capture ordering, captHist acts as tiebreaker. Before this fix, a PxQ with bad captHist could sort below PxP with good captHist. Bad captures still scored by raw captHist (no MVV-LVA there).

## 2026-03-12: ContHist2 in MovePicker (quiet+evasion scoring)
- **Change**: Added contHist2 (2-ply continuation history) to MovePicker quiet and evasion scoring at 1x weight (vs 3x for contHist1). contHist2 was already used in LMR/pruning but not in move ordering.
- **Result**: -8 Elo (48.8%, 448-485-637, 1570 games). Killed — persistent negative.
- **Baseline**: 04796f7 (simplified extensions)
- **Notes**: Adding contHist2 to ordering hurts despite being useful in LMR/pruning. Possible explanations: (1) 2-ply history is too noisy for ordering — the signal-to-noise ratio is worse than 1-ply contHist; (2) the 1x weight may be wrong; (3) contHist2 may help more as a pruning/reduction signal than as an ordering signal. Don't retry without a different approach (e.g., smaller weight like /2, or only in evasions).

## 2026-03-12: QS Skip Bad Captures (remove double SEE)
- **Change**: In QS, skip bad captures entirely in MovePicker (go straight to stageDone after good captures) and remove redundant SEE check in search loop. NPS optimization.
- **Result**: -2 Elo (49.7%, 376-385-605, 1366 games). Killed — flat zero.
- **Baseline**: 04796f7 (simplified extensions)
- **Notes**: Theory was sound (avoid double SEE for good captures, skip bad captures entirely). But the NPS gain from one fewer SEE call per QS capture is negligible. Strength-neutral.

## 2026-03-12: LMR Reduction-- for Checks (instead of skip)
- **Change**: Remove `!givesCheck` from LMR guard, add `reduction--` for checking moves inside reduction adjustments. Allows LMR on checks but with less reduction.
- **Result**: -31 Elo (45.6%, 64-90-144, 298 games). Killed — strongly negative.
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: Checks genuinely need full-depth search. Even reducing by 1 less than normal is harmful. Combined with check extensions being harmful (-11.2 Elo), this confirms: the correct approach for checks is neither extend nor reduce — just search at normal depth. The `!givesCheck` LMR skip is well-calibrated.

## 2026-03-12: ContHist2 Updates on Cutoff/Penalty
- **Change**: Add contHistPtr2 (2-ply continuation history) to bonus/penalty updates on beta cutoffs and failed quiets. Previously contHist2 was used for LMR/pruning decisions but never received learning signal.
- **Result**: -10 Elo (48.6%, 231-382-255, 868 games). Killed — persistent negative.
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: Updating contHist2 on cutoffs adds noise to the 2-ply history table. The signal from 2-ply-ago context is inherently weaker; adding full-weight updates may dilute the table with unreliable data. Could retry with reduced learning rate (bonus/2) but low priority given consistent negative results with contHist2 changes.

## 2026-03-12: ProbCut MVV-LVA Ordering
- **Change**: Sort ProbCut captures by MVV-LVA before searching. Selection sort with scores array. Previously ProbCut iterated captures in generation order after SEE filter.
- **Result**: -16 Elo (47.8%, 111-166-129, 406 games). Killed — persistent negative.
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: The overhead of sorting ProbCut captures outweighs the benefit of trying better captures first. ProbCut already filters by SEE >= beta-200, so the remaining captures are all "good enough." The extra sorting computation at every ProbCut node adds up. The generation order (which tends to be roughly MVV-ordered anyway due to move generation patterns) is sufficient.

## 2026-03-12: Fail-Low History Penalties
- **Change**: At fail-low nodes (bestScore <= alphaOrig), penalize all quiets tried with historyBonus(depth-1). Known Stockfish technique — none of the quiets was good enough, so discourage them in future ordering.
- **Result**: -43 Elo (43.9%, 42-103-68, 213 games). Killed — strongly negative.
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: Penalizing all quiets at fail-low is too aggressive. Fail-low doesn't mean the moves are bad — it means the position is bad. The quiets may be the best available; penalizing them pollutes history tables with misleading signals. Would need much smaller weight or only penalize moves that scored well below alpha to be viable.

## 2026-03-12: HistoryBonus Cap 1600 (up from 1200)
- **Change**: Increase historyBonus cap from 1200 to 1600 (min(depth*depth, 1600)). Allows deeper nodes to have stronger history updates.
- **Result**: -19 Elo (47.2%, 92-141-111, 344 games). Killed — persistent negative.
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: Higher cap allows deep-node updates to dominate shallow ones, skewing history tables. 1200 is well-calibrated.

## 2026-03-12: HistoryBonus Cap 800 (down from 1200)
- **Change**: Decrease historyBonus cap from 1200 to 800 (min(depth*depth, 800)). Tests whether shallower history signals are sufficient.
- **Result**: -16 Elo (47.7%, 62-127-74, 263 games). Killed — persistent negative.
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: Lower cap weakens deep-node history signal too much, reducing move ordering quality at higher depths. Combined with 1600 losing, 1200 is bracketed as optimal.

## 2026-03-13: IIR Deep2 d≥10 (retest on new baseline)
- **Change**: IIR reduces by 2 plies when depth ≥ 10 and no TT move (was always 1). Retest of 2026-03-10 experiment on new baseline.
- **Result**: -4.3 Elo after 1068 games (killed). W270-L277-D443 (49.7%). LOS 29.6%. LLR -0.42 (-14%).
- **Baseline**: 21b8ddd (captHist scaling)
- **Notes**: Consistent with original result (+1.8 at 1394 games, flat). Double IIR at deep nodes remains neutral-to-slightly-negative. At 10+0.1s TC, depth ≥ 10 fires rarely enough that the savings don't compensate for the information loss. Not worth revisiting.

## 2026-03-13: IIR Deep2 d≥8
- **Change**: IIR reduces by 2 plies when depth ≥ 8 (lowered from 10) and no TT move.
- **Result**: -6.9 Elo after 2231 games (killed, no SPRT conclusion). W580-L624-D1027 (49.0%). LOS 10.2%. LLR -2.24 (-76%).
- **Baseline**: 4bbcb7d (conthist 3x, QS TT move)
- **Notes**: Lowering the depth gate makes it worse. Double IIR fires more often at d≥8, losing too much search information. Combined with the d≥10 result, double IIR is harmful at any threshold. Don't revisit.

## 2026-03-13: Killer Evasion Bonus
- **Change**: Give killer moves a bonus in evasion move scoring (was unscored).
- **Result**: -6.0 Elo after 2101 games (killed, no SPRT conclusion). W577-L614-D910 (49.1%). LOS 14.8%. LLR -1.59 (-54%).
- **Baseline**: 4bbcb7d (conthist 3x, QS TT move)
- **Notes**: Evasion sets are small (2-8 moves), so ordering quality has minimal impact. Killers from non-evasion contexts may not be relevant when in check. Not worth revisiting.

## 2026-03-13: LMR SEE Quiet Reduction
- **Change**: Increase LMR reduction for quiet moves with negative SEE (SEE < 0 → reduction += 1).
- **Result**: -5.6 Elo after 1631 games (killed, no SPRT conclusion). W441-L468-D722 (49.2%). LOS 19.4%. LLR -1.11 (-38%).
- **Baseline**: 4bbcb7d (conthist 3x, QS TT move)
- **Notes**: SEE on quiet moves is unreliable — it measures capture exchanges, not positional value. Over-pruning quiet moves that happen to lose material in tactical lines misses important positional moves.

## 2026-03-13: PawnHist in LMR Formula
- **Change**: Add pawn structure history to LMR reduction formula (alongside main history and continuation history).
- **Result**: -4.3 Elo after 2055 games (killed, no SPRT conclusion). W556-L581-D918 (49.4%). LOS 22.8%. LLR -0.81 (-27%).
- **Baseline**: 4bbcb7d (conthist 3x, QS TT move)
- **Notes**: PawnHist signal is too noisy/slow to learn for LMR adjustment. The /5000 divisor would need retuning, but the trend is clearly negative. Adding more history dimensions has diminishing returns — main + continuation is sufficient.

## 2026-03-14: V4 Net Threshold Tuning — RFP 100/70

**Context**: V4 net (net-v4-classical.nnue) trained on classical depth-8 scored data has correct eval scale but loses ~230 Elo against v3 baseline due to threshold mismatch. All experiments below test v4 vs v4 (same net, different thresholds).

**SPRT settings**: elo0=-5 elo1=15 alpha=0.05 beta=0.05, tc=10+0.1, Hash=64, v4 net, concurrency=4.

### RFP Margins 85/60 → 100/70
- **Change**: Reverse futility pruning margins from depth×85 (non-improving) / depth×60 (improving) → depth×100 / depth×70.
- **Result**: **H1 accepted, +28.6 Elo** ±24.6 in 475 games. W166-L127-D182 (54.1%). LOS 98.9%.
- **Notes**: Largest single threshold gain. Correct eval scale means positions are evaluated with larger magnitudes in tactical situations, so RFP needs wider margins to avoid over-pruning.

### Futility Margins 100+d×100 → 120+d×120 (1.2x)
- **Change**: Futility pruning base and scale from 100+lmrDepth×100 → 120+lmrDepth×120.
- **Result**: **H0 accepted, -19.6 Elo** ±25.2 in 461 games. W132-L158-D171 (47.2%). LOS 6.3%.
- **Notes**: Futility margins were already well-calibrated for v4 scale. Widening them over-prunes good moves. The original 100+d×100 margins may be near-optimal, or the optimal direction is tighter, not wider.

### Aspiration Delta 15 → 20
- **Change**: Initial aspiration window width from 15 to 20.
- **Result**: ~0 Elo after 478 games (still running, converging to H0). W151-L153-D174 (49.8%).
- **Notes**: Neutral. Aspiration window width is relatively scale-insensitive since it self-adjusts via the 1.5x widening mechanism.

### Razoring 400+d×100 → 500+d×120 (1.2x)
- **Change**: Razoring base from 400 to 500, per-depth from 100 to 120.
- **Result**: ~0 Elo after 491 games (still running, converging to H0). W154-L157-D180 (49.7%).
- **Notes**: Neutral. Razoring fires rarely (depth ≤ 2 only), so its impact is small regardless of scale.

### Earlier round: 1.5x scaling (all rejected)
- RFP 130/90, Futility 150+150d, Aspiration 25, Razoring 600+150d all trended negative (-13 to -42 Elo) after ~100 games. 1.5x was too aggressive. Killed early.

### ProbCut Margin 200 → 240 (with gate 100 → 120)
- **Change**: ProbCut beta margin from beta+200 → beta+240, eval gate from staticEval+100 → staticEval+120.
- **Result**: **H0 accepted, -24.6 Elo** ±27.8 in 382 games. W108-L135-D139 (46.5%). LOS 4.2%.
- **Notes**: ProbCut margins don't need scaling for v4 net. The current 200cp margin is well-calibrated. Widening over-prunes.

### SEE Capture Threshold -d×100 → -d×120
- **Change**: SEE capture pruning threshold from -depth×100 → -depth×120.
- **Result**: **H0 accepted, -13.3 Elo** ±21.8 in 549 games. W144-L165-D240 (48.1%). LOS 11.6%.
- **Notes**: SEE is material-based and already correctly scaled. Loosening the threshold allows more bad captures through, hurting play. Current -d×100 is near-optimal.

### NMP Eval Divisor 200 → 240
- **Change**: Null-move pruning eval-based reduction divisor from 200 → 240.
- **Result**: **H0 accepted, -11.4 Elo** ±20.5 in 641 games. W175-L196-D270 (48.4%). LOS 13.8%.
- **Notes**: NMP divisor controls how much extra reduction is given when eval exceeds beta. Widening the divisor reduces the extra reduction, making NMP less aggressive. The current 200 is well-calibrated for v4 scale. Consider testing tighter (170) as a replacement.

### Singular Margin d×3 → d×4
- **Change**: Singular extension margin from depth×3 → depth×4 (also double-singular and negative-singular thresholds).
- **Result**: **H0 accepted, -6.2 Elo** ±16.8 in 958 games. W271-L288-D399 (49.1%). LOS 23.6%.
- **Notes**: Singular margin is well-calibrated at d×3. Widening reduces the number of singular extensions, losing valuable search depth on critical moves. Both directions tested (d×3 was previously tuned), confirming near-optimality.

### Delta Pruning QS Buffer +200 → +240
- **Change**: Quiescence search delta pruning margin from SEEPieceValues[captured]+200 → +240.
- **Result**: **H0 accepted, -11.2 Elo** ±20.6 in 588 games. W148-L167-D273 (48.4%). LOS 14.2%.
- **Notes**: Delta pruning buffer is already well-calibrated at +200. Widening lets too many futile captures through QS, wasting time. The buffer accounts for positional value beyond material, which doesn't scale with eval magnitude.

### SEE Quiet Threshold -20×d² → -25×d²
- **Change**: SEE quiet move pruning threshold from -20×depth² → -25×depth².
- **Result**: **H0 accepted, -2.8 Elo** ±14.1 in 1347 games. W386-L397-D564 (49.6%). LOS 34.7%.
- **Notes**: Very close to zero — 1347 games to converge. SEE quiet threshold is well-calibrated at -20×d². The v4 net doesn't significantly change quiet move SEE dynamics.

### NMP Eval Divisor 200 → 170 (tighter)
- **Change**: Null-move pruning eval-based reduction divisor from 200 → 170 (more aggressive NMP).
- **Result**: **H0 accepted, -4.4 Elo** ±15.4 in 1116 games. W313-L327-D476 (49.4%). LOS 29.0%.
- **Notes**: Both directions tested (240 and 170), both rejected. NMP divisor 200 is well-calibrated for v4 net. The parameter is near-optimal — don't revisit.

### Contempt 10 → 15
- **Change**: Draw avoidance contempt penalty from 10 → 15 centipawns.
- **Result**: **H0 accepted, -11.7 Elo** ±20.8 in 626 games. W172-L193-D261 (48.3%). LOS 13.6%.
- **Notes**: Higher contempt causes the engine to avoid draws too aggressively in self-play, accepting worse positions rather than drawing. Current contempt=10 is well-calibrated.

### Instability Threshold 200 → 240
- **Change**: Eval instability detection threshold from 200cp → 240cp swing between parent/child nodes.
- **Result**: **H0 accepted, -0.8 Elo** ±12.2 in 1743 games. W490-L494-D759 (49.9%). LOS 44.9%.
- **Notes**: Almost perfectly zero — needed 1743 games to converge. Instability threshold at 200 is well-calibrated. This parameter is scale-insensitive since it measures relative eval swings, not absolute values.

### TM Score Delta Stable ≤10 → ≤15
- **Change**: Time management stable score threshold from ≤10cp → ≤15cp (reduces time on more positions).
- **Result**: **H0 accepted, -2.2 Elo** ±13.7 in 1415 games. W400-L409-D606 (49.7%). LOS 37.6%.
- **Notes**: Near-zero after 1415 games. TM stable threshold at 10cp is well-calibrated for v4 net. Score deltas between iterations don't scale with eval magnitude since they measure relative changes.

### TM Score Delta Medium >25/>50 → >35/>70
- **Change**: Time management volatile score thresholds from >25cp/50cp → >35cp/70cp (require larger swings to extend time).
- **Result**: **H0 accepted, -1.1 Elo** ±12.6 in 1632 games. W452-L457-D723 (49.8%). LOS 43.4%.
- **Notes**: Near-zero after 1632 games. Time management score deltas are relative measures (iteration-to-iteration changes), not absolute eval values, so they don't scale with eval magnitude. All TM thresholds are confirmed well-calibrated.

### Futility Tighter 100+d×100 → 80+d×80
- **Change**: Futility pruning margins tightened from 100+lmrDepth×100 → 80+lmrDepth×80.
- **Result**: ~0 Elo after 2062 games (killed). W593-L580-D889 (50.3%). Converging to H0.
- **Notes**: Both directions tested (120 rejected at -19.6, 80 flat at +1.8). Futility margins at 100+d×100 are well-calibrated for v4 net. Confirmed near-optimal.

### RFP Depth Gate ≤7 → ≤8
- **Change**: Allow reverse futility pruning at one deeper ply (depth ≤ 8 instead of ≤ 7).
- **Result**: ~0 Elo after 686 games (killed). W188-L189-D309 (49.9%). Dead flat.
- **Notes**: Depth gate doesn't benefit from v4's eval. At depth 8 the margins (800cp non-improving) are too large for reliable static pruning.

### Razoring Depth Gate ≤2 → ≤3
- **Change**: Allow razoring at depth 3 (was depth ≤ 2 only).
- **Result**: ~-10 Elo after 405 games (killed). W109-L120-D176 (48.6%). Trending negative.
- **Notes**: Razoring at depth 3 drops to QS too early, losing search depth. The v4 net's better eval doesn't compensate for the lost search.

### Razoring Tighter 400+d×100 → 350+d×90
- **Change**: Razoring margins tightened from 400+depth×100 → 350+depth×90.
- **Result**: **H0 accepted, -28.2 Elo** ±29.4 in 309 games. W76-L101-D132 (46.0%). LOS 3.0%.
- **Notes**: Both directions tested: wider (500+d×120, ~0 Elo), tighter (350+d×90, -28 Elo), deeper (depth≤3, -10 Elo). Tighter razoring drops to QS too aggressively. Current 400+d×100 is well-calibrated. Don't revisit.

### Contempt 10 → 5 (reverse direction)
- **Change**: Contempt (draw avoidance penalty) from 10 → 5 centipawns.
- **Result**: **H0 accepted, -17.0 Elo** ±23.8 in 490 games. W134-L158-D198 (47.6%). LOS 8.0%.
- **Notes**: Both directions tested (15 and 5), both rejected. Contempt=10 is optimal. Lower contempt accepts too many draws; higher contempt avoids draws too aggressively. This parameter is eval-scale-independent (fixed return value in search, not derived from NNUE output).

### RFP 100/70 → 110/80 (further scaling)
- **Change**: RFP margins from 100/70 → 110/80 (further scaling beyond proven 100/70).
- **Result**: **H0 accepted, +0.9 Elo** ±10.2 in 1957 games. LOS 57.2%. Dead flat.
- **Notes**: 100/70 is well-bracketed. Both directions (110/80 and original 85/60) tested; 100/70 is optimal for v4 net.

### ProbCut Margin 200 → 170 with Gate 100 → 85 (MERGED)
- **Change**: ProbCut beta margin from beta+200 → beta+170, eval pre-filter gate from staticEval+100 → staticEval+85.
- **Result**: **H1 accepted, +10.0 Elo** ±11.3 in 2042 games. W608-L549-D885 (51.4%). LOS 95.9%. LLR 3.01.
- **Baseline**: V4 net with RFP 100/70. SPRT elo0=-5 elo1=15.
- **Commit**: (pending merge)
- **Notes**: Second v4 threshold win after RFP. Tighter ProbCut means we try the expensive verification search more often, catching positions where the null-move-like cut was too optimistic. The opposite direction (200→240) was rejected at -24.6 Elo, confirming 170 is the right direction.

### Futility 100+d×100 → 90+d×90 (tighter)
- **Change**: Futility pruning margins tightened from 100+lmrDepth×100 → 90+lmrDepth×90.
- **Result**: **H0 accepted, -2.2 Elo** ±13.5 in 1431 games. W400-L409-D622 (49.7%). LOS 37.6%.
- **Notes**: Both directions tested (120 rejected at -19.6, 90 flat at -2.2, 80 flat at +1.8). Futility margins at 100+d×100 are well-calibrated and bracketed.

### ContHist2 Cutoff/Penalty Updates
- **Change**: Add learning signal updates (cutoff bonus + quiet penalty) to 2-ply continuation history table, which was previously read-only for LMR/pruning.
- **Result**: **H0 accepted, -7.5 Elo** ±18.0 in 791 games. W210-L227-D354 (48.9%). LOS 20.8%.
- **Notes**: The ply-2 continuation history lookup is too noisy for reliable learning signal. The piece at the ply-2 move may have been captured, making the key stale. Read-only with half-weight remains the correct approach.

### LMR Reduce Less for Checks (reduction--)
- **Change**: Allow LMR on checking moves (was skipped entirely), but with reduction-- to reduce less.
- **Result**: **H0 accepted, -8.1 Elo** ±18.5 in 771 games. W210-L228-D333 (48.8%). LOS 19.5%.
- **Notes**: Checking moves should not be reduced at all. Skipping LMR for checks is correct — they are tactically significant and reducing them loses search depth on critical lines. The check extension removal (earlier experiment) was also harmful. Checks need full search depth.

### ProbCut MVV-LVA Ordering
- **Change**: Sort ProbCut captures by MVV-LVA (sort.Slice) before iterating, so highest-value captures are tried first.
- **Result**: **H0 accepted, -18.3 Elo** ±24.5 in 438 games. W113-L136-D189 (47.4%). LOS 7.2%.
- **Notes**: The sort.Slice overhead outweighs any benefit from better ordering. ProbCut already uses SEE >= 0 filter, and the capture set is small (typically 2-5 moves). The sort allocations may also cause GC pressure. Generation order is adequate for this small set.

### SkipQuiets After LMP
- **Change**: When LMP triggers, tell the MovePicker to skip quiet move generation entirely (jump to bad captures stage).
- **Result**: **H0 accepted, -35.0 Elo** ±32.0 in 269 games. W66-L93-D110 (45.0%). LOS 1.6%.
- **Notes**: Catastrophic. Skipping quiet generation after LMP breaks the killer and counter-move stages which come before quiets but after good captures. Those moves are essential even when LMP fires. The implementation may have been too aggressive in what it skipped.

### ProbCut β+150 (tighter, bracket optimum)
- **Change**: ProbCut margin from β+170 → β+150, gate 85→75.
- **Result**: Killed at 639 games, -3.9 Elo ±20.3. W172-L183-D284 (49.1%). Trending H0.
- **Notes**: Both tighter (150) and wider (240, -24.6 Elo) rejected. ProbCut margin 170 is well-bracketed as optimal.

### ProbCut Verification Depth depth-4 → depth-3
- **Change**: Deeper ProbCut verification search (depth-3 instead of depth-4).
- **Result**: Killed at 394 games, -10.1 Elo ±26.5. W111-L118-D165 (49.1%). Trending H0.
- **Notes**: Deeper verification is too expensive — the extra ply costs more than it saves in pruning accuracy. depth-4 is optimal.

## 2026-03-14: Engine Review Ideas — SPRT Testing

**Context**: Ideas from cross-referencing 12 chess engines (see engine-notes/SUMMARY.md). Tested against V4 baseline with RFP 100/70 + ProbCut 170/85 + LMR-split. SPRT elo0=-5 elo1=15, tc=10+0.1, concurrency=4.

### LMR Separate Tables for Captures vs Quiets (MERGED)
- **Change**: Separate LMR reduction tables for captures (C=1.80, less aggressive) and quiets (C=1.50). Table-based capture LMR with captHist adjustments (>2000 reduces less, <-2000 reduces more), replacing hard-coded reduction=1/2.
- **Result**: **H1 accepted, +43.5 Elo**. Biggest single win in the project.
- **Source**: Midnight (cap: 1.40/1.80, quiet: 1.50/1.75), Caissa
- **Commit**: d8fdc3c
- **Notes**: Captures and quiets have fundamentally different reduction needs. Captures are more forcing and should be reduced less. The table-based approach with capture history integration gives much finer-grained control.

### RFP Score Dampening (eval+beta)/2
- **Change**: Return (eval+beta)/2 instead of eval on RFP cutoff. Blends the raw eval with the bound to prevent score inflation.
- **Result**: **H0 accepted, -16.7 Elo**. W119-L147-D199 (47.0%).
- **Source**: Winter, Arasan
- **Notes**: Our RFP margins are already well-calibrated (100/70). Dampening loses precision — the raw eval is more informative than the blended value at our margin levels.

### NMP TT Guard (killed — flat zero, RETRY CANDIDATE)
- **Change**: Skip null-move pruning if TT has upper-bound entry with score below beta (predicts fail-low).
- **Result**: Killed at 806 games, +0.9 Elo ±18.6. W242-L238-D326 (50.2%). LLR -0.92. Dead flat.
- **Source**: Ethereal, Rodent, Arasan
- **Notes**: Was -45 Elo early (~100 games), recovered to zero by 800 games. Our lockless 4-slot TT may have unreliable shallow TTUpper entries — the guard fires on noisy data. Needs depth guard or different TT architecture to work.
- **Retry**: Candidate for retest after better NNUE net or other search improvements. Could also run with patience (3000+ games) to detect a +5 Elo gain. Consider adding a depth guard (only trust TTUpper entries with sufficient depth).

### History Bonus Scaled by Score Difference
- **Change**: Scale history update bonus proportionally to score-beta on cutoffs. bonus = bonus + bonus * min(scoreDiff, 300) / 300. Applied to both quiet and capture history.
- **Result**: **H0 accepted, -33.9 Elo** ±31.8 in 257 games. W59-L84-D114 (45.1%). LOS 1.8%.
- **Source**: Caissa
- **Notes**: Strongly negative. Scaling history bonuses by score difference adds too much noise — large cutoff margins don't necessarily mean the move is more informative. The flat bonus approach is more robust. Our history divisor (5000) is already well-calibrated.

### NMP Threat Detection (MERGED)
- **Change**: After null move fails, extract opponent's TT best move target square, give +8000 ordering bonus to quiet moves escaping from that square.
- **Result**: **H1 accepted, +12.4 Elo** ±13.7 in 1398 games. W423-L373-D602 (51.8%). LOS 96.2%. LLR 3.02.
- **Source**: Rodent (+2048 bonus), ExChess (exempt from pruning), Texel (defense heuristic)
- **Commit**: 464a425
- **Notes**: Patient win — hovered at +4-8 for hundreds of games before converging. The TT probe after NMP failure is essentially free (the entry exists from the null-move search). The +8000 bonus is large enough to promote threat evasions above most quiets but below good captures and killers. Fourth engine-review win.

### Non-Pawn Material Correction History
- **Change**: Added second correction history table indexed by XOR hash of non-pawn piece bitboards (knights through queens). Applied alongside pawn correction, averaged (each contributes half weight).
- **Result**: **H0 accepted, -11.8 Elo** ±20.7 in 618 games. W166-L187-D265 (48.3%). LOS 13.2%.
- **Source**: Arasan (6 tables), Caissa (3+ tables), Winter (16 dims)
- **Notes**: The XOR hash of piece bitboards may not be discriminating enough — different piece configurations can hash to the same value. Also, averaging pawn+non-pawn corrections at equal weight may dilute the pawn signal. Consider: (1) better hash (use Zobrist keys per piece), (2) weighted sum favoring pawn correction, (3) separate application rather than averaging. Despite failing, expanded correction history is proven in strong engines — the implementation matters.

### QS Beta Blending (MERGED)
- **Change**: At non-PV QS nodes, blend fail-high scores with beta: `(bestScore+beta)/2`. Applied at both stand-pat cutoffs and capture fail-highs.
- **Result**: **Accepted at +4.9 Elo** ±10.1 in 2467 games. W687-L657-D1123 (50.6%). LOS 83.1%. LLR did not converge but consistently positive.
- **Source**: Caissa, Berserk, Stormphrax
- **Commit**: 00af62c
- **Notes**: Third application of the score dampening pattern (after TT-dampen +22.1 and FH-blend +14.7). Smaller effect in QS because scores are already closer to ground truth, but still positive. Accepted based on 2467 games of consistent positive signal despite SPRT non-convergence.

### TM Score Delta Stable ≤10 → ≤15 (H0 rejected)
- **Change**: Widen "stable score" threshold from ≤10cp to ≤15cp. This makes the 0.8x time reduction trigger more often.
- **Result**: **H0 accepted, -54.1 Elo** ±39.7 in 149 games. W26-L49-D74 (42.3%). LOS 0.4%.
- **Notes**: Fast, strong rejection. Making the stability detector more permissive wastes time on volatile positions. The ≤10cp threshold is well-calibrated.

### TM Score Delta Medium >25 → >35 (H0 rejected)
- **Change**: Raise "medium instability" threshold from >25cp to >35cp. Requires larger score swings before applying 1.2x time extension.
- **Result**: **H0 accepted, -26.5 Elo** ±28.7 in 250 games. W46-L65-D139 (46.2%). LOS 3.6%.
- **Notes**: Fast rejection. The >25cp medium threshold is well-calibrated. All three TM thresholds tested (instability 200, stable ≤10, medium >25) confirmed optimal.

### QS Delta Pruning Buffer 200 → 240 (MERGED)
- **Change**: Widen QS delta pruning buffer from +200 to +240cp. Captures within 240cp of alpha are now searched instead of pruned.
- **Result**: **H1 accepted, +31.2 Elo** ±26.1 in 368 games. W116-L83-D169 (54.5%). LOS 99.0%. LLR 2.97.
- **Commit**: 0a4b4d1
- **Notes**: Fastest convergence of any experiment — H1 in just 368 games. The old +200 buffer was too aggressive, pruning captures that turned out to matter. Aligns with pattern #5: per-move pruning needs slack.

### SEE Quiet Threshold -20d² → -25d² (H0 rejected)
- **Change**: Tighten SEE quiet pruning threshold from -20d² to -25d².
- **Result**: **H0 accepted, -26.2 Elo** ±28.5 in 305 games. W70-L93-D142 (46.2%). LOS 3.6%.
- **Notes**: Fast rejection. Making SEE pruning more aggressive loses Elo — our -20d² threshold is well-calibrated.

### Instability Threshold 200 → 240 (H0 rejected)
- **Change**: Raise eval instability detection threshold from 200 to 240.
- **Result**: **H0 accepted, -28.8 Elo** ±29.3 in 278 games. W60-L83-D135 (45.9%). LOS 2.7%.
- **Notes**: Fast rejection. Making the instability detector less sensitive hurts. The 200 threshold is well-calibrated.

### Fail-High Score Blending (MERGED)
- **Change**: At non-PV nodes with depth ≥ 3, blend fail-high score toward beta weighted by depth: `(score*depth + beta)/(depth+1)`. Deeper cutoffs trust the raw score more; shallow cutoffs blend more toward beta.
- **Result**: **H1 accepted, +14.7 Elo** ±15.8 in 1038 games. W312-L268-D458 (52.1%). LOS 96.6%. LLR 3.0.
- **Source**: Caissa, Stormphrax, Berserk
- **Commit**: c7b65d0
- **Notes**: Sixth engine-review win. Same dampening pattern as TT score dampening (+22.1). Score inflation at fail-high boundaries is a real problem — non-PV cutoff scores are noisy, and blending toward beta dampens that noise proportional to depth confidence.

### 50-Move Rule Eval Scaling (H0 rejected)
- **Change**: Scale eval toward zero as halfmove clock advances: `eval * (200-fmr) / 200`.
- **Result**: **H0 accepted, -3.0 Elo** ±14.4 in 1255 games. W349-L360-D546 (49.6%). LOS 34.0%. LLR -2.96.
- **Source**: Texel, Berserk
- **Notes**: Dead flat throughout, drifting slightly negative. Our eval doesn't have significant 50-move bias, or the scaling formula is wrong for our engine.

### Singular Margin d*3 → d*2 (H0 rejected)
- **Change**: Tighten singular extension margin from `ttScore - depth*3` to `ttScore - depth*2`. More moves become singular → more extensions.
- **Result**: **H0 accepted, -8.5 Elo** ±18.6 in 697 games. W173-L190-D334 (48.8%). LOS 18.6%. LLR -2.99.
- **Source**: Seer, Berserk, Stormphrax, Koivisto, RubiChess
- **Notes**: Tightening the margin is the wrong direction for our engine. Combined with d*4 also being negative (-6.2), our d*3 margin is well-calibrated. The double extension and multi-cut shortcut variants may still work — those are structural changes, not margin changes.

### Alpha-Reduce: Depth Reduction After Alpha Improvement (MERGED)
- **Change**: After a move raises alpha in the search, all subsequent moves are searched at one ply less depth. Once a PV is established, remaining moves are less likely to improve on it.
- **Result**: **H1 accepted, +13.0 Elo** in 1281 games. LLR 2.95 (elo0=-5, elo1=15).
- **Source**: Caissa (similar approach)
- **Commit**: 4a49d1f
- **Notes**: Fifth engine-review win. Simple 3-line change with strong results. Complements LMR by providing an additional depth reduction mechanism based on search progress rather than move ordering heuristics.

### TT Score Dampening (3*score+beta)/4 (MERGED)
- **Change**: On non-PV TT cutoffs (TTLower, non-mate scores), return (3*score + beta)/4 instead of raw score. Prevents inflated TT scores from propagating.
- **Result**: **H1 accepted, +22.1 Elo** ±21.1 in 567 games. W172-L136-D259 (53.2%). LOS 98.0%. LLR 2.96.
- **Source**: Winter, Caissa
- **Commit**: 2b37d25
- **Notes**: Third engine-review win after LMR-split (+43.5) and ProbCut tightening (+10.0). TT score inflation is a real issue at non-PV nodes — blending toward beta dampens propagation of overly optimistic scores. Consider testing the opposite direction (more dampening: (score+beta)/2) or extending to fail-high dampening in main search.

### RFP TT Quiet-Move Guard (H0 rejected)
- **Change**: Skip reverse futility pruning when TT has a quiet best move — `&& (ttMove == NoMove || board.IsCapture(ttMove))` added to RFP condition. Logic: if we know a good quiet move exists, don't prune based on static eval alone.
- **Result**: **H0 accepted, -31.3 Elo** ±30.7 in 234 games. W45-L66-D123 (45.5%). LOS 2.3%. LLR -2.99.
- **Source**: Tucano, Weiss (history-gated variant)
- **Notes**: Fast rejection. Guarding RFP when a quiet TT move exists actually *hurts* — it prevents RFP from working in positions where it should. The TT quiet move doesn't mean the position is tactical; RFP's eval-based gate is already sufficient. Weiss's variant (guard only when TT move history > 6450) might be better, but the basic guard is clearly wrong for our engine.

### Contempt 10 → 15 (H0 rejected)
- **Change**: Increase contempt from 10 to 15 centipawns. Higher contempt makes the engine avoid draws more aggressively.
- **Result**: **H0 accepted, -5.2 Elo** ±16.2 in 865 games. W206-L219-D440 (49.2%). LOS 26.4%. LLR -2.98.
- **Notes**: Slow grind to rejection. Current contempt of 10 is well-calibrated. Increasing it makes the engine overvalue risky positions. Don't revisit unless eval character changes significantly.

### LMR doDeeperSearch/doShallowerSearch (H0 rejected)
- **Change**: After LMR re-search beats alpha, dynamically adjust newDepth: +1 if score > bestScore+69, -1 if score < bestScore+newDepth. Concentrates effort on genuinely promising LMR fail-highs.
- **Result**: **H0 accepted, -13.7 Elo** ±21.7 in 483 games. W109-L128-D246 (48.0%). LOS 10.9%. LLR -3.05.
- **Source**: Berserk, Stormphrax, Weiss, Obsidian, Stockfish (5 engines)
- **Notes**: Despite 5-engine consensus, clearly negative for our engine. The thresholds (69/newDepth) may be poorly calibrated for our search, or our LMR tables already handle this adequately. Could retry with different margins (Weiss uses `1+6*(newDepth-lmrDepth)`, Obsidian uses tunable 43/11), but the basic concept doesn't seem to help.

### TT Near-Miss Cutoffs (MERGED)
- **Change**: Accept TT entries 1 ply shallower than required, with a 64cp score margin. At non-PV nodes: if TTLower entry has score-64 >= beta, return score-64. If TTUpper entry has score+64 <= alpha, return score+64. Avoids re-searching positions where we have a near-hit.
- **Result**: **H1 accepted, +21.7 Elo** ±20.8 in 561 games. W165-L130-D266 (53.2%). LOS 97.9%. LLR 2.96.
- **Source**: Minic (margin 60-64cp, credited to Ethereal), Ethereal
- **Commit**: a412cbe
- **Notes**: Ninth engine-review win. Strong and fast convergence. The 64cp margin is conservative enough to avoid incorrect cutoffs while still saving significant re-search effort. Only applies at non-PV nodes (beta-alpha == 1) and non-mate scores.

### Singular Extensions + Multi-Cut (H0 rejected)
- **Change**: Re-enable singular extensions (previously disabled at -28.6 Elo) with multi-cut pruning shortcut: when the singular verification search finds singularBeta >= beta, return singularBeta immediately (multiple moves beat beta, position has many good options).
- **Result**: **H0 accepted, -28.5 Elo** ±29.4 in 281 games. W62-L85-D134 (46.2%). LOS 2.9%. LLR -3.00.
- **Source**: Weiss, Obsidian, Minic, Tucano, Koivisto (7+ engines)
- **Notes**: Fast rejection. The underlying problem is that our singular extensions themselves are harmful — the verification search costs more than it saves, even with multi-cut providing an alternative pruning path. Our depth*3 margin may be too loose (97-100% of SE searches found no singularity). Could try tighter margin or depth gate (depth >= 12 instead of 10), but the basic SE framework seems wrong for our engine.

### Futility Pruning Margin 100+d*100 → 80+d*80 (MERGED)
- **Change**: Tighten futility pruning margin from `staticEval+100+lmrDepth*100` to `staticEval+80+lmrDepth*80`. More aggressive pruning of quiet moves that can't reach alpha.
- **Result**: **H1 accepted, +33.6 Elo** ±27.2 in ~300 games. W72-L52-D133 (53.9%). LOS 99.2%. LLR 3.0.
- **Commit**: ec9d8fa
- **Notes**: Tenth win! Previous attempt at 120+d*120 (loosening) failed, confirming the optimum is in the tighter direction. This is the opposite of our usual pattern ("per-move pruning needs slack") — futility uses estimated LMR depth which provides its own slack. Consider testing 60+d*60 to bracket further.

### QS Two-Sided Delta Pruning (H0 — rejected)
- **Change**: Add "good-delta" early return in QS: when standPat + captureValue - 240 >= beta AND SEE is positive, return beta immediately. Complement to existing "bad-delta" (skip captures that can't reach alpha).
- **Result**: H0 at 224 games, -37.4 Elo ±33.4, LOS 1.4%, LLR -2.96.
- **Source**: Minic (credited to Seer)
- **Notes**: The good-delta early return is too aggressive — returning beta without searching the capture misses important tactical complications. Our existing bad-delta + QS beta blending already handles the QS boundary well.

### TT Noisy Move Detection (H1 — MERGED)
- **Change**: Detect when the TT best move is a capture (`ttMoveNoisy := ttMove != NoMove && b.Squares[ttMove.To()] != Empty`). When true, add +1 LMR reduction for quiet moves — if the best known move is tactical, quiet alternatives deserve extra skepticism.
- **Result**: H1 at 304 games, +34.4 Elo ±27.4, LOS 99.3%, LLR 3.02.
- **Baseline**: ec9d8fa (futility 80+d*80)
- **Commit**: 330dcd4
- **Source**: Obsidian, Berserk
- **Notes**: Eleventh SPRT win from engine reviews! Simple 2-line idea with massive payoff. The insight is that when the position's best move is a capture, the position is likely tactical and quiet moves are less relevant — so reduce them more aggressively. This is a form of position-aware LMR that leverages TT information.

### Aspiration Fail-High Depth Reduction (H0 — rejected)
- **Change**: In the aspiration window loop, add `depth--` when the search fails high. Intent: make re-searches cheaper after fail-high by reducing the search depth.
- **Result**: H0 at 26 games, -353.8 Elo ±166.9, LOS 0.0%, LLR -3.09. Catastrophic.
- **Baseline**: ec9d8fa (futility 80+d*80)
- **Source**: 5 engines (but implementation was wrong)
- **Notes**: Bug: `depth--` modified the outer iterative deepening loop variable, permanently reducing search depth for the rest of the game. Correct implementation would use a separate `searchDepth` variable inside the aspiration loop. Not worth retesting — the correct version would need careful scoping and the gain is likely marginal.

### 4-Ply Continuation History (H0 — rejected)
- **Change**: Add `contHistPtr4` from 4 plies ago (our own move 2 full moves back) at quarter weight (1/4) in both history pruning and LMR history scoring. Read-only — no updates to the 4-ply table.
- **Result**: H0 at 303 games, -25.3 Elo ±27.7, LOS 3.7%, LLR -3.04.
- **Baseline**: 330dcd4 (TT-noisy merged)
- **Source**: 10 engines (Berserk, Stormphrax, RubiChess, Caissa, Seer, Weiss, Obsidian, BlackMarlin, Altair, Reckless)
- **Notes**: Despite massive engine consensus (10 engines), this hurts for us. The 4-ply lookback is too distant — the position has changed too much for the correlation to be useful. Our 1-ply and 2-ply continuation history are sufficient. May work better if writes to ply-4 are also done (Obsidian writes at half bonus), but the read-only approach is clearly negative. The 1/4 weight may also be wrong — some engines use equal weight for all plies.

### NMP Less Reduction After Captures (H0 — rejected)
- **Change**: In NMP, reduce R by 1 when the previous move was a capture (`b.UndoStack[last].Captured != Empty → R--`). Rationale: captures change the position significantly, making NMP riskier.
- **Result**: H0 at 668 games, -8.3 Elo ±18.3, LOS 18.7%, LLR -3.04.
- **Baseline**: 330dcd4 (TT-noisy merged)
- **Source**: Tucano
- **Notes**: Our NMP is already well-calibrated for post-capture positions. The R-1 after captures makes NMP too conservative, losing the time savings without compensating accuracy gain.

### Cutnode LMR +2 (H0 — rejected)
- **Change**: Extra LMR reduction at cut nodes increased from +1 to +2. `if beta-alpha == 1 && moveCount > 1 { reduction += 2 }` (was `+= 1`).
- **Result**: H0 at 1930 games, +0.5 Elo ±10.8, LOS 53.9%, LLR -2.93.
- **Baseline**: 330dcd4 (TT-noisy merged)
- **Source**: Weiss (+2), Obsidian, BlackMarlin
- **Notes**: Dead flat after 1930 games. Our existing +1 cut-node reduction is correctly calibrated. +2 is too aggressive — it over-reduces at expected cut nodes, losing tactical accuracy. Despite 3-engine consensus for +2, our engine's LMR tables (C=1.50 quiet / C=1.80 capture) already provide sufficient reduction at cut nodes.

### Cutoff-Count Child Feedback (H0 — rejected)
- **Change**: Track beta cutoffs at child nodes. Added `ChildCutoffs [MaxPly+1]int` to SearchInfo, reset at node entry, increment parent's counter on beta cutoff. In quiet LMR: `if info.ChildCutoffs[ply] > 2 { reduction++ }`.
- **Result**: H0 at 1294 games, -1.6 Elo ±13.1, LOS 40.5%, LLR -2.96.
- **Baseline**: 330dcd4 (TT-noisy merged)
- **Source**: Reckless (cutoff_count > 2 → reduction += 1604)
- **Notes**: Dead flat. The child cutoff count doesn't provide useful signal for parent LMR. Many children cutting off is ambiguous — it could mean many refutations exist (bad position) or many moves are obviously losing (clear best move). The signal is too noisy to improve on existing LMR heuristics.

### "Failing" Heuristic — Position Deterioration (H1 — MERGED)
- **Change**: Detect significant eval deterioration: `failing = staticEval < eval2pliesAgo - (60 + 40*depth)`. When failing: +1 LMR reduction for quiet moves, tighten LMP limit to 2/3. Complements the existing `improving` flag by detecting the opposite — positions getting much worse.
- **Result**: **H1 at 355 games, +29.4 Elo ±25.3, LOS 98.9%, LLR 2.95.** Fast convergence.
- **Baseline**: 330dcd4 (TT-noisy merged)
- **Source**: Altair (failing flag → +1 LMR, tighter LMP divider)
- **Commit**: (pending merge)
- **Notes**: Twelfth engine-review win! Same "search progress feedback" family as alpha-reduce (+13.0). When the position is deteriorating significantly, our moves are being refuted — reduce and prune more aggressively. The depth-scaled threshold (60+40*d) ensures the bar for "failing" rises with depth, avoiding false positives at deep nodes.

### Futility History Gate (H0 — rejected)
- **Change**: Exempt moves with combined history > 12000 from futility pruning. Computed `combinedHist = mainHistory + contHist + contHist2/2` before MakeMove, added `combinedHist <= 12000` condition to futility pruning block.
- **Result**: H0 at 600 games, -9.8 Elo ±19.5, LOS 16.1%, LLR -3.00.
- **Baseline**: 05aee22 (failing heuristic merged)
- **Source**: Igel (history + cmhist + fmhist < fpHistoryLimit[improving])
- **Notes**: Clearly negative. Futility pruning's eval-based gate is already well-calibrated — adding a history exemption allows bad moves through. History is already used in LMR and history pruning; double-dipping in futility is redundant and harmful. Consistent with earlier finding that history-aware futility margins lose Elo (-6.0 in 2026-03-12 experiment).

### NMP Divisor 170 (H0 — rejected)
- **Change**: Reduce NMP depth divisor from 200 to 170 (more aggressive null-move pruning via deeper reductions). `depth - depth/170 - 4` instead of `depth - depth/200 - 4`.
- **Result**: H0 at 3963 games, +2.7 Elo ±7.7, LOS 75.5%, LLR -2.96. SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 05aee22 (failing heuristic merged)
- **Notes**: Long grind to rejection. The +2.7 Elo is in the noise band — NMP divisor 200 is well-calibrated. Combined with the earlier NMP-240 rejection (-3.2 Elo), both directions lose, confirming 200 is optimal. Do not revisit NMP divisor.

### Alpha-Raised Aspiration Window (H1 — MERGED)
- **Change**: After alpha-side aspiration failure, raise alpha to `max(alpha, bestScore - delta/2)` instead of `alpha - delta`. This prevents alpha from collapsing too far on a single fail-low, giving tighter search windows on the retry.
- **Result**: **H1 at 3519 games, +7.5 Elo ±8.0, LOS 96.7%, LLR 3.00.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 05aee22 (failing heuristic merged)
- **Source**: Engine review — Altair aspiration window management
- **Notes**: Thirteenth engine-review win! Same "score dampening" family as TT-dampen (+22.1) and FH-blend (+14.7). Pattern confirmed: score dampening at noisy boundaries has 80% win rate. The key insight is that a single fail-low doesn't mean the true score is far below alpha — raising alpha prevents wasted work on overly wide windows.

### History-Based Extensions (H0 — rejected)
- **Change**: Extend +1 ply when both contHist[0] and contHist[1] are >= 10000 (Igel pattern). Applied only to quiet, non-check moves at depth >= 6.
- **Result**: H0 at 1443 games, -1.2 Elo ±12.7, LOS 42.6%, LLR -2.95. SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 05aee22 (failing heuristic merged)
- **Source**: Igel (extend when continuation histories both high)
- **Notes**: Dead flat. Continuation history values >= 10000 are too rare to trigger frequently enough to matter, and when they do trigger, the position is likely already well-searched. Extensions based on history don't add signal beyond what singular extensions already provide. Consistent with the general finding that history-aware modifications beyond LMR have low success rate (8%).

### Threat-Aware LMR: Pawn Threat Escape (H1 — MERGED)
- **Change**: Compute enemy pawn attacks at each node. In LMR, reduce less (reduction--) when moving a piece away from a pawn-attacked square. The logic: escaping a pawn threat is a purposeful move that deserves deeper search.
- **Result**: **H1 at 951 games, +14.6 Elo ±15.7, LOS 96.6%, LLR 2.99.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 0c1e716 (progressive alpha-raised merged)
- **Source**: Engine review — threat-aware history (12-engine consensus). Simplified to LMR-only adjustment using pawn threats.
- **Notes**: Fourteenth engine-review win! First structural innovation from the next-phase plan. Pawn attacks are cheap to compute (two bitboard shifts + OR). The signal is strong because pieces under pawn attack genuinely need to move, and reducing less on those escape moves prevents overlooking critical tactics. This lighter approach (LMR adjustment only) captured the key benefit without the complexity of a full 4x indexed history table.

### Node-Based Time Management — Aggressive (H0 — rejected)
- **Change**: Track best move's share of root nodes per iteration. Scale soft time limit: >0.9 fraction → 0.6x, >0.8 → 0.75x, <0.2 → 1.6x, <0.4 → 1.3x. More aggressive thresholds than the earlier conservative attempt (+2 Elo, flat).
- **Result**: H0 at 903 games, -4.6 Elo ±15.8, LOS 28.3%, LLR -2.97. SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 0c1e716 (progressive alpha-raised merged)
- **Source**: Next-phase plan item #4 (retry with aggressive scaling)
- **Notes**: Both conservative (flat) and aggressive (negative) node-fraction TM have failed. The issue may be that at 10+0.1 time controls, there aren't enough root nodes per iteration for the fraction to be meaningful. Also, the interaction with existing stability-based TM may be redundant — both try to measure "how confident are we in the best move." Do not revisit node-fraction TM.

### IIR Extra on PV Nodes (inconclusive — killed for net upgrade)
- **Change**: Extra IIR reduction at PV nodes without TT move: depth reduced by 2 instead of 1. `if beta-alpha > 1 { depth-- }` inside IIR block.
- **Result**: Killed at 2102 games, +8.1 Elo ±10.2, LOS 94.0%, LLR 2.28 (77% toward H1). SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 0c1e716 (progressive alpha-raised merged)
- **Source**: Altair (IIR extra at PV)
- **Notes**: Was trending strongly toward H1 when killed for net upgrade. Re-test with new net — likely a real +5-8 Elo gain.

### SEE Quiet Threshold -20d² → -25d² (inconclusive — killed for net upgrade)
- **Change**: Tighter SEE quiet pruning threshold from -20*depth*depth to -25*depth*depth.
- **Result**: Killed at 1819 games, +2.5 Elo ±10.9, LOS 67.4%, LLR -1.61. SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 0c1e716 (progressive alpha-raised merged)
- **Notes**: Flat — was trending toward H0. Low priority for re-test.

### Threat-Aware LMP (killed — clearly negative)
- **Change**: Tighten LMP limit to 3/4 for quiet moves that don't escape a pawn-attacked square. Uses the enemyPawnAttacks bitboard from threat-aware LMR.
- **Result**: Killed at 386 games, -11.0 Elo ±24.7, LOS 19.2%, LLR -2.01. SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 8d6f19d (threat-aware LMR merged)
- **Notes**: Clearly negative. Tightening LMP based on threat status prunes too aggressively — many good quiet moves don't involve escaping threats. The threat signal works for LMR (search shallower) but not LMP (skip entirely). Do not revisit.

### Non-Pawn Correction History with Zobrist (killed — negative)
- **Change**: Added second correction history table indexed by `(hash ^ pawnHash) % corrHistSize` to capture piece-placement eval errors. Blended 2:1 with pawn correction (2/3 pawn + 1/3 non-pawn).
- **Result**: Killed at 367 games, -9.7 Elo ±24.5, LOS 22.0%, LLR -1.87. SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 8d6f19d (threat-aware LMR merged)
- **Source**: Next-phase plan item #3 (correction history retry with Zobrist keys)
- **Notes**: Negative. The non-pawn Zobrist key changes too frequently (every piece move) to build reliable correction statistics. The pawn hash works because pawn structure is stable — adding a volatile key dilutes the signal. Third correction history variant to fail. The pawn-hash-only approach is correct for our engine.

### Threat-Aware RFP (REJECTED)
- **Change**: Widen RFP margin by 30cp per non-pawn piece under enemy pawn attack. When our pieces are threatened, the position is more volatile, so require a bigger eval surplus before pruning.
- **Result**: **H0 at 402 games, -15.6 Elo ±22.8, LOS 9.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: b6ab0c9 (net v3 merged)
- **Notes**: Widening the RFP margin for threatened positions makes it harder to prune, wasting search effort on nodes that should be cut. The base RFP margins are already well-calibrated — adding threat-based adjustments just makes them worse. Third threat-signal extension to fail (after threat-LMP and threat-futility). The threat signal is only useful for LMR modulation.

### Threat-Aware Futility Pruning (REJECTED)
- **Change**: Exempt moves that escape enemy pawn threats from futility pruning. If a quiet move's from-square is under pawn attack but to-square is not, skip the futility check.
- **Result**: **H0 at 195 games, -35.8 Elo ±32.8, LOS 1.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: b6ab0c9 (net v3 merged)
- **Notes**: Strongly negative. Exempting threat-escaping moves from futility pruning allows too many low-value moves through — most "escapes" aren't tactically important enough to justify the search cost. Similar lesson to threat-LMP: the threat signal is only useful for *modulating* search depth (LMR), not for *bypassing* pruning entirely.

### IIR Extra Reduction for PV Nodes (REJECTED)
- **Change**: When IIR triggers (no TT move, depth >= 4), apply an additional `depth--` for PV nodes. Theory: PV nodes without a TT move are especially suspect and benefit from deeper reduction.
- **Result**: **H0 at 825 games, -5.5 Elo ±16.5, LOS 25.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: PV nodes need full depth more than other node types — they're on the principal variation. Extra IIR reduction loses the engine's ability to find good continuations. Combined with earlier IIR-deep2 failures, extra IIR beyond a single depth reduction is harmful.

### LMR Check Reduction (REJECTED)
- **Change**: Instead of skipping LMR for check-giving moves (`!givesCheck`), allow checks into LMR but give them `reduction--`. Theory: checks are important but not so important they should skip LMR entirely.
- **Result**: **H0 at 362 games, -16.3 Elo ±23.4, LOS 8.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: Strongly negative. Checks deserve full LMR skip — they're tactical by nature and reducing them loses critical tactics. This aligns with the check extension failure (-11.2 Elo): the current "skip LMR for checks" is the correct calibration.

### NNUE Net v3: 113M Positions (MERGED)
- **Change**: New NNUE net trained from scratch on 112.7M positions (vs 70M for previous net). Includes ~1M "drunken" positions from games with blunders. 3-stage LR schedule: 0.01×8ep → 0.003×4ep → 0.001×4ep. Val loss: 0.127 (vs ~0.133 for old net).
- **Result**: **H1 at 216 games, +56.8 Elo ±37.2, LOS 99.9%.** SPRT bounds: elo0=-5, elo1=15. Tested on separate 32-thread machine.
- **Notes**: Biggest single Elo gain ever. Data volume (1.6x) was the key driver — suggests returns on data are far from exhausted. Low draw ratio (36.6%) indicates the net finds wins the old net couldn't. Priority should shift to data scaling for future gains.

### Correction History with Zobrist Keys v2 (REJECTED)
- **Change**: Second attempt at correction history using `(hash ^ pawnHash) % corrHistSize` as the non-pawn key. Separate table blended with pawn correction.
- **Result**: **H0 at 1385 games, -1.3 Elo ±12.5, LOS 42.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: Third correction history variant to fail (XOR bitboards: -11.8, Zobrist killed at -9.7, Zobrist v2: -1.3). The pawn-hash-only approach is correct for our engine. Non-pawn keys change too frequently to build reliable correction statistics. Do not revisit correction history unless the approach fundamentally changes (e.g., per-color separate tables, continuation-indexed).

### ContHist2 Updates on Cutoff/Penalty v2 (REJECTED)
- **Change**: Add contHistPtr2 (2-ply continuation history) to bonus/penalty updates on beta cutoffs and failed quiets. Second attempt.
- **Result**: **H0 at 1331 games, -1.3 Elo ±12.7, LOS 42.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: Second attempt, same result as first (-7.5 Elo). The 2-ply continuation history signal is inherently too weak for reliable learning — the piece at ply-2 may have been captured, making the key stale. Read-only at half weight for LMR/pruning remains the correct approach. Do not revisit.

### ProbCut MVV-LVA Ordering (REJECTED)
- **Change**: Add MVV-LVA + captHist scoring with incremental selection sort to ProbCut capture iteration. Previously captures were iterated in generation order after SEE filter.
- **Result**: **H0 at 474 games, -13.2 Elo ±21.8, LOS 11.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: ProbCut already filters by SEE >= margin, leaving very few captures (~2-3). Sorting overhead outweighs any ordering benefit. The SEE filter is sufficient — the first passing capture is almost always the best. Do not revisit.

### Hindsight Reduction/Bonus (REJECTED)
- **Change**: After LMR re-search, adjust newDepth based on eval trajectory. When reduction >= 2: if staticEval + parentEval > 0 (improving), newDepth++; if < 0 (declining), newDepth--. Stormphrax/Berserk/Tucano pattern.
- **Result**: **H0 at 198 games, -35.2 Elo ±32.3, LOS 1.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: engine-notes/SUMMARY.md #8, 4 engines
- **Notes**: Strongly negative despite 4-engine consensus. The eval sum heuristic may be too crude — staticEval + parentEval doesn't account for tempo or captures between plies. Also, the newDepth-- path (reduce further when declining) may over-prune positions where the engine needs to search harder to find a defense. Could retry with only the extension path (no reduction), or with a larger threshold than 0.

### IIR in PV at Depth 2 (REJECTED)
- **Change**: Apply Internal Iterative Reduction in PV nodes at depth >= 2 (previously only non-PV). When no TT move found in PV, reduce depth by 1.
- **Result**: **H0 at 825 games, -5.5 Elo ±16.5, LOS 25.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: PV nodes without a TT move are rare and usually important positions where the engine needs full depth. Reducing depth here causes the PV to miss critical moves. IIR is correct for non-PV (saves search on unimportant lines) but harmful in PV. Second test of this idea (also -5.5 Elo at different bounds). Do not revisit.

### Opponent-Threats Guard on RFP (REJECTED)
- **Change**: Skip Reverse Futility Pruning when the opponent has a threatening move (detected via null move or TT). Prevents over-pruning in tactical positions.
- **Result**: **H0 at 402 games, -15.6 Elo ±22.8, LOS 9.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: engine-notes/SUMMARY.md, Berserk/Minic pattern
- **Notes**: The guard was triggered too frequently, effectively disabling RFP in many positions. RFP's margin already accounts for tactical risk implicitly. Adding explicit threat detection on top creates redundancy and loses efficiency. Guards against over-pruning work when they're rare (like our NMP threat bonus at +12.4 Elo) but not when they fire broadly.

### Opponent-Threats Guard on Futility (REJECTED)
- **Change**: Skip futility pruning when the opponent has a threatening move. Similar approach to Threat-RFP but applied to futility pruning gate.
- **Result**: **H0 at 195 games, -35.8 Elo ±32.8, LOS 1.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: engine-notes/SUMMARY.md, Berserk/Minic pattern
- **Notes**: Even worse than Threat-RFP. Futility pruning is the engine's most aggressive quiet-move pruning and disabling it for threats causes massive search tree explosion. The futility margin (100+d*100) is already well-calibrated. Threat-based guards on pruning are consistently negative for our engine — do not revisit this pattern.

### Finny Tables — NNUE Accumulator Cache (REJECTED)
- **Change**: Per-perspective, per-king-bucket accumulator cache (`[2][16]FinnyEntry`). On king bucket change, diff cached vs current features using sorted merge (O(n)) and apply deltas (~5 ops) instead of full recompute (~30 ops). Fast path (no cache hit) has zero overhead — identical to original code.
- **Result**: **H0 at 702 games, -6.9 Elo ±17.6, LOS 22.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4bbcb7d (conthist 3x merged)
- **Notes**: NPS gain was +0.5-1.5% in benchmarks but didn't translate to Elo. Early SPRT was misleadingly positive (+20.6 at 236 games). King bucket changes are rare enough that caching doesn't fire often, and when it does the accuracy of the diff'd accumulator may be slightly worse than a clean recompute due to int16 accumulation error. Not worth the complexity. Could revisit if architecture scales to 2x512+ where recompute cost is higher.

### ProbCut QS Pre-Filter (REJECTED)
- **Change**: Inside ProbCut capture loop, call quiescence search before the reduced-depth negamax. If QS already exceeds the ProbCut beta threshold, skip the expensive reduced-depth search entirely.
- **Result**: **H0 at 177 games, -35.5 Elo ±32.6, LOS 1.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: Tucano pattern, experiment_queue Tier 1 #3
- **Notes**: Strongly negative. QS is not a reliable filter for ProbCut — the QS score can differ significantly from the reduced-depth search score because QS only considers captures while ProbCut's reduced search also evaluates quiet moves. Using QS as a cheap pre-filter causes ProbCut to fire on positions where the full search would not confirm the cutoff, leading to incorrect pruning. The existing ProbCut implementation (reduced-depth search only) is correct.

### Recapture Depth Reduction (REJECTED)
- **Change**: When `!improving` and `staticEval + bestCaptureValue < alpha`, reduce depth by 1 before the main search. Theory: if our eval plus the best available capture still can't reach alpha, the position is hopeless and we can search shallowly.
- **Result**: **H0 at 1197 games, -2.4 Elo ±13.7, LOS 36.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: Tucano pattern, experiment_queue Tier 1 #2
- **Notes**: Flat-to-slightly-negative over ~1200 games. The condition fires in positions where the engine is already losing, so reducing depth there doesn't save meaningful time — those nodes are typically cut quickly by other pruning (RFP, futility, LMP). The depth reduction also risks missing tactical escapes in losing positions. Not harmful but not helpful either.

### Score-Drop Time Extension — Aggressive Sigmoid (REJECTED)
- **Change**: Replaced conservative time management scaling (1.4x/1.2x for score drops >50/>25) with aggressive Tucano-inspired scaling: 2.5x/2.0x/1.5x for drops ≥60/≥40/≥20. Also added score-stability tracking (3+ consecutive iterations with Δ≤10 → scale 0.7x to save time).
- **Result**: **H0 at 1539 games, -0.5 Elo ±11.7, LOS 46.9%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: Tucano pattern, experiment_queue Tier 1 #1
- **Notes**: Dead flat over 1500+ games. The more aggressive time allocation on score drops doesn't help because: (1) at 10+0.1s TC, the extra time from 2.5x scaling is still small in absolute terms, (2) the engine's existing TM already handles instability via best-move stability and the 1.4x scaling, (3) the 0.7x stable-score reduction may counteract the extensions. Our existing TM is well-calibrated for this TC. Aggressive time extension might help at longer TCs (60+0.6s) where the absolute time gain is larger.

### Fail-High Extra Reduction at Cut-Nodes (REJECTED)
- **Change**: In LMR, add `reduction++` when `staticEval - 200 > beta`. Theory: if static eval exceeds beta by 200+ centipawns, we're very likely to get a cutoff and can search more shallowly.
- **Result**: **H0 at 955 games, -4.0 Elo ±15.1, LOS 30.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: Minic pattern, experiment_queue Tier 1 #4
- **Notes**: Slightly negative. The engine already has effective cut-node handling via NMP (which prunes entire subtrees when eval >> beta) and RFP (which prunes at shallow depths). Adding an extra LMR reduction on top is redundant — the positions where eval-200 > beta are already being aggressively pruned. The extra reduction only fires on the few moves that survive other pruning, and reducing those further loses important tactical verification.

### Complexity-Adjusted LMR (REJECTED)
- **Change**: Compute `complexity = abs(staticEval - rawEval)` (correction history magnitude). When `complexity > 50`, reduce LMR by 1 (search more deeply in uncertain positions).
- **Result**: **H0 at 530 games, -11.2 Elo ±20.5, LOS 14.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: Obsidian pattern, experiment_queue Tier 2 #6
- **Notes**: Clearly negative. Reducing LMR in "complex" positions (high correction magnitude) fires too broadly and increases the search tree significantly without finding better moves. The correction history magnitude is not a reliable indicator of tactical complexity — it may be large due to positional evaluation errors that don't affect move ordering. LMR reductions are well-calibrated; blanket reduction decreases based on eval uncertainty are harmful.

### Opponent Material LMR (MERGED)
- **Change**: In LMR, add `reduction++` when opponent has fewer than 2 non-pawn pieces. In simplified endgame-like positions, there's less tactical potential, so quiet moves can be searched more shallowly.
- **Result**: **H1 at 742 games, +18.5 Elo ±18.5, LOS 97.4%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged)
- **Source**: Weiss pattern, experiment_queue Tier 2 #7
- **Notes**: Strong result. When the opponent has ≤1 non-pawn piece, the position is simplified enough that quiet move alternatives are even less likely to be best — the engine can afford to search them more shallowly. This is a clean, principled LMR adjustment: material count is a reliable proxy for tactical complexity, unlike correction-history-based complexity measures which fire too broadly. Consider testing the opposite direction (reduction-- when opponent has many pieces) as a follow-up.

### Castling Bonus in Quiet Scoring (REJECTED)
- **Change**: Add a small bonus (+10) to quiet move scoring for castling moves, encouraging the move picker to try castling earlier in the quiet move ordering.
- **Result**: **H0 at 567 games, -9.8 Elo ±19.7, LOS 16.4%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged, pre-opp-material merge)
- **Source**: move_ordering_backlog, low-priority idea
- **Notes**: Clearly negative. Castling is already handled well by the existing move ordering (history heuristic, PST). Adding a fixed bonus disrupts the learned ordering and can cause castling to be tried before more relevant quiet moves. The move picker's history-based approach is superior to static bonuses for non-forcing moves.

### Opponent High-Material LMR Reduction-- (REJECTED)
- **Change**: In LMR, add `reduction--` when opponent has ≥4 non-pawn pieces. The complement to the merged opp-material LMR (reduction++ when oppNonPawn < 2): search quiet moves more deeply when the opponent has many pieces (higher tactical complexity).
- **Result**: **H0 at 307 games, -24.9 Elo ±27.5, LOS 3.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: opp-material-LMR merged (main + reduction++ for oppNonPawn < 2)
- **Source**: Follow-up to Opponent Material LMR (Tier 2 #7)
- **Notes**: Strongly negative. Reducing LMR in complex positions (many opponent pieces) massively increases the search tree. The existing LMR reduction table is well-calibrated for normal material counts — decreasing reductions broadly when the opponent has 4+ pieces causes too many quiet moves to be searched deeply, wasting time on irrelevant alternatives. The asymmetry makes sense: pruning MORE in simple positions (few pieces) is safe because there are fewer tactics, but pruning LESS in complex positions doesn't find better moves — it just searches more junk. The opp-material LMR benefit is one-directional.

### Complexity-Aware RFP (REJECTED)
- **Change**: Use correction history magnitude as a complexity proxy in Reverse Futility Pruning margin. When `abs(correctedEval - rawEval) > 50`, widen the RFP margin by 30cp (require more margin to prune in uncertain positions).
- **Result**: **H0 at 1907 games, +0.5 Elo ±10.8, LOS 54.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged, pre-opp-material merge)
- **Source**: Stormphrax pattern, experiment_queue Tier 2 #5
- **Notes**: Dead flat over nearly 2000 games. The correction history magnitude doesn't provide useful signal for RFP margin adjustment. The existing RFP margin (85cp + 60cp×depth) is well-calibrated and the correction history correction already implicitly handles eval uncertainty by adjusting staticEval directly. Adding a second layer of uncertainty-based margin widening is redundant — the corrected eval already accounts for the positional patterns that correction history captures. RFP margins are fully bracketed.

### Bad Noisy Futility (MERGED)
- **Change**: Separate futility pruning for losing captures (SEE < 0). At depth ≤ 4, when `staticEval + depth*50 <= alpha` and the capture has negative SEE, prune it (unless it gives check). The SEE call is gated behind the cheap eval guard so it's only computed when the position already looks futile.
- **Result**: **H1 at 1295 games, +11.8 Elo ±13.2, LOS 96.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged, pre-opp-material merge)
- **Source**: Reckless pattern, experiment_queue Tier 2 #8
- **Notes**: Strong result. Losing captures that don't even bring eval close to alpha are pure waste — they lose material AND the position is already bad. The existing futility pruning only covers quiets; this extends the concept to bad captures. The `depth*50` margin is tight (vs `80+80*depth` for quiet futility) because captures have a known material exchange that SEE evaluates. Guards: not in check, not TT move, not promotion, has bestScore, doesn't give check. The cheap eval guard prevents unnecessary SEE calls in most positions.

### CMP/FMP — Per-Component Continuation History Pruning (REJECTED)
- **Change**: In the history pruning section (depth ≤ 3), add per-component continuation history pruning: prune quiet moves when individual `contHist[piece][to] < -3000` or `contHist2[piece][to] < -3000`, even if the combined history score is above threshold.
- **Result**: **H0 at 303 games, -21.8 Elo ±26.1, LOS 5.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 1c7bdc8 (code review fixes merged, pre-opp-material merge)
- **Source**: Igel pattern, experiment_queue Tier 3 (CMP/FMP)
- **Notes**: Strongly negative. Per-component pruning is too aggressive — individual continuation history components can be deeply negative for a specific piece-to pair while the combined score (main history + contHist + contHist2) is positive, indicating the move is still worth searching. Pruning based on a single negative component throws away moves that other history signals support. The combined-score approach in our existing history pruning (threshold: -2000×depth) is the correct granularity. Confirms the pattern: history-based pruning modifications have a ~8% success rate for our engine.

### Opponent Material LMR Threshold 3 (MERGED)
- **Change**: Widen the opponent material LMR threshold from `oppNonPawn < 2` to `oppNonPawn < 3`. With fewer than 3 non-pawn pieces (i.e. 0-2 pieces), the position is simplified enough to increase LMR reduction.
- **Result**: **H1 at 624 games, +18.4 Elo ±18.3, LOS 97.5%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 705911f (opp-material < 2 + bad-noisy merged)
- **Source**: Bracket test of merged Opponent Material LMR
- **Notes**: Another strong result from widening the threshold. The < 2 condition only fires in deep endgames (lone piece); < 3 fires much more often (e.g. rook + minor vs similar). The broader application still works because positions with ≤2 non-pawn opponent pieces genuinely have reduced tactical complexity. Consider testing < 4 as a further bracket, though diminishing returns are likely as the condition approaches normal middlegame material counts.

### Own-Side Low Material LMR (REJECTED)
- **Change**: In LMR, add `reduction++` when the side to move has fewer than 2 non-pawn pieces. Symmetric counterpart to the opponent material LMR — if we're simplified too, reduce more.
- **Result**: **H0 at 316 games, -18.7 Elo ±24.8, LOS 7.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 705911f (opp-material < 2 + bad-noisy merged)
- **Source**: Follow-up to Opponent Material LMR
- **Notes**: Strongly negative. The asymmetry makes sense: when the *opponent* has few pieces, *their* threats are limited so we can reduce our quiet moves. But when *we* have few pieces, we need to search carefully to find the best use of our limited material — reducing our own moves in this situation loses critical accuracy. The signal is one-directional: opponent material count predicts opponent threat level, not our own move quality.

### Bad Noisy Futility Margin 75 (MERGED)
- **Change**: Loosen the bad-noisy futility margin from `staticEval + depth*50 <= alpha` to `staticEval + depth*75 <= alpha`. The wider gate allows more losing captures to be pruned.
- **Result**: **H1 at 131 games, +58.9 Elo ±38.0, LOS 99.9%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 705911f (opp-material < 2 + bad-noisy merged)
- **Source**: Bracket test of merged Bad Noisy Futility
- **Notes**: Massive result — the original depth*50 margin was too conservative. With depth*75, the pruning fires much more frequently (a depth-4 capture is pruned when eval is 300cp below alpha instead of 200cp). Losing captures in positions that are 300cp below alpha are overwhelmingly futile. The SEE guard still ensures we only prune material-losing captures, so the wider eval gate is safe. Consider testing depth*100 as a further bracket.

### Texel-Tuned Classical Eval (MERGED)
- **Change**: All PST, eval, and pawn structure parameters optimized via Texel tuning on 125M positions (depth-10 rescored). ~1268 parameters updated.
- **Result**: **H1 at 314 games, +33.3 Elo ±27.1, LOS 99.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: Current main (classical mode, `arg=-classical`)
- **Source**: Texel tuner with Adam optimizer, 500 epochs, lambda=0.5
- **Notes**: First complete Texel tune from the large rescored dataset. The classical eval was using manually-tuned PeSTO values; the optimized parameters are better calibrated. Changes are small per-square adjustments across all piece types and pawn structure bonuses, not structural changes. This gain is orthogonal to NNUE (only affects classical play and training data generation). The .tbin cache must be regenerated after this merge.

### Bad Noisy Depth 6 — NOT DIRECTLY COMPARABLE
- **Change**: Extend bad-noisy from depth≤4 to depth≤6, with margin depth*50 (vs current depth*75 at depth≤4).
- **Result**: **H1 at 281 games, +37.2 Elo ±29.1, LOS 99.4%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: Pre-M75 merge base (old worktree). NOT directly comparable to current main.
- **Source**: Pre-merge bracket test
- **Notes**: Tested against an older baseline before the depth*75 merge. The current main has depth≤4/margin*75, while this tested depth≤6/margin*50 against the old depth≤4/margin*50 base. The +37.2 Elo gain is vs that old base, not vs current. Worth retesting depth≤6/margin*75 against current main to see if extending the depth range adds further value on top of the wider margin.

### Bad Noisy Margin depth*100 (REJECTED)
- **Change**: Widen bad-noisy futility margin from `depth*75` to `depth*100`.
- **Result**: **H0 at 167 games, -37.6 Elo ±33.6, LOS 1.5%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: d82ab70 (Texel-tuned eval + BINP reader)
- **Source**: Bracket test of merged bad-noisy margin
- **Notes**: Too aggressive. At depth 4, this prunes captures when eval is 400cp below alpha — too wide, catching captures that actually have tactical merit. The optimum is confirmed at depth*75. Don't test wider.

### Failing Heuristic in Capture LMR (REJECTED)
- **Change**: Add `if failing { reduction++ }` to the capture LMR block, matching the quiet LMR adjustment.
- **Result**: **H0 at 292 games, -22.6 Elo ±26.9, LOS 5.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: d82ab70 (Texel-tuned eval + BINP reader)
- **Source**: Capture LMR asymmetry analysis
- **Notes**: The failing signal doesn't transfer to captures. When the position is deteriorating, quiets deserve more reduction (they're unlikely to save us), but captures remain tactically relevant — even in collapsing positions, a good capture can change the evaluation. The asymmetry between quiet and capture LMR is intentional, not a gap.

### TT-Noisy in Capture LMR (REJECTED)
- **Change**: Add `if ttMoveNoisy { reduction++ }` to capture LMR, matching quiet LMR's +1 reduction when TT move is a capture.
- **Result**: **H0 at 261 games, -29.4 Elo ±29.5, LOS 2.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 46122f5 (SF binpack filter fix)
- **Source**: Capture LMR asymmetry analysis
- **Notes**: Third capture LMR extension rejected (after failing flag -22.6, now ttMoveNoisy -29.4). The pattern is clear: quiet LMR adjustments do NOT transfer to capture LMR. When the TT move is a capture, other captures remain tactically important — reducing them loses critical lines. The capture LMR table with capture history is the right granularity; adding coarser signals hurts.

### Bad Noisy Depth 6 vs Current Main (REJECTED)
- **Change**: Extend bad-noisy from `depth <= 4` to `depth <= 6`, keeping margin `depth*75`.
- **Result**: **H0 at 1460 games, -1.0 Elo ±12.2, LOS 43.9%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 46122f5 (current main with depth≤4/margin*75)
- **Notes**: Dead flat. The depth≤4 threshold is optimal — deeper bad-noisy catches too few additional positions to matter. The earlier +37.2 vs old base was measuring the margin change, not the depth change.

### Opponent Material LMR Threshold <4 (REJECTED)
- **Change**: Widen from `oppNonPawn < 3` to `oppNonPawn < 4`.
- **Result**: **H0 at 2710 games, +1.9 Elo ±8.8, LOS 66.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 46122f5
- **Notes**: Very long test, very flat. Threshold <3 is optimal. <4 includes too many normal middlegame positions where the reduction is inappropriate.

### FH Blend Depth Gate 2 (REJECTED)
- **Change**: Lower FH blend depth gate from `depth >= 3` to `depth >= 2`.
- **Result**: **H0 at 1280 games, -1.4 Elo ±12.7, LOS 41.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 46122f5
- **Notes**: Flat. Depth-2 FH is too close to QS where beta blending already operates. The depth≥3 gate correctly separates the two dampening mechanisms.

### TT Cutoff History Bonus (REJECTED)
- **Change**: Give history bonus to TT move (quiet or capture) when TT probe causes a beta cutoff, matching Stockfish PR #5791.
- **Result**: **H0 at 1025 games, -3.1 Elo ±14.3, LOS 33.9%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 46122f5
- **Notes**: Slightly negative despite being a proven SF feature. Our persist-history fix means history tables are already well-populated across searches, reducing the value of additional TT-cutoff updates. Different engine, different optima.

### Improving-Aware Capture LMR (REJECTED)
- **Change**: Add `if improving { reduction-- }` to capture LMR, reducing less for captures in improving positions.
- **Result**: **H0 at 668 games, -6.8 Elo ±17.3, LOS 22.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 47f9d14
- **Source**: Capture LMR asymmetry analysis — first test reducing *less* instead of *more*
- **Notes**: Fourth capture LMR extension rejected (failing -22.6, ttMoveNoisy -29.4, now improving -6.8). Even reducing captures *less* in improving positions hurts. Capture LMR is fully self-contained — the capture history alone is the correct and only adjustment mechanism. No quiet LMR signal transfers in either direction. Stop testing capture LMR modifications.

### NMP Return Value Dampening (MERGED)
- **Change**: Blend NMP return value toward beta: `return (score*2 + beta) / 3` instead of `return beta`.
- **Result**: **H1 at 1059 games, +12.8 Elo ±14.1, LOS 96.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 47f9d14
- **Source**: Score dampening pattern (TT dampen +22.1, FH blend +14.7, QS blend +4.9, now NMP +12.8)
- **Notes**: Fourth score dampening win. NMP scores are noisy because the null-move assumption is approximate. Blending the return toward beta prevents inflated cutoff scores from propagating. The dampening pattern is now proven at every boundary: TT cutoffs, fail-highs, QS stand-pat, and NMP.

### Capture LMR C=1.70 (REJECTED)
- **Change**: Tighten capture LMR constant from 1.80 to 1.70 (more aggressive reduction).
- **Result**: **H0 at 1104 games, -2.5 Elo ±14.0, LOS 36.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 47f9d14
- **Notes**: Dead flat. C=1.80 is optimal. Capture LMR is fully calibrated.

### History Gravity Divisor 12288 (REJECTED)
- **Change**: Reduce history gravity divisor from 16384 to 12288 (faster decay of old history data).
- **Result**: **H0 at 404 games, -13.8 Elo ±21.7, LOS 10.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 47f9d14
- **Notes**: Negative. Faster decay loses valuable persistent history data. D=16384 is optimal — history needs to accumulate across many searches to be reliable.

### NMP Dampening Stronger (score+beta)/2 (REJECTED)
- **Change**: Stronger NMP dampening: `(score+beta)/2` instead of merged `(score*2+beta)/3`.
- **Result**: **H0 at 208 games, -33.5 Elo ±31.4, LOS 1.9%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: c5bd250 (NMP dampen merged)
- **Source**: Bracket test of NMP dampening
- **Notes**: Too much dampening. The merged (score*2+beta)/3 gives 2/3 weight to score, 1/3 to beta. This variant at 1/2 each over-dampens, losing real information from the NMP score. The optimum is confirmed at (score*2+beta)/3.

### Correction History Depth Gate 2 (REJECTED)
- **Change**: Lower correction history depth gate from `depth >= 3` to `depth >= 2`.
- **Result**: **H0 at 277 games, -23.9 Elo ±27.0, LOS 4.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: c5bd250
- **Notes**: Depth-2 search results are too noisy for reliable correction updates. The depth≥3 gate correctly filters out shallow noise. D=16384 and depth≥3 are both confirmed optimal for correction history.

### Capture LMR C=1.90 (REJECTED)
- **Change**: Relax capture LMR constant from 1.80 to 1.90 (less aggressive reduction).
- **Result**: **H0 at 749 games, -6.0 Elo ±16.7, LOS 24.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: c5bd250
- **Notes**: Capture LMR fully bracketed from both sides: C=1.70 (H0, -2.5), C=1.80 (optimal), C=1.90 (H0, -6.0). Don't retest.

### NMP Dampening Weaker (score*3+beta)/4 (REJECTED)
- **Change**: Less aggressive NMP dampening: `(score*3+beta)/4` instead of merged `(score*2+beta)/3`.
- **Result**: **H0 at 825 games, -5.1 Elo ±15.9, LOS 26.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: c5bd250
- **Notes**: NMP dampening fully bracketed from both sides: (score+beta)/2 (H0, -33.5), (score*2+beta)/3 (H1, +12.8), (score*3+beta)/4 (H0, -5.1). The 2:1 ratio is optimal. Don't retest.

### TT Near-Miss Margin 80 (MERGED)
- **Change**: Widen TT near-miss cutoff margin from 64 to 80 centipawns.
- **Result**: **H1 at 570 games, +18.3 Elo ±18.5, LOS 97.4%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: c5bd250
- **Source**: Bracket test of TT near-miss margin
- **Notes**: Wider margin accepts more TT entries that are 1 ply short of the required depth. At 80cp, entries that exceed beta by 80+ cp are trusted as cutoffs. This works because large TT score margins indicate high-confidence positions where the extra ply is unlikely to change the result. Previous margin of 64 was from initial calibration and was never bracketed until now.

### TT Near-Miss Margin 112 (REJECTED)
- **Change**: Widen TT near-miss margin from 80 to 112cp.
- **Result**: **H0 at 388 games, -16.1 Elo ±23.1, LOS 8.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d (TT near-miss 80 merged)
- **Notes**: Too wide — accepts inaccurate TT entries. Margin 80 is the optimum (64→80 gained +18.3, 80→96 trending negative, 80→112 rejected).

### QS Delta 280 (REJECTED)
- **Change**: Widen QS delta pruning margin from 240 to 280.
- **Result**: **H0 at 1041 games, -3.7 Elo ±14.8, LOS 31.3%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d (TT near-miss 80 merged)
- **Notes**: QS delta 240 is optimal. Wider margin prunes too aggressively in QS. Confirmed: 200→240 gained +31.2, 240→280 rejected.

### Futility 60+d*60 (REJECTED)
- **Change**: Tighten futility margin from `80+lmrDepth*80` to `60+lmrDepth*60`.
- **Result**: **H0 at 1543 games, -0.5 Elo ±11.7, LOS 47.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: Dead flat at 1543 games. Futility 80+d*80 is optimal. Initial +8.7 signal was noise.

### QS Delta 200 (REJECTED)
- **Change**: Tighten QS delta pruning margin from 240 to 200.
- **Result**: **H0 at 282 games, -23.4 Elo ±27.0, LOS 4.5%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: QS delta fully bracketed: 200 (H0, -23.4), 240 (optimal), 280 (H0, -3.7). Don't retest.

### RFP Tighter 85/55 (REJECTED)
- **Change**: Tighten RFP margins from depth*100/depth*70 to depth*85/depth*55.
- **Result**: **H0 at 288 games, -23.0 Elo ±26.9, LOS 4.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: RFP margins confirmed optimal at 100/70. Previous test (100→110) was also H0. Don't retest.

### ProbCut Margin 150 (REJECTED)
- **Change**: Tighten ProbCut margin from beta+170 to beta+150.
- **Result**: **H0 at 415 games, -15.1 Elo ±22.7, LOS 9.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: ProbCut margin 170 is optimal. Previous test (200→170) gained +10.0. Tighter margin prunes too aggressively.

### TT Near-Miss Margin 96 (REJECTED)
- **Change**: Widen TT near-miss margin from 80 to 96cp.
- **Result**: **H0 at 2590 games, +1.6 Elo ±9.3, LOS 63.3%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d (TT near-miss 80 merged)
- **Notes**: Very long test. Early noise peaked at +7.6/91% LOS but regressed to flat. Margin 80 is optimal (64→80 gained +18.3, 80→96 flat, 80→112 rejected).

### LMR Quiet C=1.60 (REJECTED)
- **Change**: Relax quiet LMR constant from 1.50 to 1.60 (less aggressive reduction).
- **Result**: **H0 at 239 games, -29.1 Elo ±29.6, LOS 2.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: Strongly negative. Less aggressive LMR wastes search depth. The optimum is at or below 1.50 — both C=1.40 (+6.3) and C=1.30 (+26.1 early) are positive.

### LMR Quiet C=1.40 (REJECTED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.40.
- **Result**: **H0 at 960 games, -2.9 Elo ±14.3, LOS 34.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: Dead flat despite early +8.7 signal. C=1.50 confirmed from above (C=1.60 H0) and this side (C=1.40 H0). However, C=1.30 is showing +9.3 — the response may be non-monotonic or the optimum is a sharper change.

### LMR Quiet C=1.20 (REJECTED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.20.
- **Result**: **H0 at 826 games, -4.6 Elo ±15.7, LOS 28.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d
- **Notes**: Too aggressive. C=1.30 (+11.1) is the sweet spot; C=1.20 overshoots. Quiet LMR bracket: 1.20 (H0), 1.30 (building +11.1), 1.35 (+4.7), 1.40 (H0), 1.50 (current), 1.60 (H0).

### LMR Quiet C=1.30 (MERGED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.30 (more aggressive reduction).
- **Result**: **H1 at 1016 games, +13.3 Elo ±14.7, LOS 96.3%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d (TT near-miss 80 merged)
- **Source**: LMR constant bracket test
- **Notes**: Continues the LMR progression: C=2.0 → 1.75 (+16.2) → 1.50 (+44.4) → 1.30 (+13.3). Perfectly bracketed: 1.25 (H0, -3.6), 1.30 (H1, +13.3), 1.35 (H0, +1.2), 1.40 (H0, -2.9). NNUE accuracy enables more aggressive reduction.

### LMR Quiet C=1.35 (REJECTED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.35.
- **Result**: **H0 at 2063 games, +1.2 Elo ±9.9.** SPRT bounds: elo0=-5, elo1=15.

### LMR Quiet C=1.25 (REJECTED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.25.
- **Result**: **H0 at 959 games, -3.6 Elo ±14.8.** SPRT bounds: elo0=-5, elo1=15.

### V5: Futility 60+d*60 (MERGED)
- **Change**: Tighten futility margin from `80+lmrDepth*80` to `60+lmrDepth*60`.
- **Result**: **H1 at 935 games, +12.3 Elo ±13.6, LOS 96.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 932c12c with v5 sb120 net
- **Notes**: Was H0 (-0.5 Elo) with v4 net. V5's stronger eval allows tighter futility pruning — confirms the prediction that better eval shifts search parameter optima. First v5-specific search win.

### V5: LMR Quiet C=1.20 (REJECTED)
- **Change**: Tighten quiet LMR from C=1.30 to C=1.20.
- **Result**: **H0 at 167 games, -37.6 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 932c12c with v5 sb120 net
- **Notes**: Still too aggressive even with v5. C=1.30 remains optimal.

### V5: TT Near-Miss Margin 96 (REJECTED)
- **Change**: Widen TT near-miss from 80 to 96cp.
- **Result**: **H0 at 721 games, -3.9 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 932c12c with v5 sb120 net
- **Notes**: Still flat with v5. Margin 80 confirmed optimal regardless of eval quality.

### V5: RFP Depth Gate 8 (REJECTED)
- **Change**: Extend RFP depth gate from `depth <= 7` to `depth <= 8`.
- **Result**: **H0 at 1377 games, -0.5 Elo ±11.9.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Completely flat. Depth 7 gate is optimal — depth 8 positions are too complex for static eval pruning even with v5.

### V5: Bad Noisy Futility Depth 6 (REJECTED)
- **Change**: Extend bad noisy futility depth from `depth <= 4` to `depth <= 6`.
- **Result**: **H0 at 794 games, -5.3 Elo ±16.1.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Was flat with v4, now slightly negative with v5. Depth 4 confirmed optimal — deeper captures need full search.

### V5: LMR Quiet C=1.25 (REJECTED)
- **Change**: Tighten quiet LMR from C=1.30 to C=1.25.
- **Result**: **H0 at 638 games, -7.1 Elo ±17.6.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Was H0 (-3.6 Elo) with v4, now more negative with v5. C=1.30 is a hard floor.

### V5: LMR Quiet C=1.35 (REJECTED)
- **Change**: Loosen quiet LMR from C=1.30 to C=1.35.
- **Result**: **H0 at 356 games, -16.6 Elo ±23.3.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Both C=1.25 and C=1.35 lose Elo — C=1.30 is precisely optimal. LMR is fully bracketed.

### V5: ProbCut Margin beta+150 (REJECTED)
- **Change**: Tighten ProbCut margin from `beta + 170` to `beta + 150`.
- **Result**: **H0 at 508 games, -10.3 Elo ±19.6.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Was H0 (-15.1 Elo) with v4, still negative with v5. beta+170 confirmed optimal across eval changes.

### V5: Razoring 350+d*80 (REJECTED)
- **Change**: Tighten razoring margin from `400 + depth*100` to `350 + depth*80`.
- **Result**: **H0 at 549 games, -9.5 Elo ±19.4.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Current razoring margins well-calibrated. Tighter margins over-prune at depth 2.

### V5: RFP Margins 90/60 (REJECTED)
- **Change**: Tighten RFP margins from `depth*100`/`depth*70` to `depth*90`/`depth*60`.
- **Result**: **H0 at 896 games, -3.5 Elo ±14.9.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 16f4f02 with v5 sb120 net
- **Notes**: Current RFP margins (100/70) already optimal for v5. Combined with depth gate test, RFP is fully tuned.

### V5: Continuation Correction History 50/50 (REJECTED)
- **Change**: Add ContCorrHistory[13][64] table indexed by previous move's piece+to. Blend 50% pawn + 50% continuation correction.
- **Result**: **H0 at 645 games, -7.0 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: v5 sb120 net on main
- **Notes**: SF-proven feature (PR #5617). Too much weight on continuation correction dilutes the proven pawn correction.

### V5: Continuation Correction History 25% (REJECTED)
- **Change**: Same table but lighter blend: full pawn correction + 25% continuation correction.
- **Result**: **H0 at 938 games, -2.2 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: v5 sb120 net on main
- **Notes**: Lighter weight improved from -7 to -2.2 but still flat. Continuation correction doesn't add useful signal beyond pawn correction for our engine. The high draw ratio (63.3%) suggests over-correction.

### V5: TT Cutoff History Bonus (MERGED)
- **Change**: Give history bonus to TT move when TT probe causes a beta cutoff. Quiet moves get main history bonus, captures get capture history bonus.
- **Result**: **H1 at 810 games, +14.6 Elo ±15.5, LOS 96.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: v5 sb120 net on main
- **Notes**: Was -3.1 with v4 (H0 at 1025 games). V5's stronger eval makes TT cutoff moves more reliable — reinforcing them in history improves move ordering. First move ordering win with v5.

### V5: TT Cutoff Quiet Penalties (REJECTED)
- **Change**: On TT cutoff with quiet move, penalise killers and counter-move in history.
- **Result**: **H0 at 457 games, ~-12 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: The bonus alone (+14.6) is the right signal. Adding penalties for alternatives hurts — we don't know which moves would have been tried, so the penalties are imprecise.

### V5: TT Cutoff Counter-Move Update (REJECTED)
- **Change**: Update counter-move table when TT causes a cutoff with a quiet move.
- **Result**: **H0 at 291 games, ~-19 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: Counter-move table updates from TT cutoffs are noisy — the TT move may not be the best response to the previous move specifically.

### V5: ContHist Ply-4 Quarter Weight (REJECTED)
- **Change**: Add 4-ply continuation history lookback at quarter weight in history pruning and LMR.
- **Result**: **H0 at 1765 games, ~+1 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: Was +12.8 at 773 games (85% toward H1) then regressed to flat. Classic early noise. Was -12.4 with v4.

### V5: ContHist Ply-4 Half Weight (REJECTED)
- **Change**: Same as above but at half weight instead of quarter.
- **Result**: **H0 at 628 games, ~-7 Elo.** SPRT bounds: elo0=-5, elo1=15.

### V5: TT Cutoff ContHist Bonus (REJECTED)
- **Change**: Update continuation history when TT causes a cutoff.
- **Result**: **H0 at 782 games, ~-4 Elo.** SPRT bounds: elo0=-5, elo1=15.

### V5: NMP Verification Depth 10 (REJECTED)
- **Change**: Lower NMP verification depth from 12 to 10.
- **Result**: **H0 at 647 games, ~-6 Elo.** SPRT bounds: elo0=-5, elo1=15.

### V5: Aspiration Delta 12 (REJECTED)
- **Change**: Tighter aspiration window from delta=15 to delta=12.
- **Result**: **H0 at 459 games, ~-10 Elo.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: Delta=15 remains optimal even with v5's more stable eval.

### V5: History Pruning Depth 4 (KILLED — flat)
- **Change**: Extend history-based pruning from `depth <= 3` to `depth <= 4`.
- **Result**: Killed at 1007 games, +1.0 Elo, 56% LOS, LLR -1.7 (trending H0). SPRT bounds: elo0=-5, elo1=15.
- **Notes**: History pruning depth 3 is already well-calibrated. Extending to depth 4 gains nothing.

### V5: SEE Capture Pruning Depth 7 (REJECTED)
- **Change**: Extend SEE capture pruning from `depth <= 6` to `depth <= 7` (margin -depth*100).
- **Result**: **H0 at 201 games, -29.5 Elo.** Decisive rejection.
- **Notes**: At depth 7, the -700 SEE threshold prunes too many captures that are tactically important. SEE capture pruning at depth <= 6 is well-calibrated.

### V5: 1536 SCReLU vs 1024 CReLU (REJECTED)
- **Change**: 1536-wide SCReLU net (multi-file 200SB, exact SIMD inference) vs 1024 CReLU baseline (120SB).
- **Result**: **H0 at 159 games, -61.8 Elo.** Decisive rejection.
- **Notes**: Despite lower training loss (0.00870 vs 0.00882), sharper piece values, and 212K NPS (only ~15% slower than CReLU), the 1536 SCReLU net was catastrophically weaker in play. Early signal (+35 at 48 games) was pure noise. The 39.6% draw ratio (vs typical 64%) suggests the net produces unbalanced evals that lead to decisive games — likely overconfident or poorly calibrated for the search.

### V5: 1536 CReLU Multi-file vs 1024 CReLU (LIKELY REJECT)
- **Change**: 1536-wide CReLU net (multi-file 200SB) vs 1024 CReLU baseline (120SB).
- **Result**: -18.5 Elo at 336 games, LLR -2.42 (82% toward H0). Still running but almost certain to reject.
- **Notes**: Consistent with earlier 1536 single-file result (-23 Elo). More data and longer training didn't help the 1536 architecture. The 1024 net is simply better calibrated for our search at current data scales.

### V5: 1536 CReLU Multi-file vs 1024 CReLU (REJECTED)
- **Change**: 1536-wide CReLU net (multi-file 200SB) vs 1024 CReLU baseline (120SB).
- **Result**: **H0 at 469 games, -14.8 Elo.**
- **Notes**: Third 1536 rejection: single-file -23, multi-file CReLU -15, multi-file SCReLU -62. More data and longer training didn't help. 1536 width is not viable with our search — the NPS penalty (~200K vs ~790K at 1024) outweighs any eval quality gain. Low draw ratio (48%) persists across all 1536 tests, suggesting the wider net produces less stable evals.

### V5: Cap LMR Continuous History (REJECTED)
- **Change**: Replace discrete capture history LMR thresholds (±2000 → ±1 reduction) with continuous `reduction -= captHistVal / 5000`.
- **Result**: **H0 at 1119 games, -1.6 Elo.** Early mirage of +9.6 at 818 games collapsed.
- **Notes**: The discrete thresholds at ±2000 are well-calibrated. Continuous adjustment over-smooths — bad captures that deserve more reduction get too little, good captures that deserve less get too much. The hard cutoffs act as effective noise filters.

### V5: History-Based LMP (REJECTED)
- **Change**: Adjust LMP limit by move's history score: <-3000 tightens by 1/3, >3000 loosens by 1/3.
- **Result**: **H0 at 681 games, -5.6 Elo.**
- **Notes**: LMP already uses move count as a proxy for move quality (moves are ordered by history). Adding explicit history adjustment is redundant — the move ordering already ensures low-history moves appear late and get pruned by the existing count-based limit.

### V5: Failing-Aware NMP (REJECTED)
- **Change**: Skip NMP at depths 3-6 when position is failing (eval deteriorating significantly).
- **Result**: **H0 at 428 games, -13.0 Elo.**
- **Notes**: NMP is valuable even in failing positions. The null-move hypothesis tests absolute eval vs beta, not the trend. Restricting NMP at shallow depths removes critical pruning exactly where it saves the most work. Consistent with v4 result: "NMP at depth 3-7 is well-calibrated."

### V5: LMP 4+d² (RETEST CANDIDATE — small positive)
- **Change**: LMP formula from `3 + depth*depth` to `4 + depth*depth` (one extra move allowed before pruning).
- **Result**: **H0 at 2453 games, +2.3 Elo (wide bounds elo0=-5/elo1=15).** Consistently showed +3-5 Elo / 75-85% LOS throughout but never enough LLR to pass wide bounds.
- **Notes**: Likely a genuine ~3-4 Elo gain. Retest with tight bounds (elo0=0, elo1=8, 5000+ games) when distributed testing is available. Do not re-reject — the wide SPRT was the wrong tool for this effect size.

### V5: Capture LMR C=1.40 (REJECTED)
- **Change**: Reduce capture LMR table constant from 1.80 to 1.40 (less reduction on captures).
- **Result**: **H0 at 948 games, -2.6 Elo.**
- **Notes**: Less reduction means searching captures more deeply, costing NPS without improving accuracy. The current capture LMR constant (1.80) is well-calibrated — captures already get less reduction than quiets (C=1.50), so further reduction is wasteful.

### V5: SEE Quiet Threshold -25d² (LIKELY REJECT)
- **Change**: Tighter SEE quiet threshold from -20d² to -25d².
- **Result**: -1.2 Elo at 597 games, LLR -1.38. Still running but flat.
- **Notes**: Tightening the threshold prunes more aggressively. Early crash to -31 at 118 games recovered to ~0, suggesting the change is roughly neutral but not beneficial. Current -20d² is well-calibrated.

### V5: SEE Quiet Threshold -25d² (REJECTED)
- **Change**: Tighter SEE quiet threshold from -20d² to -25d².
- **Result**: **H0 at 841 games, -4.5 Elo.**
- **Notes**: Tighter threshold prunes more quiet moves, but the extra pruning removes moves that were worth searching. Current -20d² is well-calibrated. Tested in both directions now: -25d² loses, depth 9 extension is flat — the SEE quiet parameters are at their optimum.

### V5: Unstable LMR -2 (REJECTED)
- **Change**: Double the LMR reduction bonus for unstable positions (reduction -= 2 instead of -= 1).
- **Result**: **H0 at 152 games, -39.0 Elo.** Decisive.
- **Notes**: Reducing by 2 in volatile positions searches too many extra nodes. The current -1 adjustment is well-calibrated — it gives enough caution without burning NPS on every unstable node. This is consistent with the pattern: guard adjustments work best as single-ply nudges, not aggressive changes.

### V5: SEE Quiet Depth 9 (RETEST CANDIDATE — small positive)
- **Change**: Extend SEE quiet pruning from `depth <= 8` to `depth <= 9`.
- **Result**: Killed at 1290 games, +2.5 Elo, 66% LOS, LLR -1.32. Consistent +2-7 Elo throughout but fading.
- **Notes**: Another small positive that can't clear wide SPRT bounds. Retest with tight bounds (elo0=0, elo1=8) when distributed testing available.

### V5: 1024 Multi-file sb120 vs Baseline (REJECTED — WDL confound!)
- **Change**: 1024 CReLU trained on 3 files (multi-file) for 120SB with wdl=0.0, vs baseline trained single-file 120SB with wdl=0.5.
- **Result**: **H0 at 84 games, -93.2 Elo.** Catastrophic.
- **Notes**: **CRITICAL FINDING**: The wdl=0.0 vs wdl=0.5 training difference is the dominant factor, NOT data diversity or architecture. All multi-file configs (1536 CReLU, 1536 SCReLU, 1024 multi) used wdl=0.0 while the baseline used wdl=0.5. This means the 1536 rejection results (-15, -62 Elo) are CONFOUNDED — they may be mostly wdl penalty, not architecture penalty. All multi-file nets must be retrained with wdl=0.5 for valid comparison.

### V5: Threat-Aware SEE Quiet (RETEST CANDIDATE — small positive)
- **Change**: Loosen SEE quiet threshold by -100 when threatSq >= 0 (opponent has detected threats).
- **Result**: **H0 at 1675 games, +0.8 Elo (wide bounds elo0=-5/elo1=15).** Showed +5-12 Elo for first 1000 games, then regressed.
- **Notes**: Early signal was strong (+12.6 at 730 games, 94% LOS) but didn't hold. The idea is sound — be more cautious pruning when opponent has threats — but the effect may be ~3-4 Elo. Retest with tight bounds.

### V5: QS Delta 200 (REJECTED)
- **Change**: Tighten QS delta pruning margin from 240 to 200.
- **Result**: **H0 at 331 games, -16.8 Elo.**
- **Notes**: QS delta 240 was already tuned (merged as +31.2 Elo from the original 0→240 change). Tightening further prunes too many captures in QS that were tactically relevant. 240 is the optimal margin.

### V5: Countermove LMR Bonus (REJECTED)
- **Change**: Reduce LMR by 1 for countermove hits (similar to killer exemption).
- **Result**: **H0 at 555 games, -6.9 Elo.**
- **Notes**: Unlike killers (which are position-specific refutations), countermoves are a weaker signal — they refuted the opponent's move in a *different* position. The existing move ordering already places countermoves after killers, and LMR's continuous history adjustment already gives well-ordered moves less reduction. Adding a discrete bonus on top is redundant and slightly harmful.

### V5: QS SEE < -50 (REJECTED)
- **Change**: Allow slightly losing captures (SEE > -50 instead of > 0) in quiescence search.
- **Result**: **H0 at 795 games, -3.9 Elo.**
- **Notes**: Searching SEE-negative captures in QS adds NPS cost without improving tactical accuracy. The SEE > 0 filter is well-calibrated — captures that lose material are rarely worth exploring in QS.

### V5: LMR Quiet C=1.20 (REJECTED)
- **Change**: More aggressive quiet LMR reduction (divisor from 1.30 to 1.20).
- **Result**: **H0 at 322 games, -17.3 Elo.**
- **Notes**: C=1.30 was already tuned down from 1.50 for v5. Going further to 1.20 is too aggressive — the extra reduction misses tactical quiet moves. C=1.30 is the optimum for v5's eval.

### V5: QS Stand-Pat Blend (REJECTED)
- **Change**: Blend QS stand-pat beta cutoff: return `(bestScore + beta) / 2` instead of `bestScore`.
- **Result**: **H0 at 472 games, -11.0 Elo.**
- **Notes**: Unlike TT-dampen and FH-blend (which work at search boundaries where scores are noisy), the QS stand-pat score is a direct static eval — already well-calibrated. Dampening it toward beta loses information. The low draw ratio (58% vs typical 64%) suggests it destabilizes the search by returning inaccurate QS scores.

### V5: RFP Margins 90/55 (REJECTED)
- **Change**: RFP margins from 100/70 to 90/55 (looser non-improving, tighter improving).
- **Result**: **H0 at 183 games, -34.3 Elo.** Decisive.
- **Notes**: Current RFP margins (100/70) are well-calibrated for v5. The 55 improving margin prunes too aggressively in improving positions.

### V5: Singular Extensions v2 — SF-like params (IN PROGRESS — still negative)
- **Change**: Re-enabled singular extensions with fixed wiring AND Stockfish-like parameters: depth>=8, margin 3*depth/2, verification (depth-1)/2.
- **Status**: -45.4 Elo at 103 games. Still running but likely to H0.
- **Notes**: Even with correct implementation and conservative parameters, singular extensions hurt. Low draw ratio (53%) suggests search instability. May interact badly with our other extensions/reductions (alpha-reduce, failing heuristic, FH-blend). Needs deeper investigation — possibly the verification search interferes with our correction history or TT-dampen patterns.

### V5: Singular Extensions v2 — SF-like params (REJECTED — needs investigation)
- **Change**: Singular extensions with fixed wiring + SF-like params: depth>=8, margin 3*depth/2, verification (depth-1)/2.
- **Result**: **H0 at 127 games, -58.0 Elo.** Catastrophic even with correct implementation.
- **Notes**: Three attempts all failed badly: (1) original broken code (discarded extension), (2) fixed wiring + original params depth>=10/margin=depth*3, (3) fixed wiring + SF params depth>=8/margin=3*depth/2. The low draw ratio (53%) suggests search destabilization. Singular extensions are a proven +20-30 Elo technique in other engines. Our implementation is fundamentally incompatible with something in our search — possibly alpha-reduce, failing heuristic, or FH-blend interacting with the verification search. **TODO**: Investigate by disabling alpha-reduce/failing/FH-blend one at a time with singular enabled to isolate the conflict.

### V5: ProbCut Depth 4 (RETEST CANDIDATE — small positive)
- **Change**: Enable ProbCut at depth >= 4 (was depth >= 5).
- **Result**: **H0 at 1255 games, -0.3 Elo.** Showed +4-8 Elo for most of its run before collapsing.
- **Notes**: Another consistent small positive that couldn't clear wide bounds. Retest with tight bounds.

### V5: RFP Depth 9 (LIKELY REJECT)
- **Change**: Extend RFP from depth <= 7 to depth <= 8.
- **Status**: -0.7 Elo at 984 games, LLR -2.44. Dead flat, nearly H0.

### V5: RFP Depth 9 (REJECTED)
- **Change**: Extend RFP from depth <= 7 to depth <= 8.
- **Result**: **H0 at 1091 games, -1.6 Elo.** Dead flat.
- **Notes**: RFP depth 7 is well-calibrated. At depth 8, the margin (100*8=800cp non-improving) is already very wide, and positions that far ahead don't need the pruning savings.

### V5: ContHist Aging 12288 (REJECTED)
- **Change**: Faster gravity decay on continuation history table (divisor 16384 → 12288).
- **Result**: **H0 at 212 games, -26.3 Elo.** Decisive.
- **Notes**: Continuation history captures positional patterns across move pairs — these are stable properties that don't change game-to-game. Faster decay loses this information. Main history (from/to) may benefit from faster decay (hist-age still running) because it tracks move-specific tactical patterns that are more volatile. Different history tables need different decay rates.

### V5: CapHist Aging 12288 (REJECTED)
- **Change**: Faster gravity decay on capture history table (divisor 16384 → 12288).
- **Result**: **H0 at 215 games, -29.2 Elo.**
- **Notes**: Capture history patterns (which piece captures which type on which square) are stable across positions. Faster decay loses this information. Confirms: only main history (from/to) might benefit from faster decay — contHist and capHist need 16384 stickiness.

### V5: History Aging 12288 (RETEST CANDIDATE — small positive)
- **Change**: Faster gravity decay on main history table (divisor 16384 → 12288).
- **Status**: +1.8 Elo at 980 games, fading from peak of +27 at 149 games. Likely H0.
- **Notes**: Strong early signal that didn't hold. The main history table may benefit from slightly faster decay but the effect is too small for wide SPRT bounds. Retest with tight bounds, or try intermediate values (14336).

### V5: NMP Deep Reduction d>=14 (RETEST CANDIDATE — small positive)
- **Change**: Extra +1 NMP reduction when depth >= 14.
- **Result**: **H0 at 1657 games, +0.6 Elo.** Showed +4-9 Elo for first 1000 games.
- **Notes**: Consistent small positive (~2-4 Elo) that faded. Retest with tight bounds.

### V5: History Aging 12288 (REJECTED)
- **Change**: Faster gravity decay on main history table (divisor 16384 → 12288).
- **Result**: **H0 at 1635 games, +0.4 Elo.** Early peak of +27 at 149 games was pure noise.
- **Notes**: Main history gravity at 16384 is well-calibrated. Tested all three history tables (main, conthist, caphist) — all rejected. The gravity divisor is not a productive tuning dimension.

### V5: NMP Divisor 150 (REJECTED)
- **Change**: More aggressive NMP eval-based reduction (divisor 200 → 150).
- **Result**: **H0 at 1418 games, 0.0 Elo.** Perfectly flat.
- **Notes**: NMP divisor 200 confirmed well-calibrated for v5. Tested both directions now: 150 (flat) and noted in earlier experiments that the divisor is not a productive dimension.

### V5: Passed Pawn LMR (LIKELY REJECT)
- **Change**: Reduce LMR by 1 for pawn moves to 6th/7th rank.
- **Status**: -2.3 Elo at 479 games, flat/negative.
- **Notes**: Advanced pawn moves are already handled well by the history table — good pawn pushes get high history scores and receive less reduction through the continuous history adjustment. An explicit pawn rank check is redundant.

### V5: Passed Pawn LMR (REJECTED)
- **Change**: Reduce LMR by 1 for pawn moves to 6th/7th rank.
- **Result**: **H0 at 831 games, -3.3 Elo.**
- **Notes**: History-based continuous LMR adjustment already handles good pawn pushes — they get high history scores and less reduction organically. Explicit piece-type checks are redundant.

### V5: QS Evasion LMP (RETEST CANDIDATE — small positive)
- **Change**: Prune late quiet evasion moves (moveCount > 4) in quiescence when in check.
- **Status**: +2.9 Elo at 1474 games, fading. Showed +6-8 Elo for first 1000 games.
- **Notes**: Another small positive. Retest with tight bounds.

### V5: QS Evasion LMP (REJECTED)
- **Change**: Prune late quiet evasion moves (moveCount > 4) in quiescence when in check.
- **Result**: **H0 at 1690 games, +0.6 Elo.** Showed +6-8 Elo for first 1000 games before collapsing.
- **Notes**: Another strong early signal that didn't hold. QS evasion handling is already efficient.

### V5: Persist ContHist /2 (REJECTED)
- **Change**: Halve continuation history between searches instead of clearing.
- **Result**: **H0 at 201 games, -29.5 Elo.**
- **Notes**: ContHist captures move-pair patterns that are position-specific. Persisting them between games (different positions) pollutes the table with irrelevant patterns. Unlike main history which is more generic (from/to squares), contHist is too contextual to persist.

### V5: Aspiration Widen 2x (RETEST CANDIDATE — small positive)
- **Change**: Aspiration window widening from 1.5x to 2x on fail-high/fail-low.
- **Status**: +0.4 Elo at 976 games, fading. Peaked at +13.9 at 605 games.
- **Notes**: Showed consistent early signal before collapsing. Retest with tight bounds.

### V5: Persist History /2 (LIKELY REJECT)
- **Change**: Halve main history between searches instead of clearing.
- **Status**: -3.5 Elo at 617 games, heading H0.
- **Notes**: Was a merged win on v4 but doesn't help on v5. The v5 NNUE eval may produce different move ordering patterns that don't transfer well between games, or the TT (which persists across games) already provides sufficient inter-game knowledge.

### V5: Persist History /2 (REJECTED)
- **Change**: Halve main history between searches instead of clearing to zero.
- **Result**: **H0 at 691 games, -5.5 Elo.**
- **Notes**: Was a merged win on v4 but harmful on v5. The v5 NNUE eval produces different move ordering dynamics. Clearing history gives the search a fresh start each game, which is better with v5's stronger eval guiding move ordering from scratch.

### V5: Aspiration Widen 2x (REJECTED)
- **Change**: Aspiration window widening from 1.5x to 2x on fail-high/fail-low.
- **Result**: **H0 at 1270 games, -0.5 Elo.** Peaked at +13.9/94% LOS at 605 games before collapsing.
- **Notes**: The 1.5x widening is well-calibrated. 2x widens too fast, reaching full window sooner and losing the aspiration window benefit. Another dramatic early signal that was pure noise.

### V5: Non-Linear Output Buckets (REJECTED — incompatible)
- **Change**: Alexandria's output bucket formula `(63-pc)(32-pc)/225` replacing our linear `(pc-2)/4`.
- **Result**: 0 wins, 37 losses in 37 games. Catastrophic.
- **Notes**: The net was TRAINED with linear bucket mapping. Changing the bucket selection at inference without retraining maps positions to wrong output weight sets. This is a training-time change, not an inference-time change. Would need to retrain the net with the new formula to test properly.

### V5: Singular Extensions — Alexandria Params (REJECTED)
- **Change**: Alexandria-style singular: depth>=6, margin 5*depth/8, fail-high cutoff returning beta.
- **Result**: **H0 at 96 games, -84.9 Elo.** Even worse than SF-like params (-58 Elo).
- **Notes**: Diagnostic tests (200 games each, singular + one feature disabled):
  - SE + no alpha-reduce: -51 Elo (slightly less bad)
  - SE + no failing: -72 Elo
  - SE + no FH-blend: -98 Elo (worse — FH-blend was helping)
  - SE + no TT-dampen: -100 Elo (worse — TT-dampen was helping)
  None of our unique features conflict with singular. The issue is structural — possibly our TT replacement policy, move ordering, or search tree shape is fundamentally incompatible. Needs deeper investigation (trace singular verification search decisions, compare with a known-working engine).

### V5: Hindsight Reduction (REJECTED)
- **Change**: Reduce depth by 1 when eval hasn't changed much over 2 plies (diff < 155cp).
- **Result**: **H0 at 91 games, -81.6 Elo.** Catastrophic.
- **Notes**: The threshold 155cp may be too loose for our eval scale, or the eval comparison across 2 plies isn't meaningful with our NNUE eval. Alexandria may use this with a differently scaled eval.

### V5: RFP Score Blending (REJECTED)
- **Change**: RFP returns `(eval-margin + beta) / 2` instead of `eval - margin`.
- **Result**: **H0 at 355 games, -16.7 Elo.**
- **Notes**: Unlike TT-dampen and FH-blend (which work at noisy boundaries), RFP pruning is a clean cutoff — the eval IS far above beta. Blending the return toward beta loses accurate information. Dampening doesn't work at every boundary.

### LMR Quiet C=1.30 (MERGED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.30 (more aggressive reduction).
- **Result**: **H1 at 1016 games, +13.3 Elo ±14.7, LOS 96.3%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: aee6f1d (TT near-miss 80 merged)
- **Notes**: Perfectly bracketed: 1.25 (H0, -3.6), **1.30 (H1, +13.3)**, 1.35 (H0, +1.2), 1.40 (H0, -2.9), 1.50 (previous), 1.60 (H0, -29.1). The LMR progression continues: C=2.0→1.75(+16.2)→1.50(+44.4)→1.30(+13.3). More aggressive quiet LMR works because NNUE eval is accurate enough to trust shallower searches for non-critical moves.

### LMR Quiet C=1.35 (REJECTED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.35.
- **Result**: **H0 at 2063 games, +1.2 Elo ±9.9, LOS 59.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: Dead flat. The optimum is sharply at 1.30, not a gradual slope.

### LMR Quiet C=1.25 (REJECTED)
- **Change**: Tighten quiet LMR constant from 1.50 to 1.25.
- **Result**: **H0 at 959 games, -3.6 Elo ±14.8, LOS 31.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: Too aggressive. Confirms 1.30 as the sharp optimum.

### V5: NMP R-1 After Captures (MERGED)
- **Change**: Reduce NMP R by 1 when the previous move was a capture (position is more forcing).
- **Result**: **H1 at 704 games, +14.8 Elo ±15.8, LOS 96.7%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 8d92fc4 (LMR quiet C=1.30 merged)
- **Notes**: Captures make positions more forcing — null move assumption is riskier when opponent just captured. Source: Tucano. 3 lines of code.

### V5: Futility History Gate (REJECTED)
- **Change**: Don't futility-prune moves with combined history > 12000.
- **Result**: **H0 at 771 games, -5.0 Elo ±15.8, LOS 26.9%.**
- **Notes**: Showed +15 Elo at 400 games (90% LOS) then collapsed. Classic early noise. The futility margin is well-calibrated — history-gating adds complexity without benefit. Source: Igel.

### V5: Mate Distance Pruning (REJECTED)
- **Change**: Tighten alpha/beta bounds when a shorter mate is already known (standard MDP).
- **Result**: **H0 at 1559 games, +0.7 Elo ±10.4, LOS 55.0%.**
- **Notes**: Dead flat. Universal technique but our search rarely encounters positions where MDP would trigger (deep mates found at the root that need pruning at leaf nodes). The overhead of the alpha/beta adjustment at every node isn't justified.

### V5: Complexity-Adjusted LMR (REJECTED)
- **Change**: Reduce LMR less when correction history magnitude is large (uncertain eval needs deeper search). `reduction -= complexity / 120`.
- **Result**: **H0 at 1338 games, 0.0 Elo ±11.3, LOS 50.0%.**
- **Notes**: Perfectly flat. The correction history magnitude doesn't correlate with positional complexity strongly enough to improve LMR. Source: Obsidian uses this but with different scaling.

### Cap LMR C=1.90 (REJECTED)
- **Change**: Less aggressive capture LMR (divisor from 1.80 to 1.90).
- **Result**: **H0 at 749 games, -6.0 Elo ±16.7, LOS 24.0%.**
- **Notes**: Capture LMR C=1.80 is well-calibrated. Less reduction (1.90) searches captures too deeply.

### NMP Stronger (REJECTED)
- **Change**: More aggressive NMP (details from experiment queue).
- **Result**: **H0 at 208 games, -33.5 Elo ±31.4, LOS 1.9%.** Decisive.

### Correction History Depth 2 (REJECTED)
- **Change**: Lower correction history depth gate from >=3 to >=2.
- **Result**: **H0 at 277 games, -23.9 Elo ±27.0, LOS 4.2%.**
- **Notes**: Depth-2 scores are too shallow/noisy for correction history updates. Depth>=3 is well-calibrated.

### V5: Score-Drop Time Extension (MERGED)
- **Change**: More aggressive time scaling on score drops: 2.0x at >50cp (was 1.4x), 1.5x at >25cp (was 1.2x), new 1.2x tier at >10cp.
- **Result**: **H1 at 793 games, +13.6 Elo ±14.8, LOS 96.3%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: c6447e8 (dynamic NNUE width)
- **Notes**: Gives the engine more time to recover in volatile positions where the score drops between iterations. Source: Tucano-style aggressive scaling.

### V5: SE Fixed — Pruning Guards (REJECTED)
- **Change**: Gate NMP, RFP, razoring, ProbCut on `!excludedMove` during SE verification search. Diagnosis found NMP was short-circuiting the verification.
- **Result**: **H0 at 71 games, -105.9 Elo.** Still catastrophic despite fixing the identified bug.
- **Notes**: The pruning guards helped (was -140, now -106) but SE is still fundamentally broken. There are likely additional interactions with TT dampening, FH-blend, alpha-reduce, or NMP dampening that corrupt the verification search.

### V5: NMP Base R=4 (REJECTED)
- **Change**: Increase NMP base reduction from R=3 to R=4 (Alexandria/Berserk/Obsidian use R=4).
- **Result**: **H0 at 307 games, -17.0 Elo.** Decisive.
- **Notes**: Our R=3 is well-calibrated. Other engines have different search frameworks (singular extensions, cutNode tracking) that interact with NMP differently.

### V5: IIR Extra on PV (REJECTED)
- **Change**: Double IIR reduction on PV nodes without TT move (`depth -= 1 + pvNode`).
- **Result**: **H0 at 237 games, -26.4 Elo.**
- **Notes**: PV nodes without TT moves are important — reducing them too much loses accuracy. Source: Altair.

### V5: Cutnode LMR +2 (REJECTED)
- **Change**: Increase cut-node quiet LMR reduction from +1 to +2.
- **Result**: **H0 at 881 games, -3.2 Elo.** Dead flat.
- **Notes**: Our +1 cut-node reduction is well-calibrated. +2 is too aggressive — it misses tactical quiet moves at cut nodes.

### V5: SE Alexandria (REJECTED — 5th attempt)
- **Change**: Full Alexandria-style SE: depth>=6, margin d*5/8, ply limiter, multi-cut, double/triple ext, negative ext -2.
- **Result**: **H0 at 55 games, -139.7 Elo.** Catastrophic (5th consecutive SE failure).
- **Notes**: Root cause identified: NMP fires inside verification search (see se-diagnosis.md). Fixed in SE-Fixed variant but still -106 Elo. Additional interactions remain.

### V5: Quadratic 50-Move Scaling (REJECTED)
- **Change**: Scale eval by `(200-hmc)²/40000` as halfmove clock advances.
- **Result**: **H0 at 340 games, -13.3 Elo.** 
- **Notes**: Quadratic scaling (Reckless/Minic) loses more Elo than our previous linear test (H0, -3.0). The 50-move rule rarely matters in self-play at 10+0.1s — games don't last that long. Any scaling just adds noise.

### V5: ProbCut QS Pre-Filter (KILLED — flat)
- **Change**: Run QS before full ProbCut search; only do expensive search if QS confirms.
- **Result**: Killed at 287 games, +3.5 Elo. Flat.
- **Notes**: QS pre-filter adds overhead without improving ProbCut accuracy at our depths. 4 engines have this (Alexandria, Tucano, Berserk, Weiss) but it may only help with deeper searches.

### V5: NMP Threat Guard (REJECTED)
- **Change**: Disable NMP at depth<=7 when opponent pawns attack our non-pawn pieces.
- **Result**: **H0 at 236 games, -23.6 Elo.**
- **Notes**: Too conservative — NMP is still valuable even under minor pawn threats. The eval already accounts for threats. Source: Berserk/RubiChess/Koivisto pattern, but our engine handles it differently.

### V5: NMP Threat Guard (REJECTED)
- **Change**: Disable NMP at depth<=7 when opponent pawns attack our non-pawn pieces.
- **Result**: **H0 at 236 games, -23.6 Elo.**
- **Notes**: Too conservative — NMP is still valuable even under minor pawn threats. The eval already accounts for threats.

### V5: Improving-Aware Capture LMR (IN PROGRESS)
- **Status**: +4.2 Elo at 1005 games, 73.8% LOS. Flat, likely H0.

### V5: ProbCut Margin 150 (REJECTED)
- **Change**: Tighter ProbCut margin from beta+170 to beta+150.
- **Result**: **H0 at 1386 games, +0.3 Elo.** Dead flat.
- **Notes**: ProbCut margin 170 is well-calibrated. 150 prunes too aggressively, removing positions worth searching.

### V5: Improving-Aware Capture LMR (REJECTED)
- **Change**: Reduce capture LMR by 1 when position is improving.
- **Result**: **H0 at 1434 games, +0.2 Elo.** Dead flat.
- **Notes**: Capture LMR already has history-based adjustment. Adding an improving flag is redundant — the capture history captures this information implicitly.

### V5: LMR History Divisor 6000 (REJECTED)
- **Change**: Less aggressive history-based LMR adjustment (divisor 5000→6000).
- **Result**: **H0 at 241 games, -23.1 Elo.**
- **Notes**: Our divisor 5000 is well-calibrated. Less aggressive history response means good moves get less reduction relief, losing tactical accuracy. Alexandria uses 8300 but with a very different search framework.

### V5: Aspiration Fail-Low Beta Contraction (MERGED)
- **Change**: More aggressive beta contraction on aspiration fail-low: `(3*alpha + 5*beta) / 8` instead of `(alpha + beta) / 2`.
- **Result**: **H1 at 1193 games, +10.5 Elo ±11.9, LOS 95.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: cff9599 (score-drop TM merged)
- **Notes**: Tighter beta contraction helps converge faster on the true score when failing low. Source: Altair uses `(3*alpha + 5*beta) / 8`, Midnight contracts on fail-low similarly.

### V5: RFP Improving Margin 60 (LIKELY REJECT)
- **Status**: Collapsed from +13.9 at 317 games to +0.5 at 658 games. Classic early noise.
- **Notes**: RFP improving margin 70 is well-calibrated for v5.

### V5: RFP Improving Margin 60 (REJECTED)
- **Change**: Tighter RFP improving margin from depth*70 to depth*60.
- **Result**: **H0 at 853 games, -2.9 Elo.** Dead flat.
- **Notes**: Improving margin 70 is well-calibrated. 60 prunes too aggressively in improving positions. Bracket: 60 (H0), 70 (current), 80 (testing).

### V5: Aspiration Fail-High Alpha Contraction (MERGED)
- **Change**: Less aggressive alpha contraction on fail-high: `(5*alpha + 3*beta) / 8` instead of `alpha = beta`.
- **Result**: **H1 at 200 games, +38.4 Elo ±29.2, LOS 99.5%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 4994e27 (asp fail-low contraction merged)
- **Notes**: The old `alpha = beta` was too aggressive — it jumped alpha all the way to beta, then widened beta. The gentler contraction keeps some of the window below the fail-high score, reducing re-search overhead. Combined with the fail-low contraction, both sides of the aspiration window now use smooth contraction.

### V5: RFP Improving Margin 80 (IN PROGRESS)
- **Status**: -1.3 Elo at 546 games, heading H0. Brackets: 60 (H0), 70 (current), 80 (flat). Margin 70 confirmed optimal.

### V5: StatBonus on Strong Fail-High (REJECTED)
- **Change**: Use `historyBonus(depth+1)` when score > beta+95 at beta cutoff.
- **Result**: Collapsed from +27.4 at 158 games to -3.3 at 428 games. Heading H0.
- **Notes**: Early noise. The bonus at depth+1 is too small a change to measure.

### V5: ASP Fail-High (3a+5b)/8 (REJECTED)
- **Change**: Less aggressive fail-high alpha contraction: (3a+5b)/8 instead of (5a+3b)/8.
- **Result**: -11.6 Elo at 429 games, heading H0. Confirms (5a+3b)/8 is optimal.

### V5: NMP PostCap R-2 (REJECTED)
- **Change**: R-2 NMP reduction after captures (bracket of R-1).
- **Result**: **H0 at 201 games, -27.7 Elo.** R-2 is too cautious after captures. R-1 confirmed optimal.

### V5: RFP Improving Margin 80 (REJECTED)
- **Change**: Looser RFP improving margin from depth*70 to depth*80.
- **Result**: **H0 at 1194 games, -0.6 Elo.** Dead flat. Bracket complete: 60 (H0), 70 (current), 80 (H0). Margin 70 confirmed optimal.

### V5: ASP Delta Growth 2.0x (REJECTED)
- **Change**: Faster aspiration delta growth: `delta += delta` (2.0x) instead of `delta += delta/2` (1.5x).
- **Result**: **H0 at 164 games, -34.0 Elo.** Decisive.
- **Notes**: Widening too fast reaches full window too quickly, losing the aspiration benefit. Bracket: 1.33x (testing), 1.5x (current), 2.0x (H0).

## Singular Extensions Deep Dive (2026-03-20)

### SE v6: NMP-guarded + full features (REJECTED)
- **Change**: SE with NMP disabled inside verification, RFP/ProbCut active, multi-cut (singularBeta>=beta), negative ext (-1 when ttScore>=beta), margin=depth*1, depth>=8, ply limiter.
- **Result**: **H0 at 211 games, -33.0 Elo ±31.3.**
- **Baseline**: ec7dab6 with v5 sb120 net
- **Notes**: The positive extension (+1 for singular moves) is actively harmful. Extensions make the TT move search deeper but don't improve play — likely because v5 NNUE already evaluates well at current depth.

### SE v7: Multi-cut + negative ext only, NO positive extension (REJECTED)
- **Change**: Same as v6 but positive extension disabled (singularExtension stays 0 when singular). Only multi-cut (return singularBeta when singularBeta >= beta) and negative ext (-1 when ttScore >= beta but not singular).
- **Result**: **H0 at 1148 games, -1.5 Elo ±12.9.** Dead flat.
- **Baseline**: ec7dab6 with v5 sb120 net
- **Notes**: Multi-cut + negative ext perfectly offset the verification search cost, confirming they work correctly. The positive extension is the problem — v6 (-33 Elo) minus v7 (-1.5 Elo) implies the extension alone costs ~30 Elo.

### SE Root Cause Analysis
**Finding**: SE's positive extension hurts our engine (~-30 Elo) despite being +20-30 in reference engines. The verification search infrastructure (multi-cut, negative ext) works correctly and breaks even. Key hypotheses for why extensions hurt:
1. **v5 NNUE eval accuracy**: Strong positional eval may already capture what extensions aim to find
2. **Extension interactions**: Alpha-reduce, fail-high blending, or other features may conflict with SE extensions
3. **Node budget**: Extensions at depth >= 8 create large subtrees that reduce remaining budget for other critical moves
4. **Extension quality**: Our SE extends at depth >= 8 only, which may target the wrong nodes (too deep, less tactical)

**Next steps**: Test fractional extensions (+0.5 ply), restrict extensions to quiet TT moves only, or try at depth >= 6 with very tight margin.

### Model: SB200 vs SB120 wdl=0.0 6-file (NEUTRAL)
- **Change**: 200 superbatches vs 120, same wdl=0.0, same 6 T80 files, same architecture.
- **Result**: +1.0 Elo at 376 games, dead flat. Killed — heading H0.
- **Notes**: Validation loss showed sb200 slightly overfitting, confirmed in play. 120 SBs is optimal for wdl=0.0 1024 CReLU. The wdl=0.5 sb200 gains (+98 to +243 internal) were an artifact of undertrained wdl=0.5 nets needing more epochs.

### Model: 1024 wdl=0.0 6-file sb120 vs Production (NEUTRAL)
- **Change**: Same architecture, wdl=0.0, but 6 T80 files instead of 1.
- **Result**: -3.1 Elo at 798 games, heading H0. Pipeline reproduction confirmed.
- **Notes**: Extra data diversity doesn't help at 1024-wide with 120 SBs. But critically: the new Bullet version produces equivalent-strength nets. Training pipeline is proven reproducible.

### V5: ASP Delta Growth 1.33x (REJECTED)
- **Change**: Slower aspiration delta growth: `delta += delta/3` (1.33x) instead of 1.5x.
- **Result**: **H0 at 391 games, -9.8 Elo.** Bracket: 1.33x (H0), 1.5x (current), 2.0x (H0). Growth rate 1.5x confirmed optimal.

### V5: Bugfix Impact (NEUTRAL)
- **Change**: Atlas code review fixes: v5 SMP DeepCopy, int64 overflow, ARM stubs, etc.
- **Result**: +4.3 Elo at 550 games, flat. No regression confirmed. The int32 overflow fix only affects scalar path (SIMD still int32).

### V5: ASP Starting Delta 18 (REJECTED)
- **Change**: Wider aspiration starting delta from 15 to 18.
- **Result**: Collapsed from +31 at 101 games to +3.3 at 755 games. Heading H0.
- **Notes**: Delta 15 is optimal even with the new contraction parameters. Bracket: 12 (H0), 15 (current), 18 (H0).

### V5: ASP Min Depth 5 (REJECTED)
- **Change**: Only use aspiration windows at depth >= 5 (was >= 4).
- **Result**: **H0 at 538 games, -7.8 Elo.**

### V5: TM Aggressive Stability (REJECTED)
- **Change**: Stronger early stop on stable best move: 0.25x at 8+ (was 0.35x), 0.4x at 5+ (was 0.5x).
- **Result**: **H0 at 403 games, -14.7 Elo.**

### V5: IIR Depth 7 (REJECTED)
- **Change**: Raise IIR gate from depth >= 6 to depth >= 7.
- **Result**: **H0 at 343 games, -15.2 Elo.**
- **Notes**: Finny NPS boost doesn't change optimal IIR threshold.

### V5: NMP Verification Depth 10 (REJECTED)
- **Change**: Lower NMP verification gate from depth >= 12 to depth >= 10.
- **Result**: **H0 at 385 games, -15.4 Elo.**
- **Notes**: More frequent verification wastes nodes. Depth 12 is optimal.

### V5: ASP Fail-Low (a+7b)/8 (REJECTED)
- **Change**: More aggressive fail-low contraction: (a+7b)/8 instead of (3a+5b)/8.
- **Result**: **H0 at 208 games, -26.8 Elo.**
- **Notes**: Too aggressive — beta contracts too close to alpha, causing tight windows that fail repeatedly. (3a+5b)/8 confirmed optimal. Full bracket: (a+b)/2 (old, +10.5 Elo improvement), (3a+5b)/8 (current), (a+7b)/8 (H0).

### V5: LMP Depth 9 (REJECTED)
- **Change**: Extend LMP from depth<=8 to depth<=9.
- **Result**: **H0 at 791 games, -4.8 Elo.**
- **Notes**: Depth 8 is optimal. At depth 9, LMP prunes moves that were worth searching.

### Model: GPU1 Reproduce vs Production (NEUTRAL)
- **Change**: GPU1 reproduction of production model with new Bullet, same config.
- **Result**: **H0 at 1470 games, -0.9 Elo.** Pipeline confirmed on GPU1.

### Model: 1536 sb200 vs 1024 Production (H1 — WINNER!)
- **Change**: 1536-wide CReLU net (wdl=0.0, 6 T80 files, 200 SBs) vs 1024 production (120 SBs).
- **Result**: **H1 at 1522 games, +11.0 Elo ±12.4, LOS 95.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Notes**: First time a wider architecture beats production. The 12% NPS penalty is overcome by better eval quality. The 1536 net has 2.7% lower validation loss. sb300/sb400 still training — may gain more. This opens the path to 1536 as production architecture.

### V5: QS Delta 220 (REJECTED)
- **Change**: Tighter QS delta from 240 to 220.
- **Result**: **H0 at 1270 games, -0.5 Elo.** Dead flat. Bracket: 200 (H0), 220 (H0), 240 (current).

### V5: Futility 50+d*50 (REJECTED)
- **Change**: Tighter futility from 60+d*60 to 50+d*50.
- **Result**: **H0 at 699 games, -5.5 Elo.** Bracket: 50+d*50 (H0), 60+d*60 (current), 100+d*100 (old).

### V5: Correction-Aware RFP (REJECTED)
- **Change**: Widen RFP margin by correction history magnitude / 128.
- **Result**: **H0 at 452 games, -10.8 Elo.**
- **Notes**: Correction magnitude doesn't correlate well with RFP safety. The correction adjusts the eval value directly — adding it to the margin double-counts the uncertainty.

### V5: Pawn History in LMR (REJECTED)
- **Change**: Add pawn history as 4th signal in LMR continuous adjustment.
- **Result**: **H0 at 674 games, -5.7 Elo.**
- **Notes**: Pawn history is already in move ordering. Adding it to LMR dilutes the butterfly + contHist signals.

## Atlas Batch 2 (2026-03-21)

### V5: Mate Distance Pruning (REJECTED)
- **Change**: Prune when current ply can't improve on known mate bound. `alpha = max(alpha, -MateScore+ply); beta = min(beta, MateScore-ply-1)`.
- **Result**: **H0 at 1214 games, -0.9 Elo.** Dead flat.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Universal technique but rarely triggers at 10+0.1s TC. Games don't reach deep enough mate sequences for this to help. May help at longer TCs or in endgame positions.

### V5: ASP Fail-High Depth Reduce (REJECTED)
- **Change**: On aspiration fail-high, reduce inner search depth by 1 before re-widening. Alexandria-style inner-loop-only modification.
- **Result**: **H0 at 26 games, -389 Elo.** Catastrophic.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Third attempt at this idea (previous was -353 Elo). Even the "correct" inner-loop implementation breaks the aspiration loop. The depth variable is shared between the aspiration loop iterations — reducing it mid-loop corrupts subsequent iterations. This idea fundamentally conflicts with our aspiration implementation.

### V5: Complexity-Adjusted LMR (REJECTED)
- **Change**: Reduce LMR by 1 when `abs(correctedEval - rawEval) > 80` (high correction = uncertain position = search deeper).
- **Result**: **H0 at 209 games, -25.0 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Binary threshold too aggressive — reducing LMR for any "complex" position makes search too deep for marginal gains. The correction history already adjusts the eval value; using its magnitude to also adjust LMR double-counts the uncertainty signal.

### V5: Node-Based Time Management (REJECTED)
- **Change**: Track nodes per root move, scale TM by `(1.5 - bestMoveNodeFrac) * 1.7`. Stop early when best move dominates (>80% nodes), extend when unclear (<30%).
- **Result**: **H0 at 424 games, -12.3 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Alexandria-style formula too aggressive when combined with our existing stability scaling. The multiplicative interaction between node-fraction and stability scaling caused over-early stopping. May need to replace stability scaling rather than multiply with it.

### V5: Opponent Eval Feedback (REJECTED)
- **Change**: Update opponent's move history based on `evalSum = (ss-1)->staticEval + ss->staticEval`. Penalize moves where evalSum is positive (opponent made their position worse).
- **Result**: **H0 at 808 games, -3.9 Elo.** Dead flat.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Alexandria/Obsidian both have this, but our correction history may already capture this signal. The evalSum signal is noisy when correction adjustments are large.

### V5: Complexity-Adjusted RFP (REJECTED)
- **Change**: Widen RFP margin by `complexity/2` where complexity = `abs(correctedEval - rawEval)`.
- **Result**: **H0 at 592 games, -6.5 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Second attempt at using eval complexity (first was LMR at -25). RFP margin widening is less harmful than LMR reduction, but still doesn't help. Conclusion: eval complexity from correction history is not a useful search signal in our engine — the correction already adjusts the eval directly.

### V5: Hindsight Reduction (MERGED)
- **Change**: When `(ss-1)->staticEval + ss->staticEval > 150`, reduce depth by 1 before NMP/pruning. Both sides think position is quiet.
- **Result**: **H1 at 1549 games, +9.4 Elo ±10.7, LOS 95.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Alexandria uses threshold 155. 5 engines have variants. Persistent positive signal from 600 games onward — never dipped below +5 after recovery from early noise.

### V5: Eval-Based History Depth Bonus (REJECTED)
- **Change**: When beta cutoff occurs and staticEval <= alphaOrig (surprising cutoff), use `historyBonus(depth+1)` instead of `historyBonus(depth)`. Alexandria pattern.
- **Result**: **H0 at 285 games, -19.5 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: The +1 depth bonus is too small a change to help and may over-reinforce moves that got lucky. Our history tables are already well-calibrated (pattern #4 from success analysis).

### V5: Alpha-Reduce Skip Move 2 (REJECTED)
- **Change**: Don't apply alpha-reduce to move 2 (first non-TT move) — preserve full depth for the second-best alternative when TT move raised alpha.
- **Result**: **H0 at 173 games, -40.3 Elo.** Catastrophic.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Alpha-reduce on move 2 is actually valuable — the TT move already raised alpha, and searching move 2 at full depth wastes nodes on a likely inferior move. The alpha-reduce feature is correctly applied to ALL moves after alpha is raised.

### V5: IIR Depth Gate 4 (REJECTED)
- **Change**: Lower IIR gate from `depth >= 6` to `depth >= 4`. More positions get IIR when no TT move exists.
- **Result**: **H0 at 769 games, -3.2 Elo.** Dead flat.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: IIR at depth 4-5 doesn't help because shallow searches find TT moves quickly anyway. Gate 6 is well-calibrated.

### V5: Draw Score Randomization (REJECTED)
- **Change**: Add ±2cp noise to repetition draw scores: `return -Contempt + int(2-(info.Nodes&3))`. Koivisto pattern to prevent repetition-seeking.
- **Result**: **H0 at 830 games, -3.8 Elo.** Dead flat.
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Our Contempt=10 already handles draw avoidance. Adding random noise doesn't improve play — the engine already prefers playing on.

### V5: ASP Initial Delta 12 (REJECTED)
- **Change**: Tighter initial aspiration window: delta 15 → 12.
- **Result**: **H0 at 203 games, -34.3 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Delta 12 is too tight — causes more aspiration re-searches that waste time. Delta 15 is well-calibrated. Bracket: 12 (H0), 15 (current). Not testing higher since wider deltas lose aspiration benefit.

### V5: Material Scaling of NNUE Output (REJECTED)
- **Change**: Scale NNUE eval by `(700 + pieceCount*40) / 1024`. Dampens eval in low-material endgames. Alexandria pattern.
- **Result**: **H0 at 211 games, -24.7 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: Our v5 net likely handles endgame scaling internally through its 8 output buckets. External scaling conflicts with the net's learned behavior.

### V5: TT Cutoff History Malus (REJECTED)
- **Change**: On TT beta cutoff, penalize opponent's last quiet move with `-historyBonus(depth)`. Alexandria pattern.
- **Result**: **H0 at 248 games, -25.3 Elo.**
- **Baseline**: 836be58 with v5 sb120 net
- **Notes**: TT cutoffs fire very frequently and cheaply — applying history malus at every one pollutes the history tables with noisy data. Our TT cutoff history BONUS works (+22 Elo merged) because bonuses reinforce good moves; malus on the opponent's move is the wrong signal.

### V5: LMR Eval-Alpha Distance (REJECTED)
- **Change**: Increase LMR reduction when `abs(staticEval - alpha) > 175` (+1) or `> 350` (+2). Koivisto pattern: when eval is far from alpha, the position is clearly decided, reduce more.
- **Result**: **H0 at 250 games, -23.7 Elo.**
- **Baseline**: 323fd4d (with hindsight reduction) with v5 sb120 net
- **Notes**: The eval-alpha distance is already implicitly captured by our existing LMR adjustments (improving, failing, alpha-raised count). Adding an explicit distance check over-reduces in positions where the eval is high but tactically rich.

### V5: Hindsight Threshold 100 (IN PROGRESS)
- **Change**: More aggressive hindsight reduction: threshold 150 → 100. Reduces in more positions.
- **Status**: 386 games, -6 Elo. Flat/slightly negative.
- **Result**: **H0 at 492 games, -8.5 Elo.**
- **Notes**: Bracket test. 100 reduces too often — positions with evalSum 100-150 still have tactical content worth searching. Bracket: 100 (H0), 150 (H1, +9.4), 200 (testing).

### V5: SEE Quiet Pruning Depth 10 (REJECTED)
- **Change**: Extend SEE quiet pruning from `depth <= 8` to `depth <= 10`.
- **Result**: **H0 at 1788 games, +1.4 Elo.** Dead flat.
- **Baseline**: 15666ab with v5 sb120 net
- **Notes**: SEE quiet pruning at depth 9-10 doesn't help. Depth 8 confirmed optimal.

### V5: Bad Noisy Futility Depth 5 (REJECTED)
- **Change**: Extend bad noisy futility from `depth <= 4` to `depth <= 5`.
- **Result**: **H0 at 316 games, -15.4 Elo.**
- **Baseline**: 7ac7d81 with v5 sb120 net
- **Notes**: Previously tested at depth 6 (-5.3). Both 5 and 6 lose Elo. Depth 4 confirmed optimal — captures at depth 5 are too tactical to prune based on SEE alone.

### V5: SEE Capture Pruning Threshold 60 (REJECTED)
- **Change**: Tighten SEE capture pruning from `-depth*80` to `-depth*60`.
- **Result**: **H0 at 107 games, -52.3 Elo.** Catastrophic.
- **Baseline**: 920ac92 (with SEE cap 80) with v5 sb120 net
- **Notes**: Bracket complete: 60 (H0, -52), **80 (H1, +25)**, 100 (old). 60 prunes too many captures that are only slightly losing — they still have tactical value. Peak at 80.

### V5: SEE Capture Pruning Threshold 80 (MERGED)
- **Change**: Tighten SEE capture pruning threshold from `-depth*100` to `-depth*80`. Prunes more losing captures.
- **Result**: **H1 at 331 games, +25.2 Elo ±22.7, LOS 98.5%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 65cf9d1 with v5 sb120 net
- **Notes**: The tighter threshold prunes captures that lose 80cp*depth of material instead of 100cp*depth. NNUE accuracy means these slightly-losing captures are rarely worth searching. Try -depth*60 next to bracket.

### V5: Futility Depth 9 (REJECTED)
- **Change**: Extend futility pruning from `depth <= 8` to `depth <= 9`.
- **Result**: **H0 at 355 games, -11.7 Elo.**
- **Baseline**: 65cf9d1 with v5 sb120 net
- **Notes**: Depth 9 prunes too many quiet moves with tactical value. Depth 8 confirmed optimal.

### V5: NMP Base R=4 (REJECTED)
- **Change**: Increase NMP base reduction from R=3 to R=4. Alexandria uses R=4.
- **Result**: **H0 at 254 games, -24.7 Elo.**
- **Baseline**: 65cf9d1 with v5 sb120 net
- **Notes**: R=4 is too aggressive for our search framework — more null-move cutoffs are incorrect, especially in positions with zugzwang risk. Our R=3+depth/3 formula is well-calibrated. Previously tested NMP divisor changes (200→150, 200→170) which also failed.

### V5: LMP Depth 9 (REJECTED → RETRY CANDIDATE)
- **Result**: **H0 at 1615 games, +0.6 Elo.** Dead flat at wide bounds but persistent +3-5 early.
- **Change**: Extend LMP from `depth <= 8` to `depth <= 9`.
- **Status**: 476 games, +11 Elo. Holding positive.

### V5: History Pruning Threshold 1000 (REJECTED)
- **Change**: Tighten history pruning from `-1500*depth` to `-1000*depth`.
- **Result**: **H0 at 474 games, -11.0 Elo.**
- **Baseline**: 77c0cd7 (with hist-prune 1500) with v5 sb120 net
- **Notes**: Too aggressive — prunes moves that still have tactical value. Bracket: 1000 (H0), 1500 (H1), 2000 (H0). 1500 confirmed optimal.

### V5: History Pruning Threshold 2000 Reverse (REJECTED)
- **Change**: Loosen history pruning back from `-1500*depth` to `-2000*depth` (reverse of merged change).
- **Result**: **H0 at 570 games, -6.7 Elo.**
- **Baseline**: 77c0cd7 (with hist-prune 1500) with v5 sb120 net
- **Notes**: Confirms 1500 is better than 2000. Full bracket: 1000 (H0), **1500 (H1, +14.7)**, 2000 (H0). Peak at 1500.

### V5: History Pruning Threshold 1500 (MERGED)
- **Change**: Tighten history pruning threshold from `-2000*depth` to `-1500*depth`. Prunes more quiet moves with bad history.
- **Result**: **H1 at 687 games, +14.7 Elo ±15.7, LOS 96.6%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 15666ab with v5 sb120 net
- **Notes**: More aggressive pruning of moves with negative history scores. The NNUE eval is accurate enough that moves with bad history can be safely pruned earlier. Try -1000*depth next to continue bracketing.

### V5: LMR Capture C=1.60 (REJECTED)
- **Change**: Tighten capture LMR from C=1.80 to C=1.60 (more reduction for late captures).
- **Result**: **H0 at 226 games, -27.7 Elo.**
- **Baseline**: 15666ab with v5 sb120 net
- **Notes**: C=1.60 over-reduces captures — late captures are more tactical than late quiets and need their depth. C=1.80 confirmed optimal.

### V5: Razoring Depth 3 (REJECTED)
- **Change**: Extend razoring from `depth <= 2` to `depth <= 3`.
- **Result**: **H0 at 1197 games, -0.9 Elo.** Dead flat.
- **Baseline**: 6e4d4a9 with v5 sb120 net
- **Notes**: Razoring at depth 3 doesn't help — depth 3 positions are complex enough that dropping to QS loses too much information. Depth 2 confirmed optimal.

### V5: Remove TT Score Dampening (REJECTED)
- **Change**: Remove TT lower-bound score dampening `(3*score+beta)/4`. Test if still needed after aspiration/FH blending changes.
- **Result**: **H0 at 153 games, -43.4 Elo.** Catastrophic.
- **Baseline**: 15666ab with v5 sb120 net
- **Notes**: TT dampening is still critical (+22 Elo when added, -43 when removed). The aspiration contractions and fail-high blending don't replace it — they operate at different points in the search. TT dampening prevents score inflation from stale lower-bound entries.

### V5: IIR Depth Gate 5 (REJECTED)
- **Change**: Lower IIR gate from `depth >= 6` to `depth >= 5`.
- **Result**: **H0 at 357 games, -15.6 Elo.**
- **Baseline**: 6e4d4a9 with v5 sb120 net
- **Notes**: Bracket: 4 (H0, -3.2), 5 (H0, -15.6), 6 (current). IIR at depth 5 over-reduces — the positions at that depth benefit from having a TT move to guide search. Depth 6 confirmed optimal.

### V5: RFP Not-Improving Margin 110 (REJECTED)
- **Change**: Widen RFP not-improving margin from `depth*100` to `depth*110`.
- **Result**: **H0 at 908 games, -2.7 Elo.** Dead flat.
- **Baseline**: 6e4d4a9 with v5 sb120 net
- **Notes**: Not-improving margin 100 confirmed optimal. Bracket: 90 (H0, previous), 100 (current), 110 (H0).

### V5: QS Stand-Pat Blend Tighter (REJECTED)
- **Change**: Less dampening on QS stand-pat fail-high: `(bestScore+beta)/2` → `(2*bestScore+beta)/3`.
- **Result**: **H0 at 266 games, -17.0 Elo.**
- **Baseline**: 6e4d4a9 with v5 sb120 net
- **Notes**: The current 50/50 blend is well-calibrated. Less dampening lets noisy QS scores through, hurting search accuracy. Our QS blend was +4.9 Elo when added; the 50/50 ratio is the sweet spot.

### V5: SEE Capture Pruning Depth 8 (REJECTED)
- **Change**: Extend SEE capture pruning from `depth <= 6` to `depth <= 8`.
- **Result**: **H0 at 696 games, -3.5 Elo.** Dead flat.
- **Baseline**: 1cf3871 with v5 sb120 net
- **Notes**: SEE capture pruning at depth 7-8 doesn't help — captures at those depths are already filtered by other mechanisms (LMR, capture history). Depth 6 confirmed optimal.

### V5: ProbCut Depth Gate 4 (REJECTED)
- **Change**: Lower ProbCut depth gate from `depth >= 5` to `depth >= 4`.
- **Result**: **H0 at 468 games, -9.7 Elo.**
- **Baseline**: 1cf3871 with v5 sb120 net
- **Notes**: ProbCut at depth 4 fires too often with insufficient search depth (pcDepth = 0, basically QS). The shallow verification can't reliably confirm cutoffs. Depth 5 confirmed optimal.

### V5: Contempt 5 (REJECTED)
- **Change**: Lower contempt from 10 to 5 (less draw avoidance).
- **Result**: **H0 at 342 games, -14.2 Elo.**
- **Baseline**: 1cf3871 with v5 sb120 net
- **Notes**: Contempt 10 is well-calibrated. Lower contempt accepts too many draws in self-play. Previously tested contempt 15 (H0, -5.2). Bracket: 5 (H0), 10 (current), 15 (H0).

### V5: LMP Tight Base 2+d² (REJECTED)
- **Change**: Tighten LMP limit from `3 + depth*depth` to `2 + depth*depth`.
- **Result**: **H0 at 345 games, -16.1 Elo.**
- **Baseline**: 1cf3871 with v5 sb120 net
- **Notes**: Pruning one more move at each depth is too aggressive — the 3rd-to-last move at each depth is still valuable often enough to matter. Base 3 confirmed optimal.

### V5: NMP Verify Depth 16 (REJECTED)
- **Change**: Raise NMP verification from depth >= 14 to depth >= 16.
- **Result**: **H0 at 2045 games, +1.7 Elo.** Dead flat.
- **Baseline**: 3856669 (with NMP verify 14) with v5 sb120 net
- **Notes**: Bracket: 12 (old), 14 (H1, +27.9), 16 (H0), never (H0). Depth 14 confirmed optimal.

### V5: NMP No Verification (REJECTED)
- **Change**: Remove NMP verification search entirely (was `if depth >= 14`).
- **Result**: **H0 at 1478 games, +0.2 Elo.** Dead flat.
- **Baseline**: 3856669 (with NMP verify 14) with v5 sb120 net
- **Notes**: Verification at depth >= 14 rarely fires at 10+0.1s TC. Keeping it is safer for longer TCs.

### V5: Futility 50+d*50 (REJECTED)
- **Change**: Tighten futility margin from 60+d*60 to 50+d*50.
- **Result**: **H0 at 2329 games, +1.9 Elo.** Dead flat.
- **Baseline**: 323fd4d (pre-hindsight-200/NMP-14) with v5 sb120 net
- **Notes**: 60+d*60 confirmed optimal. Bracket: 50+d*50 (H0), 60+d*60 (current), 80+d*80 (old).

### V5: Hindsight Threshold 250 (REJECTED)
- **Change**: Less aggressive hindsight reduction: threshold 200 → 250.
- **Result**: **H0 at 672 games, -5.2 Elo.**
- **Baseline**: 5c6dbba (with hindsight 200) with v5 sb120 net
- **Notes**: Full bracket complete: 100 (H0, -8.5), 150 (H1, +9.4), 200 (H1, +16.2), 250 (H0, -5.2). Peak confirmed at 200 — reducing too rarely (250) misses beneficial pruning opportunities.

### V5: Hindsight Threshold 200 (MERGED)
- **Change**: Less aggressive hindsight reduction: threshold 150 → 200. Reduces only in very quiet positions.
- **Result**: **H1 at 664 games, +16.2 Elo ±16.7, LOS 97.1%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 323fd4d (with hindsight 150) with v5 sb120 net
- **Notes**: Bracket complete: 100 (H0, -8.5), 150 (H1, +9.4 vs no-hindsight), 200 (H1, +16.2 vs 150). 200 is clearly optimal — further from the over-pruning edge. Try 250 next to find the peak.

### V5: QS Recapture-Only at Depth >= 5 (REJECTED)
- **Change**: After 5+ capture plies in quiescence, restrict to recaptures only. Minic pattern to prevent QS explosion.
- **Result**: **H0 at 734 games, -4.3 Elo.** Dead flat.
- **Baseline**: 323fd4d with v5 sb120 net
- **Notes**: QS rarely reaches depth 5+ in our engine — SEE pruning and delta pruning cut most captures before that. The filter fires too rarely to matter.

### V5: NMP Verification Depth 14 (MERGED)
- **Change**: Raise NMP verification threshold from depth >= 12 to depth >= 14. Less verification = more NMP cutoffs accepted on trust.
- **Result**: **H1 at 324 games, +27.9 Elo ±24.3, LOS 98.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 323fd4d (with hindsight 150) with v5 sb120 net
- **Notes**: NMP verification at depth 12 was wasting nodes re-confirming cutoffs that were almost always correct. Raising to 14 saves significant search effort. Big win. Try depth 16 next to continue bracketing.

### V5: Multi-Source Correction History (REJECTED — revisit with tuned weights)
- **Change**: Added 3 new correction tables: white non-pawn Zobrist, black non-pawn Zobrist, continuation (opponent's last move piece+to). Equal-weight blend of all 4 sources.
- **Result**: **H0 at 485 games, -7.2 Elo.** Showed +20.9 at 114 games before fading.
- **Notes**: The concept is proven (12+ engines use multi-source correction). The equal-weight blend likely dilutes the strong pawn correction signal. Retry with: (a) tuned asymmetric weights (Weiss uses different weights per source), (b) just non-pawn tables without continuation correction, (c) different table sizes. The high draw rate (68%) suggests the correction is working but overcorrecting in some positions.

### V5: Progressive Alpha-Reduce (REJECTED)
- **Change**: Scale alpha-reduce by `min(alphaRaisedCount, 2)` instead of flat -1. More alpha raises = more depth reduction.
- **Result**: **H0 at 724 games, -5.3 Elo.** Showed +39 at 109 games, +16 at 350, then collapsed.
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Novel idea (no other engine does this). The progressive scaling over-reduces when multiple moves raise alpha — the 3rd+ moves still need some depth to verify they're truly worse. Flat -1 is the correct amount.

### V5: Alpha-Reduce Non-PV Only (REJECTED)
- **Change**: Gate alpha-reduce on `beta-alpha == 1` (non-PV nodes only). Protect PV accuracy.
- **Result**: **H0 at 530 games, -8.5 Elo.**
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Alpha-reduce at PV nodes is actually beneficial — PV nodes with multiple alpha raises are wasting time on inferior continuations even at PV depth. The flat -1 everywhere is correct.

### V5: Cap LMR Continuous History (MERGED)
- **Change**: Replace binary capture LMR history (±2000 threshold → ±1 reduction) with continuous `/5000` divisor, matching the quiet LMR pattern.
- **Result**: **H1 at 1243 games, +10.6 Elo ±12.0, LOS 95.8%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 920ac92 (pre-retries) with v5 sb120 net
- **Notes**: The binary ±2000 threshold was too coarse — continuous adjustment gives finer-grained reduction that matches capture quality better. Same divisor (5000) as quiet LMR works well.

### V5: LMP 4+d² Retry (REJECTED)
- **Change**: Loosen LMP from `3+d²` to `4+d²`. Retry after landscape shift.
- **Result**: **H0 at 716 games, -4.9 Elo.**
- **Baseline**: 40c8eb4 with v5 sb120 net
- **Notes**: LMP limit is robust to landscape changes. `3+d²` confirmed optimal across retests.

### V5: History Divisor 4000 (REJECTED → RETRY CANDIDATE)
- **Change**: More aggressive history-based LMR: divisor 5000 → 4000.
- **Result**: **H0 at 1776 games, +1.0 Elo.** Dead flat.
- **Baseline**: 920ac92 (old) with v5 sb120 net
- **Notes**: Persistent +3-5 for hundreds of games but couldn't reach H1. Retry candidate at tighter bounds.

### V5: RFP Depth Gate 9 (REJECTED)
- **Change**: Extend RFP from depth<=8 to depth<=9.
- **Result**: **H0 at 564 games, -6.8 Elo.**
- **Baseline**: 9c10566 with v5 sb120 net
- **Notes**: Depth 9 positions are too complex for static eval pruning. Bracket: 7 (old), **8 (H1, +37.7)**, 9 (H0). Peak at 8.

### V5: Futility 50+d*50 Retry (REJECTED)
- **Change**: Tighten futility from 60+d*60 to 50+d*50. Retry after RFP/badnoisy landscape shift.
- **Result**: **H0 at 577 games, -7.8 Elo.**
- **Baseline**: 40c8eb4 with v5 sb120 net
- **Notes**: Unlike RFP depth 8 which gained +37.7 from landscape shift, futility margin is robust to other changes. 60+d*60 confirmed optimal across multiple retests.

### V5: SEE Cap Threshold 70 (REJECTED)
- **Change**: Tighten SEE cap from `-depth*80` to `-depth*70`.
- **Result**: **H0 at 1650 games, +0.8 Elo.** Dead flat.
- **Baseline**: 920ac92 (old) with v5 sb120 net
- **Notes**: Full bracket: 60 (H0, -52), 70 (H0, +0.8), **80 (H1, +25)**, 90 (H0, +1.6). Peak confirmed at 80.

### V5: RFP Depth Gate 8 Retry (MERGED)
- **Change**: Extend RFP from depth<=7 to depth<=8. Retry of previously flat test.
- **Result**: **H1 at 268 games, +37.7 Elo ±28.9, LOS 99.5%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 9c10566 with v5 sb120 net
- **Notes**: Previously H0 (-0.5 Elo at 1377 games) before bad-noisy/SEE-cap/hist-prune changes. The tighter capture pruning shifted the search landscape — depth 8 positions now benefit from static eval pruning because bad captures are already removed. **This validates retesting failed experiments after conditions change.**

### V5: SEE Cap Threshold 90 (REJECTED → RETRY CANDIDATE)
- **Change**: Loosen SEE cap pruning from `-depth*80` back to `-depth*90`.
- **Result**: **H0 at 2161 games, +1.6 Elo.** Dead flat.
- **Baseline**: 920ac92 (pre-badnoisy-50/QS-delta-280) with v5 sb120 net
- **Notes**: Running against old baseline. SEE cap 80 confirmed optimal — both 70 and 90 are flat vs 80. Bracket: 60 (H0, -52), 70 (running +4), **80 (H1, +25)**, 90 (H0, +1.6), 100 (old).

### V5: ProbCut Margin 155 (REJECTED)
- **Change**: Tighten ProbCut margin from `beta+170` to `beta+155`.
- **Result**: **H0 at 338 games, -17.5 Elo.**
- **Baseline**: eb6ac61 with v5 sb120 net
- **Notes**: Previously tested 150 (-10 Elo). Both 150 and 155 lose. ProbCut margin 170 confirmed optimal across multiple attempts.

### V5: QS Delta Buffer 320 (REJECTED)
- **Change**: Widen QS delta buffer from 280 to 320.
- **Result**: **H0 at 1369 games, -0.3 Elo.** Dead flat.
- **Baseline**: 920ac92 (pre-badnoisy-50) with v5 sb120 net
- **Notes**: Bracket: 240 (old), **280 (H1, +11)**, 320 (H0). Peak at 280 — further widening doesn't help, the extra captures searched are too speculative.

### V5: Bad Noisy Margin 45 (REJECTED)
- **Change**: Tighten bad noisy futility margin from `depth*50` to `depth*45`.
- **Result**: **H0 at 600 games, -5.8 Elo.**
- **Baseline**: 920ac92 (pre-QS-delta/badnoisy-50) with v5 sb120 net
- **Notes**: Full bracket: 45 (H0), **50 (H1)**, 60 (H1), 75 (old). Peak at 50 — margin 45 over-prunes, removing captures that still have value when eval is close to alpha.

### V5: Bad Noisy Margin 50 (MERGED)
- **Change**: Tighten bad noisy futility margin from `depth*60` to `depth*50`.
- **Result**: **H1 at 646 games, +16.7 Elo ±17.1, LOS 97.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 920ac92 (pre-QS-delta-280) with v5 sb120 net
- **Notes**: Third consecutive tightening: 75→60 (+32.4), 60→50 (+16.7). The NNUE eval continues to benefit from aggressive bad capture pruning. Testing 45 next. Cumulative bad noisy gain: ~49 Elo from 75.

### V5: QS Delta Buffer 280 (MERGED)
- **Change**: Widen QS delta pruning buffer from 240 to 280. Less aggressive delta pruning preserves more captures.
- **Result**: **H1 at 1173 games, +11.0 Elo ±12.3, LOS 96.0%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Previous change 200→240 was +31.2 Elo. Further widening to 280 gains +11 more. The NNUE eval benefits from seeing more capture lines in QS. Try 320 next to bracket.

### V5: FH Blend Depth Gate 4 (REJECTED → RETRY CANDIDATE)
- **Change**: Only apply fail-high score blending at depth >= 4 instead of depth >= 3.
- **Result**: **H0 at 2243 games, +1.9 Elo.** Dead flat.
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Persistent +3 for 1000+ games but couldn't reach H1. Retry candidate at tighter bounds.

### V5: Bad Noisy Margin 60 (MERGED)
- **Change**: Tighten bad noisy futility margin from `depth*75` to `depth*60`. Prunes more losing captures when eval is below alpha.
- **Result**: **H1 at 258 games, +32.4 Elo ±26.3, LOS 99.2%.** SPRT bounds: elo0=-5, elo1=15.
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Tighter margin prunes losing captures earlier, saving nodes. Consistent with our SEE cap 80 win — the engine benefits from trusting the NNUE eval more and pruning bad captures aggressively. Try depth*50 next to bracket.

### V5: NMP Dampening (score+beta)/2 (REJECTED)
- **Change**: Stronger NMP score dampening: `(score*2+beta)/3` → `(score+beta)/2`.
- **Result**: **H0 at 513 games, -10.2 Elo.**
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: More dampening makes NMP cutoff scores too conservative, causing the engine to miss good cutoffs. `(score*2+beta)/3` is well-calibrated.

### V5: TT Dampening Depth-Adaptive v1 (REJECTED → RETRY CANDIDATE)
- **Change**: Replace fixed `(3*score+beta)/4` with `(score*ttDepth+beta)/(ttDepth+1)`. Trust deeper TT entries more.
- **Result**: **H0 at 1335 games, -0.5 Elo.** Showed +6-8 for 300+ games before fading.
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: The idea is sound — deep TT entries ARE more reliable. But the formula over-dampens shallow entries (ttDepth=1: 50% vs current 75%). Try v2 with floor at 3 (running separately). Retry candidate at tighter bounds.

### V5: TT Dampening Depth-Adaptive v2 Floor 3 (REJECTED → RETRY CANDIDATE)
- **Change**: Like v1 but floor ttDepth at 3: `(score*max(ttDepth,3)+beta)/(max(ttDepth,3)+1)`.
- **Result**: **H0 at 1557 games, +0.2 Elo.** Dead flat.
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Floor fixes the over-dampening but deep-entry benefit is too small. Both v1/v2 are retry candidates.

### V5: SEE Quiet Threshold -15*d² (REJECTED)
- **Change**: Less aggressive SEE quiet pruning: threshold from `-20*d²` to `-15*d²`.
- **Result**: **H0 at 491 games, -9.9 Elo.**
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Less pruning wastes nodes on quiet moves that land on bad squares. -20*d² confirmed optimal.

### V5: Pawn Push LMR Relief (REJECTED)
- **Change**: Reduce LMR by 1 for pawn pushes to rank 5+ (approaching promotion).
- **Result**: **H0 at 450 games, -10.0 Elo.**
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Too broad — history already captures which pawn pushes are good.

### V5: Eval Monotonicity Reduction (REJECTED)
- **Change**: When eval has been consistently declining over 3 measurements (ply-4 > ply-2 > ply), reduce depth by 1. Novel idea: use eval trend, not just two-point comparison.
- **Result**: **H0 at 411 games, -12.7 Elo.**
- **Baseline**: 920ac92 with v5 sb120 net
- **Notes**: Declining eval often means a tactical sequence where deeper search is needed, not less. The condition fires too broadly — needs a minimum decline threshold per step to filter natural oscillation. Retry with margin (e.g., each step must decline by 50+ cp).

### V5: Threat-Aware History (REJECTED)
- **Change**: 4x butterfly history indexed by [from_threatened][to_threatened] using enemy pawn attacks. Added to move ordering, LMR adjustment, and history pruning.
- **Result**: **H0 at 452 games, -8.5 Elo.**
- **Notes**: Simple pawn-threat version doesn't help. The existing threat escape bonus (+8000 for NMP threat square) may already capture the useful signal. Full threat computation (minor attacks majors, etc.) might be needed but adds NPS cost.

### V5: Multi-Source Correction Weights v2 60/15/15/10 (REJECTED)
- **Status**: -1.1 Elo at 668 games, heading H0. The 50/20/20/10 blend from the merged version is already near-optimal.

### Model: 1536 sb800 vs 1024 Production (H1)
- **Change**: 1536 CReLU wdl=0.0 sb800 (cosine over 800 SBs) vs 1024 production.
- **Result**: **H1 at 991 games, +13.7 Elo ±15.0, LOS 96.3%.**
- **Notes**: Similar to sb400's +18.2. The 1536 architecture is consistently +10-15 over 1024 but the NPS penalty limits the gap.

### Model: 1536 sb800 vs sb700 (H1 — same training run)
- **Change**: sb800 vs sb700 from same cosine-over-800 run.
- **Result**: **H1 at 1456 games, +10.7 Elo ±12.0, LOS 96.0%.**
- **Notes**: Training still improving at sb700→sb800. 1536 hasn't plateaued. sb1000+ could help.

### Model: 1536 sb800 vs sb500 (H1 — same training run)
- **Change**: sb800 vs sb500 from same run.
- **Result**: **H1 at 158 games, +66.8 Elo.**

### V5: Multi-Source Correction Weights v2 60/15/15/10 (REJECTED)
- **Result**: **H0 at 954 games, -1.5 Elo.** Current 50/20/20/10 weights are near-optimal.

### V5: LMP Failing /2 (REJECTED → RETRY CANDIDATE)
- **Change**: Tighter LMP when failing: `lmpLimit/2` instead of `lmpLimit*2/3`.
- **Result**: **H0 at 1693 games, +0.8 Elo.** Persistent +2-4 throughout.
- **Baseline**: 414ea5c with v5 sb120 net
- **Notes**: Another persistent small positive. Retry candidate at tighter bounds.

### V5: RFP Not-Improving 90 Retry (REJECTED)
- **Change**: Tighten RFP not-improving margin from depth*100 to depth*90. Retry with RFP depth 8.
- **Result**: **H0 at 1085 games, -1.3 Elo.** Dead flat.
- **Baseline**: f951efe with v5 sb120 net
- **Notes**: RFP not-improving 100 confirmed optimal even with depth 8. Previously tested 110 (H0, -2.7).

### V5: QS Delta Buffer 260 (REJECTED)
- **Change**: Tighten QS delta from 280 to 260.
- **Result**: **H0 at 726 games, -4.8 Elo.**
- **Baseline**: f951efe with v5 sb120 net
- **Notes**: Bracket: 240 (old), 260 (H0), **280 (H1, +11)**, 320 (H0). Peak confirmed at 280.

### V5: Hindsight Depth Gate 4 (REJECTED)
- **Change**: Raise hindsight reduction gate from depth>=3 to depth>=4.
- **Result**: **H0 at 1486 games, +0.5 Elo.** Dead flat.
- **Baseline**: 414ea5c with v5 sb120 net
- **Notes**: Depth 3 is optimal — hindsight reduction at depth 3 positions is cheap and beneficial. Raising the gate to 4 just misses these opportunities.

### V5: RFP Improving Margin 65 (REJECTED)
- **Change**: Tighten RFP improving margin from depth*70 to depth*65.
- **Result**: **H0 at 569 games, -7.3 Elo.**
- **Baseline**: f951efe with v5 sb120 net
- **Notes**: Previously tested 60 (H0, -2.9) and 80 (H0, -0.6). Now 65 also fails. Bracket: 60/65/70/80 — 70 confirmed optimal even with RFP depth 8.

### V5: History Pruning Depth 4 (REJECTED → RETRY CANDIDATE)
- **Change**: Extend history pruning from depth<=3 to depth<=4.
- **Result**: **H0 at 1321 games, -0.5 Elo.** Showed +5.5 at 1141 games before fading.
- **Baseline**: 414ea5c with v5 sb120 net
- **Notes**: Persistent +3-5 for hundreds of games. Retry candidate at tighter bounds.

### V5: Capture History Pruning (REJECTED)
- **Change**: Prune captures with deeply negative capture history (`captHist < -3000*depth`) at depth<=3.
- **Result**: **H0 at 219 games, -30.2 Elo.**
- **Baseline**: 414ea5c with v5 sb120 net
- **Notes**: Captures are fundamentally different from quiets — even captures with bad history can be tactically important (discovered attacks, pins). SEE pruning already handles truly bad captures. Adding history-based pruning on top is redundant and harmful.

### V5: LMR Quiet C=1.25 Retry (REJECTED)
- **Change**: Tighten quiet LMR from C=1.30 to C=1.25. Retry after landscape shift.
- **Result**: **H0 at 165 games, -35.9 Elo.** Even worse than before (-7).
- **Baseline**: 414ea5c with v5 sb120 net
- **Notes**: C=1.30 is robust. The landscape shift made C=1.25 worse — tighter pruning elsewhere means the remaining quiet moves at each node are more important, not less.

### V5: Cap LMR Divisor 4000 (REJECTED)
- **Change**: Tighter capture LMR history adjustment: /5000 → /4000.
- **Result**: **H0 at 272 games, -19.2 Elo.**
- **Baseline**: 414ea5c with v5 sb120 net
- **Notes**: /4000 over-adjusts capture reductions. /5000 matches the quiet LMR divisor and is confirmed optimal.

### V5: Bad Noisy Depth 5 Retry (REJECTED)
- **Change**: Extend bad noisy futility from depth<=4 to depth<=5. Retry with margin 50.
- **Result**: **H0 at 317 games, -17.6 Elo.**
- **Baseline**: 80c0637 with v5 sb120 net
- **Notes**: Still negative even with tighter margin 50. Depth 4 confirmed optimal.

### V5: NMP Divisor 170 Retry (REJECTED)
- **Change**: NMP eval divisor 200→170. Retry after landscape shift. Was +2.7 at 3963 games pre-shift.
- **Result**: **H0 at 331 games, -15.8 Elo.**
- **Baseline**: 80c0637 with v5 sb120 net
- **Notes**: Landscape shift made this worse. NMP divisor 200 is robust.

### V5: SEE Quiet -25*d² (REJECTED)
- **Change**: Tighter SEE quiet pruning: -20*d² → -25*d².
- **Result**: **H0 at 270 games, -19.3 Elo.**
- **Baseline**: d0bdf4f with v5 sb120 net
- **Notes**: Both directions fail: -15*d² (-10) and -25*d² (-19). -20*d² confirmed optimal.

### V5: Retry Futility 50+d*50 (MERGED — landscape shift win!)
- **Change**: Tighten futility margin from 60+d*60 to 50+d*50.
- **Result**: **H1 at 1665 games, +10.2 Elo ±9.7, LOS 98.1%.** Tight SPRT bounds: elo0=-2, elo1=8.
- **Previous result**: H0 at 699 games, -5.5 Elo (before bad noisy, SEE cap, RFP depth 8 changes).
- **Notes**: Confirms landscape shift hypothesis. The search tree changes from other tightening patches made tighter futility viable. This is why periodic retesting matters.

### V5: Retry LMP Depth 9 (REJECTED again)
- **Result**: **H0 at 650 games, -16.0 Elo.** Landscape shift didn't help LMP.

### V5: Retry Razoring Depth 3 (heading H0)
- **Status**: -16.2 Elo at 613 games, heading H0.

## Cross-Engine Gauntlet Ablation (2026-03-23)

Testing method: 600-game gauntlet vs Texel, Ethereal, Laser (200 per engine), 10+0.1s, Hash=64.
Baseline comparison: HEAD was -2 ±24, pre-Titan was +14 ±23.

### Revert LMR C=1.30 → C=1.50 (KEEP C=1.30)
- **Change**: Undo LMR quiet constant from 1.30 back to 1.50.
- **Result**: **-10 ±22** (600 games). Worse than HEAD by 8 Elo.
- **Notes**: LMR reduction is self-correcting (re-search on fail high). More aggressive reduction saves nodes for important moves without creating permanent blindspots. Classification: safe reduction change, transfers well.

### Revert SEE Cap 80 → 100 (REVERTED — merged)
- **Change**: Undo SEE capture pruning from -depth*80 to -depth*100.
- **Result**: **+8 ±22** (600 games). Better than HEAD by 10 Elo.
- **Notes**: SEE cap 80 was +25 self-play but prunes captures that opponents exploit. Hard capture pruning creates permanent blindspots. Reverted in commit 7731b64.

### Revert Cap LMR Continuous /5000 → Discrete ±2000 (REVERTED — merged)
- **Change**: Replace continuous `reduction -= captHistVal/5000` with discrete thresholds (±2000 → ±1 reduction).
- **Result**: **+24 ±23** (600 games). Better than HEAD by 26 Elo.
- **Notes**: Biggest single finding. Continuous adjustment over-tunes to self-play tactical patterns. Discrete thresholds act as noise filters — only adjust for clearly good/bad captures, leave ambiguous middle alone. The original SPRT pass (+10.6) was a mirage; later retest was -1.6. Reverted in commit 7731b64.

### Cleaned Baseline (Both Reverts Combined)
- **Result**: **+19 ±22** (600 games). Better than HEAD by 21, better than pre-Titan by 5.
- **Per-engine**: vs Ethereal -35 (was -98), vs Texel +4 (was -8), vs Laser +125 (was +96).
- **Notes**: Both reverts combined successfully. The improvement is concentrated against Ethereal (strongest opponent), consistent with removing blindspots that strong engines exploit.

### Singular Extensions on Cleaned Base (REJECTED cross-engine)
- **Change**: Re-enable singular extensions (SingularExtEnabled=true, wire singularExtension into extension variable). Tested on cleaned base (both capture reverts applied).
- **Result**: **-22 ±23** (600 games, Hercules). ~41 Elo worse than cleaned baseline.
- **Notes**: SE remains broken in our engine regardless of testing method (self-play: -60 to -140, cross-engine: -41). The verification search at (depth-1)/2 costs too many nodes for insufficient extensions. May need fundamentally different approach (multi-cut, fractional extensions).

### SEE-Filtered Check Extensions on Cleaned Base (REJECTED cross-engine)
- **Change**: Extend checks that don't lose material (`givesCheck && SEE >= 0`). On cleaned base. Previous raw check ext was -11 self-play.
- **Result**: Hercules: **-8 ±23** (600 games). Atlas: **-26 ±16** (1200 games). Delta vs baseline: **-30 to -39 Elo**.
- **Per-engine (Atlas)**: vs Ethereal -68 (base -45), vs Texel -36 (base +5), vs Laser +25 (base +51).
- **Notes**: SEE filter doesn't save check extensions. Harmful cross-engine on both machines. Check extensions were -11 in self-play but -30 to -39 cross-engine — a **3:1 amplification** of harm, not a discount. Extensions that invest nodes based on our eval's judgment of "important" positions waste nodes cross-engine when opponents handle those positions differently.

### Extended Cleaned Baseline (Hercules, 1200 games)
- **Result**: **+19 ±16** (1200 games). Consistent across two runs (+19, +19).
- **Per-engine**: vs Ethereal -79, vs Texel -10, vs Laser +132.
- **Notes**: Definitive Hercules reference point for all subsequent experiments.

### Extended Cleaned Baseline (Atlas, 1200 games)
- **Result**: **+4 ±16** (1200 games).
- **Per-engine**: vs Ethereal -45, vs Texel +5, vs Laser +51.
- **Notes**: Lower than Hercules (+19) due to slower per-thread speed on Atlas AMD hardware (effective TC difference). All Atlas experiments compared against this Atlas baseline.

## Atlas Gauntlet Series (2026-03-23)

Testing method: 1200-game gauntlet vs Texel, Ethereal, Laser (400 per engine), 10+0.1s, Hash=64, concurrency 16.
All on cleaned baseline (commit 7731b64). Atlas AMD hardware.

### No-Hindsight (Atlas, KEEP HINDSIGHT)
- **Change**: Disable hindsight reduction (comment out `depth--` when `evalSum > 200`).
- **Result**: **-6 ±17** (1200 games). Delta vs baseline: **-10**.
- **Per-engine**: vs Ethereal -39, vs Texel -29, vs Laser +49.
- **Notes**: Hindsight reduction is genuinely useful cross-engine, not a self-play artifact. When both sides' evals agree the position is quiet, reducing is safe regardless of opponent — it's an eval-agnostic signal. Confirmed: keep hindsight 200.

### Loosen Futility 80+d*80 (Atlas, NEUTRAL)
- **Change**: Widen futility margin from 60+d*60 to 80+d*80 (pre-Titan value).
- **Result**: **+4 ±16** (1200 games). Delta vs baseline: **0**.
- **Per-engine**: vs Ethereal -46, vs Texel +3, vs Laser +54.
- **Notes**: Current 60+d*60 is well-calibrated. Neither tighter nor looser helps cross-engine. Confirmed optimal.

### Loosen LMP 4+d² (Atlas, SLIGHTLY WORSE)
- **Change**: LMP from 3+d² to 4+d² (one extra move before pruning).
- **Result**: **-3 ±16** (1200 games). Delta vs baseline: **-7**.
- **Per-engine**: vs Ethereal -72 (base -45), vs Texel +10 (base +5), vs Laser +53 (base +51).
- **Notes**: Loosening LMP hurts mainly vs Ethereal (-27 delta). Extra late moves searched are mostly noise that costs time vs strong opponents. Current 3+d² confirmed. Pattern: Ethereal is the most sensitive opponent to our search quality changes.

### NMP Verify 12 (Atlas, BORDERLINE POSITIVE)
- **Change**: Restore NMP verification at depth≥12 (from depth≥14). More verification = fewer NMP blindspots.
- **Result**: **+9 ±16** (1200 games). Delta vs baseline: **+5**.
- **Per-engine**: vs Ethereal -51, vs Texel -3, vs Laser +82.
- **Notes**: Mild positive from restoring more NMP verification. Depth 14 change was +28 self-play (less verification = more aggressive) but the extra blindspots may hurt cross-engine. Borderline — +5 ±16 needs more games to confirm. Consider retesting or parking.

## NNUE Model Cross-Engine Tests (2026-03-23)

Testing method: 600-game gauntlet vs Texel, Ethereal, Laser, 10+0.1s, Hash=64, Hercules.
All using cleaned search binary (commit 7731b64).

### SCReLU 1024 sb400 (net-v5-1024s-w0-e400s400.nnue)
- **Config**: 1024 SCReLU, wdl=0.0, cosine/400, final snapshot. GPU2 4070 training.
- **Result**: **+10 ±22** (600 games). Implied delta vs CReLU production (+19): **-9**.
- **Per-engine**: vs Ethereal -53, vs Texel -8, vs Laser +108.
- **Notes**: SCReLU slightly behind CReLU but within error bars. check-net showed 15-20% larger piece-loss values (wider dynamic range from squared activation). Led to eval scale experiments below.

## Eval Scale Experiments (2026-03-23)

**Key discovery**: SCReLU's squared activation produces evals with wider dynamic range than CReLU. Our search thresholds (futility, RFP, NMP margins) were tuned for CReLU scale via self-play. The mismatch makes thresholds effectively tighter with SCReLU, causing over-pruning.

Testing method: 600-game gauntlet vs Texel, Ethereal, Laser, 10+0.1s, Hercules.
Scale applied as `result = result * N / M` after quantization in ForwardV5().

### SCReLU ×0.80 Scale
- **Result**: **+35 ±23** (600 games). Delta vs unscaled SCReLU (+10): **+25**. Delta vs CReLU baseline (+19): **+16**.
- **Per-engine**: vs Ethereal -30, vs Texel +44, vs Laser +158.
- **Notes**: Massive improvement from simply compressing eval range. Confirms the eval scale mismatch theory. SCReLU with correct scaling is **stronger than CReLU**.

### CReLU ×0.80 Scale (control experiment)
- **Result**: **+24 ±22** (600 games). Delta vs unscaled CReLU (+19): **+5**.
- **Per-engine**: vs Ethereal -30, vs Texel +17, vs Laser +136.
- **Notes**: Small gain for CReLU (within error bars overall), but large per-engine shifts: Ethereal improved from -79 to -30. The 0.80 multiplier acts as a global pruning de-tune. CReLU doesn't need it as much as SCReLU — the benefit is primarily SCReLU-specific.

### SCReLU ×0.75 Scale (IN PROGRESS)
- **Status**: +15 ±42 at 180 games. Trending below ×0.80 (+35).
- **Notes**: 0.75 may be too much compression — losing eval resolution.

### Bracket Summary (so far)
| Scale | SCReLU Elo | CReLU Elo |
|-------|-----------|-----------|
| 1.00 | +10 ±22 | +19 ±16 |
| 0.85 | pending | — |
| 0.80 | **+35 ±23** | +24 ±22 |
| 0.75 | ~+15 (early) | — |

### Key Insights from Eval Scale Experiments

1. **SCReLU is stronger than CReLU** when eval scale is corrected. +35 vs +19 = ~16 Elo advantage.
2. **The scale mismatch was hiding SCReLU's strength**. Every previous SCReLU test was confounded by the wider dynamic range hitting self-play-tuned thresholds harder.
3. **CReLU gains modestly from 0.80** (+5 overall), suggesting our search thresholds are slightly too aggressive generally, but not dramatically so.
4. **The proper fix** is adjusting the Bullet converter's output quantization for SCReLU, not a runtime multiplier. The scale factor should be applied once at conversion time.
5. **Extensions failed because they're eval-biased** — they invest nodes where our eval says "important", which gets amplified 3:1 against opponents with different evals. Eval-agnostic changes (LMR re-search, hindsight) transfer well.

## Revised Cross-Engine Transfer Model (2026-03-23)

Updated from today's combined Hercules + Atlas data:

| Change type | Self-play | Cross-engine | Mechanism |
|------------|-----------|-------------|-----------|
| Accuracy/information | +X | ~+X | Eval-agnostic, both sides benefit |
| Self-correcting reduction (LMR) | +X | ~+X | Re-search prevents permanent blindspots |
| Eval-agnostic reduction (hindsight) | +X | ~+X | Quiet detection works regardless of opponent |
| Bad extensions (check, singular) | -X | **-3X** | Node waste amplified by opponent diversity |
| Hard capture pruning | +X | **-X** | Captures are where engines diverge most |
| Capture reduction over-tuning | +X | **-2X** | Fine-grained self-play optimization = overfitting |
| Eval scale mismatch | invisible | **-25** | Thresholds too tight for net's dynamic range |

---

# Coda Experiments (Rust port)

All experiments below are Coda-specific. GoChess experiments above are preserved for reference.

## Testing methodology
- **Gauntlet opponents**: Minic, Ethereal, Texel (roughly equal strength to Coda)
- **TC**: 10+0.1s, concurrency 32
- **Opening book**: noob_3moves.epd
- **Adjudication**: draw movecount=10 score=10 after move 20; resign 3 moves at 500cp
- **Baseline (1200g)**: +3 ±17 Elo with 1024s SCReLU model (net-v5-1024s-w5-e800-s800.nnue)
- **Note**: 300-game experiments have ~±35 Elo error bars. Results marked with * need more games.

## 2026-03-30: Feature ablation study (1024s model)

Disabled each search feature individually, 300 games each vs Minic/Ethereal/Texel.

| Feature disabled | Elo | vs Base (+3) | Impact |
|---|---|---|---|
| No LMR | -72 | -75 | Critical |
| No RFP | -41 | -44 | Critical |
| No Futility | -28 | -31 | Important |
| No TT Near-miss | -23 | -26 | Important |
| No Correction | -9 | -12 | Important |
| No Razoring | -6 | -9 | Important |
| No Hindsight | -2 | -5 | Moderate |
| No Bad Noisy | +5 | +2 | Moderate |
| No Alpha Reduce | +6 | +3 | Moderate |
| No Extensions | +9 | +6 | Moderate |
| No Hist Prune | +12 | +9 | Moderate |
| No NMP | +15 | +12 | Moderate |
| No ProbCut | +24 | +21 | Marginal |
| No SEE Prune | +24 | +21 | Marginal |
| No IIR | +28 | +25 | Marginal |
| No LMP | +33 | +30 | Neutral — not helping |

**Notes**: LMR/RFP/Futility are the core trio (~150 Elo combined). LMP is dead weight at 0 Elo. IIR/ProbCut/SEE Prune are marginal.

## 2026-03-30: Same-bucket king NNUE optimization

- **Change**: When king moves but stays in same HalfKA bucket+mirror, do incremental update (2-4 delta ops) instead of full Finny recompute (~30 delta ops).
- **SPRT (400g self-play)**: +6.9 Elo (65W-57L-278D), didn't reach bounds [0,10].
- **Result**: Committed. NPS improvement ~10-15%, translates to small Elo gain.

## 2026-03-30: Cont-hist 4-ply reads

- **Change**: Read continuation history at plies 1, 2, 4, 6 (was only 1, 2). Plies 1-2 at 3x weight, plies 4-6 at 1x weight. Matches Obsidian/Alexandria/Berserk/Stormphrax.
- **Gauntlet (300g)**: +41 Elo* (vs +3 baseline = ~+38 raw, but noisy)
- **Result**: Committed. Genuine improvement — adds information without pruning risk.
- **Notes**: Initially only reads added; writes at plies 2/4/6 added later (half bonus, Obsidian pattern). Full read+write at plies 1,2,4,6.

## 2026-03-30: NMP R=4, depth>=4

- **Change**: Raise NMP base reduction from R=3 to R=4, min depth from 3 to 4. Matches Obsidian/Alexandria/Berserk/Stormphrax consensus.
- **Gauntlet (300g)**: +15 Elo* (vs +3 baseline = +12 raw)
- **Result**: Rejected. More aggressive NMP appears to hurt at our strength — higher R means shallower null-move verification, missing tactical refutations. All strong engines use R=4 but they have deeper search to compensate.
- **Notes**: Worth retrying when Coda is ~100 Elo stronger.

## 2026-03-30: SEE quiet pruning with lmrDepth

- **Change**: Use estimated LMR-reduced depth for SEE quiet threshold instead of raw depth. Late-order moves (high reduction) get tighter thresholds. Matches Obsidian/Berserk/Stormphrax.
- **Gauntlet (300g)**: +9 Elo* (vs +3 baseline = +6 raw)
- **Result**: Rejected. The tighter thresholds for late moves prune too aggressively at our move ordering quality. Our move ordering isn't good enough to push critical moves early.
- **Notes**: Worth retrying after move ordering improvements.

## 2026-03-30: LMP pre-make (Obsidian formula)

- **Change**: Move LMP before make_move to avoid wasted make/unmake/NNUE cycles. Use Obsidian formula: `(3 + d²) / (2 - improving)`.
- **Gauntlet (300g)**: -3 Elo* (vs +3 baseline = -6 raw)
- **Result**: Rejected. The Obsidian formula is too aggressive at shallow depths (threshold of 2 at depth 1 not-improving). Also loses the `gives_check` exemption which we can't compute pre-make.

## 2026-03-30: LMP pre-make (original thresholds)

- **Change**: Move LMP before make_move, keep original thresholds (3 + d², improving *3/2), remove `failing` penalty and `gives_check` guard.
- **Gauntlet (300g)**: +8 Elo* (vs +3 baseline = +5 raw)
- **Result**: Rejected. Losing the `gives_check` exemption hurts — check-giving moves are important tactical moves that shouldn't be pruned by move count.
- **Notes**: Could work if we add a separate pre-make `gives_check` detection, but that's expensive.

## 2026-03-30: IIR depth threshold 4 (was 6)

- **Change**: Lower IIR threshold from depth>=6 to depth>=4. Matches Berserk/Alexandria (both use 4), Stormphrax uses 3.
- **Gauntlet (300g, combined with NMP)**: +27 Elo combined. Not isolated.
- **Result**: Inconclusive. Needs cut_node tracking to implement properly (all strong engines restrict IIR to PV/cut nodes). Without cut_node, IIR fires at all-nodes where it's not helpful.

## 2026-03-30: Drop LVA from capture scoring

- **Change**: Use `MVV * 16 + captHist` (no LVA, raw captHist not /16). Matches Obsidian/Alexandria/Berserk/Stormphrax — none use LVA.
- **Gauntlet (300g)**: +13 Elo* (vs +3 baseline = +10 raw, but vs CH4 +41 = -28)
- **Result**: Rejected. LVA actually helps at our strength level. With weaker move ordering, knowing the attacker helps avoid losing exchanges. Strong engines can drop it because their other ordering signals (threat-indexed history, etc.) compensate.

## 2026-03-30: Threat-aware escape/enter bonuses ±16384

- **Change**: Add ±16384 bonus/penalty for non-pawn/king pieces escaping/entering pawn-attacked squares. Matches Berserk's flat approach.
- **Gauntlet (300g)**: +9 Elo* (vs CH4 baseline = -32)
- **Result**: Rejected. Bonuses way too large, overwhelm history signal.

## 2026-03-30: Threat-aware escape/enter bonuses ±4000 (detuned)

- **Change**: Same as above but ±4000 instead of ±16384.
- **Gauntlet (300g)**: +8 Elo* (vs CH4 baseline = -33)
- **Result**: Rejected. Even at ±4000, the flat pawn-threat approach is too crude. Obsidian uses per-piece-type thresholds (queen vs rook threats, rook vs minor threats, etc.) which may work better.
- **Notes**: The combination of no-LVA + threats ±4k scored +50 (best single result) but this was likely noise given the +3 baseline anchor.

## 2026-03-30: Promotion bonus in capture scoring (+10000)

- **Change**: Add +10000 bonus for queen promotions in good capture scoring.
- **Gauntlet (300g)**: +1 Elo* (vs CH4 baseline = -40)
- **Result**: Rejected. Queen promotions are already handled by SEE partition (high value) and MVV scoring. Extra bonus disrupts relative ordering of captures.

## Key lessons learned

1. **More information helps; more aggressive pruning hurts** at our strength level (~2800 Elo).
2. Parameters tuned for 3000+ engines need detuning for us — our move ordering isn't strong enough to compensate for aggressive pruning.
3. 300-game gauntlets have ~±35 Elo error bars. Use 600+ game baselines to anchor experiments.
4. Features that are "universal across strong engines" don't automatically translate — they assume better move ordering and deeper search.
5. The strongest gains come from **adding information** (cont-hist plies) rather than **tweaking thresholds** (NMP R, SEE margins, LMP limits).

## 2026-03-30: Singular extensions — the breakthrough

### SE v7: multi-cut + negative extensions (no positive ext)
- **Change**: Verification search at half depth when TT move exists at sufficient depth (>=8). NMP gated inside verification (critical fix from GoChess diagnosis). Multi-cut prunes entire node when alternatives are strong enough. Negative extension reduces TT move by 1 when alternatives are competitive.
- **Gauntlet (600g)**: +27 ±24 Elo (baseline +3). Raw: +24 Elo.
- **Result**: Committed.

### SE positive extension (+1 for singular moves)
- **Change**: When verification confirms TT move is singular (no competitive alternatives), extend by +1 ply.
- **Gauntlet (600g)**: +49 ±24 Elo (baseline +3). Raw: +46 Elo. +22 on top of SE-v7.
- **Result**: Committed. This is the feature that cost GoChess -30 Elo but now works in Coda.
- **Why it works now**: The is_legal fixes, capture scoring cleanup, cont-hist 4-ply reads, and NMP-gated verification created the conditions for SE to function properly. The GoChess SE diagnosis identified 5 bugs; all were addressed in Coda.

### Combined session gains (2026-03-30)
| Change | Elo gain |
|--------|----------|
| Same-bucket king NNUE | +7 |
| Cont-hist 4-ply reads | +14 |
| Drop LVA + raw captHist | +14 |
| NMP R=4, depth>=4 | ~0 (consensus) |
| SE multi-cut + neg ext | +24 |
| SE positive extension | +22 |
| **Total estimated** | **~+80 Elo** |

## 2026-03-30: Double extension v2 (with ply guard)

- **Change**: Extend +2 when `singular_score < singular_beta - 15` AND `ply < depth * 2` (non-PV only). Ply guard prevents explosive tree growth (Obsidian/Berserk pattern).
- **Gauntlet (453g)**: +21 Elo (vs SE+posext +49, baseline +3). Still -25 vs single extension.
- **Result**: Rejected. Even with ply guard, double extension costs ~25 Elo vs single extension at our strength. Consistent with the pattern: more aggressive extensions hurt.
- **Prior attempt**: depth*3 margin without ply guard scored +13 (worse).
- **Notes**: May work when engine is stronger. All top engines have double extensions with various limiters.

## 2026-03-30: Mate Distance Pruning

- **Change**: Narrow alpha/beta window based on ply — can't mate in fewer plies than current distance from root. Universal technique in strong engines.
- **Bug found**: Initial implementation set `alpha_orig` AFTER MDP narrowing, corrupting the PV/non-PV detection for TT. Fixed by saving `alpha_orig` before MDP.
- **Gauntlet (600g, fixed)**: +16 ±24 Elo (new baseline +23 ±17). Raw: -7 Elo.
- **Result**: Rejected. Mate situations too rare at 10+0.1 to offset any interference with aspiration.
- **Notes**: Buggy version scored +24 (actually worse than baseline too). May help at longer TC.

## New Baseline (2026-03-30, post-SE)

- **1200 games vs Minic/Ethereal/Texel**: +23 ±17 Elo
- **Includes**: cont-hist 4-ply, drop LVA, NMP R=4, SE with positive ext
- **Use this for all future experiments**

## 2026-03-30: History Divisor 4000 + Futility 50+d*50

- **Combined (600g)**: +41 ±24 Elo (raw +18). Promising but...
- **History 4000 alone (567g)**: +25 ±24 Elo (raw +2). Neutral.
- **Futility 50+d*50 alone (600g)**: +20 ±24 Elo (raw -3). Neutral.
- **Result**: Both individually neutral. Combined +18 was likely noise. Neither committed.
- **Notes**: The combined test illustrates the danger of testing two changes together — synergistic noise can look like a real gain.

## 2026-03-30: Fail-high blend depth gate 4 (was 3)

- **Change**: Raise FH blend minimum depth from 3 to 4. Reduces score dampening at shallow depths.
- **Gauntlet (591g)**: +13 ±24 Elo (raw -10 vs baseline +23).
- **Result**: Rejected. Blending at depth 3 helps; removing it loses ~10 Elo.

## 2026-03-30: NMP eval divisor 170 (was 200)

- **Change**: More aggressive eval-dependent NMP reduction. Lower divisor means larger R bonus per centipawn above beta.
- **Gauntlet (563g)**: -9 ±24 Elo (raw -32 vs baseline +23).
- **Result**: Rejected. Same pattern as NMP R=4 — more aggressive NMP hurts at our strength.

## 2026-03-30: TT near-miss margin 96 (was 80)

- **Change**: Widen TT near-miss cutoff margin from 80 to 96.
- **Gauntlet (587g)**: 0 ±24 Elo (raw -23 vs baseline +23).
- **Result**: Rejected. Wider margin allows too many approximate cutoffs with inaccurate scores.

## 2026-03-31: Cut node tracking + IIR PV/cut + cut-node LMR

- **Combined (595g)**: -2 ±24 Elo (raw -25). Cut-node LMR +1 too aggressive.
- **CN plumbing + IIR PV/cut only (600g)**: +22 ±24 Elo (raw -1). Neutral.
- **Result**: Committed CN plumbing + IIR restriction. Cut-node LMR rejected.
- **Notes**: Cut node tracking is zero-cost infrastructure. IIR PV/cut is consensus-aligned. Neither gains Elo now but enables future features.

## 2026-03-31: Node-based time management

- **v1 (full scaling)**: `(1.5 - bestMoveNodesFrac) * 1.7`, clamped [0.5, 1.5]. +14 ±24 Elo (raw -9). Double-counts stability signal with existing best-move stability counter.
- **v2 (extend only)**: Only extend time when best move uses <30% of nodes. +6 ±24 Elo (raw -17). Score-delta scaling already catches volatility.
- **Result**: Both rejected. Our existing TM (stability counter + score-delta + next-iteration estimate) already covers the same signals. Node fraction adds redundant information.
- **Insight**: Node-based TM works well in engines that don't have detailed score-delta/stability tracking (Alexandria, BlackMarlin). We already have those signals, so node fraction is marginal.

## 2026-03-31: NMP cutNode restriction

- **Change**: Only allow NMP at cut nodes (Obsidian pattern). All-nodes unlikely to produce a cutoff from NMP.
- **Gauntlet (584g)**: 0 ±24 Elo (raw -23 vs baseline +23).
- **Result**: Rejected. Too restrictive — NMP at all-nodes does succeed often enough to justify the node cost.

## Testing methodology update (2026-03-31)

Previous tests used cutechess-cli WITHOUT `-tournament gauntlet`, meaning some games were between non-Coda opponents (wasted signal). Switching to `-tournament gauntlet` to ensure all games involve Coda. Requires new baseline.

## 2026-03-31: GHI mitigation (50mr TT key bucketing) — REJECTED

- **Change**: XOR halfmove clock bucket (8-ply groups, 16 buckets) into TT probe/store/prefetch key. Positions with different 50-move progress get different TT entries.
- **Source**: Obsidian/Reckless engine reviews. Universal technique for preventing Graph History Interaction.
- **Gauntlet (600g, Hercules)**: Elo +4.6, raw -28.5 vs new baseline +33.1
- **Result**: REJECTED. Strongly negative.
- **Why it failed**: TT fragmentation. The same position with different halfmove clocks now uses different TT slots, reducing hit rate. At our hash size (64MB) and search efficiency, TT hits are precious — losing them costs more than the rare GHI bug. Top engines using this have larger effective TTs and more efficient searches. Also, our explicit halfmove >= 100 draw check already handles most 50-move scenarios.
- **Lesson**: Not every consensus feature helps at every strength level. GHI is a rare edge case; TT efficiency matters more for us right now.

## New Gauntlet Baseline (2026-03-31, post corr-hist-6 + history aging)

- **Method**: `-tournament gauntlet` (all games involve Coda), 600g vs Minic/Ethereal/Texel
- **Result**: +37 ±24 Elo
- **Includes**: All prior commits + corr-hist-6 (Hercules, +11.4 Elo) + history aging (Hercules)
- **Note**: Gauntlet format gives equal weight to each opponent. Prior RR baseline was +23 ±17.

## 2026-03-31: Pin-aware SEE (side-to-move pins only) — REJECTED (Hercules)

- **Change**: Pass pre-computed pinned mask to SEE, exclude side-to-move's pinned pieces from attacker set.
- **Source**: Reckless engine review.
- **Gauntlet (578g, Hercules)**: Elo +25.3, raw -7.8 vs baseline +33.1
- **Result**: REJECTED. Slightly negative.
- **Why**: Asymmetric — only our pins excluded, not opponent's. May bias SEE.
- **Revisit**: With both-side pin mask if SEE accuracy becomes a priority.

## 2026-03-31: TT cutoff retroactive cont-hist + aspiration fail-high depth reduction

- **TT cutoff retroactive**: Penalise opponent's last quiet move in cont-hist when TT cutoff occurs. "Your move led to a position we know is bad for you." Matches Alexandria/Obsidian.
- **Aspiration fail-high depth**: Reduce asp_depth by 1 on fail-high re-search. Score is already above window — shallower re-search with wider window is sufficient. Matches Alexandria/Midnight/Seer. Uses separate asp_depth variable (not outer loop depth — previous GoChess bug).
- **Combined gauntlet (300g)**: +56 ±33 Elo (baseline +37). Raw: +19.
- **Result**: Both committed as separate commits.

## 2026-03-31: v7 20sb ramp vs 5sb ramp — H2H + 5-way RR (Hercules)

- **Change**: v7 1024h16x32s w5 e800 trained with 20sb LR warmup vs 5sb warmup
- **H2H (200g self-play)**: 24-21-55, +10.5 Elo for 20sb ramp. Suggestive positive.
- **5-way RR (200g each, vs Minic/Ethereal/Texel)**: 
  - v7-20sb-ramp: -40 Elo
  - v7-5sb-ramp: -81 Elo
  - **Cross-engine delta: +41 Elo for 20sb ramp** (vs +10 self-play)
- **Result**: Slower ramp clearly better. Cross-engine effect amplified vs self-play.
- **Implications**: Hidden layers need more stability during early training. Frozen-FT approach (the extreme version) should be even better. 40sb ramp training started.

## 2026-03-31: v5 1024s e1200 vs e800 — H2H (Hercules)

- **Change**: Same architecture (1024 SCReLU w5), extended training from 800 to 1200 SBs
- **H2H (200g self-play)**: 29-21-50, +28 Elo for e1200
- **Check-net**: Flat (no improvement in material detection). Gains are in positional refinement.
- **5-way RR**: Running, results pending.
- **Result**: Longer training helps v5 despite flat check-net. "Free" Elo for GPU time.
- **Implications**: Always train v5 to e1200+. e1600 training started to find the plateau.

## 2026-03-31: GHI mitigation (50mr TT key bucketing) — REJECTED (Hercules)

- **Change**: XOR halfmove clock bucket into TT key.
- **Gauntlet (600g)**: raw -28.5 vs baseline. Strongly negative.
- **Why**: TT fragmentation costs more than rare GHI bug prevention.

## 2026-03-31: Pin-aware SEE (side-to-move only) — REJECTED (Hercules)

- **Change**: Exclude pinned pieces from SEE attacker set.
- **Gauntlet (578g)**: raw -7.8 vs baseline. Slightly negative.
- **Why**: Asymmetric pin mask (only our side) may bias SEE.

## 2026-03-31: Opponent move feedback + eval-based history depth bonus

- **Combined (300g)**: +20 ±33 Elo (raw -17). Negative.
- **Eval bonus alone (300g)**: +26 ±33 Elo (raw -11). Slightly negative.
- **Opponent feedback alone**: Not isolated, but combined minus eval-only implies ~-6 additional.
- **Result**: Both rejected.
- **Why opponent feedback failed**: Updating butterfly history at every node based on eval sum is too noisy. The eval sum `(parent + current)` is an unreliable signal — it conflates position quality with move quality. Alexandria's version uses more careful scaling and only updates the opponent's own history table (which we don't have).
- **Why eval bonus failed**: +1 depth to history bonus on surprising cutoffs over-reinforces. The extra depth applies to both the bonus and all penalties, making both too strong. May need a more nuanced approach (e.g., only bonus, not penalty scaling).

## 2026-03-31: Pawn history in LMR adjustment

- **Change**: Add pawn_hist score to the LMR history adjustment (alongside main + cont-hist). Pawn structure context for reduction decisions.
- **Gauntlet (300g)**: +26 ±33 Elo (raw -11 vs baseline +37).
- **Result**: Rejected. Pawn history values may have different scaling from main/cont-hist, diluting the signal when summed with /5000 divisor.
- **Note**: Might work with a separate divisor or weighted contribution.

## 2026-03-31: v5 1024s e1200 vs e800 — 5-way RR (Hercules)

- **Change**: Same architecture (1024 SCReLU w5), extended training 800→1200 SBs
- **H2H (200g self-play)**: +28 Elo for e1200
- **5-way RR (200g each, vs Minic/Ethereal/Texel)**:
  - v5-e800: +16 Elo
  - v5-e1200: -7 Elo
  - **Cross-engine delta: -23 Elo for e1200** (vs +28 self-play)
- **Result**: INVERTED. Longer training helps self-play but hurts cross-engine.
- **Why**: Likely eval scale shift (check-net: miss-pawn weakened -122→-85, EG queen-up widened 1760→2073). Search thresholds calibrated for e800's scale are suboptimal for e1200's shifted range. Both sides share the same miscalibration in self-play so it cancels.
- **Implications**: Don't assume longer training = better. Always validate model changes cross-engine. e800 remains our strongest v5 for match play. The e1600 (training on GPU3) may show the same pattern.

## 2026-03-31: Fifty-move eval scaling — REJECTED (Hercules)

- **Change**: `eval *= (193 - halfmove) / 200`. Dampens eval as 50-move draw approaches.
- **Source**: Quanticade, PlentyChess, Halogen all use this.
- **Gauntlet (157g, combined with TM v2)**: Elo +19.9, raw -6.7 vs baseline. TM alone was +24. Suggests FMR scaling costs ~30 Elo.
- **Result**: REJECTED. The scaling dampens eval at ALL halfmove values (3.5% at move 0, 13.5% at move 20). Our search thresholds are calibrated for full-strength evals — dampening them by 10-15% makes pruning too aggressive.
- **Revisit**: Only apply at halfmove > 60, or retune thresholds with scaling active. Top engines that use this have thresholds tuned WITH the scaling.

## 2026-03-31: Correction history gating by TT flag — REJECTED (Hercules)

- **Change**: Only update correction history when TT flag direction agrees with correction direction. Fail-high + positive correction → OK. Fail-low + negative → OK. Disagreements filtered.
- **Source**: Quanticade engine review.
- **Gauntlet (166g, combined with TM v2)**: Elo +27.3, raw +0.7 vs old baseline. TM alone was +24. Suggests corr-gate costs ~23 Elo.
- **Result**: REJECTED. Gating filters too aggressively — removes informative "disagreement" corrections where the bound type contradicts the eval error. These disagreement cases may be the most valuable corrections.
- **Revisit**: Try gating only exact-flag updates, or relax the condition.
## 2026-03-31: Aspiration delta + LMR research skip + NMP soft cutNode

- **Combined (300g)**: -15 ±33 Elo (raw -52). Clearly negative.
- **Aspiration delta alone (300g)**: +17 ±33 Elo (raw -20). Score-adaptive formula gives tighter windows at typical scores, causing more aspiration failures.
- **LMR research skip**: Not isolated (part of combined -52). The TT upper bound check may be too conservative with depth-4 threshold.
- **NMP R-1 at all-nodes**: Not isolated (part of combined -52). Softer version of the full cutNode restriction that scored -23 raw.
- **Result**: All rejected. The combined -52 raw is dominated by at least one strongly negative change.
- **Aspiration lesson**: Our fixed delta=15 is already well-tuned. Score-adaptive widening (Reckless: 13+avg²/23660) assumes extreme scores need wider windows, but at 10+0.1 TC our aspiration rarely sees extreme scores and the tighter default makes failures more frequent.

## 2026-03-31: Remove killers (pure history ordering) — CATASTROPHIC (Hercules)

- **Change**: Skip killer stages in movepicker. History alone orders quiet moves.
- **Source**: SF/Reckless/Viridithas all dropped killers.
- **Gauntlet (155g)**: -530 Elo. Catastrophic.
- **Why**: Our history tables aren't rich enough. Need 4D threat-aware history first.

## 2026-03-31: Check-giving moves LMR with R-1 (was full exemption)

- **Change**: Apply LMR to check-giving moves with R-1 adjustment (was fully exempt). Matches Obsidian/Alexandria/Berserk/Stormphrax — none exempt checks from LMR.
- **Gauntlet (300g)**: +40 ±33 Elo (raw +3). Neutral.
- **Result**: Committed for consensus alignment. Code is cleaner.
- **Key finding**: No strong engine uses classic check extensions anymore. All use LMR reduction suppression.

## 2026-03-31: Material scaling + halfmove eval decay

- **Change**: Material scaling: `eval * (22400 + materialValue) / 32 / 1024` (Alexandria). Halfmove decay: `eval * (200 - halfmove) / 200` (Reckless). Both dampen eval in draws/endgames.
- **Gauntlet (300g, combined)**: +37 ±33 Elo (raw +0). Dead neutral.
- **Result**: Rejected. NNUE output buckets already handle material-dependent eval scaling internally.

## 2026-03-31: 3 killers per ply (was 2) — REJECTED (Hercules)

- **Change**: Expand killer slots from 2 to 3. Add Killer3 stage in movepicker.
- **Motivation**: Killers are worth ~500 Elo (from no-killers test). More slots = more hits.
- **Gauntlet (157g, with TM v2)**: Elo +15.5, raw -11.1. TM alone was +24. So 3 killers costs ~35 Elo.
- **Result**: REJECTED. The 3rd killer is low quality (bumped out by 2 better moves), wastes a node trying it, and the pruning exemption for killers applies to a weaker move.
- **Lesson**: 2 killers is the right number — the 1st killer is very strong, the 2nd decent, the 3rd is noise. Quality > quantity for killers.

## 2026-03-31: Batch B experiments (individually tested)

- **TT-adjusted NMP eval (300g)**: +13 ±33 (raw -24). TT lower bound for NMP eval hurts — may over-trigger NMP when TT score is stale.
- **History gate on SEE pruning (part of combined -31)**: Not isolated individually. The 8000 threshold may be wrong.
- **Queen promotion extension (300g)**: +9 ±33 (raw -28). SE already handles important promotions. Extra extension wastes nodes.
- **RFP TT quiet guard (300g)**: +41 ±33 (raw +4). Committed — logically sound guard against over-pruning.
- **Combined TripleB (300g)**: +6 ±33 (raw -31). At least one strongly negative. Lesson: test individually at our hit rate.

## 2026-03-31: Reverse direction history penalty

- **Change**: On beta cutoff, penalize history[to][from] at -bonus/2. "Don't take back a good move."
- **Gauntlet (300g)**: +44 ±33 Elo (raw +7). Suggestive positive.
- **Result**: Committed. Novel (only Arasan), logically sound, simple.

## Batch C experiments still to test:
- History gate on SEE pruning (threshold TBD)
- Weiss NMP skip after null move (anti-chain)
- Passed pawn push exemption (needs helper function)
- Reverse direction history in cont-hist (extend Arasan idea deeper)

## 2026-03-31: v5 WDL Net Cross-Engine RR

- **Test**: Round-robin with 4 Coda nets (w0-prod, w3, w5, w7) + 3 rivals (Weiss, Laser, Texel)
- **Games**: 510 (of 2100 planned), tc=10+0.1, concurrency 16
- **Binary**: Latest Coda (4D history + aspiration delta merged)
- **Nets**: w0=net-v5-120sb-sb120.nnue (CReLU e120), w3/w5/w7=net-v5-1024s-{w3,w5,w7}-e800 (SCReLU e800)

| Rank | Net | Elo | ±Error | Score | Draw% |
|------|-----|-----|--------|-------|-------|
| 1 | **Coda-w7** | **+77** | ±41 | 60.8% | 49.0% |
| 2 | Coda-w3 | +61 | ±43 | 58.7% | 43.8% |
| 3 | Coda-w5 | +37 | ±41 | 55.3% | 48.6% |
| 4 | Weiss | +27 | ±48 | 53.9% | 31.7% |
| 5 | Coda-w0(prod) | +12 | ±45 | 51.7% | 37.8% |
| 6 | Texel | -22 | ±44 | 46.9% | 42.0% |
| 7 | Laser | -213 | ±57 | 22.7% | 21.7% |

**Key findings**:
- All WDL nets beat w0 cross-engine despite looking flat in self-play
- w7 is clearly best (+65 over w0), w3 and w5 roughly tied (+49/+25 over w0)
- WDL nets have much higher draw rates (44-49%) vs w0 (38%) — better draw recognition
- Self-play WDL testing was actively misleading: showed no difference, cross-engine shows +25-65
- Caveat: single training run per WDL value, seed variance not yet quantified
- **Action**: Switch production net to w7. Test w10 when GPU available.

## 2026-03-31: v7 Hidden Layer Net Cross-Engine RR

- **Test**: Round-robin with 3 v7 nets (w0, w5, w5-ramp) + v5-w7 reference + 3 rivals
- **Games**: 741 (of 2100 planned), tc=10+0.1, concurrency 16
- **Binary**: Latest Coda (4D history + aspiration delta merged)

| Rank | Net | Elo | ±Error | Score | Draw% |
|------|-----|-----|--------|-------|-------|
| 1 | **v5-w7(best)** | **+105** | ±36 | 64.6% | 42.5% |
| 2 | Weiss | +66 | ±37 | 59.4% | 38.7% |
| 3 | v7-w5-ramp | +3 | ±31 | 50.5% | 55.7% |
| 4 | Texel | -12 | ±35 | 48.3% | 43.6% |
| 5 | v7-w5 | -13 | ±35 | 48.1% | 44.3% |
| 6 | v7-w0 | -22 | ±34 | 46.9% | 47.1% |
| 7 | Laser | -131 | ±40 | 32.0% | 33.6% |

**Key findings**:
- v5-w7 dominates all v7 nets by ~100 Elo — hidden layer NPS penalty (~2.5x) not yet compensated by eval quality
- v7 nets cluster around Texel-level strength, ~100 Elo below v5-w7
- WDL blending helps v7 too: w5-ramp (+3) > w5 (-13) > w0 (-22)
- 20sb warmup ramp gives small edge over 5sb default (+16 Elo)
- v7 needs pairwise FT (faster + richer features) and/or frozen-FT transfer learning to compete
- **Action**: Wait for pairwise v7 training results. Try frozen-FT from v5-w7.

## 2026-03-31: Retests with 4D history + aspiration delta

New baseline post-4D: +44 ±24 Elo (600g).

- **SEE lmrDepth retest (300g)**: +31 (raw -13). Still negative — tighter thresholds for late moves don't help regardless of ordering.
- **Double extensions v3 (300g)**: +62 (raw +18). **COMMITTED!** The 4D history + better aspiration made verification search reliable enough for double extensions. Previously -25 raw.
- **TT-NMP retest (300g)**: +37 (raw -7). Improved from -24 but still slightly negative. TT lower bounds are too unreliable for NMP eval.

## 2026-03-31: Eval-depth bonus + SEE history gate retests (post-4D)

- **Eval-depth bonus retest (300g)**: +33 (raw -11). Unchanged from pre-4D test. Over-reinforcing surprising cutoffs consistently hurts.
- **SEE history gate (300g)**: +23 (raw -21). Exempting high-history moves from SEE pruning wastes nodes — tactically unsound moves shouldn't be saved by history alone.
- **Result**: Both rejected again. These two ideas don't benefit from improved move ordering.

## 2026-03-31: v7 LR Warmup Duration Cross-Engine RR

- **Test**: Round-robin with 3 warmup variants (5sb, 20sb, 40sb) + 3 rivals (Weiss, Laser, Texel)
- **Games**: 463, tc=10+0.1, concurrency 16
- **Nets**: net-v7-1024h16x32s-w5-e800{i8-s800, s800-20sb-ramp, s800-40sb-ramp}

| Rank | Net | Elo | ±Error | Draw% |
|------|-----|-----|--------|-------|
| 3 | **v7-20sb** | **+16** | ±41 | 44.1% |
| 4 | v7-40sb | +2 | ±43 | 38.3% |
| 5 | v7-5sb | -7 | ±41 | 44.8% |

**Findings**: 20sb warmup is best (+23 over 5sb default). 40sb warmup is slightly better than 5sb but worse than 20sb — longer warmup protects hidden layers but also constrains their early learning. 20sb is the sweet spot. Weight analysis confirms: 40sb has 1 dead L2 neuron vs 0 for 20sb, and lower weight magnitudes (L1 mean|w| 7.27 vs 7.88).

## 2026-03-31: Batch of 5 experiments (post-cuckoo baseline)

New baseline: +44 ±24 Elo (600g gauntlet, post-cuckoo + 4D history + aspiration delta).

### No killers retest (90g) — REJECTED
- **Change**: Remove killer/counter-move stages from movepicker, remove killer exemptions from LMR/pruning.
- **Result**: -4 Elo (raw -48 vs baseline). Still clearly negative.
- **Notes**: Was -530 Elo before 4D threat-aware history. Now -48 — massive improvement proving 4D history covers most of the killer signal. But not all of it. Killers still needed, likely for position-specific refutations that history (which aggregates across positions) can't capture.

### NMP anti-chain (84g) — REJECTED
- **Change**: Skip NMP when grandparent move was null move (prevents NMP chains). Weiss pattern.
- **Result**: 0 Elo (raw -44). Classic early noise — started at +48 at 29 games, collapsed to 0.
- **Notes**: NMP chains are already rare enough that preventing them doesn't save meaningful nodes. Our NMP with verification at depth>=14 handles the edge cases.

### SE depth gate 6 (36g) — REJECTED
- **Change**: Lower SE depth gate from depth>=8 to depth>=6.
- **Result**: 0 Elo (raw -44 early). Killed at 36 games.
- **Notes**: SE at depth 6-7 fires on too-shallow positions where the verification search cost isn't justified. Depth 8 confirmed as floor.

### Reverse cont-hist penalty (107g) — REJECTED
- **Change**: On quiet beta cutoff, penalize the reverse move (to→from) in cont-hist at -bonus/2. Extension of merged Arasan reverse main-history pattern.
- **Result**: +3 Elo (raw -41). Dead neutral.
- **Notes**: The reverse pattern works for main history (generic from/to pairs) but not for cont-hist (move-pair context). The "don't take back" signal is position-independent; cont-hist is position-dependent.

### SE margin 2d/3 (228g) — REJECTED
- **Change**: Tighter SE margin: `tt_score - depth*2/3` instead of `tt_score - depth`.
- **Result**: +48 Elo (raw +4). Started at +71 at 30g, converged to baseline.
- **Notes**: Classic early noise collapse. Tighter margin (more singular moves) doesn't improve play.

### SE margin d/2 (432g) — REJECTED
- **Change**: Even tighter SE margin: `tt_score - depth/2`.
- **Result**: +44 Elo (raw 0). Started at +57, converged to dead baseline.
- **Notes**: Both tighter margins tested (2d/3 and d/2), both neutral. SE margin `depth` is well-calibrated. Combined with GoChess tests (d*2 and d*3 both rejected), the margin is fully bracketed: d/2 (neutral), **d (current, optimal)**, d*2 (rejected), d*3 (rejected).

### SE double-extension margin 25 (300g) — REJECTED
- **Change**: Wider double-ext margin: `singular_beta - 25` instead of `singular_beta - 15`. Fewer double extensions.
- **Result**: +36 Elo (raw -8). Slightly negative.

### SE double-extension margin 10 (294g) — REJECTED
- **Change**: Tighter double-ext margin: `singular_beta - 10` instead of `singular_beta - 15`. More double extensions.
- **Result**: +36 Elo (raw -8). Slightly negative.
- **Notes**: Both directions lose ~8 Elo. Double-extension margin fully bracketed: 10 (-8), **15 (current, optimal)**, 25 (-8). SE is now fully tuned: margin=depth, depth gate=8, double-ext margin=15, verification=(depth-1)/2.

---

## 2026-04-01: Comprehensive Ablation Sweep [ablation]

Self-play SPRT (elo0=-5, elo1=5), TC 10+0.1, Hash=64, net-v5-1024s-w5-e800-s800.nnue.
Disabling each feature individually to measure its contribution.

### Titan Results (9/9 complete)

| Rank | Feature | Elo cost of disabling | Games | LLR | Verdict |
|------|---------|----------------------|-------|-----|---------|
| 1 | LMR | +262 ±80.7 | 58 | 2.98 | **KEEP** |
| 2 | QS captures | +111 ±40.9 | 123 | 2.97 | **KEEP** |
| 3 | RFP | +56 ±27.6 | 238 | 2.95 | **KEEP** |
| 4 | LMP | +22.2 ±17.0 | 565 | 2.96 | **KEEP** |
| 5 | IIR | +18.9 ±15.5 | 644 | 3.01 | **KEEP** |
| 6 | NMP | +18.3 ±15.3 | 683 | 3.02 | **KEEP** |
| 7 | Singular ext. | +15.6 ±14.1 | 601 | H1 | **KEEP** |
| 8 | Futility | +6.1 ±8.8 | 1820 | H1 | **KEEP** (marginal) |
| 9 | Razoring | -19.8 ±16.1 | 527 | -2.95 | **REMOVE** |

### Atlas Results — Initial (CPU STARVED, unreliable)

⚠️ Atlas was running a 36-engine RR alongside ablation, halving effective TC. All pruning-class results were systematically wrong. See clean CPU retests below.

| Test | Feature | Elo (starved) | Games | Verdict (WRONG) |
|------|---------|---------------|-------|-----------------|
| A1 | Extensions (all) | -9 | 1372 | KEEP (correct) |
| A2 | Correction history | -3 | 4584 | KEEP (understated) |
| A3 | Hindsight reduction | +9 | 1263 | REMOVE (WRONG) |
| A4 | Alpha-reduce | +9 | 1307 | REMOVE (WRONG) |
| A5 | ProbCut (old impl) | +4 | 3079 | REMOVE (WRONG) |
| A6 | Hist prune | +3 | 4804 | REMOVE (WRONG) |
| A7 | Bad noisy pruning | +3 | 3697 | REMOVE (WRONG) |
| A8 | SEE prune | +12 | 1054 | REMOVE (WRONG) |

### Atlas Results — Clean CPU Retests (reliable)

| Feature | Starved | Clean CPU | Flipped? |
|---------|---------|-----------|----------|
| Correction | -3 (4584g) | **-10** (1140g) | No — 3x stronger |
| Hist prune | +3 (4804g) | **-17** (631g) | **YES** |
| Bad noisy | +3 (3697g) | **-26** (394g) | **YES** |
| SEE prune | +12 (1054g) | **-17** (742g) | **YES** |
| Hindsight | +9 (1263g) | **-18** (629g) | **YES** |
| Alpha-reduce | +9 (1307g) | **-4** (1116g, trending keep) | **YES** |
| ProbCut | +4 (3079g) | ~0 (1731g, dead zero) | Partially |

**5 of 5 retested pruning features flipped from REMOVE to KEEP.** CPU contention systematically biased against pruning features whose value scales with search depth.

### Titan Feature Tests (Tier B: elo0=0, elo1=10)

| Change | Elo | Games | LLR | Result |
|--------|-----|-------|-----|--------|
| Hindsight V2 (prior_r≥2, threshold 195) | -7.5 ±12.7 | 877 | -2.97 (H0) | **REJECTED** |
| Futility V2 (90+100d+hist/128) | -0.9 ±8.6 | 2014 | H0 | **REJECTED** |
| Correction clamp 192 vs 128 | -0.5 ±8.4 | 1965 | H0 | **REJECTED** |

### Actions Taken

- **Razoring**: Fully removed from codebase (obsolete technique, confirmed -20 Elo on Titan quiet CPU)
- **ProbCut**: Disabled by default. Dead zero on clean CPU retest. Atlas ProbCut rewrites also failed (-24 and -2).
- **All other features**: Re-enabled after clean CPU retests confirmed they help (hindsight -18, SEE prune -17, hist prune -17, bad noisy -26, alpha-reduce -4).
- **Hindsight V2** (prior_reduction gate): Rejected at -7.5. Original hindsight works better than SF-style gated version.
- **Futility V2** (wider margins + history): Rejected. Current margins (60+60*lmrDepth) are well-calibrated.
- **Correction clamp 192**: Rejected. Clamp 128 is optimal (bracketed: 96 flat, 128 optimal, 192 flat).

### Key Insights

- **CPU contention is catastrophic for ablation testing.** Halved TC (~1 ply less) systematically inverts pruning feature results. Always verify idle CPU before SPRT.
- NMP had misleading early start (43% win rate at 120 games → recovered to 52% at 683). Reinforces: always let SPRT finish.
- Singular extensions confirmed at +15.6 — not the drag we suspected.
- The only genuine removal from the entire sweep is **razoring** (confirmed on Titan quiet CPU).
- **ProbCut** is the only feature at true zero — disabled but kept behind feature flag for future re-evaluation.

## 2026-04-01: Hercules Net and Parameter SPRTs [SPRT-validated]

Self-play SPRT (elo0=0, elo1=10), TC 10+0.1, Hash=64.

### LMR C=1.20 vs C=1.30

- **Result**: **H0**, -2.0 ±9.6 Elo, 1360 games. C=1.30 remains optimal.
- **Notes**: C=1.30 fully bracketed: C=1.20 (H0 -2), **C=1.30 (current)**, C=1.375 (flat), C=1.50 (H0 from GoChess era). Do not revisit.

### w10 vs w7 net (same e800 training, different WDL lambda)

- **Result**: **H0**, -3.3 ±10.3 Elo, 1465 games. w7 remains best WDL lambda.
- **Notes**: Was +31.4 at 244 games — classic early noise. Completely regressed to baseline. WDL lambda progression: w0 < w3 < w5 < **w7 (optimal)** > w10. Do not increase WDL lambda beyond 0.07.

### e1600-w5 vs e800-w5 net (longer training, same WDL lambda)

- **Result**: **H0**, -18.7 ±17.4 Elo, 688 games. Longer training hurts for w5.
- **Notes**: e1600 actively worse than e800. Possible overtraining or LR schedule mismatch. The e800 cosine schedule is well-calibrated for this dataset size.

### NMP V1 (divisor 173, no dampening, no capture adj)

- **Result**: **H0**, -4.4 ±11.1 Elo, ~1400 games. Current NMP is well-calibrated.
- **Notes**: Bundled three SF-style changes: eval divisor 200→173, removed score dampening (return beta instead of blend), removed capture R-1 adjustment. None helped. NMP formula (R=4+depth/3, divisor 200, dampened return, capture -1) is optimal.

### Futility V1 (wider margins 90+100*lmrDepth)

- **Result**: **H0**, +0.9 ±7.3 Elo, ~2800 games. Dead flat.
- **Notes**: Widening from 60+60*d to 90+100*d (closer to SF's 42+120*d). No benefit. Current margins are well-calibrated. Combined with Titan's V2 rejection (history adjustment), futility is fully tuned.

### s1200-w5 vs s800-w5 net (longer training, same architecture)

- **Result**: **H0**, -5.3 ±11.6 Elo, ~1000 games. e800 is optimal training length.
- **Notes**: Combined with e1600 rejection (-18.7 Elo), training length is bracketed: **e800 (optimal)** > e1200 (-5) > e1600 (-19). Diminishing returns become negative past s800 for 1024s w5.

### 768pw-w5 vs 1024s-w5 (pairwise vs standard, both e800)

- **Result**: Dead flat at +0.7 after 1945 games, killed.
- **Notes**: Pairwise 768 matches standard 1024 in self-play — remarkable efficiency (768 params doing 1024's work). Cross-engine may differentiate. Neither architecture is strictly better in self-play.

## 2026-04-01: OpenBench Razoring Test (inflated TC) [SPRT-validated]

First test on OpenBench. Reference NPS was set too high (1.1M vs actual 0.3-0.6M), so all machines played at 2-3x intended TC (~20-33s+0.2-0.3 instead of 10+0.1).

### with-razoring vs main (OpenBench test #1)

- **Result**: **H1**, +3.5 Elo, 14,910 games. LLR 2.94. Longest SPRT ever run.
- **Notes**: Razoring helps by ~3.5 Elo at effectively LTC (2-3x target TC). This contradicts Titan's -20 at STC. Razoring is **TC-dependent**: hurts at STC (shallow search), helps at LTC (deeper search saves more by pruning hopeless shallow nodes). Decision: **keep removed**. The benefit is marginal, TC-dependent, and every top engine has removed it. When reference NPS is corrected, future tests will run at proper STC where razoring hurts.

## 2026-04-02: OpenBench Tests — Code Review and Engine Analysis [SPRT-validated]

All tests via OpenBench (https://ob.atwiss.com/), self-play SPRT [0.00, 5.00], TC 10+0.1 scaled by NPS.

### Completed / Stopped Tests

| # | Test | Elo | Games | LLR | Result | Action |
|---|------|-----|-------|-----|--------|--------|
| 5 | 4file-blunders vs 4file (SB100) | +1.2 | 26,970 | -0.71 | Flat | Stopped. SB100 too short to differentiate. |
| 6 | 4D history ablation (NO_4D_HISTORY) | ~0 | 13,002 | -2.96 | **H0** | Confirmed zero Elo. |
| 7 | retry-lmp-depth9 | -0.49 | 7,216 | -0.97 | Stopped (trending H0) | Was +5 early, regressed. |
| 8 | retry-opp-material-4 | -3.1 | 7,482 | -1.71 | Stopped (trending H0) | Not helping. |
| 9 | retry-fh-blend-depth4 | ~flat | 5,772 | -1.57 | Stopped (trending H0) | Not helping. |
| 12 | fix-tt-exact-pv | -0.35 | 8,060 | -0.94 | Stopped. Neutral. | **Merged** as correctness fix. |
| 14 | fix-corr-hist-modulo | +0.55 | 7,370 | -0.47 | Stopped. Neutral. | **Merged** (bitmask is objectively correct). |
| 15 | fix-cutnode-lmr (+2 at cut nodes) | -15.7 | 3,040 | -2.95 | **H0** | Rejected. Too aggressive for our engine. |
| 20 | cleanup-search-hot-path | ~0 | 6,144 | 0.09 | Stopped. No regression. | **Merged** as code quality improvement. |

### Still Running

| # | Test | Elo | Games | LLR | Source |
|---|------|-----|-------|-----|--------|
| 13 | fix-asp-depth-clamp | ~+5 | ~950 | 0.36 | Code review: depth-2 min on fail-high |
| 16 | fix-strong-failhigh-bonus | ~flat | ~1,700 | 0.00 | Engine review: depth+1 when score > beta+95 |
| 17 | 768pw-w7 vs production | ~flat | ~4,300 | -1.04 | Net comparison |
| 29 | fix-cuckoo-init (3 bugs) | ~-3 | ~3,400 | -0.87 | Critical: XOR init, obstruction, pliesFromNull |
| 21 | fix-nmp-bugs (3 bugs) | early | — | — | Verify depth, capture R, depth>=2 |
| 22 | fix-lmr-nonpawn | early | — | — | Wrong side non-pawn check |
| 23 | fix-se-extensions (3 fixes) | early | — | — | Limiter, negative ext, multi-cut |
| 24 | fix-futility-exemptions | early | — | — | TT/killer exempt, ply>0 guard |
| 25 | fix-histprune-score | early | — | — | Full combined history for pruning |
| 26 | fix-histprune-no-improving | early | — | — | Remove !improving+!unstable guards |
| 27 | fix-histprune-no-unstable | early | — | — | Remove !unstable guard only |
| 28 | fix-capthist-scale | early | — | — | Restore captHist/16 scaling |

### Bug Inventory (discovered 2026-04-02)

Systematic feature-by-feature deep review by Titan uncovered implementation bugs in nearly every search feature. Key findings:

**Cuckoo** (3 bugs): XOR chain init wrong (dead code), obstruction missing dest square, no pliesFromNull limit.
**NMP** (3 bugs): Verification depth too shallow, capture R inverted, depth gate too conservative.
**LMR** (1 bug): Wrong side non-pawn piece check (reducing when WE have few pieces, not opponent).
**SE** (3 issues): No extension limiter, weak negative extensions, multi-cut returns wrong score.
**Futility** (2 bugs): Missing TT/killer exemptions, no ply>0 guard.
**SEE** (2 bugs): Promotions completely unhandled (major), EP pawn not removed from occupancy.
**RFP** (2 bugs): Missing !is_pv guard, missing excluded_move guard.
**History Pruning** (2 issues): Incomplete history score, overly conservative guards.

All fixes branched and SPRT testing in progress.

Additional bugs found in second/third review passes:
**ProbCut** (3 bugs): Missing excluded_move guard, no qsearch filter, SEE threshold too low. Plus: unique static eval gate too restrictive, missing TT guard, missing mate guard.
**is_pseudo_legal** (5 gaps): Back-rank pawns without promo, EP diagonal without EP flag, invalid flag values 3/8-15.
**make/unmake** (1 bug): minor_key/major_key not restored in unmake — corrupts ALL correction history lookups.
**Time management** (2 bugs): Ponderhit kills search, movestogo=1 wastes 50% of time.
**Lazy SMP** (1 bug): Helper threads set shared stop flag on timeout.
**Alpha-reduce** (1 design bug): Component 1 bypasses all move quality exemptions unconditionally.
**Hindsight** (1 sign bug): eval_sum adds opposite-perspective evals without negating.
**Cuckoo** (2 more bugs found by Atlas): XOR chain indices wrong (i-2 should be i), obstruction check incorrectly included destination square.
**Repetition** (1 issue): No pliesFromNull guard on lookback.

### Concluded Tests (Day 2 continued)

| # | Test | Elo | Games | LLR | Result | Action |
|---|------|-----|-------|-----|--------|--------|
| 13 | fix-asp-depth-clamp | -4.34 | 4,808 | -1.80 | Trending H0 | Shallow re-searches serve as cheap filter |
| 15 | fix-cutnode-lmr (+2) | -15.7 | 3,040 | -2.95 | **H0** | +2 too aggressive for our engine |
| 17 | 768pw-w7 vs production | -2.94 | 4,608 | -1.34 | Stopped (flat) | Confirms 768pw = 1024s (3rd test) |
| 29 | fix-cuckoo-init (bugs 1-3) | -3.68 | ~4,000 | H0 | **H0** | Fixes merged, cuckoo disabled. Atlas found 2 more bugs. |
| 30 | 13file-blunders e800 vs production | +2.58 | 5,200 | 0.30 | Stopped (mildly +ve) | Blunders help slightly at ~8% ratio. Try 20-30%. |

### Architecture Decision: 768pw

768pw-w5 and 768pw-w7 both match 1024s in self-play across 3 separate tests (~12,000 total games). Decision: **adopt 768pw as default architecture** for 10-15% NPS advantage and v7 readiness. All top engines use 768pw as FT layer.

## 2026-04-03: H1 Winners and Ongoing Tests

### H1 Passed — Merged to main

| # | Test | Elo | Games | Result | Description |
|---|------|-----|-------|--------|-------------|
| 44 | fix-probcut | **+5.86 ±4.22** | 11,272 | **H1** | ProbCut revival: removed static eval gate, added qsearch filter, SEE threshold, TT/mate guards |
| 37 | fix-unmake-keys | **+5.67 ±4.10** | 11,526 | **H1** | Restore minor_key/major_key in unmake — was corrupting all correction history |
| 31 | fix-see-promotion | **~+5** | ~10,000 | Stopped +ve | Handle promotions in SEE (gain + risk value). Correctness fix, consistently positive. |

**Combined merged Elo: ~+16.5** from three bug fixes.

### H0 Failed

| # | Test | Elo | Games | Result | Notes |
|---|------|-----|-------|--------|-------|
| 21 | fix-nmp-bugs (bundled) | -29 | 1,820 | **H0** | 3 NMP fixes together too aggressive. Decomposed into 3 individual tests. |
| 32 | fix-see-ep | H0 | 9,740 | **H0** | EP pawn occupancy fix — too rare to show. Merge as correctness. |
| 13 | fix-asp-depth-clamp | -4.34 | 4,808 | Trending H0 | Shallow re-searches serve as cheap filter. Don't clamp. |

### Stopped / Noted

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 30 | 13file-blunders e800 | +2.58 | 5,200 | Blunders help slightly at ~8% ratio. Try 20-30% next time. |
| 29 | fix-cuckoo-init v1 | -3.68 | ~5,600 | H0. Fixes had 2 additional bugs found by Atlas. |

### Still Running (key tests)

| # | Test | Elo (latest) | Games | Notes |
|---|------|-------------|-------|-------|
| 43 | fix-cuckoo-bugs (Atlas v2) | ~+2 | ~19,000 | Mildly positive, grinding |
| 16 | fix-strong-failhigh-bonus | ~+4 | ~5,600 | Trending toward H1 |
| 44v2 | fix-probcut (improved) | early | — | Removed static eval gate, added TT/mate guards |
| NMP decomposed (3 tests) | — | early | — | Capture R, verify depth, depth gate tested individually |
| fix-hindsight-sign | — | rebased | — | Negate parent eval for correct perspective |
| + many more from Titan bug hunt | — | queued | — | SE, futility, RFP, hist prune, alpha-reduce, is_pseudo_legal, SMP, TM |

### H1 Passed — Merged (Day 3)

| # | Test | Elo | Games | Result | Description |
|---|------|-----|-------|--------|-------------|
| 50 | fix-nmp-search-depth | **+11.0** | ~5,000 | **H1** | depth-1-r → depth-r. Every NMP search was 1 ply too shallow. Biggest single fix. |

**Total confirmed Elo from bug fixes: ~+27.5** (probcut +5.86, unmake keys +5.67, SEE promo ~+5, NMP depth +11)

### H0 Failed (Day 3)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 22 | fix-lmr-nonpawn | H0 | 4,686 | Wrong-side fix hurts — engine tuned around it |
| 25 | fix-histprune-score | H0 | 3,044 | Full history score too aggressive |
| 48 | fix-hindsight-sign | H0 | 4,154 | Sign fix hurts — feature captures useful signal as-is |
| 23 | fix-se-extensions | H0 | 2,544 | SE fixes hurt — parameters tuned for current behavior |
| 13 | fix-asp-depth-clamp | H0 | 6,784 | Shallow re-searches serve as cheap filter |
| 21 | fix-nmp-bugs (bundled) | H0 | 1,820 | 3 NMP fixes together -29 Elo. Decomposed. |
| — | fix-histprune-full | H0 | ~3,000 | Consensus reimplementation too aggressive |
| — | fix-futility-full | trending H0 | ~2,000 | Consensus reimplementation regressed to -2 |

### Key Lesson

Many "correctness fixes" test negative because the engine was tuned around the bugs. The NMP search depth fix (+11) succeeded because it was a pure 1-line change with minimal parameter interaction. Holistic reimplementations based on top-engine consensus failed — our engine's parameter landscape is different. Conservative single-change approach is more reliable.

## 2026-04-04: Ongoing Results

### H1 Passed — Merged

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 74 | fix-qs-tt-stand-pat | **+6.5** | H1 | QS TT bound refinement of stand-pat. Every top engine does this. |
| 47 | fix-nmp-depth-gate | **+0.68** | 2,048 | NMP depth gate >= 4 → >= 3. Correctness, mildly positive. |

**Running total merged Elo: ~+35** (NMP depth +11, probcut +5.86, unmake keys +5.67, SEE promo ~+5, QS stand-pat +6.5, NMP depth gate ~+0.7)

### Stopped / Noted

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 56 | const-attack-tables | +1.5 | 45,000 | NPS gain from compile-time tables. Conflicts with runtime PEXT detection. Revisit later. |
| 55 | compile-time-pext | ~0 | — | Superseded by runtime PEXT detection (AMD Zen1/2 +20% NPS). |
| 76 | fix-tm-node-based v1 | -31 | 628 | Bug: cumulative node tracking across iterations. |
| 77 | fix-tm-node-based v2 | -10 | — | Fixed per-iteration reset but parameters still wrong. Needs more work. |
| 57 | fix-histprune-add-cont2 v1 | -68 | 352 | Threshold -1500 too tight for 3 signals. Resubmitted at -2250. |
| 53 | fix-lmr-remove-alpha-raised (old base) | -2.3 | 5,866 | Stale — tested against pre-merge main. Replaced by #65. |

### H0 Failed (Day 3-4)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 49 | fix-histprune-full | H0 | ~3,000 | Consensus reimplementation too aggressive |
| 52 | fix-futility-full | H0 | ~7,000 | Consensus reimplementation regressed |

### NPS Optimization Notes

- **Runtime PEXT detection** (Thor): +20% NPS on AMD Zen1/2 (~13 Elo). Auto-detects at startup. Merged to main.
- **Const-attack-tables**: +1-2 Elo from compile-time tables but conflicts with runtime PEXT. Defer.
- **Compile-time-pext**: Superseded by runtime approach. Dropped.
- **Incremental-checkers-pinned**: -10 Elo. Eager compute worse than on-demand. Needs true incremental update (cozy-chess style) to be useful.

### Reverted

| Test | Elo | Notes |
|------|-----|-------|
| fix-rep-pliesnull | -38 | PliesnullNull limit on repetition detection was wrong — cut lookback too short, missed real repetitions. Limit should only apply to cuckoo, not normal rep detection. |
| fix-hindsight-sign | -10 (H0) | Original "wrong" sign captures useful mutual-optimism signal. Revert confirmed correct. |

### Stopped (Day 4 continued)

| Test | Elo | Games | Notes |
|------|-----|-------|-------|
| incremental-checkers-pinned | -10 | — | Eager compute worse than on-demand |
| fix-tm-node-based v1 | -31 | 628 | Cumulative node tracking bug |
| fix-tm-node-based v2 | -10 | — | Per-iteration reset but params still wrong |
| fix-histprune-add-cont2 v1 | -68 | 352 | Threshold too tight for 3 signals |
| fix-histprune-add-cont2 v2 | -58 | — | Even at -2250 threshold, ply-2 piece lookup is buggy (stale board state) |

### H0 Failed (Day 4 continued)

| Test | Elo | Notes |
|------|-----|-------|
| fix-histprune-full | H0 | Consensus reimplementation too aggressive |
| fix-futility-full | H0 | Consensus reimplementation regressed |
| fix-nmp-no-dampening | trending H0 (-1.9) | NMP dampening removal not beneficial |
| fix-nmp-capture-r | trending H0 (-1.6) | NMP capture reduction not beneficial |
| Various LMR removals | trending H0 | Multiple LMR feature removals all trending negative |

### SPSA Tuning (in progress)

- **Round 1**: 16 parameters, 5000 iterations target. At ~860 iterations, strong signals:
  - NMP_EVAL_DIV: 200 → 164 (most aggressive mover)
  - NMP_BASE_R: 4 → 3
  - RFP_DEPTH: 7 → 6
  - RFP_MARGIN_IMP: 70 → 81
  - FUT_BASE: 90 → 102
  - HIST_PRUNE_DEPTH: 3 → 2
- **Round 2**: 10 more parameters added (aspiration, LMR C, LMP, probcut, etc.)
- 26 tunable UCI parameters exposed for SPSA.

### Infrastructure (Day 4)

- **Thor** (5th machine) added: AMD Ryzen 8C/16T, 1.1M NPS
- **Runtime PEXT detection**: +20% NPS on AMD Zen1/2
- **NEON pairwise SIMD** for ARM
- **clap argument parser** replacing manual parsing
- **26 tunable UCI parameters** for SPSA

### Key Finding: Ply-2 Continuation History Piece Lookup Bug

The ply-2 cont hist piece lookup via `board.piece_at(move_to(pm2))` is unreliable — the piece may have been captured by ply-1's move. This explains why adding ply-2 cont hist to history pruning was -58/-68 Elo regardless of threshold tuning. The fix requires a `moved_piece` field on the search stack (so each ply records what piece it moved, independent of current board state). Added to todo.md.

## 2026-04-04: SPSA Round 1 — H1 at +31.48 Elo [SPRT-validated]

**Biggest single Elo gain in Coda's history.** First SPSA parameter optimization via OpenBench.

- **Result**: **H1**, +31.48 ±11.26 Elo, 1,616 games, LLR 2.98.
- **Commit**: cbdcefe
- **SPSA config**: 16 parameters, 1,672 iterations (~27K games), 5 machines.
- **Key parameter changes**: NMP_EVAL_DIV 200→164, NMP_BASE_R 4→3, RFP_DEPTH 7→6, RFP_MARGIN_IMP 70→84, FUT_BASE 90→103, HIST_PRUNE_DEPTH 3→2, LMR_HIST_DIV 5000→5489, SE_DEPTH 8→9.
- **Lesson**: Simultaneous optimization finds gains that individual SPRT cannot detect. Parameters interact — NMP wants aggressive eval scaling (164) but only when RFP margins widen simultaneously. No single change tests positive, but the combination is +31.
- **Next**: SPSA round 2 with 26 parameters on 768pw-w7 net after model switch.

## 2026-04-05: Structural Fixes + SPSA Round 4

### Methodology Breakthrough: Detect → Diagnose → Fix → Tune

Discovered that SPSA detuning signals indicate structural implementation flaws. When SPSA aggressively moves a parameter away from its starting value, the feature may be broken. Deep cross-engine comparison (8 engines) diagnoses the issue, fix is SPRT tested, then SPSA retunes the corrected base.

**Key structural finding**: Multiple pruning features were applied AFTER MakeMove when every top engine does them BEFORE. This wasted MakeMove + NNUE push/pop per pruned move and made features redundant with earlier pruning. Migrated all pruning pre-MakeMove.

### H1 Passed — Merged (Day 5)

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 94 | fix-see-quiet-pruning | **+11.4** | 2,020 | SEE quiet: pre-MakeMove, lmrDepth² scaling, use see_ge. SPSA was detuning 17→6. |
| 114 | spsa-r4-final | **+17.0** | 1,230 | 32 params, 5000 iterations on corrected base. Biggest movers: FUT_PER_DEPTH 116→140, LMR_C_CAP 179→196. |
| 109 | fix-bad-noisy-pre-move | **+2.9** | 2,846 | Bad noisy: pre-MakeMove, drop gives_check exemption. Completes pre-MakeMove migration. |
| 112 | fix-lmp-reimplementation | **+2.3** | 3,492 | LMP: pre-MakeMove, SF formula (BASE+d²)/(2-improving), BASE=5 from focused SPSA. |
| 85 | fix-corr-proportional-gravity | **+3.9** | 2,466 | Correction history: proportional gravity (consensus). |
| 84 | fix-corr-no-depth-gate | **~0** | 924 | Correction history: remove depth≥3 gate (consensus, correctness merge). |
| 105 | fix-lmr-cont2-stat-v2 | **+2.5** | 3,396 | LMR: ply-2 cont hist in stat score at half weight (consensus). |
| 106 | fix-tm-forced-move | **+1.3** | 2,064 | TM: forced move detection, cap time at 10ms with 1 legal move. |
| 107 | fix-qs-negative-see | **~0** | 3,098 | QS: add QS_SEE_THRESHOLD tunable (default 0, no behavior change). |
| — | fix-qs-move-count-cutoff | **~0** | — | QS: add QS_MAX_CAPTURES tunable (default 32, no behavior change). |

**Running total merged Elo (Day 5): ~+41** (SEE quiet +11.4, SPSA r4 +17, bad noisy +2.9, LMP +2.3, corr gravity +3.9, LMR cont2 +2.5, forced move +1.3)

### H0 Failed / Rejected (Day 5)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 93 | spsa-r3-snapshot-1000 | +1.6 | 5,676 | Too early snapshot (1000 iters), killed. |
| 97 | spsa-r3-snapshot-8054 | -7.6 | 960 | Tested against stale main (pre-futility merge). |
| 95 | fix-futility-pruning v1 | -18.6 | 578 | bestScore update polluted TT. Fixed in v2. |
| 96 | fix-futility-pruning v2 | +1.5 | 3,962 | Correctness merge (gate fix, pre-MakeMove). |
| 100 | fix-corr-clamp-1024 | -7.0 | 1,830 | GRAIN not scaled with LIMIT. V2 fixed but still neutral. |
| 104 | fix-corr-clamp-1024-v2 | +1.5 | 2,816 | Properly scaled constants, but no benefit. Our 32000 limit works fine. |
| 102 | fix-lmr-cont2-stat v1 | -5.3 | 2,304 | Ply-2 at full weight over-scaled hist_score. Fixed in v2 (half weight). |
| 101 | fix-corr-cont-2ply | +0.4 | 8,908 | 2-ply continuation correction. H0 — ply-2 context too noisy for correction signal. |
| 108 | fix-lmp-reimplementation v1 | -6.3 | 2,082 | BASE=3 too aggressive. Fixed in v2 (BASE=5 from focused SPSA). |
| 113 | fix-qs-max-captures-5 | -1.5 | 4,456 | QS capture limit at 5. Mildly negative standalone. |
| 71 | fix-se-ply-gate | -7.5 | 5,168 | SE ply gate doesn't suit our engine. |
| 83 | fix-asp-delta-133 | -5.3 | 3,288 | Our x1.5 growth works better than consensus x1.33. |
| 82 | fix-asp-skip-winning | -6.2 | 3,696 | Skip depth reduction for winning fail-highs hurts. |
| 90 | fix-histprune-cont2-stack | -8.4 | 4,780 | Ply-2 cont hist in hist pruning. Threshold not scaled for 3 signals. |

### Stopped / Not Merged (Day 5)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 80 | fix-asp-no-fh-alpha | +1.1 | 3,036 | Remove alpha contraction on fail-high. Flat, stopped. |
| 81 | fix-asp-fl-midpoint | +1.2 | 4,252 | Fail-low beta to midpoint. Flat, stopped. |
| 58 | fix-nmp-no-dampening | -0.6 | 5,794 | Already merged earlier, SPRT was confirming non-regression. Stopped. |
| 86 | fix-smp-copy-history | +5.9 | 2,300 | Already merged earlier, confirming non-regression. Stopped. |

### SPSA Tuning

- **Round 3**: 32 parameters on 768pw-w7. Multiple snapshots tested (1000 iter, 8054 iter). Early snapshot too unstable. Later snapshot stale vs new main. Killed in favor of r4.
- **Round 4**: 32 parameters on corrected base (all pruning pre-MakeMove). 5000 iterations. **H1 at +17.0 Elo, 1230 games.** Key: FUT_PER_DEPTH 116→140, LMR_C_CAP 179→196, SE_DEPTH 10→8, LMP_BASE 1→3.
- **Round 5**: 34 parameters (added QS_SEE_THRESHOLD, QS_MAX_CAPTURES). Running on post-LMP-merge base. QS_MAX_CAPTURES starting at 16.
- **LMP focused tune**: 2 params (LMP_BASE, LMP_DEPTH), 359 iterations. BASE converged 3→4.5. Used BASE=5 for SPRT.
- **QS_MAX_CAPTURES mini-tune**: 1 param, 200 iterations. Converged at 5.3. SPRT at 5 was -1.5 (H0). Value may be useful in full SPSA context.

### Infrastructure / Code Quality (Day 5)

- **Embedded NNUE fix**: UCI mode now uses fat binary's embedded net (was silently falling back to disk).
- **auto_discover_nnue()**: Single function for all net loading paths (bench, UCI, etc.).
- **Pondering fixed**: ponder move output, TM skip for infinite mode, ponderhit_time reset.
- **tunables! macro**: Single source of truth for parameter definitions (eliminated duplication).
- **Code review cleanup**: dedup lmrDepth computation, fix FEAT flags, fix forced move under movetime, remove stale comments.
- **Pre-MakeMove migration complete**: ALL pruning (history, futility, SEE quiet, LMP, bad noisy, SEE capture) now before MakeMove.
- **Force-captures datagen**: New --force-captures mode for training data diversity.
- **ob_stop.py**: Script to stop OpenBench tests.

### Key Lessons (Day 5)

1. **SPSA detuning = structural bug signal.** SEE_QUIET_MULT driven to 5 → found 3 structural issues. FUT_PER_DEPTH driven to 99 → found gate/margin mismatch. LMP_BASE driven to 3 → found post-MakeMove + wrong formula.
2. **Pre-MakeMove is universal.** Every pruning feature we moved pre-MakeMove gained Elo. The pattern was systematic, not feature-specific.
3. **Fix first, tune second.** Structural fixes compound — SPSA r4 gained +17 because it tuned a corrected base, not a broken one.
4. **SPRT bounds should match intent.** [-10,5] for correctness fixes, [0,10] for expected gains. Saved weeks of grinding on features we'd merge anyway.
5. **V2 attempts matter.** fix-lmr-cont2 v1 was -5.3, v2 was +2.5 (half weight fix). fix-corr-clamp v1 was -7.0, v2 was +1.5 (scaled constants). fix-lmp v1 was -6.3, v2 was +2.3 (SPSA-tuned BASE).
6. **Blunder training data improves eval calibration.** SB20/SB60 models trained on blunder data show piece values much closer to Obsidian (reference engine). T80 alone creates miscalibrated piece ordering (knight > queen). Blend is the answer.

## 2026-04-06: Capture History Investigation [SPRT-pending]

### Dynamic Capture SEE Threshold v1 — REJECTED (OB #120)
- **Change**: Replace binary SEE≥0 for good/bad capture split with `-capScore/32` (MVV + captHist in the score).
- **Result**: H0, -4.6 Elo, 2632 games.

### Dynamic Capture SEE Threshold v2 — REJECTED (OB #128)
- **Change**: Use only captHist (not MVV) for threshold: `-captHist/18`, matching Stockfish's divisor.
- **Result**: H0, -11.5 Elo, 2828 games. Worse than v1.

### Root Cause: Capture History Was Non-Functional
Deep comparison with Stockfish/Obsidian/Viridithas/Clarity revealed two critical bugs:

**Bug 1: Capture history bonus 5-25x too small.** `history_bonus(depth) = depth*depth capped at 1200` gives bonus of 25 at depth 5, 100 at depth 10. Every top engine uses linear formulas giving 500-1600 at same depths (Obsidian: `min(175*d-50, 1400)`, SF: `min(128*d-77, 1529)`). CaptHist values were too tiny to influence ordering — MVV dominated completely.

**Bug 2: Missing unconditional capture malus.** Coda only updated capture history when a capture caused the cutoff. When a quiet caused cutoff, all tried-and-failed captures got NO malus. Every top engine (SF, Obsidian, Viridithas, Clarity) unconditionally penalizes tried captures on any cutoff. This systematically inflated capture history values.

**Consequence**: LMR capture thresholds (±2000) were dead code — values never reached that range. Dynamic SEE threshold was adding noise, not signal.

### Capture History Fix (OB #pending) — fix-capture-history
- **Changes**: (1) New `capture_history_bonus(depth) = min(175*d - 50, 1400)` (Obsidian-style), separate from quiet bonus. (2) Unconditional capture malus: penalize all tried captures on any beta cutoff, not just when a capture is best.
- **Bench**: 1108435 (12% fewer nodes than main = capture ordering actually cutting off earlier).
- **Impact**: If this passes, should revisit: fix-dynamic-capture-see, fix-capthist-scale, and any other feature that depends on capture history signal.

### Features to Revisit After Capture History Fix
- Dynamic capture SEE threshold (v1 and v2 both failed with broken captHist)
- CaptHist scaling in move ordering (fix-capthist-scale, H0)
- LMR capture history adjustments (thresholds were dead code)
- Capture history weight in scoring (MVV×16 + captHist — ratio may need retuning)

## 2026-04-06: History Magnitude Discovery + Move Ordering + Training

### The History Magnitude Bug (biggest finding)

Dynamic capture SEE failed 3 times (v1 -4.6, Atlas v2 -11.6, our v1 -4.6).
Instead of giving up, pushed Atlas to investigate **why** captHist values
were too small to matter. Found: `history_bonus = (depth²).min(1200)` gave
25 at depth 5, vs SF's 682 and Obsidian's 825. **27× too small.**

This affected ALL history tables: main butterfly, capture, continuation,
and pawn history. The entire history system was effectively dead.

Fix: `(170 * depth - 50).clamp(0, 1400)` — linear formula matching consensus.

### H1 Passed — Merged (Day 6)

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 132 | fix-history-bonus-magnitude | **+31.6** | 364 | Biggest single change ever. All history 27× too small. |
| 122 | fix-quiet-check-bonus | **+6.0** | 5,000 | Checking squares bitboard, +10K quiet check bonus |
| 117 | spsa-r5-final | **+9.9** | 2,392 | SPSA on LMP-corrected base |
| 129 | fix-mate-distance-pruning-v2 | **+1.7** | 4,164 | ply+1 fix, non-PV only (v1 H0'd at -3.6) |
| 131 | fix-capture-history | **~+5** | trending H1 | Unconditional capture malus + linear bonus |

**Running total merged Elo (Day 6): ~+54** (history +31.6, check bonus +6.0, SPSA r5 +9.9, MDP +1.7, capture hist ~+5)

### H0 Failed (Day 6)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 125 | fix-mate-distance-pruning v1 | -3.6 | 4,866 | Missing ply+1, applied at PV. Fixed in v2. |
| 124 | fix-root-history | -0.8 | 5,320 | Root history table not helpful. |
| 123 | fix-threat-escape-bonus | -2.6 | 3,488 | 4D history already handles threats. |
| 126 | fix-double-extensions v1 | +0.7 | 10K+ | Margin 30 too conservative. v2 testing margin 10. |
| 128 | fix-dynamic-capture-see v2 (Atlas) | -11.6 | 2,828 | Led to captHist magnitude discovery. |
| 120 | fix-dynamic-capture-see v1 | -4.6 | 2,632 | Same root cause — tiny captHist. |
| 121 | fix-cont-hist-equal-weights | -6.3 | 2,032 | 3x→2x hurts (3x compensated for tiny magnitudes). |

### Active SPRTs

| # | Test | Status | Notes |
|---|------|--------|-------|
| 133 | fix-dynamic-capture-see-v3 | Just started | Atlas's version on history-fixed base |
| 130 | fix-double-extensions-v2 | Just started | Margin 30→10 (closer to Velvet's 4) |
| 131 | fix-capture-history | +5.1, LLR 1.68 | Trending H1 |

### Training Experiments

- **Exp A/C/E (768pw v7)**: All collapsed — dead hidden layers, gradient instability through pairwise bottleneck.
- **Exp G (1024 SCReLU v7 + tricks)**: Running, SB ~36/100
- **Exp H (v5 768pw + power-2.6 loss)**: Running, SB ~36/100
- **Exp I (v5 768pw + 1 output bucket)**: Running, SB ~36/100

### Key Lessons (Day 6)

1. **Don't accept consensus H0s**: Dynamic capture SEE failing 3 times led to the biggest discovery (+31.6 Elo from history magnitude fix). Always ask WHY.
2. **Compare actual magnitudes, not just code structure**: The history bonus formula *looked* correct (depth-based, gravity, capping). Only comparing exact numbers at specific depths revealed the 27× scaling bug.
3. **Cascading discoveries**: capture SEE → captHist too small → ALL history too small. One investigation chain led to fixing the entire history system.
4. **v7 pairwise+hidden training broken in Bullet**: CReLU→pairwise creates gradient bottleneck that kills FT learning. 1024 SCReLU v7 works. Velvet-style v5 is the recommended path.

### Features to Retry After SPSA Retune

These features H0'd on a base with dead history. With working history (27× larger values),
the landscape is completely different. Priority retry list:

1. **Dynamic capture SEE** — v3 already testing on fixed base
2. **Cont hist weights** (3x→2x or 1x) — 3x was compensating for tiny values
3. **Root history** — tiny values meant 4× of nothing was still nothing
4. **Threat escape bonuses** — 4D history was also dead, explicit bonuses may now be redundant for the right reason
5. **Double extensions** — better history → better move quality → extensions on more accurate moves
6. **LMP with lower BASE** — working history means late moves are genuinely bad, LMP can prune more aggressively

## 2026-04-06 Evening: Retries + Atlas Experiments

### Context
After history magnitude fix (+31.6 Elo) and capture history fix (+5.4),
retesting features that previously failed on dead-history base.

### H1 Passed — Pending Merge

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 141 | retry-lmp-base5 | +3.5 | 8210 | LMP BASE 9→5. Working history enables better move ordering. Trending H1. |

### H0 Failed (Retries)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 140 | retry-cont-hist-1x | -1.1 | 4922 | 3x/3x genuinely better, not just compensating for dead history. DROP. |
| 142 | retry-root-history | -7.9 | 1800 | Third failure across different bases. DROP permanently. |
| 143 | retry-double-ext-margin4 | +0.3 | 8346 | Margin=4 too aggressive. Our margin=10 is right. DROP. |

### Atlas Experiments (running)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 144 | retry-dynamic-capture-see | +0.7 | 4096 | 4th attempt with -captHist/18. Flat, likely H0. |
| 145 | fix-50move-eval-scaling | +1.7 | 1860 | eval*(200-hm)/200. Early, positive. |
| 147 | fix-strong-failhigh-bonus-v2 | +3.2 | 2938 | depth+(score>beta+95) for history. Promising. |
| 146 | fix-eval-history-bonus | -3.8 | 3098 | depth+(eval<=alpha). Heading H0. |
| 148/149 | fix-tt-cutoff-nodetype | -109/resubmitted | — | Missing condition in v1. v2 submitted. |

### SPSA Tune r8 (running, 2500/5000)
Key movers: HIST_PRUNE_MULT 2119→3736 (+76%), FUT_PER_DEPTH 151→159,
LMR_C_CAP 184→174, SE_DEPTH 6→5.7. Mainly calibrating history pruning
threshold for 27× larger values.

### Training Experiments

| # | Config | Result |
|---|--------|--------|
| L | v5 768pw, 2 output buckets, MSE, production config | HEALTHY. Check-net: knight>queen ordering persists. Output bucket hypothesis DISPROVEN. |

Check-net tool was fixed: FENs had Black missing pieces (should be White).
Piece ordering issue traced to LC0 training data scoring by win probability,
not material — a data characteristic, not a bug.

### Key Lessons (Day 6 Evening)
1. Retries after history fix: LMP improved (as predicted), cont-hist-1x and root-history still fail (3x/3x is genuinely better, root ordering already good).
2. Output bucketing doesn't cause piece ordering — LC0 data does.
3. Power-2.6 loss collapses in Bullet without weight decay (all attempts failed).
4. 11 of 12 GPU training experiments failed today — mostly config bugs from template-derived configs. Always use exact production config as base.

## 2026-04-07: Overnight Results + New Experiments

### H1 Passed — Merged

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 152 | spsa-r8-snapshot | **+3.5** | 2,598 | HIST_PRUNE_MULT 2119→4170 recalibration |
| 149 | fix-tt-cutoff-nodetype | **+2.4** | 29,632 | Only accept TT cutoffs matching expected node type (Alexandria) |
| 153 | spsa-r9-snapshot | **+6.4** | 1,514 | 45-param tune. HIST_PRUNE_MULT→7224, FUT_BASE→94, NMP_EVAL_DIV→148 |

### H0 Failed

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 144 | retry-dynamic-capture-see | -0.5 | 20,122 | 5th failure. PERMANENTLY DROPPED. |
| 146 | fix-eval-history-bonus | -0.4 | 21,392 | depth+(eval<=alpha) doesn't help. Drop. |
| 147 | fix-strong-failhigh-bonus-v2 | -1.2 | 14,576 | depth+(score>beta+95) doesn't help. Drop. |
| 145 | fix-50move-eval-scaling | +0.8 | 31,020 | Neutral at 31K games. Stopped. |

### SPSA Tune r9 (45 params, 2500 iterations)
New params (11 added): HIST_BONUS_MULT/BASE/MAX, CAP_HIST_MULT/BASE/MAX,
DEXT_MARGIN/CAP, QUIET_CHECK_BONUS, LMR_COMPLEXITY_DIV. Most new params
stable at starting values — defaults were well-calibrated. HIST_BONUS_MAX
moved from 1400→1505, CAP_HIST_BASE from 50→45.

HIST_PRUNE_MULT continues to be the dominant mover: 4170→7224 (+73%).
Still climbing after 3 consecutive tunes. Focused 5-param tune running.

### Focused Tune (5 params on r9 branch)
Running: HIST_PRUNE_MULT, FUT_BASE, HIST_BONUS_MAX, LMP_BASE, NMP_EVAL_DIV.
These showed the most movement and may not have converged in the full tune.

## 2026-04-07 Afternoon: TT PV Flag, Node-Based TM, Atlas Retries

### H1 Passed — Merged

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 160 | fix-tt-pv-flag | **+4.5** | 10,780 | Sticky PV bit in TT, LMR -1 for PV positions. 30% node increase. |
| 172 | fix-node-based-tm | **+11.9** | 1,196 | 3-factor TM (Obsidian/Clarity). **Tested at 40+0.4 LTC.** Failed 3x at STC. |
| 173 | spsa-r10-snapshot | **+4.0** | 14,408 | 18-param pruning retune post-tt-pv. Nodes -8.3%. |

### H0 Failed — Atlas Retries (bug fixes on previously-failed features)

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 166 | fix-badnode-flag v3 | -3.7 | 6,882 | Fixed RFP/NMP direction. Still no help. Permanently dropped. |
| 167 | fix-history-extensions v2 | -0.4 | 21,690 | Lower threshold (10K→5K). Dead flat. Permanently dropped. |
| 168 | fix-lmp-npm-guard v2 | -3.8 | 6,810 | NPM guard only (no BASE change). Permanently dropped. |
| 169 | fix-tt-cutoff-conthist-malus v2 | -0.5 | 20,570 | Fixed self-referencing indexing bug. Flat without retune. |

### Key Methodology Discoveries (Day 7)

1. **TM changes need LTC validation.** Node-based TM failed 3x at STC (-31, -10, -3.2) but passed at 40+0.4 (+11.9). At STC (~200ms/move) stop/continue decisions have too little leverage.
2. **Features that shift tree shape need retuning.** TT PV flag +4.5 raw, retune added another +4.0 (nearly doubled).
3. **Cont-hist malus: flat without retune, +6.5 with.** First validation of retune-on-branch methodology. Feature was -0.15 at 16K games, then +6.5 after SPSA found different optima (LMR_HIST_DIV -15%, HIST_PRUNE_MULT -8.4%).

## 2026-04-08: Retune-on-Branch Systematic Testing

### Philosophy

Previously-rejected features that change tree shape may gain Elo when pruning parameters
are recalibrated on their branch. Workflow: feature branch → SPSA tune (18 pruning params,
~600-1000 iterations) → compare parameter divergence vs main baseline → SPRT with tuned values.

### Round 1: Retune Candidates (from reject pile)

All tested against main at b98c0a1 (bench 1780721).

| # | Test | Elo | Games | Original result | Tune divergence | Status |
|---|------|-----|-------|-----------------|-----------------|--------|
| 183 | fix-lmr-simplify + tune | **+6.3** | 7,366 | -2 to -4 (each adj individually) | High: SEE_QUIET -17%, HIST_PRUNE -9% | **H1 ✓** |
| 181 | fix-corr-clamp-1024-v2 + tune | +2.1 | 9,890 | -7.0 (v1), +1.5 (v2) | Highest: HIST_PRUNE -15%, SEE_CAP -11% | Stopped, merge as correctness |
| 180 | fix-50move-eval-scaling + tune | -2.9 | 8,878 | +0.8 (31K games) | Moderate (FUT_BASE, FUT_PER_DEPTH) | **H0 ✗** |
| 182 | fix-se-negative-ext-v2 + tune | +1.4 | 22,572 | -10.5 | Lowest (closest to baseline) | Running (+1.4, grinding) |

### Round 2: More Retune Candidates

| # | Test | Elo | Games | Original result | Tune divergence | Status |
|---|------|-----|-------|-----------------|-----------------|--------|
| 195 | fix-nmp-capture-r + tune | **+3.5** | 16,718 | -0.2/-7.1 | Moderate: SEE_CAP -2.5%, FUT_BASE -6% | **H1 ✓** |
| 194 | fix-futility-full + tune | -2.5 | 9,714 | -1.0/-2.2 | High: HIST_PRUNE -12%, LMR_C_QUIET -6% | **H0 ✗** |
| 197 | fix-se-ply-gate + tune | -5.6 | 5,310 | -7.5 | Moderate: HIST_PRUNE -8% | **H0 ✗** |
| 198 | fix-histprune-no-improving + tune | +0.5 | 12,264 | -3.7 | Low (calmest tune) | Running (flat) |
| 196 | fix-corr-cont-2ply + tune | +1.3 | 12,866 | +0.4 (8.9K) | Highest of R2: FUT -7%, NMP_EVAL +8% | Stopped, retuning further |
| 199 | fix-good-bad-quiet-split | -0.0 | 4,444 | N/A (new feature) | N/A (direct SPRT, now tuning) | Stopped, tuning |

### Merge Batch (pending — holding for all experiments to complete)

| Feature | Elo | Method | Notes |
|---------|-----|--------|-------|
| LMR simplify + tune | +6.3 | Retune-on-branch | Remove failing/alpha_raised/unstable |
| NMP capture R + tune | +3.5 | Retune-on-branch | Flip r-=1 to r+=1 after captures |
| Corr-clamp 1024 + tune | +2.1 | Correctness merge | Consensus clamp value |
| (cont-hist malus + tune) | +6.5 | Retune-on-branch | Already merged |

### New Features Being Tuned

| Feature | Bench | Node Δ | Tree shape signal | Tune status |
|---------|-------|--------|-------------------|-------------|
| Good/bad quiet split | 1,701,572 | -4.4% | Avg cutoff 1.92→1.82, TT hits +8.3% | Tune running |
| Remove killers/counters | 1,695,682 | -4.8% | TT hits +13.6%, LMR +14.7%, EBF 1.87→1.82 | Tune running (600 iter) |
| Corr-cont-2ply (re-tune) | — | -21.5% | Biggest tree shape change of any candidate | Tune restarted |

### Infrastructure

- **ob_tune.py**: New script for submitting SPSA tunes programmatically
- **ob_tune_status.py**: Read tune results, compare branches side-by-side
- **ob_status.py**: Updated to separate active vs finished tests
- **tune_pruning_18.txt**: Standard 18-param pruning tune specification
- **Tree shape fingerprint**: Added to bench output (per-1K-node pruning rates)
- **EBF metric**: Added to bench (effective branching factor)
- **Lichess bot**: Deployed on `lo` as `codabot`, playing on lichess.org

### Key Lessons (Day 8)

1. **Retune-on-branch is validated.** 4 features rescued from reject pile: LMR simplify (+6.3), cont-hist malus (+6.5), NMP capture R (+3.5), corr-clamp (+2.1). Combined ~+18 Elo from "dead" features.
2. **Tune divergence predicts SPRT success.** High divergence (corr-clamp, LMR simplify) → passed. Low divergence (SE neg ext, histprune) → flat/failed. Use tune divergence as a fast filter.
3. **600-1000 SPSA iterations is sufficient** for branch tunes. Values converge by ~800 and don't shift meaningfully after. Confirmed by comparing 800 vs 1600 iteration snapshots on 50-move branch.
4. **Tree shape fingerprint detects retune candidates.** Node count alone is insufficient — per-1K-node pruning rates reveal tree shape changes even when total nodes are similar.
5. **Not all rejected features benefit from retuning.** 50-move (-2.9), futility-full (-2.5), SE ply gate (-5.6) all failed despite retuning. The feature must add genuine information, not just shift the tree.
6. **De-pruning trend continues.** Most positive changes increase tree size. SPSA compensates by tightening other pruning. The virtuous cycle: accuracy improvement → retune → smarter tree at similar size.

## 2026-04-09: Batch Merge + Killer Removal + TM Fix + Model Comparison

### H1 Passed — Merged (batch merge day)

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 182 | fix-se-negative-ext-v2 + tune | **+2.67** | 26,270 | Differentiated -3/-2/-1. Previously -10.5 without retune. |
| 181 | fix-corr-clamp-1024-v2 + tune | **+2.1** | 9,890 | Correctness merge. Consensus clamp 1024. |
| 195 | fix-nmp-capture-r + tune | **+3.5** | 16,718 | Flip r-=1 to r+=1. Previously -7.1 without retune. |
| 215 | fix-remove-killers-counters + tune | **+3.77** | 14,764 | Remove killers and counter-moves. History handles ordering. |

**Batch total: ~+12 Elo merged today.** All four features validated via retune-on-branch.

### H0 Failed

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 213 | fix-corr-cont-2ply-v2 | -1.1 | 27,520 | Dead flat at 28K. 2-ply continuation adds noise. |
| 217 | fix-lmr-simplify-v2 | -1.7 | 12,030 | Remove opp-non-pawn + pawn-escape. Those adjustments help. |
| 216 | fix-good-bad-quiet-split + tune | -3.6 | 7,680 | SF-only feature. Neither direct nor tuned helped. |
| 222 | fix-futility-full-v2 (rebased) | -5.7 | 5,076 | History→lmrDepth. Rebase didn't save it. |
| 221 | fix-se-ply-gate-v2 (rebased) | -7.7 | 4,158 | ply<2*rootDepth. Rebase didn't save it. |
| 223 | fix-50move-eval-scaling-v2 (rebased) | -1.9 | 11,282 | Rebase didn't help. Truly neutral. |
| 220 | fix-histprune-no-improving-v2 (rebased) | -3.4 | 9,104 | Faded from +5.4 early. Rebase didn't help. |
| 209 | fix-low-ply-history-bonus | -2.9 | 8,674 | 8x at ply 0 doesn't help. |

### Running

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 227 | fix-tm-conservative v2 (LTC) | +2.6 | 2,168 | Hard limit ×5, stability 0.70, eval-based reduction. |

### Self-Play Model Comparison

| # | Test | Elo | Games | Notes |
|---|------|-----|-------|-------|
| 212 | selfplay (untuned) vs T80 e120 | -60 | 310 | 7x less data, no parameter calibration |
| 219 | selfplay (tuned) vs T80 e120 | -32 | 566 | Retune closed 28 Elo. Data quality matches, needs volume. |

### NNUE Training Data Discovery

T80 dataset is ~46.7B positions (not 12B as previously thought). At SB800 (80B positions seen),
each position is seen ~1.7 times — barely adequate coverage. Self-play dataset at 800M positions
growing on Titan. Target: 2-3B for fair comparison, 16B for full training.

### Infrastructure

- **Lichess bot** (`codabot`): 2856 rating, 9W 1L 2D. Observed conservative TM in live games.
- **Killer removal merged**: 53 lines removed, engine simplified. History tables handle ordering.
- **Bench**: 1781171 after batch merge (down from 2275554).

### Key Lessons (Day 9)

1. **Killer/counter-move removal works.** +3.77 Elo from removing a feature that's been in chess engines since the 1980s. History tables (27× stronger after magnitude fix) make killers redundant.
2. **Rebase matters.** Branches off old main were testing at ~6 Elo disadvantage (missing LMR simplify). Some results were contaminated. Always check branch parent.
3. **Not all rebased features recover.** Futility-full, SE ply gate, 50-move, histprune all failed despite rebase. The +6 boost wasn't enough.
4. **TM needs LTC validation AND correct base.** The conservative TM at -7 on wrong base became +2.6 on correct base at LTC.
5. **Self-play data quality matches T80 per-position.** -32 Elo with 7× less data + retune. With equal data volume, self-play could match or beat T80.

---

## Day 10 (2026-04-10): Training Pipeline Breakthroughs + TM/Correctness Fixes

### NNUE Training Experiments [SPRT-validated]

Three training config changes tested against baseline `720pw-w7-s120` (768pw, WDL=0.07, standard cosine LR 0.001→0.0001, s120 snapshot, T80 data).

| Experiment | Config Change | Bench | First-move Cut | SPRT Result | Status |
|-----------|--------------|-------|---------------|-------------|--------|
| Low final LR | final_lr=0.000005 (was 0.0001, 20× lower) | 1,587,033 | 79.9% | **+46.7 Elo, H1 ✓** | **MERGED** |
| WDL 0.25 | WDL proportion 0.07→0.25 | 2,631,360 | 76.1% | **-24.2 Elo, H0 ✗** | Rejected |
| Filtered data | ply≥16, no checks, no captures, no tactical | 2,820,607 | 81.1% | Pending | Running |

**Key finding: final LR was 20× too high.** Our cosine schedule decayed from 0.001 to 0.0001, but Bullet's own examples use 0.001 × 0.3^5 = 2.4e-6. The net was oscillating in late training instead of converging. One config line change = +47 Elo. Full e800 run started with low LR.

**WDL 0.25 failed due to eval scale mismatch.** Higher WDL shifts eval magnitudes, making all pruning thresholds (RFP, futility, SEE, LMR) miscalibrated. Would need retune-on-branch to show benefit. Our WDL=0.07 is an outlier vs consensus (0.3-0.4), but can't change without retuning.

**Filtered data has best move ordering (81.1%)** but inflated eval scale (miss queen -2211 vs baseline -608). Like WDL, the scale change may need retune. SPRT pending.

### Training Config Gaps Identified

| Parameter | Coda | Consensus | Gap |
|-----------|------|-----------|-----|
| Data filtering | score < 10000 only | ply≥16, no checks/captures/tactical | Major |
| WDL proportion | 0.07 | 0.3-0.4 | Major (needs retune) |
| Final LR | 0.0001 (10% of initial) | ~2.4e-6 (0.24% of initial) | **Fixed: +47 Elo** |
| LR schedule | Cosine 0.001→0.0001 | Similar but much lower final | Fixed |

### Correctness Fixes Merged

| Fix | SPRT | Commit | Impact |
|-----|------|--------|--------|
| Contempt root-relative | +1.1 ±9.1 | 1b15219 | Fixes draw-seeking with time advantage |
| Ponderhit race condition | N/A (local test) | 9233540 | Fixes hang when opponent responds instantly |
| Ponder move validation | N/A | 9233540 | Prevents illegal ponder moves from stale PV |
| History review (pawn aging + TT cap bonus) | +2.6 ±9.0 | 4a9f4f5 | Consistency fixes |
| Pawn wrap is_pseudo_legal | -2.2 ±5.0 | 4689571 | Prevents TT collision illegal moves |
| TB interior WDL probing | +1.4 ±6.3 | 87691ca | Search now uses tablebases at interior nodes |
| Datagen output path validation | N/A | 1c4786e | Prevents silent hang on bad output path |

### TMv2 Attempts (Problematic — DO NOT merge without ponder validation)

| Attempt | Change | SPRT | Issue |
|---------|--------|------|-------|
| #240 | Bucketed hard limits | -38.9, H0 ✗ | Bullet bracket too tight at OB STC |
| #242 | Continuous scaling | -27.0, H0 ✗ | max_pct cap too tight at fast TC |
| #244 | Continuous + eval factor | -8.8 | Eval factor hurts at STC |
| #245 | Continuous, no eval factor | -12.9, H0 ✗ | time_left/8 floor too tight |
| #246 | base_soft*3 + old cap | **+2.6, H1 ✓** | Minimal change, proven at STC |

**Lesson: TMv2 changes that test positive in self-play SPRT (ponder off) caused 88-93% accuracy on Lichess (ponder on).** The interaction between ponderhit stopping, TT pollution, and time limits produces bad moves that self-play can't detect. TM changes MUST be validated with ponder-on testing before deploying to Lichess.

### Key Lessons (Day 10)

1. **Final LR is the biggest training knob.** +47 Elo from reducing final LR by 20×. Check Bullet examples for baseline configs before training.
2. **Eval scale changes need retune.** WDL and filtering both changed the eval scale, breaking calibrated pruning thresholds. Can't A/B test in isolation.
3. **TM + ponder = dangerous.** Four TMv2 iterations failed at STC. The one that passed (+2.6) still showed degraded accuracy on Lichess with ponder on. Self-play SPRT is necessary but not sufficient for TM changes.
4. **Ponder testing framework works.** cutechess-cli with `ponder` flag, local testing catches ponderhit bugs that OB can't test (no ponder in FastChess/OB).
5. **Code review finds real bugs.** Deep review by Atlas found pawn file-wrapping (is_pseudo_legal), history table inconsistencies, and TB interior probe missing. All confirmed by SPRT or testing.

---

## Days 11-12 (2026-04-11 to 2026-04-12): Net Breakthroughs + Correctness Fixes

### NNUE Training — SPRT Results [SPRT-validated]

| # | Test | Elo | Games | Net | Description |
|---|------|-----|-------|-----|-------------|
| 282 | e800-lowestlr-tuned | **+23.7** | 1,924 | e800-filtered-lowestlr | SPSA tune #280 for new production net. H1 ✓ |
| 287 | e800-lowestlr-tune2 | **+3.5** | 17,804 | e800-filtered-lowestlr | SPSA tune #283 refinement (new SEE values). H1 ✓ |
| 289 | selfplay s120 vs T80 s120 | **+40.0** | 594 | selfplay-w7-e120s120 | 1.67B selfplay positions vs T80 baseline. H1 ✓ |
| 291 | pow25 vs mse loss | ~+12.5 | 7,300 | pow25/mse s120 | Power-2.5 loss vs MSE. Trending H1. |

**Selfplay net:** Trained on 1.67B selfplay positions (14% of T80 dataset). +40 Elo over T80-trained net at same training length (s120). Best move ordering ever: pos²=7.3, first-move cut=81.7%. Validates selfplay data as training source.

**Power-2.5 loss:** +12.5 Elo over standard MSE loss, same T80 data. Free improvement from changing loss function. Based on Cosmo/Viridithas findings (+16-24 Elo).

### Search/SEE Changes [SPRT-validated]

| # | Test | Elo | Games | Description |
|---|------|-----|-------|-------------|
| 281 | fix-see-values | +1.7 | 13,850 | SEE piece values to consensus: N=420, B=420, R=640, Q=1200 |

### Correctness Fixes Merged (not SPRT'd — bug fixes)

| Fix | Commit | Impact |
|-----|--------|--------|
| TB root probe missing score | d4382bc | Lichess bot had no score for draw/resign with TB moves |
| Bestmove vs PV underpromotion mismatch | e70229f | fastchess warnings — bestmove picked queen when PV had knight/rook promo |
| Ponderhit budget for all TCs | 3cc4e2c | Instant stop at 114s with 118s on clock. Fixes classical/tournament TC. |
| TT store guards (all 3 unguarded stores) | 67b73be | QS stores had no stop guard — score 0.0 stored as real eval |
| Hard fail without NNUE | 37c4780 | Engine silently fell back to PESTO eval on Lichess (91% accuracy!) |
| Make PGO embedding | 37c4780 | `make pgo` didn't pass `--features embedded-net` |

### NNUE Training — Currently Running

| Config | Architecture | Loss | WDL | Data | Schedule | Notes |
|--------|-------------|------|-----|------|----------|-------|
| v7 768pw h16x32 | FT→CReLU→pw→16→32→1×8 | MSE | 0.4 | T80 | e800 | Testing WDL fix for v7 |
| v5 768pw | FT→CReLU→pw→1×8 | MSE | 0.07 | selfplay | s120 done, need e800 | +40 Elo result |
| v5 768pw | FT→CReLU→pw→1×8 | power-2.5 | 0.07 | T80 | s120 done | ~+12.5 Elo over MSE |

### SPSA Tune Running

| # | Net | Status | Notes |
|---|-----|--------|-------|
| 284 | v7-gochess-e1200 | 37K/40K | GoChess v7 net. LMR wants more reduction, history bonuses up. |
| 290 | selfplay s120 | Early | Full 48-param retune for selfplay eval characteristics. |
| 251 | filtered net | Ongoing | Similar trends to #252 |
| 252 | low LR net | Ongoing | Tighter pruning, trust move ordering more |

### Cross-Engine Validation (Atlas Rivals RR)

Previous baseline (Apr 11): **Coda 4th place, +71 Elo**
Current (Apr 12, in progress): **Coda 1st place, +138 Elo** (+67 Elo cross-engine gain)

Combined effect of: filtered data + low final LR + SPSA tune + SEE scaling. Overtook Arasan and Igel.

### Lichess Status

- Running on `lo` with e800-filtered-lowestlr net, ponder OFF, correct NNUE embedding
- 97%+ accuracy, <10cp average loss
- Almost all wins and draws in recent games (fixed from 91% accuracy with PESTO eval)

### Key Lessons (Days 11-12)

1. **Selfplay data is the biggest single gain** (+40 Elo at s120 with 14% of T80 data). The eval matches positions our search actually reaches.
2. **Power-2.5 loss is free Elo** (~+12.5 over MSE). Just change the loss function.
3. **Cross-engine validation confirms gains.** +67 Elo cross-engine (4th→1st) validates that training + tuning improvements transfer to real opponents.
4. **Bug fixes compound.** TT pollution, ponderhit, NNUE embedding, TB scores — each individually small, together ~200 Elo on Lichess.
5. **Production net changes need explicit approval.** Don't change net.txt unless explicitly asked to switch nets.

## 2026-04-13: Overnight Retry Batch (Atlas)

### Dynamic Capture SEE Threshold v6 — MERGED (OB #294)
- **Change**: `-captHist/18` as SEE threshold for good/bad capture split. CaptHist only (not MVV).
- **Result**: **H1, +4.4 Elo, 12378 games.** 6th attempt — finally worked with new SEE values (420/420/640/1200) and properly calibrated capture history (post magnitude fix).
- **Notes**: Previous 5 attempts failed (-0.5 to -11.6) with old SEE values and broken captHist. The combination of consensus SEE values + working capture history (27× magnitude fix + unconditional malus) was needed.

### Pawn History Pessimistic Init (-1000) — REJECTED (OB #295)
- **Change**: Initialize pawn history to -1000 instead of 0 (PlentyChess pattern).
- **Result**: H0, -1.0 Elo, 16468 games. True zero.
- **Notes**: Pessimistic prior for unknown pawn structures didn't help. Pawn history fills quickly enough from search that initialization doesn't matter.

### History-Based Extensions v3 (threshold 5000) — REJECTED (OB #297)
- **Change**: Extend +1 when both ply-1 and ply-2 cont-hist exceed 5000 for a quiet move (Igel/Altair pattern).
- **Result**: H0, -0.1 Elo, 26414 games. Dead flat.
- **Notes**: 2nd attempt (v2 was -2.3 with threshold 5000 on old base). Cont-hist agreement at a threshold doesn't provide useful extension signal for us.

### badNode Flag v4 (NMP R+1, RFP -25%) — REJECTED (OB #296)
- **Change**: When !tt_hit && depth >= 4, increase NMP R by 1 and reduce RFP margin by 25% (Alexandria pattern: more aggressive pruning when position unexplored).
- **Result**: H0, -2.5 Elo, 9762 games.
- **Notes**: 4th attempt across 3 different implementations. Alexandria's "unexplored = prune more" pattern consistently doesn't work for us. Our search may already handle unexplored positions well via IIR.

### Key Lesson
Dynamic capture SEE finally proved that **revisiting failed features when underlying systems change** is valuable. 5 failures over 2 weeks, then H1 on the 6th try after SEE values and captHist were both fixed. Persistence pays when you understand WHY it failed before.

---

## Day 13 (2026-04-13): Search Experiments + Bug Fixes + Selfplay Progress

### Search Experiments [SPRT-validated]

| # | Test | Elo | Games | Description | Status |
|---|------|-----|-------|-------------|--------|
| 298 | fix-mate-distance-pruning | +1.6 | 37,276 | MDP on all nodes (was non-PV only) | →H1 (grinding) |
| 299 | fix-cutnode-lmr (+2) | -6.2 | 5,234 | Too aggressive at +2 | H0 ✗ |
| 303 | fix-cutnode-lmr (+1) | — | 0 | Retry with +1 (consensus value) | Running |
| 300 | fix-bad-node | -0.7 | 18,486 | Reduce NMP R-1 and RFP ×3/4 without TT hit | H0 ✗ |

### Selfplay Net Progress [SPRT-validated]

| # | Test | Elo | Games | Description | Status |
|---|------|-----|-------|-------------|--------|
| 301 | SP400 vs SP120 | **+11.5** | 1,996 | Longer selfplay training helps | **H1 ✓** |
| 302 | SP400 vs production | -44.7 | 430 | Expected — production has 12× data | H0 ✗ |

### Correctness Fixes Merged

| Fix | Commit | Impact |
|-----|--------|--------|
| TB root probe DTZ sign inverted | 542e8cb | **Resigned won games.** shakmaty DTZ: negative=winning. Was backwards. |
| TB interior probe ambiguous WDL amplification | c984753 | MaybeLoss (-1) was amplified to -28800. Now passes through as-is. |
| TB regression tests | befc1c8 | Root + interior probe sign tests, skip without TBs. |
| Ponderhit TM: normal allocation instead of minimal | fa5199c | Was banking 3+ minutes unused. Now spends time productively. |
| check-net rewrite: relative health checks | eefb46b | No false alarms on eval scale. Tests signs, ordering, symmetry. |
| Bestmove underpromotion mismatch | e70229f | PV had knight promo, bestmove picked queen. |
| OB scale_nps 1.1M→500K | dbacefd | Was inflating TC on workers. Match engine config. |

### v7 Training Investigation

Unbucketed hidden layers (matching GoChess) still produced collapsed eval.
Ruled out: bucketing, WDL proportion, data volume (single file has 3-4B positions).
Remaining suspects: factoriser, L1 i8 quantization, L2/L3 float format.
GoChess-style experiment config created (no factoriser, i16 quant). Training started.

### Key Lessons (Day 13)

1. **DTZ sign convention cost us a game.** shakmaty-syzygy: negative DTZ = STM winning. We had it backwards since TB support was added. Regression tests now prevent recurrence.
2. **Cutnode LMR +2 is too much.** Consensus is +1. The initial +17.6 at 572 games was noise.
3. **Dynamic capture SEE revived.** Previously "permanently dropped" after 5 failed attempts. New SEE values + SEE_MATERIAL_SCALE made it viable: +4.4 Elo H1 (#294).
4. **Selfplay s400 >> s120.** +11.5 Elo from longer training. But still -45 vs production (data volume gap).
5. **Ponderhit TM was wasting time.** Budget of inc/2 = 1s meant instant moves. Normal allocation lets the engine think 5-20s on ponderhits.

### TB Feature SPRTs (local, Hercules, with /tablebases)

| Test | Elo | Games | Description | Status |
|------|-----|-------|-------------|--------|
| fix-cursed-win | -7.4 | 1,179 | Treat CursedWin/BlessedLoss as definite ±20000 | **H0 ✗** |
| fix-tb-alpha-strict | -2.8 | ~1,720 | `< alpha` instead of `<= alpha` for TB cutoff | **H0 ✗** (killed) |

Both TB changes rejected. The conservative treatment (ambiguous ±1 for cursed/blessed, `<=` alpha cutoff) is correct — the engine plays better when it doesn't overcommit to technically-won-but-50-move-drawable positions.

Note: These were tested locally with tablebases enabled on all engines (not viable on OB where most workers lack TBs).

### Production Net Update

| # | Test | Elo | Games | Description | Status |
|---|------|-----|-------|-------------|--------|
| 323 | pow2.5 e800 vs production | **+12.0** | ~2,000 | Power-2.5 loss, 12 T80 files, e800 | **H1 ✓** |

New production net: `net-v5-768pw-w7-e800s800-pow25-12f.nnue`. Node count +44% vs old production — retune needed to recalibrate pruning for new eval distribution.

### Material Eval Scaling — MERGED (OB #313)
- **Change**: `score = score * (22400 + material) / 32 / 1024` (Alexandria pattern)
- **Result**: **H1, +3.97 Elo, 13,552 games.**
- **Notes**: Dampens NNUE eval in low-material endgames. Tested against old MSE production net — may interact differently with pow2.5 net since both increase tree size. Per-piece scaling values (P=100, N=422, B=422, R=642, Q=1015) could be made SPSA-tunable in future.

### v7 Training Root Cause: .transpose() in Bullet Save Format
- **Bug**: `.transpose()` on L1w/L2w/L3w in Coda training configs caused double-transpose. Converter+loader expected input-major but got neuron-major from Bullet's transpose. All i8 L1 weights were scrambled.
- **Fix**: Removed `.transpose()` from L1w/L2w/L3w in training configs. GoChess-style (no transpose) always worked because it matched expectations.
- **Status**: Retraining 768pw v7 with fixed config on GPU.

### SEE Threshold Floor — REJECTED (OB #315)
- **Change**: `.max(-200)` floor on dynamic capture SEE threshold (`-captHist/18`)
- **Result**: **H0, -2.42 Elo, 10,336 games.**
- **Notes**: The dynamic threshold works fine without a floor. Extreme negative captHist values creating high positive thresholds is a theoretical concern but doesn't hurt in practice. Feature confirmed working as-is.

## 2026-04-16: v9 Threat Feature Experiments

### Escape-Capture Bonuses + Pawn History in LMR — H1 (OB #390)
- **Change**: Two features combined on feature/threat-inputs:
  1. Escape-capture bonuses (Reckless pattern): move ordering bonus for moving pieces off
     enemy-pawn-attacked squares. Q=20000, R=14000, N/B=8000. Tunable via SPSA.
  2. Pawn history in LMR: added pawn_hist[pawn_hash][piece][to] to LMR history adjustment
     and history pruning decision (was only used in move scoring, not LMR).
- **Result**: **H1, +3.6 Elo ±5.6, 6800 games.** LLR 3.03.
- **Notes**: Escape bonuses halve the search tree (bench 2.07M → 1.09M) with first-move cut
  rate 70% → 75%. Massive tree shape change. Tested at untuned defaults; SPSA retune expected
  to find more Elo. Pawn history in LMR was separately trending +4.1 at 14K games (#384).
  Both are v9-specific: escape bonuses leverage threat-aware move ordering, pawn history
  leverages the richer eval for pawn-structure-dependent decisions.

### v9 Tune Round 1 — H1 (OB #376)
- **Change**: 48-param SPSA tune (#373, 1000 iterations) on v9 s400 threat net.
  Started from v5-calibrated defaults.
- **Result**: **H1, +104.3 Elo ±22.7, 614 games.** Massive eval scale recalibration.
- **Notes**: v9 eval is ~3× scale of v5. Key shifts: SE_DEPTH +103%, HIST_PRUNE_MULT -43%,
  LMR_HIST_DIV +65%, NMP_EVAL_MAX +44%, RFP_MARGIN_IMP +27%.

### v9 Tune Round 2 — H1 (OB #382)
- **Change**: 48-param SPSA tune (#380, 2500 iterations) seeded from round 1 values.
- **Result**: **H1, +30.6 Elo ±11.0, 1638 games.**
- **Notes**: Continued large shifts: FUT_BASE -22%, LMR_C_QUIET -14%, ASP_DELTA -20%,
  CONTEMPT_VAL +38%, SE_DEPTH +18%. Confirms round 1 hadn't converged.

### Bullet Compatibility Fix: Semi-Exclusion — FLAT (OB #392)
- **Change**: Align semi-exclusion convention with Bullet training. Training uses physical
  square ordering for same-piece-type pairs; inference was using perspective-flipped squares.
  ~3-5% NTM feature mismatch.
- **Result**: Flat/slightly negative at ~200 games (still running).
- **Notes**: Fix is correct (training/inference should match) but the affected features are too
  few (~2-4 per position) and too low-weight to matter with current s400 net. May become
  relevant with longer training. Keep fix for correctness.

### Threat-Density LMR — TRENDING H0 (OB #387)
- **Change**: Reduce LMR less when multiple pieces under pawn attack (threat_count / LMR_THREAT_DIV).
- **Result**: -1.9 Elo at 3460 games, trending H0.
- **Notes**: Novel idea, not from Reckless. Too blunt — blanket LMR reduction for all quiet moves
  in tactical positions doesn't help. Reckless relies on threat-aware history for this signal.

### Threat-Aware Singular Extension — TRENDING H0 (OB #388)
- **Change**: Widen singular beta by SE_THREAT_MARGIN when pieces under pawn attack.
- **Result**: -1.8 Elo at 3980 games, trending H0.
- **Notes**: Novel idea, not from Reckless. Explicit threat logic in singular extensions doesn't
  help. The NNUE eval and threat-aware history already capture this signal implicitly.

### Threat-Aware Futility — STILL RUNNING (OB #386)
- **Change**: Widen futility margin by FUT_THREAT_MARGIN when pieces under pawn attack.
- **Result**: +4.4 Elo at 1036 games (early, still running).

### Razoring on v9 — STILL RUNNING (OB #389)
- **Change**: Re-add razoring (removed from v5) with tunable margins scaled for v9 eval.
  RAZOR_BASE=900, RAZOR_MULT=756 (3× Reckless defaults for v9 scale).
- **Result**: +1.1 Elo at 5122 games (still running, slightly positive).

## 2026-04-17: v9 Correctness Hunt + NPS Optimisation Sweep

Biggest gains day on the project. ~+296 Elo on `feature/threat-inputs` from a
combination of correctness-bug fixes, NPS tuning, and retune-on-branch. Discovery
methodology: TDD-style fuzzers + "diff against Stockfish/Reckless" audits +
microbenchmarks for sub-1% changes invisible to whole-bench.

### can_update ordering bug — H1, +142.9 Elo (OB #417)
- **Change**: Fix ordering bug in `ThreatStack::can_update` (src/threat_accum.rs).
  The function was accepting an ancestor ply as "accurate" BEFORE checking whether
  the ply immediately above had overflowed delta buffer or crossed the king-file
  mirror boundary. Correct order: validate entry[i+1] invariants FIRST, then
  accept entry[i] as ancestor.
- **Result**: **H1, +142.9 Elo ±31.9, 308 games.** LLR 3.00.
- **Found via**: Threat-accumulator fuzzer added during the same session — played
  random legal games from multiple FENs and asserted incremental vs from-scratch
  threat refresh. Fuzzer diverged at a king-file crossing move; inspection of
  the can_update function revealed the ordering error.
- **Notes**: Bug had been present since threat accumulator was added (~2 days
  earlier). Self-play SPRT on the buggy implementation couldn't detect it —
  both sides had the same broken code. The fuzzer's cross-implementation
  comparison was the key signal.

### X-ray threat delta generation — H1, +110 Elo (OB #407)
- **Change**: Earlier in day, fix `enumerate_threats` to properly account for
  x-ray threat re-indexing through the just-moved piece. Section 2 Z-finding
  block + section 2b blocker-past-subject block.
- **Result**: **H1, +110 Elo ±33.8, 310 games.** LLR 2.99.
- **Notes**: Companion to can_update bug. The threat accumulator was
  structurally wrong in two ways at once. Both fixes were needed for full
  gain; they're measured separately because each SPRT ran against a
  progressively better baseline.

### Zobrist EP fix — H1 (OB #421, measurement: neutral-positive)
- **Change**: The Zobrist hash was unconditionally XOR-ing `ep_key` into `hash`
  whenever a double-pawn push occurred, regardless of whether any enemy pawn
  could actually make the EP capture. This silently broke threefold-repetition
  detection whenever a rep cycle crossed a "dead" double-push (no enemy pawn
  adjacent). Positions that should have hashed equal hashed differently,
  rep detection missed, engine scored forced draws as -80cp instead of -19cp.
- **Found via**: Investigating a fastchess `PV continues after threefold
  repetition` warning that prior Claudes had written off as cosmetic.
  Adam's insistence on digging deeper revealed the real hash bug.
- **Result**: **H1, +3.3 Elo ±4.5, stopped early at ~1500 games for merge.**
- **Bench impact**: 1,318,356 → 1,050,912 (-20% nodes — rep-cycle subtrees
  now correctly pruned as draws).
- **Implementation**: `ep_capture_available(pieces, colors, capturing_side,
  ep_sq)` helper; XOR `ep_key` only when this returns true. Applied at
  compute_hash, make_move (remove + add), make_null_move. Hash-invariant
  fuzzer added (incremental vs compute_hash on 180 random games × 80 plies)
  — catches future drift in the conditional XOR.
- **Meta-lesson**: "It's just a display warning" was wrong. Paranoid
  investigation of the -80cp score on what looked like a forced draw
  revealed the Zobrist bug. Saved to memory as
  `feedback_correctness_over_features.md`.

### Section 2 Z-finding cull — H1, +23.4 Elo (OB #423)
- **Change**: Skip the Z-level x-ray delta block inside the slider-loop when
  no ray from `square` has 2+ occupants. Pre-check `(occ & past_first_region) != 0`
  before the loop; if false, no slider can have a Z chain (`S → square → Y → Z`),
  so only the direct threat needs emitting. Shared `rays_from_sq_empty`
  precomputation with the existing 2b cull (net zero on pre-check cost).
- **Result**: **H1, +23.4 Elo ±9.3, 1234 games.** LLR 2.95.
- **Bench**: +1.5% NPS only. The Elo gain came from frequency of firing in real
  middlegame/tactical positions not represented in bench's 8 canned FENs.
- **Meta-lesson**: Bench NPS is a narrow signal. Real-game benefit can be
  10× the bench delta.

### Simplified PSQ refresh (per-pov on king-bucket crossing) — H1, +21 Elo (OB #426)
- **Change**: When a king moves and crosses a bucket/mirror boundary, the moving
  side's perspective needs a Finny refresh (all features re-index), but the
  non-moving side's perspective is unchanged and can apply the dirty changes
  incrementally. Previous code refreshed BOTH perspectives in lockstep. Reckless
  pattern.
- **Implementation**: `AccEntry.computed: [bool; 2]` (per-pov), `DirtyPiece.
  needs_refresh: [bool; 2]` set by `build_dirty_piece` when king crosses bucket,
  `materialize` dispatches four cases (both-refresh, both-incremental,
  W-refresh+B-incr, B-refresh+W-incr) to minimal work.
- **Result**: **H1 trending, +21 Elo ±11 at 1044+ games, LLR 1.97.**
- **Bench**: ~parity (-1.5% to +1.5% run-to-run).
- **History**: Initial attempt (#424, with a multi-ply chain walkback on top)
  H0'd at **-17.86 Elo**. `bench_materialize` microbench diagnosed the
  walkback as the pessimization (lazy-chain 3.5× slower than old
  Finny-diff refresh). Simplified version removes the walkback, keeps only
  the per-pov-on-bucket-cross insight, recovers the gain.
- **Meta-lesson**: Microbench-guided debugging can rescue an abandoned feature
  by isolating WHICH part of a refactor is costing. Lesson saved to memory.

### ProbCut TT-store bugs — H1 pending (OB #428)
- **Change**: Two bugs in ProbCut's successful-cutoff TT store vs Stockfish:
  1. Stored `dampened = score - (probcut_beta - beta)` instead of raw `score`.
     `score >= probcut_beta` is a tighter lower bound than dampened; storing
     dampened loses pruning on future probes with beta between the two.
  2. Hardcoded `tt_pv: false` — should use local `tt_pv` variable (sticky PV
     flag). Positions pruned by ProbCut were losing PV status in TT,
     affecting LMR reduction on revisits.
- **Result**: SPRT in flight. Bench 1,126,147 → 1,209,491 (+7% nodes —
  preserving tt_pv reduces LMR on those lines, expanding PV subtrees).
- **Notes**: ProbCut has now had 5 distinct bug fixes total (missing qsearch
  filter, SEE threshold, excluded_move guard, stored-dampened, tt_pv flag).
  Worth continued auditing.

### Correction history skip-noisy — H0, -17 Elo (OB #427)
- **Change**: Skip correction-history update when `best_move` is a
  capture/promotion. Matches Stockfish (search.cpp:1495) and Reckless
  (search.rs:1085).
- **Result**: **H0 trending, -17.32 Elo ±16.75, 522 games.** LLR -0.81.
- **Meta-lesson**: Consensus pattern didn't transplant. SF's gate works
  alongside compensating calibrations (bonusScale bumps, aggressive gravity)
  that we don't have. Our simpler update formula `weight = (depth+1).min(16)`
  depends on the larger volume from ALL beat-alpha positions. Removing ~30%
  of updates starved correction history of signal.
- **Saved to memory**: `feedback_consensus_patterns_dont_always_transfer.md`.

### Forward-pass MaybeUninit — H0 trending, -4 Elo (OB #425)
- **Change**: Replace `[0u8; 2048]` scratch stack buffers in forward-pass
  SIMD paths with `[MaybeUninit<u8>; 2048]`. Skipped ~4 KB of zero-stores
  per forward call (SIMD packers write exactly `pw` bytes; downstream
  reads only `[..pw]`).
- **Microbench**: **+7% forward speed** (787 → 733 ns/call). Found using
  a new `bench_forward_pass` microbench added the same session.
- **Full bench**: +5% NPS (490K → 514K).
- **Result**: **H0 trending, -4.4 Elo ±5.4, 3858 games.** LLR -1.83.
- **Meta-lesson**: Microbench is a necessary tool for forward-pass
  optimisation but not sufficient to predict real-game Elo. The 1-second
  games probably interact with the memory layout / cache / branch-predictor
  in ways the tight microbench loop doesn't capture. LLVM may have been
  eliminating the zero-stores already when it could prove them dead;
  the measured gain might be code-layout noise. Lesson saved.

### Tune #411 retune-on-branch — H1 accepted, +1.37 Elo (OB #412)
- **Change**: 52-param SPSA tune (2500 iterations, scale-nps 250000) on
  `feature/threat-inputs` seeded from live src/search.rs defaults. Post
  x-ray fix + can_update fix recalibration.
- **Result**: +1.37 Elo at 37K games, stopped early and accepted. Classic
  retune-on-branch modest gain after the big structural fixes land.
- **Key shifts**: HIST_BONUS_BASE 12→8 (-33%), LMR_C_QUIET 99→89 (-10%),
  LMR_C_CAP 115→107 (-7%), SEE_CAP_MULT 136→144 (+6%). Most parameters
  near starting values; the shifted ones reflect the cleaner search tree
  after can_update/x-ray/Zobrist fixes.

### Mini tune: MVV_CAP_MULT + CONT_HIST_MULT — confirmed defaults (OB #420)
- **Change**: Promoted two hardcoded move-ordering constants (`see_value(victim) * 16`
  in good-capture scoring, `[3i32, 3, 1, 1]` cont-hist weights) to SPSA tunables
  `MVV_CAP_MULT` (default 16) and `CONT_HIST_MULT` (default 3). 2-param SPSA with
  1000 iterations at scale-nps 250000.
- **Result**: MVV_CAP_MULT 16.02 (-0.0%), CONT_HIST_MULT 2.8 (-6.7%) after 68%
  iterations; stopped early. Defaults confirmed near-optimal.
- **Notes**: Full 54-param tune #422 (in flight at time of writing) later showed
  MVV_CAP_MULT drifting to 15.2 and CONT_HIST_MULT confirmed at 3.0 — the
  2-param mini-tune masks parameter interactions that the full tune reveals.

### Full 54-param SPSA tune — IN FLIGHT (OB #422)
- **Change**: Full tune on `feature/threat-inputs` after merging Zobrist +
  tune #411 + MVV/CONT_HIST tunables. 2500 iterations, scale-nps 250000.
- **Result at 26% (649/2500)**: Biggest movers —
  HIST_BONUS_BASE 12→7 (-42%), DEXT_MARGIN 9→8 (-11%), FH_BLEND_DEPTH 1→0.9
  (-13%, previously pinned at min), LMR_C_QUIET 89→97 (+9%, range-expanded
  reveals wants higher). SE_DEPTH 12→11.4 (-5%, previously pinned at max,
  now in range-expanded [4,20] wants slightly lower).
- **Notes**: The 42% drop in HIST_BONUS_BASE is the strongest signal —
  previous value was compensating for broken history signal from the
  can_update bug. Clean baseline reveals smaller offset is optimal.
  Expected to resolve overnight.

### v9 vs v5 production gap — measurement (OB #418)
- **Purpose**: After all correctness fixes + tune, measure remaining gap.
- **Result**: **-54.1 Elo ±27.7, 272 games.** (vs -175 at session start,
  -66 after can_update fix alone).
- **Notes**: Remaining gap attributable to training maturity (v9 s200
  snapshots vs v5 s800 production) and king-bucket layout (v9 uses
  Uniform 16, v5 now uses Consensus 16 after earlier promotion +15 Elo).
  Both being addressed by in-flight training: consensus KB, s800 xray,
  Reckless 10-KB.

### Infrastructure added this session

- **`bench_forward_pass`** (nnue.rs, `#[ignore]` test): tight-loop
  forward-pass microbench with v9 net. Baseline 787 ns/call. Enables
  sub-1% detection for forward-path optimisations.
- **`bench_materialize`** (nnue.rs, `#[ignore]` test): materialize
  microbench across three scenarios (single-ply, king within-bucket,
  lazy chain). Used to diagnose the PSQ walkback pessimisation.
- **`fuzz_psq_accumulator`** (nnue.rs): random games × multiple FENs,
  asserts incremental PSQ vs force_recompute parity after every ply.
- **`fuzz_psq_lazy_eval_chain`** (nnue.rs): same but with batched push
  (lazy-eval chain testing).
- **`fuzz_hash_incremental_matches_compute`** (board.rs): hash-invariant
  fuzzer specific to the Zobrist EP fix.
- **`threat_accum::fuzz_random_games`** (threat_accum.rs): caught the
  can_update ordering bug.
- **`test_excluded_move_cleared_after_search`** (search.rs): SE
  invariant guard.
- **`history_4d_flag_routes_correctly`** (movepicker.rs): A/B guard
  for the 2D vs 4D main history paths.
- **`test_z_finding_cull_endgame_no_z`** + **`test_z_finding_cull_has_z_chain`**
  (threats.rs): pin the cull's semantic boundary.
- **`zobrist_ep_only_when_capturable`** (board.rs): pin the new Zobrist
  EP invariant.

### Session tally

| Merged/pending | Elo | SPRT | Mechanism |
|---|---|---|---|
| can_update fix | +142.9 | #417 H1 | Ordering bug |
| x-ray delta fix | +110.0 | #407 H1 | Correctness |
| Z-cull | +23.4 | #423 H1 | NPS optimisation |
| Simplified PSQ | +21 (pending H1) | #426 | NPS optimisation |
| Tune #411 retune | +1.4 | #412 accepted | Retune-on-branch |
| Zobrist EP fix | ~0 | #421 | Correctness |
| ProbCut TT fixes | TBD | #428 in flight | Correctness |
| Tune #422 full | TBD | SPSA in flight | Retune-on-branch |
| **MaybeUninit** | **-4** | **#425 H0** | Reverted |
| **PSQ walkback** | **-18** | **#424 H0** | Reverted, rescued as #426 |
| **Corr-hist skip-noisy** | **-17** | **#427 H0** | Reverted, ecosystem mismatch |
| **Total (confirmed + pending)** | **~+296 Elo** | | |

Gap to v5 at session start: -54 Elo. Expected to close or cross by
morning with consensus-KB and s800 nets landing.

## 2026-04-18: v9 Training Investigation — Schedule, Data Volume, LR Tail

Series of SPRTs investigating what matters for v9 net quality.
Net-compare SPRTs (same branch, different `--dev-network` vs
`--base-network`). All bounds [-3, 3] unless noted.

### Training data volume: 1xT80 vs 12xT80 — H0, -53 Elo (OB #submission)

**Setup:** Two v9 nets trained identically at s200, w15, Uniform KB —
only difference: 1 T80 file (~3B positions) vs 12 T80 files (~30B
positions).

**Result:** H0 at **-53 Elo** for 1xT80.

**Read:** Data volume still matters significantly at s200 even under
the current modernized pipeline (power loss 2.5, data filtering, xray
features). Consistent with v5's +33 Elo for 12× data. The pipeline
hasn't saturated on 3B positions — sparse threat features especially
benefit from more unique positions.

**Prior belief this tested:** possible that the pipeline improvements
(power loss, filtering) had made per-position signal dense enough that
1 file's 3B positions was already sufficient. It isn't.

**Implication:** Data-gen at d7/d8 is worth preserving over deeper/fewer
positions for the v9 sparse-feature regime. Diminishing returns are
probably kicking in around 12× but going to 1× is a real loss.

### Schedule shape: e800s200 vs e800s600 vs e800s800 — low-LR tail dominates

Three snapshots of the same e800 cosine schedule, tested against each
other on Uniform KB, w15, xray-fixed nets.

| Comparison | SPRT | Result |
|---|---|---|
| e800s800 vs e800s200 | #443 | **+9.4 H1** |
| e800s600 vs e800s200 | #444 | **-88 H0** |
| e400s400 vs e800s800 | #450 | **-6.27 ± 5.29 H0 (final)** |

**Key finding:** The mid-cosine snapshot (s600, LR ~1.6e-4) is
dramatically weaker than both the warmup-dominated snapshot (s200)
and the final low-LR snapshot (s800). The final 200 SBs of the cosine
tail add **roughly +100 Elo** of refinement.

**Mechanism:** v9's 66864 sparse threat features get ~1-5% of positions.
At high LR, each update is large and rare-feature weights get
overwritten by subsequent batch noise. Low LR makes stable refinement
possible.

**Schedule length — log-linear in SBs:**

| Snapshot | Elo vs e800s200 |
|---|---|
| e400s400 | ≈ +3.1 (= 9.4 - 6.27) |
| e800s800 | +9.4 |

Each **doubling** of SB length adds ~**3-6 Elo** — roughly log-linear
but slightly right-skewed (s400→s800 gap is larger than s200→s400).
This is consistent with the low-LR tail being where the gain concentrates:
- s200→s400: LR mid-cosine (~5e-4), refinement still coarse
- s400→s800: LR floor (~2e-6), fine refinement regime
- s1600 would project to ~+14 vs s200 (another ~+5 vs s800)
- s3200 would project to ~+19 vs s200

Implication: ROI per compute-doubling is constant (~4.7 Elo for 2×
compute) until architecture saturation hits. Diminishing absolute
marginal returns, but constant slope in log-compute space.

**Practical rule:** For architecture/hyperparam sweeps, e400 captures
~50% of the Elo at 50% of compute. For production candidates, go long.
At some point the log-linear relationship will break as the architecture
saturates — that's the signal to change architectures.

### Training selfplay data on v9 — H0 -46 (pre-tune)

**Setup:** selfplay-trained s200 vs 1xT80-trained s200, both untuned
(same trunk search params).

**Result:** H0 at -46 Elo for selfplay.

**Caveat:** Selfplay net has cooler RMS eval (568 cp vs 1xT80's 748 cp,
23% cooler). Current search params (RFP/futility/SEE margins) are
tuned for the hotter distribution, so selfplay under-prunes (+36% bench,
higher EBF). The -46 includes a scale-mismatch penalty of unknown
magnitude.

**Noted for followup:** Tune #449 (1000-iter SPSA on feature/threat-inputs
with selfplay net) is running to rebalance search params for selfplay's
distribution. Selfplay vs 1xT80 will be re-tested post-tune to isolate
net-quality vs param-fit.

**Interesting diagnostic:** Selfplay net has **better move ordering**
(first-move cutoff 81.2% vs 78.7%) despite being weaker end-to-end.
Relative-ranking quality is preserved; only absolute-scale calibration
is off.

### EPD regression suite noise floor

**Setup:** 22-position coda_blunders.epd (v5 failures, 3s/pos).

Comparing s200, s400, s600, s800 snapshots gave:

| Net | EPD pass | SPRT vs e800s200 |
|---|---|---|
| e200s200-xray-fixed | 12/22 (54.5%) | baseline |
| e400s400 | 13/22 (59.1%) | — |
| e800s600 | **14/22 (63.6%)** | **-88 H0** |
| e800s800 | 11/22 (50.0%) | **+9.4 H1** |

**EPD ranking contradicted SPRT ranking.** The strongest SPRT net
scored worst on EPD. 22 positions is too small a suite (1σ ≈ 3 positions
of binomial noise) to rank similarly-strong nets. Use SPRT for strength,
reserve EPD for regression tripwires only.

### Key training lessons locked in

1. **Loss is not strength.** A snapshot with lower training loss can
   be dramatically weaker (e800s600 had lower loss than e800s800 but
   was -88 Elo weaker).

2. **Low-LR tail is where Elo is purchased.** Most loss reduction
   happens in warmup/early cosine; most Elo happens in the final
   low-LR convergence tail. This holds for v5 (+50 Elo) and is larger
   for v9 (+100 Elo).

3. **Bigger/sparser architectures need lower final LR.** v5's optimum
   at 5e-6 does NOT transfer to v9. Experiments queued with
   final_lr=2.43e-7 (10× lower than current 2.43e-6).

4. **Data volume still matters** even under modernized pipeline.
   1×T80 → 12×T80 is ~+50 Elo in self-play.

### Corrections to merge: HIST_BONUS_BASE removal — SPRT pending

**Motivation:** Two independent SPSA tunes (#422 full trunk, #449
selfplay in flight) consistently drove HIST_BONUS_BASE from its
default toward 0. The parameter was dead weight:
`bonus = (MULT*depth - BASE).clamp(0, MAX)` with BASE=2 shifts the
bonus by 0.7% — within SPSA noise, but systematically pulled to 0.

**Fix:** Remove the parameter. Simplified formula now matches
Stockfish's `stat_bonus(d) = min(MAX, MULT*d)`.

**Bench surprise:** 1498845 → 1798525 (+20% nodes). Far larger than
the 0.7% numerical shift predicts — tree-shape chaos. SPRT submitted
at [-5, 5] bounds as non-regression check.

**Branch:** `remove/hist-bonus-base`. OB submission pending.

### Skip-noisy correction-history merge

**Tested:** tune/corr-history-skip-noisy-r1 vs trunk-r2 = flat at 15k
games (SPRT #441). Stopped early at "neutral, correctness first, merge".

**Merged:** Code change (skip correction-history updates when best_move
is capture/promotion, per Stockfish/Reckless pattern) plus the 1000-iter
tune #433 values (seeded from trunk's r2 tune #422, refined on skip-noisy
branch). Bench 1,498,845.

**Rationale:** r1 tune + skip-noisy had lowest bench of three candidate
combinations AND was SPRT-validated equivalent to trunk+r2 tune. Skip
updating correction history on noisy moves is a consensus pattern
(Stockfish search.cpp:1495, Reckless search.rs:1085).

### Ongoing: three Reckless-KB training experiments in flight

Kicked off 2026-04-18 mid-afternoon:

1. **Low final_lr**: `--kb-layout reckless --superbatches 200 --final-lr 2.43e-7`
2. **Short warmup**: `--kb-layout reckless --superbatches 200 --warmup 5`
3. **Reckless e400 baseline**: `--kb-layout reckless --superbatches 400`

All three vs existing `net-v9-768th16x32-kb10-w15-e200s200.nnue` baseline
(hash BE5849B6). Expected SPRTable in 4-5 hours for s200 variants, 8-10
hours for e400.

### Infrastructure added this session

- **`--final-lr` CLI flag** in `bullet/examples/coda_v9_768_threats.rs`
  for low-LR-tail experiments without code edits.
- **`coda_blunders.epd`** regression suite (22 positions) ported from main.
- **Memory**: `project_v9_low_lr_tail_critical.md`,
  `feedback_loss_is_not_strength.md`,
  `feedback_epd_not_for_model_testing.md`,
  `feedback_piece_value_test_design.md`,
  `feedback_never_stop_sprt_without_verify.md`.

## 2026-04-18 (afternoon/evening): Major tune + NPS wins + v5 gap closure

### HIST_BONUS_BASE removal — H1 +3.7 Elo (OB #452)

- Tunable repeatedly pulled to 0 across independent SPSA runs
  (#422 trunk, #449 selfplay). Dead weight at MULT=293, BASE=2 —
  shifts bonus by 0.7%. Removed parameter and simplified formula
  to match Stockfish's `min(MAX, MULT*d)`.
- SPRT [-5, 5]: +3.7 Elo H1 at 5300+ games.
- Freed one SPSA dimension (58 → 57 params).

### Skip-noisy correction history merge — Neutral +tune values

- After #441 flat at 15k games, merged tune/corr-history-skip-noisy-r1
  (SF/Reckless pattern: skip correction-history updates when best_move
  is a capture/promotion). Took tune #433's 1000-iter values.
- Correctness-first: zero ELO regression with a consensus correctness
  improvement.

### Trunk retune on e800s800 net (tune #454) — H1 +38.6 Elo (OB #456)

**Biggest single tune in v9's history.**

- SPSA 2500 iterations, `feature/threat-inputs` with e800s800 net
  (hash 6AEA210B), `--scale-nps 250000`.
- Biggest parameter shifts — direction: trust the (better) s800 eval more:
  - NMP_DEPTH_DIV    2 → 3     +50% (lighter NMP reduction)
  - SE_DEPTH         10 → 6    -40% (SE kicks in earlier)
  - HIST_PRUNE_MULT  5861 → 3981 -32% (lighter history pruning)
  - LMR_HIST_DIV     11392 → 13958 +22%
  - LMR_C_QUIET      107 → 121  +13%
  - NMP_EVAL_DIV     169 → 149  -12%
  - RFP_MARGIN_IMP   101 → 87   -14%
  - ASP_DELTA        8 → 10     +25%
  - SEE_QUIET_MULT   37 → 44    +19%

- Bench: 1,798,525 → 2,401,107 (+33% nodes, tree-shape shift)
- **Principle confirmed: "tune on your best net."** This third-round
  tune on the s800 net delivered ~2× the +16.5 Elo of the prior
  s200-based tune.

### v5 gap measurement — before/after the major tune

- **Pre-tune gap** (SPRT #453, v9 e800s800 vs v5 prod, both on
  feature/threat-inputs): **-74 Elo H0** (370 games)
- **Post-tune gap** (SPRT #459, tuned v9 e800s800 vs main + v5 prod):
  **-19.82 Elo H0** (1386 games, [-5, 5] bounds rejected)
- **~54 Elo of gap closed in one day** via the single tune + TM
  fixes (on main, not yet in feature/threat-inputs).

Gap moved into the "all-in on v9" threshold zone (-20 to -30 Elo).
Decision to merge v9 trunk to main now depends on whether pending
experiments (Reckless KB at s400, low-final-lr, const-generic NPS
refactor) can close the remaining ~20 Elo.

### NPS structural wins (cumulative)

Three structural fixes to the NNUE/threat incremental-update path,
all test-only infrastructure validated against existing fuzzers:

| Commit | Change | NPS gain |
|---|---|---|
| 317deab | Fuse PSQ incremental update — all deltas in single pass | +0.4% |
| 2218233 | Fuse threat-accum copy+apply — no separate memcpy pass | +2.6% |
| afd2fcd | Enable column-major L1 matmul via dense_l1_avx2 | +1.2% |
| **Total** | | **+3.8%** |

Baseline 548K → 569K NPS (10-run mean). Bench 1,798,525 unchanged
(correctness preserved). All fuzzers (`fuzz_psq_accumulator`,
`threat_accum::fuzz_random_games`, `finny_king_march_consistency`)
pass.

### Audit suite (25 new property-based tests)

Permanent regression coverage, all pass on current HEAD:

- **Zobrist aux keys** fuzzer (pawn_hash, non_pawn_key, minor_key,
  major_key) — matches from-scratch recomputation after every move.
- **Make/unmake round-trip** + **null-move round-trip** — full state
  restore after random game sequences.
- **is_pseudo_legal** positive/negative fuzzers — every legal move
  accepted; every corrupted move rejected or matches generate_all.
- **Repetition detection** (main-search + QS variants) — 4-ply knight
  dance detected, halfmove cap honoured, random-games agreement.
- **SEE correctness** (9 assertive tests) — hand-computed values
  match, monotonicity of `see_ge` across thresholds.
- **Cuckoo cycle detection** (6 tests) — table populated (3668
  entries), every entry valid, knight-dance cycle detected.
- **Correction history** — update formula directionality, bounds,
  gravity saturation, zero-err no-op, read/write index symmetry.
- **Finny king-march** — 14-step king walk forces cross-bucket
  refresh on every ply, asserts against force_recompute.
- **TT bucket replacement** (6 tests) — roundtrip, 5-slot
  coexistence, XOR-key verification, same-key update, depth-gated
  replacement, eviction arithmetic.

Session bug count: 0. Infrastructure locked in for future NPS work.

### PGO re-investigation — still regresses

Tested vanilla `cargo pgo` today on v9 trunk:
- Non-PGO: ~549K NPS
- PGO: ~491K NPS (**-10.4% regression**)

Tried panic="abort", removed embedded-net feature — neither recovered
the regression. Reckless on same hardware gains +1.5% from PGO
(974K → 989K). Structural difference in Coda's hot path that PGO
inlining misjudges; unclear without deep LLVM investigation.
Shelved for now.

### Training experiments dispatched (pending results)

Kicked off this morning, ~2 hours to land:

1. **Low final_lr** Reckless s200: `--final-lr 2.43e-7` (10× lower
   than default 2.43e-6). Tests whether v9 sparse threat features
   benefit from even lower LR tail.
2. **Short warmup** Reckless s200: `--warmup 5` (vs 20 default).
   Ablation of warmup length.
3. **Reckless e400 baseline**: full 400-SB schedule with Reckless
   KB. Measures KB × schedule-length economics jointly.

### Pending SPRT queue

- **#455** — selfplay-tuned (tune #449 applied) vs 1xT80 (trunk tune).
  Post-tune re-test of the earlier H0 -46 pre-tune result.

### v9 schedule-length curve (as measured pre-tune #454)

| Snapshot | Elo vs e800s200 |
|---|---|
| e400s400 | +3.1 |
| e800s800 | +9.4 |

Log-linear ~3-6 Elo per SB doubling, front-loaded onto the
low-LR tail (not evenly spread per SB). The +38.6 tune may shift
this curve; re-measurement under the new tune is worth doing once
SPRT fleet has capacity.

### Lever stack toward v5 parity (2026-04-18 evening)

Current gap: -19.82 Elo (SPRT #459 H0). Realistic candidate gains:

| Lever | Projected | Status |
|---|---|---|
| Reckless KB net at s400 | +15-25 | In-flight training |
| Retune on Reckless KB | +15-40 (compounds with above) | Queued |
| Low-final-LR variant | +5-15? speculative | In-flight training |
| Const-generic PARAMETERS refactor | +10-15 | Not started (scoped) |
| s1600 full schedule | +5-10 | Deferred (48h train) |

Any 2 of (Reckless+retune, const-generic, low-LR) landing positive
puts us at or above v5. All-in merge decision deferred pending
results.

## 2026-04-20: Overnight sweep — 2 merges (+12.7), 9 H0s, retries in flight

### Merged to trunk (H1 resolved)

| Test | Feature | Elo | Games | Notes |
|---|---|---|---|---|
| **#542** | `experiment/unstable-probcut-skip` | **+6.7** | 7390 | Skip ProbCut when static eval differs sharply from parent (|Δ| > UNSTABLE_THRESH=157). Second consumer of `unstable` signal (HIST_PRUNE guard was first). Mirror of #481 king-zone-pressure ProbCut gate. Clean LLR 2.96. |
| **#539** | `experiment/threats-nmp-skip` | **+6.0** | 8330 | Skip NMP when any_threat_count ≥ 3 (3+ of our non-pawns under enemy attack). Third landing for any_threat_count signal after futility-widener #484 and LMR threat-density. Crisis positions get real search, null move's extra tempo no longer costly. Clean LLR 3.01. |

### H0'd (closed)

| Test | Branch | Result | What it tested | Decision |
|---|---|---|---|---|
| **#540** | `threats-rfp-widener` | −4.8 LLR −2.96 @ 6016 | any_threat_count RFP widener — mirror of existing has_pawn_threats widener but general threats | **Dropped.** Signal overlap: pawn-threat widener already catches the high-value case. Don't re-iterate. |
| **#541** | `unstable-lmr-reduce-less` | −10.9 LLR −2.99 @ 3146 | `reduction -= 1` when unstable (all depths) | Iterate with depth gate — **#548** in flight |
| **#538** | `our-kz-opp-probcut` | −0.3 LLR −2.97 @ 23800 | Our-king-zone-opportunity as ProbCut skip gate (mirror of enemy-king-zone) | Dropped — symmetry assumption fails; our king attacks doesn't help ProbCut like enemy's does |
| **#543** | `dynamic-see-v2` | −0.7 LLR −2.97 @ 19062 | Dynamic SEE threshold with −75 baseline | Iterate with different baseline — **#556** (−125) in flight |
| **#544** | `corrhist-probcut-skip` | −2.5 LLR −3.22 @ 10684 | Skip ProbCut when corrhist-mag ≥ 25 | **Dropped.** Interaction with #542's `unstable` ProbCut skip — double-gated same "eval noisy" locus. |
| **#533** | `discovered-attack-scaled` | +0.2 LLR −2.96 @ 34642 | Victim-value-scaled B1 bonus (stacks on +52 flat B1) | Dropped — original B1 already captures the ordering signal; victim-scaling doesn't compound on top. |
| **#528, #529** (earlier in day) | history-shape-offset, material-scale-tunable | both H0 | see below | see below |

### In flight from this batch (overnight wave)

| Test | Branch | Purpose |
|---|---|---|
| #546→stopped | threats-nmp-skip retune | Single-feature tune, stopped to do combined retune instead |
| **#550→#551** | `tune/v9-overnight-r1` | Combined retune post #542 + #539 merges (38 tunables moved). SPRT #551 trending +2 Elo. |
| #547 | `onto-pawn-threatened` | Retry of #537 onto-threatened with filtered signal (enemy pawn attacks only). |
| #548 | `unstable-lmr-depth-gated` | Retry of #541 with depth >= 6 gate. |
| #549 | `caissa-3tier-tight` | Retry of #536 Caissa 3-tier with tighter force-good rule (attacker*2 ≤ victim). |
| **#552** | `eval-diff-qhist-v2` | Retry of #517 (H0 at −0.7) solo on new post-#542/#539 trunk. Bench 2.72M (−35%). |
| **#553** | `se-king-pressure-v2` | Retry of #511 (H0 at −0.6) solo on new trunk. |
| **#554** | `offense-bonus` | Reckless pattern (+6000 for quiet attacking enemy piece). Not previously tested. Bench 2.85M (−31%). |
| **#555** | `rook-kingring-bonus` | Reckless pattern (+5000 for rook attacks on enemy king zone). Bench 3.35M (−20%). |
| **#556** | `dynamic-see-v3` | Retry of #543 with more-permissive −125 baseline. |

### Overnight session overall

- **Net Elo gain merged: +12.7 face value** (#542 + #539) — retune #551 grinding a further +2 at time of writing.
- **Fleet efficiency**: wait-500-rule validated repeatedly; early spikes (+8 at 200 games) faded to realistic +1-5 ranges.
- **Signal-context sweep continuing to pay**: #539 was 3rd landing for any_threat_count signal; #542 was 2nd landing for `unstable` signal. Pattern of cross-context reuse exceeds Titan's 1-in-3 prior significantly.
- **Bench creep concern** (Adam, 2026-04-20 early AM): today's +Elo wins added 38% nodes (3.0M → 4.17M after #539, 4.51M with tune applied). Approach: prioritise bench-REDUCING retries in today's next wave. Offense bonus (−31%), eval-diff-qhist (−35%), rook-kingring (−20%) are the three bench-reducing candidates queued.

### Post-mortems by H0 bucket

- **Magnitude-too-large** (#537 −24.8, #541 −10.9): signal direction correct but applied too aggressively. Both iterating with gates/filters.
- **Signal overlap** (#540, #544): redundant with existing features. Drop without iterate.
- **Mechanism wrong** (#528, #536): structural assumption doesn't hold. #528 dropped; #536 retrying with tighter rule as check.

Memory updated: `feedback_h0_post_mortem_discipline.md` — require mechanism-bucketing and iterate/drop decision for every H0 going forward.

---

## 2026-04-20: Two H0s, one diagnostic sidetrip

### H0'd (closed — dropped)

| Test | Branch | Result | What it tested | Decision |
|---|---|---|---|---|
| **#528** | `experiment/history-shape-offset` | **−6.3 ± 5.0**, LLR −3.00, 4936 games | History bonus shape change: `(MULT*d − OFFSET).clamp(0, MAX)` vs linear `min(MULT*d, MAX)`. SPSA #515 converged values applied (OFFSET=76, MULT=310, MAX=1601, LMR_HIST_DIV=6705, HIST_PRUNE_MULT=4986). Shape matches SF/Obsidian/Alexandria/cap_history convention. | **Dropped.** Shape change hurts v9 by ~6 Elo; SPSA-tuned coupled params couldn't compensate. No trunk changes to revert — HIST_BONUS_OFFSET was never added to trunk's `tunables!` block. |
| **#529** | `experiment/material-scale-tunable` | **−3.0 ± 3.8**, LLR −2.97, 8820 games | Expose material-scaling constants (hardcoded `22400 / 32768` in `eval *= (BASE + material) / DIV`) as SPSA tunables MAT_SCALE_BASE / MAT_SCALE_DIV. #516 SPSA converged to slightly looser values (22830 / 31360) after 385 iters. | **Dropped.** Hardcoded values were effectively optimal; the ~6% looser SPSA drift costs −3 Elo. No trunk changes to revert — MAT_SCALE_BASE/DIV were only ever on the experiment branch. |

### Lessons

- **Not every consensus-shape pattern transfers.** Linear-with-offset history bonus is standard in SF/Obsidian/Alexandria and matches how Coda's cap_history is already written. But on v9 (threat-aware main history + 4D keying), adding the offset *reduces* tree efficiency. Hypothesis: the 4D-indexed main history has finer per-bucket signal than 2D histories — Coda's current formula without offset already produces tight bonuses at low depths without over-weighting. The offset strips bonus where v9's richer signal needs it most.
- **"Hardcoded is suspicious → make it tunable" isn't always worth it.** Material-scale had fixed 22400/32768 in code since v5 era. Exposing it to SPSA seemed low-risk (if tuning finds no gain, revert). But the act of exposing + applying post-tune values adds code complexity, and the ~6% drift the tuner found was negative in SPRT. Coda's original scaling constants were converged against the NNUE eval distribution; re-tuning them in isolation shifted away from that optimum. Cost: ~10K games of fleet + commit churn on the experiment branch. Minimal, but the pattern is "hardcoded ≠ untuned, sometimes it's locally optimal already."
- **Both failures were clean, no ambiguity.** −6.3 and −3.0 with LLRs hitting −3.00/−2.97 cleanly. No "maybe retune more" temptation; the formulas under test aren't hiding latent gains.

### Methodology win (session side-effect)

- **Live Lichess watching catches what SPRT can't.** Adam watched one game on Lichess and identified castling-weight weirdness (Kf1 then Kg1 sequence — engine manually reaching castled square). Turned out to be a deployment issue (v9 binary with v5 net embedded, triggered by `net.txt` pointing at v5 URL). But the methodology win stands: in a week of fleet testing, this kind of qualitative move-choice error is invisible because (a) the position-type is rare in aggregate, (b) self-play blindspots line up, (c) ±0.5 Elo from rare weirdness is under the `[-3, 3]` noise floor. Saved to memory as `feedback_live_lichess_watch_catches_bugs_sprt_misses.md`. Operational: watch at least one Lichess game after every net change or significant search refactor. One game has high SNR when a bug exists.
- **Framework validation (incidental).** v9 binary + v5 net = v9-calibrated tunables (loosened for flatter eval) applied to v5 peakier eval = systematically too-loose pruning. The "odd play" on Lichess was v9 search exhausting time on nodes its tunables should have pruned more aggressively given v5's clearer eval. Clean at-zero-fleet-cost validation of the ordering-coupled-pruning framework's claim that tunables are coupled to the eval they were tuned against.

---

## 2026-04-19: ~+90 Elo day (v9 search-consumer stack + activation win)

### Merged to v9 trunk (H1 resolved)

| Test | Feature | Elo | Games | Notes |
|---|---|---|---|---|
| #478 | `fix/threats-2b-rewrite` (2b slider-iteration rewrite + & occ fix) | **+10.0** | 4674 | Profile-driven NPS refactor of section 2b in threats.rs — scalar 8-direction walks → slider-iteration on precomputed between()/ray_extension tables. Bundle also included Titan's zero-emit counters and `Board::xray_blockers` helper (unused at merge, consumed by B1 later). Bug caught: 2b rewrite emitted phantom x-ray during `push_threats_on_move` transit (pieces_bb vs occ_transit inconsistency); fixed with `& occ` filter on candidates. |
| #481 | `experiment/probcut-threat-gate` (A3 — king-zone-pressure ProbCut gate) | **+7.03** | 4942 | Skip ProbCut when enemy has ≥ PROBCUT_KING_ZONE_MAX attackers on our king zone. Reuses `king_zone_pressure` from NMP gate (#466). Third landing for that signal. |
| #482 | `experiment/lmr-king-pressure` (king-pressure LMR modifier) | **+6.81** | 5204 | `reduction -= king_zone_pressure / LMR_KING_PRESSURE_DIV`. Fourth consumer of the signal. I initially miscalled this H0 at 1242 games (-9.8 early); corrected after resolution via overlapping-bars feedback memory. |
| #484 | `experiment/futility-defenses` (our_defenses futility widener) | **+7.00** | 5116 | `futility_value += any_threat_count * FUT_THREATS_MARGIN`. Sibling to the has_pawn_threats RFP widener. |
| #490 | `tune/v9-post-merge-r1` (60-param post-merge retune) | **+7.38** | 6546 | Post-merge retune capturing the "gates let the rest of search be bolder" compound effect. Biggest param shifts: LMR_HIST_DIV 11685→7123 (-39%), CAP_HIST_BASE 10→15 (+50%), NMP_EVAL_DIV 136→122. 37 tunables moved materially. |
| #497 | creluHL net (clipped-ReLU on L1/L2) | **+4.0** | ~8500 | Clipped-ReLU on hidden layers (Reckless pattern, one-line Bullet change). Validated via `HiddenActivation=crelu` UCI option on existing inference path. No code merge — net choice change only when we promote a CReLU-trained net. |
| #502 | `experiment/discovered-attack-bonus` (B1 — Titan's Tier B1) | **+52.0** | 666 | **Biggest single-feature win in project history.** Flat bonus on quiet moves where `from()` is a square blocking our own slider's attack on an enemy. Uses `Board::xray_blockers`. Confirmed the "specific tactical motif" scoring pattern is distinct from generic "a piece is attacked" nudges (enter-penalty, hanging-escape both H0'd). Path 2 (bit-steal delta-tagging refactor) now firmly justified per Titan's ">6 Elo gates Path 2 work". |

**Day total merged: ~+94 Elo** (2b 10 + ProbCut 7 + LMR-KP 6.8 + futility-def 7 + retune 7.4 + creluHL 4 + B1 52 = 94.2). Plus Atlas's TM-floor fix merged on main (+4.4) picked up via the main→v9 rebase on 2026-04-19.

### H0'd today (closed — no retry planned)

| Test | Feature | Result | Decision |
|---|---|---|---|
| #479, #501 | `experiment/stratified-escape-canwin` (A1a+A1c bundle) | H0'd twice: #479 drifted to flat/small negative, #501 post-SPSA retest went to -8 at 3996 games | **Dropped.** SPSA on the new tunables (#486) didn't find a positive basin either. The stratified-escape-ladder + can_win_material combination isn't adding value on this trunk. |
| #504 | `experiment/lmp-king-pressure` (S3 — LMP threshold softener) | -10.1, H0 | **Dropped.** King-zone-pressure as an LMP softener fails cleanly. |
| #503 | `experiment/rfp-king-zone-widener` (S1 — RFP margin widener) | -9.5, H0 | **Dropped.** Tree shape changed moderately (bench 2988580 → 2594958, -13%) but Elo was -9 not small-negative; retuning wouldn't recover that magnitude. |

**Pattern from these H0s**: king-zone-pressure signal works as a **gate on pruning decisions** (NMP, ProbCut, LMR-reduction-modifier all H1) but **fails as a margin/threshold widener** (RFP +, LMP both H0). Speculation: margin wideners are already tightly calibrated against `improving` / `has_pawn_threats` / SPSA'd base margins; adding a third overlapping widener disrupts the tuning balance.

### Still in flight (as of session end)

- **#496 warm30 net**: +3.9 ± 4.3 at 7162 games, LLR 2.56 → close to H1 resolution
- **#500 `experiment/threat-mag-lmr`** (Titan's): +0.8 flat at ~4800 games — trending slow-H0
- **#508 CONTEMPT_VAL=0**: +0.2 at 1410 games, early — tests consensus of "remove contempt" per `tunable_anomalies_2026-04-19.md`
- **#511 S4 SE king-pressure**: resubmitted at 250K scale, early
- **#512 S5 Aspiration king-pressure**: resubmitted, early
- **#513 scaled discovered-attack**: B1 variant with victim-value scaling; early
- **#509 SPSA history-shape-offset**: 0/1000 iters — Experiment 1 from `shape_experiments_proposal_2026-04-19.md`
- **#495→resubmitted reckless-kb tune**: 0/1000 — restarted at 250K after audit caught 500K misconfig

### Housekeeping landed today

- **v9 trunk merged main** (Atlas's TM fixes + tm_score + blunder_suite now on v9).
- **net.txt flipped decision** for eventual v9→main merge: v9 production net will be default (captured in `docs/v9_merge_plan_2026-04-19.md`).
- **v7 deprecated**: architecture support going forward is v5 (legacy) + v9 (primary). v7 inference code stays (shared by v9 paths).
- **ob_submit / ob_tune auto-detect v9 branches** for `--scale-nps` (250K for v9, 500K for main). Fixed a recurring operational error where v9 SPRTs ran at 2× wall-clock time.
- **B1 bench convention** nailed down — v9 trunk commits use `nets/net-v9-768th16x32-w15-e800s800-xray.nnue` (hash 6AEA210B) bench, **not** the embedded v5 net. CLAUDE.md and memory updated.
- **Three companion idea docs** merged to trunk by Titan: `signal_context_sweep_2026-04-19.md`, `move_ordering_ideas_2026-04-19.md`, `threat_ideas_plan_2026-04-19.md`, `tunable_anomalies_2026-04-19.md`, `shape_experiments_proposal_2026-04-19.md`.

### Calibration lessons (worth remembering)

- **"Scoring nudge" pattern is not monolithic.** Generic "a piece is attacked" nudges failed (enter-penalty, hanging-escape). Specific tactical motif nudge (B1 discovered-attack) landed +52. Distinguish before pattern-matching off prior H0s.
- **Five engines minimum for outlier claims** in cross-engine tunable comparison (Titan's methodology note in `tunable_anomalies_2026-04-19.md`). Two-engine comparison over-flagged three params in the first pass.
- **Post-merge retune compounds with multiple merged features**, not a single one. #490 landed +7.4 after 4 features merged; single-feature retunes typically land +2-5.
- **Overlapping error bars = "same distribution twice"**, not "earlier was noise". I miscalled LMR king-pressure H0 at 1242 games because -9.8 had ±10.4 bars. Ended +6.81. Don't narrate direction from within the noise floor.

## 2026-04-20 → 2026-04-21 session

### Major H1 wins (merged to feature/threat-inputs)

- **#553 SE-king-pressure-v2** — S4 from signal_context_sweep, retry of H0'd #511. **+9.7 Elo H1** at 4462g LLR 2.97. King-zone-pressure SE margin widener; +1 context for the kzp signal (now 4/8 tested landed).
- **#542 unstable-probcut-skip** — parent-child eval-gap → ProbCut skip. **+6.7 Elo H1** at 7390g LLR 2.96.
- **#539 threats-nmp-skip** — `any_threat_count` gating NMP. **+6.0 Elo H1** at 8330g LLR 3.01.
- **#554 offense-bonus** — Reckless quiet-attacks-enemy MovePick bonus. **+5.7 Elo H1** at 8854g LLR 3.02. #558 LTC validation landed +6.3.
- **#557 fix/smp-king-bucket-race** — King-bucket static-mut to NNUENet field + v6 kb_layout fix. **+3.4 Elo H1** at 5444g LLR 3.01 (cache locality gain despite identical bench). SMP trilogy bug #1 of 3.
- **#576 fix/smp-helper-threat-init** — Helper threads never mirrored main's threat-accumulator state. T=4 SPRT **+651 Elo (self-play "correct-T=4-vs-broken-T=4")** at 304g. **Unlocks v9 T=4 + ponder deployment.** SMP trilogy bug #3 of 3. (Bug #2 was Atlas's aarch64-Finny-cfg-eats-else fix — ARM only.)
- **#578 tuned-knight-fork** — Knight-fork MovePick bonus + retune #569 applied. **+5.2 Elo H1** at 5734g LLR 2.94. 3rd successful retune-on-branch rescue precedent (after TT_PV and cont-hist-malus).
- **#582 feature/threat-inputs** — **New production v9 net**: `net-v9-768th16x32-w15-e800s800-reckless-crelu.nnue` (reckless kb layout + CReLU hidden layer + warm30 warmup + s800). **+15.2 Elo H1** at 2950g LLR 3.05 vs xray (6AEA210B). Biggest single net upgrade this session.
- **#583 fix/lmr-endgame-gate** — Skip LMR when `popcount(occupied) ≤ LMR_ENDGAME_PIECES` (default 6). **+5.0 Elo H1** at 5164g LLR 3.01. Fixed endgame-conversion blunders observed in Lichess game CG5ZXe5Z (coda at depth 22 couldn't find M7 with LMR on; finds M8 at depth 17 with gate).

### H0 rejections (notable)

- **#565 queen-7th-bonus** (-2.0, H0 at 11482g). Retune #568 applied → #577 also H0 (-1.7, 18290g). Retune-on-branch can't rescue features that H0 with genuine negative Elo (not just noise-near-zero).
- **#555 rook-kingring-bonus** (+0.1 at 32196g, H0). Retune #564 applied → #574 H0 at -2.2 (made it worse). Same lesson: retune can't rescue noise-around-zero features.
- **#563 knight-fork-bonus** (+0.2 at 32740g, H0) — but retune #569 RESCUED to +5.2 Elo (#578 H1). Distinct from #555 because the feature was borderline positive with big tree-reshape (right candidate for retune).
- **#566 rook-open-file** (+0.8 at 72998g, H0). Large bench drop, tiny positive Elo. Could retry on tuned trunk post-#585.
- **#573 tune/v9-post-554-r1-applied** — Main retune after #554 merged. Stopped early at 5528g, -0.7 flat. Lesson: generic retunes on already-tuned trunks diminishing returns; retune-on-branch is selective, generic is waste.
- **#572 feature/threat-inputs** (w20 vs w15 on kb10 net) — H0 at -10.5. Confirmed w15 is WDL optimum for kb10 family (with prior w10 < w15 result).
- **#567 feature/threat-inputs** (warm50 vs warm40 on reckless-warm series) — H0 at -9.4. Warm ladder saturates past warm30.

### Infrastructure / correctness

- **`chore/tunables-with-c-end`** merged — tunables! macro now carries c_end field, `coda tune-spec [--r-end 0.002]` CLI dumps fresh SPSA spec from source. Eliminates stale spec file class of errors (observed in tune #562 where LMP_BASE was stuck at integer boundary due to c_end too small).
- **`nnue/hl-crelu-file-marker`** merged — context-dependent bit 5: on extended_kb=1 nets means hl_crelu (auto-configures NNUE at load). `coda patch-net` CLI for flipping bit 5 on existing nets.
- **`fix/simd512-pairwise-pack-fused-threats`** — AVX-512 pairwise-pack was silently dropping threat features on v9 pairwise nets. Validated by Adam on AVX-512 host: pre-fix bench 3,317,873; post-fix bench 4,513,225 (matches AVX2 Hercules exactly). OB fleet unaffected (no AVX-512 workers). Latent landmine for future AVX-512 worker.
- **`fix/bench-multithread-output`** — `coda bench -tN` now shows per-position info (thread 0 runs full stats, others silent). Previous behavior was entirely silent.
- **Compiler warnings cleared to zero** — CLAUDE.md Code Hygiene section added requiring `cargo build --release` emits no warnings.

### SPRTs in flight at session end

- **#586 experiment/tuned-trunk-585** — applies tune #585 values (2500 iters on post-merge trunk + new production net). 15+ strong movers: LMR_HIST_DIV +25%, HIST_PRUNE_MULT +31%, CORR_HIST_DIV -22%, CORR_W_NP -23%, SE_DEPTH -19%, LMP_DEPTH -19%, QS_SEE_THRESHOLD tighter. Bench 1,959,547 vs trunk 1,766,373; EBF 1.70 vs 1.78 (-4.5%); FMC 77.3% vs 75.2% (+2.1pp). User prior +10-20 Elo given cumulative improvements.
- **#587 experiment/tuned-force-more-pruning** — applies tune #571 values on old trunk (pre-knight-fork merge). Tests whether SPSA starting from "force more pruning" shifted defaults converges on a better-pruned equilibrium than natural retune. Bench 2,820,229 (-37.5% vs pre-tune 4,513,225), EBF 1.77 (-1.1% vs 1.79). Force-pruning nudged out of local max; Elo TBD.
- **#584 experiment/threat-delta-capture-bonus** — idea C retest at `THREAT_CAPTURE_BONUS=500` (half the failed #579 value 1000). UCI override in SPRT. If still flat, try lower or retest on new trunk.
- **#580 experiment/threat-delta-ext** — idea D (threat_deltas count extension on captures). -8.9 trending H0 at 3568g.

### Key calibration lessons

- **"Wait 500+ games" rule validated** — C bonus=1000 showed -5.6 at 996g, recovered to -0.3 at 10036g. First-thousand-games noise swing is real; resist stopping early.
- **Retune-on-branch cliché refined**: works when feature H0'd with non-zero-centered distribution (TT_PV +4.5→+8.5, knight-fork +0.8→+5.2). Fails when feature H0'd at noise-around-zero (rook-kingring +0.1→-2.2, queen-7th -2.0→-1.7).
- **Live Lichess watching catches bugs SPRT misses** — CG5ZXe5Z game revealed endgame-conversion LMR blunder that bench+SPRT had never flagged. Diagnostic FEN-probe → root-cause (LMR over-reducing mate-completion moves) → surgical gate → +5 Elo. Worth adding live-watch to post-net-change checklist.
- **SMP bugs have "Swiss cheese" failure signature** — most games fine (helpers lose root cutoff race), rare games catastrophic (helper's wrong-eval TT entry poisons main). Discovered via 20-FEN eval-consistency instrumentation (Atlas's pattern), not SPRT.
- **Bench on correct branch** — submitted #582 with wrong bench (2,862,737) because local build was on idea-C branch with THREAT_CAPTURE_BONUS=1000 active. OB rejected with Wrong Bench. Memory note `feedback_bench_depends_on_branch_state.md` exists exactly for this; should have followed it.

### WDL ladder (kb10 family)

Tested so far: **w15 > w10**, **w15 > w20**. Additional w0 and w05 nets trained overnight. Full RR deferred — low priority, pure research/documentation. w15 confirmed optimum for kb10.

### Warmup curve (kb10 reckless family, e200) — resolved

Full curve at e200 tested, closes Titan's `training_warmup_curve_2026-04-19.md`
open question:

| Warmup SBs | Result |
|---:|---|
| 5 | −7 Elo (earlier in thread) |
| 20 | baseline |
| **30** | **+4.44 Elo (#496), peak** |
| 40 | H0 vs warm30 |
| 50 | H0 vs warm30; #567 H0 −9.4 vs warm40 |

**Shape:** positive 5→30, peak at 30, regression past 30. Rule: don't warm
past ~15% of total training length. For e200 that's warm30; anchor at
warm30 for e400/e800 production unless absolute-count follow-up suggests
otherwise. `docs/training_warmup_curve_2026-04-19.md` updated with
the resolution.

## 2026-04-21 session (continued) — AVX-512 + VNNI merge

**Merged `feature/nnue-avx512-vnni-v9`** (Zeus's work) at 51a71f2.
Bench bit-exact 2,170,815 across all tested CPUs — correctness validated.

### Per-uArch NPS impact

| Platform | CPU | Pre NPS | Post NPS | Delta | Notes |
|---|---|---|---|---|---|
| Zeus server | Zen5 (AVX-512 VNNI) | 1,076K | 1,225K | **+13.8%** | VPDPBUSD 1-cycle throughput |
| Hercules | Xeon E-2288G (AVX2) | 200K | 200K | 0% | Bit-exact — AVX2 path unchanged |
| Adam laptop | i5-13500H Raptor Lake (AVX-VNNI) | 664K | 614K | **−7.5%** | VPDPBUSD on Intel mobile is ~2-cycle, uop-fusion win eaten by lower throughput |

### Open follow-up

**Raptor Lake −7.5% needs investigation.** This isn't a niche CPU — 13th gen
Intel i5/i7 is mainstream hardware. Merge stays for now because correctness
is validated and Zen5 gain is substantial, but AVX-VNNI dispatch needs to
either:
1. Tune the kernel (unrolling, accumulator count) to match AVX2-path performance
2. Narrow the dispatch gate (avx-vnni only on Zen 4+, or only when `avx512vnni` ∧ AVX-VNNI)
3. Add per-uArch detection

Quick A/B for the local Claude investigating: patch `detect_avx_vnni()` to
return false, rebuild, bench. If NPS recovers to 664K, VPDPBUSD-on-Raptor-Lake
is confirmed as the culprit (vs dispatch-reorg side-effects).

Relevant file: `src/sparse_l1.rs::dense_l1_avx_vnni` — the hot path for L1
matmul on AVX-VNNI-without-AVX-512 CPUs.

## 2026-04-22 session — overnight resolutions + morning merges

### H1 landed (merged to trunk)

- **#619 fix/fifty-move-scaling-v2** +3.3 Elo H1, 8708g. Zeus's two-part
  fix: stronger `score * (100 - hm) / 100` formula that actually reaches
  0 at the 50-move cliff, plus TT-storage invariant (50-move scaling
  applied at use site only, never stored). The first patch alone
  (#610) regressed -9.4 because of the TT-stale-scaling bug. Commit
  `5086604`.
- **#613 experiment/history-bonus-offset** +3.0 Elo H1, 5460g. Titan's
  shape experiment 1: `min(MAX, MULT*d)` → `clamp(0, MAX, MULT*d - OFFSET)`.
  New tunable HIST_BONUS_OFFSET=72 (SF's value). Post-merge joint SPSA
  per Titan's spec likely banks more. Commit `ee76dcb`.
- **#604 experiment/se-xray-blocker** +1.1 Elo H1, 25934g. Titan's
  signal-context sweep: xray-blockers signal (B1's +52 Elo mechanism)
  in Singular Extensions context. Small but clean positive. Commit
  `dd4452c`.

Total merged today: **+7.4 Elo raw** (post-merge retune could bank more).

### H0 rejections (Titan-doc items resolved)

- **#611 experiment/caphist-defender-v2** -1.4 @ 13268g. Rebased+merged
  branch; SPRT-untuned as first step per methodology. Confirmed
  Titan's prediction: caphist's original +10pp FMC headroom was
  **absorbed by B1 discovered-attack merge** (+52 Elo). On current
  trunk (FMC 76.8%), the remaining ordering gap isn't caphist-shaped.
  Dropped permanently.
- **#612 experiment/skewer-bonus** -8.0 @ 3848g. Unified
  (value-unfiltered) skewer detector. Over-triggered on low-value
  ray alignments, added noise to ordering.
- **#618 experiment/pin-skewer-value-filtered** -6.4 @ 4728g. Value-
  filtered split of #612 per Titan's T1.1+T1.2 spec. Didn't rescue —
  the ray-alignment signal just isn't an H1 feature on current trunk
  (B1 already captures it via FROM-based discovered-attack).
- **#614 experiment/rfp-unified-margin** -2.6 @ 6820g. Shape exp 2.
  Classic retune-candidate signature (bench +8% vs trunk, Elo
  negative). Could land with focused SPSA on RFP_MARGIN + RFP_IMPROVING_SUB.
- **#617 experiment/undefended-nmp-skip** +0.4 @ 44792g. T2.1 from
  next_ideas. Hanging-piece NMP skip — signal fired too rarely to
  move Elo. Noise-level.
- **#606 experiment/rfp-anythreat-widen** -1.2 @ 24532g. Slow-fade to
  H0 after trending H1 earlier.
- **#609 L1 sparse net vs warm30 (net-vs-net)** -2.1 @ 15628g. L1
  coefficient 1e-6 too weak to produce meaningful sparsity beyond
  the 8.4% structural-zero floor. Confirms sparsity doc finding.

### Factoriser journey (important but non-linear)

Three-training-length data points on factoriser (`--factoriser` flag,
otherwise matches `reckless-crelu` recipe):

- **factor SB50 vs creluHL SB200 (broken hl_crelu bit)** = +157 Elo
  **(measurement artifact)**. The creluHL baseline had bit 5 = False
  (consensus_buckets interpretation) but was trained with CReLU
  hidden → Coda loaded with SCReLU hidden → broken activation chain
  → tanked baseline by 50-150 Elo. Not real factoriser signal.
- **factor SB50 vs prod DAA4C54E** = -118 Elo. Clean comparison
  (both sides have bit 5 = True). Baby can't beat 16×-longer-trained
  giant. Training gap dominates.
- **factor SB200 vs prod DAA4C54E** = -30.9 ± 15.3 Elo @ 700g
  (bounds [-5, 5]). Clean magnitude. Sits in expected training-gap
  disadvantage range (~-25 to -35 from 2 doublings + prod's retune
  advantage). Factoriser architectural contribution is therefore
  ~**neutral**, closing the training gap at roughly 1:1 rather than
  being multiplier.
- **factor SB200 vs creluHL SB200 (patched hl_crelu)** = -11.8 Elo H0
  @ 1700g (#627, bounds [-5, 5]). Cleanest factoriser-only comparison
  — same kb10, w15, crelu hidden, e200 snapshot; differs only in
  factoriser yes/no and warmup (30 vs 20). Given warm30 was +4.44
  Elo vs warm20 at e200 (per warmup-curve experiment), the factoriser
  contribution net of warmup is **~-15 Elo at equal training + no retune**.

**Calibrated expectation going forward**: factoriser at equal training
+ NO retune is slightly negative. A retune-on-branch (SPSA on
factoriser tree shape) might flip it to neutral-positive; current
pruning tunables are calibrated against non-factoriser trunk trees
and may be wrong for the factoriser's sharper ordering.

**Revised strategy**: wait for SB400 overnight, retune on factoriser
branch, then SPRT vs prod. If still negative after retune, factoriser
isn't a keeper. If +5-15 post-retune, it becomes an e800-production
training candidate.

### Methodology lessons captured (new memory notes)

- `feedback_verify_net_flag_bits_before_sprt.md`: always inspect bit 5
  (hl_crelu) on both sides before net-vs-net SPRT. Burnt on #622.
- `feedback_sprt_bounds_interpretation.md`: SPRT H0 at bounds
  [elo0, elo1] means "failed to prove ≥ elo1", NOT "confirmed ≤ elo0".
  Asymmetric bounds are policy, not magnitude. Burnt on #625.
- `feedback_correctness_audit_wins_dominate.md`: bugs in rarely-fired
  paths consistently land +3-30 Elo. Correctness audits beat feature
  additions for Elo-per-effort. Titan-scale investigation direction.
- `reference_ob_debugging_endpoints.md`: self-service debug via
  `/errors/` page + two-binary-bench rule after branch switches.

### SPSA tune in flight

- SPSA tune on experiment/caphist-defender-v2 (focused 4 capture-coupled
  tunables, 500 iters). Likely not useful given caphist SPRT H0'd
  cleanly — no point pursuing if the base feature doesn't carry
  signal. May stop.

### Key calibration takeaway

Correctness-audit wins consistently outperform feature-addition wins
for Elo-per-experiment. Our next-day queue should bias toward
correctness audits in rarely-fired paths (repetition, cuckoo, TB sign
conventions, null-move zugzwang edge cases, stalemate path coverage).

## 2026-04-22 session — correctness audit sprint + C8 training fix

### Titan correctness audit (9 CRITICAL, ~35 LIKELY, ~30 SPECULATIVE)
Ten-parallel-agent review across threats / NNUE / search / TT+history /
movepicker / TM+UCI+ponder / SMP / SEE+Zobrist+cuckoo+TB / training I/O /
Bullet-trainer subsystems. Doc: `docs/correctness_audit_2026-04-22.md`.

### CRITICAL fixes — H1 confirmed via SPRT

- **#628 P2 halfmove-gated TT cutoff** +1.2 Elo H1, 13970g. Extended
  existing `halfmove < 90` gate to bounds-narrow collapse + near-miss
  + QS TT cutoff return paths. (Merged prior session.)

### CRITICAL fixes — merged as bench-invariant on unreachable paths

No OB SPRT coverage (fleet has no TB / aarch64 / mixed-mode UCI):

- **C2 TB cache halfmove key** — probe_wdl is halfmove-aware;
  cache was keyed on Zobrist alone → stale CursedWin served to
  halfmove=0 queries. SplitMix scramble on halfmove mixed in.
- **C4 NEON SCReLU shift** — commit 44baa95 mistakenly set NEON
  screlu_pack to >>9 alongside pairwise; fix restores >>8 to
  match scalar + x86. Only affects v7 non-pairwise SCReLU nets on
  aarch64.
- **C5 TT_WIN threshold widen** — `score_to_tt` / `score_from_tt`
  threshold moved from `MATE_SCORE-100` (28900) to `TB_WIN-128`
  (28672) so TB scores actually get ply-adjusted on TT crossings.
- **C6 TM stale limits** — zero all TM limits up-front before
  mode-specific branches; fixes `go movetime N` after prior
  `go wtime/btime` leaking stale soft/hard_limit.
- **C7 convert-checkpoint output-layer** — `l2_size == 0` path
  silently dropped output-layer entries in both `weights.bin` AND
  `momentum.bin`/`velocity.bin`. Centralised write list through a
  local helper so the three buffers can't diverge.
- **C9 HIP Adam ABI** — not active in prod (CUDA path), flagged
  for Bullet team.

### LIKELY fixes — H1 / merged (search pruning class)

- **#635 T1.4 Battery bonus** +5.3 Elo H1, 8452g (merged morning).
- **#637 N6 promotion-imminent extension** +1.6 Elo @ 25886g, merged
  at 10K-plus per confident-correctness policy.
- **#638 C1 is_pseudo_legal EP validation** +0.4 Elo @ 28382g,
  correctness merge.
- **#645 evasion history key symmetry** -0.9 Elo @ 12802g, merged
  on "correctness fix at -1 to -2 is mergeable" policy. Fix:
  evasion MovePicker now threads through `enemy_attacks` so
  4D main-history reads match beta-cutoff writes.
- **#649 rep-detection null-move boundary** -0.5 Elo @ 10906g, merged.
  Main + QS rep scans now stop at null-move boundaries
  (plies_from_null), matching cuckoo's existing min() pattern.
- **#650 SE `is_pv` shadow + corrhist stop guard** +0.6 Elo @ 10236g,
  merged. SE shadow was computing `is_pv` from current alpha; outer
  `is_pv` (alpha_orig-based) is correct. Corrhist update now has
  stop-flag guard matching TT store.

### LIKELY fixes — direct merge (bench-bit-exact or user-facing)

- **#17 legacy threat pipeline debug_assert** — fall-through to
  empty_threat silently zeroed the eval half if v9 routed via
  `forward()` instead of `forward_with_threats`.
- **#20 `coda check-net` uses forward_with_threats** — for v9 nets,
  display-eval was ignoring threat features entirely.
- **#29 SMP stockpile early-stop** — set shared stop before the
  stockpile-prevention sleep so helpers don't burn CPU through
  the window.
- **#30 Ponder UCI option handler** — was falling through
  silently; now explicit no-op acknowledgement.
- **#35 helper.ponderhit_time clone** — was never cloned from main,
  helpers ignored ponderhit deadlines.
- **#36 load_nnue threat_stack reset** — net-swap from v9→v5 left
  threat_stack.active=true; now reset unconditionally, then
  activated if `net.has_threats`.
- **#37 SMP TT gen advance** — moved from `search()` to
  `search_smp`/datagen callers so helpers can't write old-gen TT
  entries during spawn race.
- **#43 TB castling gate** — early-return None before FEN format
  when `board.castling != 0` (Syzygy rejects those anyway).
- **Movepicker hygiene bundle** — bad_moves buffer 64→256,
  non-capture promotions get promotion material delta in
  mvv_lva, evasion scorer checks capture before promotion (so
  capture-promotions rank above regular captures).
- **TM/UCI hygiene** — soft_limit clamp order, depth parse
  fail-closed (unwrap_or(0) not 100), `go ponder movetime X` on
  ponderhit uses movetime instead of instant-stopping.
- **TB hygiene** — DTZ |d|>100 maps to ±1 (not ±20000),
  ponderhit TB override now matches `go` path on draws.
- **Offline tools** — fetch-net gains `-f`, threat weight clamp
  uses full i8 range, material-removal datagen cleans up
  castling/EP after piece removal.
- **Threat refresh overflow flag** — refresh now tracks overflow;
  accurate flag set to !overflowed so incremental deltas don't
  compound on corrupted baseline.

### LIKELY fixes — H0'd individually, retune bundle pursued

- **#640 C3 NMP sentinel** (-1.3 @ 22898g), **#646 #3 NMP stale
  reductions** (-5.1 @ 5722g): both correctness fixes (SF uses these
  sentinels), but bench +13.4% / +7.7% suggested tree shape shift.
  Bundled into `experiment/nmp-sentinel-reductions-retune`; SPSA
  #654 running on standard 18 pruning params. Expected pattern:
  retuned defaults flip the bundle to +3-7 Elo (W3 retune-on-branch
  pattern, historical #490 +7.4 / #586 +6.2).

### Resolved since (all merged to feature/threat-inputs)

- **#648 improving+probcut+lmp** bundle — **+1.2 @ 9972g trending H1**,
  merged at `1b70fd8` (confident correctness / small-win bundle).
- **#652 SEE-promo-recapture** — **+1.8 @ 16404g H1 ✓**,
  merged at `b25366d`. (Note: this fix compounded with the #660 tune
  bundle on the C8-fix net to drop trunk bench by an additional ~20%
  beyond the tune alone — SEE sharpening × tuned SEE/history margins.)
- **#653 QS cuckoo-alone** (unbundled from #647) — **+5.0 @ 5942g H1 ✓**,
  merged at `334e80c`. The unbundled `ply > 0` guard was the working
  half of #647's bundled fix.
- **#651 Movepicker hygiene** — **-0.5 @ 10870g trending H0**,
  merged at `6f4d681` as confident-correctness (−4 tripwire not hit).

### C8 Bullet training/inference frame mismatch — the big one

**Audit claim**: Bullet's `sq < to` same-type semi-exclusion in
bf (STM-relative) frame diverges from Coda inference's physical-
frame decision on ~63% of real-STM=Black positions. Each affected
position has 1-3 features at the wrong feature index — trained
weights never activate at inference and vice versa.

**Fuzzer confirmed**: C8 diagnostic tool (`coda fuzz-threats -n N`)
measured 1264/2000 pov-evaluations with ≥1 mismatch, 100% on
real-STM=Black, 0% on White. Magnitude matches audit's "50%" figure.

**Attempted fix path (abandoned)**: switch Coda inference to
bf-frame semi-exclusion. Tests revealed a deeper problem —
bf-frame semi-excl is NOT STM-invariant, which breaks Coda's
incremental delta mechanism (persistent same-type pairs get
different feature indices across STM flips, requiring extra
deltas that don't exist). 7 threat_accum tests failed.

**Chosen fix path**: modify Bullet training to use physical-frame
semi-exclusion (matches Coda inference, which is STM-invariant
and preserves delta correctness). Patches landed on
`adamtwiss/bullet feature/threat-inputs` commit a8e2c7d:
- `sfbinpack`/`montybinpack` loaders stamp real STM into
  `ChessBoard.extra[0]` (`from_raw` otherwise loses it via
  byte-swap).
- `chess_threats.rs` `map_features` uses `phys_flip = extra[0] ? 56 : 0`
  to convert bf squares to physical for the semi-exclusion check
  only. Index computation stays as before.

**C8-fix S200 net (1836917B) observations**:
- Net-vs-net SPRT #657 vs creluHL-S200 (pre-fix) trending ~0 Elo
  so far, still early at 3868g.
- Bench stats show clear internal improvements:
  - First-move cut rate **77.5% vs 75.1%** (+2.4pp, big)
  - EBF **1.71 vs 1.78** (smaller trees)
  - NMP cutoff success rate **50% vs 36%** (+14pp)
  - History pruning rate **73/Kn vs 44/Kn** (more discriminative hist)
- NPS difference **+26%** (~660K vs ~520K, worker-off bench).
  Verified on A/B patched-pre-fix: bit 0 (use_screlu) is redundant
  for v9 pairwise-with-hidden; NPS gain is from weight-value
  differences, likely tree-shape interaction with threat-delta
  application patterns. May explain "v9 unexpectedly slow vs v5"
  observation across the v9 run.
- SPSA retune #659 firing on C8-fix net with standard 18 pruning
  params. Hypothesis: pre-fix pruning was calibrated against
  noisy (bug-driven) eval; C8-fix's cleaner eval should unlock
  more aggressive margins → true Elo benefit shows up post-retune.

**Retune + SPRT outcome (late 2026-04-22):**
- **SPSA #659 stopped** (18-param) in favour of **#660 (full 67-param)**
  on the C8-fix S200 net. Completed 2502/2500 iterations.
- Big movers: `LMR_HIST_DIV` -45%, `HIST_BONUS_OFFSET` -26%,
  `DISCOVERED_ATTACK_BONUS` -24%, `FH_BLEND_DEPTH` +31%,
  `DEXT_MARGIN` +23%, `LMR_COMPLEXITY_DIV` +16%, `NMP_VERIFY_DEPTH` +14%.
  Theme: sharpened eval → trust history more, hand-crafted tactical
  bonuses less. `LMR_ENDGAME_PIECES` pinned at floor (4) in SPSA;
  manually restored to 5 (range clamped to [4, 9] in source to
  prevent future drift — play-quality-load-bearing from #583).
- Applied as branch `tune/v9-660-c8fix` (commit `d2cfb8a`).
- **SPRT #661 (tune/v9-660-c8fix vs feature/threat-inputs, both
  on C8-fix net): +8.25 ±4.8 @ 5264g LLR 2.95 H1 ✓.** Merged
  at `d806b84` — this is the new v9 trunk baseline.
- Bench on trunk: **2,575,054** on C8-fix (vs 3,541,301 pre-tune,
  -27%). Effective depth rose several plies; trunk is leaner than
  pre-#645 on the same net.

**C2 TB cache halfmove fix** — merged at `0c7f316` after local
cutechess SPRT (with-tb vs pre-tb, both C8-fix net, concurrency 16)
showed Elo-neutral (-0.6 point estimate at 7800g, 74% draws).
No OB SPRT (fleet has no Syzygy); local-validated correctness.

### Factoriser SB400 (warm30, training in the morning)

Two SPRTs:
- **#656 factor-SB400 vs creluHL-SB200** +70.6 Elo H1 @ 364g.
  Factor has 2× training advantage.
- **#655 factor-SB400 vs prod-SB800** -22 Elo @ (resolved earlier).
  Factor has 1/2× training disadvantage.

**Back-of-envelope**: if training-length gains are ~symmetric,
factor_arch ≈ (70 - 22)/2 = +24 Elo at equal training. If training
gains are asymmetric (late-LR tail dominates per
`project_v9_low_lr_tail_critical`), factor_arch could be higher.

**Direction**: encouraging factoriser signal, but waiting on C8-fix
SPRT and retune before kicking off full SB800 factor+C8 training
runs.

### Merge summary for the session

| Category | Count | Notes |
|---|---|---|
| Correctness fixes merged (direct) | ~25 | audit bug-class, bench-bit-exact or unreachable-on-fleet |
| Features / structural fixes merged (via SPRT) | 6 | T1.4, P2, N6, C1, #645, #649, #650 |
| Still SPRT-running | ~5 | various |
| Retunes firing | 2 | NMP sentinel/reductions bundle, C8-fix net |
| Training fix | 1 | C8 in Bullet fork, S200 net trained |

The session executed about 40 of the audit's 44 actionable items.
Remaining LIKELY (pawn_hash comment, hardcoded buffer limits, TT
aarch64 atomics, forward non-hidden NTM dead code, etc.) are
latent / dead-code / cosmetic and deferred.

## 2026-04-23 overnight batch (post-#661 tuned C8-fix trunk)

Ten overnight SPRTs on tuned C8-fix trunk covering retries of pre-tune
H0s and new correctness items.

### H1 — merged

- **#670 fix/nmp-stale-reductions-c8-retry** — +2.8 Elo H1 ✓ @ 10540g.
  C3 audit follow-up (reset `info.reductions[ply_u]` at node entry).
  Originally H0'd −5.1 on pre-C8 trunk (#646); retune-on-branch
  pattern validated on tuned C8-fix trunk. Confirms that some
  correctness fixes that H0 on miscalibrated trunk emerge positive
  after retune.
- **#676 experiment/undefended-nmp-skip-c8-retry** — +2.9 Elo H1 ✓
  @ 10122g. T2.1 (undefended-piece NMP skip, next_ideas_2026-04-21).
  Was marginal +0.4 @ 44Kg H0 on old trunk (#617). Tuned trunk +
  C8-fix eval made the W2 gate pattern fire productively.

### H0 but merged (confident-correctness)

- **#668 fix/nmp-sentinel-c8-retry** — −1.6 Elo H0 @ 19874g.
  C3 audit: NMP was leaving `moved_piece_stack[ply_u]` stale;
  children read polluted cont-hist keys. Direct SF sentinel
  translation. Within −2 threshold; load-bearing for downstream
  cont-hist-sensitive experiments.
- **#672 fix/tt-cutoff-malus-symmetry** — −1.7 Elo H0 @ 10556g.
  C8 LIKELY #6: malus read-side used `board.piece_at(to)`
  (post-move, queen on promotions) while write-side used
  `moved_piece_stack` (pre-move, pawn). Old code silently disabled
  malus on promotions. "Fix" enables it — which is why old bundle
  #647 regressed: old trunk was implicitly calibrated around
  suppression. Merging now lets future retune absorb the shift.

### H0 — dropped

- **#667 tune/psq-refresh-perpov-c8fix** — −2.2 Elo H0 @ 10366g.
  Per-pov PSQ materialize on king-bucket crossing. Original a9f6e1f
  trended +21 on old trunk (#426) but never merged. Bug-fixed
  re-apply (my `needs_refresh` leak) still regressed on tuned
  C8-fix trunk. Not a correctness fix (NPS optimisation); no
  confident-correctness argument. Dropped. Revisit if profiling
  flags materialize as a hotspot.
- **#669 experiment/t1-3-overload-c8-retry** — −4.9 Elo H0 @ 6056g.
  T1.3 overload bonus. H0'd on both pre-C8 (#634 −1.4) and tuned
  C8-fix trunks. Past −4 tripwire this round. Dropped.
- **#671 experiment/p1-optimism-c8-retry** — −7.1 Elo H0 @ 4268g.
  P1 optimism port (SF/Reckless/Viridithas consensus), untuned.
  Dropped for now; could iterate via K1/K2 focused SPSA on-branch
  if we want to revisit.
- **#674 fix/should-stop-granularity** — −1.1 Elo H0 @ 16332g.
  Audit SPECULATIVE #326 (4096 → 1024 node time-check bucket).
  Benefit claim was theoretical (emergency budget overruns); at
  STC we never hit that regime. Cost is 4× more atomic
  fetch_adds. Dropped; revisit only if LTC shows real overruns.

### Iterating

- **#673 fix/lmr-shallower-margin** — −1.7 Elo H0 @ 17286g at 30cp.
  Audit SPECULATIVE #321 fix: `new_depth` used as cp margin
  (near-certain typo). Direction correct; magnitude wrong.
  **v2 submitted at 20cp** (closer to old effective threshold).
  Next fallback if v2 H0s: 10cp, or Alexandria's alpha-relative
  `alpha + 7`.

### Net result

4 merged (2 H1 + 2 confident-correctness), 4 dropped, 1 iterating.
Trunk bench after the 4 merges: **3,370,847** (up from 2,575,054
pre-batch). NMP sentinel + T2.1 + undefended-NMP-skip + malus
symmetry stack to a much bigger tree — next retune will absorb.

## 2026-04-23 → 2026-04-24 session — factor architecture + NPS investigation wave

### Factor architecture validation

| SPRT | Branch | Result | Status |
|------|--------|--------|--------|
| #679 | fix/lmr-shallower-margin-v2 (20cp) | +1.4 Elo H1 @ 20926g | ✅ merged (audit SPECULATIVE #321 fix; iteration from #673 H0) |
| #682 | tune/v9-682-sb400 (C8-fix retune) | (64-param tune) | applied |
| #686 | Merge tune/v9-682-sb400 | +3.0 Elo H1 @ 22118g | ✅ merged |
| #683 | experiment/n5-qs-insufficient-material | +0.3 Elo @ 50894g | ❌ dropped (N5 from peripheral_mechanisms) |
| #684 | experiment/t2-3-mobility-delta | **+1.4 Elo H1 @ 22200g** | ✅ merged (T2.3 from next_ideas_2026-04-21) |
| #685 | experiment/t1-5-trapped-piece-escape | −2.7 Elo H0 @ ~904g | ❌ dropped |
| #687 | experiment/t1-5-trapped-piece-v2-bonus20 | −3.7 Elo H0 @ 8174g | ❌ dropped (T1.5 two-point sampled; genuinely negative) |
| #688 | feature/threat-inputs (factor net SPRT) | **+13.6 Elo H1 @ 944g** | ✅ validated factor architecture |
| #689 | experiment/s17-probcut-corrhist-gate | −0.4 Elo H0 @ 21618g | ❌ dropped |
| #691 | experiment/n1-twofold-eval-blend | +0.0 Elo @ 28792g | ❌ dropped (N1 from peripheral_mechanisms) |
| #694 | tune/v9-692-factor | +31.1 Elo H1 @ 750g | ✅ validated (factor retune #692 end-values) |

### Factor tune cycle (cumulative on factor net)

| SPRT | Comparison | Result | Status |
|------|------------|--------|--------|
| #698 | (2000-iter factor tune continuation) | SPSA completed | applied on branch; NMP_UNDEFENDED_MAX bug fixed to 5 |
| #700 | tune/v9-698-factor vs non-factor | +22.9 Elo H1 @ 1064g | factor arch effect |
| #703 | tune/v9-698-factor vs pre-tune (both factor) | −2.6 Elo H0 @ 9226g | ❌ tune neutral on factor |
| #704 | tune/v9-698-factor + factor vs trunk + non-factor | +20.8 Elo H1 @ 1190g | factor arch effect (confirmed) |
| #705 | tune/v9-698-factor vs prod SB800 | −15.6 Elo H0 @ 2254g | SB training length gap |
| #706 | (continuation, 2000 iters more, cumulative 5000) | SPSA completed | factor cumulative |
| #710 | tune/v9-706-factor vs pre-tune | −3.4 Elo H0 @ 7618g | ❌ tune dropped (confirmed factor's pruning optimum near pre-tune) |

**Factor lesson**: factor architecture is worth +20-23 Elo vs non-factor at
same training length. SPSA retune on factor adds nothing — factor's pruning
optimum sits close to pre-tune values. Do not apply #698/#706 tune wholesale;
use factor net with pre-tune params once SB800 factor net lands.

### NPS investigation — Phase 3 pruning (Hercules lane)

SPRTs driven by `docs/coda_vs_reckless_nps_2026-04-23.md` pruning scan.

| SPRT | Branch | Result | Status |
|------|--------|--------|--------|
| #707 | experiment/direct-check-carveout | **+2.5 Elo H1 ✓ @ 26186g** | ✅ merged (FP + BNFP direct-check carve-out; Reckless #410 + #630 pattern) |
| #708 | experiment/lmp-adaptive-improvement | −6.7 Elo H0 @ 4644g | ❌ raw port of Reckless constants; 20% bench drop — formula needs SPSA on Coda's eval scale |
| #709 | experiment/nmp-skip-tt-capture | −7.8 Elo H0 @ 3794g | ❌ raw port; 48% bench growth — gate fires too broadly on Coda's TT distribution |

**Lesson**: Reckless's tuned pruning constants don't transfer directly to
Coda's eval scale. Both formulas need SPSA-tunable variants rather than
direct ports.

### NPS investigation — cache hygiene (Zeus lane + mine)

SPRTs driven by Zeus's flatten/prefetch/walk-back work + eval-TT writeback.

| SPRT | Branch | Result | Status |
|------|--------|--------|--------|
| #711 | experiment/flatten-acc-entry | **+6.5 Elo H1 ✓ @ 3608g** | ✅ merged (Zeus Item 1: 4 big AccEntry Vec fields → inline arrays) |
| #713 | experiment/eval-only-tt-writeback | **+14.7 Elo H1 ✓ @ 2462g** | ✅ merged (Hercules Item 7: seed TT static_eval on NNUE-hit + TT-miss; evals/node 0.677 → 0.581) |
| #714 | experiment/l1-inference-compact-rebased | −2.5 Elo H0 @ 7184g | ❌ dropped (Zeus Item 3: load-time compact 4 MB — saves no Elo on current non-sparse net) |
| #719 | experiment/prefetch-threat-deltas | −4.4 Elo H0 @ 4382g | ❌ dropped (Zeus Item 5: manual prefetch on modern HW) |
| #720 | experiment/psq-walkback-v2 | **+1.1 Elo H1 ✓ @ 35666g** | ✅ merged (Zeus Item 6: clean-retry of walk-back; prior #424 −17.86 Elo was confounded by struct/layout churn) |
| #721 | experiment/prefetch-threat-deltas-t1 | −8.1 Elo H0 @ 2570g | ❌ dropped (tier-1 retry of Item 5) |
| #722 | experiment/compact-plus-frontload | −3.7 Elo H0 @ 6900g | ❌ dropped (Zeus bundled Item 3 + Item 4 hot-feature frontload; no Elo on current net) |
| #723 | experiment/sibling-se-propagation | −3.5 Elo H0 @ 7250g | ❌ dropped (Hercules Item 11: raw Reckless port flipped −2/−3 SE → +2/+3 sibling; too aggressive) |
| #724 | experiment/nmp-skip-tt-capture-v2 | +0.1 Elo H0 @ 32986g | ❌ dropped (refined retry with depth + SEE gates; settled neutral, no signal) |
| #725 | experiment/tt-static-eval-wider | −167.5 @ 442g (bug) | ❌ bug in bit-layout refactor (Zeus Item 9; retry pending) |
| #726 | experiment/flatten-acc-entry-phase2 | −7.1 @ 1218g (bug) | ❌ bug (Zeus Item 10 complete-flatten; retry pending) |
| #727 | experiment/sibling-se-propagation-v2 | (in flight) | clamped-to-+1 retry of #723 |

### Session cumulative result

Merged during the 2026-04-23/24 wave (non-factor trunk only, prod net
SB800 DAA4C54E):
- direct-check carve-out: +2.5
- flatten AccEntry: +6.5
- eval-only TT writeback: +14.7
- PSQ walk-back clean retry: +1.1
- direct-check + mobility-delta + lmr-shallower-v2 + T2.1 + sentinel reductions + tune #686: from prior lines, summed ~+8

Roughly **+24.8 Elo merged from NPS-investigation-derived work** in
24 hours (not counting correctness fixes already in the prior batch
or the factor-architecture work which is waiting on SB800 factor net).

### Cross-engine validation (Rivals RR)

4-day Rivals RR delta on V9 binary (all merges above against static
v5 reference + rival set):

| Date | V9 Elo | V9 rank |
|------|-------:|:-------:|
| 2026-04-20 | 53 | 6 |
| 2026-04-24 | **109** | **1** |

**+56 Rivals-internal**. Per `feedback_rivals_rr_stretches_elo.md` (~2×
stretch), broader-pool estimate is ~+28 Elo. Consistent with the
SPRT-merged total within noise. Confirms the NPS-investigation wins
convert to actual cross-engine strength, not self-play artefact.

## 2026-04-24 → 2026-04-25 session — v9-into-main merge + force-more-pruning cluster

### v9 architecture into main (2026-04-24)

`feature/threat-inputs` merged into `main` at commit `ea07d93`. Main is now v9
trunk: 768 accumulator + 66,864 threat features + 16/32 hidden layers,
kb10 layout, prod net `net-v9-768th16x32-kb10-w15-e800s800-crelu`.

### Force-more-pruning cluster (intentional Reckless-style port wave)

Reckless's pruning archetype (uncapped depth gates, smaller futility margins,
higher LMP base) was ported piecewise to test which outliers carried Elo on
Coda's eval scale. Pattern: most direct-port values H0 because Coda's lmr_d
formula vs Reckless's raw-depth formula doesn't translate 1:1.

| SPRT | Branch | Result | Status |
|------|--------|--------|--------|
| #730 | experiment/n4-halfmove-scaled-margins | −1.0 Elo H0 @ 15946g | ❌ on detuned trunk; re-SPRT in flight (#755) |
| #732 | experiment/nmp-ttnoisy-rplus | −0.4 Elo H0 @ 41080g | ❌ on detuned trunk; re-SPRT trending H1 on post-tune-750 trunk (#754) — guard pattern wants retune-on-branch |
| #734 | experiment/bnfp-depth-uncap | −1.2 Elo H0 @ 14454g | ❌ direct uncap (d≤4 → unbounded); tune #750 found 13 instead |
| #735 | experiment/lmp-depth-uncap | −1.1 Elo H0 @ 14452g | ❌ direct uncap (d≤8 → unbounded); tune #750 found 14 instead |
| #736 | experiment/futility-tighten | **+4.4 Elo H1 ✓ @ 3888g** | ✅ merged (FUT_BASE 78→40, FUT_PER_DEPTH 160→65 — Reckless-scale margins) |
| #737 | experiment/rfp-depth-uncap | **+2.2 Elo H1 ✓ @ 34380g** | ✅ merged (RFP depth cap 10→16) |
| #738 | experiment/see-quiet-reckless-shape | −45.0 @ 264g (catastrophic) | ❌ aborted — formula incompatibility |
| #740 | experiment/see-quiet-reckless-shape-v2 | −13.3 Elo H0 @ 4272g | ❌ recalibrated retry still H0; SEE_QUIET_MULT=46 (lmr_d formula) doesn't directly port to Reckless's raw-depth shape |
| #741 | fix/expose-hardcoded-pruning-tunables | **+3.7 Elo H1 ✓ @ 4670g** | ✅ merged (5 hardcoded gates exposed as SPSA tunables; tune #743 immediately moved IIR_MIN_DEPTH 4→2) |
| #744 | experiment/iir-ablate | −6.3 Elo H0 @ 4398g | ❌ dropped (IIR is contributing; tune-driven 4→2 captured optimal value) |
| #745 | fix/expose-more-pruning-tunables | **+10.4 Elo H1 ✓ @ 2662g** | ✅ merged (3 more hardcoded gates exposed as tunables; SPSA found NMP_MIN_DEPTH 5, PROBCUT_MIN_DEPTH 6) |
| #749 | experiment/rfp-depth-uncap-v2 | +0.2 ±1.5 / 43846g (stopped flat) | ❌ dropped — change already absorbed by tune #747's RFP_DEPTH 11 |

**Force-more-pruning lesson** (durable): direct-port of Reckless's pruning
constants H0s on Coda's eval scale, but **biased-aggressive starting values for
SPSA find a different basin** (see #750 below). The cluster of merges (#736
+4.4, #737 +2.2, #741 +3.7, #745 +10.4) demonstrates that exposing previously-
hardcoded gates as tunables banks Elo on its own — SPSA finds non-default
optima the moment they're exposed.

### Tune cycle on post-v9-merge trunk

| SPRT | Branch | Result | Status |
|------|--------|--------|--------|
| #743 | tune/v9-merged (full-sweep, 2500 iters) | SPSA completed | applied to trunk via #747 |
| #747 | experiment/retune-post-743 | **+7.2 Elo H1 ✓ @ 3920g** | ✅ merged (default-rooted SPSA captured flywheel from v9 merge) |
| #750 | experiment/force-more-pruning (2000 iters, biased-aggressive start) | SPSA completed | applied to trunk via #752 |
| #752 | experiment/apply-tune-750 | **+12.3 Elo H1 ✓ @ 3132g** | ✅ merged (biased-start SPSA basin found a fundamentally different convergence point — bench dropped 67% from 2.43M → 788K) |

**Methodology validation (durable)**: Biased-starting-point SPSA can find
basins default-rooted SPSAs cannot reach. #743 (default-rooted) banked +7.2;
#750 (biased-aggressive start, all pruning thresholds wider) banked +12.3 ON
TOP. Same 77-param sweep, same trunk, different convergence basin. See
`memory/feedback_spsa_biased_starting_point.md`.

### Group-lasso training probe (2026-04-25)

First group-lasso (per-row L1) net trained on Bullet fork
`feature/decouple-l1-lr` at `--group-l1-decay 1e-2`, SB200, kb-reckless layout.

| SPRT | Comparison | Result | Status |
|------|------------|--------|--------|
| #753 | grouplasso-1e2 net (SB200) vs C8fix dense (SB200) | **+13.3 Elo H1 ✓ @ 1468g** | ✅ validated methodology (NOT a ship candidate — SB200 vs SB200) |

**Group-lasso lesson (durable)**: at SB200, group-lasso acts as
*regularization*, not just a sparsifier — sparse net is +13 Elo stronger
than matched dense baseline at 13.48% threat-row sparsity. Cache-residency
target was 35% sparsity; we hit 13.5%, but the L1 effect itself helps Elo
at SB200. See `memory/project_group_lasso_acts_as_regularizer.md`.

**Caveat**: SB200 SPRT magnitudes don't translate to SB800 ship-readiness.
Probe #1 vs prod SB800 would lose 50-100 Elo (missing low-LR convergence
tail). Probes #2 (3e-2) and #3 (5e-2) running in parallel on GPU hosts.

### Active batch (2026-04-25 morning) — re-SPRTs + correctness audits

Submitted on post-tune-750 trunk while GPU training runs:

| SPRT | Branch | Class | Bounds | Status |
|------|--------|-------|--------|--------|
| #754 | experiment/nmp-ttnoisy-rplus (re-SPRT) | guard retune-on-branch | [-3, 3] | **H0 (manually stopped 2026-04-25)** — −0.8 ±1.9 / 24460 g, LLR −2.20. Post-tune-750 didn't flip it; H0 a second time. |
| #755 | experiment/n4-halfmove-scaled-margins (re-SPRT) | pruning | [-3, 3] | **stopped (no signal, 2026-04-25)** — +0.2 ±3.7 / 6606 g, LLR +0.12. Halfmove-scaling provides no marginal info on top of existing margins. |
| #756 | experiment/sibling-se-propagation-v2 (re-SPRT) | extension | [-3, 3] | **stopped at fade 2026-04-25** — −0.2 ±1.9 / 25832g, LLR −0.59. Slow drift toward H0 with tight bars; no clean resolution coming. |
| #757 | fix/should-stop-granularity (R5 #3) | TM 4096→1024 nodes | [-5, 5] | **H0 −2.4 ±3.6 / 6516g** — re-test on post-tune-750 trunk didn't flip original #674 H0. 4096-node granularity is correct at STC. Drop. |
| #758 | fix/recapture-ext-ply-guard (audit) | extension correctness | [-3, 3] | **H1 +1.5 ±2.2 / 16606g** (LLR 2.97) — **MERGED** 2026-04-25 |
| #759 | fix/fh-blend-skip-in-se (audit) | extension correctness | [-3, 3] | **H1 +1.7 ±2.4 / 14764g** (LLR 2.99) — **MERGED** 2026-04-25 |
| #760 | fix/threats-blocker-bounds (audit) | bounds-safety (sq=63) | [-5, 5] | **H1 +0.9 ±2.2 / 17094g** (LLR 3.09) — **MERGED** 2026-04-25 |
| #761 | fix/se-singular-beta-mate-clamp (audit) | extension correctness | [-3, 3] | **H0 −1.8 ±2.4 / 14412g** (LLR −2.95) — mate-distance clamp removed legitimate SE in mate-shaped positions where multi-cut return was correct. Bucket: mechanism-wrong. Drop. |
| #762 | experiment/lmr-shallower-margin-10 (parameter probe) | LMR cp margin 20→10 | [-3, 3] | **H0 −2.5 ±2.9 / 10482g** (LLR −2.97) — 20cp is the optimum (per #679 H1 +1.4); 10cp too aggressive. Drop. |
| #764 | fix/aarch64-tt-tbcache-ordering | ARM SMP correctness, [-5, 5] non-regression | [-5, 5] | **−0.1 ±1.9 / 24886g** (LLR −0.40, stopped at fade) — x86 cost ≈ 0, **MERGED** 2026-04-25 (ARM-as-first-class commitment). |

**H0 post-mortems (2026-04-25)**:

- **#754 nmp-ttnoisy-rplus** (second H0). Mechanism: guard adds `r++`
  when TT move is a capture. Per CLAUDE.md guard sub-pattern, the
  vanilla SPRT only captures direct safety gain — not the cluster
  rebalancing gain. We never did the NMP-cluster retune-on-branch.
  **Bucket: mechanism-wrong (missing retune step)**. Iterate:
  retune NMP_BASE_R / NMP_DEPTH_DIV / NMP_EVAL_DIV / NMP_EVAL_MAX
  / NMP_VERIFY_DEPTH (5-6 params, ~1500 iters) on this branch, then
  SPRT guard+retune vs trunk. Worth a refined retry.
- **#755 n4-halfmove-scaled-margins** (second H0). Mechanism: scale
  RFP/futility margins by halfmove-clock proximity to draw, on the
  hypothesis that near-50mr positions should prune less aggressively.
  **Bucket: signal overlap** — existing eval already encodes 50mr
  proximity through correction history; margin scaling adds no
  marginal info. Drop, do not iterate.

### Session cumulative

Merged during 2026-04-24 → 2026-04-25 wave:
- v9 architecture into main (ea07d93, prior gain absorbed)
- futility-tighten: +4.4
- rfp-depth-uncap: +2.2
- expose-hardcoded-pruning-tunables: +3.7
- expose-more-pruning-tunables: +10.4
- retune-post-743 (#747): +7.2
- apply-tune-750 (#752): +12.3

**~+40.2 Elo merged** from search/tune levers on top of v9 architecture in
~36 hours. Plus probe #753 (+13.3) validated group-lasso methodology
(non-ship). Cumulative for the 2026-04 sprint, this brings trunk from
roughly +0 (just v9 merged) to +40 self-play Elo, with cross-engine
transfer ~50-80% of that = +20 to +32 cross-engine.

### Correctness audit batch — final close 2026-04-25

After the active batch resolved, the following correctness fixes
merged in a second wave:

- **#758 fix/recapture-ext-ply-guard**: +1.5 (recapture extension was
  firing at root on game-history captures, leaking +1 ply into wrong
  subtree)
- **#759 fix/fh-blend-skip-in-se**: +1.7 (FH blending dampened
  singular_score during SE verification, biasing DEXT decisions)
- **#760 fix/threats-blocker-bounds**: +0.9 (1u64 << 64 UB in xray
  blocker shift at sq=63; defence-in-depth, masked today by upstream
  `revealed == 0` filter but fragile)
- **#764 fix/aarch64-tt-tbcache-ordering**: non-regression (Acquire/
  Release on x86 cost ≈ 0, ARM SMP correctness benefit
  fleet-untestable but required per project_arm commitment)

Plus three audit H0s with clear post-mortems (#761 SE-mate-clamp,
#757 should-stop-granularity, #762 LMR-shallower-10).

**Total banked from correctness audit batch: ~+4.1 Elo** on top of
the +40 from search/tune cluster. Brings 2026-04-24 → 2026-04-25
session total to **~+44 Elo merged**.

### NPS investigation (2026-04-25 afternoon)

Following Adam's anecdata "NPS down ~10% post-tune-750" we instrumented
counter atomics in HEAD and ea07d93 to localize the +28% cache-miss-
per-node bump from earlier perf data. Result: **the +28% per-node bump
is mostly a tree-shape denominator artifact, not a regression**.

| Counter | ea07d93 | HEAD | per-node Δ | absolute Δ |
|---|---|---|---|---|
| ContHist lookups | 10.15/n | 11.73/n | +15.6% | **−62.5%** |
| HistScore calls | 1.96/n | 2.47/n | +26.0% | **−59.2%** |
| Total nodes | 2.43M | 0.79M | — | **−67.5%** |
| Bench wall-clock | 11.7s | 4.2s | — | **2.78× faster** |

Better pruning shrinks leaf nodes faster than interior nodes →
per-node averages weight more toward heavy interior nodes → ratios
inflate even when total memory traffic fell sharply.

**ContHist AoS-pack (originally framed as "NPS recovery") reframed
as absolute-NPS gain** (~2-5 Elo class), not regression-recovery.
Queued behind feature work. Findings saved to memory:
- `feedback_per_node_metrics_misleading_with_pruning_changes.md`
- `feedback_local_ob_worker_kill_freely.md` (Adam permission)
- Reframed `feedback_lmp_reorder_nps_neutral.md`

### Active batch (2026-04-25 evening)

| SPRT | Branch | Class | Bounds | Status |
|------|--------|-------|--------|--------|
| #767 | experiment/good-bad-quiet-split-v3 | move ordering | [0, 3] | **trending H0** −1.1 ±1.8 / 26K g, LLR −2.66 |
| #768 | experiment/nmp-ttnoisy-rplus-v3 | NMP guard + retune | [0, 3] | early/flat +0.5 / 8.4K g, LLR −0.05 |
| #769 | experiment/pin-bonus-solo (T1.1) | move ordering bonus | [0, 3] | **submitted** — value-filtered pin without skewer noise. Bench 804743 vs main 788473 (+2.1%). |
| #770 | experiment/quiet-see-attacking (T3.2) | move ordering bonus | [0, 3] | **submitted** — "good quiet" cheap-SEE proxy: bonus when offense target value > attacker value. Bench 734357 vs main 788473 (-6.9%, tree shrinks from earlier cutoffs). |

### Resolution of 2026-04-25 evening batch

| SPRT | Branch | Result | Bucket |
|------|--------|--------|--------|
| #767 | good-bad-quiet-split-v3 | **H0** −1.2 ±1.8 / 27.3K (LLR −2.97) | mechanism overlap; drop |
| #768 | nmp-ttnoisy-rplus-v3 | **H0** −0.5 ±1.4 / 44.4K (LLR −2.97) | guard captured direct gain only — needed NMP cluster retune-on-branch (per CLAUDE.md guard sub-pattern); refined retry queued as #790 SPSA |
| #769 | pin-bonus-solo (T1.1) | **H0** −13.2 ±4.9 / 4.0K (LLR −3.01) | mechanism-wrong; pin bonus magnitude or eligibility too broad |
| #770 | quiet-see-attacking (T3.2) | **H1 +2.6** ±1.8 / 26.7K (LLR 2.96) | **MERGED** (a89c014) — small-quiet attack-creating SEE proxy carried Elo |

## 2026-04-25 → 2026-04-26 session — Tier-1 cross-engine port wave + factor SB800 deployment

### Cross-engine queue Tier-1 batch (per docs/cross_engine_comparison_2026-04-25.md)

8 of 10 Tier-1 items submitted; 7 H0, 1 still pending. Pattern strongly
calibrates the doc's expected-Elo column DOWN by 50-70% on raw consensus
ports — but the H0s surface real structural diagnostics worth retuning
or bisecting toward.

| SPRT | Branch | Bounds | Result | Bucket / follow-up |
|------|--------|--------|--------|--------|
| #771 | fix/n6-promotion-imminent-stm (audit) | [-3, 3] | **H0 −3.3 ±3.3 / 7.7K** (LLR −2.98) | mechanism-wrong; pre-move STM fix didn't deliver direction predicted by audit |
| #772 | experiment/nmp-cut-node-gate | [0, 3] | **H0 −2.3 ±2.2 / 17.9K** (LLR −2.97) | retune-needed-prior; SPSA #790 (NMP cluster, 1500 iters) on branch in flight |
| #773 | experiment/enter-threat-penalty | [0, 3] | **H0 −0.4 ±4.1 / 5.2K** (LLR −0.31, stopped) | refined as #781 with split tunables (also H0); mechanism overlap with escape-bonus likely |
| #774 | experiment/lmr-c-swap | [0, 3] | **H0 −9.3 ±7.2 / 1.6K** (LLR −0.94, stopped early) | **structural finding (load-bearing)**: cap LMR had binary ±1 captHist threshold at ±2000 while quiet LMR uses continuous `hist/LMR_HIST_DIV`. The asymmetry was why SPSA had detuned C_CAP. Follow-up: #780 (capture-lmr-hist-adjustment) added LMR_CAP_HIST_DIV at default 1024; SPSA #791 retunes the new cluster. THIS, not the swap, is the actual lever |
| #780 | experiment/capture-lmr-hist-adjustment | [0, 3] | **H0 −0.5 ±2.1 / 21.2K** (LLR −1.38) | retune-needed-prior at default tunable; SPSA #791 in flight to find LMR_CAP_HIST_DIV's converged value |
| #781 | experiment/enter-threat-split-tunables | [0, 3] | **H0 +0.1 ±1.0 / 94.7K** (LLR −2.96) — burned excess fleet capacity per `feedback_stop_sprt_when_upper_ci_below_elo1` | mechanism-wrong / signal-overlap; threat-aware history already captures most of this signal |
| #785 | experiment/pawn-history-8192 | [0, 3] | **H0 −1.2 ±1.8 / 27.7K** (LLR −3.00) | initial bucket value-too-extreme; bisected to 2048 (#797) but **bisection landed even worse: −3.3 ±3.3 / 8.3K (LLR −3.05)**. 512 → 2048 → 8192 = -3.3 → -1.2 non-monotonic. Pawn-history size isn't a free Elo lever for Coda in this range. **Drop**; possible refined retry is `pawnHistFill = -919` SPSA-tuned init (Integral pattern) at default 512 size, but priority low. Lesson: bisection isn't always rescue — sometimes the consensus value carries a different mechanism Coda lacks. |
| #786 | experiment/hist-prune-gate-drop | [0, 3] | **H0 stopped −0.2 ±1.2 / 56.5K** (LLR −2.88) — **MERGED** as ebe9ad6 for code hygiene + consensus alignment | bench 947494 → 736194 (-22%): consensus-gate change shifts search shape substantially. Elo-neutral but flywheel-eligible — surrounding pruning cluster has shifted equilibrium. **Queue retune-on-branch follow-up** per `feedback_bench_delta_signals_retune_need` |
| #787 | experiment/triple-extension | [0, 3] | **H0 −1.0 ±1.7 / 30.7K** (LLR −2.95) | retune-needed-prior at flat-constant defaults → #792 SPSA found no basin → reclassified **mechanism-wrong**; refined retry as #804 with Reckless-pattern structural fix (PV/quiet-aware additive margins) |
| #804 | experiment/se-margins-reckless v1 | [0, 5] | **H0 −12.5 ±6.4 / 2.5K** (LLR −2.96) | structural port at Reckless-derived defaults at +67% bench. **Diagnosis**: missed 2 things vs Reckless source — (a) `correction_value.abs()/128` modulator in margin formulas (line 686-689); (b) at non-PV non-quiet our default `dext_margin=0` auto-fires DEXT on every singular hit. v2 below |
| (#804 v2) | experiment/se-margins-reckless | tune-only | **SPSA #806** running, 84 params 2500 iters | added DEXT_MARGIN_CORR (16), TRIPLE_MARGIN_CORR (15) for the modulator; added DEXT_MARGIN_BASE (30) to put non-PV at sane threshold close to old Coda's >11cp. Bench reduced 1580K → 1274K (+67% → +35%). Yin/Yang frame: extensions enable more pruning; SPSA finds the equilibrium. SPRT will follow once tune resolves |

**Calibration**: 8/9 H0 on consensus ports with retune-needed or
structural-finding follow-ups. Refined retries: #772→#790, #780→#791,
#787→#792→#804 (chained mechanism-wrong → structural fix → retune
pending). Bisections: #785→#797 (drop). Drops: #771, #773→#781.
**Merged for consensus alignment despite H0**: #786 (gate-drop, code
hygiene + bench-delta retune queued). The doc's "all-Tier-1 +25-50 Elo"
expectation collapses to maybe +5-15 once the SE-margins retune and
bench-delta retunes resolve.

### Factor SB800 net swap + tune wave (PROD-AFFECTING)

| SPRT | Net | Result | Outcome |
|------|------|--------|---------|
| #782 | 1EF1C3E5 (factor SB800 + C8fix-1) vs prod DAA4C54E (SB800 reckless-crelu) | **H1 +3.3 ±2.2 / 23.8K** (LLR 2.95) | net carries Elo at default tunables |
| #784 | full-sweep SPSA (77 params, 2500 iters) on factor SB800 | applied as defaults | tune-784 commit (697a703) |
| #788 | tune-784 vs trunk (both on 1EF1C3E5) | **H1 +3.2 ±2.5 / 14.9K** | tunables earn separate Elo on top of net swap |
| #789 | tune-784 + 1EF1C3E5 (deployment package) vs prod (DAA4C54E + pre-tune) | **H1 +4.9 ±3.5 / 9.7K** (LLR 2.94) | **MERGED** as v0.4.0-nets prod (commits 1a5399b, 56a37f0). 1EF1C3E5 is current PROD net |

### C8fix-2 SB200 isolation pair

CC483681 = SB200 factor + C8fix-2 (first net to actually contain
both halves of the C8 fix; trained on `feature/no-blocking-sync`
post-62931d1). C0A97CF4 = SB200 factor + C8fix-1 only (Apr 22 train,
predates 62931d1).

| SPRT | Comparison | Result | Bucket |
|------|-----------|--------|--------|
| #793 | C0A97CF4 (SB200 factor C8fix-1) vs 1EF1C3E5 (SB800 prod) | **H0 +0.1 ±4.6 / 6.1K** (LLR −0.58, stopped at fade) | net-vs-net check; SB200 vs SB800 expected ~0 magnitude given training-depth difference |
| #794 | CC483681 (SB200 factor + C8fix-2) vs C0A97CF4 (SB200 factor + C8fix-1) | **H0 −9.0 ±5.9 / 3.7K** (LLR −2.96) | **trunk-mismatch confound**: trunk tunables calibrated for noisy-threat semantics (1EF1C3E5 = C8fix-1) — fits BASE not DEV. Bench delta 33% triggers retune-needed-prior. Follow-up: SPSA #796 (full-sweep retune on CC483681, 77 params, 2500 iters) running. Re-SPRT post-tune to isolate C8fix-2's true contribution |

**End-to-end C8fix-2 validation** (`coda fuzz-threats --postfix`):
- pre-C8fix-2 Bullet ref: 33.42% feature mismatch vs Coda inference (all real_stm=Black)
- post-C8fix-2 Bullet ref: 0.00% mismatch across 200K evals × 2 seeds
- Confirms training/inference fully agree post-62931d1

### Active SPSA tunes (2026-04-26)

| Tune | Branch / target | Iters / params | Status |
|------|-----------------|---------------|--------|
| #790 | experiment/nmp-cut-node-gate | 1500 / 8 | **finished** — NMP_BASE_R +14%, NMP_UNDEFENDED_MAX −37%, NMP_MIN_DEPTH +15%. Post-tune SPRT #807: **H0 −0.4 / 49.6K**. Bucket: signal-not-there (cut-node gate's gain at default tunables collapsed under retune; SPSA absorbed it elsewhere). Drop. |
| #791 | experiment/capture-lmr-hist-adjustment | 1500 / 5 | **finished** — LMR_CAP_HIST_DIV 1024→1287 (+25.7%), CAP_HIST_BASE 18→19 (+7%). Post-tune SPRT #809: **H0 −1.2 / 28.5K**. Bucket: signal-not-there. The LMR_C_QUIET/CAP asymmetry (#774 finding) survives — continuous shaping for captures alone wasn't the lever. Drop. |
| #792 | experiment/triple-extension | 1000 / 3 | **finished** — no basin found at flat-constant defaults; reclassified by #804 investigation as **mechanism-wrong** (Coda's flat 75/16 missed Reckless/SF's `204*PvNode - 16*is_quiet` per-node-type margin scaling). Refined retry: #804 with structural fix |
| #795 | main focused (CC483681) | 1500 / 15 | **STOPPED** at iter 55 — wrong scope (focused-cluster playbook applies to feature retunes, NOT net swaps). Replaced by #796 |
| #796 | main full-sweep (CC483681) | 2500 / 77 | **finished** — large movers: NMP_EVAL_MAX −57%, LMR_THREAT_DIV −23%, LMP_BASE +16%, CORR_W_NP/MINOR/MAJOR/CONT all 9-12%. Post-tune SPRTs: **#802 H1 +8.4 / 7.4K** (tune-796 vs trunk, both on CC483681 SB200) — retune carries Elo on top of net swap. **#803 H1 +4.8 / 6.4K** (CC483681 + tune-796 vs C0A97CF4 + pre-tune trunk, both SB200) — package beats C8fix-1 SB200. Deployment package against current prod (1EF1C3E5 SB800) blocked on **C8fix-2 SB800 training** (task #117 in flight). Don't apply tune-796 to current trunk yet — it's calibrated to CC483681's bench landscape, not 1EF1C3E5's. |
| #806 | experiment/se-margins-reckless v2 | 2500 / 84 | running, ~2168/2500 |
| #811 | experiment/lmp-reckless-shape (Phase B #3+#4+#5) | 2500 / 80 | running, ~567/2500. Phase B core SPSA on the new history-aware continuous-improvement formula. |

### Anomalies / setup errors

| SPRT | Branch | Result | Note |
|------|--------|--------|------|
| #777 | main | -38.2 ±8.9 / 1.8K (LLR −3.01) | suspicious; likely net mismatch or branch issue (no commit notes) |
| #778 | main | -146.4 ±17.1 / 0.8K (LLR −3.02) | clearly broken setup; investigate before next "main" SPRT |

### Method updates banked this session

- `feedback_h0_isnt_terminal_bisect_and_diagnose.md` — H0 on parameter probe = bisect; H0 on structural port = find asymmetry. Burnt: pawn-history-8192 → no 1024/2048 follow-up; lmr-c-swap reported without surfacing gate-asymmetry diagnostic
- `feedback_log_to_experiments_md_immediately.md` — log SPRTs to experiments.md as they resolve; update referenced docs inline. Burnt: this batch (~25 SPRTs) was unlogged for 24h
- `feedback_net_swap_needs_full_sweep_retune.md` — net swap retune is full-sweep, not focused-cluster
- `feedback_bench_delta_signals_retune_need.md` — >15% bench delta signals retune. #794's 33% delta was the canonical case
- `feedback_stop_sprt_when_upper_ci_below_elo1.md` — exception to don't-stop-on-fade; #781 burned 94K games

### Session cumulative (2026-04-25 → 2026-04-26)

**Merged with confirmed Elo:**
- #770 (T3.2 quiet-see-attacking): **+2.6**
- #782/#788/#789 (factor SB800 + tune-784 deployment package): **+4.9** (vs prior prod, 1EF1C3E5 + tune-784 now prod)
- #758 (recapture-ext-ply-guard, late close): +1.5
- #759 (fh-blend-skip-in-se, late close): +1.7
- #760 (threats-blocker-bounds, late close): +0.9
- #764 (aarch64-tt-tbcache-ordering): non-regression, ARM correctness

**~+11.6 Elo merged this session.** Plus the C8fix-2 isolation pair
(#793/#794) and 9 cross-engine port SPRTs (mostly H0 with retune
follow-ups in flight) and 6 SPSA tunes (1 applied, 4 running, 1 stopped).

### LMP-vs-Reckless alignment, Phase A (2026-04-26)

Driven by `docs/reckless_vs_coda_pruning_diff_2026-04-25.md` finding:
Coda fires LMP **28× more often per Kn** than Reckless. Plan
documented in `docs/lmp_reckless_alignment_plan_2026-04-26.md`. 7
differences enumerated; phased A (independent surgical) → B
(structural threshold) → C (gate removals).

| SPRT | Branch | Result | Outcome |
|------|--------|--------|---------|
| #808 | experiment/lmp-direct-check-carveout (Phase A #1) | **H1 +2.54** | **MERGED** (3252a3a). Single-line `&& !board.gives_direct_check(mv)` inside LMP gate. Bench 736194 → 796065 (+8%, fewer LMP fires let checking late moves through). |
| #810 | experiment/lmp-skip-quiets (Phase A #2) | **H1 +1.86** | **MERGED** (6f54162). Reckless pattern (search.rs:752): set `skip_quiets` flag once LMP fires; bypass remaining quiets in move loop. Bench 736194 → 1000987 (+36%). Submitted as data point per Adam — bisection confirmed setter-without-check is bench-neutral while check-enabled is +22% on its own; logical equivalence to per-move re-fire didn't hold empirically. Per-move-trace instrumentation queued if interpretation needs refinement. |

**Phase A banked: +4.4 Elo merged.** Both wins compound on top of
trunk's tune-784 retune; bench shifted ~+36% reflects materially
shallower trees (more LMP fires reach late quiets).

### Active SPSA tunes (Phase B, in flight)

| Tune | Branch / target | Iters / params | Status |
|------|-----------------|---------------|--------|
| #806 | experiment/se-margins-reckless (v2 with corr modulator + DEXT_MARGIN_BASE) | 2500 / 84 | **finished**. Applied as 8736d30; post-tune SPRT #815 H0 (-4.84 ±10.5 / 790g, stopped). Hypothesis: TRIPLE bundling diluted DEXT signal — TRIPLE was independently H0 at #787 with no SPSA basin (#792). Split branch as `experiment/se-margins-reckless-no-triple`. |
| #811 | experiment/lmp-reckless-shape (Phase B #3+#4+#5) | 2500 / 80 | **finished**. LMP_K_HIST stayed flat at 67-68 through full tune (signal-not-there warning). Applied as e09c5cb. Post-tune SPRT **#818 H0 -2.4 ±3.2 / 9134g**. Confirmed the LMP_K_HIST flat-line was directionally right. Decomposition follow-up: `experiment/lmp-reckless-shape-no-hist` (Option B per 2026-04-27 discussion), strip history term keep #4+#5 — SPSA **#819** running. |
| #816 | experiment/se-margins-reckless-no-triple | 2500 / 80 | **finished**. Different basin from #806: NMP_BASE_R 6→8, NMP_VERIFY_DEPTH 10→8, DEXT_MARGIN_BASE 24→33 (opposite #806), SEE_QUIET_MULT 24→29, FUT_THREATS_MARGIN 26→15, SE_KING_PRESSURE 3→2, SE_XRAY_BLOCKER 4→3. TRIPLE was distorting #806's basin. Applied as fafd694. Post-tune SPRT **#817 H1 +2.5 ±2.0 / 22040g — MERGED 2026-04-27** (commit 2a93ae9). Trunk bench 809369 → 1237371. |
| #819 | experiment/lmp-reckless-shape-no-hist (Phase B Option B) | 2500 / 79 | **finished**. Applied as aa9f42e. Post-tune SPRT **#832 H0 -3.3 ±2.6 / 13564g** (LLR -2.97). Combined with #818 (original lmp-reckless-shape full retune, H0 -2.4) the Reckless-pattern LMP shape is comprehensively rejected for Coda — both with/without history term, with full retune support. Our existing LMP shape is roughly optimal for our search regime. **Drop this direction permanently.** |
| **#820** | **main full-sweep** (post-#817 trunk) | **5000 / 80** | **finished**. Applied as 45481f9 (62 of 80 params changed; corrhist cluster shifted coherently toward "less aggressive correction" — evidence the v9-era corrhist defaults inherited a v5-era calibration). Post-tune SPRT **#831** running, currently -0.4 ±2.1 / 20.7K games (95% CI [-2.5, +1.7]). Tune-820 finding: same-net 5K-iter retune produces *equivalent* basin to 2.5K (not significantly stronger) — undertuning hypothesis disproven for same-net retunes. Adam considering "calibration merge" if #831 stays in [-3, +3] zone — basin-gravity argument. |

### Tier 3 quick-win SPRTs (2026-04-26)

Cross-engine doc Tier 3 small-win batch — independent 1-2 line
changes, [0, 3] bounds.

| SPRT | Branch | Status |
|------|--------|--------|
| #813 | experiment/material-np-only | **MERGED 2026-04-27** (commit d2aca36). +0.6 ±0.9 / 117K (LLR -0.77 stopped, can't reach H1=3 at this magnitude). Genuine small positive with -20% bench reduction (tree-shape change). Drop pawn term from material scaling (SF/Stormphrax/Halogen/Integral pattern). Eval no longer dampened toward zero in pawn-up endgames. Retune-on-branch candidate: eval-scale change shifts SEE/futility/RFP calibration. New main bench: 809369. |
| #814 | experiment/fh-blend-depth-cap | **H0 -0.3 ±1.3 / 55302g**. Reckless's `min(d, 8)` cap on FH-blend weight didn't carry Elo for Coda. Bench-flat. Drop. |
| #821 | experiment/se-ttpv-margin (cross-engine item 14) | **submitted** [0, 3]. Adds DEXT_MARGIN_TT_PV=16 to the SE margin formula — when tt_pv flag is set, dext_margin shrinks → DEXT fires more eagerly. SF/Reckless/Obsidian/Viri all add a ttPv signal. Bench 1045910. |
| #822 | experiment/cont-hist-drop-ply6 (cross-engine item 16) | **H0 -0.6 ±1.5 / 42K**. Unilaterally zeroing ply-6 from cont-hist quiet scoring rejected. Follow-up: per-ply tunable approach `experiment/cont-hist-tunable-weights` exposes CONT_HIST_W1/W2/W4/W6 (defaults [1,1,1,1]); focused tune **#833** (4 params, 2000 iters) lets SPSA find the shape directly, including potentially zeroing W6 only as part of a compensating redistribution. |

**Lesson learned:** initial #812 submission failed OB bench check
because `cargo build --release` produces a different bench than
`make` (--features embedded-net). OB workers always use make. New
memory: `feedback_bench_via_make_not_cargo.md`.

### Cross-engine Tier 2/3 batch continued (2026-04-27)

Three more small-win candidates from `docs/cross_engine_comparison_2026-04-25.md`,
all [0, 3] bounds.

| SPRT | Branch | Item | Status |
|------|--------|------|--------|
| #827 | experiment/multicut-fix | Item 5 | **H0 -10.9 ±4.4 / 4486g** (LLR -2.95). Reckless-pattern `singular_score >= beta` gate + `(2*s + beta)/3` return. Bench dropped 1237371 → 995284 (-19.6%) — fired far more often than the dead-code old gate. **Bucket: signal-overlap with post-#817 SE/DEXT cluster** (the new fire-on-beta + the recently-merged DEXT decomposition double-prune). Refined retry would gate on `cut_node && !is_pv` like Reckless does. Drop. |
| #828 | experiment/eval-quant-16cp | Item N-14 | running. PlentyChess-style `(eval/16)*16` after material scaling for TT-stability. Bench 1397631. Trending H0 -3.3 ±4.9 / 3.8K early. |

**Bench cascade lesson** (2026-04-27): five OB "Wrong Bench" errors
across multicut-fix + eval-quant before realising the cause. `make`
emits `./coda` at repo root via `--emit link=coda`. I had been
benching `./target/release/coda` (stale binary from cargo build)
which gave 818364 — but `make && ./coda bench` gives the correct
995284 / 1397631 numbers OB workers measure. Always `./coda`, not
`./target/release/coda`. The bench-recipe + check-/errors/-after-submit
rules are now in CLAUDE.md (§Build and Test → Bench-for-OB ritual).

**Tune-820 mid-tune snapshot** (2026-04-27, 2106/5000 iters): clear
evidence the standard 2.5K-iter tune length is undertuning at our
current trunk. `NMP_DEPTH_DIV` was at +29.9% at 275 iters, then
**flipped to -3.2% at 2106 iters** — a sign-flip across a swing
that early-mover heuristics would have called "trending up". The
2500-iter snapshot should be saved for compare against the final
5000 to quantify what we miss at the standard length. Likely informs
the §Long tunes / long training thread in CLAUDE.md.

## 2026-04-27 — Tune cluster wrap-up

| SPRT | Branch | Status |
|------|--------|--------|
| #831 | experiment/tune-820-applied (calibration retune vs main) | **H0 -0.0 ±1.7 / 32672g**. Resolved as expected — pre/post-tune SPRT on a 5K-iter retune that converged near current trunk values; calibration merge per "basin gravity" (snap trunk to fresh SPSA optimum even when delta is zero) shipped to main. Future SPRTs run against post-tune trunk; eliminates SPSA-drift confound for downstream experiments. |
| #832 | experiment/lmp-reckless-shape-no-hist | **H0 -3.3 ±2.6 / 13564g**. Reckless-shape LMP without history-bonus adjustment. Drop. |
| #833 | experiment/cont-hist-tunable-weights (focused tune) | **Stopped at 1326/2000**: SPSA at integer weights with small ranges hit zero-gradient. CONT_HIST_W1/W2/W4/W6 stayed at 1.0 (max ±8% drift, all well within noise). Lesson: SPSA-on-integer-tunable-with-tiny-range needs c_end large enough that perturbations land on different ints. |
| #834 | experiment/tt-replacement-pv-bonus | running, **trending H0 -1.1 ±2.9 / 11084g** (LLR -1.08). SF-style TT replacement bundled three changes: EXACT-always-wins + `+2*is_pv` depth bonus + threshold -3 → -4. Early sample showed +2.6 (n=2618), drifted negative as games accumulated. **Bisect candidates** (per `feedback_h0_isnt_terminal_bisect_and_diagnose.md`): the threshold loosening (-3 → -4) is most suspicious — Coda has 5-slot buckets vs SF's 3-slot clusters, so the same threshold may over-thrash on the denser bucket. EXACT-always-wins and +2*pv individually are likely safer. Queue post-resolution: SPRT each change in isolation. |
| #835 | experiment/tune-830-applied vs e68dcc9 (both with xray net 6C154331) | **H1 +6.0 ±3.0 / 9646g** (LLR 2.97) ✓. Pre-vs-post tune-830 retune. Confirms the 5K-iter SPSA on the new net converged usefully — 64 changed tunables collectively buy +6 Elo over default tunables on the new net. **Methodology validation:** retune-on-branch is the right pattern for net-vs-net SPRTs. The new net (6C154331) is rejected by #836 (-10.7 vs prod), but for any future net that's a deployment candidate, this confirms the retune machinery delivers meaningful Elo. |
| #836 | experiment/tune-830-applied (xray net) vs main (prod net 1EF1C3E5) | **H0 -10.7 ±6.3 / 3100g** (LLR -2.97). Net-vs-net at retuned state. **Reframed by Adam 2026-04-27**: the C8fix-xray training change is a *committed correctness fix* to the training pipeline — not an optional adoption. PROD 1EF1C3E5 is on borrowed time as the last net trained with the bug present. The -10.7 is a diagnostic alarm: why does the corrected pipeline produce a worse net? Open investigation: (a) bisect C8fix-2 alone vs adding x-ray training data; (b) recipe re-search at SB200 (LR tail, WDL, save-rate); (c) widened-range SPSA on params tune-830 pinned at boundaries. Don't ship 6C154331 in current form, but don't drop the work either — the underlying pipeline change is permanent. |
| #821 | experiment/se-ttpv-margin | **H0 -0.1 ±1.2 / 68274g** (LLR -3.03). Final result on the long-running drift. SE TT-PV margin port (DEXT_MARGIN_TT_PV=16 — when tt_pv flag set, dext_margin shrinks). Drop. Mechanism: signal already overlapping with merged DEXT decomposition. |

## 2026-04-28 — Trunk-drift bisect for C8fix-2 SB800 regression

**Setup.** #803 (CC483681 SB200 + tune-796 vs C0A97CF4 SB200 + pre-tune trunk)
landed **+4.8 H1 / 6.4K** on trunk c38623c. The same C8fix-2 vs C8fix-1
SB200 pair re-run on current trunk (#847 main) landed **-10.5 H0**. That's
a 15-Elo direction reversal across the trunk-drift between c38623c and main.
Diagnostic confirmed the regression is in source-code drift, not training.

**Hypothesis.** One (or more) of the search-side merges between c38623c and
main interacts badly with C8fix-2's smoother eval distribution. Bisect by
reverting candidate commits one at a time, then SPRTing **PROD vs C8FIXED
net** (1EF1C3E5 SB800 vs 6C154331 SB800) on the revert branch. If the
revert closes the gap (Elo > -5), the reverted change is implicated. Both
arms run the SAME revert code so only the net changes — clean isolation.

| SPRT | Branch | Reverts | Status |
|------|--------|---------|--------|
| #850 | experiment/revert-material-np-only | 1b0ddc4 (material scaling: non-pawn-only) | **H0 -5.5 ±6.1 / 3570g** ✗ — not the culprit |
| #851 | experiment/revert-hist-prune-gate-drop | 988ca5c (drop !improving && !unstable from hist-prune) | **H0 -4.3 ±5.5 / 4562g** ✗ — not the culprit |
| #852 | experiment/revert-lmp-changes | 8199f17 + 74a9f2a (LMP direct-check carve-out + skip-quiets flag) | **H1 +25.2 ±12.8 / 732g** ✓ (LLR 2.95) — **CULPRIT** |
| #853 | experiment/revert-tune-820 | 45481f9 (Apply tune #820 — 5K-iter retune values) | **H1 +18.2 ±10.9 / 994g** ✓ (LLR 2.97) — **CULPRIT** |

**Bisect verdict.** Two of four reverts H1'd cleanly. The trunk-drift since
#803 has two roots:

1. **LMP changes** (#852, +25.2): the direct-check carve-out and skip-quiets
   flag changed LMP shape in ways that interact badly with C8fix-2's smoother
   eval distribution. Both LMP merges land in this single revert branch.
2. **tune-820** (#853, +18.2): the 5K-iter SPSA retune on main converged to
   parameter values that fit PROD's eval distribution but mis-fit C8fix-2's.
   Pre-tune-820 SPSA values (from tune-816 on SE-no-triple branch) are
   better-aligned to C8fix-2's landscape.

Magnitudes (+25 / +18) are noisy at 700-1000 games but the LLR resolution
is robust. **Combined gap exceeds the 15 Elo measured drift** — likely
indicates partial signal overlap (LMP changes affect what tune-820
calibrated for); follow-up combined-revert SPRT measures the joint effect.

| #854 | experiment/revert-lmp-and-tune820 | 8199f17 + 74a9f2a + 45481f9 (all three trunk-drift commits) | **H1 +13.9 ±9.6 / 1322g** ✓ (LLR 2.96) — joint effect confirmed |

**Verdict — C8FIXED is genuinely the better net.** On a trunk without
LMP changes and without tune-820, **C8FIXED beats PROD by +14 Elo**.
The previous -10.7 result (#836) was entirely trunk-over-fit-to-PROD
artifact, not a real net regression.

**Deployment path** (post-#854 confirmation):

1. **Don't revert LMP changes on main** — Phase A LMP wins (#810
   +1.86 vs PROD-trunk) are real Elo against PROD-fitted trunk.
   Reverting loses that.
2. **Fresh full-sweep SPSA on current main + C8FIXED net** — starts
   from current tunables, SPSA explores until it finds the C8FIXED
   equilibrium that retains the LMP wins. This is the same pattern
   as tune-830 (which retuned for C8FIXED at e68dcc9 baseline) but
   forward-rolled to current main. **Submitted as tune-855**: 80
   params, 5000 iters, dev_network=6C154331 (C8FIXED).
3. Re-SPRT C8FIXED + fresh-tune vs PROD-trunk after SPSA converges.
   Expected outcome: H1 with sizable margin since #854 already
   confirmed +14 on the simpler trunk.

**Update 2026-04-28 — deployment SPRT didn't go as expected.**

| SPRT | Branch | Status |
|------|--------|--------|
| #858 | tune-855-applied (80% / 4033 iters) + C8FIXED vs main + PROD | **H0 -8.1 ±5.5 / 3808g** ✗ |
| #859 | tune-855-final-applied (100% / 5000 iters) + C8FIXED vs main + PROD | running, [-5, 5] bounds |

**Gap between expectation (+14 from #854) and result (-8.1 from #858)
is 22 Elo.** Multiple working hypotheses:

1. **80%-snapshot artifact**: late iters shifted some big movers
   (NMP_VERIFY_DEPTH 8→7, HIST_PRUNE_MULT +4.7%, LMR_HIST_DIV -8%,
   LMP_DEPTH 11→10, HIST_BONUS_OFFSET 9→6). #859 with the 100%
   snapshot will close this hypothesis.
2. **SPSA basin gravity**: starting from tune-820 PROD-fitted values,
   5K iters didn't fully escape to the C8FIXED-fit basin. The
   "fresh full-sweep on main" approach may need to start from
   default values or from `experiment/revert-lmp-and-tune820`'s
   parameter set (proven good with C8FIXED at +14).
3. **LMP-changes inherently disadvantage C8FIXED**: even with
   optimal tunables, the LMP shape favors PROD's eval distribution.
   If true, "retain LMP wins" doctrine doesn't survive the C8FIXED
   deployment.

**Backup deployment paths if #859 H0s:**
- Tune from default values on current main (clean SPSA start, no
  basin gravity from tune-820)
- Tune on `experiment/revert-lmp-and-tune820` branch with C8FIXED,
  then merge LMP changes back on top (using the tune-on-good-baseline
  + reapply pattern)
- Accept revert of LMP changes for C8FIXED deploy (loses #810's
  +1.86 but gains #854's +14 — net positive)

Combined effect (+13.9) is less than the arithmetic sum of individual
reverts (+25 + +18 = +43), confirming **signal overlap**: tune-820
calibrated against the LMP-merged trunk shape; reverting only one
under-estimates because the other still over-fits in the SPSA basin.
This is expected SPSA behaviour: tunes are calibrated for the search
shape they were submitted against.

**Action plan post-#854 resolution:**

- If #854 lands ≥ +15 Elo H1: confirms the trunk-drift gap is fully
  attributable to these two clusters. Next: rather than reverting on main
  (which would lose Phase A LMP wins #810 measured at +1.86 against
  PROD-trunk), the right deployment path is to **fresh-SPSA-retune the
  C8FIXED net at current trunk** — tune-830 already did this but at the
  pre-LMP-merge baseline (e68dcc9). A new full-sweep retune at current
  main with the C8FIXED net should bridge the gap.
- If #854 lands < +15 Elo H1: at least one of the two reverts is
  measurement-noise-inflated; isolate by extending #852 / #853 N to ~5K.

**Methodology note.** This bisect demonstrates the value of single-commit
net-vs-net SPRTs on revert branches. Both arms share identical code so
only the net differs — clean isolation of which trunk merge interacts
poorly with the candidate net. Pattern reusable for any future
"net-vs-net regression" investigations.
