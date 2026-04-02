# Igel Chess Engine - Crib Notes

Source: `~/chess/engines/igel/`
Author: Volodymyr Shcherbyna (derived from GreKo 2018.01)
NNUE: HalfKP-like with 32 king buckets, 22528->2x1024->16->32->1 x8 output buckets, pairwise mul + dual activation (SCReLU + CReLU)
RR rank: #14 at +87 Elo (GoChess-era gauntlet; 195 above GoChess-v5)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Lazy SMP with vote-based best-move selection across threads. Derived from GreKo but heavily modernized.

### Aspiration Windows
- Enabled at depth >= 4 (`aspiration = depth >= 4 ? 5 : CHECKMATE_SCORE`)
- **Very tight initial delta: 5** (vs Coda's 15)
- On fail-low: `beta = (alpha + beta) / 2`, widen alpha by aspiration
- On fail-high: widen beta by aspiration
- Delta growth: `aspiration += 2 + aspiration / 2` (multiply by ~1.5 each iteration)
- No fail-high depth reduction

### Draw Detection
- Two-fold repetition, fifty-move rule, insufficient material (no pawns + MatIndex < 5)

### Mate Distance Pruning
- Standard MDP: `rAlpha = max(alpha, -CHECKMATE + ply)`, `rBeta = min(beta, CHECKMATE - ply - 1)`
- **IMPLEMENTED in Coda**: Coda does not have mate distance pruning. This remains an unimplemented idea.

**Source verified**: Lines 155-160 of search.cpp confirm the implementation.

---

## 2. Pruning Techniques

### Reverse Futility Pruning (Static Null Move Pruning)
- Conditions: `!inCheck && !onPV`, depth <= 8
- Margin: `85 * (depth - improving)`
  - Improving: 85*(d-1); Not improving: 85*d
- Uses TT-corrected `bestScore` (not raw staticEval)
- Compare to Coda: Coda uses improving?70*d:100*d, depth<=7. Different formula structure -- Igel's `85*(d-improving)` is a single expression vs Coda's branching margin.

**Source verified**: Line 287 of search.cpp: `bestScore - 85 * (depth - improving) >= beta`.

### Razoring
- Depth <= 2, margin 150 (fixed)
- Compare to Coda: Coda uses 400+100*d, depth<=2. Igel's margin is much tighter (150 vs 500-600).

**Source verified**: Line 280: `staticEval + 150 < alpha`.

### Null Move Pruning
- TT guard: `!ttHit || !(type==HASH_BETA) || ttScore >= beta`
- Reduction: `R = 5 + depth/6 + min(3, (bestScore-beta)/100)`
- Returns beta for mate scores, raw nullScore otherwise
- **No NMP verification** at deep searches
- Compare to Coda: Coda uses `R = 3 + depth/3 + (eval-beta)/200`, verify at depth>=12. Igel's base R is higher (5 vs 3), depth divisor smaller (6 vs 3). The TT guard is a feature Coda lacks.

**Source verified**: Lines 294-302 confirm R formula and TT guard.

### ProbCut
- Margin: beta + 100, depth >= 5
- **Two-stage QS pre-filter**: runs qsearch first, only does full depth-4 search if QS passes
- SEE filter: `SEE(move) >= betaCut - staticEval`
- Compare to Coda: Coda has ProbCut (disabled by default) with margin 170, gate 85. No QS pre-filter.

**Source verified**: Lines 309-338 confirm two-stage approach.

### LMP
- Table-based thresholds: not-improving `{0,1,2,3,5,9,13,18,25}`, improving `{0,5,7,11,17,26,36,48,63}` for depth 1-8
- Compare to Coda: Coda uses `3+d^2` with improving/failing adjustments. Similar concept, different thresholds.

### CMP / FMP (Countermove / Followup History Pruning)
- CMP: depth <= 3 (not improving) or depth <= 2 (improving), prune if cmhistory below threshold
- FMP: same structure for follow-up move history
- **Not in Coda**: Coda has general history pruning but not per-component CMP/FMP.

### Futility Pruning
- Margin: `90 * depth`, depth <= 8
- **History gate**: only prunes if combined history < threshold (12000 not-improving, 6000 improving)
- Compare to Coda: Coda uses 60+lmrDepth*60. The history gate is a feature Coda lacks.

**Source verified**: Lines 391-396 confirm history gate with `m_fpHistoryLimit[improving]`.

### SEE Pruning
- Quiet: -60*depth, Noisy: -10*depth^2, depth <= 8
- Compare to Coda: Coda uses quiet -20*d^2, capture -d*100.

**Source verified**: Lines 404-412 confirm SEE margins.

---

## 3. Extensions

### Singular Extensions
- Depth >= 8, margin: `ttScore - depth` (1*depth)
- **Double extension**: +2 if `score < betaCut - 50` (non-PV)
- **Multi-cut**: if `betaCut >= beta`, return betaCut directly
- **Negative extension**: -2 when `ttScore >= beta`
- Compare to Coda: Coda has singular extensions with margin `1*depth`, multi-cut (return singular_beta when score >= beta), and negative extension (-1). Coda's singular depth >= 8. Double extension is not in Coda.

**Source verified**: Lines 424-437 confirm SE implementation.

### Check Extension
- +1 unconditionally when in check

### History-Based Extension
- Extend +1 when both cmhist and fmhist > 10000
- **Not in Coda.**

---

## 4. LMR

### Table
- Single table: `0.75 + log(depth) * log(moves) / 2.25`
- Compare to Coda: Coda uses split tables -- captures C=1.80, quiets C=1.30.

**Source verified**: Line 77 confirms the formula.

### Adjustments
- +1 cut node, -2 PV, -1 killer, history-based `/5000`
- Only quiet moves get LMR (no capture LMR)
- No DoDeeper/DoShallower, no improving-based adjustment

### What Coda has that Igel doesn't
- Split LMR tables (capture/quiet)
- Capture LMR
- DoDeeper/DoShallower
- Alpha-raised count scaling
- Failing heuristic in LMR

---

## 5. Move Ordering

All scores computed upfront with selection sort (not staged).

1. TT move: 7,000,000
2. Good captures: MVV-LVA based
3. Killers: 5,000,000
4. Quiets: history + continuation history (plies 1, 2)
5. Bad captures: 1,000,000 + SEE

### History Tables
- Main: `[color][from][to]`, gravity 32/512
- Continuation: plies 1 and 2 only
- **No capture history**, no counter-move heuristic, no pawn history

Compare to Coda: Coda has staged generation, capture history, counter-move, pawn history, and continuation history at plies 1, 2, 4, 6.

---

## 6. Time Management

- Hard limit: `remaining/13 + inc/2`, soft limit: `hard/4` or `hard/6`
- Score-drop: `softLimit *= min(1 + (drop/80), 1.5)`
- Enemy time bonus: uses opponent's time pressure

---

## 7. NNUE Architecture

22528->2x1024->16->32->1 x8 output buckets. 32 king buckets, pairwise mul, dual SCReLU+CReLU activation, output skip connection, material scaling, 50-move eval decay.

Compared to Coda: Coda uses v5/v6/v7 architectures with 16 king buckets, 8 output buckets, SCReLU/CReLU, pairwise mul, Finny tables. Igel's dual activation path and output skip connection are distinctive.

---

## 8. Transposition Table

4-slot buckets, 16 bytes/entry, XOR-verified, modulo indexing, age-based replacement. 50-move safety: skips TT cutoffs at `Fifty() >= 90`.

Compare to Coda: Coda uses 5-slot buckets, XOR-verified, power-of-2 masking. Does not have 50-move safety check.

---

## 9. Lazy SMP

Vote-based best move: each thread votes `(score - worst_score + 20) * depth`, highest-voted move wins. Compare to Coda: Coda uses main-thread-decides approach.

---

## 10. Parameter Comparison Table

| Feature | Igel | Coda |
|---------|------|------|
| RFP margin | 85*(d-improving), d<=8 | improving?70*d:100*d, d<=7 |
| Razoring | eval+150 < alpha, d<=2 | 400+100*d, d<=2 |
| NMP base R | 5 + d/6 | 3 + d/3 |
| NMP eval scale | (bestScore-beta)/100 | (eval-beta)/200 |
| NMP TT guard | Yes | No |
| NMP verification | No | depth >= 12 |
| ProbCut margin | beta + 100 | beta + 170 (disabled) |
| ProbCut QS pre-filter | Yes | No |
| LMR | single table, base 0.75, div 2.25 | split: cap C=1.80, quiet C=1.30 |
| LMR captures | No | Yes |
| LMR history div | /5000 | /5000 |
| LMP depth | <= 8, table-based | 3+d^2, d<=8 |
| Futility margin | 90*d, d<=8 | 60+lmrDepth*60, d<=8 |
| Futility history gate | Yes (12000/6000) | No |
| SEE quiet | -60*d, d<=8 | -20*d^2, d<=8 |
| SEE noisy | -10*d^2, d<=8 | -d*100, d<=6 |
| CMP/FMP | Yes | No |
| SE depth | >= 8 | >= 8 |
| SE beta | ttScore - depth | ttScore - depth |
| SE double ext | +2 at betaCut-50 | No |
| SE multi-cut | Yes | Yes |
| SE neg ext | -2 | -1 |
| History ext | cmhist/fmhist > 10000 | No |
| Aspiration delta | 5 | 15 |
| Cont-hist plies | 1, 2 | 1, 2, 4, 6 |
| Correction history | None | 5 tables (pawn/wp-NP/bp-NP/minor/major) |
| TT buckets | 4-slot | 5-slot |

---

## 11. Ideas Worth Testing from Igel

### Still relevant for Coda:

1. **Mate distance pruning** -- 3 lines, free. Coda still lacks this.

2. **NMP TT guard** -- Skip NMP when TT has fail-high with score < beta. Prevents NMP in positions where TT data contradicts the null-move hypothesis. 1 condition.

3. **CMP/FMP** -- Per-component continuation history pruning with granular thresholds. More precise than Coda's general history pruning.

4. **Futility history gate** -- Don't futility-prune moves with combined history > threshold. Prevents pruning tactically important moves. 2 lines.

5. **History-based extensions** -- Extend when both cmhist and fmhist exceed threshold. 2 lines.

6. **ProbCut QS pre-filter** -- Two-stage ProbCut (QS first, then full search). Could help if Coda re-enables ProbCut.

7. **Aspiration delta=5** -- Much tighter than Coda's 15. More re-searches but tighter bounds.

8. **SE double extension** -- +2 when score < betaCut - 50. Coda only does +1.

9. **NMP base R=5** -- Significantly more aggressive than Coda's R=3. At depth 6: Igel R=6, Coda R=5. At depth 12: both R=7 at eval=beta, but Igel's eval scaling is 2x more aggressive (/100 vs /200).

10. **TT 50-move safety** -- Skip TT cutoffs at Fifty() >= 90. Trivial guard, 1 condition.

11. **Vote-based Lazy SMP** -- Consensus best-move selection. Medium complexity.

12. **Razor at depth<=2, margin 150** -- Much more aggressive than Coda's margins. Could be net-dependent.

### IMPLEMENTED (remove from testing queue):

- **Multi-source correction history** -- Coda now has 5 tables (pawn, white-NP, black-NP, minor, major).
- **Deeper continuation history** -- Coda now reads plies 1, 2, 4, 6.
- **SE multi-cut** -- Coda has this.
- **SE negative extension** -- Coda has -1 (Igel has -2).
- **Aspiration fail-low contraction** -- Coda has this.
- **Score-drop time extension** -- Coda has this.
- **SCReLU / pairwise mul / Finny tables** -- Coda has all of these in NNUE.
