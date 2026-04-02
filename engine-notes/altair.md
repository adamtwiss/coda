# Altair Chess Engine - Crib Notes

Source: `~/chess/engines/AltairChessEngine/`
Author: Alexander Tian
Version: 7.2.1
RR rank: #11 at +112 Elo (GoChess-era gauntlet; 220 above GoChess-v5)
NNUE: (768x5 -> 1024)x2 -> 1x8 (SCReLU, 5 king buckets, 8 output buckets)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Lazy SMP with shared TT (not lockless -- potential data races). C++ with `search_parameters.h` exposing all tunable params with SPRT-style ranges (OpenBench-compatible).

### Aspiration Windows
- Enabled at depth >= 6
- Initial delta: `max(6 + 85/(asp_depth-2), 9)` -- starts ~49 at depth 6, shrinks over iterations
- Fail-low: widen alpha, contract beta: `beta = (3*alpha + 5*beta) / 8`
- Fail-high: widen beta, **reduce depth by 1**
- Delta growth: multiply by ~1.4 each re-search
- Compare to Coda: Coda uses fixed delta=15. The depth-adaptive delta and fail-high depth reduction are absent from Coda.

**Source verified**: search_parameters.h confirms all parameter values.

### Draw Detection
- Two-fold repetition, fifty-move rule
- **Draw score randomization**: `3 - (node_count & 8)` (between -5 and 3 to avoid draw-seeking)
- Compare to Coda: Coda uses fixed draw score. Randomization is not in Coda.

### Mate Distance Pruning
- Standard: `alpha = max(alpha, -MATE_SCORE + ply)`, `beta = min(beta, MATE_SCORE - ply)`
- **Not in Coda**. Trivial to add, 3 lines.

**Source verified**: Lines 613-618 of search.cpp confirm MDP.

---

## 2. Pruning Techniques

All forward pruning gated by: `!pv_node && !in_check && !singular_search && abs(beta) < MATE_BOUND`.

### Reverse Futility Pruning
- Depth <= 7, margin: `110 * (depth - improving)`
- **Returns `(eval + beta) / 2`** (score dampening)
- Compare to Coda: Coda uses improving?70*d:100*d, d<=7, returns raw eval. The score dampening is not in Coda.

**Source verified**: Lines 705-708 confirm margin=110 and `(eval + beta) / 2` return.

### NMP
- Min depth: 1 (extremely low)
- Depth-dependent activation gate: `static_eval >= beta + 128 - (11 + 3*improving) * depth`
- Material guard: `non_pawn_material >= 1 + (depth >= 10)`
- R = `5 + depth/4 + clamp((eval-beta)/400, -1, 3)`
- **Verification at depth > 15**: full `negamax(beta-1, beta, depth-R)` with do_null=false
- Compare to Coda: Coda uses R=3+d/3+(eval-beta)/200, verify at depth>=12. Altair's base R=5 is significantly more aggressive.

**Source verified**: Lines 713-746 confirm NMP formula and verification at depth <= 15 returns early, else verifies.

### IIR
- `depth -= 1 + pv_node` -- **extra -1 on PV nodes**. Novel.
- Depth >= 4
- Compare to Coda: Coda does -1 only, at depth >= 6.

**Source verified**: Lines 750-751.

### LMP -- Two tiers
**Tier 1**: `legal_moves >= depth * 10`, d<=4, breaks all moves
**Tier 2 (quiet)**: `legal_moves >= 2 + d^2 / (1 + !improving + failing)`
- The `failing` flag tightens LMP for deteriorating positions
- Compare to Coda: Coda uses `3+d^2` with improving/failing adjustments. Very similar to tier 2.

**Source verified**: Lines 857-864 confirm both tiers with `LMP_depth=4`, `LMP_margin=10`, `LMP_margin_quiet=2`.

### Futility Pruning
- Margin: `(depth - !improving) * 170 + 70`, d<=7
- Triggers break (prunes ALL remaining moves)
- Compare to Coda: Coda uses 60+lmrDepth*60, per-move continue. Altair's margins are much wider.

**Source verified**: Lines 868-871 confirm `FP_multiplier=170`, `FP_margin=70`, `FP_depth=7`.

### History Pruning
- Threshold: `(depth + improving) * -9600`, d<=10
- Applies to quiets and losing captures
- Compare to Coda: Coda uses -1500*depth, d<=3. Altair's threshold is much larger (includes cont-hist sum) and goes deeper.

**Source verified**: Lines 874-877 confirm `history_pruning_divisor=9600`, `history_pruning_depth=10`.

### SEE Pruning
- Quiet: `-40*depth`, Noisy: `-85*depth`
- **History gate**: only prune by SEE when `move_history_score <= 5800`
- Depth range varies: base 2, +0 noisy, +5 PV
- Compare to Coda: Coda uses quiet -20*d^2, capture -d*100, d<=8/d<=6. No history gate.

**Source verified**: Lines 880-888 confirm SEE parameters and history gate.

---

## 3. Extensions

### Singular Extensions
- Depth >= 6, margin: `ttScore - depth`
- **Double extension**: +2 when `score < singular_beta - 12`, limited to 7 per branch
- **Multi-cut**: return score when `score >= beta`
- **Negative extension**: -1 when `ttScore >= beta || ttScore <= alpha`
- Compare to Coda: Coda has SE at depth >= 8, multi-cut, negative extension (-1). Coda lacks double extension with limiter.

**Source verified**: Lines 916-955 confirm `SE_base_depth=6`, `SE_dext_margin=12`, `SE_dext_limit=7`.

### Passed Pawn Extension
- +1 for passed pawns reaching 7th rank (when `move_score >= 0`)
- +1 for queen promotions
- **Not in Coda.**

### Extension Capping
- `extension = min(2, MAX_DEPTH - 1 - depth)` -- prevents over-extension.

---

## 4. LMR

### Table
- Single table: `log(d)*log(m)/2.30 + 1.40`
- Compare to Coda: Coda has split tables (cap C=1.80, quiet C=1.30).

**Source verified**: search_parameters.h: `LMR_divisor_quiet=230`, `LMR_base_quiet=140`.

### Adjustments
- -1 noisy, -1 PV, -1 tt_pv, -1 improving, +1 failing
- -1 "interesting" (passed pawn, queen promo, gives check, is killer)
- -1 in_check
- History: `-= move_history_score / 10000`
- **Alpha-raised count**: `+= alpha_raised_count * (0.5 + 0.5 * tt_move_noisy)`
- +1 cutnode
- Compare to Coda: Coda has failing heuristic and alpha-raised count (both implemented). Coda's history divisor is 5000 (more aggressive). Coda lacks tt_pv, "interesting" catch-all flag, and in_check adjustments.

**Source verified**: Lines 992-1025 confirm all LMR adjustments.

### DoDeeper / DoShallower
- `+1` if `return_eval >= best_score + 80`
- `-1` if `return_eval < best_score + new_depth`
- Compare to Coda: Coda has DoDeeper/DoShallower (implemented).

**Source verified**: Lines 1045-1046 confirm `deeper_margin=80`.

---

## 5. Move Ordering

Staged generation with MaxHeap optimization for expected-all-nodes (PV or exact TT).

1. TT move: 500,000
2. Good captures (SEE >= -85): capture_history + MVV-LVA
3. Queen promotions: 100,000
4. Killers: 30,000 / 25,000
5. Castling: 1,200
6. Quiets: **threat-aware butterfly history** + continuation history (3 plies)
7. Bad captures: base(-3000) + MVV-LVA + capture_history

### History Tables
- **Threat-aware butterfly**: `history_moves[12][64][2][2]` -- piece, target, from_threatened, to_threatened
- Capture history: `[2][12][12][64]` -- winning_capture_flag, attacker, victim, target (2-tier)
- Continuation history: plies 1, 2, and **4** (`LAST_MOVE_PLIES = {1, 2, 4}`)
- **No counter-move table**
- History bonus: `depth * (depth + 1 + null_search + pv_node + improving) - 1` with `+ (alpha >= eval)` for surprising cutoffs

**Source verified**: Lines 345-351 confirm threat-aware history indexing. Lines 124-129 confirm continuation history structure.

Compare to Coda: Coda uses flat butterfly history (not threat-indexed), has counter-move table, continuation history at plies 1, 2, 4, 6.

---

## 6. Correction History

4 tables: pawn, non-pawn, major, minor. Dedicated hash keys per table. Weight: `min((1+depth)^2, 144)`. All corrections summed equally.

**Source verified**: Lines 93-94 show `correction_history_np`, `_major`, `_minor` tables. Lines 131-155 confirm update formula.

Compare to Coda: Coda has 5 tables (pawn, white-NP, black-NP, minor, major) with weighted combination (384/154/154/204/104 out of 1024). Coda's is more refined.

---

## 7. Time Management

Position-phase scaling: material + pawn structure analysis.
- With inc: `remaining/20 + inc*0.75`, scaled by position factor (0.2-1.2)
- Node-based soft time scaling: `(1.35 - best_node_pct) * 1.80`
- Score-based: asymmetric -- drops extend more than gains
- Hard limit: `time_amt * 2.81`, soft limit: `time_amt * 0.78`

Compare to Coda: Coda uses simpler `timeLeft/movesLeft + 80% inc`, cap at 50% remaining. No node-based or position-based scaling.

---

## 8. Transposition Table

Single-entry per index, 24 bytes. Power-of-2 masking. **Not lockless** (shared between SMP threads without atomics). PV flag stored and used in replacement (+2 depth bonus for PV entries).

Compare to Coda: Coda uses 5-slot buckets, lockless XOR-verified atomics. No PV flag in TT.

---

## 9. NNUE

(768x5 -> 1024)x2 -> 1x8, SCReLU, 5 king buckets, 8 output buckets. No Finny tables. Eager incremental updates.

Compare to Coda: Coda has v5/v6/v7 with 16 king buckets, Finny tables, lazy accumulator, AVX2/AVX-512 SIMD.

---

## 10. Parameter Comparison Table

| Feature | Altair | Coda |
|---------|--------|------|
| RFP margin | 110*(d-improving), d<=7 | improving?70*d:100*d, d<=7 |
| RFP return | (eval+beta)/2 | raw eval |
| NMP base R | 5 | 3 |
| NMP depth divisor | 4 | 3 |
| NMP eval divisor | 400 | 200 |
| NMP verification | depth > 15 | depth >= 12 |
| LMR table | single, div=2.30, base=1.40 | split cap/quiet C=1.80/1.30 |
| LMR history div | 10000 | 5000 |
| LMP tier 1 | depth*10, d<=4 | N/A |
| LMP tier 2 | 2+d^2/(1+!imp+fail) | 3+d^2 with failing |
| Futility | (d-!imp)*170+70, d<=7 | 60+lmrDepth*60, d<=8 |
| History pruning | -9600*(d+imp), d<=10 | -1500*d, d<=3 |
| SEE quiet | -40*d | -20*d^2 |
| SEE history gate | <= 5800 | No |
| SE depth | >= 6 | >= 8 |
| SE beta | ttScore - depth | ttScore - depth |
| SE double ext | +2 (margin 12, limit 7) | No |
| SE multi-cut | Yes | Yes |
| SE neg ext | -1 (ttScore>=beta or <=alpha) | -1 |
| IIR | depth-=1+pvNode, d>=4 | depth-=1, d>=6 |
| Aspiration delta | depth-adaptive (49 to 9) | fixed 15 |
| ContHist plies | 1, 2, 4 | 1, 2, 4, 6 |
| Correction tables | 4 (pawn/np/major/minor) | 5 (pawn/wNP/bNP/minor/major) |
| History tables | threat-aware [2][2] | flat butterfly |
| TT | single entry, 24B, PV flag | 5-slot, 64B, no PV flag |

---

## 11. Ideas Worth Testing from Altair

### Still relevant for Coda:

1. **Mate distance pruning** -- 3 lines, free. Coda still lacks this.

2. **Threat-aware butterfly history** -- `[piece][to][from_threatened][to_threatened]`. 4x table size. Altair is another confirming engine for this widely-used technique.

3. **NMP base R=5** -- Much more aggressive than Coda's R=3. Combined with activation gate, could be net positive. Test R=4 first.

4. **SE double extension with limiter** -- +2 when `score < singular_beta - 12`, max 7 per branch. The limiter prevents explosive tree growth.

5. **SEE pruning history gate** -- Skip SEE pruning when `history > 5800`. Prevents over-pruning tactically important moves. 1 condition.

6. **IIR extra reduction on PV** -- `depth -= 1 + pv_node`. Aggressive for PV without TT move.

7. **RFP score blending** -- Return `(eval+beta)/2`. Note: GoChess-era test was -16.7 Elo, but formula may have differed.

8. **TT PV flag** -- Store PV node status in TT, use for replacement priority (+2 depth) and LMR (-1 for tt_pv). Medium complexity.

9. **Position-phase time scaling** -- Material + pawn structure to allocate time. Coda's time management is simple.

10. **Node-based time scaling** -- Scale soft limit by best-move node percentage. Standard in strong engines, not in Coda.

11. **Draw score randomization** -- `3 - (nodes & 8)` to avoid draw-seeking. Trivial.

12. **Passed pawn extension** -- +1 for passed pawns on 7th rank. Coda has recapture extensions but not passed pawn.

13. **Aspiration depth-adaptive delta** -- Starts wide, shrinks over iterations. More resilient early, tighter late.

14. **LMR in_check adjustment** -- -1 reduction when in check (position volatile). Trivial.

### IMPLEMENTED (remove from testing queue):

- **Multi-source correction history** -- Coda has 5 tables (more than Altair's 4).
- **Continuation history ply 4** -- Coda has plies 1, 2, 4, 6.
- **Failing heuristic** -- Coda has this in LMR/LMP.
- **Alpha-raised count in LMR** -- Coda has this.
- **DoDeeper/DoShallower** -- Coda has this.
- **SE multi-cut** -- Coda has this.
- **SE negative extension** -- Coda has -1.
- **Aspiration fail-low contraction** -- Coda has this.
- **SCReLU activation** -- Coda supports SCReLU in NNUE.
- **Score-drop time extension** -- Coda has this.
