# RubiChess Engine - Crib Notes

Source: `~/chess/engines/RubiChess/`
Author: Andreas Matthies
Version: 2026 (git)
NNUE: Dual architecture -- V1 (HalfKP 40960->256x2->32->32->1) and V5 (HalfKAv2_hm, 1024x2->16+16x32->1, 8 output/PSQT buckets)
Strength: #6 in RR at +247 Elo (GoChess-era gauntlet; 355 above GoChess-v5)

---

## 1. NNUE Architecture

### V5 (Default, SFNNv5 compatible)
- HalfKAv2 with horizontal mirroring, 22528 input features, 32 king buckets
- FT: 22528 -> 1024 (configurable)
- L1: 1024 -> 16 (neuron 15 is bypass/"fwd" output via scaled direct contribution)
- Dual activation on L1: SqrClippedReLU on first 15 + ClippedReLU on first 15 = 30-dim L2 input
- L2: 30 -> 32, Output: 32 -> 1
- 8 output buckets + 8 PSQT buckets by piece count
- PSQT blending: `((128-7)*psqt + (128+7)*eval) / 128 / 16`
- Accumulator cache (FinnyTable equivalent)

### NNUE-Classical Hybrid
- Falls back to classical eval when `|PSQT_eval| >= 760` (extreme material imbalance)
- Phase-scaled: `score * (116 + phcount) / 128`
- Adds tempo (20cp)

Compare to Coda: Coda uses v5/v6/v7 with 16 king buckets, 8 output buckets, SCReLU/CReLU, pairwise mul, Finny tables. RubiChess's dual activation path and bypass neuron are distinctive. Coda is NNUE-only (no classical fallback).

---

## 2. Search Architecture

Standard iterative deepening, PVS, Lazy SMP with skip-depth pattern (Laser/Ethereal style). Best thread selection: picks thread with highest score among those that completed to main thread's depth.

### Aspiration Windows
- Initial delta: 15 (after depth > 4)
- Fail-low: beta contraction `beta = (alpha + beta) / 2`
- Fail-high: widen beta
- Growth: `delta += delta/4 + 2` (~1.25x + 2)
- Root LMR modulated by aspiration window state: `inWindowLast - 1`

**Source verified**: RubiChess.h confirms `aspinitialdelta=15`. Search.cpp lines 1143 confirm root LMR `+ inWindowLast - 1`.

### Mate Distance Pruning
- Standard MDP (template-gated by `Pt == MatePrune`)
- **Not in Coda.** Trivial to add.

**Source verified**: Lines 420-428.

---

## 3. Pruning Techniques

### RFP
- Depth <= 8, margin: `depth * (39 - 4*improving)`
- **Threat guard**: `POPCOUNT(threats) < 2` -- skips RFP when 2+ pieces threatened
- Returns raw staticeval
- Compare to Coda: Coda uses improving?70*d:100*d, d<=7. Coda has no threat guard. RubiChess's margin is much smaller (39 vs 70-100).

**Source verified**: RubiChess.h: `futilityreversedepthfactor=39`, `futilityreverseimproved=4`. Search.cpp line 591.

### Razoring
- Depth <= 2, margin: `360 + 67*depth` (depth 1: 427, depth 2: 494)
- Depth 1: full alpha/beta QS; depth 2: null-window QS with confirmation
- Compare to Coda: Coda uses 400+100*d. Very similar margins.

**Source verified**: RubiChess.h: `razormargin=360`, `razordepthfactor=67`. Lines 558-576.

### Threat Pruning (Koivisto-inspired)
- Depth == 1, `!threats`, `staticeval > beta + (improving ? 3 : 29)`
- Returns beta
- **Not in Coda.** Very targeted depth-1-specific RFP variant.

**Source verified**: Line 580.

### Futility Pruning (Forward)
- Margin: `12 + 80*depth`, d<=8
- Sets `futility = true`, per-move application on quiet non-check moves
- Compare to Coda: Coda uses 60+lmrDepth*60. Very similar.

**Source verified**: RubiChess.h: `futilitymargin=12`, `futilitymarginperdepth=80`.

### NMP
- Conditions: `!isCheckbb && !threats && depth >= 4`
- **Threat guard**: `!threats` (disables NMP when any piece is threatened)
- R = `min(depth, 3 + depth/5 + (eval-beta)/125 + 2*(!PVNode))`
- Uses `bestknownscore` (hashscore if available, else staticeval)
- **Verification at depth >= 11**: SF-style with `nullmoveply` and `nullmoveside` to prevent repeated NMP
- Compare to Coda: Coda uses R=3+d/3+(eval-beta)/200, verify at d>=12. RubiChess has threat guard and PV factor (2*!PVNode) that Coda lacks. RubiChess's eval divisor (125) is more aggressive.

**Source verified**: RubiChess.h: `nmmredbase=3`, `nmmreddepthratio=5`, `nmmredevalratio=125`, `nmmredpvfactor=2`, `nmverificationdepth=11`. Lines 602-628.

### LMP
- Table-based: not-improving `2.5 + 0.43*d^0.62`, improving `4.0 + 0.70*d^1.68`
- Compare to Coda: Coda uses `3+d^2`. RubiChess is more nuanced with tunable exponents.

**Source verified**: RubiChess.h: `lmpf0=43`, `lmppow0=62`, `lmpf1=70`, `lmppow1=168`. Lines 52-54.

### SEE Pruning
- `seeprunemarginperdepth * depth * (tactical ? depth : seeprunequietfactor)` = `-14*d*(d for tactical, 3 for quiet)`
- Quiet: -42*d, Tactical: -14*d^2
- Compare to Coda: Coda uses quiet -20*d^2, capture -d*100.

**Source verified**: RubiChess.h: `seeprunemarginperdepth=-14`, `seeprunequietfactor=3`. Line 731.

### ProbCut
- Depth >= 7, margin: beta + 108
- Two-stage: QS first, then full search at depth-4
- Compare to Coda: Coda has ProbCut (disabled) with margin 170, depth >= 5. RubiChess's tighter margin and QS pre-filter are notable.

**Source verified**: RubiChess.h: `probcutmindepth=7`, `probcutmargin=108`. Lines 635-659.

### QSearch Delta Pruning
- Position-level: `staticeval + 389 + bestCapture < alpha`
- Per-move: `staticeval + materialvalue + 389 <= alpha`
- Compare to Coda: Coda uses delta 240. RubiChess's is much larger (389).

**Source verified**: RubiChess.h: `deltapruningmargin=389`.

---

## 4. Extensions

### Check Extension
- +1, guarded by extension budget (`extensionguard < 3 * 16`)
- Max 3 check extensions per branch
- When not extending for check, computes threats instead

**Source verified**: RubiChess.h: `extguardcheckext=3`. Lines 526-536.

### Singular Extensions
- Depth >= 8, margin: `hashscore - 1*depth`
- **Double extension** (+2): `redScore < sBeta - 17`, limited by `(extensionguard & 0xf) <= 7`
- **Multi-cut**: `bestknownscore >= beta && sBeta >= beta` => return sBeta
- **Negative extension** (-1): `hashscore >= beta`
- Extension guard: unified budget tracking check + double SE extensions
- Compare to Coda: Coda has SE at depth >= 8, margin 1*depth, multi-cut, neg ext (-1). Coda lacks double extension with limiter and extension guard.

**Source verified**: RubiChess.h: `singularmindepth=8`, `singularmarginperdepth=1`, `singularmarginfor2=17`, `extguarddoubleext=7`. Lines 748-788.

### Endgame Capture Extension
- +1 when `phcount < 6 && capture >= knight`
- **Not in Coda.**

**Source verified**: Lines 790-793.

### History Extension (Auto-Tuning Threshold)
- Extend when both `conthistptr[ply-1]` and `conthistptr[ply-2]` exceed `he_threshold`
- **Self-calibrating**: every 4M nodes, adjusts threshold to maintain 0.003%-0.2% extension rate
- **Not in Coda.** Unique auto-tuning mechanism.

**Source verified**: Lines 800-821, RubiChess.h: `histextminthreshold=9`, `histextmaxthreshold=15`.

---

## 5. LMR

### Tables
Two separate tables for improving vs not-improving:
- Not improving: `1 + round(log(d * 1.31) * log(m) * 0.41)`
- Improving: `round(log(d * 1.52) * log(m*2) * 0.30)`

**Source verified**: RubiChess.h: `lmrlogf0=131`, `lmrf0=41`, `lmrlogf1=152`, `lmrf1=30`, `lmrmindepth=3`. Lines 41-47.

### Adjustments
1. +1 cutnode (quiets only)
2. History: `-= stats / (908 * 8)` = `-= stats / 7264`
3. -1 PV node
4. -1 PV node with good hash
5. **-1 for opponent's high move count** (`CurrentMoveNum[ply-1] >= 24`)
6. **Fail-high count feedback**: if `failhighcount[ply] < 4`: `-= (5 - failhighcount) / 2`
   - 0 fail-highs: -2, 1: -2, 2: -1, 3: -1
   - Updated: `+= (!hashmovecode + 1)` (2 for surprising cutoff, 1 for expected)

**Source verified**: RubiChess.h: `lmrstatsratio=908`, `lmropponentmovecount=24`. Lines 830-867.

Compare to Coda: Coda has split tables (cap/quiet), failing heuristic, alpha-raised count. Coda lacks opponent move count adjustment, fail-high count feedback, and the double PV reduction.

---

## 6. Move Ordering

Staged: HASHMOVE -> TACTICAL -> KILLER1 -> KILLER2 -> COUNTER -> QUIET -> BAD_TACTICAL

### History Tables
- **Threat-aware main history**: `history[2][65][64][64]` -- color, threat_square (0-64), from, to
  - Index 0-63 = square of first threatened piece, 64 = no threats
  - Finer granularity than boolean approaches
- Continuation history: plies -1, -2, -4 for scoring; updated at offsets 0, 1, 3; read at -4 and -6 at half weight for ordering
- Tactical history: `[7][64][6]` (attacker_type, to, victim_type)
- Counter-move: `[14][64]` (piece, to)
- Bonus: depth^2, capped at 400, gravity `32*delta - entry*|delta|/256`

**Source verified**: Lines 122-156 confirm threat-aware history with `threatSquare`, cont-hist read at plies -1, -2, -4, updated at offsets 0, 1, 3.

Compare to Coda: Coda uses flat butterfly history (not threat-indexed), counter-move table, cont-hist at plies 1, 2, 4, 6. The 65-value threat index is a unique approach vs boolean alternatives.

---

## 7. Correction History

3 tables: pawn + white non-pawn + black non-pawn.
- Non-pawn keys: Zobrist-based, incrementally updated, per-color
- Weight: `min(1 + depth, 16)`
- Update: exponential moving average with divisor 256
- Clamped to [-8192, 8192]
- Division ratios: pawn 102, non-pawn 102

**Source verified**: RubiChess.h: `pawncorrectionhistoryratio=102`, `nonpawncorrectionhistoryratio=102`. Lines 184-207.

Compare to Coda: Coda has 5 tables (pawn, white-NP, black-NP, minor, major) with weighted combination. Coda's is more comprehensive. Both use per-color non-pawn Zobrist keys.

---

## 8. Transposition Table

3-entry buckets, 10 bytes/entry, 32 bytes/cluster. Power-of-2 masking. Huge page support. Age-based replacement: `depth - age_penalty * 2`. Replace if: exact bound, different hash, or `new_depth + 3 >= old_depth`.

Compare to Coda: Coda uses 5-slot buckets, 64 bytes, lockless XOR-verified atomics.

---

## 9. Time Management

Dual limits (soft endtime1, hard endtime2). Phase-aware allocation blending material and move number. Best-move node ratio: `128 * (2.5 - 2 * bestNodes/totalNodes)`. Score drop: halves `constantRootMoves`, indirectly extending time.

Compare to Coda: Coda has simpler allocation. No node-based ratio, no phase-aware allocation.

---

## 10. Notable Unique Techniques

1. **Threat-aware history (65-value index)** -- actual threatened square, not boolean
2. **Fail-high count feedback in LMR** -- child cutoff count adjusts parent LMR
3. **Self-tuning history extension threshold** -- auto-calibrates every 4M nodes
4. **Extension guard (unified budget)** -- single integer tracking check + double SE extensions
5. **Threat guard on NMP and RFP** -- NMP: `!threats`; RFP: `threats < 2`
6. **NNUE-classical hybrid** with PSQ threshold
7. **Opponent move count in LMR** -- `-= (CurrentMoveNum[ply-1] >= 24)`
8. **Root LMR modulated by aspiration window state**

---

## 11. Parameter Comparison Table

| Feature | RubiChess | Coda |
|---------|-----------|------|
| RFP margin | d*(39-4*imp), d<=8 | improving?70*d:100*d, d<=7 |
| RFP guard | threats < 2 | none |
| Razoring | 360+67*d, d<=2 | 400+100*d, d<=2 |
| NMP R | 3+d/5+(eval-beta)/125+2*!PV | 3+d/3+(eval-beta)/200 |
| NMP guard | !threats | none |
| NMP verification | depth >= 11 | depth >= 12 |
| ProbCut margin | beta + 108, d>=7 | beta + 170, d>=5 (disabled) |
| LMR tables | improving/not-improving | cap/quiet split |
| LMR history div | 7264 | 5000 |
| LMR cutnode | +1 (quiets only) | +1 |
| LMR fail-high feedback | yes | no |
| LMR opponent movecount | yes (-1 at >=24) | no |
| LMP | table, tunable exponents | 3+d^2 |
| Futility | 12+80*d, d<=8 | 60+lmrDepth*60, d<=8 |
| SEE quiet | -42*d | -20*d^2 |
| SE depth | >= 8 | >= 8 |
| SE beta | hashscore - 1*depth | ttScore - depth |
| SE double ext | +2, margin 17, limit 7 | No |
| SE multi-cut | Yes (return sBeta) | Yes |
| History table | threat-aware (65 indices) | flat butterfly |
| Cont-hist plies | -1,-2,-4 (read -4,-6 half) | 1, 2, 4, 6 |
| Correction history | 3 tables (pawn + 2x NP) | 5 tables |
| Aspiration delta | 15 | 15 |
| Delta pruning | 389 | 240 |

---

## 12. Ideas Worth Testing from RubiChess

### Still relevant for Coda:

1. **Fail-high count feedback in LMR** -- `-(5 - failhighcount) / 2` reduction based on child cutoff count. Novel child-to-parent feedback. Clean implementation.

2. **Threat guard on NMP and RFP** -- NMP: disable when threats exist. RFP: disable when 2+ pieces threatened. Prevents over-pruning in tactical positions.

3. **Threat-aware history (65-value index)** -- Finer granularity than boolean approaches. RubiChess uses actual threatened square as index. 65x table expansion per color.

4. **Self-tuning history extension threshold** -- Auto-calibrates to maintain target extension rate. Eliminates a hard-to-tune parameter.

5. **Extension guard (unified budget)** -- Prevent explosive tree growth from unbounded extensions. Critical for SE double extensions.

6. **SE double extension** -- +2 when `score < sBeta - 17`, limited to 7 per branch.

7. **Opponent move count LMR adjustment** -- `-= (opponentMoveNum >= 24)`. Cheap, 1 line.

8. **Threat pruning at depth 1** -- Koivisto-inspired. `!threats && eval > beta + margin` at depth 1 only.

9. **Endgame capture extension** -- +1 for knight+ captures with < 6 pieces. Simple.

10. **Mate distance pruning** -- 3 lines, free. Coda still lacks this.

11. **Root LMR modulated by aspiration state** -- Interesting coupling.

### IMPLEMENTED (remove from testing queue):

- **Multi-source correction history** -- Coda has 5 tables (more than RubiChess's 3).
- **Deeper continuation history** -- Coda has plies 1, 2, 4, 6. RubiChess reads -4 and -6 at half weight.
- **SE multi-cut** -- Coda has this.
- **SE negative extension** -- Coda has -1.
- **Aspiration fail-low beta contraction** -- Coda has this.
- **Score-drop time extension** -- Coda has this.
- **SCReLU / pairwise mul / Finny tables** -- Coda has all.
- **NMP verification** -- Coda has at depth >= 12.
- **ProbCut** -- Coda has this (disabled by default due to ablation results).
- **Failing heuristic** -- Coda has this.
- **Alpha-raised count in LMR** -- Coda has this.
- **DoDeeper/DoShallower** -- Coda has this.
