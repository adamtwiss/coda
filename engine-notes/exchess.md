# EXchess 7.97b - Engine Analysis Notes

Source: `~/chess/engines/exchess/src/`
Author: Daniel C. Homan, 1997-2017. GPL licensed.
Board: mailbox 8x8 array (`sq[64]`), piece lists (`plist[side][ptype][index]`), no bitboards for move generation.

---

## 1. Search Architecture

- **PVS with iterative deepening**, aspiration windows (delta=15, widened on fail by 1.5x the overshoot).
- Optional MTD(f) (`USE_MTD=1`, granularity=8), but PVS is the default.
- **Fail-high depth reduction trick** (Robert Houdart, CCC): on aspiration fail-high, the root re-search is done at `max_ply - 1 - fail_high` (i.e., one ply shallower to resolve fail-highs faster).
- Max main tree depth: 80 plies. Total (including qsearch): 100.
- Hash table: 4-slot buckets, lockless XOR verification (Hyatt/Mann style). Depth-priority replacement with age check.
- Separate PV hash: PV nodes XOR the hash key with a constant `h_pv`, so PV entries are stored separately and don't collide with zero-window entries. **Double-store**: PV nodes store to both the normal and PV-modified key.

---

## 2. Pruning Techniques

### 2a. Null Move Pruning
- **Conditions**: not PV, not in check, not after null move, `premove_score > beta`, no hash move available, opponent's hash move != last move played, pieces[stm] > 1, beta > -(MATE/2), ply < MAXD-3.
- **Also requires**: `null_hash == 1` (set to 0 if hash returned FLAG_A or FLAG_P with score < beta, preventing null move when hash suggests we're unlikely to beat beta).
- **Reduction R**:
  - Default R = 3
  - R = 5 if `premove_score > beta + 400` and `depth >= 6`
  - R = 4 if `premove_score > beta + 100` and `depth >= 5`
  - Every 4th thread (`(ID+1) % 4 == 0`): R = depth/2 (aggressive null move for thread diversity)
- **No verification search** -- just returns beta on cutoff.
- **Threat detection from null move**: if null move fails low and the refutation was a capture, the capture target square is recorded as `threat_square`. Also records `threat_check` if the null-move refutation delivered check. These threat squares are used later to exempt threat-evasion moves from pruning/reduction.

### 2b. Futility Pruning
- **Margin formula**: `MARGIN(x) = 50 + 100*x*x` where x = max(0, depth+depth_mod)
  - depth 1: 150
  - depth 2: 450
  - depth 3: 950
- **Conditions**: not PV, `depth + depth_mod < 4`, and `premove_score + MARGIN + pawn_bonus + captured_piece_value + promotion_value < alpha`
- `pawn_bonus` = 35 * game_stage if capturing a pawn on the 7th/2nd rank
- Applied only to moves with score < 2,000,000 (i.e., not killers, safe checks, etc.)

### 2c. Search Abort (Late Move Pruning / Move Count Pruning)
- **Depth guard**: depth < 7
- Uses a precomputed `abort_search_fraction[8]` table indexed by `depth + (premove_score - beta) / 100`, clamped to [0,7]:
  - `{0, 128, 406, 625, 717, 808, 900, 1100}` (divided by 1000)
  - Index 0: abort after 0% of moves (always abort)
  - Index 7: abort after 110% (never abort, since > 100%)
- Aborts entire move loop (returns alpha) if `mcount > fraction * moves.count / 1000`
- **Exception**: if a threat square was identified from null move, individual moves are skipped instead of aborting the whole loop, until a threat-evasion move is reached.
- Only applies to non-killer, non-check, non-pawn-push-7th moves (score < 2,000,000).
- Not applied during singular extension searches (`move_to_skip != 0`).

### 2d. No explicit RFP (Reverse Futility Pruning)
- EXchess does NOT have a standalone reverse futility / static null move pruning step. The null move itself serves a similar role since it requires `premove_score > beta`.

### 2e. No explicit Razoring
- No razoring step is present.

### 2f. No explicit ProbCut
- No ProbCut implementation.

### 2g. No explicit SEE Pruning on Quiets
- SEE (called `swap()`) is used for move ordering (to classify captures as winning/losing) and to validate safe checks, but there is no standalone SEE-based quiet pruning in the move loop.

---

## 3. Extensions

### 3a. Singular Extension
- **Root**: depth > 7, hash move exists, pieces > 1, alpha > -MATE/2, no fail-low/fail-high in progress, more than 1 root move. Search at `depth/2` with window `[g_last - 25 - 1, g_last - 25]`. If score < test_score, extend first move by 1 ply.
- **Interior nodes**: `depth > max(7, 2*sqrt(max_ply))` (dynamic depth threshold based on iteration depth), hash move exists, last move was real (not null/IID), hash score > alpha, hash flag is FLAG_B or FLAG_P (not upper-bound only), pieces > 1, ply+depth < MAXD-3. Test at `depth/2` with margin 25 (or 75 at ply==2 for "easy move" detection). The move being tested is excluded via `move_to_skip`.
  - Additional filter: only runs for every 3rd thread OR in PV/near-PV nodes OR after opponent's hash move OR non-capture hash move.
- Singular status is stored in the hash table (encoded in the move's type bits).

### 3b. Check Extension
- **Single reply to check**: if `moves.count == 1` and `pos.check`, extend 1 ply.
- **Check giving move**: if `next.pos.check` and (PV or near-PV node or king vulnerability `qchecks[stm] > 0`) and move score > 1,000,000 (i.e., killer/good-capture tier), extend 1 ply.
- **Root**: check extension only when depth < 3.

### 3c. No Passed Pawn Extension
### 3d. No Recapture Extension

---

## 4. Move Ordering

### 4a. Stages / Scoring (all computed at move generation time, no lazy staging)
All moves generated at once, scored during `add_move()`, then picked with selection sort (`MSort`). Late move generation: if a hash move exists and not in check, only the hash move is tried first; full movegen happens at move index 1.

**Score tiers** (from highest to lowest):
1. **Hash move**: 50,000,000
2. **Queen promotion**: 20,000,000 (other promotions: 11,000,000 / 10,999,940 / 10,999,930)
3. **Winning/equal captures** (MVV-LVA or SEE >= -50): 10,000,000 + 1000*victim_type - attacker_type + pawn_bonus
   - Also includes losing captures with discovered check (via `slide_check_table`)
   - +50 threshold on SEE allows minor piece exchanges (e.g., BxN)
   - Bishop pair bonus added when capturing one of two bishops
4. **Combination move (cmove)**: adds 9,000,000 bonus to any move that matches
5. **Reply/counter move**: 8,000,000
6. **Killer 1**: 6,000,000
7. **Killer 2**: 4,000,000
8. **Killer 3**: 2,000,000
9. **Safe checks** (not in capture tier): 2,000,000 (or 12,000,000 if qchecks[stm] > 2, meaning king is in danger)
10. **Safe pawn push to 7th** (when few enemy pieces and swap >= 0): 2,000,000
11. **Quiet history**: `history[piece_id][to_square]` (can be negative)
12. **Losing captures**: score = 0

### 4b. History Table
- **Dimensions**: `history[15][64]` -- indexed by piece ID (0-14) and to-square.
- **Update on beta cutoff (non-capture, non-check, non-checking move)**: `+= 11 * (depth + depth_mod)` (only when depth+depth_mod > 1, capped at 1,000,000)
- **Update on fail-low (non-capture, non-check)**: `-= (depth + depth_mod)` (capped at -1,000,000)
- **Asymmetric**: gains are 11x losses per depth. This means a move that fails high once at depth 5 needs ~11 fail-lows at depth 5 to become negative.
- **Aging**: halved (`/= 2`) at the start of each new search (unless pondering).

### 4c. Killer Moves
- **3 killers** per side-to-move (not per ply! -- `killer1[wtm]`, `killer2[wtm]`, `killer3[wtm]`).
- Updated as a FIFO: new killer pushes killer1 -> killer2 -> killer3.
- Only updated for non-check positions.
- Cleared at search start.

### 4d. Counter (Reply) Move
- `reply[piece_id][to_square]` -- indexed by the piece/square of the **opponent's last move**.
- Single move per slot, always-replace.
- Only stored for non-capture beta-cutoff moves that weren't already the counter move.

### 4e. Combination Move Hash (Novel!)
- A dedicated hash table (`cmove_table`, ~25% of non-main-hash memory) that stores refutation moves keyed by the XOR of position hashes 2 plies apart.
- Key: `plist[turn+ply-3] XOR current_hcode`, with side-to-move correction.
- **2-layer depth-priority table**: each entry has a primary and secondary slot. New entries replace by depth.
- Stores moves that caused beta cutoffs, excluding: forced check responses, winning captures (victim >= attacker value).
- Scored at 9,000,000 + base score in move ordering.
- Similar to "Last Best Reply" (Steven Edwards) but keyed by position hash difference rather than move pair.

### 4f. Root Move Ordering
- Best move (from TT or PV) placed first.
- Remaining moves sorted by aggregated history across all threads.

---

## 5. LMR (Late Move Reductions)

### 5a. Interior Node LMR
**Conditions**: not first move, not in check, next position not in check, no extension, not a threat-evasion move, `best > -(MATE/2)`, and the relaxed "not early PV" condition:
  - `!in_pv || pos.hmove != pc[0][ply] || best >= alpha`
  - Plus: `premove_score < alpha + mcount^2 * 200` (progressive widening -- very unusual! At move 5, allows premove_score to be up to alpha + 5000 before entering the reduction zone)

**Base reduction**: -1 ply for all moves scored < 2,000,000 (non-killer, non-safe-check, etc.)

**Additional reductions based on static eval**:
- If `premove_score < alpha - 350`: additional `-(depth/5)` plies

**Additional reductions based on history score**:
- If `history_score < -10 * depth`: additional `-(1 + depth/12)` plies
- If `history_score < -23 * depth`: further `-(1 + depth/8)` plies on top
- Example at depth 24: base -1, then if history bad: -1 -(24/12)= -3, then if very bad: -1 -(24/8) = -4, total = -8 plies

**No reduction formula/table** (like log*log). It's a step function: either -1 or -1 plus history/eval-based extras.

**Killers, safe checks, pawn push to 7th (score >= 2,000,000)**: Not reduced, but still subject to futility pruning.

**Re-search on fail-high**: if reduced search returns > alpha, re-search at full depth (depth_mod = 0) with full window if PV node.

### 5b. Root LMR
- **Conditions**: depth > 4, not in check, `best > g_last - 85` (NO_ROOT_LMR_SCORE), not first move, not singular, next not in check, not the "next best" move from singular search, move score < 2,000,000, `best > -(MATE/2)`.
- **Base**: -1 ply
- **History-based extras**: same thresholds as interior (-10*depth, -23*depth), same additional reductions.

---

## 6. Internal Iterative Deepening (IID)

- **Condition**: no hash move, depth > 3, not during singular extension search.
- **Also requires**: in_pv OR prev was PV node OR opponent's hash move == last move OR no last move (first move of search).
- **Depth**: `depth - 3` with the same alpha/beta window.
- If IID returns > alpha, its best move becomes the hash move for the main search.
- Inherits `depth_mod` from parent node.

---

## 7. Quiescence Search

- **Stand pat**: `premove_score` used as initial best score (when not in check).
- **Delta pruning**: `delta_score = alpha - best - 50` (used in capture generation to skip captures that can't reach alpha even with the captured piece value).
- **Capture generation with futility filtering**: the `captures()` function takes `delta_score` and skips captures where `victim_value < delta_score` (minus piece-specific adjustments).
- **Checks in qsearch**: generated only when:
  - First ply of qsearch (`qply == 0`) AND `qchecks[stm] > 0` (king danger indicator from eval) AND `alpha - best < 650` (NO_QCHECKS)
  - OR continuation of forced check sequence: `qply <= 2` AND previous position was in check with only 1 move available
- **Check evasions**: full move generation when in check, but only on `qply == 0`; at `qply > 0`, returns static eval clamped to [alpha, beta].
- **Hash probing**: yes, at `hdepth >= -1` (which means qsearch results are stored at depth -1).
- **Draw detection**: 3-fold rep and 50-move rule checked at qply == 0 only.

---

## 8. Time Management

### 8a. Time Allocation
- `limit` = base time per move (from time control calculation)
- `max_limit` = min(8 * limit, max(limit, timeleft / 4))
  - For non-xboard non-TC: `max_limit = timeleft`
  - Capped by `max_search_time * 100`
  - For xboard with TC: `limit = min(max_limit / 2, limit)`

### 8b. Time Extension on Fail-Low (Root)
- If first move fails low and `score <= g_last - EXTEND_TIME_SCORE * (time_double + 1)` where EXTEND_TIME_SCORE = 15:
  - Double `limit` (up to `max_limit / 2`)
  - Requires at least limit/4 time already spent
  - Up to 3 doublings (since fail-low window expands iteratively)
- The scaling `15 * (time_double + 1)` means: first extension at -15 score drop, second at -30, third at -45.

### 8c. Time Reduction on Singular Move
- If first move is singular AND `score > alpha`: halve `limit` (down to `max_limit / 16`).
- This is the "easy move" mechanism.

### 8d. Abort Decision
- At each iteration: abort if `elapsed >= limit` (after at least 3 plies).
- Within search: abort if `elapsed >= min(2 * limit, max_limit)`.
- Check interval dynamically adjusted: targets checking every ~10% of remaining time.

### 8e. Short Time Panic Mode
- When timeleft < min(3 * time_cushion, max_margin) where:
  - `min_cushion` = min(base_time, 600 centiseconds)
  - `time_cushion` = max((mttc + 3) * average_lag, min_cushion)
  - `max_margin` = min(base_time * 10, 6000 centiseconds)
- Forces `limit = max(1, timeleft - 2 * time_cushion)` and disables time extensions.

### 8f. Aspiration Window Interaction
- Fail-low: `root_alpha = max(-MATE, g + 1.5 * (root_alpha - g_last))`
- Fail-high: `root_beta = min(+MATE, g + 1.5 * (root_beta - g_last))`
- If window exceeds +/- 500 from last score, widen to full [-MATE, MATE].

---

## 9. SMP (Shared Memory Parallel)

- **Approach**: shared root move list with move skipping (not Lazy SMP, not ABDADA).
- All threads search the same root position with the same depth (no depth variation across threads, though there's commented-out code for `ADDED_DEPTH`).
- **Work sharing**: at root and PV nodes, each thread publishes its current move via `share_data[ply][thread_id]`. Other threads skip moves being actively searched by siblings and add them to a "skipped" list to revisit later.
- All threads share the hash table, pawn hash, score hash, and cmove hash (all lockless).
- Per-thread: search nodes, PV array, killers, history, reply table, counters.
- History aging at search start is per-thread.
- First completed thread's result replaces the main thread's result if it searched deeper or the main thread timed out.

---

## 10. Evaluation

- Classical HCE (no NNUE).
- 4-stage piece-square tables (opening, early-middle, late-middle, endgame) blended by `AVERAGE_SCORE(early, late)` using game stage (0-15 based on total non-pawn pieces).
- Pawn hash table with pawn structure scores, pawn attack bitboards, passed pawn flags.
- Score hash table caches full eval.
- King safety: tropism-based + attack count, scaled by `attack_scale[16]` lookup table.
- `qchecks[side]` flag from eval indicates king vulnerability; used to enable check extensions and qsearch checks.
- Piece values: P=100, N=377, B=399, R=596, Q=1197, K=10000.
- Side-to-move bonus: 10 (opening), 0 (endgame).

---

## 11. Novel / Unusual Features

### 11a. Combination Move Hash Table
The most distinctive feature. A separate hash table keyed by `hash(pos_2_plies_ago) XOR hash(current_pos)` stores refutation moves. This captures the idea: "given the last two moves played (by both sides), what was the best reply?" It's a positional generalization of the counter-move heuristic -- while counter-moves key on a single (piece, to-square), this keys on the full position change across two plies. Uses 2-layer depth priority.

### 11b. Threat-Aware Pruning
Null move failure identifies a `threat_square` (the target of the refuting capture). Later in the move loop, moves that defend this square (by moving the threatened piece to a safe square, or capturing on the threat square with a good capture) are exempted from LMR and LMP. After a reasonable evasion is tried, the exemption expires. This is a lightweight version of "threat move" detection.

### 11c. Threat Check Detection
If the null move refutation delivered check and the opponent's king is vulnerable (`qchecks > 0`), moves that evade by moving the king are given special treatment.

### 11d. Progressive LMR Gating
The condition `premove_score < alpha + mcount^2 * 200` means that at high move counts, even positions where static eval is well above alpha can enter the reduction zone. At move count 1 (second move), this is `alpha + 200`; at move 10, it's `alpha + 20,000` (essentially always). This progressively tightens the reduction exemption as we go deeper into the move list.

### 11e. History Asymmetry
History increments (+11*depth) are much larger than decrements (-depth), making the history table biased toward remembering successes. A single beta cutoff at depth 10 adds +110, which takes 110 fail-lows at depth 1 to cancel out.

### 11f. Per-Side Killers (not Per-Ply)
Only 3 killers per side-to-move total, not per-ply. This means the killers are global across the tree for each side. Unusual -- most engines use per-ply killers.

### 11g. Score Hash / Eval Cache
A dedicated eval cache (`score_table`) stores full evaluation scores keyed by position hash. Avoids recomputing eval when the same position is reached multiple times.

### 11h. Knowledge Scaling for Skill Levels
At knowledge_scale < 50, the engine probabilistically skips moves based on a hash-derived random condition. This creates "oversights" that weaken play naturally.

### 11i. No Bitboards
Pure mailbox with piece lists. Uses precomputed 64-bit lookup tables (`check_table`, `knight_check_table`, `slide_check_table`, `bishop_check_table`, `rook_check_table`) for fast check detection, but move generation is loop-based.

---

## 12. Features We Have That EXchess Lacks

- NNUE evaluation
- Bitboard move generation (magic bitboards)
- Staged move picker (lazy generation)
- Reverse futility pruning (static NMP)
- Razoring
- ProbCut
- SEE pruning on quiets
- Continuation history
- Capture history
- Per-ply killers
- Passed pawn extensions
- Recapture extensions
- History-based LMP (we use move count)
- Correction history

## 13. Features EXchess Has That We Might Not

- **Combination move hash table** -- a positional 2-ply counter-move keyed by hash difference. Worth investigating, though our continuation history may serve a similar role.
- **Threat square from null move** -- exempting threat-evasion moves from pruning. We could test: after null move fails low, if the refutation is a capture, exempt moves that defend that square from LMR/LMP.
- **Progressive LMR gating** via `premove_score < alpha + mcount^2 * 200` -- interesting alternative to "improving" flag.
- **History-based deep reductions** -- the `-10*depth` and `-23*depth` thresholds for additional 1+depth/12 and 1+depth/8 reductions are more aggressive than typical log-based LMR.
- **Per-side (not per-ply) killers** -- simpler, might be worth testing.
- **Score hash (eval cache)** -- separate from TT, avoids eval recomputation.
- **Aspiration fail-high depth reduction** (Houdart trick) -- searching at `depth - 1` on aspiration fail-high to resolve it faster.
