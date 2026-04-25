# Cross-Engine Comparison — Coda vs Top 17 Engines (2026-04-25)

Five parallel research agents surveyed the top 17 engines on the Bullet TC RR (Coda is rank 21 at +62 Elo; gap to top-10 ~120 Elo, gap to #11 Caissa ~90 Elo). Engines surveyed: **Stockfish, Reckless, Obsidian, PlentyChess, Integral, Berserk, Viridithas, Quanticade, Clover, Alexandria, Tarnished, Halogen, Astra, RubiChess, Stormphrax, Horsie, Caissa**. New-to-our-survey engines: **Integral, Clover, Tarnished, Astra, Horsie**.

Five mechanism classes:
1. Move ordering + history tables
2. Pruning gates + reductions
3. Extensions
4. Eval post-processing + NNUE integration
5. Correction history + uncategorized novel

Sources cite `~/chess/engines/<Engine>/src/<file>:<line>` throughout.

`experiments.md` is the source of truth for resolved SPRTs.

## Headline findings — Coda outliers vs 5+ engine consensus

These are where Coda is structurally **different** from a clear consensus across 5+ top engines. Each has high confidence + concrete file:line + a precedented Elo magnitude. Listed in approximate ROI order.

### 1. **`LMR_C_QUIET=120` vs `LMR_C_CAP=93` — captures reduced MORE than quiets**

`src/search.rs:92-93`. **Inverted vs SF, Reckless, Obsidian, Tarnished, Halogen — all of which reduce captures less.** Almost certainly an SPSA detuning artifact (we've had similar inversions before). Reckless cap-additive 1449 < quiet 1806; Obsidian shares table but adds histDiv tilt that softens captures.

Recommended swap: `LMR_C_QUIET=93, LMR_C_CAP=120`, then SPSA. **Expected +6 to +10 Elo.**

### 2. **No triple extension**

`src/search.rs:2553-2560`. **13/14 surveyed engines have it.** Coda jumps from +1 to +2 only. Alexandria's clean formula:

```
if quiet && singular_score < singular_beta - TRIPLE_MARGIN { singular_extension = 3 }
```

`TRIPLE_MARGIN ≈ 75` (Alexandria) is the consensus starting point. Existing `DEXT_CAP` already provides safety. **Expected +3 to +8 Elo.**

### 3. **No stm dimension on main quiet history**

`src/movepicker.rs:27`. Coda's `[ft][tt][from][to]` (4D) mixes white and black history at the same `from-to`. **9/12 surveyed engines have stm.** White's e4-e5 has no relation to black's e4-e5; mixing halves effective resolution. Consensus is unambiguous. **Expected +3 to +8 Elo.**

Memory cost: 8 MB → 16 MB main hist.

### 4. **No symmetric enter-threat penalty in quiet move scoring**

`src/movepicker.rs:608-616`. Coda has escape-bonus only. **All 6 threat-aware engines (SF, Reckless, Plenty, Obsidian, Viri, Clover) apply both:**
- Bonus for piece moving off threatened square (Coda has)
- Penalty for piece moving onto enemy-attacked square (Coda missing)

SF formula: `int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to)); m.value += PieceValue[pt] * v;` — symmetric pair. Coda has only the `from` half.

**Expected +3 to +6 Elo. Single most universal missing pattern across the survey.**

### 5. **Multi-cut is dead code in Coda**

`src/search.rs:2536-2539`. Extra `singular_beta >= beta` gate makes multi-cut almost never fire — `singular_beta = tt_score - margin - kp - xray` rarely exceeds beta. Consensus pattern (SF/Reckless/Stormphrax/PlentyChess) drops the `singular_beta >= beta` clause entirely:

```
if singular_score >= beta && !is_decisive(singular_score) {
    return (2 * singular_score + beta) / 3;  // Reckless
}
```

Cheap fix; uses existing code path. **Expected +1 to +4 Elo standalone, more after retune realigns SE_KING_PRESSURE_MARGIN and DEXT_MARGIN.**

### 6. **Pawn history size 512 — smallest in the field**

`src/movepicker.rs:594`. **Consensus: 1024 (Obsidian/Viri) → 8192 (SF/Plenty) → 16384 (Clover).** Coda is 2-32× smaller than peers. With 64-bit `pawn_hash % 512`, severe overwriting (~10⁶ pawn structures visited per game, 512 buckets). Raise to 8192. Optionally negative-init like Integral (`pawnHistFill = -919` SPSA-tuned).

**Expected +3 to +7 Elo.**

### 7. **Halfmove eval scale `(100-hm)/100` is too aggressive**

`src/search.rs:698-706`. **All five surveyed halfmove-decay engines (Reckless, Stormphrax, Astra, Horsie, Tarnished) use `(200-hm)/200`.** Only SF reaches zero at hm=100, but does so via secondary `r50/199` formula that's gentler at mid-clock. Coda's recent change to `/100` over-decays at hm 60-90 where eval should still be confident. Revert toward `/200` shape.

**Expected +1 to +4 Elo.**

### 8. **Material scaling includes pawns**

`src/search.rs:682`. **SF, Stormphrax, Halogen, Integral all use non-pawn material only.** Coda's pawn-inclusive form `(22400 + material) / 32 / 1024` scales eval to zero too aggressively in pawn-only endgames where eval is still decisive. Switch to npMaterial.

Bonus: the two-step `/32 /1024` truncates twice; fold to single `/32768`.

**Expected +2 to +5 Elo.**

### 9. **Single-row LMP with B=8**

`src/search.rs:97, 2648`. **Consensus is B=3 with two-row improving formula** (Halogen, Reckless, Alexandria, Ethereal). Coda's row at depth 4 imp=false: 24 vs SF's 19. Single-row vs two-row matters most at d=2..4 where pruning volume is highest.

Halved-d² shape with separate improving multiplier:
```
limit = if improving { LMP_BASE_IMP + d²/2 } else { LMP_BASE_NOIMP + d²/2 }
```

**Expected +5 to +8 Elo** (already on next_ideas as T-something, was previously researched).

### 10. **History pruning gate inverts intent**

`src/search.rs:2591`. `!improving && !unstable` gate. **SF/Obsidian/Halogen/Reckless gate hist-prune on none of these conditions** — only `score > worst` and shallow depth. Coda's gates suppress firing in ~50% of nodes hist-prune was meant for. Drop the `!improving && !unstable` clause.

**Expected +3 to +6 Elo.** Combine with FUT_LMR_DEPTH 16→13 and BAD_NOISY_DEPTH 13→8 (both currently 3+ plies past consensus).

## Tier-2 outliers (5+ engine consensus, lower expected Elo)

| # | Item | Coda location | Consensus | Expected |
|---|------|--------------|-----------|----------|
| 11 | No FH-blend depth cap | `search.rs:3190` | Reckless `weight = min(d, 8)`; current Coda formula approaches `score` at depth 25+ | +1 to +3 |
| 12 | NMP post-capture R++ uses "previous move was capture" | `search.rs:2203-2205` | Obsidian `R += ttMoveNoisy` (signal is "TT move is strong noisy", structurally cleaner) | +2 to +4 |
| 13 | Recapture extension unconditional | `search.rs:2696-2702` | Consensus is **no recapture ext** (Caissa is only other one with PV+ttMove gate) | +1 to +3 |
| 14 | No `ttPv` tracking in SE | `search.rs:2497-2520` | SF/Reckless/Obsidian/Viri all add `ttPv` term to margin or depth gate | +1 to +3 |
| 15 | No threat-bit on captHist | `src/movepicker.rs:32` | Reckless/Stormphrax/Clover/Viri/Quanticade have it (the **single defended-bit** is consensus-light, not the 6D expansion on stop list) | +2 to +5 |
| 16 | Cont-hist plies 1,2,4,6 weights 3:3:1:1 | `movepicker.rs:584` | Plenty 2:1:1:½, Stormphrax 1,1,½ (3 plies), Integral 1,2,4 (no ply-6); Coda's flat weights + ply-6 inclusion are outliers | +1 to +3 |
| 17 | Bonus formula single linear for all categories | `movepicker.rs` | Plenty/Clover/Viri use separate base/factor/max for bonus AND malus per category | +2 to +5 |

## Novel mechanisms (1-3 engines have, Coda doesn't)

These are higher-variance bets — fewer engines have them but they're well-validated where present.

### N-1. **Factorized main history** (Reckless / Viridithas / Stormphrax)

Three top engines split main history into a baseline factorizer + a context-specific bucket:

- Reckless: `factorizer[stm][from][to]` (cap 1852) + `bucket[stm][from][to][ft][tt]` (cap 6324). `Reckless/src/history.rs:14-66`.
- Viridithas: `piece_to_hist[ft][tt][piece][to]` averaged with `from_to_hist[ft][tt][from][to]`. `viridithas/src/movepicker.rs:213-216`.
- Stormphrax: butterfly + pieceTo averaged at `Stormphrax/src/history.h:144-146`.

Three-engine consensus = real signal. **+5 to +10 Elo. Higher implementation cost than items 1-10 (two tables to update, averaged read).**

### N-2. **Pawn-push attacker-graduated bonus** (Viridithas)

`viridithas/src/movepicker.rs:225-253`. When our pawn moves to a square defended by another pawn AND attacks an enemy piece, score graduates by victim type (10k king → 1k pawn). **Same family as Coda's discovered-attack +52, knight-fork +5.2, offense +5.7 wins.** Eval-free tactical motif.

**+2 to +5 Elo.** Fits Coda's W1 winning pattern; high confidence.

### N-3. **Threat-bucketed continuation correction history** (PlentyChess)

`PlentyChess/src/search.cpp:558`. Cont-corr indexed by `[stm][piece][to][threat(from)][threat(to)]`. **Of all engines surveyed only PlentyChess does this.** Directly intersects Coda's threat-input agenda — main history's threat-aware key won big; corr-hist is the natural follow-on.

**+2 to +4 Elo. Highest expected of corrhist-only items.**

### N-4. **TTMoveHistory** (Stockfish)

`Stockfish/src/history.h:216`, `Stockfish/src/search.cpp:607,1437`. Global StatsEntry tracking how often the TT move is actually `bestMove`. Updated `bestMove == ttMove ? +805 : -787` post-search. Used in singular-extension margin (`search.cpp:1161`) and triple-margin. Tiny code addition (~8 KB), zero NPS cost.

**+1 to +3 Elo.** Cheap to test; cleanly orthogonal to existing mechanisms.

### N-5. **LowPlyHistory** (Stockfish)

`Stockfish/src/history.h:139`, `Stockfish/src/movepick.cpp:179`. `[ply][move]` history table for early plies (ply<5), blended into MovePicker as `8 * lph[ply][m] / (1+ply)`. Targets the part of the tree that affects all subtrees.

**+1 to +3 Elo.** Mid-Elo gain in SF history. Cleanly orthogonal.

### N-6. **Parent cutoff_count signal** (Stockfish / Clover / Quanticade)

`Stockfish/src/search.cpp:1225`, `Reckless/src/search.rs:830-832`. Track `(ss+1)->cutoffCnt`; when child has many cutoffs, current node's LMR adjusts. Reckless: `td.stack[ply+1].cutoff_count > 2 → r += 1515`. Three engines have it; Coda has no equivalent parent-cutoff tracker.

**+1 to +3 Elo.**

### N-7. **Variance-driven aspiration window** (SF / Reckless / Viridithas)

Initial delta `= base + avg² / divisor` (SF: `5 + threadIdx%8 + abs(meanSqScore)/10208`; Reckless: `15 + avg²/25833`). Asymmetric growth (`delta += 28*delta/128` low, `62*delta/128` high). Coda uses constant initial.

**+1 to +3 Elo.** Proportional to score volatility tightens windows when the engine is sure.

### N-8. **SF small ProbCut TT-trust shortcut**

`Stockfish/src/search.cpp:1003-1006`. After moves_loop label, if `tt_flag == LOWER && tt_depth >= depth-4 && tt_score >= beta+416`, return `beta+416` immediately. Pure TT-trust shortcut, ~zero NPS cost. Coda has plain ProbCut but not this shortcut.

**+2 to +5 Elo.**

### N-9. **Reckless adaptive ProbCut depth**

`Reckless/src/search.rs:622-633`. `probcut_depth = base − (score - pcβ) / 319`; re-search at `adjusted_beta`. Higher qsearch surplus → less depth needed. Coda uses fixed depth-4.

**+1 to +3 Elo.**

### N-10. **Cont-hist gravity with combined base modulator** (Stormphrax / Viridithas)

`Stormphrax/src/history.h:120`. When updating cont-hist, gravity modulator is `getConthist(...) + getMainHist(...)/2`, not just the entry being updated. Drives correlation between main hist and cont hist.

**+1 to +3 Elo. Trivial code change.**

### N-11. **Source-specific corrhist update boosting** (Stockfish)

`Stockfish/src/search.cpp:111-114`. `bonus * 153/128` minor, `bonus * 187/128` NP at *write* time, separate from read weight. Effectively per-source learning rate. Coda uses one shared `CORR_HIST_LIMIT = 1024` across all 5 sources.

**+1 to +2 Elo.** Cheap to add; combine with retune.

### N-12. **Cont-hist split by `[in_check][capture]`** (Reckless)

`Reckless/src/history.rs:7,156-167`. Quadruples cont-hist storage. Past moves from in-check positions update a different table than out-of-check positions. Obsidian has partial `[isCap]` only at `Obsidian/src/history.h:21`.

**+2 to +5 Elo, but higher memory cost.**

### N-13. **King-wall-pawn malus** (Reckless)

`Reckless/src/movepick.rs:226`. `-4000 * wall_pawns.contains(mv.from())` where wall_pawns are pawns adjacent to a castled king. Discourages king-safety-degrading pushes. Eval-free tactical motif.

**+1 to +3 Elo.** Same family as Coda's tactical motif wins.

### N-14. **PlentyChess 16-cp eval quantisation**

`PlentyChess/src/evaluation.cpp:49`. `(eval / 16) * 16` after material+halfmove scaling. TT-stability trick — more eval-ties = more TT hits across slightly-different paths. One line.

**0 to +2 Elo.** Minimal cost.

### N-15. **CloverEngine combined material+halfmove single-multiply**

`CloverEngine/src/evaluate.h:24-34`. `scale = EvalScaleBias + Σ(piece × seeVal)/32 - hm × EvalShuffleCoef`. Folds halfmove into material scale; one multiplication, one tunable. Coda does two separate multiplies (truncates twice).

**+1 to +3 Elo.** Cleaner pipeline.

### N-16. **Stormphrax stm-asymmetric NP corrhist weights**

`Stormphrax/src/correction.h:85-87`. Different read weight for stm-side and non-stm-side NP corrhist. Coda symmetric. **+0 to +2 Elo, ~free.**

### N-17. **Integral score-bound guard on corrhist update**

`integral/.../correction_history.h:119-126`. Only update corr-hist when score-bound is consistent with eval. Reduces noise in corrhist learning. **+0 to +2 Elo.**

### N-18. **TT cluster size 5 vs consensus 3**

`src/tt.rs`. SF, Reckless, Obsidian all use 3-slot. Coda is the only 5-slot in the survey. May trade extra collisions for cache footprint pressure (each 64-byte bucket holds 5 × 12B entries — same cache line, but more keys to scan per probe). Worth A/B testing 5 → 3.

**+0 to +3 Elo, possibly small NPS gain.**

### N-19. **Stockfish 2-fold-as-draw within ply**

`Stockfish/src/position.cpp:390-403`. Treat 2-fold rep as instant draw if the repetition occurred within `ply` plies of root. Coda requires 3-fold. Some engines (Stormphrax-leaning) cite this as load-bearing.

**Mixed signal across engines (only SF clearly has it). +1 to +3 Elo, but disrupts repetition semantics — needs careful gating.**

## Ranked queue for Hercules

Suggested order; each line is "branch name → expected Elo → key change":

**Tier 1 — high confidence consensus fixes (likely +20-40 Elo aggregated):**

1. `experiment/lmr-c-swap` → +6 to +10 → swap `LMR_C_QUIET=93, LMR_C_CAP=120`
2. `experiment/triple-extension` → +3 to +8 → Alexandria pattern with `TRIPLE_MARGIN=75`
3. `experiment/main-history-stm-dim` → +3 to +8 → add stm to main quiet history 4D→5D
4. `experiment/enter-threat-penalty` → +3 to +6 → symmetric to escape bonus
5. `experiment/lmp-two-row` → +5 to +8 → halved d² + improving multiplier separated
6. `experiment/pawn-history-8192` → +3 to +7 → 512→8192, optionally with `-919` init
7. `experiment/multicut-fix` → +1 to +4 → drop `singular_beta >= beta` clause
8. `experiment/halfmove-200-revert` → +1 to +4 → `(200-hm)/200` with bench A/B
9. `experiment/material-np-only` → +2 to +5 → `(BASE + npMaterial) / 32768`
10. `experiment/hist-prune-gate-drop` → +3 to +6 → drop `!improving && !unstable` + tighten depth caps

**Tier 2 — novel single-source mechanisms (likely +10-25 Elo aggregated):**

11. `experiment/factorized-main-hist` → +5 to +10 → Reckless/Viri pattern, baseline + bucket
12. `experiment/pawn-push-attacker-graduated` → +2 to +5 → Viridithas tactical motif
13. `experiment/threat-bucketed-cont-corr` → +2 to +4 → Plenty pattern
14. `experiment/sf-small-probcut` → +2 to +5 → SF TT-trust shortcut
15. `experiment/optimism-full-p1` → +3 to +7 → K1+K2+optBase+divisor SPSA on Stormphrax-shape

**Tier 3 — small wins, low risk (each +1-3, but cheap):**

16. `experiment/ttmove-history` → +1 to +3 → SF global stat
17. `experiment/lowply-history` → +1 to +3 → SF early-ply table
18. `experiment/parent-cutoff-count` → +1 to +3 → SF/Clover/Quanticade
19. `experiment/variance-aspiration` → +1 to +3 → SF/Reckless/Viri
20. `experiment/conthist-combined-modulator` → +1 to +3 → Stormphrax/Viri
21. `experiment/fh-blend-depth-cap` → +1 to +3 → `min(d, 8)` weight
22. `experiment/eval-quantisation-16cp` → 0 to +2 → Plenty pattern
23. `experiment/clover-combined-mat-hm` → +1 to +3 → single-multiply
24. `experiment/sf-source-bonus-boost` → +1 to +2 → SF write-time scaling

## Cross-cutting observations

- **Coda's tunable space has SPSA-detuning artifacts** — `LMR_C_QUIET > LMR_C_CAP` (item 1) is structurally inverted vs every consensus engine and almost certainly an artifact of partial-feature tuning. The fix-then-retune cycle should expose more such inversions.
- **The "consensus H0 = dig deeper" lesson applies broadly.** From CLAUDE.md: capture history magnitude was +31.6 Elo when fixed despite 3 prior H0s. Several items here (single-row LMP, no triple extension, pawn-history 512) are the same shape — Coda has tried adjacent variants and H0'd, but the consensus-form has not been tried.
- **Tactical-motif scoring** keeps paying off. Items N-2 (pawn-push attacker), N-13 (king-wall-pawn malus) extend the W1 family of wins (B1 +52, knight-fork +5.2, offense +5.7).
- **Threat-aware mechanisms have headroom.** Item 4 (enter-threat penalty) and N-3 (threat-bucketed cont-corr) are direct extensions of work that already paid Elo. Plenty has the most threat-bucketed mechanisms and is at rank 4 on the RR.
- **Cumulative envelope across Tier 1 alone is +20-40 Elo before retune** per CLAUDE.md's +6-8 retune-per-3-merges pattern. Realistically: 8 merges × landed-conditional-on-Elo + 3 retunes ≈ +30-50 Elo.

## Companion docs

- `next_ideas_2026-04-21.md` — prior round of ideas; intersect on T1.x tactical motifs, T2.x threat signals.
- `peripheral_mechanisms_2026-04-22.md` — non-pure-search mechanisms (P1 optimism is item N-15-ish here).
- `correctness_audit_2026-04-25.md` — fresh audit findings; some interact (Coda's `SEE_CAP_MULT` dead tunable was found there).
- `research_threads_2026-04-24.md` — R2 LMP/NMP threads pre-figured items 1, 5.
- `experiments.md` — source of truth for resolved SPRTs.

## Stop-trying list (verbatim — mostly carried forward)

- Don't try another dynamic-SEE-threshold variant without changing the `score` input (must be captHist-only, not MVV-dominated).
- Don't try the **6D captHist expansion** with from+to AND threat bits (the single defended-bit / threat-bit is OK; full 6D is on stop list).
- Don't try threat-magnitude as LMR-amount / extension-amount.
- Don't try another margin widener for a signal that already has a gate landed (RFP/LMP king-pressure proved this).
- Don't add "generic piece-attacked" nudges (use specific tactical motifs).
- **NEW:** don't expand `SEE_CAP_MULT`'s use without first confirming it's wired in (it's currently dead code per the 2026-04-25 audit).
- **NEW:** don't re-port Reckless's NMP TT-capture skip as binary; use the dynamic R++ form instead (item 12).
