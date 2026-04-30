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

## Empirical cross-validation — `reckless_vs_coda_pruning_diff_2026-04-25.md`

The instrumented Reckless-vs-Coda diff (`docs/reckless_vs_coda_pruning_diff_2026-04-25.md`, run 2026-04-25) measured pruning fire rates and NMP cutoff rate at depth 12 on matched per-position node counts. **It inverts a prior CLAUDE.md framing** ("Coda under-prunes vs Reckless") and validates several of the cross-engine findings in this doc:

| Metric | Coda | Reckless | Ratio |
|---|---:|---:|---:|
| First-move-cut rate | 81.7% | 82.85% | -1.15pp Coda |
| **NMP cutoff rate** | **30%** | **57%** | **0.53× — Reckless 1.9× more efficient** |
| **LMP fires / Kn** | 573.9 | 20.4 | **28× more in Coda** |
| Futility (incl BNFP) / Kn | 707.0 | 138.2 | **5.1× more in Coda** |
| SEE prunes / Kn | 573.8 | 189.5 | **3.0× more in Coda** |

**Key reframing.** Coda doesn't *under*-prune; it fires shallow-pruning gates 3-28× more often per Kn than Reckless. Two compounding causes:

1. Worse move ordering (1.15pp FMC gap) → more moves reach the pruning gates, so each gate fires more often.
2. Some Coda gates are configured to fire deeper / on more conditions — directly explaining the 28× LMP gap (consensus-vs-Coda findings #9, #10 below).

**The empirical NMP cutoff-rate gap (30% → 57%) is the single highest-leverage open lever** at ~2× efficiency delta. The cleanest mechanism for closing it is Reckless's `cut_node` gate on NMP — promoted from "noted divergence" to **Tier-1 top item** in the queue below.

**Implications for the queue ordering:**
- **Move-ordering items lead** — they fix the cause, not the symptom (every FMC point compounds across the tree).
- **Pruning-gate-rate fixes** (LMP two-row, hist-prune gate, depth caps) keep their position in the queue and their +Elo magnitude — they treat measurable over-firing — but are framed as symptom fixes.
- **`cutoff_count` propagation** (was a Tier-3 novel mechanism) promotes to Tier 2 — Reckless uses it in both NMP and LMR; load-bearing for the empirical efficiency gap.
- **Methodological caveat:** Reckless bench is 51 positions vs Coda's 8. Per-position node counts (~63K-67K) are matched, but the Coda 8 may not exercise the same regime mix. Magnitudes (28×, 5×, 3×) too large to invalidate; porting the same `dbg_hit` instrumentation to Coda + running matched bench locks in exact magnitudes.

## Headline findings — Coda outliers vs 5+ engine consensus

These are where Coda is structurally **different** from a clear consensus across 5+ top engines. Each has high confidence + concrete file:line + a precedented Elo magnitude. Listed in approximate ROI order.

### 1. **`LMR_C_QUIET=120` vs `LMR_C_CAP=93` — captures reduced MORE than quiets**

`src/search.rs:92-93`. **Inverted vs SF, Reckless, Obsidian, Tarnished, Halogen — all of which reduce captures less.** Almost certainly an SPSA detuning artifact (we've had similar inversions before). Reckless cap-additive 1449 < quiet 1806; Obsidian shares table but adds histDiv tilt that softens captures.

Recommended swap: `LMR_C_QUIET=93, LMR_C_CAP=120`, then SPSA. **Expected +6 to +10 Elo.**

> **Tested 2026-04-26 (#774): H0 −9.3 ±7.2 / 1.6K** — but the H0 surfaced
> the actual finding. **Structural diagnosis**: cap LMR had a binary ±1
> captHist threshold at ±2000 while quiet LMR uses continuous
> `reduction -= hist_score / LMR_HIST_DIV`. The asymmetry was why SPSA
> had detuned C_CAP downward — captures got effectively no history-
> driven LMR shaping, forcing C_CAP to compensate. Obsidian's "histDiv
> tilt softens captures" notes the same primitive.
>
> **Follow-up (the actual lever):** added `LMR_CAP_HIST_DIV` (default
> 1024) so captures get continuous shaping like quiets — branch
> `experiment/capture-lmr-hist-adjustment` (#780 H0 −0.5 at default
> tunable, retune-needed-prior). SPSA **#791** finished
> (LMR_CAP_HIST_DIV 1024→1287 +25.7%, CAP_HIST_BASE +7%). Post-tune
> SPRT **#809: H0 −1.2 / 28.5K**. Bucket: signal-not-there — the
> structural diagnosis was correct (asymmetry exists) but continuous
> shaping for captures alone doesn't carry Elo even with retune.
> Drop. Possible refined retry: histDiv-tilt form (Obsidian) where
> the same table feeds quiet AND capture LMR with a tilt term, not
> separate divisors.

### 2. **No triple extension**

`src/search.rs:2553-2560`. **13/14 surveyed engines have it.** Coda jumps from +1 to +2 only. Alexandria's clean formula:

```
if quiet && singular_score < singular_beta - TRIPLE_MARGIN { singular_extension = 3 }
```

`TRIPLE_MARGIN ≈ 75` (Alexandria) is the consensus starting point. Existing `DEXT_CAP` already provides safety. **Expected +3 to +8 Elo.**

> **Tested 2026-04-26 (#787): H0 −1.0 ±1.7 / 30.7K**. Initial bucket:
> retune-needed-prior. SPSA **#792** (3 params, 1000 iters) on branch
> finished with tiny movements only — TRIPLE_MARGIN 75→78 (+3.5% *),
> DEXT_MARGIN 10→10 (flat), DEXT_CAP 16→15 (−3.9% *). All single-star,
> no basin found. Compare #790 (NMP_UNDEFENDED_MAX −26%) or #796
> (NMP_EVAL_MAX −57%) for what a retune-needed cluster looks like.
>
> **Reclassified bucket: signal-not-there.** Triple extension's
> incremental gain over Coda's existing DEXT (margin 10, cap 16)
> appears to be ~zero — DEXT already fires on the same cases. **Drop
> branch**, do not re-SPRT post-tune. Possible refined retry: change
> the trigger condition (gate on `tt_pv`, `cut_node`, or depth) rather
> than re-tuning the 3 magnitudes.

### 3. **No stm dimension on main quiet history**

`src/movepicker.rs:27`. Coda's `[ft][tt][from][to]` (4D) mixes white and black history at the same `from-to`. **9/12 surveyed engines have stm.** White's e4-e5 has no relation to black's e4-e5; mixing halves effective resolution. Consensus is unambiguous. **Expected +3 to +8 Elo.**

Memory cost: 8 MB → 16 MB main hist.

### 4. **No symmetric enter-threat penalty in quiet move scoring**

`src/movepicker.rs:608-616`. Coda has escape-bonus only. **All 6 threat-aware engines (SF, Reckless, Plenty, Obsidian, Viri, Clover) apply both:**
- Bonus for piece moving off threatened square (Coda has)
- Penalty for piece moving onto enemy-attacked square (Coda missing)

SF formula: `int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to)); m.value += PieceValue[pt] * v;` — symmetric pair. Coda has only the `from` half.

**Expected +3 to +6 Elo. Single most universal missing pattern across the survey.**

> **Tested 2026-04-26**: vanilla form #773 H0 −0.4 ±4.1 / 5.2K (stopped
> early). Refined v2 with split tunables for from/to magnitudes #781
> H0 +0.1 ±1.0 / 94.7K. Bucket: signal-overlap — Coda's threat-aware
> main history (4D `[ft][tt][from][to]`) already captures most of the
> "moving onto attacked square is bad" signal we expected the explicit
> penalty to add. Drop unless threat-history dimensionality changes
> (e.g. if main-history-stm-dim or factorized-main-hist lands).

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

> **Tested 2026-04-26 (#785): H0 −1.2 ±1.8 / 27.7K** at 8192. Bisection
> follow-up #797 at 2048 also H0'd, **worse**: −3.3 ±3.3 / 8.3K (LLR
> −3.05). 512 → 2048 → 8192 = -3.3 → -1.2 — non-monotonic, no smooth
> Elo gradient in the 512-8192 range. Bucket: signal-not-there for
> Coda's regime. Possible refined retry: `pawnHistFill = -919` SPSA-
> tuned negative init (Integral pattern) at default 512 size — that
> changes the table's prior, not its size. Low priority.
>
> **Lesson noted**: bisection isn't always a rescue. Sometimes the
> consensus value's Elo comes from an adjacent mechanism we lack
> (e.g., negative-init), not the size itself.

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

> **Phased Reckless alignment in flight 2026-04-26** — see
> `docs/lmp_reckless_alignment_plan_2026-04-26.md`. Empirical 28× LMP
> firing gap motivated a 7-difference enumeration + phased plan.
> **Phase A merged**: #808 lmp-direct-check-carveout (+2.54),
> #810 lmp-skip-quiets (+1.86). Combined +4.4 Elo banked.
> **Phase B running**: #811 SPSA on `experiment/lmp-reckless-shape`
> (formula change to Reckless history-aware continuous-improvement
> form, 4 new tunables: LMP_K_BASE/IMP/DEPTH/HIST, 80-param full
> sweep). Phase C (drop !is_pv gate, drop depth cap) deferred until
> Phase B retune lands.

### 10. **History pruning gate inverts intent**

`src/search.rs:2591`. `!improving && !unstable` gate. **SF/Obsidian/Halogen/Reckless gate hist-prune on none of these conditions** — only `score > worst` and shallow depth. Coda's gates suppress firing in ~50% of nodes hist-prune was meant for. Drop the `!improving && !unstable` clause.

**Expected +3 to +6 Elo.** Combine with FUT_LMR_DEPTH 16→13 and BAD_NOISY_DEPTH 13→8 (both currently 3+ plies past consensus).

> **Tested 2026-04-26 (#786): trending H0 −0.2 ±1.3 / 49.5K (LLR −2.50)**.
> Drop-both-clauses form. Bucket: signal-overlap — the existing gates
> co-fire with futility/LMP catches in Coda's tree, so dropping them
> doesn't expose new prunable nodes. Possible bisection: drop ONE
> clause at a time (`!improving` only, `!unstable` only) to find which
> gate is actually load-bearing. Not queued; lower priority than the
> retune-needed batch.

## Tier-2 outliers (5+ engine consensus, lower expected Elo)

> **Status (2026-04-30):**
> - **Item 11 (FH-blend depth cap):** H0 #814 -0.3 ±1.3 / 55.3K. Drop.
> - **Item 14 (`ttPv` in SE):** H0 #821 -0.1 ±1.2 / 68.3K — signal-overlap with merged DEXT decomposition (#817). Drop.

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

## Status overview (2026-04-30 update)

**Tier-1 batch resolution: most resolved.** Of the 10 rows below: 7 H0, 1 merged (material-np-only), 1 LMP-shape split (Phase A merged, Phase B H0), 1 NMP-cut-node-gate ultimately merged on third attempt.

**2026-04-30 evening update** — items #3 and #7 finally tested, both H0:
- #3 main-history-stm-dim → **H0 #888 −2.8 ±2.4 / 15K**
- #7 halfmove-200-revert → **H0 #886 −1.2 ±1.8 / 28K**

Plus two more 2026-04-30 closures from Tier-2/3:
- #10 factorized-main-hist (Tier 2) → **H0 #889 −2.6 ±2.4 / 16K**
- #19 lowply-history (Tier 3) → **H0 #887 −3.5 ±2.7 / 12.4K**

**Cluster lesson**: #887/#888/#889 H0'd cleanly negative on the same fleet pass — three top-engine-consensus history-shape ports failing at −2.6 to −3.5 Elo. Coda's 4D threat-aware main history already encodes the gradient these features surrogate. Saved as `memory/feedback_naive_ordering_ports_dont_transfer.md`. Implication: **stop firing history-shape ports off simpler-history engines** (lowply, STM-dim, factorisation, per-piece × main-history dual-update). Future ordering ideas should add a dimension Coda doesn't already have, not surrogate the threat-aware key.

Plus two cross-engine *port* closures from the same fleet pass — different mechanism but same "signal-overlap with our existing landscape" pattern:
- #883 cutoff-count-propagation (Tier 2 #9, LMR-only port) → **H0 −0.0 ±1.0 / 76K**
- #884 sf-small-probcut (Tier 2 #13) → **H0 −0.9 ±1.6 / 33.5K**

| # | Item | OB | Result | Follow-up |
|---|------|-----|--------|-----------|
| 1 | lmr-c-swap | #774 | H0 −9.3 | **structural finding (gate asymmetry); #780 + #791 retune confirmed signal-not-there post-#809 H0 -1.2** |
| 2 | triple-extension | #787 | H0 −1.0 | **#792 retune found no basin (3-4% noise drift); drop — signal-not-there** |
| 3 | main-history-stm-dim | — | not tried | high priority, untouched |
| 4 | enter-threat-penalty | #773, #781 | H0 / H0 | drop (signal-overlap with threat-history) |
| 5 | multicut-fix | #827 | **H0 -10.9** | drop; signal-overlap with post-#817 SE/DEXT cluster |
| 6 | pawn-history-8192 | #785, #797 | H0 −1.2 / H0 −3.3 | **bisection went worse — drop; possible -919 init refinement** |
| 7 | halfmove-200-revert | — | not tried | |
| 8 | material-np-only | #813 | **MERGED +0.6 / 117K** | -20% bench reduction (commit d2aca36); retune-on-branch candidate |
| 9 | LMP single-row B=8 | #808/#810 (Phase A) / #818/#832 (Phase B) | **Phase A MERGED +4.4 banked; Phase B H0** | Phase A: direct-check carve-out (+2.54) + skip-quiets (+1.86); Phase B Reckless-shape full retune comprehensively rejected |
| 10 | hist-prune-gate-drop | #786 | H0 stopped -0.2 / 56.5K, **MERGED for consensus alignment** (ebe9ad6) | |
| (extra) | nmp-cut-node-gate | #772, #807, #864 | H0 / H0 / **MERGED** +0.5 / 59.2K (3rd attempt, directional positive, commit f58a690) | yin-yang retune #870 didn't extend gain |

**Calibration**: doc's "Tier-1 +25-50 Elo" expectation collapses to
maybe +5-15 once retunes resolve. Consensus-port priors should be
discounted ~50-70% for Coda's current strength regime.

Two items (#3, #9) are still untried — those are the freshest
candidates. Three items (#1, #2, retune-needed for SPSA cluster)
have refined retries in flight (#791, #790, #792).

## Ranked queue for Hercules

Suggested order; each line is "branch name → expected Elo → key change":

**Re-prioritised after empirical diff doc** (`reckless_vs_coda_pruning_diff_2026-04-25.md`). Ordering principle: **fix causes (move ordering) before symptoms (over-firing pruning gates)**, but bank both since the symptom-fixes have +Elo magnitude on their own.

**Tier 1 — highest leverage (likely +25-50 Elo aggregated):**

> **Status update (2026-04-30):** Items 1, 3, 4, 5, 6, 7 (Phase B), 8 — RESOLVED
> (most H0, item 1 ultimately merged as directional positive on third
> attempt, item 7 split into Phase A merged + Phase B rejected, item 8
> merged for consensus alignment). Items 2 (main-history-stm-dim) and the
> Tier-1 pruning-shape work remain untested.

1. ~~`experiment/nmp-cut-node-gate-only` → **+5 to +12**~~ — **TESTED H0**. Default-tunable SPRT trended low; SPSA #790 found big movements (NMP_BASE_R +14%, NMP_UNDEFENDED_MAX −37%, NMP_MIN_DEPTH +15%) but post-tune SPRT **#807 H0 −0.4 / 49.6K**. Bucket: signal-not-there at our trunk's NMP balance. Drop. The 30%→57% gap is real but `cut_node` gating alone, even with retune, doesn't close it for us. **UPDATE 2026-04-29: third attempt #864 merged as directional positive +0.5 / 59.2K (commit f58a690).** Yin-yang retune #870 didn't extend the gain.
2. `experiment/main-history-stm-dim` → +3 to +8 → add stm to main quiet history 4D→5D. **Move-ordering cause-fix; FMC compounds.**
3. ~~`experiment/enter-threat-penalty` → +3 to +6~~ — **H0** vanilla #773 -0.4 / 5.2K, refined-with-split-tunables #781 H0 +0.1 / 94.7K. Drop (signal-overlap with threat-aware history).
4. ~~`experiment/pawn-history-8192` → +3 to +7~~ — **H0** #785 -1.2 / 27.7K at 8192; #797 bisection at 2048 worse (-3.3 / 8.3K). Drop.
5. ~~`experiment/triple-extension` → +3 to +8~~ — **H0** #787 -1.0 / 30.7K + SPSA #792 found no basin. Drop (signal-not-there; DEXT already covers same cases).
6. ~~`experiment/lmr-c-swap` → +6 to +10~~ — **H0** #774 -9.3 / 1.6K (stopped). Surfaced gate-asymmetry diagnostic; follow-up `experiment/capture-lmr-hist-adjustment` #780 H0 + retune #791 + post-tune #809 H0 -1.2 / 28.5K. Drop.
7. `experiment/lmp-two-row` → +5 to +8 → halved d² + improving multiplier separated. **Empirical: 28× LMP fire rate gap.** — **PARTIALLY RESOLVED**: Phase A `lmp-direct-check-carveout` (#808 +2.54) and `lmp-skip-quiets` (#810 +1.86) MERGED. Phase B Reckless-shape full retunes #818 H0 -2.4 + #832 H0 -3.3 — drop the formula change. +4.4 Elo banked from Phase A.
8. ~~`experiment/hist-prune-gate-drop` → +3 to +6~~ — **H0 stopped -0.2 / 56.5K** (#786) but **MERGED** for consensus alignment / code hygiene (commit ebe9ad6). Drop FUT_LMR_DEPTH/BAD_NOISY_DEPTH tightening sub-items unless retune-on-branch turns up signal.

**Tier 2 — high-confidence novel mechanisms (likely +12-30 Elo aggregated):**

9. ~~`experiment/parent-cutoff-count` → +2 to +5~~ — **H0 #883 −0.0 ±1.0 / 76K** (LMR-only port). Signal-overlap with our existing pruning landscape. Reckless's cutoff_count closes a hole their main-history can't see; on Coda's 4D-threat-aware history that hole is already covered. Same shape as `feedback_naive_ordering_ports_dont_transfer.md`. **Status: closed (LMR-only port).** NMP-side variant tested as #885 (still in flight, trending H0).
10. ~~`experiment/factorized-main-hist` → +5 to +10~~ — **H0 #889 −2.6 ±2.4 / 16K**. 3-engine consensus failed cleanly. Coda's 4D threat-aware key already encodes the factorisation gradient; adding a `[from][to]` factor on top splits gradient between two partially-overlapping cells. See `feedback_naive_ordering_ports_dont_transfer.md`. **Status: closed.**
11. `experiment/pawn-push-attacker-graduated` → +2 to +5 → Viridithas tactical motif (same family as B1 +52, knight-fork +5.2, offense +5.7).
12. `experiment/threat-bucketed-cont-corr` → +2 to +4 → Plenty pattern; intersects Coda's threat agenda.
13. ~~`experiment/sf-small-probcut` → +2 to +5~~ — **H0 #884 −0.9 ±1.6 / 33.5K**. Signal-overlap with our existing ProbCut. SF Step 12 pre-ProbCut shortcut pays in engines whose full ProbCut runs less aggressively; on Coda's tunables the full ProbCut already catches these cases. **Status: closed.**
14. `experiment/optimism-full-p1` → +3 to +7 → K1+K2+optBase+divisor SPSA on Stormphrax-shape (5 engines).
15. ~~`experiment/multicut-fix` → +1 to +4~~ — **H0 #827 -10.9 ±4.4 / 4.5K**. Bench dropped 19.6%; signal-overlap with post-#817 SE/DEXT cluster. Drop (possible refined retry: gate on `cut_node && !is_pv` like Reckless).

**Tier 3 — small wins, low risk:**

16. ~~`experiment/halfmove-200-revert` → +1 to +4~~ — **H0 #886 −1.2 ±1.8 / 28K**. The (100-hm)/100 form is correctly more aggressive at the 50-move cliff; (200-hm)/200 retains 50% at hm=100 which hurts end-of-rule positions. Obsidian/Reckless consensus is right for our search. **Status: closed.**
17. ~~`experiment/material-np-only` → +2 to +5~~ — **MERGED** #813 +0.6 ±0.9 / 117K (LLR -0.77 stopped, can't reach H1=3 at this magnitude). Genuine small positive with -20% bench reduction (commit d2aca36).
18. `experiment/ttmove-history` → +1 to +3 → SF global stat.
19. ~~`experiment/lowply-history` → +1 to +3~~ — **H0 #887 −3.5 ±2.7 / 12.4K**. SF early-ply table; surrogates the threat-aware gradient Coda's 4D main hist already has. See `feedback_naive_ordering_ports_dont_transfer.md`. **Status: closed.**
20. `experiment/variance-aspiration` → +1 to +3 → SF/Reckless/Viri.
21. `experiment/conthist-combined-modulator` → +1 to +3 → Stormphrax/Viri.
22. ~~`experiment/fh-blend-depth-cap` → +1 to +3~~ — **H0 #814 -0.3 ±1.3 / 55.3K**. Bench-flat, didn't carry Elo. Drop.
23. ~~`experiment/eval-quantisation-16cp` → 0 to +2~~ — **H0 #828** (Plenty pattern, trended H0 early). Drop.
24. `experiment/clover-combined-mat-hm` → +1 to +3 → single-multiply.
25. `experiment/sf-source-bonus-boost` → +1 to +2 → SF write-time scaling.

**Cross-cutting:** port the `dbg_hit` instrumentation pattern from Reckless into Coda's `src/search.rs` to enable matched 51-position-bench measurements after each Tier-1 merge. Validates that fire-rate gaps are closing.

## Cross-cutting observations

- **Framing inversion** (per `reckless_vs_coda_pruning_diff_2026-04-25.md`). Coda doesn't under-prune; we fire shallow gates 3-28× more often than Reckless. CLAUDE.md's prior framing of "Coda needs to prune harder" is wrong for LMP/FP/SEE and probably ProbCut. The actual gap is **NMP cutoff rate (30 vs 57%)** and **move ordering quality (1.15pp FMC)**. Treat ordering improvements as cause-fixes and threshold tightenings as symptom-fixes.
- **Coda's tunable space has SPSA-detuning artifacts** — `LMR_C_QUIET > LMR_C_CAP` (item 6) is structurally inverted vs every consensus engine and almost certainly an artifact of partial-feature tuning. The fix-then-retune cycle should expose more such inversions.
- **The "consensus H0 = dig deeper" lesson applies broadly.** From CLAUDE.md: capture history magnitude was +31.6 Elo when fixed despite 3 prior H0s. Several items here (single-row LMP, no triple extension, pawn-history 512) are the same shape — Coda has tried adjacent variants and H0'd, but the consensus-form has not been tried.
- **Tactical-motif scoring** keeps paying off. Items N-2 (pawn-push attacker), N-13 (king-wall-pawn malus) extend the W1 family of wins (B1 +52, knight-fork +5.2, offense +5.7).
- **Threat-aware mechanisms have headroom.** Item 3 (enter-threat penalty) and N-3 (threat-bucketed cont-corr) are direct extensions of work that already paid Elo. Plenty has the most threat-bucketed mechanisms and is at rank 4 on the RR.
- **Empirical NMP gap (30%→57%) is the single highest-leverage measured lever.** `cut_node` gate alone may close half of it; combine with `cutoff_count` propagation (item 9), TT-noisy guard (#768 in flight), and the verify-depth widening to close the rest.
- **Cumulative envelope across Tier 1 alone is +25-50 Elo before retune** per CLAUDE.md's +6-8 retune-per-3-merges pattern. Realistically: 8 merges × landed-conditional-on-Elo + 3 retunes ≈ +35-60 Elo. That'd close ~40-65% of Coda's 90-Elo gap to top-10.

## Companion docs

- `reckless_vs_coda_pruning_diff_2026-04-25.md` — **empirical instrumented diff**; cross-validates this doc's findings on LMP/FP/SEE over-firing and surfaces the NMP cutoff-rate gap as the single highest-leverage open lever.
- `next_ideas_2026-04-21.md` — prior round of ideas; intersect on T1.x tactical motifs, T2.x threat signals.
- `peripheral_mechanisms_2026-04-22.md` — non-pure-search mechanisms (P1 optimism is item 14 here).
- `correctness_audit_2026-04-25.md` — fresh audit findings; some interact (Coda's `SEE_CAP_MULT` dead tunable was found there).
- `research_threads_2026-04-24.md` — R2 LMP/NMP threads pre-figured items 7, 8.
- `experiments.md` — source of truth for resolved SPRTs.

## Stop-trying list (verbatim — mostly carried forward)

- Don't try another dynamic-SEE-threshold variant without changing the `score` input (must be captHist-only, not MVV-dominated).
- Don't try the **6D captHist expansion** with from+to AND threat bits (the single defended-bit / threat-bit is OK; full 6D is on stop list).
- Don't try threat-magnitude as LMR-amount / extension-amount.
- Don't try another margin widener for a signal that already has a gate landed (RFP/LMP king-pressure proved this).
- Don't add "generic piece-attacked" nudges (use specific tactical motifs).
- **NEW:** don't expand `SEE_CAP_MULT`'s use without first confirming it's wired in (it's currently dead code per the 2026-04-25 audit).
- **NEW:** don't re-port Reckless's NMP TT-capture skip as binary; use the dynamic R++ form instead (item 12).
- **NEW (2026-04-30):** **don't fire history-shape ports off simpler-history engines.** Coda's 4D `[from_threatened][to_threatened][from][to]` already encodes the gradient lowply (#887), STM-dim (#888), and factorisation (#889) surrogate. All three H0'd −2.6 to −3.5 Elo on the same fleet pass. Future ordering ideas should add a dimension Coda doesn't already have, not surrogate the threat-aware key. See `feedback_naive_ordering_ports_dont_transfer.md`.
- **NEW (2026-04-30):** don't port pre-ProbCut TT-trust shortcuts (sf-small-probcut #884 H0). Our existing ProbCut already catches these cases; the shortcut adds noise.
- **NEW (2026-04-30):** don't port LMR-only `cutoff_count` (Reckless port #883 H0). Signal-overlap with our 4D-threat-aware history. NMP-side variant (#885) trending H0 in flight; if confirmed, full mechanism class closes.
