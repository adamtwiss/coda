# search.rs Fresh Audit — 2026-06-11

Six-track parallel audit of src/search.rs (5436 lines, post NMP-de-gate /
v6-s5 promotion): QS, singular-extension machinery, ID/aspiration/root/
time/SMP, in-loop pruning + LMR, history/corrhist plumbing, perf+quality.
Cross-engine reference set: Stockfish, Reckless, Obsidian, Berserk,
PlentyChess, Alexandria, Clover, Integral, Stormphrax, Starzix (all
stronger than Coda). EXCLUDED (covered by other 2026-06 audits): NMP, RFP,
movepicker internals, TT internals, movegen.

Line numbers reference main @ 4bbb25d.

## Tier 1 — Bug-class findings (mechanism divergence, do first)

### T1.1 TT bound-narrowing cutoff gives history BONUS to fail-LOW TT moves
`search.rs:2944-3000`. The `alpha >= beta` collapse after window narrowing
fires in BOTH directions: LOWER raising alpha (fail-high — bonus correct)
and UPPER dropping beta below alpha (fail-low). In the UPPER case the
stored tt_move is the best move of a FAILED node (UPPER entries carry
moves because best_move is set whenever score > best_score, even below
alpha — :4158), yet it receives a fail-high-sized main/capture-history
bonus. Fires routinely: at cut nodes the primary cutoff's node-type guard
(:2885) blocks UPPER cutoffs, dropping them into the narrowing path. The
merged feature (+14.6, "V5: TT Cutoff History Bonus") was specified
fail-high-only. SF gates on `ttData.value >= beta` (search.cpp:786-791);
no surveyed engine bonuses a fail-low TT move.
**Fix:** gate bonus block on `tt_entry.flag == TT_FLAG_LOWER`. SPRT [0,3].

### T1.2 Stale `info.reductions[ply]` visible to LMR re-search children
`search.rs:4047` (store), 4057/4076/4126 (searches), reset only at node
entry (:2692). The slot is set to R before the reduced search and never
zeroed, so children of the FULL-DEPTH re-searches (exactly the fail-high
moves) read `prior_reduction = R` and mis-fire hindsight reduce/extend on
a false premise. SF: `ss->reduction = 0` immediately after the reduced
search; Stormphrax search.cpp:1052-1054 identical; PlentyChess scopes via
flag. Related: HINDSIGHT_MIN_DEPTH_10X is SPSA-pinned at floor (eff 0,
hindsight-reduce fires at depth 1 → drops to qsearch; consensus floor
d>=2) — plausibly SPSA compensating for the polluted signal.
**Fix:** zero `info.reductions[ply_u]` after the reduced search returns.
SPRT [-3,3]; if flat, focused hindsight-cluster retune (HINDSIGHT_THRESH,
HINDSIGHT_MIN_DEPTH_10X, ~500-1000 iters).

### T1.3 doDeeper/doShallower adjustment dropped by the full-window re-search
`search.rs:4069-4076` vs 4124-4126. Zero-window verification runs at
`new_depth + do_deeper_adj` but the PV-window re-search reverts to
`new_depth`. 6/6 engines mutate new_depth itself (SF, Obsidian,
Alexandria, Stormphrax, Integral, Reckless).
**Fix:** `new_depth += do_deeper_adj;`. Bundle with T1.2 as one
"LMR re-search correctness" branch (same block, mechanically independent
of other findings). SPRT [-3,3].

### T1.4 Multicut can return an unproven decisive (mate) score
`search.rs:3665-3671`. `singular_score` is fail-soft from a reduced
`(depth-1)/2` search with the TT move EXCLUDED; if it's a mate score it
propagates and gets TT-stored at the parent's full depth as LOWER —
unproven mate distance with full-depth credibility. SF: `value >= beta &&
!is_decisive(value)` (search.cpp:1180); Reckless same; Obsidian/Berserk
return singularBeta (structurally non-decisive). #1215 ported SF's return
value without the companion guard. Per #761 H0, don't suppress firing —
fix only the returned value.
**Fix:** when `singular_score >= MATE_SCORE - 100`, return `singular_beta`
instead. Also: the SE entry guard (:3647, ±28900) admits TB scores
(TB_WIN=28800) — see T2.3 is_decisive. SPRT [-3,3].

### T1.5 QS stores TT_FLAG_EXACT
`search.rs:4673-4679` (evasion), 4827-4833 (captures). A QS score is never
exact (captures-only + stand-pat); these EXACT entries later satisfy the
unconditional EXACT stand-pat refinement (:4717) and non-PV EXACT cutoff
(:4557) at full confidence. Unanimous: SF/Reckless/Obsidian store only
LOWER/UPPER in QS.
**Fix:** collapse EXACT arm into UPPER in both stores. SPRT [-3,3].

### T1.6 Small-correctness bundle (one branch, [-2,1])
- **QS depth cap returns static eval while in check** (:4497-4500): cap
  check runs before check detection; unanimous consensus returns draw when
  in check at the cap.
- **Malus exclusion by list position not identity** (:3862-3874, 4183,
  4240, 4320): non-capture promotions causing cutoffs enter the quiet
  branch but were never pushed (`!is_promo` push gate), so
  `saturating_sub(1)` wrongly exempts the last real loser quiet; same on
  list overflow (64/32). SF/Berserk/Reckless all exclude by identity.
- **LMR re-search cont-hist nudge has no stop guard** (:4076-4121): a stop
  during the second re-search bubbles 0 → spurious malus in 4 cont-hist
  offsets. Corrhist got exactly this fix (C8 #12).
- **tm_max_time missing from the C6 per-go reset block** (:1934-1940):
  same stale-field class as the H7 instant-emit bug. Defensive.
- **sel_depth never reset per ID iteration** (:1873): UCI seldepth is a
  search-lifetime max; all references reset per iteration. Output-only.

## Tier 2 — Strength opportunities (consensus gaps with mechanism)

### T2.1 Correction history wiped every `go` (+ helpers never seeded)
`search.rs:1878-1879` (`clear_correction_history()` per search),
1427-1443. Flagged independently by two audit tracks. Every other learned
table is aged ×0.8 and retained; corrhist alone is zeroed per `go`, so
every move re-learns eval miscalibration from scratch — and with the ±3cp
err clamp it needs ~50+ updates to saturate. ALL five surveyed engines
clear corrhist only on ucinewgame. The per-search clear dates to 0bc0f10
(2026-03-29), pre-dating the multi-table system; never revisited. Coda's
own history: blanket history persistence was +19.2.
**Fix:** delete the per-go clear (keep ucinewgame clear; optionally age
×3/4); seed helpers in copy_from. SPRT [0,3].

### T2.2 Corrhist update lacks the bound-direction consistency gate
`search.rs:4393-4412`. On fail-high, best_score is only a LOWER bound; if
`scaled_eval > best_score` we train corrhist downward on a score whose
true value may be above eval. All five engines gate this (SF:
`(bestValue > ss->staticEval) == bool(bestMove)`; Stormphrax/Obsidian/
Berserk/Reckless equivalents). One line:
`&& !(best_score >= beta && best_score <= scaled_eval)`. SPRT [0,3].

### T2.3 TB scores pass every `MATE_SCORE - 100` guard
MATE_SCORE=29000, TB_WIN=28800; guards at ±28900 admit all TB scores —
corrhist trains on tb_floor-raised scores (:4353 runs before the update),
SE margin arithmetic runs on coarse TB values, the NMP "skip for mate/TB"
comment is wrong about the TB half. References use is_decisive() covering
TB range.
**Fix:** add `fn is_decisive(s) = s.abs() >= TB_WIN - MAX_PLY` and use in
corrhist gate + SE entry + audit other sites. Mandatory prerequisite for
T2.4. Bundle, no standalone SPRT.

### T2.4 Corrhist err pre-clamp ±3cp — the full consensus bundle (known-partial)
`search.rs:247, 1244-1245`. Max update 21 vs gravity cap ~341 → sign-only
integrator. Every engine feeds full depth-scaled error, clamping only the
result. #1248/#1249 tried the clamp-only removal (marginal; the
experiments.md verdict: needs the full /128+GRAIN bundle). Critically
INTERACTS with T2.1: testing clamp removal against a per-search-wiped
table understates it.
**Plan:** corrhist overhaul branch = T2.1 + T2.2 + T2.3 + full err bundle,
retune-on-branch CORR_* cluster (~1500 iters), then SPRT [0,3].
(Alternatively stage: T2.2 one-liner first, then T2.1, then bundle.)

### T2.5 QS stand-pat bypasses correction history
`search.rs:4690-4699`. Main search corrects static eval (:3105); QS uses
raw scaled eval. Every QS leaf is uncorrected while corrhist trains on
main-search scores. Unanimous consensus: SF/Reckless/Obsidian all correct
in QS. Keep storing RAW eval to TT (invariant preserved — correct at
consumption).
**Fix:** apply corrected_eval to stand-pat. SPRT [0,3];
retune-on-branch candidate (QS delta/SEE margins calibrated against
uncorrected stand-pat). Synergistic with the corrhist overhaul.

### T2.6 Razoring absent — 10/10 unanimous among stronger engines
Removed d996d6f (2026-04-01, pre-v9, wide-bounds-era ablation). All 10
surveyed have qsearch-verified non-PV razoring (SF eval < alpha−502−306d²;
Obsidian eval < alpha−352d, d<=5, |alpha|<2000; etc.). Classic
"consensus H0 → dig deeper" profile: stale evidence, different eval.
**Fix:** re-add `!is_pv && !in_check && depth<=4 && alpha.abs()<2000 &&
static_eval + RAZOR_MULT*depth <= alpha` → qsearch verify, return if
<= alpha. RAZOR_MULT≈250-300 tunable. SPRT [0,3].

### T2.7 Aspiration loop: fail-high depth reduction never resets and is uncapped
`search.rs:2098-2104` (+ helper duplicate :1819-1823). asp_depth
monotonically decreases for the loop lifetime; on fail-high→fail-low
oscillation the fail-low re-search runs reduced and never recovers. All 4
surveyed reset on fail-low (SF failedHighCnt=0; Reckless reduction=0;
Stormphrax cap 3; Obsidian also skips the increment near mate scores).
The 2026-03-31 experiment ported the decrement without the reset/cap.
**Fix:** fail_high_cnt with reset-on-fail-low, cap ~3, optional near-mate
skip; both loops. SPRT [0,3].

### T2.8 Aspiration centres on raw prev_score, not smoothed average
`search.rs:2077-2080`. Variable literally named `avg`, comment cites
Reckless — but only the delta sizing was ported. Reckless/SF/Obsidian all
centre on a running average (`avg = (avg + score)/2`), damping one-off
spikes. ~4 lines. SPRT [0,3].

### T2.9 Mid-iteration stop discards proven root progress (no root-moves list)
`search.rs:2114-2131, 2085-2111`. (a) A completed root move that raised
alpha before the stop is thrown away; (b) a root fail-high whose widened
re-search times out loses the PROVEN-better move and plays the previous
iteration's. These are exactly the moves TM interrupts on (it extends on
instability/fail-high). All four references keep a RootMoves vector,
stable-sorted after every pass including failed ones.
**Incremental fix:** in the fail-high branch, before re-searching,
validate pv_table[0][0] against root_legal and promote to
best_move/stable_pv (proven lower bound — always safe). SPRT [0,3].
**Structural follow-up:** real root-move list (enables prev-score root
ordering, MultiPV, fixes the near-dead TT fallback :2152-2167).

### T2.10 QS in-check: unlimited quiet evasions + move-count counts pruned moves
- Quiet evasions unbounded (:4614-4664): all three references stop
  searching quiet evasions once best_score isn't a loss (SF `if (!capture)
  continue` inside !is_loss; Obsidian one-quiet-then-break; Reckless
  skip_quiets gate).
- QS_MAX_CAPTURES counter (:4753-4757) increments BEFORE delta/SEE
  pruning, so pruned moves consume budget, and has no non-losing gate.
  Comment cites "Obsidian: 3"; SPSA pushed default to 24 (near-disabled) —
  the classic "SPSA detunes a mis-implemented feature" signature.
**Fix:** count only searched moves, exempt promotions/direct checks, gate
on non-losing, then focused SPSA toward consensus 3-5. Retune-on-branch;
SPRT [0,3]. Tree-shrinking — bench will move.

### T2.11 BNFP margin omits the victim value
`search.rs:3803-3810`. Capture futility should credit optimistic material
gain; Coda prunes losing QxQ identically to losing PxP. 4/4
same-mechanism engines include a victim term (SF full value + captHist;
Reckless 8%; PlentyChess full; Clover full). SEE<0 gate doesn't
substitute (it judges the exchange, not the stand-in material swing).
**Fix:** `+ see_value(victim) * BNFP_VICTIM_PCT/100`, try 100 and ~10.
SPRT [0,3]; retune BNFP pair if flat.

### T2.12 SE gated on `!in_check` — zero-engine support, never probed
`search.rs:3634`. None of SF/Reckless/Obsidian/Berserk/Stormphrax gate SE
on check. Deep in-check nodes (often a single forced evasion — maximally
singular) never get SE/multicut/negative-ext. Mechanically safe to remove
(SE path doesn't read static_eval; correction_value is position-keyed).
**Fix:** drop `!in_check`. SPRT [0,3]; retune-on-branch if flat+bench-shift.

### T2.13 Missing fail-low prior-countermove bonus
Node-end block (:4335-4433) does nothing for the opponent's previous move
on an UPPER-bound node — direct evidence that move was good. SF
(search.cpp:1444-1474) and Reckless (search.rs:1033-1065) both bonus the
prior quiet via main+cont(+pawn) hist with factor scaling. The previously
rejected idea (malus to OUR tried quiets at fail-low) is a different
mechanism; the PCM bonus has never been tried.
**Fix:** port minimal Reckless shape (quiet PCM only). SPRT [0,3].

### T2.14 Missing "early refuted quiet" malus at ordinary beta cutoffs
Coda penalizes the opponent's last quiet only on TT cutoffs (:2907-2939).
SF/Reckless also apply it on searched cutoffs when the parent's move_count
was <=1+ttHit (low-noise gate). Needs a parent-move_count stack ([u8;
MAX_PLY]). SPRT [0,3].

### T2.15 Lower-priority strength probes
- **Malus magnitude coupled to bonus boosts** (:4188-4196): NFH-cascade and
  depth boosts (best-move-confidence signals) amplify the malus applied to
  up to 63 unrelated quiets. 3/4 engines use separate malus formulas.
  Probe: un-boosted malus first; then separate HIST_MALUS_* tunables.
- **Single-move-keyed cont-corr** (:726, 1261): consensus keys continuation
  correction by move PAIR ((ss-2) move × (ss-1) move); Coda's 768-entry
  single-move table is a much weaker signal. After T2.1 (persistence).
- **Capture LMR non-PV only** (:4003): 7/8 reduce captures at PV too, but
  four prior capture-LMR H0s — lowest priority.
- **Cut-node LMR term uses !is_pv** (:3901): every engine has a separate,
  larger cut_node term; Coda's prior attempts all ADDED reduction and
  H0'd. The untested direction is the SWAP (all-nodes reduce less). Low
  priority given history.
- **QS stand-pat TT store missing** (:4723): all three references store
  (depth-unsearched, LOWER, raw eval) on stand-pat cutoff with !tt_hit.
  Cheap [0,3].
- **SMP vote lacks mate/TB guards** (:1724-1737): Obsidian's three-rule
  tiebreak (proven-win protection etc.). Rare-case; [-2,1].

## Tier 3 — Performance (bench-identical bundle, [-2,1])

Same class as the validated +4.0 movepicker bundle. Candidates, by
expected impact:

1. **Threat/attack/xray block computed before TT cutoff, TB probe, and QS
   dispatch** (:2717-2744). Runs on EVERY negamax entry including all
   depth<=0 dispatches and TT-cutoff exits; xray_blockers is a nested
   slider loop. First real consumers are all post-dispatch. Move below
   the QS dispatch + TT cutoffs; compute locally in the one cutoff branch
   that needs it (:2983). Biggest single per-node win available. (Also fix
   the actively-wrong cost comment at :2716.)
2. **Depth<=0 nodes probe TT twice** (:2836 then :4534 same hash). Pass
   the probed entry into a quiescence_entry variant (bench-identical
   form). The alternative (dispatch before probe) changes nodes — [0,3].
3. **Snapshot FEAT flags + tunables into locals/struct per node/search**
   (~10 FEAT + 15-30 tp() relaxed atomic loads per MOVE; opaque to the
   optimizer, blocks hoisting/CSE). setoption only happens between
   searches, so a per-search SearchParams snapshot is semantics-preserving.
4. **LMP threshold before gives_direct_check** (:3785-3792) + cache
   gives_direct_check once per move (currently 2× per quiet at d<4:
   futility carve + LMP carve), + hoist LMP above SEE-quiet/futility in
   the prune chain (7/10 engines LMP-first; behavior-identical reorder).
5. **Pass negamax's `pinned` into MovePicker::new** (recomputed :3042 →
   movepicker.rs:279).
6. **lmr_d computed for captures that never use it** (:3603) — gate on
   !is_cap.
7. **PV/info string + board clone + Vec built when silent** (:2189-2268)
   — gate on !info.silent (datagen throughput).
8. **TT-cutoff PV-stuff validation pays pinned+checkers per cutoff**
   (:2898, :2966) on a path that "empirically never fires" — skip PV
   stuffing at non-PV cutoff nodes (defence-in-depth preserved by the
   PV-print legality walk).

Separate (not SPRT-visible at 1 thread):
- **Per-go helper churn** (:1654-1694): thread spawn + ~17MB zeroed per
  helper per move; helpers lose pawn_hist/corr learning every move.
  Persistent helper pool + aging. Verify via Threads=4 NPS bench.
- **Pawn-hist aging walks 13.6MB scalar per go** (:1882-1888): the i32
  round-trip blocks autovectorization; ~1-2ms + cache flush per move at
  STC. Vectorizable rewrite or epoch-lazy aging.
- **Node-accounting flush asymmetry** (:908-913, negamax checks
  pre-increment :2754, QS post-increment :4505): boundaries double-fire or
  skip; a skipped boundary doubles the time-check interval to ~8192 nodes
  (~33ms at 250K NPS) eating most of the 50ms forfeit margin. Delta-flush
  (`last_flushed_nodes`) + aligned check gating.

## Tier 4 — Deployment correctness (ponder path, ARM)

- **Ponderhit deadline trio: three Relaxed stores read non-atomically**
  (uci.rs:809-811 → search.rs:2038-2056). Reader-publish pattern on
  data-dependent atomics — exactly the class the ARM standard forbids.
  Floor can be observed 0 (one-shot latch → post-ponderhit min-think floor
  silently absent for that move). Store floor/hard first then soft with
  Release; load soft with Acquire. Bench-neutral.
- **GUI `stop` erased in the ponderhit→fresh-search window**
  (uci.rs:562-567): `stop.store(false)` unconditionally after wait-loop
  exit; a stop landing in the window is wiped and the GUI blocks for the
  full search. Gate on !external_stop. Verify with scripted UCI session
  (not OB-testable).

## Cross-cutting / needs-decision

- **Pawn-hash index inconsistency** (:3520 vs :3953): move-ordering slot
  uses PARENT pawn structure (& mask, pre-make); LMR history adjustment
  uses CHILD structure (% len, post-make). For pawn moves these read
  different slots for the same move. If intentional, comment it; if not,
  reuse the outer ph_idx (changes nodes → [0,3]).
- **CLAUDE.md drift**: futility described as "history adjusts effective
  lmr_depth (SF pattern)" — code adds main_hist/128 to the VALUE, not the
  depth. Doc fix.
- **Housekeeping from the SE track**: #1201 (verification cut_node) queued
  LTC retry never ran; #1202 (-1 all-node negative ext) SE-cluster retune
  still pending in the audit queue.
- **Root forced-move verification pollutes root_move_nodes TM accounting**
  (:2384 vs :4148-4152): gate accumulation on excluded_move[0]==NO_MOVE.
  TM-class methodology.

## Quality (no SPRT; verify bench unchanged)

- Triple `is_pv` definitions with different windows (:2696, 2884, 3158) —
  shadowing rebind invites the exact bug class the 2026-05-31 audit fixed.
- 6+ copies of the NNUE/threat push/make/pop block, several with mangled
  indentation (:3286, 3339, 3442, 3818, 4144, 4636, 4797) — extract helper.
- Duplicated pawn-hist gravity inline with hardcoded 16384/32000
  (:4230-4237, 4277-4287) — drift risk vs MAX_HISTORY.
- Dead guards: `ply_u <= MAX_PLY` ×10 after the :2684 early return; QS
  evasion `qs_safe_ply <= MAX_PLY` (:4628).
- Dead tombstone fn `_phase13_deleted_old_compute_tm_budgets` (:1613).
- Duplicate piece_at lookups in the quiet-malus loop (:4252, 4279).
- Helper seeding comment claims "aged" but copy happens pre-aging;
  pawn_hist/corrhist not seeded at all (:1427-1438).
- Stale "six source tables" docs — corrhist is now 4 sources / 3 weights.

## Verified clean (checked, not flagged)

Root pruning exemptions complete (every prune ply>0-gated); per-ply
clears + stable-PV snapshot correct; stop-guards on TT/corrhist writes
present; excluded-move plumbing through TT/corrhist/NMP/RFP/ProbCut/TB all
correct; double_ext_count propagation/reset correct; aspiration
contraction shapes individually SPRT'd; QS draw detection (50mr,
insufficient material, repetition with null boundary) clean; QS
delta-pruning shape = consensus; QS SEE gate -26 in band; killers
intentionally absent (+3.77); conthist offsets {1,2,4,6} = Reckless; TT
stores raw eval; no allocations/strings in hot path; PruneStats plain
per-thread u64; FEAT flags Once-parsed; NNUE lazy-push barriers intact;
helper vote formula = SF/Obsidian shape.
