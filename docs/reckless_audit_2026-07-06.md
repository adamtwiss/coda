# Reckless Search / Move-Ordering / TT Audit — 2026-07-06

Deep cross-engine read of **Reckless v0.10.0-dev** (`22d7558`, fresh pull at
`/tmp/Reckless`) against Coda's current `main`. Reckless bakes its
SPSA-tuned constants inline as literals, so every number below is a current
value. `file:line` refs are into Reckless `src/search.rs` unless noted;
Coda refs into `src/search.rs` / `src/movepicker.rs` / `src/tt.rs`.

**Why now:** the fresh pull is **121 commits ahead** of the copy in
`~/chess/engines/Reckless`, spanning ~40 search/ordering/TT PRs. Reckless
and Stockfish are the only two engines above Coda; we've spent recent
effort on SF, this pass goes deep on Reckless. Reckless's recent Elo delta
concentrates in **LMR correction terms** and **history/ordering shaping** —
both areas where Coda has concrete, still-open gaps.

The three feature-by-feature comparison tables are long; this doc leads with
the **actionable ranked lever list**, then records the full comparison and
the "already ahead / dead-end" ledger so we don't re-walk it.

---

## Ranked candidate levers (Elo × tractability)

Gating per CLAUDE.md: tree-shape changes → `[0,3]` STC-first then LTC, with
a retune-on-branch SPSA if a cluster shifts; complexity-free one-liners on
existing state → `[-1,2]`; NPS-only → bench + `[-2,1]`.

### Tier 1 — concrete gaps, low effort, where Reckless's recent delta lives

1. **LMR correction-term battery** (search). Coda's LMR lacks four terms
   Reckless added/tuned recently, each a one-line gate on state already in
   hand at the LMR site:
   - `is_win(beta)` → reduce **more** (winning-beta, #1087; Reckless
     `+1024/1024`).
   - `tt_score <= alpha` → reduce more (loose-alpha window, #1044;
     `+464`).
   - `tt_depth < depth` → reduce more (`+326`).
   - quiet correct-expectation `+ k*(alpha - static_eval).clamp(lo,hi)`
     (#1022; Reckless `+418*(alpha-est).clamp(-65,91)/128`).
   Verified absent in Coda (`grep` for each = 0 in the LMR block). This is
   the CLAUDE.md `[-1,2]` "master one-liner, no new state" class. Test as a
   small cluster or individually; retune the LMR cluster after. **Effort
   low; expected small-but-bankable each — this is the highest-density
   pocket of recent Reckless search Elo.**

2. **`cutoff_count` propagation** (search). Add `[i32; MAX_PLY+2]` to the
   search stack, increment at the beta-cutoff site, reset per node, and
   read `cutoff_count[ply+1]` in LMR (`+R` when `>2`, plus an extra bump at
   non-PV non-cut nodes) and optionally the NMP gate. **Confirmed still
   absent** (`grep cutoff_count src/` = 0) — flagged as an open gap since
   `docs/reckless_vs_coda_pruning_diff_2026-04-25.md` §3 and never closed.
   Reckless threads it through LMR, NMP, and node-shaping. **Effort medium
   (new state + a few read sites); highest structural upside on this list;
   tree-shape → retune.**

3. **Threatened-TO malus** (move-ordering). Penalize a quiet moving *onto*
   an enemy-attacked square. Coda has the symmetric **escape-FROM** bonus
   (`ESCAPE_BONUS_*`, `movepicker.rs:595-664`) but **not** the TO-malus;
   Reckless has both (`-8875 * threatened[pt].contains(to)`,
   `movepick.rs:209`). Cheapest precise form reuses the per-node
   `self.threats` mask already computed: `if self.threats & (1<<to) { score
   -= MALUS }`. **Start with the all-threats one-liner** — the
   lower-value-attacker-keyed version needs per-node `piece_threats` masks,
   which `memory/…offense_precompute_negative_2026-06-24` measured
   NPS-negative. **Effort low; a core SF/Reckless ordering term Coda simply
   lacks; tree-shape → retune.**

4. **`key_after` pre-make TT prefetch** (TT, NPS). Reckless prefetches the
   child TT entry **before** `make_move` + NNUE-push (#1085,
   `search.rs:1425`), overlapping DRAM latency with the whole make+NNUE
   cost; Coda prefetches **after** `make_move` (`search.rs:4905` etc.), so
   the fetch has only the short pre-probe window to land. Requires a new
   incremental "Zobrist key after `mv` without mutating the board" on
   `Board` (EP/castle/promo/capture + side + castling-rights delta) —
   **confirmed no such helper exists** (the `hash_after_two` hits in
   `board.rs` are a test local). An approximate key is fine — a wrong key
   only wastes a fetch. **Effort medium; the single biggest tractable NPS
   item here; NPS-only → bench + `[-2,1]`.**

### Tier 2 — cheap behavioral probes

5. **History term in quiet-SEE and LMP thresholds** (search). Coda's
   quiet-SEE (`-SEE_QUIET_MULT*lmr_d²`) and LMP limit ignore history;
   Reckless folds `history/1024` into both (`:823-825`, `:780`) so
   good-history quiets survive later. Coda's *capture*-SEE already carries
   capt-hist — this just extends the pattern to quiets. **Effort low;
   retune.**

6. **Index-decayed quiet malus** (#1038, move-ordering). Reckless scales
   each failed quiet's malus by `1024²/((1024+45*i)²/1024)` — later-tried
   quiets penalized less (`:1085-1096`). Coda applies flat `-malus` to
   every pre-cutoff quiet. Add the per-index scale in the existing malus
   loop (applies to main + cont + pawn malus). **Effort low; fresh Reckless
   win; retune.**

7. **Limit good captures to 3 when the TT move is quiet** (#992,
   move-ordering). After a quiet TT move, defer good captures beyond the
   3rd into bad-noisy (`movepick.rs:65`). Coda always exhausts all good
   captures before quiets. **Effort low-medium (counter + gate in the
   GoodCaptures stage); staging change → retune.**

8. **Relax the TT-cutoff node-type guard at deep nodes** (TT). Reckless's
   guard relaxes above depth 5 (`(!cut_node || depth>5)` /
   `(cut_node || depth>5)`, `:390-391`); Coda's `cut_node ==
   score_above_beta` guard **never relaxes**, forgoing deep cutoffs
   Reckless takes. Add `|| depth > GUARD_RELAX_DEPTH` with a tunable
   (default ~5). **Effort ~1 line + tunable; tree-shape → `[0,3]`.**

9. **PV-entry replacement protection** (TT). Give existing tt_pv entries
   one extra ply of survival in the same-key overwrite gate: require
   `depth > slot_depth - 4 - 2*slot_tt_pv` (Reckless `:253`). **Effort ~1
   line on existing state → `[-2,1]`/`[0,3]`.**

10. **NMP `improving` term + `estimated_score`** (search). (a) Coda's NMP R
    has no `improving` term; add `r += improving` (Reckless `+917/1024`,
    #975). (b) Bigger structural item: Reckless prunes off `estimated_score`
    — the static eval **tightened toward `tt_score`** when the TT bound
    aligns (`:470-481`) — in RFP/NMP/razoring, and runs the NMP null search
    against `tt_score` instead of `beta` when
    `tt_bound==Lower && beta>tt_score && depth-2<=tt_depth` (#1031). Coda
    prunes off plain corrected eval. **Effort low (R term) / medium
    (estimated_score at ~3 gate sites); tree-shape → retune.**

### Tier 3 — higher effort or more speculative

11. **Low-depth singular extension** (#1033, search). When the SE gate
    can't run: `depth<=7 && !in_check && cut_node && static_eval <= alpha-k
    => extension=1` (`:734-736`). Cheap cut-node extension on nodes expected
    to fail-high but eval-weak. **Effort low; retune.**

12. **Cont-hist ordering-weight split + separate cont bonus/malus shape**
    (move-ordering). Reckless weights cont plies 1/2/4/6 as `1.58/1.04/
    1.06/1.03×` (only ply-1 boosted) and gives cont its own steep
    bonus/malus formula with a `cut_node` reduction; Coda uses `[cm,cm,1,1]`
    (ply-1 **and** ply-2 at ~3× main) and reuses the main `bonus` for cont
    with no cut_node term. A real relative-weighting divergence, but Coda's
    values are deeply SPSA-tuned and possibly net-specific — treat as a
    tuning-shape experiment, not a bug. **Effort low-medium; must SPSA
    against Coda's existing NFH/sibling scaling to avoid double-counting.**

13. **Two-ply continuation correction history** (move-ordering/eval).
    Reckless samples cont-corrhist at ply-2 **and** ply-4 via a proper
    `[in_check][capture][piece][to]→[piece][to]` subtable, 50mr-bucketed,
    huge-paged (#1081); Coda samples one ply, flat `[piece][to]`. Keep
    Coda's transition-corrhist alongside. **Effort medium-high (table
    restructure + stack plumbing); eval-path → `[0,3]`.**

14. **Eval-delta quiet-history nudge** (#1055). Bonus the parent quiet by
    `clamp(k*(-(eval+prevEval)), lo, hi)` gated `depth<6 || tt_miss`
    (`:490-502`). Coda has the static-eval stack but no such update.
    **Effort medium; retune.**

15. **Low-priority odds & ends**: wall-pawn ordering malus
    (`-k*wall_pawns.contains(from)`); capture-history `to_threatened`
    dimension (Reckless's capt-hist is threat-aware, Coda's isn't); LMR SMP
    **jitter** `(nodes+id*k)%128 - offset` (Reckless `:895/957`, ~0 on
    single-thread OB, maybe small on multi-thread lichess); QS TT cutoff at
    PV nodes for non-decisive scores; non-power-of-2 TT sizing via Lemire
    index (recovers ~50% Hash only at odd sizes we don't deploy).

---

## Where Coda is already ahead or at parity (don't chase)

- **TT correctness — AHEAD, do not regress.** Coda's TT is fully atomic
  with disciplined Acquire/Release key-before-data ordering and 32-bit XOR
  torn-read verification (ARM-correct). Reckless runs **non-atomic entries,
  a 16-bit key, and no memory ordering** — fine on x86, strictly UB under
  Lazy SMP and unsafe on aarch64. Its density (8B/entry, 3/cluster, SWAR
  key lookup #974) is *enabled by* that racy model; adopting it would
  violate Coda's ARM standard. **Off-limits.**
- **Huge-page TT alloc** — Coda's 3-tier `MAP_HUGETLB → aligned+MADV_COLLAPSE
  → madvise` is more robust than Reckless's single `MADV_HUGEPAGE`.
- **Near-miss TT cutoff** — Coda accepts 1-ply-short entries with an 80cp
  margin at zero-window non-decisive nodes; Reckless has no analogue.
- **TT-score dampening, mate/TB round-trip, 50mr-mate-downgrade, full
  PV-move legality validation** — all richer in Coda.
- **Tiered negative extensions** (−1/−2/−3 by node type) vs Reckless's
  binary −3.
- **Transition (zobrist-delta) correction history** — Coda-unique source;
  no Reckless analogue.
- **Discovered-attack (x-ray) ordering bonus** and **SEE-gated quiet-check
  bonus** — Coda-only / more precise than Reckless's flat bonuses.
- **`num_fail_highs` + sibling multiplicative history scaling** — a
  cutoff-confidence mechanism Reckless lacks.
- **Threat-aware main history** (4D from/to) and **pawn-history-in-ordering
  (#1088)** — **parity**, Coda already does both; pawn-hist dims (512
  buckets × piece × to) match Reckless exactly. Reckless's recent #1088 is
  a catch-up to something Coda already had.
- **ProbCut improving-margin + improving-depth (#1086), forced-win guards
  on LMP/futility/SEE (#1025), eval-only TT writeback, aspiration
  root-depth-reduce-on-fail-high, RFP-before-NMP, hindsight reduce+extend**
  — all already present in Coda.
- **Root-depth-aware RFP/LMR/ProbCut STC↔LTC self-adaptation** — Coda-specific.

## Dead-ends — do NOT port from Reckless

- **History pruning** (Reckless `:818-821`): Coda removed it after **3
  consecutive H0s** (#1562/#1691/#1697); genuinely not a fit for Coda's
  tree shape.
- **Optimism / contempt** (`td.optimism`): Coda shelved both (contempt SPRT
  #508 net-positive to *remove*; optimism 2× H0, monotonically worse with
  magnitude). This is the eval-flywheel problem, not a search knob.
- **Triple extensions**: Coda deliberately excludes; a prior test exploded
  bench.
- **main-history stm dimension** (Reckless 5D): doubles memory, many strong
  engines omit it, uncertain value.

---

## Notable structural divergences (context for the levers)

- **LMR shape.** Reckless **deleted the move-count term** (#995/#996); its
  base is purely `269 * log2(depth)` (1024-scaled fixed point), and a long
  battery of *correction terms* does the rest (improvement, `|corr|`,
  exact-bound, tt_score/tt_depth, is_win(beta), history, PV window width via
  `root_delta`, tt_pv, cut-node, in-check, cutoff_count, singular margin,
  parent-reduction consistency, SMP jitter). Coda uses a classic 2D
  `ln(d)*ln(m)` table + adjustments. Reckless's recent gains are all in
  *adding correction terms*, not reshaping the base — which is exactly why
  the Tier-1 lever #1 is a clean fit (bolt terms onto Coda's table without
  touching its base).
- **Pruning eval.** Reckless prunes off `estimated_score` (TT-tightened);
  Coda off corrected static eval. Systemic — lever #10b.
- **History factorizers.** Reckless *removed* both quiet and noisy
  factorizers (#1037/#989). Coda never had one — nothing to remove, and the
  removals confirm not to add one.

## CLAUDE.md staleness found while verifying (flagged, not yet fixed)

Two factual drifts in CLAUDE.md caught against current source — worth a
correction pass (left to Adam; not touched as part of this audit):
1. Pawn history is **512 buckets** (`PAWN_HIST_SIZE=512`, indexed
   `pawn_hash & 511`), not `pawnHash%8192` as CLAUDE.md §History-tables
   states.
2. Correction-history sources are **pawn / white-NP / black-NP /
   continuation / transition** — the "minor, major" tables CLAUDE.md lists
   were **ablated to 0 and dropped 2026-05-18/19** (comment at
   `search.rs:79,269`).

---

## Lever status ledger (live — 2026-07-06)

Coordinated split with Hercules to avoid double-work:

| Lever | Owner | Status |
|---|---|---|
| T1.1 LMR correction-term battery (#1087/#1044/#1022/tt_depth) | **Hercules** | in progress |
| T1.2 `cutoff_count` propagation | **Hercules** | queued |
| #3 threatened-TO ordering malus (SEE-gated, stratified) | Fable | OB #2593 `[0,3]` running |
| #4 `key_after` pre-make TT prefetch | Fable | OB #2595 `[-2,1]` — **SURPRISE H1: +5.5 ±3.5 @ N=8.7k, LLR 2.6**. Flat on a clean single-process Zeus bench but a real fleet gain — the binary is NODE-IDENTICAL to main (bench 2302849=main), so it can only be speed. Reconciles via CONTENTION: the fleet runs ~6 concurrent games/worker with shared L3+bandwidth; hiding DRAM latency pays off there but not in a clean single-process bench. Same shape as the 2026-07 ponder saturated-A/B. **METHODOLOGY: measure NPS/memory changes under contention (concurrent games / saturated cores), not clean pinned bench — the clean bench has no bandwidth pressure and structurally under-reads prefetch/cache wins.** **CONFIRMED per-host split (machine SIMD flags via /machines/<id>/, game-weighted): AVX2-only hosts +9.0 ±10.2 (n=4604); AVX512 hosts -1.1 ±9.9 (n=4898) = flat. Aggregate faded +5.9→+3.8 as early-N noise washed out.** The whole win is on the AVX2/older-VPS half under all-core memory contention; AVX512/Rackspace is neutral (strong hw prefetchers + bandwidth headroom make explicit prefetch redundant). Strictly non-negative everywhere. Relevant to codabot (4-core VPS, AVX2-class → gains); coda_bot (Ryzen AVX512 → flat). Merge candidate (Adam's call); optional refinement = gate the pre-make prefetch to non-AVX512 to shed the (noise-level) -1.1, likely not worth the complexity. |
| #6 index-decayed quiet malus (#1038) | Fable | OB #2597 `[0,3]` running |

Next unclaimed non-LMR candidates (Fable queue): #7 (limit good captures to
3 when TT move is quiet, #992), #5 (history term in quiet-SEE/LMP
thresholds), #8/#9 (deep-node TT-cutoff relax / PV replacement protection),
#10 (NMP improving term + `estimated_score`). Hold on anything touching the
LMR reduction expression or `cutoff_count` — Hercules owns those.
