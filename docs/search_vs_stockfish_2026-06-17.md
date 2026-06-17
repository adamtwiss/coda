# Coda vs Stockfish — search/pruning deep dive (2026-06-17)

**Motivation.** Fixed-node comparison shows Coda reaches *less depth than
Stockfish at the same node count* — i.e. Coda **under-prunes / under-reduces**,
so its effective branching factor (EBF) is higher. Eval quality is close to SF,
so the strength gap that this surfaces is a **search-efficiency** problem, not an
eval problem. This audit compares Coda (`src/search.rs`, `src/movepicker.rs`)
against current-dev Stockfish (`engines/Stockfish/src/search.cpp`,
`movepick.cpp`, commit `133731f3`, 2026-05-19) technique-by-technique, looking
specifically for places Coda prunes/reduces *less* than SF.

All Coda numbers are resolved from the live `tunables!` defaults. All SF numbers
read from current dev. Nothing was changed — this is the analysis that seeds the
experiment branches.

---

## Headline: the gap is on the *reduction / late-move* side, not static pruning

A clear pattern fell out of all three clusters:

- On the cheap **static forward cuts** (razoring, ProbCut margin, RFP base
  margin), Coda is already **as aggressive as or more aggressive than SF**.
  Coda razors a *wider* eval band (`275·d` vs SF's `465+300·d²`), its ProbCut
  margin is *smaller* (90/117 vs 155/214), and its RFP base margin is
  competitive. These are SPSA-calibrated to Coda's eval scale — leave them.
- Coda pays all of that back, and more, on the **late-move / reduction side**:
  LMR at cut nodes, NMP reduction depth, and the SEE/futility pruning of
  individual moves are all materially *less* aggressive than SF. That is where
  the "less depth at fixed nodes" comes from.

So the lever is **reduce more on late moves and prune more individual moves**,
not "add another static cut."

---

## Ranked findings

Priority = expected EBF-leverage × confidence ÷ effort. Bounds are Coda's
standing `[0,3]` unless noted. Items that change the reduction table or node
count are **tree-shape-changing** → follow the retune-on-branch / guard
sub-pattern (SPRT at default to confirm direction, focused SPSA on the adjacent
cluster, then SPRT tuned-vs-trunk); a vanilla SPRT will undersell them.

### Tier 1 — biggest depth levers (structural, retune-on-branch)

**1. Cut-node LMR reduction is ~4 plies too small.**
Coda adds a flat `+1` to the reduction at non-PV late moves
(`search.rs:4108`); SF adds `+3995/1024 ≈ +3.9` plies at cut nodes, `+1059`
more (≈ +4.9) when there's no TT move (`search.cpp:1262`). Cut nodes are the
majority of interior nodes, so this term dominates EBF.

| node (non-pv, non-impr, cut, TT move) | Coda r | SF r |
|---|---|---|
| d8,  m6  | 3 | 6 |
| d10, m10 | 4 | 8 |
| d12, m15 | 5 | 9 |
| d20, m20 | 7 | 10 |

*Change:* replace the flat `+1` with a depth/state-scaled cut-node bump (new
tunable `LMR_CUTNODE_BUMP`, let SPSA push it well above 1; add an extra step
when no TT move). **Highest expected Elo in the audit.**

**2. NMP under-prunes two independent ways.**
- *Reduction too shallow at depth.* Coda `R = 8 + depth/8`; SF `R = 7 +
  depth/3` (`search.rs:~3540` vs `search.cpp:~967`). At d24 Coda nulls at
  depth−11 while SF nulls at depth−15 — Coda spends ~4 extra plies per null
  verify *and* gets fewer fail-highs.
- *Eval gate far stricter.* Coda requires `static_eval ≥ beta` (plus a
  `nmp_threat_margin` surcharge that can raise the bar); SF attempts NMP whenever
  `static_eval ≥ beta − 14·depth − 45·improving + 374`, i.e. up to ~330cp
  *below* beta at low depth. Coda attempts NMP on a much narrower band → far
  fewer attempts.

*Change:* one NMP branch — SF-shape depth-scaled eval gate + steeper reduction
(`depth/3`–`depth/4`) — then a focused NMP-cluster SPSA retune (`NMP_BASE_R`,
`NMP_DEPTH_DIV`, `NMP_EVAL_DIV`, `NMP_EVAL_MAX`, `NMP_VERIFY_DEPTH`). The shape
is structurally different from SF, not just a tuned constant, so this is high
value.

**3. Capture-SEE pruning ~2.5× too permissive + hard depth cap SF lacks.**
Coda prunes a bad capture only if `depth ≤ 7` and `SEE < −215·depth`
(`search.rs:3781`); SF uses `−175·depth` with **no depth cap**. Normalized to
pawns (SF pawn=208, Coda pawn=100): Coda lets through captures losing up to
**2.15 pawns × depth**, SF only **0.84 pawns × depth**. Largest single
in-loop-pruning divergence found.

*Change:* split a dedicated `SEE_CAP_MULT` off `SEE_MATERIAL_SCALE` (currently
shared with the QS delta path — must decouple first), lower it, and
raise/remove the `depth ≤ 7` cap. Partly softened today by BNFP, so confirm via
SPRT + retune.

**4. Quiet-SEE pruning ~2.7× too permissive + missing history inflation.**
Coda prunes quiets at `SEE < −33·lmr²` with **no history adjustment of
`lmr_d`** (`search.rs:3807`); SF uses `−25·lmr²` on `lmrDepth` that is itself
inflated by move history. Per-pawn: Coda 0.33·lmr² vs SF 0.12·lmr².

*Change:* lower `SEE_QUIET_MULT` (33 → ~22–25) **and** add SF's history
inflation of `lmr_d` so strong-history quiets survive while weak ones prune
harder. Retune-on-branch (interacts with futility).

### Tier 2 — strong, several are cheap/no-retune

**5. LMP: searches more moves + disabled past depth 8.**
Coda `(6 + depth²)/(2−improving)`, gated `depth ≤ 8` (`search.rs:3997`); SF
`(3 + depth²)/(2−improving)`, **no depth cap** (`search.cpp:1114`). Coda
searches 1–2 more quiets at every shallow depth and has **no move-count pruning
at all at depth 9+**.

*Change:* lower `LMP_BASE` toward SF's 3, and raise/remove `LMP_DEPTH`. This
continues the already-banked `LMP_BASE 8→5` thread (+1.7–4.1, memory
`project_rfp_lmp_flat_wins`); the depth-cap removal is the fresh half. Safe to
SPRT directly.

**6. Missing `cutoffCnt` "next ply fails high a lot" reduction.**
SF adds up to +2.4 plies when the child node has produced many cutoffs
(`search.cpp:1270`); Coda tracks no `cutoffCnt` at all. Pure missing reduction.
New per-stack counter, cheap. Retune-on-branch.

**7. Quiet check bonus is not SEE-gated (cheap ordering fix).**
Coda adds `+14805` to *any* direct-check quiet (`movepicker.rs:667`); SF only
adds its check bonus when `see_ge(m, −75)` (`movepick.cpp:240`). Coda therefore
orders losing check-sacs into the first-searched slot. Add a `see_ge(mv, −75)`
gate. Very cheap, low-risk, no retune.

**8. Main-history under-weighted in quiet ordering.**
Coda weights main history `×1`, same as a single cont-hist plane
(`movepicker.rs:628`); SF weights `main ×2` and `pawn ×2` relative to one
cont-hist plane (`movepick.cpp:231`). Main history is the lowest-variance,
most-populated table — under-weighting it mis-sorts and costs first-move
cutoffs. Add a `MAIN_HIST_MULT` (default 2); pair with a focused ordering tune
since the cont-hist weights were SPSA-converged against the current 1×.

**9. Missing "quiet moves INTO a lesser-piece attack" penalty.**
Coda has the *escape* half (`ESCAPE_BONUS_*`) but no penalty for a quiet that
walks onto a square attacked by a lesser piece; SF has the symmetric graded term
`PieceValue·20·(threatByLesser&from − threatByLesser&to)` (`movepick.cpp:244`).
Demotes blunder-quiets before they're searched. Needs per-piece-type
lesser-attacker bitboards (Coda already computes most of the inputs). Medium
effort.

### Tier 3 — lower magnitude / speculative

**10. Quiet futility margins ~2–4× wider than SF** (per-pawn) and history-adjust
is main-history-only vs SF's full multi-table fold into `lmrDepth`
(`search.rs:3960` vs `search.cpp:1158`). Retune-on-branch; FUT cluster already
SPSA-converged so magnitude uncertain.

**11. ProbCut: add SF's free TT-ProbCut early return.** SF returns on a TT
lower-bound at `beta + 428` with no search (`search.cpp:1046`); Coda lacks this
cutoff entirely. Cheap new cutoff. Also two Coda-only ProbCut suppressors
(`king_zone_pressure`, `unstable`) SF doesn't have — ablate separately.

**12. Singular extension fires ~2 plies too shallow + no ttPv margin penalty.**
Coda `depth ≥ 4` and flat margin (`search.rs:3819`); SF `depth ≥ 6 + ttPv`
with a ttPv-widened margin (`search.cpp:1190`). Coda extends more readily →
shallower effective depth. Retune-on-branch.

**13. History-update parity gaps.** Coda's bonus is depth-only and lacks SF's
`+382·(best==tt)` and `+statScore/30` terms (`search.cpp:1880`); Coda's malus
slope equals its bonus slope, whereas SF's malus slope is ~7× steeper; Coda
lacks SF's cont-hist "positive consistency" multiplier (`search.cpp:1925`). Roll
into a history-shape focused tune. Low confidence individually.

**14. RFP Coda-only multiplicative suppressors** (`has_pawn_threats ×4/3`,
`unstable ×4/3`, up to ×16/9) that SF lacks (`search.rs:3434`). These widen the
RFP margin in tactical positions → fewer cuts than SF. Ablate each at `[-2,1]`
— but they're plausibly load-bearing safety guards, so direction is genuinely
uncertain.

### Not under-pruning — do not touch
- **Razoring** — Coda razors a *wider* band than SF; already aggressive.
- **ProbCut margin / min-depth** — Coda's 90/117 + depth-2 floor is *more*
  aggressive than SF's 155/214 + depth-3; SPSA-calibrated to Coda's eval scale.
- **RFP base margin** — linear `43·d` is competitive at STC; only the
  multiplicative suppressors (#14) are the surface.
- **Capture ordering** — `MVV×28·see_value + capt_hist` vs SF
  `7·PieceValue + capt_hist` are well-aligned in ratio (~6× vs ~7× capt-hist /
  pawn-MVV).

---

## Recommended sequencing

1. **Fire the cheap, no-retune wins first** (parallel, direct `[0,3]` SPRTs):
   #7 (check-bonus SEE gate), #8 (main-history ×2), #5 (LMP base + depth-cap).
   These don't need a reduction-table retune and have high confidence.
2. **Then the Tier-1 structural branch with the most leverage**: #1 (cut-node
   LMR bump) on its own branch, default-SPRT to confirm direction, then focused
   LMR+adjacent SPSA, then tuned SPRT. This is the single biggest expected gain.
3. **Then #2 (NMP shape)** as a separate branch with its own focused NMP retune.
4. #3/#4 (SEE margins) need the `SEE_MATERIAL_SCALE` decouple first; do after #1.
5. Tier 3 as fill-in.

Each is one change per branch (CLAUDE.md: never stack untested changes). The
reduction-table items will undersell on a vanilla SPRT — budget the
retune-on-branch step into the plan, don't drop them at +1.

**Meta-conclusion for the strength frontier:** the audit substantiates the
"effective depth is the target" thesis — Coda's loss vs SF here is concentrated
in *reduction aggressiveness on late/cut nodes*, exactly the "+3–6 ply
ordering/pruning carve-out" class the strength-frontier doc says Coda's
loss-profile favours. This is search-side Elo that does not depend on the eval
flywheel and can run in parallel with the bishop-blindness training work.
