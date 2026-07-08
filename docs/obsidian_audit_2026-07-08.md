# Obsidian dev-16.15 — fresh idea-mining for Coda (2026-07-08)

Adam-commissioned fresh look at Obsidian (#3 in our RR, under-studied relative
to SF/Reckless). Three parallel deep-reads (NNUE/eval, search/TM,
ordering/history/TT) against current Coda, each told what Coda has already
absorbed from Obsidian (pairwise FT, NNZ sparse L1, Finny, cont-hist weights,
complexity-aware LMR, cutNode-NMP) so they hunt for what's *not* yet there.
Obsidian dev-16.15 hasn't moved since Sep 2025, so this is a current-state
comparison (no recent-commit delta like the Reckless audit had).

Two agent claims were FALSE POSITIVES on cross-check (logged so we don't
re-chase): (a) "Coda lacks material/phase eval scaling" — WRONG, Coda applies
`score*(22400+material)/32/1024` at the eval-consumption site (`search.rs:1537`,
non-pawn-only per #813, TT-halfmove-independent by design; more considered than
Obsidian's `(230+phase)/330`); (b) "explicit move-into-threat penalty" — that's
the Reckless threatened-TO malus already tested to H0 (#2593). Both dropped.

## Bottom line

Obsidian is largely a **leaner** version of what Coda already has, and on the
LTC/deep-tree theme **Coda is ahead** on the mechanism (Coda has explicit
deep-spine LMR protection — `LMR_ROOT` term + depth-decaying all-node
inflation — that Obsidian entirely lacks; Obsidian's LMR is pure `ln·ln`). Two
hoped-for levers are NOT in Obsidian: it does **not** scale its per-move TM
fraction with the clock either (flat 2.5% cap, same as Coda), and its eval
edges (wider 1536 FT, 13 king buckets, active dual-L1-activation) are all
retrain/capacity items — not Coda's binding constraint. So Obsidian is not an
eval seam and not the TM-scaling reference.

But three **genuine, verified-absent** candidates came out, two of them
flagged independently by two agents:

## Candidate ideas (verified against Coda source)

### A. Eval-swing retroactive "opponent move quality" history (EvalHist) — TOP PICK
**Flagged by both the ordering and search/TM agents; SF + Obsidian both have
it.** After computing static eval at a node, write a magnitude-scaled
bonus/penalty to the OPPONENT's last move's **main (butterfly) history** based
on the eval swing it caused: Obsidian `theirLoss = prev_sEval + cur_sEval − c;
bonus = clamp(−k·theirLoss/64, ±cap); addToHistory(mainHist[~stm][prevMove])`
(`search.cpp:840-844`). Coda only skims this on fail-lows into *cont*-hist
(`FAIL_LOW_PREV_BONUS`, `search.rs:5814`) — different table, narrower trigger.
Verified genuinely absent. **Ordering-only change (no new pruning branch),
`[0,3]` SPRT with a light history retune-on-branch.** Effort low-medium; the
`static_evals[]` stack + prev-move-quiet detection already exist.

### B. Continuation-history `isCap` split — structural
Index cont-hist by whether the **parent** move was a capture:
`cont_hist[isCap][prev_piece][prev_to][piece][to]` (Obsidian `search.cpp:338`,
SF too). Quiet-after-capture patterns differ structurally from
quiet-after-quiet; conflating dilutes both. **Verified absent** in Coda
(grep = 0). This is a STRUCTURAL change — the class Coda's ordering has
historically been *receptive* to (unlike the value tweaks that H0'd in the
Reckless audit). Effort medium (doubles cont-hist array, threads a per-ply
`played_cap` bit — Coda already tracks `undo.captured` — into the sub-table
selection in the 3 MovePicker ctors + update sites). Risk: ~2× cont-hist
memory (largest history table — check footprint/NPS); tree-shape → retune.

### C. Cross-*move* score-trend term in TM — LTC-relevant
Obsidian carries `searchPrevScore` (best score of the *previous* `go`) across
moves and folds `~0.025·(prevScore − score)` into its time factor
(`search.cpp:1466,1531`). Coda's `tm_prev_score` resets each search → only
*within-search* iteration trend. A game-horizon term gives more time when the
position has been deteriorating across MOVES — where LTC games are lost. Low
effort; **TM change → validate via local cross-engine RR + non-regression
SPRT** (self-play SPRT undersells TM changes).

### Already in flight / already owned
- **Aspiration `score²`-div**: Obsidian's is 13000 vs Coda's 33378 (2.6× larger
  → widens far less at LTC scores) — confirms the "dead window at LTC" read.
  **Already being addressed** in the Track-A LTC tune (#2620 seeds
  ASP_SCORE_DIV at 15000).
- **LMR flatter basin**: Obsidian base 0.99/div 3.14 vs Coda 0.21/div 1.54
  (~2× steeper slope) — the flatter basin is the on-theme target, but this is
  **Hercules's LMR reshape + LTC retune** domain (and Coda's correction battery
  + deep-spine machinery already narrow the gap; Coda leads on the mechanism).
  Confirms Hercules's direction; not a separate Fable lever.

### Ruled out / off-theme
- Material/phase eval scaling — already present (agent false positive).
- Move-into-threat penalty — = Reckless threatened-TO malus, already H0 (#2593).
- TM per-move fraction scaling — Obsidian doesn't do it either.
- Triple extension — Coda deliberately removed (`search.rs:385`); high tree-shape
  risk; only revisit if LMR-basin work doesn't restore deep-spine density.
- 50mr-aware TT key — Coda mitigates via halfmove-independent eval + hmc≥89 gate;
  park unless a TT-draw bug surfaces.

## Where Coda is already ahead of Obsidian (don't re-study)
- **Deep-spine LMR protection** (LMR_ROOT + all-node depth decay) — Obsidian has
  neither. The single most important deep-regime area, and Coda leads.
- **NNUE threats** (x-ray, +187 Elo) — Obsidian has zero in-net enrichment.
- **Main history** 4D threat-aware vs Obsidian 2D butterfly.
- **Correction history** 5 sources (+ transition, + fortress material-damping)
  vs Obsidian 4.
- **Move-ordering signal density** (offense, knight-fork, discovered-attack,
  SEE-gated check, graded escape) vs Obsidian's lean ~40-line scorer.
- **TT SMP correctness** (lockless XOR-atomics, ARM-correct, 5 slots) vs
  Obsidian's non-atomic racy 3-slot.
- **Output-bucket layout**, **L1 width 32 vs 16**, **native VNNI + NNZ kernels**,
  **RFP deep-knee**, **contempt removed** (decided).

## Recommended order
A (EvalHist) → B (cont-hist isCap split) — both ordering/structural, verified
absent, strong SF+Obsidian priors, cheap. C (cross-move TM trend) is a good
LTC-relevant follow-up but needs local-RR validation. Name any branches by
mechanism (`eval-swing-history`, `conthist-iscap-split`, `tm-cross-move-trend`),
not by engine.
