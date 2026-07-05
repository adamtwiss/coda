# SF-vs-Coda Search Audit — 2026-07-05

**Trigger:** Adam's TC/nodes/depth decomposition (2026-07-05). At FIXED NODES
(no TM, no NPS) SF leads −51 @ 20k nodes growing to **−79 @ 100k** — SF's
quality-per-node advantage COMPOUNDS with budget. At fixed depth: Coda +110
@ d8, −28 @ d12 — SF's nominal plies are thin, Coda's dense. The search
domain is not mined out; this audit maps where the remaining Elo lives.

**Method:** five parallel audit agents compared **latest SF master
(`d5bbc6b6`, 2026-07-03, /tmp/Stockfish)** against Coda main across:
reductions/extensions, move ordering/history, pruning gates,
TT/aspiration/eval-pipeline, and a novelty diff vs the ~March checkout.
All agents cross-checked June audit docs + experiments.md so prior H0s are
respected (or explicitly challenged with receipts). Full detail lives in the
agent transcripts; this doc is the ranked synthesis.

---

## The mechanism story (why SF wins per node)

Three pillars, all budget-compounding:

**1. Thin but protected.** SF prunes/reduces harder than Coda almost
everywhere a tree is big — depth-PROPORTIONAL quiet thinning
(`conthist < −4313·d`), history-shifted lmrDepth feeding futility/SEE
(bad-history moves pruned at ANY nominal depth), capture futility that
prunes SEE-POSITIVE captures, an NMP eval-gate that RELAXES with depth
(`beta + 374 − 14·d`), and a depth-DECAYING all-node reduction
(`r·272/(256d+285)` ≈ +26% @ d3 → +5% @ d20, vs Coda's flat +1 = a full
extra ply at d20). What makes the aggression safe is spine protection:
**followPV** (previous-iteration PV line is exempt from IIR and, at PV
nodes, from the entire quiet-pruning block), TT child-verification on deep
cutoffs, and verification searches. Coda is the mirror image: denser
interior (less thinning, flat penalties that don't decay) and NO spine
protection — we pay for breadth everywhere and still lose the one line
that matters to reduction/pruning erosion at depth.

**2. Learn from every node.** SF harvests ordering signal from fail-lows
(prior-countermove bonus — the MAJORITY node class in a big tree),
alpha-raises/PV-exact bests, eval-deltas at every static-eval node, a
global ttMoveHistory reliability scalar, and scales bonuses by
searched-move count. Coda learns ONLY from beta cutoffs + TT cutoffs. In a
100k-node tree most nodes are fail-lows — SF updates there, Coda is
silent. Signal density scales with tree size; so does the gap.

**3. TT hygiene at scale.** Deep trees are TT-dominated. SF spends effort
keeping the TT TRUSTWORTHY at saturation: child-consistency verification
before deep cutoffs (d≥7), +2·pv replacement protection, terminal/TB facts
stored at depth+6, secondary aging. Coda's TT is well-audited (margin-4,
EXACT-override, near-miss, node-type guard all banked) but has none of the
verification/retention mechanisms. Separately: Coda's TT-cutoff history
bonus is FULL cutoff magnitude vs SF's ~0.46× — and TT-hit rate RISES with
node count, so over-crediting TT cutoffs progressively dilutes real search
evidence: the one finding whose distortion PROVABLY grows with budget.

---

## Ranked action list

Discipline: one change per branch; `[0, 3]` default; retune-on-branch when
bench moves >15-20%; items marked LTC should SPRT at 40+0.4 with STC
cross-check.

### Tier 1 — fire now (cheap, untested, deep-regime-direct)

| # | Item | Source | Size |
|---|------|--------|------|
| 1 | **followPV spine protection.** Thread `follow_pv` (parent followed ∧ move == previous-iteration PV[ply]; `stable_pv` infra exists). Probe (a): IIR gate `!follow_pv`. Probe (b): quiet-prune block exemption `!follow_pv \|\| !is_pv`. Two branches. | SF search.cpp:752-755, 1030, 1180. Flagged independently by 4/5 agents. Untested (post-dates June audits). | S |
| 2 | **Fail-low prior-countermove history bonus.** On fail-low with quiet prev-move: bonus `min(141·d − 82, 1472)`-class to conthist(prev)+mainHist(opponent); capture-prior analog +901. Start with the simple core, skip statScore term. | SF search.cpp:1523-1553. Coda: zero history writes on the majority node class. Untested. | S-M |
| 3 | **TT-cutoff child-verification (d≥7).** Before a deep TT cutoff: make ttMove (board-only), probe child TT, unmake; cut only if child agrees or is absent. Measure NPS via bench first. | SF search.cpp:873-892. Untested; tightens where #2124/#2300 loosened (both H0). | S |
| 4 | **TT-cutoff bonus down-weight.** Coda credits TT cutoffs at FULL history bonus; SF at ~0.46× cap. Add `TT_CUT_BONUS_PCT` (default 100, probe 50). | SF :860-868 vs search.rs:3549-3573. Trivial; provably budget-growing distortion. | XS |
| 5 | **`improving \|= static_eval >= beta`** after NMP — upgrades LMP/(2−imp), ProbCut margins, LMR !improving for the whole move loop. | SF :1025. One-liner, untested (flagged by 3 agents). | XS |

### Tier 2 — next wave

| # | Item | Notes |
|---|------|-------|
| 6 | **History credit on alpha-raise/PV-exact bests** — move the update block from `alpha>=beta`-only to `best_move exists` (SF :1515-1518); PV spine bests currently uncredited. | S |
| 7 | **Depth-decaying all-node reduction** — replace flat +1 with SF's `r·K/(256d+285)` shape. Integer-grain interim: all-node +1 only at depth ≤ D (tunable). | Pairs with the fractional-LMR enabler below. |
| 8 | **IIR min-depth LTC probe** — Coda fires at eff. depth 4 (STC-SPSA'd, wants 2); SF at 6 with an explicit "aggressive IIR scales poorly" scaler note. One knob 37→60, **LTC-target**. | Natural bundle with #1a. |
| 9 | **Damped RFP return** — return `(716β + 308·eval)/1024`-class blend instead of raw `eval − margin`; bounds RFP optimism propagation in big trees (RFP = Coda's dominant pruner, 301/Kn; shallow-FP rate 42-45%). Also unblocks re-testing P3.1 TT-refined eval. | S |
| 10 | **Singular `depth++` node bump** — deepen the whole node (all later siblings) when TT move proves singular (SF :1239-1251). The one untested item left in the SE block. | XS |
| 11 | **NMP depth-relaxed eval gate** (audit N4, never ran) — `eval >= beta + BASE − PER_DEPTH·d − 45·imp`; deep quiet nodes get NMP cutoffs Coda's flat gate refuses. | S + NMP cluster SPSA |
| 12 | **History→lmrDepth for SEE-quiet threshold** (SEE audit S2, never ran) — `lmr_d += hist/DIV` before the quadratic SEE bar; per-move thinning discrimination Coda lacks. | S + retune |
| 13 | **Capture futility SF-shape** (SEE audit S3, never ran) — stage (a): add `see_value(captured)` (+captHist) to BNFP margin. Stage (b): drop SEE<0 gate, key on lmr_d<7. | S then M |

### Tier 3 — enabler-gated / retune-heavy / re-tests with cause

- **Land `experiment/lmr-fractional`** (H1 non-reg #2192 + LTC cluster #2195
  already applied) — the ENABLER for: LMR base-offset + moveCount linear
  rebate (SF's `+1027 − 62·mc` shape; prior base H0s #2225/#2243 were
  integer-grain-confounded), the `delta·617/rootDelta` window term, and the
  optional full-shape cutoffCnt retry (prior H0s #2226/#2238 tested 4× SF's
  magnitude without the ttMove else-branch).
- **ContHist pruning re-test** (`hist < −K·d`): all three H0s (#1562/#1691/
  #1697, −7..−9) PREDATE the #2432 probcut-conthist-pollution fix whose log
  says "unblocks cont-hist-sensitive experiments." Confound-suspect; re-test
  with retune-on-branch, note the invalidation in the submission.
- **Optimism, retune-first** (SF :378-380 + evaluate.cpp:54-60): dynamic
  root-score-derived, material-weighted, sign-symmetric — NOT the removed
  static contempt. #671's untuned H0 isn't a fair verdict; port + focused
  SPSA before the verdict SPRT. Must live in the single eval space (all
  consumers) per the material-scaling v2 post-mortem.
- **Conthist `[in_check][is_capture]` context split** (SF history.h:241,
  4 universes vs Coda's 1) + micro-branch first: in-check write truncation
  to plies 1-2 (SF writes only 1-2 in check).
- **|correction| into LMR (`r −= |cv|/K1`) and RFP margin (`+|cv|/K2`)** —
  Coda computes the signal, consumes it only in DEXT. Two near-free lines;
  rescale divisors (~4× smaller corr authority than SF).
- **Draw-score randomization ±1** (`VALUE_DRAW − 1 + (nodes & 0x2)`) — 3
  lines, directly on the draws-not-losses LTC axis; contempt-era "random
  noise" prior is stale. LTC lens.
- **Terminal/TB TT stores at depth+6** (mate/stalemate results currently
  return WITHOUT storing; TB cutoffs too) — `[-2,1]` + local EGTB RR (OB is
  TB-blind).
- **Minor-piece corrhist source** (last untested corrhist source; SF weight
  ~70% of pawn's) — CORR cluster retune.
- **Eval-diff history update** (SF :957-965, every static-eval node) —
  stale H0s #517/#552 (2026-04-20, ~15 trunk generations ago); SF's current
  form has new terms (+62 bias, !ttHit pawn gate). Re-test framed as such.
- **Searched-count bonus scale** (`bonus += bonus·moves_searched/256`,
  non-PV; SF Jun-6) + **ttMoveHistory** (global TT-reliability scalar →
  DEXT margin) — two cheap history-signal upgrades.
- **TT micro-bundle**: `+2·pv` same-key replacement bonus (T7) + ttPv
  fail-low parent propagation — one bundled `[0,3]` shot (sibling change
  #2450/#2472 was neutral; expectations tempered).

### Deployment-only (OB-blind — local-rr T=4 / lichess)

- **Shared histories across SMP threads** (SF: NUMA-shared atomic conthist/
  pawn/corr) vs Coda per-thread+copy — T=4 lever, SPRT-invisible at T=1.
- Aspiration per-thread delta jitter (`threadIdx % 8`).
- Root instant-stop on proven mate; forgotten-mate PV rollback (analysis QoL).

### Closed by receipts (do NOT re-fire)

Small-probcut TT return (#884 H0), TT penalize-on-mismatch (#2330 H0),
aspiration center-on-average (#2503 H0), TT-refined pruning eval (P3.1,
−9.5, blocked on undamped RFP — see Tier-2 #9), residual corrhist training
(#2453/#2500), paired cont-correction (#2317), malus shape/3× (#1047/#1068
−39.6), lowply/factorised history (#887/#889), hist-prune as tested
(removal was +3.0 — the RE-test above is the SF shape with the pollution
fix, a different claim), LMR-extension via negative r (#865 −8.0), SEE
capture ceiling raise standalone (#2117), cutoffCnt as tested (#2226/#2238
— only the full-SF-shape fractional variant remains arguable), QS family
(audited/banked), LMP family (closed; only improving-augment touches it),
NMP verification depth (#2268), helper depth offsets (removed; SF agrees).

---

## Cross-agent contradiction resolutions

- Small ProbCut TT-lower cut: pruning agent flagged as untested; TT agent
  produced the receipt (#884 H0 −0.9) → CLOSED stands.
- TT penalize: novelty agent ranked it top-3; TT agent had #2330 H0 −0.3 →
  CLOSED stands (secondary aging variant also deprioritized — SF's own
  comment says it needs VVLTC to verify).
- Draw randomization: novelty "untested" vs TT agent's contempt-era prior →
  prior is stale and about a different mechanism (random noise vs 3-fold
  dither); kept, priced small, LTC lens.

## Reading the decomposition against this list

The fixed-nodes slope (−51→−79) is consistent with pillar 2+3 (signal
density + TT trust both scale with nodes); the fixed-depth flip (+110 @ d8
→ −28 @ d12) with pillar 1 (Coda's dense plies win when depth is capped
low; SF's thin-but-protected plies win as depth grows). If the Tier-1 set
lands even half its expected value, re-running Adam's node-sweep is the
verification instrument: the −51→−79 slope should flatten.
