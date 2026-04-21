# Understanding Coda's Move Ordering — Deep Investigation (2026-04-19)

## Status as of 2026-04-21 — ABSORBED (diagnostic doc)

This doc's diagnostic is complete — root cause identified as nonlinear
eval flattening the best-move-separation distribution, not TT-move
unreliability. The actionable follow-up is `caphist_retune_proposal_
2026-04-19.md` (proven 72% → 82% first-move-cut in diagnostic, awaiting
rebase + retune).

Downstream wins that draw on this investigation's framing:
- **B1 discovered-attack #502 +52 Elo** — specific-tactical-motif
  movepicker bonus (vs generic scoring nudge). The doc's finding that
  "best move is in top-k but not #1" maps directly to why targeted
  ordering bonuses work: they promote specific good moves that are
  currently mis-ranked.
- **#554 offense-bonus +5.7, #578 tuned-knight-fork +5.2** — same
  pattern: specific-motif ordering bonuses land.
- **Generic scoring nudges** (enter-penalty, hanging-escape) all H0'd.
  Reinforces the doc's conclusion that surface-level "prefer moves
  from attacked squares" isn't the lever — specific tactical motifs are.

No standalone follow-ups. Next concrete action: execute
caphist_retune.

---

**Goal**: understand what move ordering is doing for Coda, why it degraded from v5 (82.3% first-move cut) to v9 (72.2%), and what hidden-layer engines do differently.

**Not the goal** (yet): immediate fixes. This doc is about understanding the mechanism. Actionable recommendations flow from understanding, not guessing.

---

## 1. The core observation

| Net | Architecture | First-move cut | Δ from v5 |
|---|---|---|---|
| v5 | linear FT → output | **82.3%** | — |
| v7 (1024h16x32s-w0) | hidden layers, no threats | 68.7% | −13.6 pp |
| v9 w0 | hidden layers + threats | 73.6% | −8.7 pp |
| v9 w15 (best) | hidden layers + threats | 72.2% | −10.1 pp |

**Same ordering code throughout.** Only the eval differs.

**What this tells us**:
- Hidden-layer non-linearity costs ~13.6 pp of first-move-cut rate.
- Threats *recover* ~4 pp of that loss (v7 68.7% → v9 72%). Not the other way around.
- WDL blending contributes <2 pp.

**What this does NOT tell us**:
- Why hidden layers specifically degrade ordering — is it TT move reliability? History table noise? Movepicker bonus miscalibration?
- Whether Coda's current movepicker is optimal for the v9 eval or whether it's a v5-era artifact.

## 2. Why would hidden-layer eval degrade the same ordering code?

### Mechanism: flatter eval distribution per-position

Prior eval-variance investigation on 30 WAC positions:

| Metric | v5 | v9 w0 | v9 w15 | Interpretation |
|---|---|---|---|---|
| Mean std of child evals | 207 | 238 | 248 | v9 has *wider* absolute spread |
| Mean `gap_n` (best vs 2nd-best, in σ) | **1.82** | 1.36 | 1.39 | v9's best move is less separated from peers |
| Median `gap_n` | 1.79 | 1.01 | 1.06 | 40-42% reduction in winner-takes-all clarity |
| Top-1 agreement v5 vs v9 | — | 60% | 60% | genuinely different rankings, not just less confidence |

So v9 produces:
- Wider absolute spread of evals across moves (more variance)
- But LESS concentration at the top — the best move is only ~1σ above the pack, not ~1.8σ

**Analogy**: picture two bell curves centered on the "best move" score. v5's curve is tall-and-narrow (one clear winner). v9's curve is wider but flatter (many close contenders).

### Why this hurts pre-search ordering

Move ordering signals (history, MVV+captHist, bonuses) have finite discriminative power. They're noisy estimators of "which move is best".

- Under v5's peaked distribution: noise in the ordering signal doesn't matter much. Even a moderately noisy signal points at the right move because the right move stands clearly above peers.
- Under v9's flatter distribution: the signal-to-noise ratio has to be *higher* to identify the correct top move from close peers. Same signal quality → more ordering errors.

This is the mechanism. **Coda's ordering code isn't worse for v9 — it's the same quality signal, but it's being applied to a problem where that quality is no longer sufficient.**

## 3. What the other hidden-layer engines do

Survey of quiet move scoring across the 3 strong hidden-layer engines + Coda:

### Scoring components — full comparison

| Component | Reckless | Obsidian | PlentyChess | Coda v9 |
|---|---|---|---|---|
| **main_history (threat-indexed)** | ✓ `quiet_history.get(threats, side, mv)` | ✓ 2D `mainHist[side][from-to]` | ✓ (internal) | ✓ **4D** `[from_thr][to_thr][from][to]` |
| **conthist plies 1,2,4,6** | ✓ all at 1× weight | ✓ all at 1× weight | ? | ✓ plies 1,2 at `CONT_HIST_MULT=3`, plies 4,6 at 1× |
| **pawn_history** | — | ✓ | — | ✓ |
| **Escape-from-threat** | ✓ **attacker-type-stratified** `[0, 8k, 8k, 14k, 20k, 0]` per moving-piece | ✓ **attacker-type-stratified** ±32k/±16k/±16k | ✓ **attacker-type-stratified** Q/R/Minor | ✗ pawn-attackers only, moving-piece-type-scaled |
| **Onto-threatened penalty** | ✓ −8000 stratified | ✓ same magnitude as bonus | ✓ same | ✗ not present |
| **Offense bonus** (move attacks enemy piece safely) | ✓ +6000 per-piece-type offense bitboard | ✗ | ✗ | ✗ |
| **Rook into king-ring-ortho** | ✓ +5000 | ✗ | ✗ | ✗ |
| **Quiet check bonus** | ✓ +10000 | ✗ | ✗ | ✓ `QUIET_CHECK_BONUS=8270` |
| **Discovered-attack (x-ray)** | ✗ | ✗ | ✗ | ✓ `DISCOVERED_ATTACK_BONUS` (#502 landed +52 Elo) |
| **Null-move threat-sq escape** | ✗ | ✗ | ✗ | ✓ +8000 |

### What Coda is *uniquely missing* compared to the hidden-layer consensus

1. **Attacker-type-stratified escape bonuses** (3/3 hidden-layer engines have; Coda doesn't). Coda uses moving-piece-type magnitudes (Q > R > Minor) but doesn't condition on attacker type. Reckless's logic is crisper: a queen escaping a rook is different from a queen escaping a pawn; the bonus magnitude encodes "this is a winning exchange if captured".
2. **Onto-threatened penalty** (3/3 have; Coda tried unstratified version #465 which H0'd, stratified SEE-gated version not yet tested)

### What only Reckless uses

3. **Offense bonus**: +6000 for quiet moves that land on squares attacking an enemy piece. Reckless is the only top engine with this. Could be a Reckless-specific invention. Never tested in Coda.
4. **Rook king-ring-ortho bonus**: +5000 for rook moves into squares attacking the enemy king zone orthogonally. Also Reckless-unique.

### What Coda has that consensus doesn't

5. **Discovered-attack (x-ray) bonus**: requires x-ray feature enumeration, which no other engine has. #502 landed at +52 Elo.
6. **Null-move threat-sq escape**: leverages the fact that Coda extracts threat_sq from the null-move's TT after NMP fails. Other engines don't do this.

### Weight magnitudes (informative)

| Engine | Typical escape magnitude | Typical check bonus | Pattern |
|---|---|---|---|
| Obsidian | ±32768 (queen), ±16384 others | — | big, stratified by attacker |
| Reckless | 8k-20k escape / 10k check | 10000 | medium |
| Coda | 10k-16k escape / 8.3k check | — | medium |
| PlentyChess | SPSA-tuned (unknown exact) | — | unknown |

Obsidian's weights are 2-3× bigger than Coda's. Obsidian's simpler movepicker with big weights vs Coda's richer movepicker with moderate weights: different design philosophies.

## 4. Structural insight — pre-search vs post-factum framing

From `ordering_coupled_pruning_2026-04-19.md`:

- **Pre-search (future-predicting)** signals: TT move, MVV + captHist for captures, escape/check/offense bonuses in movepicker. Quality depends on eval smoothness + a-priori heuristics.
- **Post-factum (search-derived)** signals: main history, conthist, pawn-hist, corrhist. Quality depends on how cleanly past searches produced cutoffs.

Under v9's flatter eval distribution:
- **Pre-search signals lose discriminative power** proportionally to how much they depend on eval-derived features. History and bonuses based on raw move attributes (check, on-threatened-square) are less affected. Eval-derived rankings (TT move from prior iteration, MVV-based capture ordering) are hit harder.
- **Post-factum signals also degrade** because past cutoffs were noisier → per-bucket history scores are less precise. But they degrade less than pre-search because they aggregate many observations.

This predicts that Coda's current movepicker reliance on history (4D threat-indexed, pawn-hist, conthist at plies 1,2,4,6) is appropriate for v9, but its pre-search bonuses (unstratified escape, QUIET_CHECK_BONUS, discovered-attack) may be under-utilized for the kind of fine-grained distinctions v9's eval requires.

## 5. PV stability measurement — null result is itself informative

**Setup**: 50 WAC positions + 50 SBD ("Silent but Deadly") positions, run `go depth 3/5/7/9` separately on each, record `bestmove` from each. Compare v5 vs v9 across depths.

**Result**: **100% stability on both corpora for both engines.**

| Corpus | v5 stability (all 4 transitions) | v9 stability | v5-vs-v9 agreement at depth 9 |
|---|---|---|---|
| WAC (tactical) | 50/50 positions | 50/50 | 100% |
| SBD (quiet-positional) | 50/50 positions | 50/50 | 100% |

Both engines find the same move at depth 3 as at depth 9, and they agree with each other on every position. No depth-to-depth churn, no engine-to-engine divergence at the root.

### Confirmation on third corpus (Arasan)

After the initial WAC+SBD null result, re-ran on **100 Arasan positions at depths 3/5/7/9/11** (1000 searches). Same result: 100% stable on both engines, 100% v5/v9 agreement at every depth.

**Measurement is now solid across 200 positions from three different-character test suites.** The finding generalizes.

### Interpretation

The hypothesis was wrong in the way I framed it. The first-move-cut gap is **not** caused by top-level PV instability — v9 finds the same root-level answers as v5, just slower.

**The real mechanism must be at internal nodes during search**, not at the root. Inside the tree, v9 is doing more work to confirm the same conclusion. Specifically:

- The 60% v5-vs-v9 top-1 agreement I measured earlier was at **depth 1** (pure static eval of each child move, no search). That's where the flatter eval distribution shows up.
- At depth 9, search has compensated for the flatter eval by doing more nodes of work. Answer quality converges to v5's.
- The 53% extra bench nodes at depth 13 (v5 2.16M vs v9 3.31M) is the measured cost of this compensation.

### Corrected framing

**v9's eval isn't less correct — it's less efficient.** Search reaches the same answer but burns more nodes doing it. The compensation tax is paid per-node via the weaker first-move-cut rate (72% vs 82%), not via different final decisions.

### Direct diagnostic (2026-04-19): WHERE the first-move-cut gap lives

Added instrumentation to `search.rs` decomposing the first-move-cut rate by move type. Ran bench on v5 and v9 (same binary, same net-switching). Results:

| Cutoff type | v5 | v9 | Δ |
|---|---|---|---|
| TT-move cut (first move is TT move, cuts) | 22.5% | 17.2% | −5.3pp |
| **1st capture cut (no TT, MVV-best cap cuts)** | **41.7%** | **32.2%** | **−9.5pp** |
| 1st quiet cut (no TT, no caps, 1st quiet cuts) | 15.8% | 23.0% | +7.2pp |
| Cut at move 2 | 10.3% | 12.6% | +2.3pp |
| Cut at 3-5 | 6.3% | 9.7% | +3.4pp |
| Cut at 6+ | 3.4% | 5.3% | +1.9pp |
| **TT move available at cutoff** | **29.2%** | **23.0%** | −6.2pp |
| **TT-cut rate WHEN available** | **77.1%** | **75.0%** | −2.1pp |

**Two critical findings from the diagnostic**:

1. **TT move reliability is NOT the problem.** When a TT move is available, it cuts at 77% in v5 and 75% in v9 — essentially equivalent. The hypothesis that v9 stores unreliable TT moves is **false**.

2. **The gap lives in capture ordering at non-TT nodes.** "1st cap cut (no TT)" dropped 9.5pp — by far the biggest component of the 10pp gap. The capture-scoring logic `16 × captured_value + captHist` is less effective at picking cutoff-causing captures in v9.

Secondary: first-quiet cut rose +7.2pp. Mostly because more of v9's tree explores positions without good captures at all (tree is 48% bigger; more "quiet" internal nodes reached).

### Why capture ordering specifically?

- **MVV** (`captured_value * 16`) is eval-independent — same for v5 and v9.
- **captHist** accumulates from prior search results. If v9's eval makes cutoffs less decisive, captHist learns noisier per-(piece, to, victim) statistics.
- Result: the 2nd, 3rd captures by MVV are less reliably distinguished by captHist in v9.

### Actionable implications

This is the most specific diagnostic we have. It points at **capture scoring** as the primary leverage point, not TT or history generally:

- **Capture history (captHist) update schedule or weight** — since this is the post-factum signal for capture ordering that's degrading.
- **Capture pre-search signals** — MVV alone is a weak discriminator; anything that augments MVV with eval-independent info (SEE magnitude? piece activity? attacking-king-zone?) could close the gap.
- **Capture move-ordering re-score** — consider running MVV+captHist-ordered captures through a cheaper NNUE-delta at the top to re-rank the top-K captures. Expensive but directly targets this finding.

The Reckless "offense bonus" discussion is tangential to this finding — offense is a QUIET move signal, addressing the smaller +7.2pp quiet-cut shift, not the bigger −9.5pp capture-cut gap. Still worth testing, but not where the biggest gain lives.

### Implication for Hercules-priority ordering

Given this diagnostic:

1. **Capture history mechanics** are the highest-leverage area to investigate. Any improvement here targets the 9.5pp locus.
2. **Offense bonus / king-ring bonus** (Reckless-unique) address the smaller quiet-cut gap. Still worth testing but not the primary win.
3. **TT move ordering** isn't the issue — don't spend cycles there.

### Deeper investigation (2026-04-19 follow-up)

Adam pushed back on the "same code same SEE values" framing — so what specifically propagates from the net difference to capture ordering degradation? Followed up with several diagnostic threads:

#### (a) Tunable drift from v5 (main) to v9 (trunk) — SPSA compensation

Same binary contains the tunables SPSA has settled on. Compared v5-tuned main vs v9-tuned trunk capture-related parameters:

| Tunable | v5 (main) | v9 (trunk) | Δ | Direction |
|---|---|---|---|---|
| SEE_QUIET_MULT | 18 | 45 | +150% | v9 prunes quiets much less aggressively |
| SEE_CAP_MULT | 120 | 146 | +22% | v9 prunes captures less aggressively |
| BAD_NOISY_MARGIN | 95 | 125 | +32% | v9 keeps more bad captures |
| CAP_HIST_MULT | 193 | 263 | +36% | v9 boosts captHist bonuses more per depth |
| CAP_HIST_BASE | 36 | 15 | -58% | v9 less offset |
| CAP_HIST_MAX | 1474 | 1635 | +11% | v9 slightly higher cap |

**Pattern**: v9-tuned params uniformly lean "less aggressive pruning, bigger compensatory captHist bonuses." This is SPSA compensating for v9's ordering weakness by (a) not pruning on noisier signals and (b) cranking up the post-factum captHist signal.

**Implication**: when I measured "v5 first-move-cut 80%" earlier using the v9 trunk binary, I was running v5 net with v9-tuned params — not canonical v5. The canonical v5 (main binary + v5 params + v5 net) is the 82.3%. My diagnostic showed "80% with v9-tuned params on v5 net" — still valid for the decomposition, but the 82.3% vs 72.2% gap is the right apples-to-apples reference.

#### (b) The `experiment/caphist-defender` H0 — retune candidate

While investigating, found an already-tested attempt: [commit 8d0c1fc](../search.rs). Added a `defended` dimension to capture history: `capture[piece][to][captured][defended]` where `defended=1` iff `to` square is attacked by any enemy piece.

**This is exactly Reckless's `noisy_history[piece][to][captured][to_threatened]` structure.** Reckless has it, Coda briefly had it.

Result:
- **First-move-cut rose to 82.0%** (fully closing to v5's canonical level!)
- **Bench dropped -22%** (2.64M → 2.06M, major tree efficiency gain)
- **SPRT #483: H0 at −1.5 Elo after 14520 games** — **no retune attempted**

**This is a textbook retune-on-branch case** per CLAUDE.md's methodology:

> **Retune-on-Branch Methodology**: Some features are neutral without retuning but gain significant Elo when pruning parameters are recalibrated on their branch.
> Pattern: "big bench/node change but flat Elo → retune candidate"

-22% nodes is a massive tree shape change. The v9 pruning params (SEE_QUIET_MULT=45, BAD_NOISY_MARGIN=125, etc.) were tuned against a 2.64M-node tree. With caphist-defender the tree is 2.06M and those params over-prune what's still needed.

**Hypothesis (testable)**: retune-on-branch of caphist-defender would convert the −1.5 Elo result into a positive H1. SPSA has already demonstrated it can compensate for ordering quality (the v5→v9 drift); given a BETTER ordering signal (caphist-defender), the compensation unwinds back toward the cleaner v5-tuned direction.

**Recommended action**: kick off a focused SPSA tune on `experiment/caphist-defender` branch for the 18 standard pruning params (NMP, RFP, futility, SEE, LMP, history pruning, LMR coefficients). If SPSA pulls params back toward v5-tuned values (lower SEE_QUIET_MULT, tighter BAD_NOISY_MARGIN, smaller CAP_HIST_MULT — less compensation needed), that validates the retune-on-branch thesis. Then SPRT tuned values.

This is a **specific, concrete, immediately-actionable recommendation** that emerged from the diagnostic + historical investigation.

### Additional data point: Reckless's noisy_history is exactly this structure

Reckless uses `[piece][to][captured_piece_type][to_threatened]` — 4D with the to-threatened bit. Factorized with a separate global piece-to score.

Same mechanism Coda tried in caphist-defender. Reckless has it landed because their pruning params were tuned WITH it. Coda's unretuned attempt failed.

**Coda has the feature built; just needs retune to land.**

Caveat: this measurement covers 100 middlegame positions from two test suites. It's possible in **noisier positions** (near-equal endgames, wild complications) v9 and v5 diverge more. A larger-corpus measurement with more-varied positions could surface those cases. But the base-rate finding — "on defined-best-move positions, they agree" — holds solidly on this sample.

### What this doesn't rule out

- **Internal-node TT move quality** could still be worse on v9. The root's PV from depth N-2 is just one TT entry; most TT entries in the tree are at internal nodes. Measuring those would require code instrumentation (Hercules's lane).
- **Short-of-PV move quality** — the 2nd-best, 3rd-best, ... moves' scores — is almost certainly less-discriminated on v9, consistent with the flatter eval distribution. Bench stats (avg cutoff pos 2.10 vs 1.57) already confirm this.

### Correction to Section 2's mechanism claim

I said "Coda's ordering code isn't worse for v9 — it's the same quality signal, but it's being applied to a problem where that quality is no longer sufficient." That's still accurate but needs refinement:

The inefficiency is **per-internal-node**, not per-iteration-at-root. v9's ordering code at each internal node has to search 1.34× more moves on average (2.10 vs 1.57 cutoff position) to find the cut, compounding across the tree into 53% more total nodes.

## 6. Corrhist comparison (validates "strong post-factum infrastructure" claim)

Correction-history sources across engines:

| Engine | # sources | Sources |
|---|---|---|
| **Coda** | **5 (6 applications)** | pawn, whiteNP, blackNP, **minor**, **major**, cont |
| Viridithas | 5 | pawn, major, minor, nonpawn, cont |
| Reckless | 5 | pawn, minor, whiteNP, blackNP, cont |
| Alexandria | 4 | pawn, whiteNP, blackNP, cont |
| Obsidian | 3 | pawn, nonpawn (2x), cont |

Coda sits at the rich end. Its minor+major split is more granular than any other engine's structure. This validates the claim that **Coda's post-factum infrastructure is built out to handle a lot of signals**.

Per-signal weight magnitudes (raw weight / apply-divisor / grain-divisor to get effective influence):

| Engine | Effective per-unit influence (pawn source) |
|---|---|
| Obsidian | ~0.059 (30 / 512) |
| Coda | ~0.030 (301 / 1263 / 8) |
| Viridithas | formula-incompatible |

Obsidian puts ~2× the weight per-unit, but Obsidian has fewer sources. Coda has more sources each lightly-weighted. Both are valid — they're different bets about where correction signal lives (concentrated in few channels vs spread across many).

Given v9's flatter eval distribution, **Coda's many-lightly-weighted strategy looks better suited** — aggregating across sources reduces noise, whereas Obsidian's fewer-heavily-weighted strategy works best when each source is high-SNR.

## 7. Key takeaways — understanding-focused

These are *mechanistic* claims about move ordering on v9, not fixes.

1. **The 10pp first-move-cut gap is mechanical, not algorithmic.** v9's eval is not less correct; it's less peaked. Same ordering code gets hurt by the geometry of the eval surface.

2. **At root, v9 and v5 agree 100% on best moves.** Measured on 100 middle-game positions (50 WAC + 50 SBD) at depths 3-9. The inefficiency is **internal-node**, not root-level — v9 reaches the same answer but pays 53% more nodes getting there.

3. **Hidden layers cost 13.6 pp; threats recover 4 pp.** Threats aren't the variance source the earlier hypothesis suggested — they actively help.

4. **Coda's movepicker is missing consensus features from hidden-layer engines**, specifically:
   - Attacker-type-stratified escape bonuses (3/3 hidden-layer engines have)
   - Onto-threatened penalty (3/3 have; tested unstratified version failed)
   - Offense bonus (Reckless-unique but plausibly applicable)

5. **Coda has unique features other engines lack**:
   - Discovered-attack via x-ray (#502 landed +52 Elo)
   - Null-move threat_sq escape bonus
   - 4D threat-indexed main history

6. **Coda's post-factum infrastructure is the richest among top engines** (5+ corrhist sources including minor/major split, 4D history, pawn-hist, conthist at plies 1/2/4/6). The many-lightly-weighted strategy matches the flatter eval of v9.

7. **The pre-search vs post-factum framing predicts** Coda's relative strengths and weaknesses: strong post-factum infrastructure (validated), weak pre-search bonuses for v9's fine-grained discrimination needs (validated by the missing stratified signals).

## 7. What NOT to conclude

- Don't conclude "Coda's movepicker is bad". It was built to match consensus; the eval changed under it.
- Don't conclude "just port all Reckless signals". Several of those (stratified onto-threatened penalty) were tested in unstratified form and failed. Porting without understanding costs time; porting with understanding of the coupling (pre-search + escape-stratification + SEE-gate) is the productive move.
- Don't conclude "retrain the eval to be smoother". v9's eval is stronger than v5's — the eval tradeoff is correct; the ordering code is what needs to adapt.

## 8. Ongoing work / follow-up threads

- **PV stability measurement** (Section 5) — numbers pending.
- **Stratified escape + SEE-gated onto-threatened** — Hercules experimenting; #501 post-SPSA retest running.
- **Shape experiments** (`shape_experiments_proposal_2026-04-19.md`) — Hercules has `experiment/history-shape-offset` in flight.
- **#502 discovered-attack-bonus** — landed +52 Elo. One of the unique Coda signals paid off hugely.

## 9. Methodology notes

- For cross-engine comparison, 5 engines is the minimum for trustworthy consensus claims (per `tunable_anomalies` second pass).
- For tunable anomalies, check for *coupling* to known Coda properties before recommending SPRT at consensus value (per `ordering_coupled_pruning`).
- For move ordering specifically, the ordering signal's quality and the eval surface's geometry are both relevant — looking at one without the other produces wrong conclusions.

## Companion docs

- `tunable_anomalies_2026-04-19.md` — cross-engine tunable comparison
- `ordering_coupled_pruning_2026-04-19.md` — move ordering → pruning compensation with pre-search/post-factum framing
- `signal_context_sweep_2026-04-19.md` — signal × context sweep matrix
- `move_ordering_ideas_2026-04-19.md` — move-ordering ideas catalogue (earlier)
- `threat_ideas_plan_2026-04-19.md` — threat-signal experiments with results
- `shape_experiments_proposal_2026-04-19.md` — formula-shape experiments (Hercules handoff)
- *This doc* — understanding what move ordering is doing on v9 (deep investigation)
