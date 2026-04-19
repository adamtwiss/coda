# Understanding Coda's Move Ordering — Deep Investigation (2026-04-19)

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

## 5. PV stability measurement — in progress

Running 50 WAC positions at depths {3, 5, 7, 9} on v5 and v9. Measures how often the best move at depth N persists to depth N+2.

**Hypothesis**: if v9's eval produces less clear cutoffs, iterative deepening's best move should be less stable across iterations — a depth-N best move is less often the depth-N+2 best move for v9. This would mean **TT moves are less reliable** in v9, which is what makes the first-move-cut rate drop.

Measurement pending. Will update this doc with the actual numbers when the data collection finishes (~30-60 min).

## 6. Key takeaways — understanding-focused

These are *mechanistic* claims about move ordering on v9, not fixes.

1. **The 10pp first-move-cut gap is mechanical, not algorithmic.** v9's eval is not less correct; it's less peaked. Same ordering code gets hurt by the geometry of the eval surface.

2. **Hidden layers cost 13.6 pp; threats recover 4 pp.** Threats aren't the variance source the earlier hypothesis suggested — they actively help.

3. **Coda's movepicker is missing consensus features from hidden-layer engines**, specifically:
   - Attacker-type-stratified escape bonuses (3/3 hidden-layer engines have)
   - Onto-threatened penalty (3/3 have; tested unstratified version failed)
   - Offense bonus (Reckless-unique but plausibly applicable)

4. **Coda has unique features other engines lack**:
   - Discovered-attack via x-ray (#502 landed big)
   - Null-move threat_sq escape bonus
   - 4D threat-indexed main history

5. **The pre-search vs post-factum framing predicts** that Coda has relatively strong post-factum infrastructure (good history tables) and relatively weak pre-search bonuses for v9's fine-grained discrimination needs. The missing stratified signals would directly address this.

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
