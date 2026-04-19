# Move Ordering → Pruning Compensation — Cross-Parameter Analysis

## Origin

SPRT #519 tested `LMP_BASE=3` (SF/Obsidian consensus) vs Coda's current
value of 13. **Crashed at −44 Elo after 206 games**, LLR −1.15 trending
strong H0. This confirmed: SPSA's drift of LMP_BASE from 3 → 13 was
**compensation** for Coda's worse move-ordering quality, not drift like
contempt was.

Mechanism: Coda v9's first-move-cut rate is ~72% vs SF/Obsidian's
~82-85%. When the "best" move is often not move #1, LMP has to let
more moves through before pruning, or it cuts legitimate refutations.
LMP_BASE=13 at d=3 gives threshold 11 vs consensus threshold 6 — 83%
more moves searched before pruning.

This doc asks: **does the same compensation pattern show up in other
move-ordering-coupled pruning parameters?**

Findings: yes for 4 parameters, ambiguous for 2. The pattern is real
but not uniform — different pruning mechanics react differently to bad
ordering.

## Parameters examined

Reference engines: Stockfish dev, Reckless, Obsidian, Viridithas,
Alexandria. Where formulas aren't directly comparable, noted.

### Category A — confirmed compensation

These params are tuned in the direction of "prune less aggressively"
relative to consensus.

#### A1. LMP_BASE (confirmed by SPRT #519)

| Engine | Base |
|---|---|
| Stockfish | 3 |
| Obsidian | 3 |
| Viridithas | ~2.5 |
| **Coda** | **13** |

Same formula `(base + d²) / (2 - improving)`. Coda's base is 4× consensus.
At low depths, Coda searches 1.5-2.5× more moves before LMP fires.

**Mechanism**: worse first-move-cut → real cutoff moves are often later
in ordering → need bigger move-count budget before LMP. Coda's base=13
is a correct compensation.

**Test result**: LMP_BASE=3 crashed at −44 Elo after 206 games (#519).

#### A2. SE_DEPTH (starts SE earlier)

| Engine | SE starting depth |
|---|---|
| **Coda** | **5** |
| Reckless | 5-6 |
| Obsidian | ~6 |
| Viridithas | ~6 |
| Alexandria | ~8 |
| Stockfish | ~5 (recently lowered) |

Coda sits at the low end of consensus (5-8). SE fires earlier → TT
move uniqueness verified more often.

**Mechanism**: worse TT-move-correctness rate (a symptom of weaker
ordering — when iterative deepening's stored move is less reliable)
means SE's verification is more valuable at shallower depth. Coda's
low SE_DEPTH compensates by running SE more frequently.

**Status**: not an outlier (consensus spans 5-8) but sits at low
bound, consistent with compensation.

#### A3. SEE_QUIET_MULT (less aggressive SEE pruning on quiets)

| Engine | Formula for quiet SEE threshold | Coefficient |
|---|---|---|
| **Coda** | `-SEE_QUIET_MULT · d²` | **−45** |
| Obsidian | `PvsQuietSeeMargin · lmrDepth²` | **−21** |
| Stockfish | (different, razoring-scaled) | — |
| Viridithas | flat `-93` | — |

Both Coda and Obsidian use `d²` scaling. Coda's coefficient is 2.1× Obsidian's, meaning Coda's threshold is more permissive (only prunes quiet moves that lose MORE material before pruning).

**Mechanism**: with worse ordering, a quiet move that appears to lose material per SEE might still be the refutation. Coda's higher coefficient compensates by requiring a bigger apparent SEE loss before pruning.

**Status**: first pass flagged this as "structural formula difference." Second look — same formula, 2× coefficient. Consistent with compensation.

#### A4. HIST_PRUNE_MULT (less aggressive history pruning)

| Engine | Formula (prune if `history < threshold · d`) | Coefficient |
|---|---|---|
| **Coda** | `history < -HIST_PRUNE_MULT · d` | **−5148** |
| Obsidian | `history < HistPrDepthMul · d` | **−7471** |

Coda's threshold is closer to zero → prunes LESS aggressively. Obsidian prunes a move with history score below −7471·d; Coda only prunes below −5148·d.

**Mechanism**: with worse ordering, history scores are noisier (same move in similar positions gets different treatment). Trusting the signal less and pruning less aggressively compensates.

**Status**: consistent with compensation.

### Category B — counter-pattern / ambiguous

These params are tuned in the direction *opposite* compensation, or without a clean cross-engine comparison.

#### B1. LMR_HIST_DIV (more history influence on LMR, not less)

| Engine | Divisor | Relative trust in history |
|---|---|---|
| **Coda** | **7123** | Higher influence (smaller divisor) |
| Obsidian | 9621 | Lower influence |

Coda has *more* history influence on LMR reduction than Obsidian. Under the pure compensation theory this is backward — noisier history should be trusted less → larger divisor.

**Possible explanations**:
- Coda's 4D threat-aware history (indexed by `[from_threatened][to_threatened][from][to]`) is *more* reliable per-bucket than Obsidian's 2D history, despite the overall move-ordering tax. The threat bits partition position state finely, giving each bucket sharper statistics.
- SPSA hasn't fully converged on this axis (just hasn't been re-tuned since another related change landed).
- LMR's reduction-scaling by history is a *continuous* signal, and the compensation math differs from binary cutoffs. See meta-finding below.

**Status**: ambiguous. Does not falsify the compensation thesis because of the 4D indexing argument.

#### B2. LMR C coefficient (reduces more, not less)

| Engine | Formula | Effective reduction at d=10, m=10 |
|---|---|---|
| **Coda** | `ln(d) · ln(m) / 1.24` | **4** |
| Obsidian | `0.99 + ln(d) · ln(m) / 3.14` | 2 |

Coda has no base term; Obsidian has `+0.99`. At high depth/move-count Coda reduces *more* than Obsidian.

Under compensation theory this is backward — with worse ordering, reducing more on late moves risks missing refutations.

**Possible explanations**:
- Coda has many additional LMR reduction adjustments (history, complexity, threat-count, king-pressure, threat-mag) that SUBTRACT from the base table lookup. The raw table value is a starting point, not the final reduction. The additional adjustments may pull Coda's effective reduction closer to Obsidian in practice.
- Different formula families with different effective behaviours.

**Status**: ambiguous, likely because Coda's LMR is a multi-stage stack where the table lookup is just one component.

### Category C — orthogonal (not ordering-coupled)

These don't use `move_count` and don't depend on ordering quality:

- **RFP**: `margin * d`, only uses depth and improving
- **Futility**: `FUT_BASE + lmr_d * FUT_PER_DEPTH`, depth-based
- **NMP**: depth-and-eval-based
- **ProbCut margin**: fixed margin, no move_count
- **Aspiration**: root-only, no move ordering dependence

Tuning differences here (first-pass and third-pass found a few) are explained by other factors (eval variance, eval scale differences, architectural differences in how improving is detected). Not part of this analysis.

## Meta-finding — the pattern isn't uniform

Simple hypothesis: "worse ordering → prune less aggressively everywhere." **The evidence doesn't support this as a universal rule.**

Better hypothesis: the compensation direction depends on how the parameter uses ordering signals.

| Signal type | Compensation direction under bad ordering |
|---|---|
| **Binary cutoff on move-count** (LMP, SE gate) | Loosen threshold / fire earlier |
| **Binary cutoff on SEE** (SEE pruning) | Loosen threshold |
| **Binary cutoff on history** (history pruning) | Loosen threshold |
| **Continuous scaling by history** (LMR hist adjust) | Unclear — depends on whether the history signal itself is more or less reliable for this specific engine's history-table structure |
| **Formula structure** (LMR base table) | Influenced by what other adjustments stack on top |

**Concrete implication**: binary cutoffs are where compensation is cleanest. LMP, SE gate, SEE prune, hist prune are all "loosened" in Coda relative to consensus. This is the compensation tax paid for worse ordering.

Continuous scalings (LMR coefficients, LMR hist div) don't follow the same rule — they respond to ordering quality plus the reliability of the signal they scale on.

## Strategic implications

### 1. Move ordering is a hidden Elo tax

Coda's pruning surface is visibly detuned across 4+ parameters to compensate for bad ordering. Each detuning leaves Elo on the table — more nodes searched at shallow depth, less pruning, bigger tree. The tax manifests as reduced effective search depth.

Rough estimate: if move ordering were at SF/Obsidian parity (~85% first-move cut), the compensation tax could be unwound: LMP_BASE back toward 3, SEE_QUIET_MULT toward 21, HIST_PRUNE_MULT toward 7471. That's probably worth +10-20 Elo from tuning alone, on top of whatever direct Elo the ordering improvement gives.

### 2. Ordering-improvement changes have compounding value

Any merge that improves first-move-cut rate by N percentage points has two sources of Elo:
- Direct: fewer wasted nodes searching non-best moves = better PVs at depth
- Indirect: allows subsequent retune to tighten compensated-loose parameters = more effective pruning

The #502 discovered-attack-bonus landing at +52 Elo is a huge example. Part of that +52 is probably the indirect effect of SPSA retunes absorbing the ordering improvement into tighter pruning.

This suggests a retune cadence rule: **after any win that improves move-ordering stats (first-move-cut, avg-cutoff-pos), retune the ordering-coupled pruning params**. The joint retune should recover extra Elo beyond the raw feature.

### 3. Counterexamples to watch

B1 (LMR_HIST_DIV) and B2 (LMR C) are the interesting counterexamples. If these continue tracking the opposite-of-compensation direction even as ordering improves, it suggests something specific about Coda's history tables / LMR structure that's genuinely different from consensus.

Worth flagging on future retune cycles: do these parameters move toward consensus as ordering improves, or do they stay put? Answer informs whether they're genuinely compensation or just different-formula artifacts.

## What to act on now

- **Don't SPRT more consensus-value reversions on ordering-coupled params** without a coupling check first. LMP_BASE=3 was the clean test; the crash confirms the method works. Re-trying the same on SEE_QUIET_MULT or HIST_PRUNE_MULT would likely crash similarly.
- **Track first-move-cut rate as a leading indicator for retune cadence.** After any merge that meaningfully improves it, expect a retune cycle that tightens 4+ ordering-coupled params in aggregate.
- **When proposing new ordering-improvement ideas** (Tier B/C from the sweep doc, or from Hercules's active branches), note that the expected direct Elo understates total value because of the compensation-retune compounding.

## Companion docs

- `tunable_anomalies_2026-04-19.md` — cross-engine tunable comparison (3 passes). Third pass flagged LMP_BASE=13 as outlier; #519 crash refuted the "drift" interpretation. This doc is the follow-through.
- `signal_context_sweep_2026-04-19.md` — signal × context experiment matrix
- `move_ordering_ideas_2026-04-19.md` — move-ordering-specific improvement ideas
- `shape_experiments_proposal_2026-04-19.md` — formula-shape experiments (Hercules handoff)
- `threat_ideas_plan_2026-04-19.md` — threat-signal experiments with results

## Methodology note — lesson for future analyses

Before proposing "SPRT at consensus value" on an anomaly, ask: *is this parameter coupled to another Coda-specific property that we know is non-consensus?*

In Coda's case, the non-consensus property is move ordering quality. Any parameter that interacts with `move_count` or uses signals derived from ordering should be assumed compensated until proven drift. Contempt was unambiguously drift (no coupling). LMP was unambiguously compensation (direct coupling to move_count).

Rule of thumb: **run the coupling check before writing the SPRT proposal.**
