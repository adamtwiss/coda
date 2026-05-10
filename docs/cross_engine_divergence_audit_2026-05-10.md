# Cross-engine divergence audit — 2026-05-10

Question Adam raised: where is Coda drifting from cross-engine norms,
and is the drift signposting (a) implementation bugs, (b) calibration
to a different operating point because some other feature absorbs the
slack, or (c) genuine novelty that's working for us?

Frame for each divergence:
- **Magnitude** — how big is the |Coda − consensus| gap?
- **Bug-shape** — does the formula structure look broken?
- **Slack-shape** — could another feature be absorbing this divergence?
- **Action** — code fix, focused SPRT, or leave alone?

---

## P1 — Likely formula bug: corr-hist clamp at err vs bonus

**Coda** (`search.rs:1018-1021`):
```rust
let err = (search_score - raw_eval).clamp(-err_max, err_max);  // err_max=1
let weight = (depth + 1).min(weight_max);  // weight_max=9
let cap = CORR_HIST_LIMIT(1024) / cap_div(2);  // = 512
let bonus = (err * weight).clamp(-cap, cap);
```

The err magnitude is clamped to ±1 BEFORE multiplying by depth-weight.
With `err_max=1` (SPSA floor-pinned), a 50cp eval miss banks bonus = ±9
even at depth 8. The cap=512 is essentially never reached.

**Stockfish/Obsidian/Viridithas pattern:**
```
bonus = (err * weight).clamp(-bound, bound)   // err NOT pre-clamped
```

A 50cp miss at depth 8 banks bonus ≈ 450, scaled by magnitude.

**Why SPSA pinned `CORR_HIST_ERR_MAX=1`:** in Coda's formula, larger
err_max lets the err-magnitude through scaled by weight=9, swamping the
banked corr signal at any non-trivial eval miss. SPSA found that
clamping at 1 is the least-bad operating point for the formula shape.

**This is bug-shaped.** The fix is restructuring to clamp at the bonus
level. After the change, `CORR_HIST_ERR_MAX` becomes either redundant
(can be deleted) or a different scale. `CORR_HIST_GRAIN_T=11` (the
2.9× over-divider on the read side, audit doc 5.8) likely needs to
drop to 1 simultaneously, since it was compensating for the bonus
being under-magnitude on the write side.

**Risk:** Coda's eval may be NOISIER than SF/Obsidian — uncalibrated
threat-trained nets give larger raw err magnitudes that need
clamping. The `err_max=1` may be a noise-tolerance adaptation, not a
bug. Distinguishable only via experiment.

**Proposed experiment** (post-current-tunes):
1. Implement SF-style: drop the err pre-clamp; clamp at bonus
2. Drop `CORR_HIST_GRAIN_T` to 1 (or remove the second division)
3. SPSA re-fit `CORR_W_*`, `CORR_HIST_DIV`, `CORR_BONUS_CAP_DIV` (8 params, focused 2K iter)
4. SPRT vs main `[0, 5]` (port + retune, prior is bigger gain)

**Expected gain if real:** +3-8 Elo (per Obsidian retune deltas on similar fix).

---

## P2 — Aggressive low-depth RFP + permissive high-depth allowance

**Coda RFP** (`search.rs:92-94, 2482`):
- `RFP_DEPTH = 18`, range `2..20` (near upper ceiling)
- `RFP_MARGIN_IMP = 30 × depth`
- `RFP_MARGIN_NOIMP = 84 × depth`

**SF/Reckless:** RFP_DEPTH cap at 7-9, MARGIN_IMP at 50-90 × depth.

So Coda has:
- 2× the depth coverage (more permissive on when RFP can fire)
- ~½ the margin scaling (more aggressive on what RFP cuts at low depth)

Net effect: RFP fires often at depth 1-6 (small margin), occasionally
at depth 7-18 (small margin × depth still produces large absolute
margin for high-depth fires but the raw eval would need to be very
high to clear).

**Slack-shape interpretation:** Coda's small `RFP_MARGIN_IMP` shifts
the pruning workload toward RFP (away from futility/LMP/SEE). Tunables
on those features then look "weaker than SF" because they see harder
positions (RFP already pruned the easy ones). This is the
"ordering picks up slack" pattern Adam called out.

**Test**: SPRT `RFP_MARGIN_IMP=50` in isolation. If H0 (current is
better), the small-margin shape is real. If H1, we've been over-RFPing
at low depth.

---

## P3 — Extension gates floor-pinned at low values

| Tunable | Coda | SF | Reckless | Note |
|---|---:|---:|---:|---|
| IIR_MIN_DEPTH | 2 | 6 | ~4 | Floor-pinned (was just queued for prior_reduction gate via #1088) |
| PROBCUT_MIN_DEPTH | 3 | 5 | 4 | Floor-pinned (now testing 4 via #1085) |
| SE_DEPTH | 6 | 6+ttPv | 5+ttPv | Now aligned post-#1066 (just added ttPv via #1087) |

Pattern: SPSA pushes our extension gates DOWN (more permissive) than
cross-engine consensus.

**Why?** Two competing hypotheses:
(a) Our move ordering at low depth is weaker than SF/Reckless, so
    extension features (IIR, ProbCut) help clean up bad ordering at
    low depth — where SF doesn't need them
(b) Our search is calibrated to a different equilibrium where these
    gates' interaction with other features (LMR, NMP) balances at
    lower thresholds

The #1088 (IIR with `prior_reduction <= 3`) and #1085 (PROBCUT 3→4)
SPRTs will distinguish: if both H1, we were under-gating. If both H0,
the equilibrium is real.

---

## P4 — NMP gate cascade (audit doc Tier 1, deferred until post-retune)

| Tunable | Coda | SF |
|---|---:|---:|
| NMP_DEPTH_DIV | 6 | 4 |
| NMP_EVAL_DIV | 132 | 200 |
| NMP_MIN_DEPTH | 5 | 3 |
| NMP_VERIFY_DEPTH | 12 | 14 |

Coda's NMP fires at higher min_depth, with weaker depth-driven
reduction (DIV=6 vs 4) and tighter eval-driven extra reduction
(EVAL_DIV=132 vs 200, so eval-extra-R is BIGGER for Coda — more
aggressive). The cascade hypothesis: a too-tight gate caused
SPSA to drive `NMP_MIN_DEPTH=5` (was 3 default) and `NMP_EVAL_MAX=1`
(was higher). Audit doc Tier 1 has the full proposed restructure.

**Action**: deferred per audit doc — fix the gate margin
(SF-style), widen ranges, retune. ~+3-8 Elo expected. Not for
auto-fire today; needs research + SPSA budget.

---

## P5 — Move-ordering relative position

(Speculative — needs instrumentation to confirm.)

The "different order picks up slack" pattern shows up in pruning
calibration but is hard to test without per-feature dbg_hit
counters. If we instrument:
- "RFP cutoffs that would NOT have happened under SF parameters"
- "LMP cutoffs that would NOT have happened with SF's `(BASE+d²)/2-improving` at BASE=3"
- ...

we'd see exactly which features are absorbing slack. This is a
research item, not a near-term experiment. Add to instrumentation
backlog.

---

## Workflow proposal

When fleet capacity returns post-tune-1070/1071:

1. **P1 corr-hist formula fix** — highest expected gain (+3-8 Elo). Code change + focused SPSA + SPRT
2. **P2 RFP_MARGIN_IMP=50 ablation** — single-tunable SPRT, [-3, 3]. Cheap, high signal for "slack-shape" hypothesis
3. **P4 NMP gate cascade** — biggest single audit-doc pending item. Code + SPSA + SPRT
4. **P5 instrumentation** — when there's research time, build the dbg_hit per-feature comparison

If P1 lands +5+ Elo, that's evidence the "implementation bug" lens
finds more wins. If P2 lands H0, the slack-shape shape is real and
porting cross-engine values directly is the wrong move. Either way
we learn something durable.
