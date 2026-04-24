# Formula-Shape Experiments — Proposal

## Status as of 2026-04-21 — UNTRIED (priority 3 in execution plan)

All three experiments remain untried. Priority 3 in current plan,
queued after caphist-retune and B2 skewer detector. Full original
design preserved below.

Note: the doc references `tunable_anomalies_2026-04-19.md` as
motivation, but that file was never created — Titan consolidated the
cross-engine parameter analysis into `ordering_coupled_pruning_
2026-04-19.md` instead. The shape experiments here are still
well-motivated: they address formula structure that SPSA can't reach
by tuning values alone.

Recommended execution order:
1. **Exp 1 (history-bonus offset)** — mirrors Stockfish shape, lowest
   structural risk, easiest to SPSA-retune (4 params).
2. **Exp 2 (RFP unified margin)** — flat improving discount vs Coda's
   depth-scaled. Orthogonal to Exp 1, run in parallel if fleet has
   capacity.
3. **Exp 3 (futility TT-modulated)** — design-only, expand if Exp 1/2
   land.

---

Motivated by the cross-engine tunable comparison in
`tunable_anomalies_2026-04-19.md` (never authored — see
`ordering_coupled_pruning_2026-04-19.md` for the actual cross-engine
data). The anomalies there are about **formula shape**, not parameter
values — SPSA can't escape a shape basin by tuning values. These
experiments swap shapes.

Sequencing per Hercules (foundational first):
1. History bonus shape
2. RFP unified shape
3. Futility TT-hit modulated

Each experiment specifies: current formula, proposed formula, tunable
changes with starting values, and SPSA parameter set for the focused
retune.

---

## Experiment 1 — History bonus shape (linear → linear-with-offset)

### Why

Coda's `stat_bonus(d) = min(HIST_BONUS_MAX, HIST_BONUS_MULT * d)` has
slope 300, cap 1584. At d=5 it saturates (bonus = 1500, clipped at 1584).
Moves at d=5 and d=10 therefore receive the **same** bonus magnitude —
no depth discrimination past d=6.

SF uses `min(1690, 133 * d - 72)` — linear with negative offset and
higher cap. Bonus is zero around d≈1, grows slowly, and plateaus later.
This gives wider depth discrimination.

Coda's cap-history already uses the offset shape
(`CAP_HIST_MULT * d - CAP_HIST_BASE`, offset=15). Main history is the
only inconsistent one.

### Shape change

**Current**:
```rust
fn stat_bonus(depth: i32) -> i32 {
    (HIST_BONUS_MULT * depth).min(HIST_BONUS_MAX)
}
```

**Proposed** (`src/search.rs:2799`):
```rust
fn stat_bonus(depth: i32) -> i32 {
    (HIST_BONUS_MULT * depth - HIST_BONUS_OFFSET).clamp(0, HIST_BONUS_MAX)
}
```

The `clamp(0, MAX)` floor at 0 handles `HIST_BONUS_OFFSET > HIST_BONUS_MULT * d` at very shallow depth (avoids negative bonuses that would corrupt history table updates).

### Tunable changes

**Add** one new tunable:
```rust
(HIST_BONUS_OFFSET, 72, 0, 400),
```

Starting value 72 mirrors SF's `-72` offset. Range wide so SPSA can explore from "no offset" (0) to "very late-starting bonus" (400).

**Existing tunables retained unchanged**:
- `HIST_BONUS_MULT` (current 300, range 50-400)
- `HIST_BONUS_MAX` (current 1584, range 500-3000)

### Sequencing within the experiment

Per Hercules: the history shape change alters raw history magnitudes, which feed into three other places. Joint SPSA needed to rebalance them concurrently, otherwise measurement conflates two effects.

Downstream consumers:
- **Futility**: `hist_adj / 128` in the futility_value formula (src/search.rs:2295)
- **LMR**: `hist / LMR_HIST_DIV` subtracted from LMR reduction (src/search.rs ~2414)
- **History pruning**: `history < -HIST_PRUNE_MULT * depth` threshold (history pruning gate)

### SPSA parameter set (focused, ~1000 iters)

```
HIST_BONUS_MULT, int, 300, 50, 400, 25, 0.002
HIST_BONUS_OFFSET, int, 72, 0, 400, 25, 0.002
HIST_BONUS_MAX, int, 1584, 500, 3000, 150, 0.002
LMR_HIST_DIV, int, 7123, 2000, 100000, 1000, 0.002
HIST_PRUNE_MULT, int, 5148, 500, 50000, 500, 0.002
```

5 params, 1000 iters should converge. Values reflect the current tunables! macro defaults as of feature/threat-inputs HEAD.

### What to watch for

- `HIST_BONUS_OFFSET` drifting toward 0 → the offset shape isn't helping; SF's shape-difference is a red herring for Coda.
- `HIST_BONUS_OFFSET` drifting to 100-200 range → shape difference is real; SPSA found a depth-discrimination basin.
- `HIST_BONUS_MULT` drifting high alongside OFFSET → shape compensates; net bonus at moderate depth unchanged, but wider spread across depths.
- `HIST_BONUS_MAX` rising toward 2000+ → new shape supports a higher cap (bonus no longer saturates at d=5).

Post-SPSA SPRT of tuned values vs trunk. Bounds `[0, 5]`.

### Expected outcome

One of:
- **Neutral to +5 Elo**: offset shape finds a slightly better basin, confirming the hypothesis.
- **H0 flat**: shape doesn't matter; current linear-with-clip is fine.
- **H0 negative**: new shape loses Elo. Try the Alexandria-asymmetric-malus variant (below) as a fallback.

Fallback variant (if Experiment 1 H0s): **asymmetric malus split** (Alexandria-style). Malus formula is separate from bonus:
```
bonus: min(MAX_B, MULT_B * d + OFFSET_B)
malus: min(MAX_M, MULT_M * d + OFFSET_M)   # separate params
```

Would need to audit current code for where malus is applied. Most likely under-uses malus compared to bonus — hence Alex's separate tuning.

---

## Experiment 2 — RFP unified shape (split → base+improving_subtraction)

### Why

Coda uses `margin = depth * (improving ? RFP_MARGIN_IMP=82 : RFP_MARGIN_NOIMP=128)` — two coefficients, improving scales with depth.

SF uses `margin = futilityMult * d - improving_correction`, where the improving correction is **flat** (not per-depth). At d=5: SF improving saves ~199, at d=10: still ~199. Coda's equivalent saves 230 at d=5 and 460 at d=10.

**Structural difference**: Coda's improving discount grows linearly with depth; SF's is flat. One or the other is probably more accurate depending on how "improving" correlates with depth. SPSA on current shape cannot escape Coda's per-depth structure; needs a shape swap.

### Shape change

**Current** (src/search.rs:1983):
```rust
let mut margin = if improving {
    depth * tp(&RFP_MARGIN_IMP)
} else {
    depth * tp(&RFP_MARGIN_NOIMP)
};
```

**Proposed**:
```rust
let mut margin = depth * tp(&RFP_MARGIN) - (improving as i32) * tp(&RFP_IMPROVING_SUB);
```

### Tunable changes

**Remove** 2 existing tunables:
- `RFP_MARGIN_IMP` (current 82)
- `RFP_MARGIN_NOIMP` (current 128)

**Add** 2 replacement tunables:
- `RFP_MARGIN` (starting 128 = current NOIMP; same effective margin when improving=0)
- `RFP_IMPROVING_SUB` (starting 230; gives same d=5 behavior as current under improving: 128*5 - 230 = 410 = 82*5)

Net: same tunable count (2 replaces 2). Zero initial Elo movement — the initial-values-equivalent-to-current at d=5 is deliberate. Shape freedom is orthogonal.

```rust
(RFP_MARGIN, 128, 50, 200),
(RFP_IMPROVING_SUB, 230, 0, 600),
```

### SPSA parameter set

```
RFP_MARGIN, int, 128, 50, 200, 10, 0.002
RFP_IMPROVING_SUB, int, 230, 0, 600, 30, 0.002
```

Just 2 params; focused SPSA ~500-1000 iters.

### What to watch for

- `RFP_IMPROVING_SUB` drifts high toward 400-500 → flat subtraction works, improving positions prune harder at high depth than current shape allows.
- `RFP_IMPROVING_SUB` drifts low toward 100-150 → current per-depth scaling was better at high depth; new flat shape didn't help.
- `RFP_IMPROVING_SUB` stays near 230 → shape is neutral; new tunable didn't find a new basin. Revert.

Post-SPSA SPRT vs trunk at `[0, 5]`.

---

## Experiment 3 — Futility TT-hit modulated coefficient

### Why

Coda's futility uses a fixed `FUT_PER_DEPTH=163` regardless of TT state. SF uses `futilityMult = 76 - 21 * !ttHit` — 76 with TT hit, 55 without. Margin is SMALLER when no TT hit → more aggressive pruning.

Counterintuitive at first but the logic: no TT hit means no prior search info, and if raw eval is already well above beta (no TT evidence suggesting otherwise), the position is likely genuinely quiet. Coda's flat margin doesn't exploit this information.

Ratio: SF scales coefficient by 72% when no TT hit. Coda's equivalent would be FUT_PER_DEPTH=163 → 117 when no TT hit (163 * 0.72), or an explicit subtraction of ~46.

### Shape change

**Current** (src/search.rs:2295):
```rust
let futility_value = static_eval + tp(&FUT_BASE) + lmr_d * tp(&FUT_PER_DEPTH)
    + hist_adj + threats_adj;
```

**Proposed**:
```rust
let fut_per_depth = tp(&FUT_PER_DEPTH) - (!tt_hit as i32) * tp(&FUT_PER_DEPTH_NOTTHIT);
let futility_value = static_eval + tp(&FUT_BASE) + lmr_d * fut_per_depth
    + hist_adj + threats_adj;
```

### Tunable changes

**Add** one new tunable:
```rust
(FUT_PER_DEPTH_NOTTHIT, 46, 0, 100),
```

Starting 46 ≈ 28% of 163 (matches SF's 21/76 ratio). Range 0-100 to let SPSA explore "no modulation" through "aggressive modulation".

Existing `FUT_PER_DEPTH=163` retained unchanged.

### SPSA parameter set

```
FUT_PER_DEPTH, int, 163, 40, 250, 10, 0.002
FUT_PER_DEPTH_NOTTHIT, int, 46, 0, 100, 6, 0.002
FUT_BASE, int, 69, 20, 200, 10, 0.002
```

3 params.

### What to watch for

- `FUT_PER_DEPTH_NOTTHIT` drifts to 0 → TT-hit modulation doesn't help; shape change is noise. Remove.
- `FUT_PER_DEPTH_NOTTHIT` drifts to 30-70 → real signal; TT-hit gating works. Keep.
- Combined shift: both `FUT_PER_DEPTH` and `FUT_PER_DEPTH_NOTTHIT` move together → shape is genuinely orthogonal, the modulation unlocks a new basin.

---

## Summary of tunable budget

| Experiment | New tunables added | Net tunable count change |
|---|---|---|
| 1 (History shape) | HIST_BONUS_OFFSET | +1 |
| 2 (RFP unified) | RFP_MARGIN, RFP_IMPROVING_SUB (replaces 2) | 0 |
| 3 (Futility TT-hit) | FUT_PER_DEPTH_NOTTHIT | +1 |

Net +2 tunables across all three experiments. Current count ~58 → ~60 post.

## Non-goals

- **Not changing MVV or SEE scaling**: those are structurally different across engines for reasons unrelated to shape (SEE uses piece values, MVV uses victim value; formulas not directly comparable).
- **Not touching LMR formula**: Coda's separate-table quiet/cap LMR is structurally different from SF/Obs/Vir's unified log-based table. Would be a much bigger refactor. Out of scope.
- **Not re-testing contempt**: separate SPRT already running as #508.

## Hand-off

Hercules has implementation lane. This doc provides:
- Drop-in formula specs (pseudocode + file:line)
- Starting tunable values (chosen to match current Coda behaviour at reference depth where possible)
- Focused SPSA parameter sets
- Watch-for patterns in the SPSA result

Full code patches not required — Hercules can translate from these specs.

## Companion docs

- `tunable_anomalies_2026-04-19.md` — cross-engine comparison that motivated these
- `signal_context_sweep_2026-04-19.md` — broader signal × context ideation
- `threat_ideas_plan_2026-04-19.md` — threat-signal experiments with live results
