# Time Management Redesign

## Problem Statement

Codabot consistently stockpiles time on Lichess (3+2 blitz):
- Average time utilisation: 62% (should be 85-95%)
- Finishes with 1:40/3:00 remaining in won endgames
- Spends ~3.5s/move in middlegame vs SF's ~4.5s/move
- Ponder hits bank time instead of spending it on depth

The current TM is too conservative across all game phases, especially
in the middlegame where depth matters most.

## Design Goals

1. **Use 85-95% of available time** by move 50 in a typical game
2. **Adapt to TC regime** — generous increment vs no increment vs ultra-bullet
3. **Spend more in middlegame** (moves 15-35) where games are decided
4. **Spend less in opening** (book covers this) and **won endgames** (eval-based)
5. **Never flag** — maintain a safety buffer proportional to risk

## TC Regimes

The ratio `increment / (base_time / 50)` characterises the TC:

| TC | Base | Inc | Inc ratio | Regime | Strategy |
|----|------|-----|-----------|--------|----------|
| 1+0 | 60s | 0 | 0.00 | Ultra-bullet | Very conservative, latency dominates |
| 1+1 | 60s | 1 | 0.83 | Bullet + inc | Inc IS the time budget |
| 3+0 | 180s | 0 | 0.00 | Blitz no-inc | Conservative, can't recover time |
| 3+2 | 180s | 2 | 0.56 | Blitz + inc | Aggressive — inc covers most moves |
| 5+3 | 300s | 3 | 0.50 | Blitz + inc | Similar to 3+2 |
| 10+5 | 600s | 5 | 0.42 | Rapid | Deep thinks, very generous |
| 15+10 | 900s | 10 | 0.56 | Rapid + inc | Classical-like depth |

### Key insight: inc_ratio > 0.3 = "safe" TC

When increment covers >30% of per-move base time, flagging is nearly
impossible. The engine can be aggressive with base time knowing increment
provides a safety net. When inc_ratio = 0 (no increment), every second
is irreplaceable.

## Target Time Profiles

### 3+2 (180s base, 2s increment)

Available time = 180s base + ~100s increment over 50 moves = 280s total.
Target: use 250s by move 50 (89%), leave ~30s buffer.

```
Moves  1-10: 0.5s avg (book/opening, accumulate increment)
Moves 11-20: 6.0s avg (early middlegame, building advantage)
Moves 21-30: 8.0s avg (critical middlegame, spend heavily)
Moves 31-40: 5.0s avg (late middlegame/early endgame)
Moves 41-50: 2.0s avg (endgame, mostly technique)
Moves 50+:   1.0s avg (deep endgame, increment covers)

Total: 5 + 60 + 80 + 50 + 20 + 10 = 225s used + ~25 buffer
```

### 3+0 (180s base, no increment)

Available time = 180s total. No recovery.
Target: use 160s by move 50 (89%), leave 20s emergency buffer.

```
Moves  1-10: 0.5s avg (book)
Moves 11-20: 4.5s avg (middlegame)
Moves 21-30: 5.5s avg (critical)
Moves 31-40: 3.5s avg (late game)
Moves 41-50: 1.5s avg (endgame)
Moves 50+:   0.5s avg (flag avoidance)

Total: 5 + 45 + 55 + 35 + 15 + 5 = 160s + 20 buffer
```

## Proposed Architecture

### Phase 1: Base allocation

```
safety_buffer = if inc > 0 { move_overhead * 3 } else { time_left * 0.10 }
spendable = time_left - safety_buffer
inc_budget = inc * 0.85  // 85% of increment, reserve 15% for latency spikes

// Game phase factor: spend more in middlegame
// Reckless-inspired exponential ramp
phase_factor = 0.025 + 0.040 * (1.0 - exp(-0.050 * fullmove))
// Move 5: 0.034, Move 20: 0.053, Move 40: 0.059 (plateau)

soft = phase_factor * spendable + inc_budget
```

This naturally:
- Spends less in opening (phase_factor low)
- Peaks in middlegame (phase_factor rising)
- Plateaus in endgame (asymptote)
- Adapts to remaining time (spendable shrinks)
- Uses increment as guaranteed per-move budget

### Phase 2: Hard limit

```
// Hard limit scales with TC regime
hard_mult = if inc > 0 { 6.0 } else { 3.0 }
hard = min(soft * hard_mult, spendable * 0.25)
```

With increment: allow 6× soft (confidence that inc prevents flagging).
Without increment: only 3× soft (must preserve time).
Cap at 25% of spendable to prevent any single move from catastrophic time loss.

### Phase 3: Dynamic factors (after each iteration)

Same 3-factor multiplicative model, with additions:

```
// Factor 1: Node fraction (unchanged, Obsidian pattern)
nodes_factor = 0.63 + (1.0 - best_frac) * 2.0

// Factor 2: Stability (unchanged)
stability_factor = max(0.70, 1.60 - tm_best_stable * 0.08)

// Factor 3: Score trend (ENHANCED with cross-search trending)
prev_iter_drop = tm_prev_score - prev_score
cross_search_drop = if tm_search_prev_score != 0 {
    tm_search_prev_score - prev_score
} else { 0 }
score_factor = clamp(0.86 + 0.010 * prev_iter_drop + 0.025 * cross_search_drop,
                      0.81, 1.50)

// Factor 4: Eval magnitude (SMOOTHED, not discrete)
eval_factor = if is_mate_score(prev_score) {
    0.3
} else {
    1.0 - (prev_score.abs() as f64 / 1500.0).min(0.6)
    // 0cp: 1.0, 300cp: 0.8, 750cp: 0.5, 1500cp+: 0.4
}

// Factor 5: Fail-low bonus (NEW — extend time on complex positions)
fail_low_factor = if search_failed_low { 1.0 + fail_low_count * 0.25 } else { 1.0 }
fail_low_factor = fail_low_factor.min(1.50)

scale = nodes_factor * stability_factor * score_factor * eval_factor * fail_low_factor
adjusted_soft = (soft * scale).min(hard)
```

### Phase 4: Ponder time accounting

When a ponder hit occurs, the time saved should be partially "spent":
```
// On ponderhit: we gained opponent's think time for free
// Don't just bank it — allow the next move to use some of the bonus
if ponderhit {
    ponder_bonus = opponent_think_time * 0.3  // Use 30% of saved time
    adjusted_soft += ponder_bonus
}
```

### Phase 5: Forced-move detection (future)

After each iteration at depth >= 10:
```
// Search depth/2 excluding best move
// If no good alternative exists, reduce time
if is_forced_move(board, best_move, depth) {
    adjusted_soft *= 0.4
}
```

## Validation Plan

### Step 1: Profile comparison (no SPRT needed)

Run 50 games at 3+2 against a fixed opponent (SF/Reckless) with clock logging.
Produce per-game profile:

```
Game N: vs Reckless (W/L/D)
  Phase     Coda avg  Opp avg   Coda SD   Opp SD
  Opening   0.5s      0.3s      0.2       0.1
  Middle    7.2s      6.8s      3.1       2.9
  Late      3.5s      3.2s      1.8       1.5
  Endgame   1.2s      1.0s      0.5       0.3
  Time remaining at move 40: 35s vs 28s
```

**Target metrics:**
- Coda avg middlegame ≥ opponent avg (currently we're 50% less)
- Time remaining at move 40: < 40s (currently ~140s)
- SD of time per move > 3.0 (currently ~1.5, too flat)

### Step 2: Regime testing

Repeat for 3+0, 1+1, 10+5 to verify no flagging and reasonable profiles.

### Step 3: SPRT at LTC

Once profiles look correct, SPRT at 40+0.4 for Elo validation.

### Step 4: Lichess deployment

Update codabot, monitor rating change and time profiles in live games.

## Risks

1. **Flagging at no-increment TCs**: Mitigated by safety_buffer and hard_mult=3
2. **Over-spending in middlegame**: Mitigated by phase plateau and hard cap
3. **Network latency spikes**: Mitigated by move_overhead and inc reserve
4. **Ponder bonus abuse**: Capped at 30% of saved time

## Implementation Order

1. Cross-search score trending (10 min, low risk)
2. Smooth eval factor (5 min, low risk)
3. Phase-based allocation with TC regime awareness (1 hour)
4. Fail-low bonus (15 min)
5. Profile comparison testing (2 hours)
6. Forced-move detection (2-3 hours, after validation)
