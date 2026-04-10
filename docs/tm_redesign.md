# Time Management Redesign (v2 — updated 2026-04-10)

## Lessons Learned (hard-won)

### What went wrong with v1

1. **Hard limit `soft×5` was catastrophic at blitz.** At 3+2, a single move
   could use 44s (5× the 8.9s soft). One overspend ruins the entire game.
   The old `soft×3` was safer. We saw 40-second moves at move 3.

2. **Ponderhit allocation was far too generous.** Giving a full phase-based
   budget on top of ponder time meant the engine spent 15+ seconds per
   ponderhit. The ponder search already did the heavy lifting — ponderhit
   should only verify, not re-search.

3. **No dynamic TM during ponder transition.** After ponderhit, soft_limit
   stays at 0 so the ID loop's stability/node-fraction checks don't fire.
   The engine uses the ENTIRE ponderhit budget with no early stop.

4. **Starting iterations without time to finish them.** During ponder, the
   next-iteration guard (hard_limit check) doesn't fire because hard_limit=0.
   The engine starts depth 25 with 2 seconds of ponderhit budget, gets stopped
   mid-search, and pollutes the TT with incomplete entries → blunders.

5. **Same hard limit multiplier for all TCs.** `soft×5` at 40+0.4 is 10% of
   time (fine). At 3+2 it's 22% of time (fatal). Hard limit must scale with TC.

6. **Self-play SPRT can't catch TM bugs.** Both sides are equally affected,
   so TM overspend is invisible. Cross-engine RR caught the LTC benefit but
   Lichess (with ponder) exposed the blitz regression.

7. **The old "broken" ponderhit was accidentally safe.** Instant stop on
   ponderhit meant clean TT entries and no overspending. "Fixing" it exposed
   deeper issues. Sometimes a bug masks a worse bug.

### What worked

- Phase-based allocation concept (spend less in opening, more in middlegame)
- TC regime awareness (different strategies for inc vs no-inc)
- Eval-based time reduction (less time when winning/losing clearly)
- 50-move eval scaling (correctness fix, SPRT-neutral but prevents 200-move draws)
- The TM changes genuinely helped at LTC (+47 Elo in cross-engine RR)

## Problem Statement (revised)

Two distinct problems:

**Problem 1: Blitz overspending** (3+2 and faster)
- Hard limit allows catastrophic single-move overspend
- Ponderhit gives too much time on top of ponder search
- Engine burns all time in middlegame, rushes endgame
- Observed: 40s on move 3, 0:10 remaining at move 29

**Problem 2: LTC underspending** (40+0.4 and slower)
- Underspends in middlegame vs SF (5.0s vs 6.5s)
- Stockpiles time in won endgames (140s remaining)
- Ponder time is wasted (instant stop or no additional search)

These are CONFLICTING requirements. The solution must be TC-aware,
not a single set of parameters.

## Design Goals (revised)

1. **Never allow a single move to use >15% of remaining time** at any TC
2. **Hard limit must scale with TC** — tighter at blitz, wider at rapid
3. **Ponderhit must be conservative** — verify only, not full re-search
4. **Dynamic TM must work after ponderhit** — stability/node-fraction
5. **Never start an iteration you can't finish** — guard in ponder too
6. **Test with ponder on** — SPRT without ponder misses critical bugs

## TC Regimes

| TC | Base | Inc | Strategy | Hard mult | Max single move |
|----|------|-----|----------|-----------|----------------|
| 1+0 | 60s | 0 | Ultra-conservative | 2× | 5% of time |
| 1+1 | 60s | 1 | Increment-driven | 2.5× | 8% of time |
| 3+0 | 180s | 0 | Conservative | 2.5× | 8% of time |
| 3+2 | 180s | 2 | Moderate | 3× | 10% of time |
| 10+5 | 600s | 5 | Moderate-generous | 4× | 12% of time |
| 40+0.4 | 2400s | 0.4 | Generous | 5× | 15% of time |

Key: hard_mult and max-single-move INCREASE with TC. At blitz, the
penalty for one bad move is game-losing. At rapid, one deep think
is acceptable.

## Hard Limit Formula

**CRITICAL: Hard limit must be based on BASE soft, not dynamically-scaled
soft.** Dynamic factors can scale soft by up to ~2.5x (stability=0 ×
node_fraction=2.23). If hard_mult applies to the scaled value, the
effective maximum becomes soft × 2.5 × 3 = soft × 7.5, recreating
the overspend problem. The hard limit is an ABSOLUTE safety cap.

```
// TC-aware hard limit multiplier (applied to BASE soft, not scaled)
let seconds_per_move = time_left / 25;  // rough estimate
let hard_mult = if seconds_per_move < 2 { 2.0 }       // bullet
    else if seconds_per_move < 5 { 2.5 }               // blitz
    else if seconds_per_move < 15 { 3.0 }              // rapid
    else { 4.0 };                                       // classical

// Hard limit from BASE soft (before dynamic factors)
hard = min(base_soft * hard_mult, time_left * max_single_pct / 100)

// Dynamic factors adjust soft but NEVER exceed hard
adjusted_soft = (base_soft * scale).min(hard)
```

The `max_single_pct` prevents any single move from being catastrophic:
```
let max_single_pct = if seconds_per_move < 2 { 5 }
    else if seconds_per_move < 5 { 8 }     // was 10, tightened per review
    else if seconds_per_move < 15 { 12 }
    else { 15 };
```

## Ponderhit Allocation

### Principle: ponder did the work, ponderhit just verifies

The ponder search runs for the full opponent's think time (often 5-15s).
By ponderhit, the engine already has a deep, well-searched result. The
ponderhit allocation should be SMALL — just enough to complete the current
iteration or verify the best move.

**Key finding: instant ponderhit stop (0 seconds) produced 99% accuracy
at 2950 Lichess rating.** The ponder search result is usually good enough.
Any ponderhit budget is gravy, not necessity. Start conservative.

```
// Ponderhit budget: very conservative — ponder already searched deeply
// Start at inc * 1.5, can loosen later if proven safe
let ponderhit_soft = min(
    time_left / moves_left + inc * 0.8,  // normal per-move budget
    time_left / 10,                       // never more than 10% of remaining
    inc * 1.5                             // never more than 1.5x increment
);
```

For 3+2 at move 20 with 120s remaining:
- Normal budget: 120/25 + 1.6 = 6.4s
- 10% cap: 12s
- 1.5× increment: 3s
- Result: min(6.4, 12, 3) = **3s** (conservative, can increase later)

For 3+2 at move 5 with 180s remaining (the problem case):
- Normal budget: 180/25 + 1.6 = 8.8s
- 10% cap: 18s
- 1.5× increment: 3s
- Result: min(8.8, 18, 3) = **3s** (capped by increment, prevents overspend)

### Ponderhit must set both soft and hard limits

The search's next-iteration guard uses hard_limit. If hard_limit stays
at 0 (from ponder mode), the guard doesn't fire and the engine starts
iterations it can't finish.

```
// On ponderhit: set BOTH limits via the atomic
// The deadline conversion (budget + elapsed) is critical
let elapsed = search_start_time.elapsed();
let deadline = elapsed + ponderhit_soft;
ponderhit_flag.store(deadline, Relaxed);
// Also need hard_limit set for next-iteration guard
// This requires either making hard_limit atomic, or
// piggybacking on ponderhit_time with a separate check
```

### Dynamic TM after ponderhit

Currently, the ID loop's dynamic TM only fires when soft_limit > 0.
During ponder, soft_limit = 0. After ponderhit, it stays 0.

Fix: when ponderhit is detected in should_stop(), also set soft_limit
so the iteration-level checks work:
```
if ponderhit_detected {
    self.soft_limit = ponderhit_deadline;  // enables dynamic TM
    self.hard_limit = ponderhit_deadline + 2000;  // small hard margin
}
```

This requires soft_limit/hard_limit to be atomic or the ponderhit
conversion to happen in the search thread (where they're mutable).

## Validation Plan (revised)

### Step 0: Revert to known-good state
Revert all TM changes. Confirm codabot plays 98%+ accuracy on Lichess.
This is the baseline to beat.

### Step 1: Implement and profile locally
Apply changes. Run cutechess games at 3+2, 3+0, 10+5 against SF.
Compare time profiles using tm_profile.py.

**Critical metrics:**
- No single move uses >15% of remaining time at 3+2
- Middlegame avg within 20% of SF's allocation
- SD > 3.0 (adaptive time use)

### Step 2: Lichess test (WITH PONDER)
Deploy to codabot. Monitor first 5 games via SF analysis.
Must see 95%+ accuracy, 0 mistakes, 0 blunders.
If any game has a mistake, STOP and investigate.

### Step 3: SPRT at LTC (ponder off)
Validate no regression at LTC where the changes should help.

### Step 4: Cross-engine RR
Compare against rivals at both STC and LTC to confirm no regression.

## Additional Considerations (from code review)

### Dynamic TM after ponderhit — DEFER

Setting soft_limit when ponderhit is detected is architecturally complex
(needs atomic soft_limit or cross-thread signaling) and introduces
instability risk. **Defer entirely until everything else is stable.**
Instant ponderhit stop is simple and proven at 99% accuracy.

### Time-rich scenarios

When the opponent moves very fast (1-2s per move) and our ponder prediction
rate is high, we accumulate time. In a 3+2 game we might have 4+ minutes.
The engine should NOT overspend even when time-rich — the hard limit and
max_single_pct caps must apply regardless of how much time we have.

### Eval factor — bring back with higher threshold

Removing eval_factor entirely means spending full time on mate-in-5.
Bring it back with a conservative threshold:
```
let eval_factor = if is_mate_score(prev_score) { 0.3 }
    else if prev_score.abs() > 800 { 0.75 }  // clearly won/lost
    else { 1.0 };
```
Higher threshold (800 vs 300/500 before) and gentler reduction (0.75 vs
0.5) to avoid affecting competitive positions.

## Risks (revised)

1. **Blitz overspending** — HIGHEST RISK. Mitigated by TC-aware hard limit
   and max-single-move cap. Must test at 3+2 specifically.
2. **Ponderhit TT pollution** — fixed by next-iteration guard, but must
   verify with ponder-on testing (not available in cutechess/fastchess).
3. **Self-play masking** — TM bugs are invisible in self-play SPRT.
   Must use cross-engine testing and Lichess for validation.
4. **Regression at LTC** — unlikely (v1 showed +47 at LTC) but must verify.

## Implementation Order (revised)

1. **Revert to known-good TM** — stabilise codabot
2. **TC-aware hard limit** — single most important change
3. **Conservative ponderhit allocation** — cap at 3× increment
4. **Dynamic TM after ponderhit** — enable stability/node-fraction
5. **Next-iteration guard during ponder** — prevent TT pollution
6. **Profile at 3+2 and 40+0.4** — verify both TCs
7. **Lichess deployment with ponder** — the real test
8. **Phase-based soft allocation** — only AFTER hard limit is safe
9. **Cross-search score trending** — low risk enhancement
10. **Forced-move detection** — future, after everything else stable
