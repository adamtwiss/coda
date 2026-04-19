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

### Time-rich ponderhit scenario (discovered 2026-04-10)

When the opponent moves quickly and our ponder prediction rate is high,
we accumulate massive time advantages (e.g. 1:34 vs 0:17 in 3+2). But
with instant ponderhit stop, every correct prediction is played with 0s
of verification — even when we have minutes of banked time.

**Observed in Lichess game:** Over half the moves were instant (correct
ponder predictions). Engine accumulated 1:22 vs opponent's 0:11. But
inaccuracies occurred on the few moves where it searched normally. Result:
93% accuracy in a position it should have converted easily.

**The problem:** Instant ponderhit is optimal when time-tight (don't
risk overspending), but suboptimal when time-rich (we should verify the
ponder result). The engine should spend more time when it has a large
time advantage, not less.

**Proposed approach:** Scale ponderhit budget by time ratio:
```
let time_ratio = our_time / opponent_time;
let ponderhit_budget = if time_ratio < 1.0 {
    0  // less time than opponent: instant stop (safe)
} else if time_ratio < 2.0 {
    inc * 0.5  // moderate advantage: brief verify
} else {
    min(inc * 1.5, our_time / 20)  // big advantage: use some surplus
};
```

This is a natural extension of the conservative ponderhit allocation
above. Defer until the base ponderhit allocation is stable.

### Ponderhit race condition (fixed 2026-04-10)

When the opponent responds instantly to our ponderhit (0ms), cutechess
sends `go ponder` + `ponderhit` in the same millisecond. The search
thread clears `stop=false` at startup, **overwriting** the ponderhit's
`stop=true`. The engine then searches infinitely in ponder mode with
no time limits, causing time loss.

**Fix:** Clear the stop flag in the UCI thread (before spawning the
search thread), not in `search()`/`search_smp()`. This way, ponderhit
can safely set stop after the spawn without being overwritten.

### Time-rich scenarios (general)

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
2. **Ponderhit TT pollution** — fixed by next-iteration guard, must
   verify with ponder-on testing (cutechess-cli supports ponder; FastChess
   does NOT — see scripts/run_ponder_test.sh).
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

---

## Outstanding bugs / issues (identified 2026-04-17)

A follow-up investigation into the "1-in-4 games have clustered blunders"
pattern on Lichess surfaced a UCI-layer bug (now fixed on branch
`fix/ponder-wait-loop`) AND exposed several TM weaknesses that are
invisible under self-play SPRT but bite hard in production-style play.

### Issue 1: aggressive drain — no meaningful safety margin

**Observed, 60+1 tournament vs Caissa** (stage 13 debug trace of one game):

| Coda move | wtime remaining (s) |
|---|---|
| 1  | 60.0 |
| 5  | 44.3 |
| 10 | 21.2 |
| 15 | 10.6 |
| 20 | 2.4  ← danger zone |
| 23 | 1.3 |

Coda's TM steam-rollers through the opening at ~3–5s per move, running
the clock down to ~1–3 seconds by move 20. That's with a 1s increment.
**Any single-move variance (a long iteration, OS scheduling blip, helper-
thread join latency) tips us over zero** and forfeits. Observed rate in
local cutechess-cli tournaments at 60+1 with ponder on: **5-7% of games
forfeit on time**. Zero at 3+2 (Stage 2 main) but clustered failures
appear when production conditions (ponder, SMP, real clocks) combine.

Concretely: the existing `max_hard = time_left / 20 + our_inc` caps one
move at 5% of remaining time — reasonable by itself — but with small
inc:time ratio (60+1 = 1.67%) Coda *consistently* spends close to that
cap, so the clock drains at `hard − inc` per move. There is no floor
below which TM says "stop, preserve reserve."

**Fix direction**: introduce a minimum-reserve constraint that TM
cannot violate. Something like "never allocate so much this move that
the resulting clock drops below `max(5×inc, 3 seconds)`". If the soft
allocation would break that floor, scale soft (and hard) down instead.
SPSA-tunable minimum_reserve parameter.

### Issue 2: `max_hard` formula misbehaves for classical TC (40/15)

The current `max_hard = time_left / 20 + our_inc` assumes inc > 0 and
plenty of moves remaining. At classical TC with movestogo, the formula
breaks as movestogo decreases:

At 40/15 with 10 moves remaining (200s clock, inc=0):
- `soft = 200s / 10 = 20s` (correct expected allocation)
- `max_hard = 200000/20 + 0 = 10000ms = 10s`
- `max_hard < soft`, so the later clamp `if soft > hard { soft = hard }`
  forces soft down to 10s — **half of what we actually need**.

The `max_hard` cap was designed as a blitz safety net but it applies
uniformly. For classical (movestogo > 0) it is *too tight*; for blitz
it is *too loose* (see Issue 1).

**Fix direction**: `max_hard` should depend on TC shape.
- If `movestogo > 0`: let the movestogo path drive hard (it already
  caps at `time_left × hard_pct/100`). Don't apply the additional `/20`
  cap.
- If `our_inc > 0` (sudden-death with inc): use a tighter fraction when
  `inc/time_left` is small to preserve reserve (see Issue 1).
- If `our_inc == 0 && movestogo == 0`: true sudden-death. Current `/20`
  is OK but benefits from a minimum-reserve floor.

### Issue 3: hard_limit honored but ponderhit-chained moves drain clock fast

When cutechess's ponder prediction is frequently correct, Coda executes
many short post-ponderhit searches back-to-back, each using ~hard_limit
of Coda's clock. The drain during a ponderhit streak is `(hard - inc)
× streak_length`. With 3s hard and 1s inc, a 10-move ponderhit streak
burns 20 seconds of our clock in real wall-time.

In the observed forfeit game, 4 consecutive ponderhit moves between two
"normal" go commands (t=56s → t=81s real time) burned Coda's clock by
~9.5s across those 4 moves — each using hard_limit consistently.

Per-move the usage is correct (≤ hard_limit). The problem is there's
no "this is the Nth fast move in a row, slow down" logic. TM doesn't
see the context.

**Fix direction**: this might just need the Issue 1 fix. If minimum-
reserve floor applies per move, a streak can't drain below the floor
because later moves get downscaled.

### Issue 4: Rust stdout is BLOCK-buffered when piped to another process

Not a TM bug per se but it surfaced during this investigation and affects
TM's ability to honor time.

`println!()` in Rust writes to `io::stdout()` which is a LineWriter
**only when connected to a terminal** (per
https://doc.rust-lang.org/std/io/fn.stdout.html). When piped (as cutechess
/ Lichess does), stdout is **block-buffered**. `println!("bestmove ...")`
writes into an internal buffer that doesn't flush until the buffer fills
or explicit `flush()` is called. The GUI doesn't see bestmove until
flush — clock keeps ticking.

Fix applied in `fix/ponder-wait-loop` c37113e / 0a91e2b: explicit
`std::io::stdout().lock().flush()` after bestmove emission.

Worth also flushing after every info string at search.rs:1320 to avoid
cutechess seeing stale PV during time-pressure. (Not yet done, low risk.)

### Issue 5: ponder wait-loop raced with `info.stop`

In `search_smp`, after the main search returns, `info.stop.store(true)`
is set to tell Lazy-SMP helpers to exit. The ponder wait-loop in uci.rs
was watching `info.stop` to decide whether to wait for GUI ack. Result:
search_smp's internal "I'm done" signal was misread as a GUI-sent stop/
ponderhit, and the engine emitted bestmove before the GUI acked —
"Premature bestmove while pondering" 928× per 30 games.

Fix applied in `fix/ponder-wait-loop` c37113e: introduced
`external_stop: Arc<AtomicBool>` set ONLY by UCI handlers
(stop/ponderhit/quit/go/ucinewgame/setoption). Wait loop watches
`external_stop` instead of `info.stop`.

### Issue 6: TM features invisible to SPRT (methodology gap)

Stage 2 (main with ponder broken): 40% score vs +180 Elo opponents,
928 protocol violations, 0 time forfeits, 3 blunders / 30 games.
Stage 3 (F1 fix applied): 41.7% score, 0 protocol violations, **2 time
forfeits**, 0 blunders.

F1 eliminated all the catastrophic blunders (consistent with the 1-in-4
lichess pattern) but traded them for a 5-7% time-forfeit rate that only
appears under cutechess-with-ponder-at-60+1 / 3+2. Openbench SPRT at
10+0.1 with `ponder=false, Threads=1` is blind to ALL of these effects.

In one OpenBench test (#426, v9 psq-refresh-simplified +9 Elo aggregate),
per-worker results ranged from +40 Elo (ionos6, cache-rich) to -9 Elo
(ionos3, memory-constrained) on identical hardware-class VPSes, 7%
spread. Production hardware matters.

Concrete methodology additions worth considering:
- Track time-forfeit count in automated cross-engine runs and flag rates
  above some threshold (say 0.5%)
- Add a lichess-mirror gauntlet at 60+1 with ponder=on as a preflight
  before declaring a TM-touching patch "ready"
- Capture per-worker Elo breakdown in SPRT reports; flag when spread
  exceeds 2 SEs — indicates hardware-sensitivity

### Issue 7: drift gap we could not explain

Even with full instrumentation (spawn_overhead, join_us, search, wait,
println, stdout flush), the sum of Coda's tracked thinking time for a
forfeit game was ~93.5s against a 110s budget (50 moves at 60+1). Coda
was ~16.5s under budget by its own accounting but cutechess's clock
still hit zero. Only ~100ms/game of drift is visible to the in-process
instrumentation.

The other ~95% of the drift must be on cutechess's accounting side, or
in timings we can't instrument (pipe-read latency on cutechess's end,
etc.). Worth flagging as an open question: **if we re-deploy F1 to
Lichess and forfeits don't appear there, the forfeit is cutechess-
specific, not Coda's TM fault, and we can ship F1 as-is**.

Relevant scripts for reproducing:
- Tournament harness: Stage 6/9/11 under `/tmp/coda_stage*/` with
  `-tournament gauntlet` and a mix of Caissa/Berserk/Halogen
- Drift-trace binary: rebuild `fix/ponder-wait-loop` with the
  TM_TRACE eprintln in uci.rs (stderr-only; cutechess ignores stderr
  for clock).
- Cutechess's view (`-debug` flag on cutechess-cli) — gives exact send/
  receive timestamps per UCI line; compare to engine's reported time
  to catch protocol-level drift.

### Summary — what to prioritise

**Blocker for merging TM changes**:
1. Minimum-reserve floor (Issue 1) — simplest fix, biggest impact.
2. `max_hard` classical-TC fix (Issue 2) — only exposed when we
   actually run 40/15 games, but currently we'd time-underspend there.
3. Before shipping anything TM, deploy to Lichess and measure forfeit
   rate against cutechess's rate to disambiguate Issue 7.

**Already fixed** (in `fix/ponder-wait-loop`):
- Premature bestmove / protocol violation (Issue 5)
- Block-buffered stdout flush (Issue 4)

**Open investigations**:
- Where does the ~95% of forfeit drift come from? (Issue 7)
- Does F1 reproduce forfeits on Lichess? (needs production test)
