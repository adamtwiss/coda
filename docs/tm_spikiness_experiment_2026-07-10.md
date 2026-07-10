# TM Spikiness Experiment — design (2026-07-10)

## Motivation: the measured deficit

Decomposition of the Coda↔SF gap on Atlas (32-concurrency RRs, noob_4moves,
Hash=512; SF18 = dev-20260318-d173a065):

| condition | Coda vs SF18 |
|---|---|
| fixed 15k nodes (no NPS, no TM) | −69 |
| fixed 50k nodes | −64 |
| fixed 0.2s/move (NPS in, TM out) | −52 ±14 |
| real STC 10+0.1 (950-game top-20 RR) | −73 ±18 |

NPS under RR contention is near-neutral (in-situ per-node cost ratio 1.16×;
idle single-thread it's 1.70× — platform question, separate track). The
fixed-movetime → real-TC drop isolates **active TM: ~20 ±15 Elo vs SF**.
(Run C, the 1000-game/engine STC ladder rerun, refines the TC-side number —
see kill criteria.)

Mechanism-level signature (per-move spends from the same PGNs, pooled):

| engine | med | p90 | p90/med | time forfeits |
|---|---|---|---|---|
| Coda | 0.170 | 0.44 | **2.6** | 0 (all runs, incl. VSTC) |
| SF14–SF18 | 0.14–0.15 | 0.50–0.55 | **3.4–3.7** | up to 18 @VSTC |

Coda spends **more on routine moves and less on critical ones** than every SF
version, at both STC and VSTC. Max single-move spend is actually *higher*
than SF's (4.0s vs 2.2–3.0s) — the tail exists (fail-low extension works);
the missing discrimination is in the mid-quantiles.

## Code-level diagnosis (search.rs Phase-13 dynamic TM, T=1 path)

Factor product on `soft_limit` (73% of clock share — high baseline):

1. **Stability table `[1.71, 1.20, 0.90, 0.80, 0.75]`** — floor 0.75 at 4+
   stable; head 1.71 (lowered from Viridithas's 2.50 after Phase 13.1's −41
   Elo opening overspend at 30+0.25). Up-side and down-side both compressed.
2. **Fail-low `1 + 0.34·min(2, fl)`** — caps at 1.68.
3. **Forced-move 0.386/0.627** — strong down-side, but fires ≤1×/search and
   only at depth ≥ 8 with margins 400/170.
4. **Subtree `(1.62 − frac)·1.4`, depth > 9** — **neutral point is
   frac = 0.905**. Typical best-move node fractions (0.5–0.7) yield
   1.2–1.5×, i.e. this "confidence" factor *inflates routine moves*. On the
   easy moves where SF banks time, we spend extra. Prime suspect for the
   high median.
5. **Score-trend clamp [0.80, 1.55]**, cross-move term.
6. Inc-ceiling at STC ≈ 13 (not binding); no-inc clamp 2.5 (untouched here).

Structural: the soft check (`elapsed_since_tm >= adjusted_soft`) runs **only
at iteration end**. With ~2–3× per-iteration growth, small `adjusted_soft`
values can't be honored — spend quantizes UP to the iteration boundary,
flattening the low end. (SF/Reckless granularity to be confirmed from source
before acting — see C4.)

## Metric

`p90/median` of pooled per-move spends vs the SF pool at 10+0.1 (spend regex
`([0-9]+\.[0-9]+)s\b` — NOT the score-first non-greedy trap). Baseline 2.6;
SF consensus 3.4–3.7; **target ≥ 3.2 with median ↓ toward 0.14s**.
Diagnostic only — Elo gates every promotion; do not Goodhart the ratio.
Guards: 0 time forfeits at STC (T=1, MoveOverhead default); LTC 40+0.4
non-regression before merge.

## Phase 0 — instrument (no fleet, no SPRT needed)

1. Extend the `TMDebug` per-move log line with the five factor values +
   `frac` + `adjusted_soft` + overshoot `(elapsed_at_stop − adjusted_soft)`.
   Debug-gated print, bench-neutral.
2. ~200 games Coda vs SF17 @10+0.1 conc 8 with TMDebug on. Produce:
   - distribution of best-move node fraction at stop (validates the
     frac-0.905 neutral-point diagnosis of factor 4);
   - stability-index-at-stop histogram (how often we sit at the 0.75 floor);
   - overshoot histogram (quantifies iteration quantization → is C4 needed?);
   - which factor binds the product, per move.
3. Add a `--shape` quantile mode (med/p75/p90/p95, per engine) to
   `tm_pattern_inspect.py` so the metric is one command on any PGN.

## Phase 1 — one-knob candidates (one branch each, never stacked)

**C1 — re-center the subtree factor (prime suspect).**
`TM_SUBTREE_BASE_100` tunable, default 162 → probe 130 (neutral at
frac ≈ 0.49), keep ×1.4 slope, clamp floor 0.55. Expected: routine-move
spend ↓15–25%, median ↓, ratio ↑. Risk: low — value shape matches
Viridithas/SF direction (their equivalents *reduce* on high effort).

**C2 — extend the stability decay tail.**
Table → `[1.71, 1.20, 0.90, 0.80, 0.75, 0.68, 0.62]`, index `min(6)`.
Very-stable moves (5+, endgame shuffles, recaptures) decay further, as SF's
timeReduction does. Expected: median ↓ on long-stable moves; opening
untouched (stab resets each move).

**C3 — raise the fail-low up-side.**
`1 + 0.50·min(3, fl)` (cap 1.68 → 2.50). The asp-fail-low is the
highest-precision "this move is hard" signal, and boosting it avoids the
known-bad stability-head raise (Phase 13.1 prior: head 2.50 = −41 Elo at
30+0.25 from opening overspend; fail-low ≠ opening-correlated). Expected:
p90 ↑ on churny moves, median unchanged.

**C4 — root-move-granularity soft stop (conditional on Phase 0).**
Only if overshoot quantization dominates (median overshoot ≳ 40% of
adjusted_soft): check the soft budget between root moves at the current
iteration (reusing the incomplete-iteration best-move machinery that hard
stops already exercise). Cross-engine consensus check FIRST (SF, Reckless,
Obsidian sources) per methodology — do not assume. Highest complexity;
run last.

Ordering: C1 → C2 → C3 (→ C4). C1+C2 shrink the median (freeing clock);
C3 spends the freed clock on hard moves. If C1 and C2 both pass shape+RR
gates individually, a C1+C2 combination branch is allowed (both down-side,
non-overlapping signals) before OB.

## Phase 2 — validation ladder (per candidate)

1. **Shape gate** (30 games vs SF17, conc 8, ~10 min): mechanism fired?
   ratio moved the designed direction? forfeits 0? Else fix or drop.
2. **Local RR** (~1800 games, conc 32, ~1.5h): 4-engine RR
   `{cand, base, SF17, SF18}` (2-Coda max rule respected). Read the
   **differential** (cand − base) vs the SF pair, ±~12 Elo resolution.
   Gate: differential ≥ +5 or clearly positive trend + shape target met.
3. **OB SPRT** `[0, 3]` STC first (TM has STC-visible history, e.g. #1568),
   then LTC. TM-class merge rule applies: accept any LTC verdict short of
   clear regression; don't gate merge on LTC magnitude.
4. **Deployment check for the final winner**: ponder-enabled cross-engine RR
   at 30+0.5 (ponderhit paths share the soft/floor machinery; Phase-13
   ponder constants are deliberately non-tunable — verify no interaction).

## Phase 3 — focused SPSA after a direction wins

Tunable-ize the shape constants touched by the winning candidate(s)
(subtree base, fail-low coef/cap, stability tail entries as `_100` ints;
6–8 params max — do NOT dump the table into the core set), LTC
`40.0+0.4 Threads=1 Hash=256`, 1000–1500 iters, warm-start from defaults.
SPRT the tuned values `[0, 3]` before merge.

## Kill / reprioritize criteria

- **Run C says no TC-side deficit** (STC ladder gap ≈ movetime gap −52±10):
  the ~20 Elo TM estimate collapses; downgrade to a `[0,3]` probe of C1 only
  (it's justified independently as a factor-shape correction) and stop.
- Shape moves but differential RR flat twice → the flatness is not
  Elo-relevant at STC; park with results logged in experiments.md.
- Any STC forfeit or VSTC forfeit rate > 0.3% → candidate rejected as-is.

## Bookkeeping

- One change per branch; log every H0/H1 + RR in experiments.md.
- Baseline shape numbers and decomposition PGNs: Atlas
  `/tmp/claude-1001/.../scratchpad/run{B,A150,A500,C}*.{log,pgn}` (session
  2026-07-10); re-derive with tm_pattern_inspect.py if needed.

## Phase 0 RESULTS (2026-07-10, 200 games vs SF17 @10+0.1 conc 8, 12,136 moves)

**Q1 — subtree factor (C1): CONFIRMED.** The "confidence" factor inflates
66% of all frac-computed moves (median subf 1.12, p90 1.72). Only 33% of
moves reach the frac>0.905 deflate zone; the 0.70–0.905 band (32% of moves —
high confidence!) still gets 1.0–1.29× inflation.

**Q2 — stability floor (C2): CONFIRMED, stronger than designed.** 58% of
moves stop at stability 10+, another 17% at 6–9 → ~75% of all moves saturate
the table at index 4 and share the same 0.75 floor. A stab-15 dead recapture
and a stab-4 position get identical treatment. Extend the decay tail through
stab ~8–10 (toward 0.60), not just 2 extra entries.

**Q3 — quantization (C4): borderline-GO, reshaped.** 56% of moves overshoot
the in-force budget by >20% (28% by 20–50%, 17.5% by 50–100%, 10.7% by
>100%); median overshoot among overshooters 29% (design gate was 40%).
Key observation: the next-iteration affordability estimate (2× last iter)
guards only the HARD limit — there is no soft-side "don't start an iteration
that will blow past adjusted_soft" check. That one-sided gate is a cheaper
C4 than mid-iteration aborts and directly attacks the >100% tail.

**NEW — fail-low factor is not a precision signal (C3 DEMOTED).** flf
p10/p50/p90 = 1.00/1.34/1.68: more than half of all searches hit ≥1
aspiration fail-low, so the "hard move" extension fires on the majority of
moves and inflates the median rather than the tail. Raising it (original C3)
would flatten FURTHER. Either the aspiration windows are too narrow (check
ASP_* calibration vs peers — separate diagnostic thread) or the fail-low
counter needs a depth/severity gate before it deserves a bigger multiplier.

**Factor medians (p10/p50/p90):** stabf 0.75/0.75/1.20 · flf 1.00/1.34/1.68 ·
forcedf 0.63/1.00/1.00 · subf 0.91/1.11/1.72 · trendf 0.91/1.00/1.11 →
product 0.66/1.17/2.56. The median MOVE runs at 1.17× soft despite the
stability floor: flf and subf systematically cancel the stability discount.
That is the flat-spend mechanism in one line.

**Revised Phase-1 order: C2 (extend stability decay) → C1 (subtree
re-center) → C4' (soft-side next-iteration gate) → C3 parked pending
aspiration-window review.**
