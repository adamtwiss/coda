# Plan: threat-generation ray pre-gating (NPS lever) — 2026-06-17

**Class:** bit-identical NPS optimization (eval unchanged). Improvement-portfolio
thread: *engine NPS / effective depth*. Measurement-gated — cheap Stage 0 kills it
early if the waste isn't reclaimable.

## TL;DR

The SF speed gap in threats is **not** one clean lever. It decomposes into three,
and only one is open:

1. **Delta *volume* (Coda 8.69/move vs SF 4.46)** — mostly Coda's x-ray
   "slider-sees" richness (the +187 Elo feature, SPRT #2014). SF's threat model is
   genuinely *leaner* (one discovered piece per ray). Cutting our volume to SF's =
   cutting eval. **Off the table.**
2. **Apply-side `double_inc_update` (cross-ply combine)** — SF uses it; Coda ported
   it as `recapture-combine` and it **SPRT'd H0 −2.4 (#2015)**. Replay-gap profile
   shows only **16.7% of materializations are gap≥2** (the only case it can help),
   so the opportunity is bounded near-nothing. **Rejected, correctly.**
3. **Wasted ray-scanning in x-ray *generation*** — the x-ray/slider sections burn
   **~22% of total cycles** at **73–78% zero-emit**. Pre-gating dead rays is
   bit-identical (pure speed). **The one open lever — but partly already captured.**

**Expected NPS: ~2–3% (best guess), 2–4% feasible, ~5% optimistic, with real
probability of ~neutral.** Elo ~flat to +2–3 STC. Worth a measurement-gated try;
not a needle-mover.

## How we got here (investigation, 2026-06-17)

- **SF does NOT relocate cost to make/unmake.** SF updates the accumulator lazily
  at eval (`AccumulatorStack::evaluate` → `forward/backward_update_incremental`),
  walking back to the nearest computed accumulator and applying deltas forward —
  the same strategy as Coda's Finny/lazy. So the gap is genuine per-update
  efficiency, not an accounting artifact.
- **This SF version has threats too** (`DirtyThreats`, `update_piece_threats`,
  `AccumulatorState<ThreatFeatureSet>`), so "Coda 31% threat-apply vs SF 5.5%" is a
  *same-work* comparison, not "the price of a feature SF lacks."
- **Both generators are localized** (per dirty piece: moved from/to, captured, rook)
  — neither re-enumerates the board. SF: `update_piece_threats(pc, s, dts)` per
  dirty piece. Coda: `compute_move_deltas` (threats.rs:1407) scans only the moved
  piece's old/new attacks + attackers on from/to + sliders through them.
- **Coda already pre-gates the easy cases:** `do_z_finding` (threats.rs:~1181)
  skips the x-ray Z-block when no ray can produce a Z delta; `ray_extension` tables
  replace magic lookups. So the obvious win is taken — residual zero-emit is harder.

## The target

Per-section generation profile (prod net 549C20A5, `--features profile-threats`,
`bench 12`):

| section | %cyc | cy/call | deltas/call | zero-emit |
|---|---|---|---|---|
| direct (step 1, base) | 7.8% | 47.7 | 1.51 | 29.5% |
| own-xray (step 1b) | 5.7% | 35.2 | 0.41 | **73.8%** |
| sliders (step 2, x-ray) | 9.9% | 60.9 | 0.82 | 51.0% |
| sliders-2b (step 2b) | 6.5% | 39.7 | 0.37 | **78.2%** |
| nonsliders (step 3, base) | 5.0% | 30.7 | 0.57 | 58.5% |

- Generated **8.69 deltas/move** (caching-immune); apply 9.93/call (lazy-inflated).
- Replay gap: **83.3% gap==1, 16.7% gap≥2** (bounds the dead `double_inc` lever).
- The three x-ray sections (1b, 2, 2b) = **22.1% of cycles, ~14% of total cycles is
  zero-emit** (own-xray 4.2% + sliders 5.0% + sliders-2b 5.1%).

## Stage 0 — diagnostic (DONE 2026-06-17 via reason counters)

Adam + another agent landed the instrumentation (commit `7cc4980`: per-reason
counters for own-xray / sliders-2b emission paths + `eval-bench make-unmake-all` +
move-type delta buckets). Ran it; results:

**Reason funnels (full bench 12, prod net):**
- own-xray (8.08M calls, 75% zero-emit): cheap early exits `nonslider` 2.70M +
  `no-direct` 0.29M; **did-work-emit-nothing bucket = `no-behind` 3.08M** (found a
  blocker, nothing behind).
- sliders-2b (7.24M calls, 78% zero-emit): cheap early exit `no-candidates` 2.14M;
  only `exact1` 1.22M emit; **did-work-emit-nothing bucket = `blockers0` 3.39M**
  (had candidate sliders, zero blockers between → x-ray impossible).

**make-unmake-all microbench (isolated, bit-identical harness):**
- Delta volume by move type: quiet **10.34**/move, capture **15.76**, castle
  **27.0**. x-ray sections = **10.6% of microbench cycles** (own-xray 3.1% +
  sliders 3.9% + sliders-2b 3.6%), 45–62% zero-emit (lower than full-bench 73–78%
  — workload-dependent).

**Verdict: inconclusive-leaning-positive.** There ARE large did-work-emit-nothing
buckets (~3M+3M). But reclaimability is NOT yet proven — it hinges on whether
`no-behind`/`blockers0` are determined by an *expensive* magic-lookup/ray-walk
(reclaimable with a cheap pre-test) or the *already-cheap* `ray_extension` table
read (the prior `do_z_finding` pass already got it → not reclaimable). **That is
the Stage 0.5 code read below — the real kill gate.** Estimate stays ~2–4%; do NOT
nudge up until the code shows the determination is expensive.

## Stage 0.5 — the actual kill gate (code read) → **STOP, CLOSED 2026-06-17**

Read the determination paths. **The SF-style pre-gate is already implemented:**
- Section 2 (sliders): Z-finding via `ray_extension(slider_sq, square) & occ` —
  table reads, gated by `do_z_finding` (threats.rs:~1308).
- Section 2b (sliders-2b): `blockers0` determined via `between(S, sq) & occ` —
  cheap table read + AND. Code comment: *"replaces the previous 8-direction scalar
  ray walks with a slider iteration driven by the precomputed `between()` table.
  Per-call work now scales with number of sliders on aligned rays (typically 0–4)."*
  This is exactly SF's `noRaysContaining`/`RayPassBB` gating technique.

So `no-behind`/`blockers0` are determined CHEAPLY (table reads), not via expensive
magic-lookups/ray-walks. A prior optimization pass already captured the lever
(`do_z_finding` + `ray_extension` + the `between()`-table 2b rewrite). The residual
zero-emit cost is irreducible aligned-slider iteration, not reclaimable waste.

**Decision: CLOSED, negative.** Expected gain ~neutral / <1–2% — the rewrite would
largely re-do existing work. The threat-generation path is already pre-gated to
SF-style efficiency; the remaining 22%-cycle cost is (a) x-ray richness = the +187
Elo eval (keep) and (b) inherent per-slider table-read iteration. No free
implementation win in the threat path.

Reconciliation with "SF is more efficient at threats": true, but the advantage is a
LEANER threat model (8.41 vs 4.46 deltas/move = our x-ray eval richness, not free) +
code techniques Coda already matches — NOT an un-ported optimization. File to
strength_frontier: threat NPS gap is the eval-richness tradeoff + rejected apply
combine, structurally not cheaply closable. Perf headroom, if any, is elsewhere
(non-threat paths).

## Stage 1 — port SF's pre-gate technique (only if Stage 0 passes)

SF gates rays with bitboard masks *before* scanning:
- `noRaysContaining` — skip rays that can't carry a relevant threat.
- `RayPassBB[sliderSq][s]` intersected with occupancy to detect the single
  discovered piece (`assert(!more_than_one(discovered))`) instead of a walk.

Port the analogous board-level pre-gate into the own-xray (1b) and sliders-2b (2b)
sections — the two highest zero-emit (73.8% / 78.2%). Concretely: compute a
once-per-square mask of rays that *could* contain a blocker→target pair, and skip
the per-slider inner work when the mask is empty (extends `do_z_finding`'s idea to
1b/2b, which currently lack it).

Keep it bit-identical: the pre-gate must only skip work that provably emits nothing.

## Verification & acceptance

1. **Bit-identical:** `CODA_VERIFY_NNUE=1 ./coda bench 8` → 0 mismatches; `./coda
   bench` node count UNCHANGED vs HEAD (it's pure speed — any node delta = a bug).
2. **NPS on a deployment-class host, NOT Hercules** (Zen5 9700X). Hercules is the
   memory-bound outlier and over-states bandwidth opts AND under-states... actually
   for a *compute* cycle-saving, measure single-thread on Zen5 + a contended run,
   since deployment is contended. Report both.
3. **SPRT:** bit-identical perf change → bounds `[-2, 1]` (non-regression; the gain
   is NPS, which SPRT may under-detect — `feedback_worker_variance_only_for_perf_
   sensitive_changes`, `feedback_hercules_bench_overstates_bandwidth_opts`). Accept
   on non-regression + a measured Zen5 NPS gain; do not gate merge on SPRT Elo
   magnitude for a bit-identical speed change.
4. Pull per-worker breakdown (perf-sensitive change → 50+ Elo worker spread).

## Risks / why this might be neutral

- **Easy gates already done** (`do_z_finding`, `ray_extension`) — residual waste may
  be inherent. This is the #1 risk; Stage 0 exists to catch it.
- **Threat micro-opt track record is neutral** (`project_threat_index_microopts_
  neutral`) — small slices haven't converted to measured Elo before. This lever is
  more targeted (the zero-emit angle specifically) but the prior is real.
- **Bandwidth-bound under contention** — single-thread cycle savings convert poorly
  at the contended deployment regime (`project_coda_bandwidth_starvation_under_
  concurrency`).
- **Pre-gate overhead** — the gate adds a test to *every* call (emit and zero-emit);
  if the emit fraction is high in some sections, net could be flat or negative.

## Decision

Run Stage 0 (cheap). If it shows reclaimable ungated scan, do Stage 1 + verify. If
not, log the negative and close — the threat NPS gap is then established as
"x-ray richness (eval, keep) + already-optimized generation + rejected apply
combine," i.e. structurally not cheaply closable, which is itself a useful,
file-able conclusion for the strength-frontier doc.
