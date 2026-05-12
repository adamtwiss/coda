# Improvement Portfolio — Diversified Threads

Living reference doc. Names and characterises the parallel work threads
through which Elo gains arrive. Used as a tag on individual experiments
("which thread does this sit in?") to avoid accidental concentration on
one axis. Edit incrementally as threads start/stop delivering.

Beyond the flywheel (eval → ordering → pruning → depth → eval), the
gap-closing strategy runs multiple ORTHOGONAL threads in parallel.
Correlated work can plateau together; diversified threads hedge.

## Threads

**Eval-search flywheel** — compounding loop: better eval → better
ordering → safer pruning → more depth → better self-play training data
→ better eval. Captured via **retune-on-branch** after each feature: a
+1-2 Elo raw feature is often +5-10 Elo post-retune. Reckless-value
imports fail because their values are calibrated against their
flywheel; ours is different. Outliers are directional signals;
SPSA-retune-on-branch is the portable-translation mechanism.

**Correctness audits** — bugs in rarely-fired paths historically
deliver +3-30 Elo (50-move rule, LMR endgame gate, SMP races, TB
integration, `is_pseudo_legal` EP hole, N6 STM, cont_hist
off-by-one). Highest Elo-per-hour lane, under-invested. If torn
between a +2 Elo feature and a correctness audit branch, audit first.

**Comparative engine review with instrumentation** — reading
top-engine source tells you *what*; patching them with `dbg_hit`-style
counters tells you *when and how often*. Different information class.
A half-day of instrumented comparison often surfaces gaps a full day
of source-reading misses. `scripts/reckless_evalbench.patch` is the
entry point.

**Training hyperparameters + data** — historical big wins are
1-line schedule changes (low-final-LR, ply≥16 quiet filter) and
data-diversity additions. Iteration time is now ~2.5h for SB200 with
idle GPUs available — the constraint is ideas, not wall-clock. Open
sub-threads: filtering sweep (ply thresholds, quiet definition
variations), output-bucket layouts, FT-width follow-on (1024 has been
tried), late-stage fine-tuning (so far unsuccessful but believed to
have legs), distillation, curriculum / position weighting.

**Long tunes / long training** — top engines run 25K-iter SPSA
routinely. Per `feedback_spsa_snr_scales_inverse_sqrt_n`, SNR scales
as √N. Full-sweep 80-param tunes at 10K iters worked (#1070 H1
+2.8); 15K iters delivered an additional +1.8 (#1119 H1). Pattern:
don't half-bake — complete the schedule you start. Diminishing
returns past ~15-20K for our regime; not much evidence longer alone
helps beyond that.

**Time management** — historically had correctness bugs costing real
Lichess Elo. Now uniform with top engines (see Lichess move-time
graph comparisons). Potential +5-10 Elo from making TM params
SPSA-tunable, **but only if validated at LTC** — TM features are
mostly invisible in STC SPRTs (~200ms per move). Plan the validation
regime before parameterising.

**Infrastructure (Lichess-visible, SPRT-invisible)** — opening book
A/B, TB entry timing / DTZ walkback quality, TM edge cases. Live-watch
on Lichess catches qualitative bugs that SPRT/SPSA can't.

**Loss analysis** — meta-thread that informs the others rather than
delivering Elo directly. Rerunning periodically (post-NMP-cascade,
post-next-net) surfaces where the gap currently sits. See
`loss_analysis_2026-04-28.md` and re-author when redoing.

## Thread-selection heuristic

Name the thread first ("flywheel", "correctness", "comparative",
"training", "long-tune", "TM", "infrastructure") before picking the
next experiment — prevents accidental concentration on one axis. If a
thread hasn't delivered in 4+ weeks, prefer it.

Different threads use different resource pools:
- Flywheel / long-tune / TM / pruning retunes → SPRT fleet
- Training → GPU hours
- Correctness / comparative review / infrastructure → dev time
- Loss analysis → dev time, informs all others

When deciding what to work on next, check which pool is constrained.
Idle GPU + busy SPRT fleet → favour training.
