# LMP Alignment with Reckless — Phased Plan

**Date:** 2026-04-26
**Owner:** Hercules
**Status:** Active

## Hypothesis

Coda fires LMP **28× more often per Kn** than Reckless (per
`docs/reckless_vs_coda_pruning_diff_2026-04-25.md`). The doc's
original framing ("Coda over-prunes, weak ordering") is partially
right — but at same node budget Coda d13 ≈ Reckless d12, so the
heavy gate firing is doing real work.

Adam's deeper hypothesis: **NMP cutoff rate (Coda 30% vs
Reckless 57%) is partly a side effect of over-aggressive LMP**.
When LMP fires at a parent, the child subtree is never explored
and its NMP attempt never happens. 28× more LMP at parents →
fewer downstream NMP opportunities.

Tune-796 trajectory analysis (2026-04-26) showed empirical
support: NMP cluster moved deeper (MIN_DEPTH +12.8%) AND more
aggressive (BASE_R +9.2%, DEPTH_DIV +12.3%) while LMP_DEPTH cap
pulled IN (-6.5%) and HIST_PRUNE_DEPTH pulled IN (-12.1%).
Each gate is finding a depth niche.

## Differences vs Reckless

7 differences identified between Coda's and Reckless's LMP:

| # | Diff | Direction | Independent? |
|---|---|---|---|
| 1 | Direct-check carve-out (`!is_direct_check(mv)`) | Reckless has, Coda lacks | Yes |
| 2 | `skip_quiets = true` flag after LMP fires | Reckless has, Coda re-evaluates | Yes |
| 3 | History term in threshold (`+ K * history / 1024`) | Reckless has | Coupled with #4, #5 |
| 4 | Continuous improvement scaling (`+ K * improvement / N`) | Reckless continuous, Coda discrete `/(2-imp)` | Coupled with #3, #5 |
| 5 | Threshold magnitude (Reckless 2-2.7× higher base) | Coupled formula change | Coupled with #3, #4 |
| 6 | `!is_pv` gate (Coda has, Reckless lacks) | Removing INCREASES firing | Yes (but wrong direction at present) |
| 7 | `depth <= LMP_DEPTH (13)` cap (Coda has, Reckless lacks) | Removing INCREASES firing | Yes (but wrong direction at present) |

## Phased execution

### Phase A — independent surgical changes (in flight or imminent)

- **#1 — `experiment/lmp-direct-check-carveout`** — single-line add `&& !board.gives_direct_check(mv)` inside LMP gate. **MERGED 2026-04-26** (SPRT #808 H1 +2.54). Bench 736194 → 796065 (+8% — fewer LMP fires let checking late moves through).

- **#2 — `experiment/lmp-skip-quiets`** — Reckless pattern (search.rs:752) — set flag when LMP fires; bypass remaining quiets in loop. **Submitted as SPRT [-3, 3]** 2026-04-26. Bench 736194 → 899738 (+22%). Bisection: variable declaration alone is bench-neutral; check-disabled-but-setter-on is bench-neutral; check enabled is +22%. By logic this should be equivalent to letting LMP re-fire on each late quiet (move_count past lmp_limit on all subsequent quiets), but empirically it isn't. SPRT'd as a data point per Adam — outcome interpretation needs care since the "perf-only" framing didn't hold. Possible explanations: (1) compiler/optimisation quirk, (2) gate-equivalence assumption is wrong somewhere, (3) move-ordering interaction. Per-move trace instrumentation is the follow-up if SPRT result demands explanation.

### Phase B — structural threshold change (after Phase A)

Build on whichever of #1/#2 land H1.

- **#3 + #4 + #5 — `experiment/lmp-reckless-shape`** — replace threshold formula with Reckless's history-aware continuous-improvement form:
  ```rust
  let lmp_limit = (
      tp(&LMP_K_BASE)
      + tp(&LMP_K_IMP) * improvement / 16
      + tp(&LMP_K_DEPTH) * depth * depth
      + tp(&LMP_K_HIST) * main_hist / 1024
  ) / 1024;
  ```
  New tunables (4): `LMP_K_BASE`, `LMP_K_IMP`, `LMP_K_DEPTH`, `LMP_K_HIST`. `LMP_DEPTH` cap retained.
  SPRT [0, 5]. Full-sweep retune on branch. SPSA **#811** finished; LMP_K_HIST stayed flat at 67-68 through 2500 iters (no SPSA gradient). Post-tune SPRT **#818 H0 -2.4 ±3.2 / 9134g** 2026-04-27.

  **Phase B Option B follow-up (2026-04-27):** mirroring the #816/#817 SE diagnostic that paid +2.5 Elo, strip the history term (#3) and retune the simpler #4+#5 formula. Branch `experiment/lmp-reckless-shape-no-hist` (commit c23e4cc, bench 1148288) — drops `LMP_K_HIST` tunable + `lmp_main_hist` lookup, keeps continuous-improvement scaling and magnitude rebase. SPSA **#819** running (79 params, 2500 iters from tune-811 starting values). Post-tune SPRT vs main is the test.

  **If Option B H1:** the formula reshape carries Elo, the history term was the noise feature. Ship it, queue Phase C.

  **If Option B H0:** Phase B as a whole doesn't pay. Phase A (+4.4 banked) was the bulk of the LMP-alignment Elo. Skip Phase C (gate removals would re-introduce 28× firing problem from a worse starting point).

### Phase C — gate removals (later, only if Phase B lands)

After Phase B's retune finds the basin:

- **#6 — drop `!is_pv` gate** — let LMP fire at PV nodes. Currently increases firing in the wrong direction; only viable after Phase B has shifted Coda's overall LMP volume down.
- **#7 — drop depth cap** — let LMP fire at all depths. Same logic.

Each tested independently, both with SPSA retune-on-branch.

## Success metric

End state we're aiming for, after all phases:

- Coda LMP fires/Kn drops from 573 toward Reckless's 20 ballpark
  (or some intermediate equilibrium SPSA finds).
- Coda NMP cutoff rate rises from 30% toward 57%.
- Net Elo: positive after retune-on-branch at each phase.
- EBF reduction (track in bench output).
- Re-run `dbg_hit` instrumented compare with Reckless after Phase B
  to confirm the gate-firing distribution shifts as expected.

## What we are NOT doing

- Pure ablation (NO_LMP) — per `feedback_ablation_vs_equilibrium_for_overlapping_gates.md`, ablation breaks equilibrium and tells us nothing about the alternative balance.
- Forcing Reckless's exact tunable values — Coda's pruning context is different; SPSA finds Coda's optimum.
- Quick-port and ship without retune — the gate-vs-gate dynamics require full-sweep on branch (per `feedback_extensions_yin_yang_pruning.md` and `feedback_net_swap_needs_full_sweep_retune.md`).

## Decision log

- **2026-04-26 plan written.** Adam: prefer phased independent tests for #1/#2, build #3/#4/#5 on top assuming #1/#2 hold. Defer #6/#7 to Phase C; flag for later inclusion as "the right balance" likely involves them too.
