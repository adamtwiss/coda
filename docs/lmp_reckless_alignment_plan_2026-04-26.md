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

- **#1 — `experiment/lmp-direct-check-carveout`** — single-line add `&& !board.gives_direct_check(mv)` inside LMP gate. **Submitted as SPRT [-3, 3]** 2026-04-26. Bench 736194 → 796065 (+8% — fewer LMP fires let checking late moves through).

- **#2 — `experiment/lmp-skip-quiets`** — **ABANDONED 2026-04-26.** Intended as a perf-side optimisation (set flag when LMP fires; bypass remaining quiets in loop). Produced an unexplained bench delta of +22% (736K → 900K), incompatible with "perf-only" framing. Bisection isolated: variable declaration alone is bench-neutral; check-disabled-but-setter-on is bench-neutral; check enabled is +22%. The check `if skip_quiets && !is_cap && !is_promo { continue; }` should be equivalent to letting LMP fire on the same moves (move_count is already past lmp_limit; LMP gate re-fires each iteration). It isn't, for reasons not localised in this debug session. Need per-move trace instrumentation to find the divergence. Not blocking; #1 alone advances Phase A.

### Phase B — structural threshold change (after Phase A)

Build on whichever of #1/#2 land H1.

- **#3 + #4 + #5 — `experiment/lmp-reckless-shape`** — replace threshold formula with Reckless's history-aware continuous-improvement form:
  ```rust
  let lmp_limit = (
      tp(&LMP_BASE_NEW) * 1024
      + tp(&LMP_IMPROVEMENT_K) * improvement / 16
      + tp(&LMP_DEPTH_K) * depth * depth
      + tp(&LMP_HIST_K) * main_hist / 1024
  ) / 1024;
  ```
  New tunables (5): `LMP_BASE_NEW`, `LMP_IMPROVEMENT_K`, `LMP_DEPTH_K`, `LMP_HIST_K`, plus possibly retain `LMP_DEPTH` as upper-cap for now.
  SPRT `[0, 5]`. Likely H0 at default tunables; **full-sweep retune on branch is mandatory**, then SPRT post-tune package vs main.

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
