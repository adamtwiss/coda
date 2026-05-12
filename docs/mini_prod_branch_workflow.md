# Mini-prod branch workflow

`mini-prod` is a long-lived branch carrying trunk tunables that are
SPSA-calibrated for the current **baby-prod S200 net** (currently
`cal-day0-factor-w15-warm30-hlcrelu-s200`, SHA8 `61115E7F`). It exists
to remove **tune-flation** in S200 experiment SPRTs — the asymmetry
where a freshly-tuned experiment net beats a baseline that uses
untuned (S800-prod-calibrated) trunk defaults.

The branch name stays `mini-prod` across baby-prod net rotations; the
current reference net is recorded in `net.txt` and a comment at the top
of the `tunables!` macro in `src/search.rs`.

See `MINI_PROD.md` (top of repo on the `mini-prod` branch) for the
quick-reference / current state. This document is the methodology +
agreed policies behind that workflow.

## Why this exists

Without `mini-prod`, S200 experiment SPRTs compare:
- **DEV**: experiment-net + freshly-SPSA'd trunk → optimal
- **BASE**: baby-prod-net + main's trunk (calibrated for S800-prod) → suboptimal

The DEV side gets ~+4-6 Elo of "free" tune-flation just from being
freshly tuned, independent of any real improvement. With `mini-prod`,
BASE uses trunk-tuned-for-baby-prod, matching DEV's tune freshness.

This was directly measured 2026-05-11 via #1117 (mini-prod tuned-vs-
untuned, same net on both sides): the latent tuning headroom on the
S200 net was ~+4 Elo. See
`memory/feedback_s200_paired_probes_carry_recipe_handicap_noise.md`.

## Branch invariants

- `mini-prod` defaults are SPSA-tuned for the current baby-prod net.
- The baby-prod net is named in net.txt AND in a comment at the top of
  the `tunables!` macro on the branch.
- All search/eval CODE matches main; only tunable defaults differ
  (between refreshes — see merge cadence below).
- Bench is recorded in the most recent commit on `mini-prod`.

## Why main does NOT need the same treatment

Main's tunables are always calibrated for its current net (the SB800
production net). There is no S800-vs-different-S800 tune-flation
problem because main is the canonical reference: any SB800 experiment
fork starts from main's tunables and they're correctly calibrated for
the eval scale the experiment will be measured against.

The asymmetry mini-prod addresses is **specifically** about main's
tunables being calibrated for a *different net* than the S200 experiment
runs on. That doesn't happen on main itself.

(This was an open question in the 2026-05-11 draft of this doc and is
resolved 2026-05-12 — Adam.)

## How S200 experiments use mini-prod

Fork from `mini-prod` (NOT main):

```bash
git checkout mini-prod
git checkout -b experiment/foo-s200
# ... make S200 experimental change (different net via net.txt, an
#     architecture probe, training-recipe change, etc.) ...
make && ./coda bench
```

SPRT submits with `--base-branch mini-prod`:

```bash
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py experiment/foo-s200 <bench> \
    --base-branch mini-prod \
    --base-bench <mini-prod-bench> \
    --dev-network <foo-s200-sha> \
    --base-network 61115E7F   # current baby-prod
```

Both sides have S200-natively-tuned trunks → the SPRT measures pure
net/feature quality difference, not tune-freshness asymmetry.

**If the experiment lands (H1)**, merge it back through main as any
other experiment — see "Merge cadence" below. The mini-prod tunings
stay on mini-prod; main's tunings stay calibrated for prod.

## Merge cadence (asymmetric — important)

**To main** (normal SPRT flow):
- H1 search/eval wins merge to main ASAP after passing self-play SPRT.

**To mini-prod** (lagged):
- Same wins propagate to mini-prod ONLY during scheduled refresh
  windows when **no S200 experiments are in flight**.
- A mid-flight base shift would invalidate in-progress mini-prod SPRTs.

So mini-prod intentionally lags main between refreshes. That's the
design.

## Refresh procedure (rebase + retune)

Agreed 2026-05-12 (Adam): mini-prod stays in sync with main via
**periodic rebase + retune**, NOT recreation from scratch. This
preserves mini-prod's cumulative tuning evolution while taking main's
structural changes forward.

### When to trigger a refresh

When ANY of:

1. Main has changed the `tunables!` macro structure (added, removed,
   renamed, or widened range on a tunable).
2. Main landed a search-shape change (new pruning feature / gate /
   extension) that may interact with tuned values.
3. Main has accumulated **~5+ Elo** of merged changes since the last
   mini-prod refresh.

Plus: **a fresh refresh** (not a rotation) is appropriate when those
triggers fire AND no S200 experiments are in flight.

### Procedure

```bash
# 1. Fetch main, rebase mini-prod onto main
git fetch origin
git checkout mini-prod
git pull origin mini-prod
git rebase origin/main
```

**Conflict resolution policy** (on tunable-default conflicts during
rebase):
- Take main's STRUCTURAL changes (new tunables, renames, range
  widenings, new features).
- Keep mini-prod's tuned VALUES for any tunable that exists on both
  sides (those are S200-calibrated; main's values are S800-calibrated
  and inappropriate for mini-prod).

```bash
# 2. Build + bench fresh
make
./coda bench   # record this number

# 3. Commit the rebase result
git add -A
git commit -m "mini-prod: rebase onto main @ <main-sha>

Bench: <new-bench>"

# 4. Fire focused full-sweep SPSA refit (~1500 iter)
OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_tune.py mini-prod \
  --iterations 1500 \
  --dev-network 61115E7F \
  --priority 0

# 5. Wait for tune convergence (~few hours fleet)
# 6. Apply tune outputs
curl -s -u <ob_creds> 'https://ob.atwiss.com/api/spsa/<TUNE_ID>/outputs/' \
  > /tmp/refresh.txt
python3 /tmp/apply_tune.py /tmp/refresh.txt
make && ./coda bench

git commit -am "mini-prod: apply tune-#<TUNE_ID> outputs

Bench: <new-bench>"

# 7. SPRT-validate refresh didn't regress
OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_submit.py mini-prod \
  --base-branch <pre-refresh-mini-prod-sha> \
  --bounds '[-3, 3]'

# 8. On H1 or no-regression: push. On H0: investigate.
git push origin mini-prod

# 9. Update MINI_PROD.md "Current state" section + push
```

## Baby-prod net rotation (different from refresh)

A **net rotation** is when a new training methodology produces a
DIFFERENT baby-prod net (different architecture or training recipe —
not just a more-baked version of the same net). The eval scale and
landscape shift enough that rebase + retune isn't appropriate;
start fresh.

```bash
# Archive the old mini-prod for diagnostics
git checkout mini-prod
git branch mini-prod-<old-net-shortname>-archive
git push origin mini-prod-<old-net-shortname>-archive

# Reset mini-prod from current main
git checkout main
git pull origin main
git branch -D mini-prod
git checkout main -b mini-prod

# Set net.txt to the new baby-prod, commit
# ...

# Fire a fresh ~2500-iter full-sweep tune from main defaults
OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_tune.py mini-prod \
  --iterations 2500 \
  --dev-network <new-baby-prod-sha> \
  --priority 0

# Apply, validate (this time validate vs a known untuned-trunk
# baseline at S200 to confirm tune-flation has been removed)
# ...

git push origin mini-prod -f   # force-push because we recreated from main
```

Note the `-f` push: rotation is the one time mini-prod's history
restarts. Archive branches preserve the old timeline.

## Cross-references

- `MINI_PROD.md` (on the mini-prod branch, top-level) — quick
  reference, current state, runnable commands.
- `docs/cross_engine_divergence_audit_2026-05-10.md` — generalised
  tune-flation problem.
- `memory/feedback_s200_paired_probes_carry_recipe_handicap_noise.md`
  — empirical measurement of the headroom mini-prod neutralises.
- `CLAUDE.md` §SPRT Testing Policy — the binding policy that any S200
  experiment SPRT must fork from mini-prod, not main.
