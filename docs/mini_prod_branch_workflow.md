# Mini-prod branch workflow

`mini-prod` is a long-lived branch that carries trunk tunables tuned
for the **baby-prod** S200 net (currently `cal-day0-factor-w15-warm30-hlcrelu-s200`,
SHA8 `61115E7F`). It exists to remove **tune-flation** in S200
experiment SPRTs — the asymmetry where a freshly-tuned experiment net
beats a baseline that uses untuned (S800-prod-calibrated) trunk
defaults.

See `docs/cross_engine_divergence_audit_2026-05-10.md` for the
generalised tune-flation problem this addresses.

## Why this exists

Without `mini-prod`, S200 experiment SPRTs compare:
- DEV: experiment-net + freshly-SPSA'd trunk → optimal
- BASE: baby-prod-net + main's trunk (calibrated for S800-prod) → suboptimal

The DEV side gets ~5-15 Elo of "free" tune-flation just from being
freshly tuned. With `mini-prod`, BASE uses trunk-tuned-for-baby-prod,
matching DEV's tune freshness.

## Branch invariants

- `mini-prod` defaults are SPSA-tuned for the baby-prod net.
- The baby-prod net is named in a comment in the `tunables!` macro at
  the top of `src/search.rs` (e.g. `// Tuned for: cal-day0-factor-...-s200, refit YYYY-MM-DD`).
- All search/eval code matches main; only tunable defaults differ.
- Bench is recorded in the most recent commit on `mini-prod`.

## How S200 experiments use it

Fork from `mini-prod` (not main):

```
git checkout mini-prod
git checkout -b experiment/foo-s200
# ... change ...
make && ./coda bench
```

SPRT submits with `--base-branch mini-prod`:

```
ob_submit.py experiment/foo-s200 <bench> \
  --base-branch mini-prod \
  --base-bench <mini-prod-bench> \
  --dev-network <foo-s200-sha> \
  --base-network 61115E7F  # cal-day0-S200 baby-prod
```

Both sides have S200-natively-tuned trunks → comparison is honest.

If the experiment lands (H1), merge it back to main via the same path
any other experiment uses. The mini-prod tunings stay on mini-prod;
main's tunings stay calibrated for prod.

## When mini-prod is refreshed (rebase + retune)

Rebase `mini-prod` onto current main when ANY of:

1. Main adds, removes, renames, or changes the range of a tunable
   (the `tunables!` macro structure changes — must re-merge defaults)
2. Main lands a search-shape change (new pruning feature, new gate,
   new extension) that may interact with tunings
3. Main has accumulated ~5+ Elo of merged changes since the last
   mini-prod refresh
4. A new baby-prod net is deployed (the "tuned for" reference changes)

Don't refresh during in-flight tunes against mini-prod — pause until
those resolve.

## Refresh procedure (procedural — runnable by Claude unattended)

```bash
# 1. Fetch main, rebase mini-prod onto main
git fetch origin
git checkout mini-prod
git pull origin mini-prod
git rebase origin/main
# Resolve any tunable conflicts (typical: main bumped a default for
# a small win — keep the mini-prod value if it was SPSA-found, else
# take main's). Document conflict resolution in commit message.

# 2. Build + bench fresh
make
./coda bench   # record this number for the rebase commit

# 3. Commit the rebase result with bench
git add -A
git commit -m "mini-prod: rebase onto main @ <main-sha>

Bench: <new-bench>"

# 4. Fire focused SPSA refit (~1500 iter, full-sweep)
OPENBENCH_PASSWORD=$OPENBENCH_PASSWORD python3 scripts/ob_tune.py mini-prod <bench> \
  --params-file scripts/tune_postmerge_A.txt \
  --iterations 1500 \
  --dev-network <baby-prod-sha> \
  --priority 0

# 5. Wait for tune to converge (~few hours fleet)
# 6. Apply tune outputs to mini-prod's tunables macro defaults
# 7. SPRT-validate: mini-prod (refit) vs mini-prod (pre-refit) at
#    [-3, 3] to confirm refit doesn't regress
# 8. If H1 or no-regression: push mini-prod
# 9. If H0 (regression): investigate — likely SPSA noise, may need
#    longer tune or rebase context investigation

git push origin mini-prod
```

## Naming archived mini-prods

When a baby-prod net is deployed (new training methodology produces a
new S200 reference net), archive the old mini-prod and start fresh:

```
git branch mini-prod-cal-day0  # archive old (named after the old baby-prod net)
git checkout main
git checkout -b mini-prod-fresh
# ... apply new baby-prod-tunings ...
git branch -M mini-prod-fresh mini-prod  # promote to canonical
git push origin mini-prod
```

This keeps `mini-prod` always pointing at the current canonical S200
baseline, while preserving history for diagnostic comparisons.

## Update CLAUDE.md SPRT methodology section

When mini-prod becomes operational, add a paragraph to CLAUDE.md
SPRT Testing Policy noting that S200 net-vs-net experiments should
fork from mini-prod and SPRT against it. Otherwise they're inflated.

## Open question: same approach for main?

The generalised tune-flation problem (any retune-on-branch experiment
gets unfair tune-freshness advantage if main has gone stale) suggests
the same mechanism for main: a `main-fresh-tune` branch that's the
canonical retune-on-branch fork point. We're starting with mini-prod
for S200 only because the asymmetry is biggest there (different net
+ different tune freshness compounding). If the mini-prod workflow
proves manageable, extend to a `main-fresh-tune` branch for trunk
experiments.
