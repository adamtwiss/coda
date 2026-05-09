# Post-Tier-0 trunk retune plan

When the Tier 0 / Tier 4 SPRT cluster from 2026-05-09 (correctness audit)
lands on main, trunk needs a full-sweep retune. The Tier 0 fixes change
tree shape substantially:

- **Tier 0.1 N6 STM fix**: bench 3593209 → 4090568 (+13.8%). Trunk
  tunables were calibrated for "extension dead"; with the extension
  active, NMP/SE/extension cluster is miscalibrated.
- **Tier 0.2 cont_hist king-include**: small bench delta (+0.7%) but
  cont_hist signal now flows through movepicker for king moves at all
  4 plies for the first time. May shift LMR / move-ordering balance.
- **Tier 0.3 LMR_ENDGAME_PIECES = 5**: bench +34% (more endgame nodes
  searched without LMR). Endgame-specific tree shape change.

Plus any Tier 4/5 ablations that land H1 (each shifts the equilibrium
of the rest of the parameters).

## When to fire

After all of the following are resolved:
- Tier 0.1, 0.2, 0.3 SPRTs (`#1060`, `#1061`, `#1062`)
- Tier 4 ablations: A1 (`#1063`), A2 (`#1064`), A3 (HIST_BONUS — not yet
  fired), A4 (`#1067`)
- Tier 5 cross-engine: 5.1 SE_DEPTH (`#1066`), 5.3a LMP_BASE (`#1065`)

Merge anything H1 (or neutral correctness fix per user). Drop H0s. Then
fire trunk retune against the new merged trunk.

## Retune spec

**Scope**: full-sweep, all 83 tunables in `tunables!` macro at
`src/search.rs:60-260`. Generate spec from macro:

```bash
python3 -c 'import re
for line in open("src/search.rs"):
    m = re.match(r"    \((\w+), (-?\d+), (-?\d+), (-?\d+), ([\d.]+)\),", line)
    if m and m.group(1) != "name":
        print(f"{m.group(1)}, int, {m.group(2)}, {m.group(3)}, {m.group(4)}, {m.group(5)}, 0.002")' \
> scripts/tune_post_tier0.txt
```

**Iterations**: 12K-16K iter target for ~83 params (per √N rule:
~150-200 iter/param — see `feedback_spsa_snr_scales_inverse_sqrt_n`).
Tune-861 at 10K iter / 80 params SPRT'd negative for being undersized.

**Net**: production net per `cat net.txt` at trunk time. Per
CLAUDE.md "always tune against the net in net.txt". Pass matching
`--dev-network <SHA8>` to `ob_tune.py`.

**Submission**:
```bash
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune.py main \
  --params-file scripts/tune_post_tier0.txt \
  --iterations 14000 \
  --dev-network <SHA8>
```

**Bench**: re-measure on trunk after Tier 0 cluster merges
(`make && ./coda bench`). Pass as positional arg to `ob_tune.py`.

## Application rules

When tune resolves:

1. Compare per-param drift vs current trunk defaults
2. Watch for SPSA-vs-play-quality conflicts:
   - **LMR_ENDGAME_PIECES**: SPSA will try to push it back to 4. Floor
     is now 5. Honour the floor (or restore explicitly if floor
     somehow allows it). See
     `memory/project_lmr_endgame_pieces_play_quality.md`.
3. Apply outputs, re-bench, SPRT applied vs trunk at `[0, 3]`
4. If H1 → merge as new trunk
5. If H0 → diagnose which param flipped wrong direction; re-tune
   subset

## What this enables next

Post-retune, fire Tier 1 (NMP gate cascade) — biggest expected payoff
+3-8 Elo. Tier 1 was deliberately held in the 2026-05-09 plan to
avoid stacking structural changes before the trunk recalibrates.
