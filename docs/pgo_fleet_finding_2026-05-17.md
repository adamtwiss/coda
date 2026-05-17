# PGO Re-Test: Fleet-Fragile, Not Shipping (2026-05-17)

Follow-up to the SMP scaling investigation
(`smp_scaling_investigation_2026-05-17.md`) which flagged "verify PGO
works again with cgu=16" as a likely +5-10 Elo win.

**Result: not shipping.** PGO with the bench-13 profile is
hardware-fragile across the fleet and is at best neutral on the
production-equivalent hardware (Lichess host = "fast ionos" cohort).
Local NPS measurements on a dev box overstated the gain by ~10×.

## What we tested

Branch `experiment/pgo-openbench-default`. On the branch, `make` (the
default target) routes through `cargo pgo instrument build → ./coda
bench 13 → cargo pgo optimize build`. Workers needed
`cargo install cargo-pgo` + `rustup component add llvm-tools-preview`.
Main was left as plain release. SPRT #1299, T=1, 10+0.1, [0, 3]
bounds.

## Headline result

OB SPRT: **-1.7 ±2.9 at 16334 games, →H0 lock** (stopped early once
the cohort split was understood).

## The cohort split

Per-machine NPS ratio (dev = PGO, base = plain) and per-machine Elo,
grouped by hardware:

| Class | CPU | Hosts | Mean NPS Δ | Mean Elo | Notes |
|---|---|---|---:|---:|---|
| Slow ionos (6c) | Intel, older Xeon-class | ionos1/2/3/7/8/12/13 | +1.1% | +1.6 | PGO mildly helpful |
| Fast ionos (6c) | Intel, newer Xeon-class | ionos6/9/11/14/15/16 | -1.0% | +0.4 | ~wash — **production-equivalent (Lichess host class)** |
| hercules (8c/16t) | **Intel Xeon E-2288G** (Coffee Lake, ~2019) | 1 host | +16.7% | +8.7 | Largest win |
| titan (32t) + Atlas (dev box) | **AMD EPYC 7351P** (Zen 1, 2017) | 1 fleet host (+ dev) | +4.2% | +1.1 | Local dev measurement of +9% bench NPS did *not* fully replicate (titan: +4.2%) |
| **zeus (8c/16t)** | **AMD Ryzen 7 9700X** (Zen 5, 2024) | 1 host | **-10.0%** | **-18.5** | **Off the cliff** |

**The pattern is simple: PGO benefit decreases monotonically with
hardware age.**

- **Older silicon → PGO helps.** Slow ionos (older Intel),
  hercules (Coffee Lake Xeon, 2019), titan/Atlas (Zen 1, 2017) — all
  show positive Elo.
- **Newish silicon → PGO is neutral.** Fast ionos (newer Intel
  Xeon-class) lands at ~+0.4 Elo / -1% NPS. This is the
  Lichess-host class — production-equivalent.
- **New silicon → PGO hurts.** Zeus (Zen 5, 2024) is sharply
  negative.

Cross-vendor agreement on the age axis (Intel hercules wins, AMD Zen
1 titan wins, newer-Intel fast-ionos is flat, Zen 5 zeus loses) rules
out a simple vendor split. Both 8C/16T hosts (hercules + zeus) sit on
opposite sides of the result, so core count isn't it either.

## Why local measurement was misleading

Dev measurement on Adam's box (Atlas): plain ~370K NPS, PGO ~396K
NPS → +9% bench NPS. Standard rule (~1.4 Elo per 1% NPS) predicted
~+10 Elo at T=1.

What the fleet actually showed:
- Average NPS gain across the fleet: **+0.76%** (not +9%).
- Average Elo gain: **+0.30** (not +10).
- Linear fit: Elo ≈ 0.67 × NPS%, well below the 1.4 rule.
- Variance per machine is enormous: ±15-20% NPS, ±10-20 Elo.

**The +9% NPS on Atlas was specific to Atlas's microarchitecture
and to the bench-13 workload.** It did not generalize.

## Probable mechanism

Three compounding effects, roughly tracking hardware age:

1. **Static layout helps simpler pipelines more.** Older CPUs benefit
   more from PGO's explicit code layout / branch hints because their
   front-end + branch predictor + prefetcher do less of that work
   themselves. Newer CPUs increasingly do this work dynamically and
   well, leaving PGO with smaller wins.

2. **Per-uarch LLVM cost-model maturity.** PGO sharpens whatever the
   cost model already says. On mature targets (anything 5+ years
   old: Coffee Lake Xeon, Zen 1), the cost model is right and PGO
   sharpens good decisions. On a brand-new target (Zen 5, 2024),
   the cost model is still converging — PGO can confidently sharpen
   the *wrong* decisions.

3. **Narrow profile.** The PGO profile comes from `./coda bench 13`
   — 48 positions, fixed depth, single-threaded. Hot-path
   distribution is much narrower than real game-TC search. This
   amplifies (1) and (2) — narrow specialization is most likely
   to actively mispredict when the cost model is also weak.

The titan-vs-zeus contrast localizes the dominant effect to (1)+(2):
both AMD, both ran the identical bench-13 profile, identical
pipeline. Same vendor → very different result, just on a much
newer uarch.

## What we're NOT going to do

- **Ship `make` → PGO** on main. The 3× build slowdown isn't justified
  by neutral-to-negative production Elo.
- **Ship via `make openbench`** while leaving local `make` plain. Same
  Elo problem; the build target plumbing isn't the bottleneck.

## What's worth trying later (lower priority)

- **Richer PGO profile.** Replace `coda bench 13` in the instrument
  step with a short game-TC self-play burst (e.g. 50 games at 10+0.1
  vs a fixed opponent or self). Profile would reflect actual search
  decisions at deployment TC, not narrow bench paths.
- **AutoFDO** (sampling-based PGO via `perf record`). Doesn't
  instrument-then-rerun, so the profile reflects actual uncounted
  execution. Was mentioned as the v9-era alternative in the prior
  Makefile comment.
- **Investigate the hercules vs zeus split.** Same 8C/16T nominal
  class, very different result (+8.7 vs -18.5 Elo). Specifically:
  is the Zen 5 regression an LLVM toolchain maturity issue (would
  improve with a future rustc upgrade), or a fundamental incompatibility
  between bench-13 profile decisions and Zen 5's pipeline? Disassembling
  hot functions in both PGO binaries and comparing layout would
  localize it.

None of these are queued as active work — listed for completeness.

## Branch disposition

`experiment/pgo-openbench-default` is parked, not merged, not
deleted. Contains the working `make = pgo` Makefile recipe if we
revisit PGO with a different profile strategy. Workers retain
`cargo-pgo` + `llvm-tools-preview` from the fleet install loop,
so a future PGO attempt won't need provisioning again.

## Cross-reference

Updates the "verify PGO with cgu=16" followup from
`smp_scaling_investigation_2026-05-17.md`. Answer: PGO works
(no longer crashes / regresses to -10% as v9-era did with cgu=1),
but the cgu=16 fix doesn't deliver the predicted +5-10 Elo win
because the bench profile generalizes poorly. The SMP investigation
itself stands — cgu=16 is still load-bearing for the T=1 SMP bundle.
