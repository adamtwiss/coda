---
name: local-rr
description: Local cutechess-cli round-robin / gauntlet / net-vs-net testing for Coda — adjudication, standard args, and the CPU-contention prerequisite. Use for things OpenBench can't do (net-vs-net of bench-unstable nets, ponder/TM cross-engine, deployment-TC RR, EGTB) or quick local validation.
---

# Local RR / gauntlet testing (cutechess-cli)

Use this for testing OpenBench can't or shouldn't do: net-vs-net where a net's
bench isn't OB-reproducible, ponder-enabled / TM cross-engine work, deployment-TC
round-robins, or a quick local head-to-head. For ordinary "does this change
help?" SPRTs, use **OpenBench** (see the `ob` skill), not local RR.

## When modifying OTHER engines' checkouts (ablations, patches)

Restore the checkout to its starting state when the builds are staged —
**source AND local binary** (Adam, 2026-07-04). `git checkout -- <file>`
cleans the source but leaves `target/release/<engine>` as the PATCHED build;
the RR pool or any rebuild-skipping script then silently runs the modified
engine. Sequence: build stock → copy binary out → patch → build → copy
patched binary out → `git checkout -- .` → **copy the stock binary back over
`target/release/<engine>`** (or rebuild). Verify with `cmp` + `git status`.
Keep experiment binaries in a dedicated dir (e.g. `~/chess/ablation-*/`),
never in the engine's own tree.

## ALWAYS FIRST: kill CPU contention

**Before any local CPU-bound measurement — profiling, NPS, a local bench, an RR,
or a gauntlet — make sure nothing else is using the cores.** Background load
halves effective TC and silently distorts marginal results. This is common sense
but it's the #1 way local numbers go wrong.

**The OB worker runs on every machine, including dev hosts (Hercules/Atlas/Titan)
— and it will contend with your local run.** Stop it for the duration, restart
after:

```bash
~/code/OpenBench/ob-worker.sh stop     # before local measurement (hyphen, not underscore)
# ... run your RR / profile / bench ...
~/code/OpenBench/ob-worker.sh start    # when done
```

Also check for stray `cutechess-cli` / `coda` / build jobs (`htop`, or
`pgrep -a coda`) before starting.

## Standard gauntlet / RR invocation

```bash
cutechess-cli \
  -engine name=dev  cmd=./coda arg=-n arg=nets/<dev-net>.nnue \
  -engine name=base cmd=./coda arg=-n arg=nets/<base-net>.nnue \
  -each proto=uci option.Hash=512 tc=0/10+0.1 \
  -rounds 100 -concurrency 16 \
  -openings file=~/chess/books/noob_4moves.epd format=epd order=random \
  -pgnout gauntlet.pgn -recover -ratinginterval 20 \
  -draw movenumber=20 movecount=10 score=10 \
  -resign movecount=3 score=500 twosided=true
```

Recommended standard arguments (the block above):
- **`option.Hash=512`** — use a healthy hash, especially at LTC (a starved hash
  at long TC distorts pruning economics ~25 Elo). 512 MB per engine; budget RAM
  vs concurrency.
- **`tc=0/10+0.1`** — sudden-death 10s + 0.1s inc shown; bump for LTC
  (`0/40+0.4`, `0/60+1`, etc.). cutechess `tc` = `moves/base+inc` (0 moves =
  whole game).
- **`-rounds 100 -concurrency 16`** — 100 rounds (×2 games with `-repeat`-style
  pairing if added). conc 16 on an 8C/16T host for non-ponder; drop to **8** for
  ponder gauntlets (a pondering engine uses ~1 extra thread on the opponent's
  turn).
- **`-openings noob_4moves.epd order=random`** — standard book.
- **`-pgnout gauntlet.pgn -recover`** — `-recover` restarts a crashed engine so
  one bad game doesn't kill the run.
- **Adjudication** (speeds runs, reduces variance):
  - **`-draw movenumber=20 movecount=10 score=10`** — call a draw once, after
    move 20, 10 consecutive plies stay within ±10 cp.
  - **`-resign movecount=3 score=500 twosided=true`** — resign when 3 consecutive
    plies hit ≥500 cp; `twosided=true` requires BOTH engines to agree on the
    sign (guards against a single-engine eval blip throwing the game).

## Notes
- **Net-vs-net locally is clean even when the net's OB bench is ISA-unstable** —
  both engines run on the same CPU, same SIMD path, so the comparison is fair
  (this is the fallback when an OB net-vs-net Wrong-Benches across the fleet).
- Max **2 Coda variants** in a single RR — more amplifies shared-eval bias
  (Coda-on-Coda contamination).
- TM-class changes need **ponder-enabled** cross-engine RR. Enable it with the
  bare **`ponder`** keyword on each `-engine` line, NOT `option.Ponder=true`:
  `-engine name=dev cmd=./coda ponder`. cutechess reserves "Ponder" as a
  self-managed option, so passing `option.Ponder=true` is *rejected* with
  `Warning: <engine> doesn't have option Ponder` and the match silently runs
  **without pondering** (verified 2026-07-02 — a whole ponder RR was wasted this
  way; always grep the log for that warning and confirm it's absent before
  trusting a ponder run). See CLAUDE.md §TM-class changes.
- Read results from the cutechess stdout score line / the PGN; parse per-move
  spend from PGN comments with `([0-9]+\.[0-9]+)s\b` (the FIRST decimal is the
  score, not the spend).
