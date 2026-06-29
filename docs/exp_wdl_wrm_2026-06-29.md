# Experiment log — wdl24 × wrm-sf 2×2 (S200, current prod-shape) — 2026-06-29

**Question:** do `wdl 0.24` and the SF win-rate-model (WRM) loss stack, or
fight? Both have prior S200-positive evidence individually (wdl24 ~+2; wrm
+2 untuned / +3 tuned, tune not load-bearing) but were never tested
together, nor under the *current* code/recipe. They are the only pair that
can plausibly interact (both reshape the WDL-vs-cp trust tradeoff).

**Not in this probe:** the 4 new older-LC0 files (disk space; orthogonal,
established — folded into the eventual long train, not crossed in here).

## Setup

- Host: **GPU2** (RTX 5080), bullet HEAD `c886f36` (WRM commit), `cargo run`
  so the binary rebuilds current.
- Data: `/workspace/data` (standard 11-file T80 pool). Val: nov-2023 holdout.
- Recipe = identical to the GPU4 blindspot ablation so results cross-compare.
  Only `--wdl` and the WRM block differ between cells.

Shared recipe (every cell):
```
cargo run --release --features cuda --example coda_v9_768_threats -- \
  --dataset-dir /workspace/data \
  --superbatches 200 --warmup 20 --final-lr 1e-6 --seed 44 \
  --swa-start-sb 180 --save-rate 200 \
  --fen-skip-prob 0.5 --kb-layout reckless --hidden-activation crelu \
  --factoriser --ft-size 1024 --l1-size 32 --qat \
  --data-order interleave --mse-power 3 \
  --soft-early-ply 28 --soft-early-ply-floor 0.25 \
  --val-data /workspace/all/test80-2023-11-nov-2tb7p.min-v2.v6.binpack \
  <cell flags> --net-id <id>
```
WRM block (SF values) = `--win-rate-model --in-scaling 300 --in-offset 300 --out-scaling 350 --out-offset 300`

## Cells (train order: base → both → wdl24 → wrm)

| order | cell | net-id | cell flags |
|---|---|---|---|
| 1 | base | `ww-base-wdl20` | `--wdl 0.20` |
| 2 | both | `ww-wdl24-wrm-sf` | `--wdl 0.24` + WRM block |
| 3 | +wdl24 | `ww-wdl24` | `--wdl 0.24` |
| 4 | +wrm | `ww-wrm-sf` | `--wdl 0.20` + WRM block |

Rationale for order (Adam): land base+both first and SPRT that pair (the
headline "do they stack" read); the two individuals (prior-tested, but not
under current code) fill in attribution after.

## Read-out

- Net-vs-net SPRT on `main`, each cell's `-swa` net vs `ww-base-wdl20-swa`
  (bench each with its own net), bounds `[-1.5, 1.5]`.
- Cheap proxies: `net_pref_score.py` (overrate / wandering corpora),
  `eval_quality.py` (Spearman vs LC0). No trunk retune (wrm reads untuned).
- Interaction = is `both` ≈ `wdl24 + wrm` (additive → stack) or `<` either
  (fight).

## Results (fill as they resolve)

| cell | bench | SPRT vs base (id) | Elo ±CI | verdict | net_pref | eval_quality |
|---|---|---|---|---|---|---|
| base | — | (baseline) | — | — | | |
| both | — | | | | | |
| +wdl24 | — | | | | | |
| +wrm | — | | | | | |

_status: training launched 2026-06-29 on GPU2; base running._
