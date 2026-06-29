# Eval-blindspot training-data generation (the 150/80 harvest)

How we mine T80 binpacks for the positions Coda's NNUE values *worst*, to build
a corrective fine-tune set. First run: Jan–Jun 2024 (6 monthly files, ~318M
coda-worse positions in `/training/blindspot/`). This doc is the reproducible
recipe so the harvest can be extended to the remaining T80 datasets.

## The idea

Coda and Stockfish both train on identical T80/LC0 data, and the binpack
carries LC0's MCTS score (800 nodes, ~depth-20-equivalent) as a shared
ground-truth oracle. A *blindspot* is a **quiet** position where:

1. **Coda's static eval is far from the LC0 oracle** — `|coda − lc0| ≥ 150cp`
   (calibrated). This is the "150".
2. **SF's static eval is materially closer** — `|coda − lc0| − |sf − lc0| ≥ 80cp`
   (calibrated). This is the "80".

Condition (2) is load-bearing: SF static (no search, same training data) getting
it right rules out "only deep search resolves this" (a tactic) and isolates a
**Coda-specific learned eval error** — a pattern we mis-weighted that SF did
not. The corrective label written out is the **LC0 score** (STM-POV), so a
fine-tune over-samples these positions toward ground truth.

Quiet-only (`--quiet-only`), in-check skipped, and an oracle band
(`--max-abs-lc0 600`) keep the set aligned with how NNUE is consumed (quiet QS
leaves) and concentrated in the balanced/decidable range rather than already-won
blowouts.

## Pipeline (per monthly binpack)

Driver: `scripts/blindspot_run_all.sh` (32-way across cores; intermediates
deleted after each file so peak disk ≈ one file's scratch, ~40GB).

**Stage 1 — Coda harvest (eval-dist, sharded).** `N=32` copies of
`coda eval-dist` with the streaming-shard flags added to `main.rs`:

```
coda eval-dist --input <file>.binpack -n <net> \
  --shard k/32 --quiet-only --min-error 150 --max-abs-lc0 600 \
  --csv shard_k.csv
```

- `--shard k/N` — process only entries where `global_index % N == k`. Union of
  all N = the whole file. Implies scan-to-EOF + streaming CSV (no in-memory
  distribution stats, so it survives billion-row files). The binpack is
  chain-compressed (sequential, no seek) so every copy decompresses everything,
  but the file caches in RAM once and decompression is cheap next to the NNUE
  eval.
- `--min-error 150` — emit only rows where `|coda_eval − lc0| ≥ 150cp`
  (post-NNUE-eval gate; the cheap pre-filter for the harvest).
- `--max-abs-lc0 600` — emit only rows where `|lc0| ≤ 600cp` (pre-eval band
  reject; also saves compute).

CSV columns: `fen, white_result, coda_eval_white_cp, lc0_score_white_cp`.

**Stage 2 — SF static eval (32-way).** For each shard, build an `id<TAB>fen`
input and run SF's `evalfile` batch command (raw static NNUE eval, white-POV —
NOT a search, which would defeat the not-a-tactic logic):

```
stockfish evalfile sf_in_k.txt sf_out_k.txt   # patched evalfile build
```

**Stage 3 — calibrate + filter (32-way).** `scripts/blindspot_filter_shard.py`
joins the Coda CSV + SF output row-for-row, maps both engines onto the LC0 scale
with a per-engine linear fit `lc0 ≈ a·eval + b` (a pure scale offset must not
masquerade as error), then keeps rows where calibrated `|coda − lc0| ≥ 150`
**and** `(coda_err − sf_err) ≥ 80`. Writes `fen<TAB>lc0_stm_cp` — the corrective
label.

The calibration constants live in `calib.txt` (4 floats: `coda_a coda_b sf_a
sf_b`), fit once on a sampled subset via `scripts/sf_static_eval.py`'s
least-squares step and **reused across all months** (the engines' scales don't
drift between files). Current fit: coda `a=0.781 b=3.286`, sf `a=2.730
b=3.013`.

**Stage 4 — import to binpack.**

```
coda import-tsv -i import_M.tsv --fen-col 0 --score-col 1 \
  -o /training/blindspot/t80_blindspot_150_80_<mon>_<year>.binpack
```

Output names carry the **month _and_ year** (`..._may_2024.binpack`) — the T80
set has several Mays/Aprils across years, so bare-month names would collide.
The driver's `FILE` map is keyed `<mon>_<year>` and the output/candidate names
derive from that key.

Per-month candidate TSVs are kept gzipped (`cands_<mon>_<year>.tsv.gz`) for
provenance; heavy scratch (shard CSVs, SF in/out) is deleted.

## Supporting scripts

- `scripts/blindspot_run_all.sh` — the end-to-end driver (the exact one used for
  Feb–Jun). Idempotent (skips a month whose output binpack exists) and
  continue-on-error.
- `scripts/blindspot_filter_shard.py` — stage-3 per-shard calibrate+filter.
- `scripts/sf_static_eval.py` — stage-2 analysis tool + the calibration fit
  (where `calib.txt` comes from).
- `scripts/t80_misrate_scan.py` — stage-1 SF-free error-magnitude scan /
  candidate emitter (billion-scalable; the analysis precursor to the sharded
  eval-dist harvest).

## Yield (first run, 150/80, Jan–Jun 2024)

| Month | coda-worse positions |
|---|---|
| Jan | ~50M |
| Feb | 41.8M |
| Mar | 52.5M |
| Apr | 73.8M |
| May | 53.0M |
| Jun | 47.4M |
| **Total** | **~318M** |

## Scaling note

~318M positions is a solid first corrective set, but likely **undersized** — it
may want to be several× larger. The 150/80 filter is deliberately strict
(~15–25% of the 150-prefiltered survivors pass the SF-worse-by-80 gate), so the
limiting factor is **how many T80 monthly files we feed it, not the filter**. We
have substantially more T80 data available (more 2024 months on `/training/sf`,
and the full SF training set on GPU4) — extending the harvest to those files is
the straightforward way to grow this 5× without changing the recipe. The driver
is parameterized by the `FILE` map at the top of `blindspot_run_all.sh`; add
files there and re-run.

When training a corrective net on this set, **SPRT against the current prod net**
(as of this writing `E6C62000`, v4-swa), not the net the harvest was generated
against — the labels are LC0 ground truth and net-agnostic, but the baseline to
beat moves with prod.
