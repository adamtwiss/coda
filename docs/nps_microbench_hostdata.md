# NPS Microbench — Per-Host Results

Cross-host data points for the Coda vs Reckless NPS investigation
(`docs/coda_vs_reckless_nps_2026-04-23.md`). Each host appends its
block below. Aim is to answer:

1. Is the 2× NPS gap uArch-specific or structural?
2. Does the gap scale with cache size (matrix spills out of L3)?
3. Which levers (training shrink, AccEntry flatten, hot-feature
   frontloading) deliver the biggest wins on which hardware class?

## How to contribute a host row

```bash
# On a fresh host (one-time setup, ~2 min):
cd ~/code/coda   # or git clone git@github.com:adamtwiss/coda.git ~/code/coda
git checkout feature/threat-inputs && git pull
./scripts/nps_microbench_setup.sh     # clones Reckless, patches, fetches v9 net

# Stop OB worker for clean CPU (optional but recommended):
~/code/OpenBench/ob-worker.sh stop

# Run the matrix (~2-4 min):
./scripts/nps_microbench.sh

# Restart OB worker:
~/code/OpenBench/ob-worker.sh start

# Append results to this doc:
./scripts/nps_microbench_append.sh    # formats + appends a section below
# (or manually paste the `CSV written to:` file's summary)

# Commit the doc update + CSV:
git add docs/nps_microbench_hostdata.md nps_microbench_*.csv
git commit -m "nps_microbench: <host> data point"
git push
```

The script refuses if `./scripts/nps_microbench.sh` hasn't been run
in the current directory yet (looks for `nps_microbench_${HOST}_*.csv`).

---

## Zeus — 2026-04-23

**CPU**: Ryzen 7 9700X (Zen 5, 8c/16t, ISA: AVX-512 + VNNI)
**L3**: 32 MB

| Mode | Coda native | Coda AVX-2 | Reckless native | Reckless AVX-2 | R/C native | R/C AVX-2 |
|---|---:|---:|---:|---:|---:|---:|
| refresh / fresh | 1,330,926 | 1,129,948 | 1,478,598 | 1,268,017 | 1.11× | 1.12× |
| incremental | 3,759,675 | 2,669,350 | 6,112,511 | 5,882,765 | 1.62× | **2.20×** |
| make-unmake (threats on) | — | — | 19,623,393 | 21,818,698 | — | — |

**Key numbers**:
- Coda L1-dcache miss rate on incremental eval: **15.72%**
- Reckless L1-dcache miss rate on incremental eval: **0.55%**
- Coda make-unmake (threats on): 22,912,303/s (faster than Reckless here)

**Source**: `docs/coda_vs_reckless_nps_2026-04-23.md` + Zeus's own run.

---

## Hercules — 2026-04-23

**CPU**: Intel Xeon E-2288G @ 3.70 GHz (Coffee Lake Refresh, 8c/16t,
ISA: AVX-2 only)
**L3**: 16 MB

| Mode | Coda native | Reckless native | Ratio (R/C) |
|---|---:|---:|---:|
| fresh (scalar debug) | 27,862 | — | — |
| refresh / fresh (SIMD prod) | 499,218 | 585,901 | 1.17× |
| incremental | 958,680 | 2,935,135 | **3.06×** |
| make-unmake (observer on) | 11,924,398 | 11,820,114 | 0.99× |
| make-unmake-null (no observer) | — | 20,879,518 | — |

**Search bench (depth 13, trunk)**:
- Total nodes: 2,267,989
- Full rebuilds: 679,576 (44.55%) / Incremental: 846,010 (55.45%)
- TT static-eval hits: 102,802 (6.31%)
- Evals / node: 0.673
- Search NPS: 511,257

**Observations**:
- Incremental gap is **3.06× vs Zeus's 2.20×** (both on AVX-2) → gap
  widens on smaller L3, consistent with the 49 MB threat matrix
  spilling out of L3.
- Coda make-unmake is essentially tied with Reckless's threats-on
  version (11.9M vs 11.8M) → byteboard-splat deprioritisation
  confirmed on Intel too.

**Raw CSV**: `nps_microbench_hercules_2026-04-23.csv`

---

## Titan — 2026-04-23

**CPU**: AMD EPYC 7351P 16-Core (Zen 1 / Naples, ISA: AVX-2 only)
**L3**: 64 MB total (16 MB per CCX, single-thread sees ~16 MB close)

| Mode | Coda native | Reckless native | Ratio (R/C) |
|---|---:|---:|---:|
| fresh (scalar) | 27,938 | — | — |
| refresh / fresh (SIMD) | 272,626 | 242,538 | **0.89× (Coda faster)** |
| incremental | 552,429 | 999,573 | **1.81×** |
| make-unmake (observer on) | 6,538,794 | 3,133,312 | **0.48× (Coda 2× faster)** |
| make-unmake-null (no observer) | — | 3,926,200 | — |

**Observations**:
- Incremental ratio **1.81×** — between Zeus native (1.66×) and Zeus
  AVX-2 forced (2.20×), better than Hercules Coffee Lake (3.06×).
  The "smaller cache = worse ratio" thesis doesn't track linearly
  across uArchs — Zen's CCX-local memory hierarchy helps here vs
  Intel's monolithic L3.
- First host where **Coda's SIMD refresh beats Reckless's fresh**
  (272K vs 243K, R/C=0.89). Not sure why yet — possibly Reckless's
  refresh path has worse codegen on Zen 1's older decoder.
- **Coda make-unmake 2× faster** than Reckless here (6.5M vs 3.1M).
  Byteboard-splat deprioritisation strengthens across hosts: Coda
  1.17× faster on Zeus, tied on Hercules, 2× faster on Titan.

**Submitted by**: Titan (separate Claude instance).

---

<!-- Append new host blocks above this line. Keep the format consistent
     so a cross-host summary table can be regenerated mechanically. -->
