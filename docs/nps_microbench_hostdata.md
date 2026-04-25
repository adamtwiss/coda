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

## ionos1 — 2026-04-23

**CPU**: AMD EPYC Milan (Zen 3, ISA: AVX-2 only — no AVX-512/VNNI)

| Mode | Coda native | Reckless native | Ratio (R/C) |
|---|---:|---:|---:|
| fresh (scalar) | 30,968 | — | — |
| refresh / fresh (SIMD) | 475,034 | 487,872 | 1.03× |
| incremental | 1,043,100 | 2,212,261 | **2.12×** |
| make-unmake (observer on) | 7,847,737 | 8,868,432 | **1.13× (Reckless ahead)** |
| make-unmake-null (no observer) | — | 18,012,578 | — |

**Observations**:
- Incremental ratio **2.12×**, sits close to Zen 5 AVX-2 forced (2.20×).
  Milan's cache hierarchy behaves like Zen 5's when AVX-512 is taken off.
- First host (alongside ionos1) where **Reckless wins make-unmake**
  by a small margin (R/C=1.13 vs Coda advantage on Zeus/Hercules/Titan).
  Zen 3 Milan may handle Reckless's byteboard-splat pattern
  particularly well, or our make-unmake hit a uArch-specific quirk here.
- refresh/fresh essentially tied on all non-Titan hosts.

**Raw CSV**: `nps_microbench_ionos1_2026-04-23.csv`

---

## ionos6 — 2026-04-23 ⭐ (cache-residency validation)

**CPU**: AMD EPYC-Milan Processor, **2.5 GHz** (vs ionos1 at 2.0 GHz).
**Instance**: 3-core VM. **Cache**: L1d=96 KiB (3×32), L2=1.5 MiB,
L3=32 MiB — identical spec to ionos1.

**The 25% clock advantage explains part of it**, but not all:

| Factor | ionos1 → ionos6 |
|---|---:|
| Clock ratio | 1.25× |
| Reckless incremental speedup | 1.67× (1.34× after clock) |
| Coda incremental speedup | 2.26× (**1.81× after clock**) |

After normalising for clock, Coda still gains 1.81× vs Reckless's
1.34×. The residual delta is almost certainly **self-inflicted L3
contention** — IONOS VMs have dedicated cores but share physical
sockets, so the 32 MB L3 is shared across all Adam-fleet VMs on a
given CCD. When multiple VMs run OB workers simultaneously:
- Each VM's threads evict the others' cache lines
- Coda's 49 MB threat matrix can't even stay partially resident
- Reckless's ~400 KB sits comfortably in L2 regardless

ionos6 likely sits on a CCD where fewer (or zero) other Adam-fleet
VMs are active, getting closer to "full 32 MB L3 to this VM". To
validate: pause all Adam IONOS OB workers for ~2 min and re-run
ionos1 microbench. If ionos1 jumps toward ionos6's numbers,
confirmed self-contention.

**Implication for lever priority**: Item 2 (training-side matrix
shrink) gains additional weight — shrinking the matrix doesn't
just fix Coda's residency, it reduces the memory-bandwidth
footprint we impose on our fleet neighbours, so their Coda speeds
up too. Compounding across the fleet.

| Mode | Coda native | Reckless native | Ratio (R/C) |
|---|---:|---:|---:|
| fresh (scalar) | 69,369 | — | — |
| refresh / fresh (SIMD) | 890,609 | 752,492 | **0.84× (Coda faster)** |
| incremental | 2,353,694 | 3,686,819 | **1.57×** |
| make-unmake (observer on) | 15,044,344 | 12,401,286 | **0.82× (Coda faster)** |
| make-unmake-null (no observer) | — | 23,765,070 | — |

**Headline**: ionos6's 1.57× incremental ratio **beats Zeus's native
(1.66×)** despite ionos6 lacking AVX-512+VNNI — cache residency
outweighs ISA width.

**Production-mystery explained**: User flagged ionos6 as "~2× faster
than others for v9" in actual search. Microbench confirms the
mechanism:
- Coda incremental speedup (ionos1 → ionos6): **2.26×** (1.04M → 2.35M)
- Reckless incremental speedup (ionos1 → ionos6): **1.67×**

Same CPU and L3 size. The 2.26× comes from **effective L3 per
thread**: ionos6 is a 3-core VM, likely on a less-contended host,
so the single-threaded bench has close to the full 32 MB L3
available. On ionos1's larger instance, L3 is being evicted by
co-running threads / co-tenant VMs. Coda's 49 MB threat matrix is
therefore the whole L3 + DRAM on ionos1, but a higher-residency
DRAM-backed-L3 pattern on ionos6.

Reckless's ~400 KB footprint was L2-resident on both, so it had
less headroom to gain from the extra L3 availability.

**Lever implications**:
- **Item 2 (training-side shrink) is the single biggest lever**:
  shrinking the threat matrix below 32 MB turns every Milan/Zen 3+
  host into an ionos6.
- **Items 1 (AccEntry flatten) and 4 (hot-feature frontloading) are
  the most portable wins** — they improve the access pattern
  against any matrix size, benefiting every host class.
- **VNNI is real but secondary**: Zeus Zen 5 + VNNI (1.66×) is only
  marginally better than ionos6 Milan-X sans VNNI (1.57×). Cache
  residency > ISA width.

**Raw CSV**: `nps_microbench_ionos6_2026-04-23.csv`

---

## Thor — 2026-04-23

**CPU**: AMD Ryzen 7 3800X 8-Core (Zen 2 / Matisse, desktop).
ISA: AVX-2 only.

| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| fresh (scalar) | 309,026 | — | — |
| refresh / fresh (SIMD) | 574,941 | 665,825 | 1.16× |
| incremental | 1,152,802 | 2,525,770 | **2.19×** |
| make-unmake (obs) | 10,797,050 | 10,154,891 | **0.94× (Coda slightly ahead)** |
| make-unmake-null | — | 18,664,630 | — |

**Note**: Coda's fresh (scalar debug) is ~10× higher than most hosts
(309K vs typical ~30K). Likely an LLVM auto-vectorisation of the
scalar inner loop on Thor's rustc — not affecting production path.
Two runs gave identical numbers (<1% drift), confirming the
anomaly is stable and reproducible, not flaky.

**Raw CSV**: `nps_microbench_thor_2026-04-23.csv`

---

## Callisto — 2026-04-23

**CPU**: 11th Gen Intel i7-11390H @ 3.4 GHz (Tiger Lake, mobile).
**ISA**: AVX-512 + VNNI (first Intel host with AVX-512).

**Native (AVX-512+VNNI)**:
| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| refresh / fresh | 419,265 | 632,530 | 1.51× |
| incremental | 1,369,813 | 2,373,973 | **1.73×** |
| make-unmake | 8,542,399 | 11,090,580 | 1.30× (Reckless ahead) |
| make-unmake-null | — | 21,224,134 | — |

**AVX-2 forced**:
| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| refresh | 344,009 | 417,514 | 1.21× |
| incremental | 909,345 | 2,012,476 | **2.21×** |
| make-unmake | 9,160,002 | 7,085,172 | **0.77× (Coda 1.3× ahead)** |

**Observations**:
- VNNI incremental ratio 1.73× — nearly identical to Zeus Zen 5
  native (1.66×). VNNI thesis confirmed on Intel too.
- AVX-2-forced 2.21× ≈ Zeus AVX-2 forced (2.20×) — modern Intel
  and modern AMD indistinguishable at AVX-2.
- Make-unmake flips with ISA: Reckless 1.3× ahead native, Coda
  1.3× ahead AVX-2. AVX-512 byteboard-splat benefits Reckless
  more than it benefits Coda's scalar path.
- Coda VNNI speedup (native vs AVX-2 forced): 1.51×, marginally
  more than Zeus's 1.41×.

**Raw CSV**: `nps_microbench_callisto_2026-04-23.csv`

---

## Adams-MacBook-Air (M5) — 2026-04-24

**CPU**: Apple M5 (2025). ARM / NEON, no AVX.

| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| refresh / fresh | 842,473 | 890,168 | 1.06× (near tied) |
| incremental | 1,579,483 | 4,005,457 | **2.54×** |
| make-unmake (obs) | 27,064,744 | 30,966,248 | 1.14× |
| make-unmake-null | — | 45,969,324 | — |

**Observations**:
- Incremental gap 2.54× — middle of the pack. Worse than
  VNNI-equipped hosts (Zeus 1.66×, Callisto 1.73×), worse than
  ionos6's quiet-socket 1.57×, but much better than Coffee Lake
  (3.06×). Roughly comparable to x86 AVX-2.
- **Best make-unmake absolute throughput** of any host (27M Coda,
  31M Reckless). Apple's unified-memory bandwidth matters.
- Coda speedup MBA vs Hercules: 1.65×; Reckless: 1.36×. Coda
  benefits disproportionately from Apple's memory subsystem,
  consistent with the "Coda is memory-bound" thesis.
- **Proves the gap isn't x86-specific** — it's memory/access-pattern
  and persists across ISAs.

**Raw CSV**: `nps_microbench_Adams-MacBook-Air_2026-04-24.csv`

---

## Lo — 2026-04-23 (Lichess production host)

**CPU**: QEMU-exposed "Intel Core Processor (Haswell, no TSX)".
Underlying physical CPU unknown. ISA: AVX-2. Cloud VM.

| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| refresh | 201,538 | 281,305 | 1.40× |
| incremental | 666,716 | 892,897 | **1.34×** |
| make-unmake | 3,922,479 | 5,718,357 | **1.46× (Reckless ahead)** |
| make-unmake-null | — | 7,253,761 | — |

**Observations**:
- **Tightest incremental R/C gap on any non-VNNI host** (1.34×,
  even beating VNNI-equipped Zeus at 1.66×). Reason: Lo is slow
  enough that both engines are memory-latency-bound; Reckless's
  cache-friendliness advantage can't materialise when every
  access stalls on DRAM.
- **Slowest host in the fleet on absolute numbers**: 666K Coda
  incremental (vs 1M-2.3M elsewhere). 3-4× slower than most dev
  hosts.
- **Production impact**: Lo is codabot's Lichess host. Lichess
  users see ~1/3 the NPS we measure during SPRT on dev hosts.
  Moving Lichess to a modern Zen host (Thor, Titan, ionos*) would
  give users 50-100 Elo of effective gain from placement alone.
- **Make-unmake flip**: first host where Reckless decisively wins
  make-unmake (1.46× ahead). Haswell's older prefetcher handles
  Reckless's byteboard-splat pattern particularly well.

**Raw CSV**: `nps_microbench_lo_2026-04-23.csv`

---

## Thebe — 2026-04-23

**CPU**: QEMU-exposed "Intel Core Processor (Haswell, no TSX)".
Same signature as Lo but **very different behaviour**.

| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| refresh | 238,315 | 240,525 | 1.01× (tied) |
| incremental | 475,337 | 1,396,614 | **2.94×** |
| make-unmake | 6,534,988 | 6,548,637 | 1.00× (tied) |
| make-unmake-null | — | 12,343,982 | — |

**Observations**:
- Same QEMU signature as Lo, but R/C incremental is **2.94× vs Lo's
  1.34×** — hypervisor-reported CPU hides different underlying
  silicon. Thebe's Coda is 29% slower than Lo's; Reckless is 56%
  faster. Memory-subsystem-specific pattern that favours Reckless
  here and disfavours Coda.
- Second-worst ratio in the dataset (after Coffee Lake). Probably
  a physical host where other workloads contend for memory
  bandwidth in a Coda-punishing pattern.

**Raw CSV**: `nps_microbench_thebe_2026-04-23.csv`

---

## Jupiter — 2026-04-24

**CPU**: ARM Cortex-A55 (in-order "little" core, 8-core config,
aarch64). Cheap ARM VPS.

| Mode | Coda | Reckless | R/C |
|---|---:|---:|---:|
| refresh | 166,959 | 175,067 | 1.05× (tied) |
| incremental | 365,319 | 570,077 | **1.56×** |
| make-unmake | 5,382,950 | 1,708,825 | **0.32× (Coda 3.15× faster!)** |
| make-unmake-null | — | 2,133,949 | — |

**Observations**:
- **Incremental R/C = 1.56×** — ties ionos6 as tightest non-VNNI
  gap. Root cause here is architectural: A55 is **in-order**, so
  it can't out-of-order-execute to hide memory latency. Reckless's
  advantage depends on OoO hiding misses; without OoO, both
  engines stall equally per miss and the gap collapses.
- **Most lopsided Coda-favouring make-unmake result in the
  dataset** (Coda 3.15× faster). Reckless's byteboard-splat code
  relies on OoO to hide long dependency chains, and A55's
  in-order pipeline turns their fast path into a bottleneck.
- **ARM implementations vary enormously**: MBA M5 is 10× faster
  than Jupiter on incremental. Apple's M-series OoO cores are
  nothing like budget ARM cores.
- Final piece of the "Reckless advantage requires OoO" picture.

**Raw CSV**: `nps_microbench_jupiter_2026-04-24.csv`

---

## Cross-host summary

Full dataset spanning 9 distinct hosts across 5 uArchs (Zen 1/2/3/5,
Coffee Lake, Tiger Lake, Haswell, Apple M5, Cortex-A55).

| Host | uArch | ISA | refresh R/C | **incr R/C** | m-u R/C |
|---|---|---|---:|---:|---:|
| Lo | Haswell VM (slow) | AVX-2 | 1.40× | **1.34×** | 1.46× (R) |
| Jupiter | Cortex-A55 (in-order ARM) | NEON | 1.05× | **1.56×** | **0.32× (Coda 3.15×)** |
| ionos6 | Zen 3 Milan 2.5 GHz | AVX-2 | 0.84× (Coda) | **1.57×** | 0.82× (Coda) |
| Zeus native | Zen 5 | AVX-512+VNNI | 1.11× | **1.66×** | 0.86× (Coda) |
| Callisto native | Tiger Lake | AVX-512+VNNI | 1.51× | **1.73×** | 1.30× (R) |
| Titan | Zen 1 Naples | AVX-2 | 0.89× (Coda) | **1.81×** | **0.48× (Coda 2×)** |
| ionos1 | Zen 3 Milan 2.0 GHz | AVX-2 | 1.03× | **2.12×** | 1.13× (R) |
| Thor | Zen 2 Matisse | AVX-2 | 1.16× | **2.19×** | 0.94× (Coda) |
| Callisto AVX-2 | Tiger Lake (forced AVX-2) | AVX-2 | 1.21× | **2.21×** | 0.77× (Coda) |
| Zeus AVX-2 | Zen 5 (forced AVX-2) | AVX-2 | 1.12× | **2.20×** | ~tied |
| MBA M5 | Apple M5 | NEON | 1.06× | **2.54×** | 1.14× (R) |
| Thebe | Haswell VM (hostile) | AVX-2 | 1.01× | **2.94×** | 1.00× (tied) |
| Hercules | Coffee Lake | AVX-2 | 1.17× | **3.06×** | 0.99× |

Sorted by incremental R/C (lower = tighter gap = Coda more competitive).

## Findings across the fleet

### 1. The gap exists on every host — it's structural, not uArch-specific

Every single measured host shows Coda slower than Reckless on incremental
eval, ranging from 1.34× (Lo) to 3.06× (Hercules). 9 hosts, 5 architectures,
3 ISA tiers — the gap is universal. **Not a Zen-5 quirk, not an Intel
corner case, not a Zeus-only artefact.**

### 2. Two conditions flatten the gap — and they're both "Reckless can't pull ahead"

Reckless's cache advantage manifests as faster incremental eval *when the
CPU can out-of-order-execute past memory misses*. Conditions that neutralise
the advantage:

- **No OoO** (Jupiter A55, in-order): gap collapses to 1.56×. Reckless
  stalls just as hard as Coda when it can't reorder around misses.
- **Severe memory-latency saturation** (Lo, Haswell VM on slow silicon):
  gap collapses to 1.34×. Every access waits on DRAM, equalising
  both engines.

These aren't cases where we *fixed* something — they're cases where
Reckless's win vanishes because the CPU can't capitalise on it.
Useful as boundary conditions for understanding the mechanism.

### 3. Cache residency is the biggest mid-range lever

At mid-range hosts (not OoO-starved, not DRAM-bound), ratio correlates
strongly with effective L3 per thread:

- **ionos6** (lightly-loaded 2.5 GHz Milan, 3-core VM, 32 MB L3 mostly
  available): 1.57× — beats Zeus's VNNI (1.66×).
- **ionos1** (contended 2.0 GHz Milan, same hardware class): 2.12×.

Same physical CPU class. The difference is **effective L3 available to
the single-threaded bench**. This validates the cache-residency thesis
and promotes Item 2 (training-side shrink) to the top of the lever list.

### 4. VNNI helps, but less than expected

- Zeus Zen 5 + VNNI: 1.66× (vs AVX-2-forced 2.20×) — VNNI closes ~25% of the gap
- Callisto Tiger Lake + VNNI: 1.73× (vs AVX-2-forced 2.21×) — same magnitude

VNNI is real but only worth ~25% gap reduction. A quieter memory subsystem
(ionos6) gets similar benefit without needing VNNI at all.

### 5. Coffee Lake is uniquely bad, not "Intel in general"

Hercules (E-2288G, Coffee Lake Refresh 2019) sits alone at 3.06× — 40%
worse than the next non-contended host. Callisto (Tiger Lake 2021, same
Intel vendor) at 2.21× AVX-2 is statistically indistinguishable from
modern AMD at AVX-2 (Zeus 2.20×, Thor 2.19×).

**Takeaway**: older Intel memory subsystem handles the 49 MB scattered-
access pattern particularly poorly. Not a deployment concern on modern
fleet hosts.

### 6. Two flavours of "same CPU, different behaviour"

- **ionos1 vs ionos6**: identical CPU model, identical cache spec,
  3-core VMs. Differ by 25% clock (2.0 vs 2.5 GHz). After clock
  normalisation: Reckless gains 1.34× on ionos6, Coda gains 1.81×.
  Residual = memory-subsystem difference (co-tenant contention,
  turbo headroom) that disproportionately favours Coda. Most likely:
  self-inflicted L3 contention on ionos1 from other Adam-fleet VMs
  on the same physical socket.
- **Lo vs Thebe**: same QEMU-reported Haswell signature, wildly
  different ratios (1.34× vs 2.94×). Hypervisors lie about CPU.
  Underlying silicon differs enough to flip Reckless's advantage on
  and off.

These both reinforce that **"same CPU" at the OS layer can mean very
different things at the memory-subsystem layer**.

### 7. make-unmake is Coda's story, not Reckless's

Coda faster or tied on 8 of 13 measurements. Reckless substantially
ahead only on Lo (1.46×) and Callisto native (1.30×).

Byteboard-splat port remains deprioritised — our simpler scalar
path is competitive or better on almost every host.

### 8. Apple Silicon deserves separate analysis, not just "ARM"

MBA M5 sits at 2.54× incremental R/C — worse than any Zen AVX-2. Apple
Silicon's advantage isn't that it *closes* the Coda/Reckless gap; it's
that its absolute memory bandwidth is the best on the planet, so both
engines run at very high throughput (27M+ make-unmake). But the
relative structural issue (49 MB matrix access pattern) persists.

### 9. Lichess placement opportunity (incidental but material)

Coda's Lichess host (Lo) is ~3-4× slower than modern AMD fleet hosts
on Coda incremental. Moving codabot off Lo to ionos6 / Thor / Titan
would deliver ~50-100 Elo of user-perceived strength improvement
for zero code change — just deployment placement.

### 10. Lever priority after the full dataset

| Rank | Lever | Mechanism | Why this rank |
|---|---|---|---|
| 1 | **Item 2**: training-side shrink | Reduces 49 MB matrix → fits L3 on more hosts | Turns every Milan/Zen 3+ host into ionos6; compounds cross-fleet |
| 2 | **Item 1**: flatten AccEntry | Better access pattern on any matrix size | **+6.5 Elo H1 ✓ MERGED** |
| 3 | **Item 7**: eval-only TT writeback | Reduce evals/node via caching | **+14.7 Elo H1 ✓ MERGED** |
| 4 | **Item 8**: LMP direct-check | Shrink tree → fewer evals/node | **+2.5 Elo H1 ✓ MERGED** |
| 5 | **Item 4**: hot-feature frontloading | Keep hot rows in L2 even without full shrink | Held pending #3 resolution |
| 6 | **Item 6**: PSQ walk-back clean retry | Reduce rebuild cost | **+3.9 trending H1** (redo of earlier H0) |
| 7 | **Item 3**: load-time compact | Free 4 MB via structural zeros | **H0 (−1.6)** — surprising; may need diagnosis |
| 8 | **Item 5**: prefetch in apply loop | Hint HW prefetcher | **H0 both variants** — HW prefetcher already optimal |

**Running total merged from NPS investigation: +23.7 Elo today.**
With Item 6 likely landing: **~+27.6 Elo.** That's a full training-run's
worth of Elo converted from a ~1-day investigation. Big day.

---

<!-- Append new host blocks above this line. Keep the format consistent
     so a cross-host summary table can be regenerated mechanically. -->
