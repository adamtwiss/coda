# History pruning + cont-hist diagnostic data (2026-05-08)

Step 0 of the investigation in `history_prune_cont_hist_review_2026-05-08.md`:
add lightweight always-on counters to bench output, capture baseline numbers,
let the data redirect the experiment plan before any SPRT.

The data inverts several of the predictions from the cross-engine review.
Documenting here so subsequent sessions don't repeat the mistake of
reasoning from peer code without ground truth.

## Instrumentation added

`src/search.rs` PruneStats now carries:
- `hist_prune_eligible` — count of moves reaching the gate
- `hist_prune_ratio_buckets[8]` — score / threshold ratio histogram
- `cont_hist_mag_buckets[4][5]` — per-offset {1,2,4,6} read magnitudes
- `cont_hist_writes[4]`, `cont_hist_write_mag_sum[4]` — per-offset write counts + average bonus magnitude
- `main_hist_density[4]` — 4D `[ft][tt]` cell-count with `|val|>1000`
- `main_hist_bucket_reads[4]` — read counts per `[ft][tt]` bucket

All sampled at the hist-prune gate (one sample per gate-eligible move),
zero behaviour change (verified — bench unchanged at 3,796,568).

## Baseline data — current main, depth-12 bench, 49 positions

### History pruning gate

```
History prunes:     3251  (0.1% of eligible 2566399)
Hist-prune score / threshold buckets (sum=eligible):
    >= +1.0      (positive history)            22912 (  0.9%)
    [0.0, +1.0)                              1157611 ( 45.1%)
    [-0.5, 0.0)                              1322528 ( 51.5%)
    [-1.0, -0.5)  (close to gate)              60292 (  2.3%)
    [-1.5, -1.0)  FIRES (just over gate)        2804 (  0.1%)
    [-2.0, -1.5)  FIRES                          238 (  0.0%)
    [-3.0, -2.0)  FIRES (deep)                    14 (  0.0%)
    < -3.0        FIRES (very deep)                0 (  0.0%)
```

**Headline finding: 96.6% of eligible moves are in `[-0.5, +1.0)` of threshold.**
The threshold doesn't intersect the bulk of the score distribution. To fire,
a move's history needs to be deeply negative (worse than `-12825 * depth`),
but most poor moves cluster only slightly below zero, far from the gate.

Reframe: the `depth ≤ 3` gate is NOT the binding constraint — even with
that limit removed, only 3,256 / 2,566,399 = 0.13% would fire (the [-1.5, -∞)
buckets sum to 3,056). The threshold magnitude is the problem.

### Cont-hist read magnitude per offset

```
    offset    [0,200)   [200,1k)    [1k,5k)   [5k,10k)    [10k+)
    ply-1  1999242(79.3%)  228176( 9.1%)  248165( 9.8%)   33834( 1.3%)   10857( 0.4%)
    ply-2  1907284(74.9%)  365657(14.4%)  233175( 9.2%)   32599( 1.3%)    8080( 0.3%)
    ply-4  1537542(63.6%)  483352(20.0%)  343207(14.2%)   44723( 1.8%)    9418( 0.4%)
    ply-6  1203020(55.2%)  509761(23.4%)  404918(18.6%)   50658( 2.3%)   10191( 0.5%)
```

**Headline finding: signal magnitude INVERTS with ply distance.** ply-1 is
mostly noise (79% < 200 magnitude); ply-6 carries the most signal (45%
non-zero, 18.6% in `[1k, 5k)` band).

This contradicts the standard intuition that "closer plies should be more
informative" — which underpins the cross-engine review's recommendation
to weight ply-1 highest. The reverse appears true for Coda.

### Cont-hist write counts and average magnitudes

```
    ply-1  count    255847  avg |bonus|   780.3
    ply-2  count    226180  avg |bonus|   405.2
    ply-4  count    218521  avg |bonus|   394.7
    ply-6  count    198164  avg |bonus|   382.5
```

Write counts within 22% across offsets. Average magnitudes confirm the
asymmetry: ply-1 gets ~2× the bonus magnitude of ply-{2,4,6} (the
`bonus` vs `bonus/2` split in code).

### 4D main-history `[ft][tt]` bucket distribution

```
    [ft=0][tt=0]  reads  1757251 ( 68.5%)   |val|>1000 cells:  6061
    [ft=0][tt=1]  reads   303858 ( 11.8%)   |val|>1000 cells:  2771
    [ft=1][tt=0]  reads   357479 ( 13.9%)   |val|>1000 cells:  3733
    [ft=1][tt=1]  reads   147811 (  5.8%)   |val|>1000 cells:  1879
```

Per-cell sample density (reads / 4096):
- `[0][0]`: 429 reads/cell — well-sampled
- `[0][1]`: 74 reads/cell
- `[1][0]`: 87 reads/cell
- `[1][1]`: 36 reads/cell — 12× sparser than `[0][0]`

**Headline finding: 4D bucketing is highly uneven.** 68.5% of move-ordering
reads go through one of four buckets. Per-cell sample density differs by
12×. The "lost magnitude" Hercules flagged is real — `[1][1]` cells
accumulate signal far more slowly than `[0][0]`.

## What this means for the experiment plan

### Hypotheses INVALIDATED by data

1. **"CONT_HIST_MULT > 1 should help"** — false. ply-1 reads are mostly
   noise (79% < 200). Multiplying by `cm > 1` amplifies noise. SPSA→1 is
   the correct response. **B1 (write symmetry) was likely to make this
   WORSE** — making writes uniform `bonus` would damp ply-{2,4,6} signal
   faster, shrinking the only useful signal we have.

2. **"Berserk's `[2,2,1,1]` reads pattern transfers"** — false. That
   pattern weights ply-1 highest, but Coda's ply-1 is noise. The shape
   peers use is downstream of how their writes accumulate; it doesn't
   transfer to a different write shape.

3. **"depth ≤ 3 cap is the rare-firing cause"** — only partly. Even
   without any depth cap, only 0.13% of eligible moves are below
   threshold. Threshold magnitude is the dominant cause.

### Hypotheses STRENGTHENED by data

1. **HIST_PRUNE_MULT is too tight.** Distribution clusters in `[-0.5, +1.0)`;
   threshold at `-1.0` excludes 96.6% of eligible moves. Lowering MULT
   to ~6000 (half current) brings the gate into the `[-0.5, -1.0)` bucket
   (60,292 moves), increasing fire rate ~20×. Lowering further to ~3000
   would tap the dense `[-0.5, 0.0)` bucket (1.3M moves) which is too
   aggressive.

2. **4D bucketing makes peer thresholds untransferable.** Coda's `[1][1]`
   cells accumulate at 1/12 the rate of `[0][0]`. SF's `-4097*d` is
   calibrated against a 3D table where every cell is well-sampled. Coda's
   per-cell magnitudes scale very differently across buckets — the
   threshold needs to be relative, not absolute.

3. **Cont-hist signal lives at deeper offsets.** The per-write magnitude
   asymmetry (`bonus` at ply-1, `bonus/2` at ply-{2,4,6}) combined with
   the gravity formula means ply-{2,4,6} cells accumulate to richer
   magnitude distributions. This is "accidentally working" in our favour
   despite the weight asymmetry being inherited as a "Obsidian pattern"
   that doesn't actually match Obsidian's design.

### Revised Phase 1 experiments — informed by data

Replace the original Phase 1 plan. New ranking:

**Experiment 1: Lower HIST_PRUNE_MULT to ~6500 (half current)**
- Direct response to the threshold-distribution mismatch.
- Single-line change: `(HIST_PRUNE_MULT, 6500, 500, 50000, 1300.0)`.
- Then SPSA-retune the single param around the new starting point
  (~600-800 iters) and SPRT [0, 3] the retuned value.
- Expected: 5-10× fire rate increase, +2-5 Elo if the mid-range firings
  catch poor moves earlier.

**Experiment 2: Add ply-2 cont-hist to hist-prune score**
- Currently uses `main + cont[ply-1] + pawn`. Adding ply-2 strengthens the
  negative signal (ply-2 has 75% noise but 25% real signal at varying
  magnitudes) and aligns Coda with Reckless/Obsidian/SF/Berserk who all
  sum at least cont1+cont2.
- Two-line change at `src/search.rs:2803-2805`. Bench will change.
- Expected: gate score distribution shifts negative; combine with E1 retune.

**Experiment 3 (Phase 1.5, after E1+E2 settle): per-offset cont-hist read weights**
- Replace `[cm, cm, 1, 1]` with `[CHW1, CHW2, CHW4, CHW6]` as four
  independent SPSA params. Match Viridithas's design.
- The data suggests ply-{4,6} should carry MORE weight than ply-{1,2},
  not less. SPSA will surface the optimal shape.

### NOT to do (data contradicts the prior)

- **B1 — "uniform bonus writes"** — this was the cross-engine review's top
  recommendation. The data says it would damp the deeper-offset signal that
  SPSA has already learned to rely on. Skip until/unless E3 surfaces a
  different shape.

- **"Match Berserk's `[2,2,1,1]` reads"** — Coda's signal distribution is
  inverse, so this would amplify noise.

- **A4 (skipQuiets action)** — at current fire rate (0.1% of eligible),
  it's not worth the monotonicity-check work. Becomes interesting only
  after E1+E2 raise the fire rate to a few percent.

## Update — value sweep + ordering-time sampling

### HIST_PRUNE_MULT sweep (local bench, no fleet cost)

Built 5 binaries at MULT ∈ {3000, 4500, 6500, 8000, 12825} and ran bench:

| MULT  | Nodes     | Fires   | Eligible   | Fire rate |
|------:|----------:|--------:|-----------:|----------:|
| 3000  | 3,204,668 | 266,346 | 2,215,949  |  12.0%    |
| 4500  | 4,279,499 | 199,161 | 2,752,784  |   7.2%    |
| 6500  | 3,711,696 |  63,955 | 2,521,194  |   2.5%    |
| 8000  | 3,957,914 |  43,693 | 2,502,627  |   1.7%    |
| 12825 | 3,796,568 |   3,251 | 2,566,399  |   0.1%    |

**Fire rate is geometric in MULT**, as expected. **Node count is NOT
monotonic** with MULT — moderate pruning at MULT=4500 actually causes
*more* total nodes than at MULT=3000 or 8000. SPSA likely landed at
12825 because it found a search-shape local optimum where hist-prune
is essentially off, not because all tighter values regress.

8000 is the chosen test point — 17× current fire rate, modest enough
that pruning shouldn't tank tactical positions.
**SPRT submitted (`experiment/hist-prune-mult-8000`).**

### Ordering-time vs gate-time cont-hist sampling

Added a second sampling site at the LMR-history adjust point
(different population: any-depth LMR-eligible quiets, not depth ≤ 3
late move-loop quiets). Bench, compare distributions, then revert
(local-only diag).

`[0, 200)` (noise band) by offset:

| Offset | Gate-time | LMR-time |
|--------|----------:|---------:|
| ply-1  | 79.3%     | 61.9%    |
| ply-2  | 74.9%     | 58.5%    |
| ply-4  | 63.6%     | 51.5%    |
| ply-6  | 55.2%     | 46.4%    |

LMR-time samples have ~17 percentage points less noise per offset
(deeper search context = more developed tables). But the **relative
ordering is preserved**: ply-1 is the noisiest offset at both sample
sites. **The ply-1-is-noise finding is not sampling bias.**

This reinforces the earlier conclusion: the cross-engine review's
recommendation to weight ply-1 highest (Berserk `[2,2,1,1]`, B1
symmetric writes) is structurally wrong for Coda's current write
shape. SPSA→1 on `CONT_HIST_MULT` is the correct response.

### Why is ply-1 the noisiest? Hypothesis

Cont-hist write magnitudes:
- ply-1: avg `|bonus|` = 780.3
- ply-{2,4,6}: avg `|bonus|` ≈ 395 (the `bonus/2` halving)

Gravity formula: `entry += bonus - val * |bonus| / MAX_HIST`.

When |bonus| is large, the gravity damping is also large → entries
get pushed back toward zero more aggressively. ply-1 gets the
biggest writes AND the biggest dampening, so entries oscillate around
zero. ply-{4,6} get smaller writes with weaker damping, so entries
settle at non-zero equilibrium points.

**This means the asymmetric writes are CAUSING the ply-1 noisiness,
which CAUSES the SPSA→1 floor on CONT_HIST_MULT.**

Counter-intuitive corollary: making writes SYMMETRIC at full `bonus`
(B1 from the cross-engine review) would amplify ply-{2,4,6} damping,
collapsing them toward zero too. We'd lose the only signal we have.

The right fix shape is the OPPOSITE direction: make writes uniform
at `bonus/2` (i.e., halve ply-1 writes too). Then all offsets have
similar damping and similar signal-to-noise. SPSA can then probably
lift CONT_HIST_MULT above 1.

This is testable as an isolated experiment but tightly coupled to
HIST_PRUNE_MULT — we shouldn't bundle them. Let HIST_PRUNE_MULT=8000
SPRT resolve first.

## Open questions for next session

1. **Sampling bias check.** Is the cont-hist read distribution at hist-prune
   gate (depth ≤ 3, late in move loop) representative of the distribution
   at move-ordering time? Worth adding a parallel sampler in
   `generate_and_score_quiets` to compare.

2. **Why is ply-1 mostly noise?** Two candidate mechanisms: (a) gravity
   damping is faster at higher write magnitudes (ply-1 writes 2× heavier
   than others); (b) ply-1 cells get hit by every cutoff, but the
   `(prev_piece, prev_to)` index is highly diverse so individual cells
   are touched rarely on average. Worth instrumenting per-cell write
   counts to disambiguate.

3. **`[1][1]` 4D bucket sparsity.** Per-cell sample density is 12× lower
   than `[0][0]`. Should we:
   - Drop the 4D bucketing (collapse to 2D `[from][to]`)?
   - Use density-aware threshold scaling (per-bucket multiplier)?
   - Accept it as design choice (the bucket distinction encodes a
     real signal, even if sparser)?
   This is a separate experiment thread.

## Files

- Instrumentation: `src/search.rs` (PruneStats fields + gate hooks +
  cont-hist write hooks + bench output).
- Findings doc: this file.
- Source review: `docs/history_prune_cont_hist_review_2026-05-08.md`.
