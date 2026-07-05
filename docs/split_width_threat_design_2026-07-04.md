# Split-width threat projection — design & implementation plan (2026-07-04)

**Idea:** project threat features into a NARROWER slice of the FT
accumulator (default probe: 512 of 1024 channels) while PSQ keeps the
full width. Threat weight rows halve (1 KiB → 512 B), the threat weight
matrix halves (65.3 → 32.6 MiB), and the threat accumulator halves —
attacking the one cost block nothing in the 2026-07 NPS campaign could
touch: `apply_threat_indices` row streaming (11.9% of Zeus cycles
post-splat). The feature SPACE is unchanged — same 66,864 indices, same
x-ray semantics, same enumeration (the splat), same semi-exclusion —
only the projection width changes.

Status: DESIGN ONLY. Queued for the next net training cycle. Companion
training-side rider: the l1reg activation-sparsity probe (see §8).

## 1. Why (performance model)

From the 2026-07-04 post-splat Zeus profile and the density-probe
session (experiments.md):

| Cost block | Today (w=1024) | At w=512 | Mechanism |
|---|---|---|---|
| `apply_threat_indices` | 11.9% of cycles | ~6% | rows 1 KiB→512 B; threat-acc copy 2 KiB→1 KiB per apply; widening adds halve |
| threat refresh | 1.1% | ~0.6% | same scaling |
| threat stack | 1.17 MiB/thread | ~0.6 MiB | copies stay hotter in L2 |
| threat matrix | 65.3 MiB | 32.6 MiB | **super-linear bonus**: ~20 random 1-row touches/node vs a 32 MiB L3 currently mostly miss; at 32.6 MiB the Zipf-hot subset largely fits |

Estimate: **+6-9% NPS on AVX-512 hosts, +8-12% on small-cache/VPS
hosts** (the hugepage win's shape suggests the footprint effect is
larger under KVM). Bigger than the entire splat campaign if eval holds.

## 2. Eval-risk priors (honest)

- Scale of the signal at stake: x-ray ablation measured **-187 ±20 Elo**
  (OB #2014, S800 net) — pure eval value (ablation keeps the compute
  cost). Even the CI's lower edge dwarfs the +5-8 speed prize, so any
  variant that measurably dents threat expression is a net loss.
- FOR: threat information looks intrinsically low-rank — ~24 active
  features per position (refresh histogram) acting as sparse adjustment
  signals over the dense PSQ picture; hard to believe it needs 1024
  dims of expression. Bottlenecks preserve accuracy when the signal is
  low-rank.
- AGAINST: **no reference engine splits width** (SF / Reckless /
  PlentyChess all project threats full-width). We'd be first. Also the
  shared-affine co-adaptation between threat and PSQ channels is lost
  in the top 512 channels; cross-mixing then only happens in L1 (32
  wide).
- Prior: eval cost centered ~−5 Elo, fat tails both ways (could be
  free; could be −15). Against +5-8 Elo of speed: positive-EV
  experiment, not a sure thing. Fallback point on the curve: w=768
  (−25% streaming, less risk). DO NOT skip the short-bake derisk (§7).

## 3. The load-bearing design choice: WHICH slice (two candidate layouts)

Pairwise activation multiplies channel `i` (a-half, 0..512) with
channel `i+512` (b-half). Today's full-width threat projection makes
every product `(psq_a+thr_a)·(psq_b+thr_b)` — THREE threat term
classes: `thr_a·psq_b`, `psq_a·thr_b`, and the quadratic `thr_a·thr_b`
(mutual attacks, x-ray stacks — the tactically dense class where the
187-Elo signal earns its keep). The slice layout decides which classes
survive (Hercules review R1):

**Layout A — a-half (channels 0..pw):** every product becomes
`(psq_a+thr)·psq_b` — only the single cross-term survives; the
quadratic class vanishes from the pairwise layer entirely (composable
only in L1, 32 wide), and ALL threat expression is multiplicatively
gated by psq_b's clamp (b-channel CReLUs to 0 → threat contribution
erased). Pack simplifies to ONE fused add (`a = adds(psq_a, thr)`).
Fastest/simplest; highest representational risk.

**Layout B — interleaved (thr[0..256] → a-front, thr[256..512] →
b-front, i.e. channels 512..768):** the front 256 products retain all
three term classes; the back 256 become threat-blind. Same row width,
same streaming bytes, same cache win; pack does two half-width adds
(trivially different NPS). Hedge, not a fix — ¾ of products still lose
the quadratic terms — but it preserves an ungated threat expression
channel.

**Resolution is empirical, not aesthetic: the short-bake is a TRIPLE
(baseline / A / B), §7.** For general widths: layout A requires
`threat_width <= pw`; layout B requires `threat_width/2 <= pw` per
half. The probe uses threat_width = pw = 512 at ft=1024.

## 4. Training side (bullet fork, ~2-3 days — the riskier half)

Current: `training/configs`-driven `coda_v9_768_threats.rs` builds ONE
fused affine `l0: (768×NKB + 66864) → ft_size` — one sparse input
tensor, combined index space (threat indices offset by 768×NKB).

Changes:
1. **Loader**: emit TWO sparse input tensors per position (psq features;
   threat features un-offset). This is fork data-pipeline work — the
   place surprises live. Keep the combined-single-tensor path intact
   behind the existing config so old recipes still run.
2. **Graph**: `l0_psq: 768×NKB → ft_size` and `l0_thr: 66864 →
   thr_width`; combine as
   `ft = concat(slice(l0_psq, 0..thr_width) + l0_thr,
                slice(l0_psq, thr_width..ft_size))`
   (or an equivalent pad-add if slice-add is awkward in bullet's graph
   API — verify what exists; budget for a small custom op). Everything
   downstream (pairwise, L1, L2, out) unchanged.
3. **Export**: quantised.bin gains a second weight block of width
   thr_width; bump the checkpoint layout version. Biases stay single
   (ft biases live with l0_psq; l0_thr trains bias-free, matching
   today's inference where threat rows carry no bias).
4. **Smoke validation**: loss-curve sanity on ~50 SB vs baseline; a
   thr_width == ft_size configuration must reproduce the fused
   architecture's loss trajectory (equivalence degenerate case).
   CAVEAT (review R3): two separately-initialized tensors won't
   bit-reproduce the fused run unless the init RNG streams are
   seed-matched — either match seeds or compare converged loss
   statistically; don't chase a phantom regression.
5. **Fork coordination** (review R3): the loader work touches the file
   carrying the live skip-campaign machinery. Branch from current
   recipe state; the fused single-tensor path must stay bit-identical
   (a silent regression there confounds every in-flight recipe
   experiment).

New config flag: `--threat-width <N>` (default = ft_size → today's
topology; probe value pw).

## 5. Inference side (Coda, ~2 days — mostly parameter threading)

1. **Net format**: header gains `threat_width` (v10 format bump or a
   reserved field; nets with threat_width == hidden_size are exactly
   today's semantics — keep loading old nets unchanged).
   `convert-bullet` reads the new checkpoint block and writes
   `66864 × threat_width` i8 rows (~0.5 day).
2. **Loader** (`nnue.rs`): `threat_weights` sized by threat_width;
   `NNUENet.threat_width` field; hugepage path unchanged.
3. **Apply kernels** (`threats.rs`): `apply_threat_indices` /
   `add_weight_rows` and the AVX-512/AVX2/NEON/scalar variants already
   take the width as a parameter — pass threat_width. Tail handling
   already covers non-multiple widths; 512 is a clean multiple of all
   chunk sizes.
4. **Threat accumulator** (`threat_accum.rs`): `ThreatEntry.values`
   hard-sized `[[i16; MAX_FT_SIZE]; 2]` → keep the buffer, use
   `threat_width` as the live extent. Precisely (review R3): this
   halves COPY/replay traffic (which is what §1's "stays hotter in L2"
   rests on) but not the allocation itself; the 1.17 MiB/thread
   footprint wart needs a separate sizing change if ever worth it.
   Refresh/update paths pass threat_width.
5. **Pairwise pack** (`nnue.rs` + NEON + scalar): layout-dependent —
   layout A: one fused add on the a-half; layout B: two half-width adds
   (a-front and b-front). Implement BOTH behind the net header's layout
   field so the triple-bake candidates run on one binary. Gate: the
   existing pack equivalence tests (incl. the i16-rail case) extended
   with (threat_width, layout) parameters.
6. **Tests**: incremental/fuzz suites parameterized by threat_width;
   add a synthetic-net test (random weights, threat_width=512) so the
   whole inference path is validated BEFORE any real training run —
   the two sides are independently testable.

Untouched: enumeration (splat + scalar), threat_index/tables,
semi-exclusion, fuzz-threats parity tooling, Finny (PSQ-only), TT,
search. The 2026-07-04 semantics contract is projection-width-agnostic.

## 6. Quantization note

Threat weights stay i8 at QA=255 scale (same clamp as today). Halving
fan-out doesn't change per-row scale; the FT clamp [0, QA] and
FT_SHIFT are untouched. No EVAL_SCALE interaction expected, but check
RMS on the candidate per the standard net-deploy discipline.

## 7. Validation plan (order matters)

1. **Inference first, synthetic nets**: random-weights nets for BOTH
   layouts at 512 (plus 768/layout-A for curve context — near-free once
   kernels take the width parameter, review R3); full test suite +
   bench determinism + NPS on Zeus. This prices the whole
   speed-vs-eval trade before any GPU time — if neither 512 layout
   reaches ≥+5% on Zeus, stop.
2. **Short-bake TRIPLE (the eval derisk)**: baseline / layout-A-512 /
   layout-B-512, identical S200-class recipe, all fully-baked at the
   short schedule (the "complete the schedule you started" rule).
   Net-vs-net on main, `[-1.5, 1.5]`, as --dev/--base-network
   overrides. Answers the term-class question (§3) empirically for a
   few GPU-hours.
2.5. **End-to-end numerical parity** (review R2 — the historically
   dangerous step): run the fork-side fp32 `--eval-fens` FEN list
   through each trained checkpoint AND through `coda eval-fens` on the
   converted .nnue; demand the known error floor. This is the ONLY
   step that exercises the new format block through the full
   train→convert→infer chain (convert flag mismatches corrupt
   silently — standing CRITICAL failure class).
3. **Prod-length run** only for a (2)-winner within noise of baseline:
   full recipe, candidate vs prod net-vs-net, then the standard deploy
   discipline (retune plan, net_catalog.md entry, hash-based name).
4. NPS validation on titan + ionos-class + Zeus per the multi-host
   rule; SPRT `[0,3]` for the final candidate (speed + eval combined).

## 8. Rider: l1reg activation-sparsity probe (same GPU session)

`training/configs/v7_768pw_h16x32_l1reg.rs` already implements an L1
penalty on FT CReLU activations (v7-era; transplant the penalty branch
into the v9/v10 config, ~half a day). Motivation: the 2026-07-04
density probe measured 67% pairwise-chunk density — dense L1 wins
today, but a lambda-tuned retrain reaching ~30% chunk density would
re-open the fused-NNZ sparse-L1 kernel (all-6-reference-engine
consensus form). Zero inference changes needed for the probe itself;
judge sparsity-vs-loss at s100 per the config's own guidance
(start lambda=0.001). Shares the GPU session, NOT a net (review R3):
l1reg changes activation density, and under this design activation
density GATES threat expression (§3 layout A) — stacking the levers in
one net makes results unattributable.

## 9. Effort summary

| Piece | Estimate |
|---|---|
| Fork: loader two-tensor path + graph split + export | 2-3 days |
| Coda: format + loader + kernels param + pack + tests | ~2 days |
| Synthetic-net inference validation + NPS | 0.5 day |
| Short-bake pair + net-vs-net | GPU hours + 1 SPRT |
| l1reg rider (transplant only) | 0.5 day |

Decision gates: synthetic-net NPS ≥ +5% on Zeus for at least one 512
layout (else stop); short-bake triple has a layout within `[-1.5,1.5]`
noise of baseline (else stop — w=768/layout-A only as a curve point if
the triple shows a width-not-layout pattern). Both gates are cheap
relative to a prod-length training run.

Sequencing (review R3): explicitly AFTER the skip-consolidation full
bake — this design's EV must not preempt measured eval work in flight.

---

## Review (Hercules, 2026-07-05) — endorse with amendments

Design is sound and correctly gated (synthetic-NPS before GPU, short-bake
before prod-length). Proceed. One substantive objection, one missing
validation step, and smaller amendments:

### R1. §3's "no product is threat-blind" is true but glosses the capacity loss

Today (full-width) each pairwise product is `(psq_a+thr_a)·(psq_b+thr_b)` —
THREE threat term classes: `thr_a·psq_b`, `psq_a·thr_b`, and the pure
quadratic `thr_a·thr_b`. With threats confined to the a-half, every product
is `(psq_a+thr)·psq_b` — only the single cross-term survives:

- **Threat-threat interactions vanish from the pairwise layer entirely.**
  Mutual attacks, x-ray stacks, overloaded defenders can then only be
  composed downstream in L1 (32 wide). That's the tactically-dense class
  where eval precision matters most, and threats are a +187-Elo-class
  signal for Coda.
- **Threat influence is gated by psq_b's clamp**: wherever `psq_b` CReLUs
  to 0, the threat contribution to that product is erased. Threats keep no
  expression channel independent of the b-half's activation pattern.

This reshapes the fallback ladder. w=768 is a poor first fallback: it
breaks the §3 pack simplification anyway (768 > pw → back to two adds, a
different NPS point). The same-cost, eval-safer variant is **interleaving
the 512 threat channels across both halves** (256 into each half's front):
same row width, same streaming bytes, same cache win, but the front 256
products retain all three term classes. Costs the single-fused-add
elegance (two half-adds); trivial NPS delta against a potentially decisive
eval difference. **Recommend the short-bake pair be a TRIPLE: baseline /
a-half-512 / interleaved-512.**

### R2. Missing validation: end-to-end numerical parity (train→convert→infer)

§7's synthetic net validates inference in isolation; the s50 smoke
validates training in isolation. The historical killer is the CHAIN
(convert-bullet flag mismatches corrupt silently — standing CRITICAL
memory). The fork-side fp32 `--eval-fens` (built for the C8 parity work)
exists for exactly this: same FEN list through the trained checkpoint and
through `coda eval-fens` on the converted net, demand the known error
floor. Add as §7 step 2.5 — free, and the only step that exercises the new
format block end-to-end.

### R3. Smaller amendments

- **§4 equivalence test**: two separately-initialized tensors won't
  bit-reproduce the fused run's loss trajectory unless init RNG streams are
  matched — seed-match the init, or compare converged loss statistically.
  Otherwise someone chases a phantom regression.
- **§5.4 wording**: keeping the hard-sized buffer with a live extent makes
  the COPIES cheap but does not fix the 1.17 MiB/thread allocation wart.
  §1's "stack stays hotter in L2" rests on copy-extent, which holds — say
  it precisely.
- **Price the fallbacks up front**: once kernels take `threat_width` as a
  parameter, synthetic NPS at 512, 768, AND interleaved-512 is nearly free
  in §7.1 — measure all three so the eval-vs-speed trade is fully priced
  before the short-bake result forces a choice.
- **§8 rider: don't stack levers in one net.** l1reg changes activation
  density, which interacts with pairwise and with this design's premise.
  Sharing a GPU session is fine; sharing a net is not.
- **Fork coordination**: the two-tensor loader work touches the file
  carrying the live skip-campaign machinery (pc spline, wld filter,
  max-score, wrap salts, pz counter). Branch from current recipe state and
  keep the fused path bit-identical — a silent regression there confounds
  every in-flight recipe experiment.
- **Sequencing**: "next net cycle" is right — explicitly AFTER the
  skip-consolidation full bake (pc-milder +12.2 / fs9 +3.6 measured and
  waiting). This design's EV (+5-8 speed minus eval risk) shouldn't preempt
  that.
