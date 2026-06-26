# Cinder — engine review (ideas to test in Coda)

**Source:** `~/chess/engines/cinder` — Cinder v0.4.1 by Bruno Dutra (`brunocodutra`),
modern/actively-developed Rust engine, ~Coda's strength on Lichess. Reviewed
2026-06-26 (cinder HEAD `2a751e9`). ~20k LOC.

> **Strength context (2026-06-26, in-progress local RR):** Cinder is **#9 (+5
> Elo), BELOW Coda at #6 (+18)** — a near-peer but ~13 Elo weaker. Per our
> cross-engine rule (CLAUDE.md: only engines *stronger* than Coda are valid
> "should we do X" consensus references), **Cinder is weak evidence, not
> consensus.** Its design choices are at most corroboration. This is borne out
> below: of the top search ideas, **4 (A1/A2/A3/A6a) were already in Coda** and
> Coda is the stronger engine. So treat the novel survivors (A4/A7/A9) as
> *speculative cheap probes* — a good idea can still come from a weaker engine,
> but the prior that any given Cinder divergence helps Coda is low. **Do not
> keep mining Cinder for more ports expecting wins; the cheap-and-clean probes
> are already fired.**

This captures only where Cinder **diverges** from Coda — things Coda doesn't
already do — framed as concrete, SPRT-able hypotheses. Filtered against Coda's
known feature set; ranked by promise/transferability. Pure-logic and
compute/layout ideas rank highest (they transfer cross-engine); signal-filtering
and eval-class ideas are flagged lower-confidence per
`feedback_consensus_patterns_dont_always_transfer` /
`feedback_naive_ordering_ports_dont_transfer`.

**Architectural framing.** Cinder is the *inverse* of Coda's NNUE bet: **plain
HalfKA-mirror inputs (no threat features) + a deep f32 MLP tail with a residual
skip and per-phase buckets.** Coda = rich threat/x-ray inputs + shallow tail.
Coda already measures SF-class on eval quality (`project_eval_quality_is_sf_class`),
so the transferable value is almost entirely in Cinder's **compute/layout kernels
and search structure**, not its topology. Also note: Cinder treats
depth/reductions/extensions/margins as continuous `f32` driven by SPSA-fit
*quadratic* feature models (`convolve`, `engine.rs:17-34`) — port individual
*structure*, never their numbers (different eval scale + tree), and respect
CLAUDE.md's loose-knob / SPSA-dimensionality guidance.

---

## A. Search ideas (ranked)

### A1. Singular search = multi-cut loop + singular *reduction* (negative ext) — HIGH
`engine.rs:773-824`. Cinder's singular verification loops over **all** non-TT
moves at `se_depth-1` around `se_beta`, and:
- **multi-cut:** if `min(se_score, se_beta) >= beta` → return (whole node fails high);
- **singular reduction:** if not singular (`se_score >= se_beta`) → apply a *negative*
  extension (reduce the TT move, it's demonstrably non-unique);
- **proportional extension:** if singular, `ext = clamp(gamma*(se_beta-se_score)^delta, 0..cap)`
  — graded by how far the best alternative fell short, not a fixed double-ext step.

Coda has singular + double extensions with fixed `DEXT_MARGIN`/`DEXT_CAP`, no
multi-cut fold-in, no singular reduction. Three separable tests:
- **A1a multi-cut** (fail node high when verification clearly beats beta) — pure logic, SF/Berserk prior. SPRT `[0,3]`.
- **A1b singular reduction** (reduce TT move when proven non-singular) — pure logic. SPRT `[0,3]`.
- **A1c proportional extension** (graded vs binary single/double) — tree-shape, retune-on-branch.

### A2. Aspiration fail-high *depth reduction* — HIGH
`engine.rs:1075-1092`. Each consecutive root fail-high re-searches at progressively
**reduced depth** (`reduction += aw_fh_reduction[0]`, capped); fail-low decays it and
lerps the upper bound inward. Coda widens the window on failure but doesn't appear
to drop depth on repeated fail-highs (standard SF/Ethereal pattern → less time wasted
re-searching a deep tree on volatile roots). Test: on the Nth consecutive root
fail-high, reduce re-search depth ~N (capped). SPRT `[0,3]` + LTC.

### A3. Continuation-history term in the LMR amount — MED-HIGH
`engine.rs:896-897`. LMR reduction includes both butterfly-history *and*
**continuation (counter-move, ply-1) history** terms. Coda uses main history to
adjust effective lmr_depth in futility; confirm whether Coda's *LMR amount* reads
cont-hist. If not, add it (reduce less when ply-1 cont-hist is high). Retune-on-branch.
Med confidence — Coda's 4D threat history may already encode this gradient
(`feedback_naive_ordering_ports_dont_transfer`); gate on the retune showing real
param movement.

### A4. Zobrist-delta ("transition") correction-history source — MED-HIGH
`engine.rs:247-249`, `correction.rs`. A correction source indexed by
`zobrist(ply-i) XOR zobrist(ply)` (i=0,1) — a hash of the *transition between two
recent positions*, capturing "this structural *change* tends to be mis-evaluated"
(distinct from continuation-correction's piece-to index). Coda's 6 sources don't
include this. Add a 7th source, small table, proportional gravity like the others.
Cheap (infra exists). SPRT `[0,3]`. Risk: redundancy with continuation correction.

### A5. Butterfly-keyed "attention" node-effort TM factor — MED-HIGH (TM-class)
`attention.rs`, `control.rs:140-144`, `engine.rs:564,988`. A butterfly `[from][to]`
node counter per root move; the soft time bound is scaled by the fraction of total
nodes spent on the **current best** root move. Because it's keyed by *move identity*
and persists across iterations, a fresh best move inherits a low effort-fraction →
automatic "best move just changed → keep thinking." Coda already has a per-root-move
node-fraction TM factor; the novel bit is the **persistent butterfly indexing**. If
Coda's factor resets each iteration, switch to a persistent per-move accumulator.
Validate TM-class (LTC + cross-engine ponder RR), not just STC SPRT.

### A6. ProbCut improving-aware margin + TT node-type gate — MED
`engine.rs:740,745`. ProbCut margin is improving-aware, and ProbCut only runs when
the TT entry is a fail-high **non-quiet** lower bound (`is_fh && was_cut && !was_quiet`).
Coda gates on `beta+margin`, staticEval, `SEE>=0` but margin isn't improving-aware.
- **A6a** add improving term to Coda's ProbCut margin — cheap, SPRT `[0,3]`.
- **A6b** add TT-node-type gate — may be redundant with the staticEval gate; med.

### A7. Graded TT near-miss (fail-high pruning over a depth window) — MED
`engine.rs:655-668`. Accepts TT fail-highs up to `K` plies short if
`ttLower - margin(depthGap) >= beta`, with `margin = fhp_margin_depth*(depth-t.depth)
+ fhp_margin_scalar`. Coda's near-miss is fixed (1 ply, 80cp). Generalize to a
depth-gap-scaled margin over a small window. SPRT `[0,3]`. Med — near-miss already
captures most of this; win is in allowing 2-3 ply gaps.

### A8. Node-type-dependent IIR amount — MED-LOW
`engine.rs:613-619`. IIR is a fractional decrement that differs for all-nodes vs
cut-nodes, applied at all no-TT-move depths. Coda's IIR is a fixed integer decrement
≥ IIR_MIN_DEPTH. Test: node-type-dependent IIR reduction (one new tunable). SPRT
`[0,3]`. Watch loose-knob concern.

### A9. Threat-aware continuation *correction* — MED
`correction.rs:50,62-67`. Cinder's continuation correction is indexed by
`[turn][is_check][from][to][from_threatened][to_threatened]` — threat- and
check-conditioned like its histories. If Coda's continuation correction isn't
threat/check-conditioned, add those dims (mirrors what Coda already does for main
history). Low-risk consistency port, SPRT `[0,3]`.

### A10. King-relative (attacker/defender) move histories — MED-LOW
`history.rs:54-150`. AttackerHistory `[is_quiet][enemy_king_sq][piece][to][to_threatened]`
and DefenderHistory `[is_quiet][own_king_sq][piece][to][threat]` — histories of moves
*relative to king squares*, summed into move rating alongside butterfly. A genuinely
different axis than Coda's threat history. Worth one cheap shot, but prior says
ordering-history ports rarely transfer to Coda. SPRT `[0,3]`, don't over-invest.

### A11. Other smaller probes
- **QS TT store** (`engine.rs:548-550`): Cinder writes TT from QS + PVS-in-QS +
  `break`-on-futility (gain-monotone ordering). If Coda's QS doesn't store TT, add it
  (cheap, standard). Low-med.
- **Targeted quadratic margin term** (`convolve`, `engine.rs:17-34`): Cinder's margins
  are quadratic in (depth, improving, …). Cheap probe = add one `depth²` term to RFP
  or futility margin (one tunable). Low-med. Do NOT import the whole polynomial/14-vector
  LMR model — conflicts with SPSA-dimensionality guidance.
- **rule50 / material eval scaling** — see B-section interplay; eval-class.

---

## B. NNUE / eval ideas (ranked)

> Cinder topology (so the rest reads): FT 768×16-bucket → **2048 i16** accum/persp;
> CReLU + **pairwise-mul** 2048→1024 (`lin.rs:32-50`); tail `Lin(2048→32 int8) →
> 2×Lnn(32→32 f32, two-sided SqrReLU) → Lro residual skip → Lno(32→1)`, **per-phase
> (8 piece-count buckets)** with a per-phase residual weight. **No threat features.**

### B1. Arity-specialized fused accumulator apply — HIGH (targets named hotspot)
`transformer.rs:136-143`, `evaluator.rs:345,399`. `accumulate` / `accumulate_in_place`
dispatch on the **exact `(sub,add)` arity** (8 arms: 1-sub, 1-add, sub+add, 2-sub+add…),
so the common quiet case is a single branchless fused loop — no per-element Option
checks, illegal arity is `unreachable_unchecked`. This is exactly SF's clean
arity-specialized apply. **Directly targets Coda's biggest known untouched NPS lever:**
threat-accumulator apply is ~31% cyc vs SF ~5.5% (`project_threat_index_microopts_neutral`).
**Action:** audit Coda's threat-apply — variable-length dirty loop with per-element
branches, or arity-specialized fused loops? If the former, specialize the common
(skewed) arities. Measure via the existing cycle-attribution harness; SPRT `[-2,1]`.
**Chase this hardest.** Pure inference, no retrain.

### B2. nnz-gather sparse L1 int8 matmul (vpcompress find_nnz) — HIGH
`lin.rs:52-104`, `nzs.rs`. After CReLU+pairwise the L1 input is sparse; Cinder builds
a compressed non-zero index list (AVX-512-VBMI2 `_mm512_maskz_compress_epi16` path +
256-entry trailing-zeros LUT fallback) and runs `mul_add_4x8` (VNNI `dpbusd` / AVX2
`maddubs+madd`) **only over active inputs** — the SF "find_nnz" trick. **Verify first**
whether Coda's `sparse_l1.rs` already gathers nnz via vpcompress. If not, port the
VBMI2 path (Zeus/gpu3 have it) + LUT fallback. NPS bench AVX-512 vs AVX2; SPRT `[-2,1]`.
Compute-side → transfers. Pairs with B5.

### B3. `mulhi`-fused pairwise activation — HIGH
`lin.rs:38`, `mulhi.rs`. Pairwise product of accum halves computed as one signed
high-multiply with a fixed 9-bit shift (`_mm*_mulhi_epi16` / NEON `vqdmulhq_s16`) —
multiplies **and** rescales out of QA² in one instruction, no clamp-square-shift,
no byte-decomposition. Coda's pairwise does split/clamp/multiply/byte-decompose.
Replace with `mulhi`-with-shift fuse at the FT→L1 boundary (runs every evaluated node).
NPS bench all ISAs. Pure inference but **needs the quant scales to line up** (`<9>`
encodes QA=255/QB) — may need a one-time requantize, not a full retrain. Med risk on
exact scale.

### B4. Asymmetric single-side Finny refresh — MED-HIGH
`evaluator.rs:35-47,250-311`. Per-side `pending: Option<Update|Refresh>`; `push` marks
**only the moving side** `Refresh` if its king crossed a bucket boundary — the other
perspective keeps incrementally updating. `evaluate` walks backward per side collapsing
`Update`s to the nearest computed accum/Refresh, then replays forward. **Verify** Coda
only refreshes the side whose king changed bucket; if it refreshes both, gate it.
Low-risk inference, SPRT `[-2,1]`.

### B5. Co-designed pre-blocked/interleaved weight layout — MED
`nnue.rs:44-90,112-117`. L1 int8 weights reshaped to `[i8;4]` blocks + interleaved at
load so the nnz-gather inner loop streams contiguous weight blocks per active input;
float L2/L3 also `arrange_in_blocks`'d. Free runtime win. Coda knows "cache > compute
for weights" + "L1d layout ±5 Elo, bench-invisible" (`project_v9_nps_findings`,
`feedback_layout_variance_is_a_known_unknown`). Ensure Coda's L1 weight layout matches
its sparse-gather access order. Layout-only, bench-measurable, SPRT `[-2,1]`. Pairs with B2.

### B6. SIMD mailbox-diff refresh — MED (rare path)
`evaluator.rs:320-353`. Finny refresh diffs the cached `[Option<Piece>;64]` mailbox via
one `u8x64::simd_ne(...).to_bitmask()` (vs iterating 12 piece bitboards), then 2-sub/2-add
batches into the arity-matched apply. Refresh is rare (bucket change only) → low Elo;
only chase if bucket-refresh shows in profiling.

### B7. Retrain-gated topology bits — MED-LOW (mostly unfavorable prior)
For completeness; all require a net retrain and most conflict with prior negatives
(`project_psqt_dual_regress_s800`, multi-stage tail-capacity regressions). Don't propose
without a cheap S200 paired-probe (canonical recipe) first.
- **Per-phase full tail (8 buckets) incl. residual** (`nnue.rs:98,174`) — different
  capacity allocation; tail-capacity adds have regressed at S800.
- **Lro residual skip** (`lro.rs`, `trainer.rs:436`) — only motivated if pursuing a
  deeper tail (ties to `project_modern_hw_eval_richness_pivot`).
- **Two-sided SqrReLU** (`concat(x,-x)` then square, `lin.rs:111-114`) — keeps sign through
  squaring, doubles inter-layer width; only meaningful inside a deeper-tail experiment.

### B8. Training-data filter mechanics — LOW-MED (retrain-gated)
`trainer.rs:99-279`. Novel vs Coda's ply≥16/quiet/WDL stack: **adaptive piece-count
balancing** (online atomic counters maintaining a target material distribution,
`:172-199`) and a **placeholder-zero guard** (drops score=0 skip-markers not real evals,
`:234-257`). Also spline early-ply ramp (`:202-223`) and result-sign score-anomaly
(`:226-231`) — but Coda already does ply/quiet filtering. Port *mechanics not constants*
(LC0-WDL scale, `feedback_aggressive_filters_overfit_at_s200`). Piece-count balancing is
one we haven't logged trying — S200 paired probe. Placeholder-zero guard is a cheap
correctness add if our binpacks ever carry skip-marker zeros (relevant to the stamper).

---

## C. Combined test queue (cheapest-first, highest-confidence)

**Pure-logic search (SPRT `[0,3]`, no retrain):**
1. A1a singular multi-cut · 2. A1b singular reduction · 3. A2 aspiration fail-high
   depth reduction · 4. A4 zobrist-delta correction · 5. A6a ProbCut improving margin.

**Inference/NPS (audit-first, SPRT `[-2,1]`, no retrain):**
6. **B1 arity-specialized threat-apply** (highest value — named hotspot) ·
7. B2 nnz-gather sparse L1 (+ B5 weight layout) · 8. B3 mulhi-fused pairwise ·
9. B4 single-side refresh gate-check.

**Retune-on-branch / tree-shape:** A1c proportional ext · A3 cont-hist LMR.

**TM-class (LTC + ponder RR):** A5 persistent butterfly attention factor.

**Retrain-gated (S200 probe first, mostly unfavorable prior):** B7, B8.

## D. Not worth porting
- Full `convolve` quadratic model + 14-vector LMR model — too many loose knobs
  (CLAUDE.md SPSA-dimensionality). Cherry-pick single terms (A1c, A8, A11) instead.
- Tunable `piece_values` for SEE — Coda's SEE values are consensus-aligned already.
- Cinder's topology wholesale — Coda's threat/x-ray inputs are a strict richness
  advantage Cinder lacks; copying its plain-input/deep-tail bet throws that away.
