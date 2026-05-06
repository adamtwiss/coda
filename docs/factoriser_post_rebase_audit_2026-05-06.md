# Factoriser post-rebase audit — 2026-05-06

## Bug signature

`--factoriser` training on the post-rebase Bullet trainer produces
degenerate weights. Compared to v4-simple (same branch, same recipe,
factoriser disabled):

| Layer | factor probe | v4-simple | ratio |
|---|---:|---:|---:|
| PSQ \|mean\| | 11.4 | 37.9 | 0.30× |
| threats \|mean\| | 11.4 | 19.5 | 0.58× |
| **l0b \|mean\|** | **1.8** | **64.1** | **0.03×** |
| l1w \|mean\| | 0.65 | 0.85 | 0.76× |
| **l1b \|mean\|** | **0.25** | **5.3** | **0.05×** |
| l2w \|mean\| (milli) | 63.5 | 108 | 0.59× |
| **l2b \|mean\| (milli)** | **3.0** | **188** | **0.02×** |
| **output_w \|mean\|** | **2.5** | **27.5** | **0.09×** |

**The unusual signature: BIASES are far more starved than weights.**
l0b/l1b/l2b are at 0.02-0.05× scale (~init-frozen) while weights got
0.3-0.76× partial training. This pattern is uncommon — typically when
gradient flow is blocked, both weights and biases of an affected layer
are blocked together.

## Hypotheses considered & ranked

### REFUTED — l0f clip saturation cascade

The factor probe used `--l0f-clip 4.0` (loose), yet biases still
degenerated to 0.02-0.05×. So the simple "l0f saturates default 0.99
clip → FT input distribution shifts → biases starve" mechanism is
ruled out by the data.

### REMAINING (highest confidence) — IR/autograd interaction with non-leaf weights

The pre-rebase Bullet IR was based on `acyclib` with explicit per-op
`forward_pass` / `backward_pass` device-function methods. Post-rebase
uses a unified TensorIR with autograd-by-rewrite (TakeGradient walks
dependent ops, constructs SubGraph per op).

The key construct in factor-enabled training:
```rust
l0.weights = l0.weights + full_matrix;
```
This MUTATES the Affine's `weights` field — replacing the leaf
parameter Node with a non-leaf computed Node (the AddOp result). The
forward-pass uses `self.weights` (now an AddOp output) for the FT
matmul. Gradient flow back through this AddOp distributes the gradient
to BOTH inputs (`(grad, grad)` per `pointwise.rs`), then the
SubGraph/Inline machinery copies the gradient to each input.

This SHOULD be correct. Code inspection didn't find a smoking-gun bug.
But the agent could not RULE OUT a subtle autograd graph-construction
issue specific to the "Affine.weights field is non-leaf AddOp" pattern,
which would not have applied to the pre-rebase mechanism.

## Why pre-rebase Bullet escaped this (honest framing)

The user reasonably asked: clipping was the same on both sides, the
example is byte-identical, defaults match. So why did factor work
pre-rebase and break post-rebase? **I don't have a confirmed
mechanism — only a ranked list of suspects with what would falsify
each.** The `--l0f-clip 4.0` evidence already rules out the simplest
"saturation cascade" story (biases still degenerated), so the
divergence must live in the IR/autograd machinery itself, not in
clip values.

### Suspect 1 (most likely): Repeat-op gradient semantics differ

Pre-rebase `l0f.repeat(N)` was implemented as `reshape →
matmul-with-ones → reshape` (confirmed by reading
`feature/threat-encoder-compact:crates/acyclib/src/graph/builder/node.rs`
`fn repeat`). The matmul-with-ones is **literally a sum-reduce on
the backward pass** — l0f's gradient = Σ over buckets of (l0w's
gradient on that bucket's slice). Provably correct via standard
matmul backward (`grad_A = grad_C @ ones^T`).

Post-rebase, the same surface `repeat()` calls into a new
`BroadcastAcrossDimension` op with its own dedicated forward+backward
in `crates/compiler/src/tensor/operation/autograd/broadcast.rs`.
Backward = `ReduceAcrossDimension::new(dtype, [outer, repeats, inner],
1, Sum)` per `BroadcastAcrossDimension::invert()`.

**Code-trace finding (2026-05-06):** I verified the math by hand.
For `l0f` shape `[768, 768]` and `repeat(N)` with `dim=batched=0`:
- Forward: outer=1, inner=589824, output[r*589824+j] = input[j].
  Memory layout = N concatenated copies of l0f.
- Backward (Reduce on shape [1, N, 589824], dim=1, sum):
  output[j] = Σ_r input[r*589824+j]. Same as matmul-with-ones
  backward. Mathematically equivalent to pre-rebase.

So the broadcast forward + reduce backward is mathematically
correct. The op-level math is NOT the divergence point.

Possible remaining places for Suspect-1-class bugs:
- Numerical precision differences (cuBLAS gemm vs scalar reduce
  loop) — unlikely to produce a 100× weight-magnitude collapse.
- The reduce kernel implementation on GPU may have an aliasing /
  in-place issue when its input is a SHARED-READER tensor (the
  scenario from Suspect 2 below).

**Falsification:** the gradient-stats hook will show l0f's gradient
magnitude. If it's wrong by an obvious factor (N×, 1/N, near-zero),
the bug is in the Reduce kernel or its scheduling. If it matches
non-factor's l0w gradient magnitude, Suspect 1 is dead.

### Suspect 2: SubGraph construction order for AddOp + non-leaf weights

Post-rebase autograd-by-rewrite walks dependent ops in a specific
order to build the backward graph. The pattern here is unusual:

```
SparseMatmul(input, l0.weights = AddOp(l0w_leaf, repeat(l0f_leaf)))
```

`l0.weights` is **not a leaf** — it's an AddOp output that is
*also referenced from the Affine's field*. When TakeGradient walks
back from the loss, it needs to:

1. Compute grad-of-l0.weights (the SparseMatmul backward output for
   the weights side).
2. Distribute that grad through the AddOp to (l0w_leaf, repeat(l0f_leaf)).
3. Recurse into the repeat backward to get grad-of-l0f_leaf.

If step 2 happens correctly but step 3's repeat-backward is computed
in a context where `l0w_leaf` already consumed the AddOp's output
gradient buffer (e.g., gradient-buffer aliasing on the AddOp output),
l0f could read a zeroed buffer. Pre-rebase's per-op explicit
backward never had this hazard because the op author wrote each
gradient flow by hand.

**Code-trace finding (2026-05-06):** TakeGradient
(`crates/compiler/src/tensor/transform/autograd.rs:83-138`)
**explicitly handles** the `(grad, grad)` case from `CABinary::Add`
backward via `unique_igrads` deduplication and `igrad_map`
multi-input-per-grad. After processing AddOp, both inputs end up
pointing to the *same* SubGraph output tensor ID in the `grads`
map. Subsequent Broadcast backward reads that shared tensor ID
correctly. The IR-level construction looks correct.

If the bug is here, it must be downstream of TakeGradient — in
either:
  (a) `InlineSubgraphs` (`crates/compiler/src/tensor/transform/inline.rs`) —
      whether the inliner preserves the multi-reader semantics of
      the shared-gradient tensor, or accidentally rewrites one of
      the readers in a way that disconnects it.
  (b) The backend kernel scheduler / gradient-buffer allocator —
      whether liveness-analysis correctly extends the shared
      tensor's lifetime to span BOTH consumers (l0w accumulation
      AND Broadcast→l0f path).

This is a narrower suspect now, but still not falsified.

**Falsification:** gradient-stats showing `l0w` gets normal gradient
but `l0f` gets near-zero gradient.

### Suspect 3: Init RNG stream order differs

Pre-rebase and post-rebase both use a global RNG stream for
`vec_f32` weight init, but the order in which weights are
constructed (PSQ → l0w → threats → l0f → bias → l1w → ...) may
differ between IRs. If l0f's init scale is materially different
post-rebase (e.g., consumed at a later RNG offset that happens to
produce a higher-stdev draw), the network starts in a different
basin.

This is **less likely** to be the dominant bug — RNG-stream-order
shifts are stochastic and the bias-freeze signature is
deterministic across factor-enabled runs.

**Falsification:** dumping `l0f`'s SB0 (post-init, pre-train) mean
and abs-mean and comparing pre-rebase vs post-rebase. If they
differ by >10%, RNG order is shifted and worth pinning.

### What is NOT the divergence

- Clipping cap values (defaults match: ±1.98 default; example
  overrides l0w to ±0.99 in both).
- Affine struct shape (Clone+Copy + `weights, bias` fields in both).
- AdamW step kernel (verified identical step semantics).
- The forward `self.weights.matmul(input) + self.bias` expression
  (verified identical both sides).

### The pragmatic shortcut: gradient-stats first

All three suspects produce **distinct gradient-magnitude signatures**:

| Suspect | l0f grad | l0w grad | bias grad |
|---|---|---|---|
| 1: Repeat backward N× off | Wrong magnitude | Normal | Cascading-distorted |
| 2: AddOp + SubGraph aliasing | ~zero | Normal | Cascading-distorted |
| 3: Init RNG order | Normal | Normal | Normal (would NOT match — refuted by symptom) |

Suspect 3 doesn't match the empirical symptom on inspection, so
the gradient-stats probe primarily disambiguates 1 vs 2, both of
which are tractable to fix once identified.

### REFUTED — Affine fused bias

The post-rebase Affine forward (`crates/trainer/src/model/builder.rs:246-248`)
is identical to pre-rebase: `self.weights.matmul(input) + self.bias`.
Bias gradient flows through standard broadcast-across-batch / sum-reduce
paths, which were verified correct on inspection. Not the bug.

### REFUTED — AdamW kernel, gradient-buffer aliasing

Verified by code inspection. AdamW's clip/decay/eps substitutions are
non-colliding for the values used. Gradient buffers are unique per
weight ID (asserted in builder.rs:190). Neither is the bug.

## Most actionable next diagnostic

**`BULLET_DUMP_GRADIENT_STATS` env hook** is now implemented on
`factor-probe-clean` (commit 3b78e42, `crates/trainer/src/run.rs`).
Mirrors `BULLET_DUMP_WEIGHT_STATS`. Reads the LAST batch's gradient
buffer at each SB boundary, before any other op can mutate it.

Output line: `grad-stats sb=N id=ID n=... mean=... |mean|=... max|g|=... zero=PCT%`

Pair with `BULLET_DUMP_WEIGHT_STATS` in a single run so weights and
gradients are sampled at the same SB boundaries.

**Decision tree from gradient stats:**

- If factor's l0b/l1b/l2b **gradients are tiny** vs non-factor:
  gradient-flow bug. Narrows search to autograd graph for the AddOp +
  non-leaf weights chain in `crates/compiler/src/tensor/transform/autograd.rs`
  (TakeGradient / SubGraph construction).
- If factor's gradients are **normal-magnitude but weights don't move**:
  optimiser bug. Narrows to AdamW step semantics for weights with
  mutated Affine refs in `crates/trainer/src/optimiser.rs`.

Either path narrows the search to a localised code area where direct
inspection is much more tractable.

## Files referenced

Pre-rebase canonical:
- branch `feature/threat-encoder-compact`
- example: `examples/coda_v9_768_threats.rs:339-396` (factor build code,
  identical pre/post rebase — bug is in the framework not the example)

Post-rebase relevant:
- `crates/trainer/src/model/builder.rs:131-154` (new_weights, new_affine,
  init_with_effective_input_size — all correct)
- `crates/trainer/src/model/builder.rs:166-236` (build → fwd/bwd graph
  construction, gradient registration)
- `crates/compiler/src/tensor/operation/autograd/broadcast.rs:8-17`
  (broadcast bwd is sum-reduce — correct)
- `crates/compiler/src/tensor/operation/autograd/pointwise.rs:13-41`
  (CABinaryOp::Add bwd returns `(grad, grad)`)
- `crates/compiler/src/tensor/transform/autograd.rs:47-144`
  (TakeGradient — gradient accumulation logic)
- `crates/compiler/src/tensor/transform/inline.rs:12-58` (InlineSubgraphs)
- `crates/trainer/src/optimiser/adam.rs:33-50` (AdamW kernel with
  hard-coded WMIN/WMAX clip)
- `crates/bullet_lib/src/value.rs:181-201` (existing
  BULLET_DUMP_WEIGHT_STATS, easy to mirror for gradients)

## Status

- Bug confirmed (degenerate weights observed empirically).
- Single-line code bug NOT identified by inspection.
- Most likely class: subtle autograd/IR interaction with the
  Affine.weights-as-non-leaf-AddOp pattern.
- Next step: gradient-stats instrumentation, ~30 min implementation
  + ~30 min for two SB1-SB5 probes (factor + no-factor).
