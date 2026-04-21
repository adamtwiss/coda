# Factoriser Research — v9 Design Note

**Date:** 2026-04-21
**Status:** Research — no code changes yet
**Context:** Factoriser is a standard technique for bucketed NNUE nets but
our prior v7 attempt contributed to collapsed models. This doc captures
what we know, why v7 failed, and a plan for a cautious v9 retry.

## What is a factoriser?

A factoriser is a **shared feature-transformer weight matrix** added to each
king-bucket's per-bucket weights during training. It's a regulariser:
common patterns across buckets are learned once, per-bucket weights only
learn the *residual* difference.

Bullet syntax (from our v5 config):

```rust
let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
l0.init_with_effective_input_size(32);
l0.weights = l0.weights + expanded_factoriser;
```

At **save time**, the factoriser is folded into the per-bucket weights:

```rust
SavedFormat::id("l0w").transform(|store, weights| {
    let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
})
```

So **inference code is identical** to a non-factored net. The factoriser
only exists during training.

## Why bother?

Bucketed PSQ wastes data. With 10 king buckets × 768 PSQ features = 7,680
l0w rows, each bucket sees only ~1/10 of positions. Features that are
universally true ("bishop pair is good", "passed pawn on 7th is dangerous")
have to be re-learned independently in every bucket.

A factoriser lets every bucket share the learning of these universal
patterns, while retaining per-bucket specialisation for king-safety
asymmetries. The Elo payoff in other engines is typically **+5 to +20** on
well-tuned bucketed configs.

Since v9 uses the Reckless 10-bucket layout (which Reckless trains with a
factoriser successfully), we have a strong prior that factoriser + 10 KB
is a compatible setup.

## Prior Coda history

### v5 — factoriser worked (production)

Our v5 production net (`B9BF926F` — `net-v5-768pw-consensus-w7-e800s800`)
was trained with a factoriser. Config: `training/configs/v5_1024.rs`. It
works. That's the baseline that shipped.

Pattern used:
- `l0f` shape `[ft_size, 768]`, `InitSettings::Zeroed`
- `l0.init_with_effective_input_size(32)` — He init assuming 32 active inputs
- Factoriser folded into `l0w` at save via `.transform`

### v7 — factoriser coincided with collapse

v7 added hidden layers (FT → CReLU → pairwise → L1 SCReLU → L2 SCReLU →
output). The config `training/configs/experiments/exp_a_baseline_screlu.rs`
had the same factoriser pattern as v5, plus hidden layers. The net
**collapsed** (from experiments.md 2026-04-15):

> Unbucketed hidden layers (matching GoChess) still produced collapsed
> eval. Ruled out: bucketing, WDL proportion, data volume (single file has
> 3-4B positions). Remaining suspects: factoriser, L1 i8 quantization,
> L2/L3 float format.

**Crucially: the factoriser was a *suspect*, never isolated as *cause*.**
We removed it along with other changes (quantization format, etc.) and
got a working v7. The training configs now carry the comment:

```rust
// No factoriser — plain FT (factoriser kills hidden layers)
```

But this attribution is not proven. GoChess-style (no factoriser, no
bucketing) ALSO collapsed in our attempts. Whatever went wrong was at
most correlated with the factoriser, not conclusively caused by it.

### v9 — no factoriser carried forward

v9 inherited "no factoriser" from v7 without re-testing:

```rust
// src/nnue_coda_v9_768_threats.rs:200 (bullet fork)
// No factoriser (kills hidden layers, see v7 findings)
let l0 = builder.new_affine("l0", 768 * num_input_buckets + 66864, ft_size);
```

So the v9 production recipe (`DAA4C54E` reckless-crelu) has never
included a factoriser.

## What changed between v7 (collapsed) and v9 (works)

Beyond the removal of the factoriser, v9 differs from v7 in many ways.
Any of these could have been the actual fix:

| Component | v7 exp_a (collapsed) | v9 current (works) |
|---|---|---|
| Factoriser | Yes (suspect) | No |
| Loss | `squared_error` | `power_error(target, 2.5)` |
| LR warmup | 10 SBs | 30 SBs (warm30) |
| Final LR | `0.00001` | tuned lower (5e-6 range) |
| FT activation | CReLU + pairwise | CReLU + pairwise (same) |
| Hidden activation | SCReLU (both L1 + L2) | **CReLU** on current prod (reckless-crelu) |
| Bucket count | 16 (square-based) | 10 (reckless layout) |
| Bucket select for hidden | `select(output_buckets)` then act | shared hidden (same buckets on output only) |
| Threats | No | Yes (+66864 features) |
| Init clipping | AdamW +/- 0.99 on l0w, l0f | AdamW +/- 0.99 on l0w only |
| Data filtering | Mixed | ply≥16, no checks/captures (+22 Elo) |

Any subset of {hidden activation, warmup, LR schedule, loss function, bucket
layout, data filtering} could have been the actual cure. Factoriser may have
been innocent.

## Why v9 is probably safe to add factoriser to

1. **Reckless uses 10 KB nets with a factoriser that works.** Same
   bucket layout as v9, with hidden layers and threat features. Their
   production training recipe must include a factoriser that doesn't
   collapse.

2. **We fixed other things since v7.** Our current v9 recipe has:
   better warmup (30 vs 10 SBs), better loss (power_error 2.5 vs
   squared), better data filtering, CReLU on hidden layers (for
   reckless-crelu prod). All of these are known to improve
   training stability.

3. **Factoriser is fold-in at save time.** Inference is unaffected.
   The risk is entirely "training run wastes GPU time"; no inference
   correctness risk and no Elo regression risk once a net is produced.

4. **The collapse attribution was never conclusive.** v7
   "no-factoriser + other changes" = works. The isolated effect of the
   factoriser was never measured.

## Plan

### Phase 1: smoke test (1 training run, ~4h)

Reproduce the current v9 reckless-crelu recipe *exactly*, add nothing
but the factoriser. Compare:

- Loss curve vs trunk
- Final eval RMS (baseline 580)
- Eval sanity on 500-position EPD sample

Success criteria: loss curve looks healthy, eval RMS within 15% of
baseline, no catastrophic collapse in eval symmetry test.

If Phase 1 crashes or evals degenerate → stop, bisect (see failure-mode
section below).

### Phase 2: SPRT (if Phase 1 passes)

Convert the factorised net, SPRT against current v9 prod `DAA4C54E`.
Bounds `[0, 5]` (expected gain: +5 to +20). Wait for resolution.

- H1: factoriser is an Elo win → promote to v9 prod, retune.
- H0: factoriser is neutral/negative on v9 → compare to Phase 1
  metrics to understand why it trained cleanly but didn't gain Elo.

### Phase 3: tune (if SPRT passes)

Retune-on-branch. Factoriser changes the weight distribution; SPSA
may move pruning params.

## Implementation steps

### Bullet side (fork of `adamtwiss/bullet`)

Add CLI flag `--factoriser` to `examples/coda_v9_768_threats.rs`.
When set, mirror the v5/exp_a pattern:

```rust
let use_factoriser: bool = args.contains(&"--factoriser".into());

// In the network builder:
let l0 = if use_factoriser {
    let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
    let expanded = l0f.repeat(num_input_buckets);
    // Pad the threat portion with zeros so shapes match 768*NKB + 66864
    let full_expanded = ... expand to cover threat features with zeros ...
    let mut l0 = builder.new_affine("l0", 768 * num_input_buckets + 66864, ft_size);
    l0.init_with_effective_input_size(32);
    l0.weights = l0.weights + full_expanded;
    l0
} else {
    let mut l0 = builder.new_affine("l0", 768 * num_input_buckets + 66864, ft_size);
    l0.init_with_effective_input_size(32);
    l0
};
```

**Subtlety:** v9's l0 has `768 * NKB + 66864` input rows. The factoriser
only covers the PSQ part (`768 * NKB`), not threats. Either:

1. Make the factoriser `[ft_size, 768]` and expand only over the PSQ
   rows (threat rows have no factoriser).
2. Or use two factorisers — one for PSQ (shared across KB), one for
   threats (no expansion needed since threats are already shared).

Option 1 is cleaner and matches Reckless's assumed pattern.

Add corresponding `SavedFormat` transform for `l0w` that folds `l0f`
only into the PSQ rows, leaving threat rows untouched.

Also add `set_params_for_weight("l0f", stricter_clipping)` like v5.

### Coda side

**No inference changes required.** The factoriser is fully absorbed into
`l0w` at save time. `coda convert-bullet` reads `l0w` verbatim.

One sanity check: after conversion, bench and EPD sanity must match
expectations. Existing `check-net` script already covers this.

## Failure modes (if Phase 1 collapses)

Bisect in this order (each variable 1 training run = ~4h):

1. **Factoriser only on PSQ, untouched threats** (current plan) vs
   **no factoriser at all** — confirms the v7 suspicion.
2. **Factoriser + linear decay LR** instead of cosine — LR schedule
   sometimes interacts with wider effective-weight space.
3. **Factoriser + lower max_weight clipping** (e.g. 0.5 instead of
   0.99) — prevents runaway weight magnitude.
4. **Factoriser + SCReLU hidden** (the original v7 setup) vs
   **factoriser + CReLU hidden** (our successful v9 prod) — isolates
   the hidden activation interaction.

If any of these resolves, we have a valid "factoriser + X" recipe.

## Inference-side risk

Zero. The factoriser is a training-time concept. `coda convert-bullet`
reads post-fold `l0w` like any other net. The only way factoriser
affects inference is through the resulting weight values themselves.

## Cost estimate

- 1 GPU training run for Phase 1 smoke test: ~4h
- 1 SPRT for Phase 2: ~10-20K games (~4-6h fleet time)
- 1 SPSA retune if H1: ~2500 iters on 8 pairs (~6-10h fleet time)

**Total:** ~1 day end-to-end if everything works first time, ~3 days if
we need to bisect a collapse.

## Open questions for Adam

1. **Should the factoriser cover just PSQ or also threats?** Threats
   are already shared (no bucket repetition), so a factoriser on them
   makes no sense. Decision: factoriser on PSQ rows only.
2. **Which base net to build factoriser onto?** Current v9 prod is
   reckless-crelu (10 KB + CReLU hidden). Propose: use that exact
   config, only adding the factoriser. Keeps variables controlled.
3. **Smoke test data volume:** same as current recipe (full T80), or
   shorter for faster feedback? Proposal: full T80, 200 SBs, matches
   current config so loss curves are directly comparable.

## References

- v5 factoriser (works): `training/configs/v5_1024.rs:72-88`
- v7 factoriser (collapsed alongside other issues):
  `training/configs/experiments/exp_a_baseline_screlu.rs:63-101`
- v7 attribution comment: `training/configs/v7_768pw_h16x32.rs:92`
- v9 carried-forward comment: `bullet/examples/coda_v9_768_threats.rs:200`
- Bullet affine init helper:
  `bullet/crates/acyclib/src/graph/builder/affine.rs:25-30`
- Collapse investigation log: `experiments.md` 2026-04-15 section
  ("v7 Training Investigation")
