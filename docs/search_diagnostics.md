# Search diagnostics

These modes are measurement tools, not playing features. They are disabled by
default and leave the normal depth-12 bench at 2,659,499 nodes. Run them at a
fixed depth with one thread; sampled verification deliberately changes the
sampled tree and its shared search state.

## Decisive-beta RFP verification

```sh
RFP_DECISIVE_SAMPLE=64 ./target/release/coda bench 12
```

Every RFP would-cut whose beta is in a decisive-loss band is counted. A stable
hash selects roughly one event in `N`; that event declines RFP at the current
node and searches the same node at full depth while descendants retain RFP.
The report separates TB/downgraded and mate beta bands and buckets candidates
by remaining depth, ply proximity, and cutoff surplus.

`disagree` means that the declined search returned below beta. It is useful
counterevidence to the cutoff, but not proof of a false cutoff: the verification
search perturbs TT/history state and the original RFP result is a lower bound,
not an exact score. Stop-interrupted samples are excluded.

`RFP_VERIFY_SAMPLE=N` performs the same experiment across ordinary RFP cutoffs.
If both modes are enabled, decisive candidates use the decisive sampler and the
general sampler handles the rest.

## Correction-history calibration

```sh
CORR_AUDIT=1 ./target/release/coda bench 12
```

The aggregate table compares raw and corrected evaluation using bound-aware
loss: a fail-high only proves a lower bound and a fail-low only proves an upper
bound. The source table then removes one of pawn, white non-pawn, black
non-pawn, continuation, or transition correction from the production blend.
Its `gain` is:

```text
loss(without source) - loss(full correction)
```

Positive gain is evidence that the source helps calibration in that bucket;
negative gain is evidence that it hurts. These leave-one-out effects are not
additive because sources interact and the blend has integer rounding. `RFP
on/off` counts threshold changes attributable to removing that source; more
than one source can be credited at the same node.

The report also shows update clipping by ply. A high clip rate means the update
cap, rather than the observed residual, is controlling learning in that region.
Halfmove buckets with few samples should not drive a change without a targeted
position corpus.

