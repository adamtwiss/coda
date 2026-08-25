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

## Targeted mate and tablebase corpus

`testdata/search_audit.epd` contains full six-field FENs rather than ordinary
four-field EPD records, so halfmove clocks survive intact. Its deterministic
mix is:

- 32 fixed-depth mate roots and their 32 legal child positions;
- 64 five-piece Syzygy roots;
- 64 six-piece roots with a legal capture into the five-piece tables;
- retained interior counterexamples found by this diagnostic.

Each tablebase placement is represented at halfmove clocks 0, 50, 90 and 99.
The runner clears the TT and persistent histories between positions because
clock variants have the same board Zobrist key but different draw state.

Build the corpus again from the local tactical suites and five-piece tables:

```sh
./target/release/coda build-search-corpus \
  --syzygy /home/adam/chess/tablebases
```

Run all positions, or isolate one metadata category:

```sh
RFP_DECISIVE_SAMPLE=1 ./target/release/coda search-corpus --depth 10
RFP_DECISIVE_SAMPLE=1 ./target/release/coda search-corpus --depth 10 \
  --kind tb-transition
```

The search path deliberately bypasses UCI's root-DTZ shortcut. Five-piece
roots therefore search normally while their children exercise interior WDL
score propagation. `--kind` accepts `mate-root`, `mate-child`, `tb-direct`,
`tb-transition`, or `tb-counterexample`.

A retained interior FEN records the board and observed window, but searching it
as a new root is not expected to recreate the event: root search has a different
window, TT state and ply. The deterministic `tb-transition` run is the complete
reproduction path; the interior record makes the discrepant node inspectable.

### TB-loss window audit

Interior WDL results are directional bounds: a definite win is a lower bound
and a definite loss is an upper bound. In particular, a synthetic TB-loss score
that lies above `beta` cannot justify a fail-high return. The retained
counterexample exposed that reversed-bound case.

RFP also needs care when an ordinary centipawn eval meets a TB-loss beta. The
search now declines that cutoff only in three measured contexts: a maximum-men
node below `SyzygyProbeDepth`, a one-piece-larger node with a capture into the
tables, or a probe-confirmed definite loss. The corpus report splits those
counts as `max-men below depth`, `transition capture`, and `probed loss`.

With the five-piece tables, `RFP_DECISIVE_SAMPLE=1` found zero TB-band
disagreements in the direct depth-10 run and in transition runs at depths 8, 10
and 12. Mate-band disagreements remain visible and are intentionally not
folded into this TB-only candidate; a context-wide loss guard more than doubled
the unsampled transition tree at depth 10.
