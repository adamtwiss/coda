# Fixing eval blindspots via the training corpus (NNUE)

**Context.** The overrate corpus (`docs/overrate_corpus_thread_2026-06-15.md`)
found that ~60% of current trunk's real-game LIVE blindspots **persist at
depth** — the static NNUE eval mis-ranks the move, more search doesn't help.
Concrete recurring class: **over-valued active bishop sorties** (Bxf4, Bd6,
Bf3, Bb3, Bd2, Bd4+) and some loosening pawn pushes (f5/f6). In an NNUE world
you don't fix this by hand-tuning a term — you change what the net learns.
This note is the methodology.

## Why an NNUE has a persistent blindspot

**Training setup (correct as of 2026-06-17, per Adam):** Coda trains on the
**same SF/LC0-generated datasets that SF trains on** (T80 etc.), scored by LC0
or SF. **No self-play training.** So the labels are high-quality and identical-
in-kind to SF's — the blindspot is *not* a label-quality or self-play-echo
problem. That rules out two tempting hypotheses and narrows the cause to one:

1. **Sparse coverage + weaker generalization on the sparse region (the cause).**
   The net is an interpolator over its training distribution: accurate where
   (position, label) pairs are dense, less accurate where they are sparse.
   Bishop sorties that are *bad* are off-the-beaten-path of strong play —
   LC0/SF self-play rarely enters them, so the post-sortie positions are rare
   in the shared data. The net extrapolates there and guesses high.
2. **Why Coda and not SF, on the same data?** Because it's a *generalization*
   gap, not a *data* gap: SF's net is larger/differently-shaped and converges
   better on those sparse regions, and — critically — SF's *search* papers over
   any residual static error (recall the RpZ9LbYM start position, where Coda's
   static eval +1.87 was actually *more* accurate than SF's −0.47; SF wins via
   depth, not a better static read everywhere). So the lever for Coda is either
   (a) **more density in the sparse region** of the shared corpus, or (b) a
   bigger/better-converged net (the L1=32 direction already pushes (b)). This
   note is about (a).

## CAUSE, MEASURED (2026-06-17): the bishop-sortie class is ~70% THREAT-driven

The sparse-coverage framing above is the *general* model, but for the specific
**forward-active-bishop sortie** class we now have a direct measurement that
narrows it to **feature-design, not (only) data density.** Method: a diagnostic
flag `CODA_NO_THREAT_ACC=1` (nnue.rs, off by default, bit-identical) zeros the
threat-accumulator contribution → FT-only eval. Re-running the sortie-preference
test (mover-POV cp by which Coda prefers the bad sortie `am` over the correct
`bm`, vs SF) with threats ON vs FT-only:

| sortie | pref WITH threats | FT-only | SF | effect of threats |
|---|---|---|---|---|
| Bf3 | +408 | **−552** | +39 | **flips to correct move** |
| Rf2 | +301 | **−368** | +74 | **flips** |
| Bd6 | +195 | **−35** | +56 | **flips** |
| Bd2 (seed) | +90 | **−53** | +6 | **flips** |
| f6 | +464 | +126 | +158 | 73% from threats |
| Bxf4 (capture) | +857/+733 | +806/+509 | +142/+82 | mostly FT (material) |

Forward-bishop sortie class: mean preference **+457 (with threats) → +135
(FT-only) → SF +65**. So **~70% of the over-credit is the threat features**, and
on the *pure* sorties (Bf3, Rf2, Bd6, Bd2 — not captures) **turning threats off
FLIPS Coda to the correct quiet move.** On the seed (Bd2), threats are literally
why Coda plays it (+90 → −53).

**Mechanism:** the threats reward "my bishop now attacks these squares" but
under-encode/under-weight that the bishop is loose/attacked in return, so raw
activity reads as pure upside. Captures (Bxf4) are the exception — there the
over-valuation lives in the FT/material, not threats.

**Two corroborating facts:** (i) the threat *color-asymmetry* (the C8 tie-break,
median 0cp on this corpus, max 62cp) is NOT the driver — it's symmetric
over-credit; (ii) the over-valuation is systematic across all v9 nets incl.
future-prod multi-v6 (all ~30% on overrate.epd), so it won't fall out of the
normal train-bigger/longer flywheel.

**Fix implication — this splits the lever in two:**
- **Threat-side (the dominant driver for pure sorties):** either (a) targeted
  data so the threat *weights* learn forward-active-but-loose bishops are bad
  (keeps the architecture — the features exist, the weights mis-rank), or (b) a
  threat-feature change ensuring the attacked-in-return signal is present and
  weighted (retrain + revalidate). Open sub-question being checked: does an
  *incoming*-threat feature even fire on the bishop after Bf3/Bd2? If not → a
  feature-coverage gap; if yes → a weight/training-emphasis gap.
- **FT-side (captures + residual):** the targeted-data approach below.

So the training-corpus method below is still right for the FT-side and for
teaching the threat weights — but the finding is that the bishop-sortie blindspot
is **substantially a threat-feature over-credit**, which the original
"sparse-coverage, not a bug" framing under-weighted.

## The fix: targeted hard-example mining → re-label → mix → verify

This is the eval-flywheel made deliberate. The overrate corpus is the
*detector*; turn detected blindspots into *correctly-labeled training data*.

1. **Confirm the class at scale.** A handful of corpus positions (from lost
   games) is a hypothesis, not a law. The Coda-vs-SF gauntlet corpus (in
   flight) tests whether the bishop-sortie / pawn-push themes recur across
   many losses. Only intervene on a class that shows up repeatedly.

2. **Generate the position TYPE in volume, and FORCE entry into the sparse
   region.** A few dozen exact positions won't move a net trained on billions;
   you need hundreds–thousands of the *type*. The key difficulty: these
   positions are sparse *precisely because* strong play avoids them, so normal
   strong-vs-strong games won't produce them. You have to deliberately steer
   in. Options (combine):
   - **Forced-move openings**: from each corpus position (or a few plies
     before), *play the bad sortie* and continue with strong play — generates
     the refutation positions that LC0/SF games skip. The point is to densely
     populate the region the net is blind to.
   - **Perturbation**: material/structure variations around the seeds (e.g.
     `coda datagen` material-removal) to broaden the motif without leaving it.
   - **Structural mining**: scan existing T80 binpacks for positions matching
     the signature (side-to-move has an active bishop sortie) — they exist but
     are under-weighted; re-surfacing them is cheap.

3. **Label with the SAME pipeline as the base data: LC0 or SF.** This is the
   one part that was already right — the targeted positions must be scored by
   LC0/SF so they're consistent in kind with the T80 base (same eval scale,
   same WDL convention) and so they carry the *correct* ranking (we confirmed
   deep SF ranks the sortie correctly). The earlier draft wrongly framed this
   as "don't let Coda re-label its own self-play" — Coda has no self-play in
   the training loop; the only requirement is that the supplementary
   positions are LC0/SF-scored like everything else. The novelty is purely
   *which positions* get added (dense in the blind region), not how they're
   labeled.

4. **Mix a modest fraction into the next run.** Too much targeted data distorts
   the net / overfits the motif; too little does nothing. Either blend a few %
   into the main training mix, or add a short **fine-tune stage** (low LR, a
   few SBs) on the targeted set after the main run. The supplementary set is an
   *addition* to the shared T80/LC0 base, not a replacement — keep the broad
   distribution, add density only where the net is blind.

5. **Verify cheaply, then SPRT.** Static/shallow eval of the retrained net on
   `testdata/overrate.epd` should now rank `bm` over `am` on the targeted
   positions (the cheap inner loop). Then SPRT the new net vs prod to confirm
   net-positive overall — targeted data can fix one motif while hurting
   elsewhere, so the EPD passing is necessary, not sufficient.

## Coda-infra specifics

- The supplementary set is generated to force entry into the sparse region
  (play the bad sortie, continue with strong play) and **scored by LC0/SF**,
  matching the base T80 data exactly — same format (SF BINP binpack), same
  scoring source. No Coda-scored data enters the training mix.
- Training: Bullet (`adamtwiss/bullet`), `coda convert-bullet` → `.nnue`.
- Whether this beats simply training a bigger/better-converged net on the
  existing data (the L1=32 direction) is an empirical question — the targeted-
  data route is the cheaper, more surgical test, and the EPD suite makes its
  effect directly measurable.

## The general framing

This isn't a one-off bishop-sortie patch. The loop — *mine real-game blindspots
(overrate corpus) → generate + strongly-label the type → mix into training →
verify on the EPD + SPRT* — is a reusable strength engine that converts
deploy/gauntlet losses directly into eval improvements. The bishop-sortie class
is just the first customer. Each turn of the loop both fixes a blindspot and
refreshes the corpus with whatever the improved net is *now* worst at.
