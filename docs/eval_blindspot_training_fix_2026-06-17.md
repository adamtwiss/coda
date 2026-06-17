# Fixing eval blindspots via the training corpus (NNUE)

**Context.** The overrate corpus (`docs/overrate_corpus_thread_2026-06-15.md`)
found that ~60% of current trunk's real-game LIVE blindspots **persist at
depth** — the static NNUE eval mis-ranks the move, more search doesn't help.
Concrete recurring class: **over-valued active bishop sorties** (Bxf4, Bd6,
Bf3, Bb3, Bd2, Bd4+) and some loosening pawn pushes (f5/f6). In an NNUE world
you don't fix this by hand-tuning a term — you change what the net learns.
This note is the methodology.

## Why an NNUE has a persistent blindspot

The net is an interpolator over its training distribution: accurate where
(position, label) pairs are **dense and correctly labeled**, wrong where they
are **sparse or mislabeled**. A persistent over-valuation of a move type means
one (or more) of:

1. **Sparse coverage (most likely).** Bishop sorties that are *bad* are
   off-the-beaten-path of strong play — strong engines/players rarely enter
   them, so the post-sortie positions barely appear in T80 (Lc0) data. The net
   never saw the refutation, so it extrapolates and guesses high. Blindspots
   live exactly where the training games don't go.
2. **Self-play echo chamber.** If the net's *own* (blind) eval generates
   training data, both sides allow the sortie and the resulting positions get
   labeled with the net's own too-rosy number → the blindspot is reinforced,
   never corrected. Coda's bulk data is T80/Lc0 (good), but supplementary
   `coda datagen` self-play is labeled by **Coda search at depth 8** (see
   `coda datagen` default) — shallow enough to share the effective-depth
   blindspot. Any of that in the mix teaches the wrong ranking.
3. **Shallow labels.** Same root as (2): a label is only as good as the search
   that produced it. Depth-8 self-play labels on tactically-deep positions are
   systematically wrong in the engine's blind direction.

## The fix: targeted hard-example mining → re-label → mix → verify

This is the eval-flywheel made deliberate. The overrate corpus is the
*detector*; turn detected blindspots into *correctly-labeled training data*.

1. **Confirm the class at scale.** A handful of corpus positions (from lost
   games) is a hypothesis, not a law. The Coda-vs-SF gauntlet corpus (in
   flight) tests whether the bishop-sortie / pawn-push themes recur across
   many losses. Only intervene on a class that shows up repeatedly.

2. **Generate the position TYPE in volume.** A few dozen exact positions won't
   move a net trained on billions; you need hundreds–thousands of the *type*.
   Options (combine):
   - **Seeded self-play**: start games a few plies *before* each corpus
     position (or from the games' openings/structures) and play out — surfaces
     many positions where the bad sortie is available and gets refuted.
   - **Perturbation**: `coda datagen` material-removal around the seeds to
     vary material/structure while keeping the motif.
   - **Structural mining**: scan existing T80 binpacks for positions matching
     the signature (side-to-move has an active bishop sortie) — they exist but
     are under-weighted; re-surfacing them is cheap.

3. **Re-label with a STRONG, blindspot-free engine — this is the crux.** The
   blindspot exists because labels were sparse/shallow; the cure is *correct*
   labels. Label the targeted set with **deep Stockfish** (or deep Coda at
   d24+, or Lc0) — anything that ranks the sortie correctly (we confirmed deep
   SF does). **Do NOT re-label with Coda's depth-8 self-play** — that re-teaches
   the blindspot. If you generate positions via Coda self-play (fine, to hit
   the on-distribution-for-Coda blind positions), the *labels* must still come
   from the strong source. Break the echo at the label, not the generator.

4. **Mix a modest fraction into the next run.** Too much targeted data distorts
   the net / overfits the motif; too little does nothing. Either blend a few %
   into the main training mix, or add a short **fine-tune stage** (low LR, a
   few SBs) on the targeted set after the main run. WDL is a useful blindspot-
   free signal here (the sortie's game outcome is "lost"), so these positions
   benefit from outcome labels too — but a sharp *eval* re-label is more
   targeted than globally bumping WDL weight (already tuned to w0.15 for v9).

5. **Verify cheaply, then SPRT.** Static/shallow eval of the retrained net on
   `testdata/overrate.epd` should now rank `bm` over `am` on the targeted
   positions (the cheap inner loop). Then SPRT the new net vs prod to confirm
   net-positive overall — targeted data can fix one motif while hurting
   elsewhere, so the EPD passing is necessary, not sufficient.

## Coda-infra specifics

- Generation/labeling: `coda datagen` (self-play + material-removal,
  binpack out) — but **raise its label depth well above 8** for this purpose,
  or label with SF via a small adapter. Shallow labels are the enemy here.
- Training: Bullet (`adamtwiss/bullet`), `coda convert-bullet` → `.nnue`.
- The targeted set is a *supplement* to T80, not a replacement — keep the
  broad distribution, add density where the net is blind.

## The general framing

This isn't a one-off bishop-sortie patch. The loop — *mine real-game blindspots
(overrate corpus) → generate + strongly-label the type → mix into training →
verify on the EPD + SPRT* — is a reusable strength engine that converts
deploy/gauntlet losses directly into eval improvements. The bishop-sortie class
is just the first customer. Each turn of the loop both fixes a blindspot and
refreshes the corpus with whatever the improved net is *now* worst at.
