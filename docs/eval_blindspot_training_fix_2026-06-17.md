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
  weighted (retrain + revalidate).
  **RESOLVED (2026-06-17, `dump-threats`):** the incoming "bishop is attacked"
  features DO fire after each sortie (`w Bf4→b Bd2`, `b Be7→w Bd6`, `b Re3→w
  Bf3`) — so it's **(a), a weight/training-emphasis gap, not a coverage gap.**
  Sharper: on the *pure* sorties (Bf3, Bd2) the bishop has ~0 *outgoing* threats
  (Bf3 attacks no enemy piece), so the over-credit is the net threat-*delta of
  relocating* the bishop (collateral line shifts + under-penalized incoming)
  coming out positive, NOT "active bishop attacks more." Fix = targeted
  SF/LC0-labeled forced-sortie data, dense enough to re-teach the *threat*
  weights on this sparse class; no architecture change. (Next probe to fully
  pinpoint: per-feature threat-contribution decomposition — is the incoming
  feature mis-signed, or is it collateral threats dominating?)
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

---

## CONFIRMED 2026-06-21: ply-skipping reduces the bishop blindspot (mechanism)

> **⚠️ SUPERSEDED 2026-06-24 — this "confirmation" was small-N noise.** The
> result below rests on a 5–6-position forward-bishop subset (per-position SE
> ~150cp). Re-tested on a 119-position mined corpus
> (`testdata/wandering_bishop_corpus.epd`), the soft-ply v6-s4→v7-s4 effect is
> **+6 ± 9cp — within noise** (the −68cp it shows on the old n=6 subset does
> NOT generalize). See the "## RE-TESTED 2026-06-24 at n=119" section at the
> bottom. Keep this section for history; **do not cite ply-skip as a
> blindspot-reducing lever** — only the relabeled (stamped) corpus survives the
> larger N.

`blindspot_eval` on the two multi-stage s4 nets — **v7-s4 (with soft-ply f25)
vs v6-s4 (no ply)**, same arch (L1=32 v9), same multistage point — moved the
bishop metrics the *right way on every axis*:

| metric (s4 nets) | v6-s4 (no ply) | v7-s4 (ply f25) |
|---|---|---|
| overrate.epd pass | 2/20 (10%) | **5/20 (25%)** |
| mean sortie over-pref vs SF | +109cp | **+66cp** |
| sorties now correct | 6/17 | **7/17** |
| forward-bishop net_pref (n=5) | +327cp | **+270cp** (SF ~+65) |

**This confirms the root-cause hypothesis: the wandering-bishop over-scoring is a
TRAINING-DATA artifact**, not an architectural limitation. The over-represented
early-opening positions are where the forward-bishop over-scoring gets
reinforced; **soft-ply (a pure data-*reshaping* lever, down-weighting those
positions) measurably pulls the bias back** — no relabeling, no architecture
change. Validates the "training-set issue" read over any "net can't represent it"
alternative, and it's an *independent* (eval-side, no-search, no-tunables) reason
soft-ply is net-positive, on top of its two clean +15/+16 SPRTs (S200 #2082,
S800 #2119).

### Two complementary levers on the SAME root cause
1. **Reshaping** (soft-ply) — cheap, confirmed, **partial** (+327→+270, still far
   from SF's +65). Down-weight the positions that reinforce the bias.
2. **Relabeling** (SF-rescore / sentinel-stamp corpus, in progress) — the bigger
   hammer: replace Coda's *wrong* bishop evals on Coda-to-move positions with
   SF's correct ones, so we stop *teaching* the bias at all.

Both attack the same mechanism (training data over-represents / mis-labels the
positions where the bias lives). Soft-ply moving the metric proves the lever
class works; the relabeled corpus should move it *further* toward SF's +65.

### A validated yardstick
The operational win: `blindspot_eval`'s **forward-bishop net_pref** is now a
*demonstrated-sensitive* metric — it moved under one lever (327→270). So it's a
clean before/after gauge for the relabeling validation: train on the stamped
corpus, measure whether forward-bishop net_pref drops further toward SF's +65 —
no full SPRT needed to see if the eval moved the right way.

(Caveats: mid-training s4 nets, not final; forward-bishop n=5. But the same-
direction move across all four metrics makes the direction robust.)

---

## RE-TESTED 2026-06-24 at n=119: only the relabeled corpus survives

The original blindspot metrics (this doc + the n=6 `wbench`/`blindspot_eval`
forward-bishop subset) rest on **5–6 hand-picked positions** with per-position
net_pref SD ~130–180cp → SE ~150cp at n=6 / ~100cp paired. That is far too
noisy to separate a real ~40cp eval shift from chance — multiple "confirmed"
reductions below turned out to be noise.

**New corpus.** `testdata/wandering_bishop_corpus.epd` — **119** SF-refuted,
quiet, positional forward-bishop sorties mined from ~137k Coda-vs-SF datagen
games (OB datagen 2061/2062/2094) via the pipeline:
`scripts/bishop_sortie_prefilter.py` (cheap, engine-disagreement-gated) →
`scripts/bishop_sortie_validate.py` (deep-SF bm + move_loss 100–600, net_pref
gate) → `scripts/mine_bishop_corpus.sh` (parallel orchestration). Score any
net with `scripts/net_pref_score.py --epd testdata/wandering_bishop_corpus.epd`.
Paired SE drops to **~7–16cp** (≈10× tighter than n=6), making the metric a
real net-discriminator.

**Re-test (paired net_pref delta, baseline − treatment; >0 = treatment reduces
the over-credit):**

| lever | nets | n=119 paired Δ | verdict | prior small-N claim |
|---|---|---|---|---|
| **SF-relabel (stamped) — full** | bishop-A-baseline − B-stamped | **+39 ± 16cp** | ✅ **real reduction** | held up |
| **SF-relabel (stamped) — 2%** | bishop-A-2pct − B-2pct-stamped | **+43 ± 10cp** | ✅ **real, ~4σ** | held up |
| Soft-ply f25 (multistage) | multi-v6-s4 − multi-v7-s4 | +6 ± 9cp | ❌ null | "327→270 CONFIRMED" (n=5) |
| Soft-ply dose (S200) | baseline +82 vs f10–f40 +91…+105 | none (ply ≥ baseline) | ❌ null/worse | optimum claimed at f25 |
| v7-s5 newer data | v7-s5 − v7-s5-ws-newer | −13 ± 7cp | ❌ no reduction (marginally worse) | "significantly reduced" |
| QAT | qatpair-baseline − qatpair-qat | −0 ± 9cp | ❌ null (expected; quant lever) | — |

**Conclusion.** Of every lever claimed to dent the wandering-bishop blindspot,
**only the SF-relabeled (stamped) supplementary data robustly reduces it**
(~40cp, ~35–40% of the over-credit, significant at both dilution levels). Soft-
ply, newer-data, and QAT all come back null at n=119 — their earlier
"reductions" lived inside the ≤6-position noise floor. This both (a) validates
the relabel lever this doc advocates and (b) is a cautionary tale: **never
declare a blindspot lever confirmed on the n=6 subset** — use the 119-corpus.

**Caveat (corpus selection).** The 119-corpus was gated on the *prod* net
(035195DB) over-crediting (net_pref ≥ 0), so absolute levels for prod-like nets
are selection-inflated and not cross-comparable to dissimilar nets. Every Δ
above is **within-arm** (both nets non-gate, same fixed positions), so the
*deltas* are unbiased; but a fully clean absolute read — and an airtight null
for nets unlike prod — wants an **ungated** rebuild (all SF-refuted quiet
sorties regardless of any net's view) or a union-gate across candidate nets.
Cheap to regenerate via `mine_bishop_corpus.sh` (drop `--coda-net`).

---

## PLAN 2026-06-24: corrective-data scaling + binpack skip-flag (compression)

Two linked workstreams, set down so we don't lose the thread.

### A. Does *more* corrective data help? (the volume curve)
The dilution test (2% vs 6.5%) varied the **T80 base count**, holding the
**corrective set fixed at 230M** — so it tested dilution *rate*, NOT corrective
*volume*. Volume is untested. Prior (Adam + the sparse-region-density argument):
more independent corrective points should help.

**In flight (GPU2, S200):** `bishop-B2-halfstamped` = 1 T80 file (jan) + **half**
the stamped shards (even 0–14 of 2062+2094 ≈ 115M corrective pts, ~3.2% dilution,
in the tested band). Compare net_pref on `wandering_bishop_corpus.epd` vs:
- `bishop-A-baseline` (1 file, no stamped) — baseline
- `bishop-B-stamped` (1 file, full 230M) — full corrective
Read: B2≈B ⟹ past the knee (volume saturated); B2≪B reduction ⟹ curve still
climbing ⟹ generating more is justified. (Single-run; trust a clear gap. A 3rd
point at 57M sharpens curvature.)

**Two distinct sizing questions:**
1. *Push past 40%* at small scale — needs the curve shape (≥3 points); residual
   floor after full-B is still +54cp (likely the threat-weight ceiling).
2. *Hold 40% at production* — arithmetic: 40 T80 files ≈ 40× base ⟹ ~40× the
   corrective data (~9.2B positions) just to keep 6.5% dilution. This is the
   real generation target regardless of (1). We can generate 2–4× more on the
   fleet as a low-throughput job over a few days.

### B. Binpack skip-flag (compression fix) — enables the scale-up

> **⚠️ SUPERSEDED 2026-06-25 — the bloat was a different bug; the skip-flag is
> NOT needed.** Direct measurement showed the sentinel oscillation costs only
> ~10–20% (1.6 B/pos). The real ~2× bloat was **chain fragmentation**: the
> stamper re-derived each position via `from_fen`, whose halfmove (rule50) clock
> disagreed with sfbinpack's `after_move`, breaking `is_continuation` ~13% of
> plies → ~7-position chains instead of ~100 → the 32-byte stem paid ~14×/game.
> Fixed in `coda-stamp` (main `7c8e971`) by chaining positions via `after_move`:
> 7.61 → 3.54 B/pos (2.15×), content-identical; the stamped corpus was
> re-stamped (3.4G → 1.56G). The raw-score skip-flag below is therefore moot —
> keep this section only as the record of how we got here. Subsection A
> (corrective-data *volume*) stands.

**Problem (originally hypothesised):** stamping Coda-to-move positions with the
32000 sentinel makes the per-position score stream oscillate `[real, 32000,
real, …]`, defeating the chain's score-delta coding: **7.4 B/pos** (14.8 per
*trainable* pos) vs ~2 B/pos for raw-scored T80. (This framing was wrong — see
the superseded banner above; the gap was chain fragmentation, not the score.)

**Fix:** stop stamping. Keep **raw smooth scores** (both engines' real PGN evals
compress normally) and skip Coda-to-move positions at *train time* via a
**per-chain marker + reader parity-skip**.

**Why per-chain (not per-position):** the binpack stores explicit ply/result
**only on the chain stem**; continuations are pure movetext (move + score-delta).
The score is the *only* per-position field — which is why the sentinel had to
live in the score. So a smooth-score marker must be on the stem and the reader
reconstructs the skip per position by side-to-move parity.

**Marker encoding (tooling-safe, backward-compatible):** the `PlyResult` u16 is
`result:2 (bits 14–15) | ply:14 (bits 0–13)`. `result` is full (2 bits), but real
ply ≪ 4096, so **bits 12–13 of the ply field are spare**. Use a **2-bit skip-code
there**: `0 = none` (⟹ all existing T80/old-stamped data is unchanged — the
load-bearing back-compat property), `1 = skip white-to-move`, `2 = skip
black-to-move`. Writer sets it from Coda's color; reader masks `ply &= 0x0FFF`
before any ply use (e.g. soft-early-ply) and applies the skip. (Assumes stem ply
< 4096 — assert in the writer.)

**Four pieces + files:**
1. **Writer** — `tools/src/stamp.rs`: add `--skip-marker` mode: keep raw evals for
   *both* movers (Coda's eval is in the PGN too), and set the 2-bit skip-code on
   the **first position of each game** (= the chain stem), `1` if Coda is White
   else `2`. Default (sentinel) path unchanged. *[STARTED — see branch.]*
2. **Reader** — `crates/bullet_lib/src/value/loader/sfbinpack.rs` (+ the forked
   `sfbinpack` crate): track chain boundaries, read the stem's skip-code, hold it
   for the chain, and in the per-position `filter` skip entries where
   `side_to_move == coda_color`. Mask the marker out of ply before use.
3. **Regenerate** binpacks from raw `dg*.pgn.bz2` with `--skip-marker` (no SF
   search, pure CPU). ~2 B/pos.
4. **Round-trip test** (gate before any training use): write a small marked
   binpack → read it back → assert (a) ~half the positions are skipped, (b) the
   *right* half (Coda-to-move), (c) scores are smooth (no sentinels), (d) ply
   decodes correctly after masking, (e) old (code-0) binpacks are byte-for-byte
   unaffected.

**Sequencing:** B2 result → if volume helps, fit the curve → size generation →
land the skip-flag (small, but touches the forked sfbinpack crate) → generate at
scale with cheap compression.
