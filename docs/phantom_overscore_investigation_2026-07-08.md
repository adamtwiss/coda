# Phantom Overscore Investigation (2026-07-08)

**Status: cause localized, mechanism open.** Coda draws games it "thinks"
it is winning because the **network's static eval over-values a specific
class of position** (materially-level, queens-on middlegames) by ~0.8–0.9
pawns. Search, correction history, conversion technique, WDL calibration,
capacity, and threat features are all **ruled out** as the cause. The open
question — *why does the net over-value these positions* — is not yet
answered.

This doc consolidates the diagnostic chain so it survives for when we pick
the "why" up (esp. once GPU4 / the 1B overscore corpus is back).

---

## 1. The symptom: an unbeatable engine that draws everything

At long TC (60+1 local RR, 20-engine pool) Coda posted **30 draws in 30
games** — literally nobody beat us, and we beat nobody, landing mid-table
at 50%. The same signature shows on lichess (very high draw rate, rating
bled by *unconverted* wins, not losses). Rank in an RR comes from farming
the tail; an engine that draws everyone scores 50% regardless of how
unbeatable it is, so "unbeaten, mid-table" is arithmetic, not paradox.

The question this raised: are these draws **failures to convert real
edges**, or something else?

## 2. Phantom edges: we were never actually ahead

**Test:** in drawn LTC games where Coda's own eval peaked ≥ +1.0, take the
position at that peak and compare three evals — ours, the **opponent's**
(from the same PGN), and **Stockfish as a fresh arbiter**.

**Result (n=275 drawn LTC games, opponent; 60-game SF subsample):**

| bucket | n | our peak (median) | opponent (median) | SF arbiter |
|---|---|---|---|---|
| **DRAWN**, we saw ≥ +1.0 | 275 | +1.45 | **+0.40** | ~refuted 86% |
| **DRAWN**, we saw ≥ +1.5 | 129 | +2.12 | **+0.61** | — |
| **WON**, we saw ≥ +1.0 (control) | 54 | +9.86 | **+5.66** | confirms |

The won-games control is the clincher: **the arbiter's absolute eval
tracks the game outcome, not our eval.** When we drew, the position really
was level (~+0.4) at the exact moment we thought we were up; when we won,
it really was winning (+5.66). Our eval said "ahead" in both. So the draws
are **phantom edges** — positions we over-value, not edges we failed to
convert. (Initial 9/9 hand-check, then widened to 275.)

## 3. Structural signature: level queens-on middlegames

Profiling the 275 phantom positions vs the 55 real-win positions — they
are structurally **opposite**:

| | Phantom draws (n=275) | Real wins (n=55) |
|---|---|---|
| Game phase | **middlegame** (median phase 50) | endgame (phase 13, 76%) |
| Queens on board | **77%** | 30% |
| Material | **dead level** (median +0; 81% level-or-down) | up +2 (83% ahead) |
| OCB / fortress | 4% / rare | rare |

**Corpus decomposition (2026-07-08, heldout overrate set, 7845 pos, vs
LC0 truth) — TWO effects.** Signed static error (coda − LC0) bucketed by
our material:

| our material | n | mean overscore | median LC0 truth |
|---|---|---|---|
| down (≤−1) | 2728 (35%) | +45 | −257 |
| level (0..+1) | 4301 (55%) | +31 | +41 |
| up exch (+2/+3) | 747 (10%) | **+111** | +65 |
| up piece+ (≥+4) | 69 | **+131** | +21 |

1. **Pervasive optimism (~+30–45 cp) at *all* material levels** — we
   overscore even when *down* material (+45, truth −257). This is the bulk
   (90% level-or-down) and it is the mechanism behind the phantom *draws*
   (level-material queens-on middlegames).
2. **Material overvaluation in the up-material tail** — overscore triples
   (+110–130 cp) when up an exchange/piece; ~⅓ of those are LC0 ≤ 0
   (compensation blindness — we count the material, miss the
   attack/initiative/passed pawns). Worst individual overscores, ~10% of
   cases.

Tested and NULL as the driver: threat-feature *count* (phantoms have
*fewer* threats than matched controls, 49 vs 56); king-as-attacker
threats (rare in middlegames); slider/bishop *mobility* (corr with error
≈ 0; raw mobility ≠ the "wandering-bishop illusion", so bishop
overvaluation isn't refuted, just not linear-in-mobility). The dominant
effect is a **pervasive positive eval bias**, worst in complex/up-material
positions — not a single clean hand-computable feature.

The phantom class is **not** fortresses, opposite-coloured bishops, or
endgame technique. It is complex, materially-level, queens-on middlegames
where our eval hallucinates ~+1.5 out of nothing.

## 4. Localization: it is the NET's STATIC eval, not search

**Static-vs-search 2×2** on 79 phantom positions:

| | Coda | SF | truth (outcome) |
|---|---|---|---|
| **static** (no search) | **+1.28** | +0.44 | — |
| **search** (2 s) | ~+1.00 | ~+0.40 | +0.48 |

The overscore is entirely in the **raw net's static eval** (+1.28 vs SF's
+0.44; the +0.84 gap matches the true ~+0.48 outcome). **Search does not
amplify it — search *corrects* it** (pulls +1.28 down to +1.0). An earlier
"search amplifies to +2" reading was an artifact of taking the *max over
all game plies* (a survivorship statistic) rather than a typical value.

Mechanism: the net statically over-values this position class by ~0.8–0.9
pawns; a game's "peak" is simply where search reached the local maximum of
that already-inflated eval → phantom edge → repetition draw.

## 5. What we ruled out

| Hypothesis | Test | Verdict |
|---|---|---|
| **Adjudication artifact** (our honest 0.00s trigger early draw adjudication) | Termination taxonomy of the 60+1 draws | **Ruled out** — our draws adjudicate at the *pool* rate (34–35%); mostly 3-fold repetition, in *longer*-than-average games. |
| **Conversion / endgame technique** | won-vs-drawn arbiter control (§2) | **Ruled out** — we don't fail to convert; we were never ahead. |
| **Correction-history feedback loop** (as in the fortress-drift bug) | `NO_CORRECTION` on/off, 80 phantom FENs, 2 s search | **Ruled out** — eval ON +0.96 vs OFF +1.02 (drop +0.02; 0/80 dropped >0.5). Corrhist was slightly *helping*. Opposite of the fortress case (where OFF → 0.00). Note the fortress fix's piece-count damping only guards low material, so these high-material positions were genuinely un-damped — yet corrhist still isn't the driver. |
| **Search amplification / TT out-of-bounds** | static-vs-search 2×2 (§4) | **Ruled out** — search moves the number the *right* direction (down). No need to disable TT features. |
| **WDL calibration** (ours 0.20 vs SF 0.26) | w20 (prod) vs w24 matched pair (`multi-v9-s3-swa` vs `multi-v9-w24-s3-swa`) | **Ruled out as *the* fix** — biased screen +1.33→+1.23 (10 cp, still overscores); neutral `net_report` general eval identical (Spearman .849/.848), w24 tail marginally *worse*; blindspot overscore +44.2→+35.9 (~8 cp). WDL shaves ~8–10 cp of an ~86 cp gap at a hair of eval-quality cost. It is a robust *representational* property, not a cp-vs-outcome calibration knob. |
| **Capacity** (we assumed SF's net was bigger) | `docs/sfnnv13_architecture_review_2026-05-23.md` | **Ruled out — and the premise was false.** SF and Coda are near-identical in size; with Coda now at FT=1024 / L1=32 we *match* SF (FT 1024, first-dense 32, L2 32). No capacity gap. |
| **Threat features being a Coda-unique bias** | same review doc | **Ruled out** — SF has FullThreats too (since SFNNv12, Feb 2026), yet SF does *not* overscore these positions. Threats are not the differentiator. |

Data is also ~identical (both train on LC0 T80; Coda uses the full SF set
for prod builds). So: **same data, matched architecture size, both have
threats — yet we overscore these positions and SF does not.**

## 6. What remains — candidate causes of the "why"

The difference between our net and SF's on these positions is **not size,
data, threats, WDL, corrhist, or search.** It is in the harder-to-change
recipe/architecture details where we actually differ from SF
(`sfnnv13_architecture_review`):

**PSQT output buckets & dual activation — TESTED (S800, 2026-06-16),
regressed then, on the REVISIT list (not dead).** SF has both; Coda has
neither. Matched-triplet S800 test (`project_psqt_dual_regress_s800`):
- **PSQT** #2037 H0 −12.5. Decomposition: eval quality **matched**
  baseline (Spearman-vs-LC0 0.884 vs 0.880) — the skip-connection did
  **not** improve position ranking; it regressed only via a ~12% search
  NPS tax. So on that recipe PSQT would **not** have fixed the phantom
  overscore (it didn't move eval quality either way).
- **Dual** #2036 H0 −27.9. Eval quality **degraded** (Spearman 0.863).
- Combined #2043 H0 −104 (scale incompatibility, not additive).

**BUT (Adam, 2026-06-30 + 2026-07-08): "don't re-propose" is too strong.**
Those tests ran under a recipe now known-miscalibrated vs SF (ours mse
**3.0** / wdl **0.20** vs SF's verified **~2.44 / 0.26**), and SF makes
PSQT+dual work on a **near-identical architecture with similar data** — so
the mechanism demonstrably transfers; one stale-recipe regression closes
nothing. They are worth revisiting **under the current recipe**, with the
caveat that PSQT specifically *matched* (didn't improve) eval quality when
tested, so it's a weaker phantom-fix candidate than dual on that evidence.
NB: PSQT inference was never merged to main (nets are v11 / branch-only),
so a revisit needs that path rebuilt first.

- **Corrective data at scale.** The 1B+ overscore corpus (filtered from
  the ~200B SF set for our-eval-vs-SF/LC0 divergence, currently on the
  offline GPU4). Prior small-scale corrective tests gave ~10–20 cp — same
  order as every other cheap lever; scale is the untested variable.
- **Other recipe (untested for phantom):** MSE exponent **ours 3.0 vs SF
  ~2.44** (we weight large errors *more*), eval scale ×600 vs ×400, loss
  specifics (qp-asymmetry, in/out scaling/offset in SF's threats.yaml),
  SWA/schedule details.

**Every cheap lever tested gives ~10–20 cp of an ~86 cp gap** (WDL,
small-scale corrective). PSQT/dual regressed on the *old* recipe. No knob
tried so far fixes it; the live candidates are corrective data at scale,
a PSQT/dual revisit under the current recipe, and the recipe axes
(esp. MSE exponent) — none yet tested against the phantom metric.

**Context — the noise floor.** SF *also* overscores a comparable *count*
of positions vs LC0; some fraction of any net's overscore tail is
irreducible label noise. But SF does *not* overscore *our* phantom
positions (SF static +0.44 on them), so this class is specifically
hard-for-Coda, not universally noisy — it is a real Coda-specific gap, not
just the noise floor.

## 7. Why it matters (value framing)

The phantom overscore converts what should be **wins into draws**,
concentrated against the weaker/equal field (the 30/30 RR wall, lichess
drawishness) — real Elo, but field-facing, not the top-table gap. It does
*not* cost much against SF/Reckless (those draw/lose regardless).
Critically: **no search or tune Elo converts against the field while this
stands**, because search operates correctly on a mis-calibrated eval — so
this is a distinct net-research track, run in parallel with (not blocking)
search work.

## 8. Next diagnostic for the "why"

To find *what* the net over-values (feature attribution), the cheap next
step is a **feature/ablation decomposition on the phantom class**: does the
static overscore correlate with a specific signal (threat-feature count,
king-safety proxy, mobility, space) across positions? A coherent single
driver → architecture/representation gap (points at dual-activation / PSQT
/ a specific over-weighted feature); a diffuse driver → data-coverage gap
(points at corrective data at scale). This is characterizable from the
phantom FENs we already have, no GPU4 needed.

## 9. Instruments built (reusable)

- **Phantom-edge audit** — peak-eval vs opponent/SF-arbiter refutation
  rate, stratified by outcome. The before/after metric for candidate nets.
- **Structural profiler** — phase / material / OCB / queens / locked-pawn
  signature of a position set.
- **Static-vs-search split** — `coda eval-fens` (static) vs `go` (search)
  vs SF, the net-vs-search localizer.
- **`net_report.py`** — neutral overscore vs LC0 (signed overscore + p90/
  p99/max tail) on a general binpack; the unbiased net-quality readout.

Scratch scripts on Hercules: `phantom_wide.py`, `corr_test.py`; phantom
FEN set `phantom_fens.tsv`.

## 10. Update (2026-07-08 eve) — learnable, not a floor; corrective test running

**Key correction to §6-7's "near-floor" pessimism.** The blindspot harvest
filter is `|coda_static − lc0_dynamic| > X` AND **SF-static also disagrees
with Coda** (`coda_err − sf_err ≥ ~80`). The SF-*static* clause is
load-bearing: it excludes tactical positions (static can't see, search
can) and keeps only positions a good STATIC eval gets right — SF proves
it by nailing them without search. So these are **learnable static-eval
errors, not intrinsic difficulty and not tactics.** The ~208 mean|err|
"floor" across our nets is therefore a coverage/training gap, not a
capacity wall — corrective data CAN work with the right, directly-targeted
positions.

**Two corrective datasets, distinguished:**
- **Blindspot harvest** (directly-targeted): T80 positions filtered by the
  exact criterion above, native LC0 labels = correct corrective signal.
  6 files (~10GB, jan-jun 2024, "150_80") on **gpu3:/workspace/blindspot**
  (an earlier harvest survived; the ~1B full harvest is on offline GPU4).
- **Stamped SF-vs-Coda** (proxy): Coda-vs-SF game positions, SF-labeled;
  over-samples overscore-*adjacent* but the trainable half is SF's replies,
  not the filtered error positions. Less directly targeted.

**Running now (gpu3, S200, 4×T80, ms10000, seed 42, identical recipe):**
3-arm matched test — `exp-base-4t80` / `exp-blindspot-4t80` (+6 blindspot
files) / `exp-stamped-4t80` (+stamped). Readout: net_report blindspot
mean|err| (headline) + wandering-bishop (independent gauge). Prior s1
evidence: stamped helps WB, hurts broad blindspot — but s1 was undertrained
special-data; this clean bake + the directly-targeted blindspot arm is the
real test. If a corrective arm moves broad mean|err| without wrecking
Spearman → the fix is validated and the GPU4 full harvest at scale is the
path. Whichever way eval-metrics land, GAMES are the ultimate arbiter
(does it convert phantom draws to wins).

**Cross-net context (why no silver bullet):** across v8/v9/w24/dual/ranger/
stamped/ms32, broad blindspot mean|err| is pinned ~208-227; nothing breaks
below ~208. Wandering-bishop (independent) DOES move (18→34 correct) under
corrective/dual/ms32. Interventions that cut directional overscore do it by
FLATTENING (dual: overscore +22 but Spearman 0.785, mean|err| 215) not by
getting positions right; only Ranger nudges mean|err| down (~4cp) at no
quality cost. Depth helps modestly (s1 218 → prod-multistage 208).

## 11. Q3 ANSWERED (prior-Claude symmetric mirror) — near-symmetric, NOT a Coda defect

**Corrects §5/§7's "Coda-specific gap" framing.** The blindspot filter only
ever pointed at *our* errors (Coda ≥150 off, SF closer), so it *structurally
cannot* show SF's own blind spots. The symmetric mirror scan (100k unbiased
jun-2024 sample, calibrated per-net scales) counts both directions:

| population | rate |
|---|---|
| Coda-specific (Coda ≥150 off, SF closer by 80) | **2.59%** |
| SF-specific (SF ≥150 off, Coda closer by 80) | **2.36%** |
| shared (both ≥150 off) | **8.57%** |
| mean\|err\| | Coda 65.4 vs SF **65.0** |
| p50/p90/p99 | 37/168/352 vs 37/165/353 |

**SF has blind spots at essentially our rate.** Error distributions are
indistinguishable at every quantile. Our idiosyncratic excess is ~10%
relative (2.59 vs 2.36), not a structural defect; the biggest bucket is the
*shared* 8.57% ("dynamics not statically fittable"), hitting both nets
equally.

**Why ours are visible and SF's aren't:** (a) we only harvest our own
direction; (b) **adversarial self-sampling** — in games each engine's search
steers into positions *it* overrates, so our tail surfaces in our games and
SF's in SF's. Same mechanism as §4, now shown symmetric across engines.

**Consequences for the whole investigation:**
- **Ceiling is modest.** A perfect fix moves our idiosyncratic tail 2.59% →
  ~2.36% (SF's floor); the shared 8.57% is common to all engines and likely
  statically irreducible (needs search/dynamics). This is a *refinement*,
  not the big Elo lever — SF has the same tail, so it is NOT why SF beats us.
- **But self-sampling gives leverage.** Search lands us in our *own* tail, so
  shrinking it reduces the positions our games reach — modest tail reduction,
  possibly outsized game effect. Worth pursuing, calibrated as refinement.
- **Corrective data can't teach wrong values** (labels are LC0 ground truth);
  the only failure mode is dilution/redistribution.
- **THE READOUT = the symmetric mirror metric**, not net_report alone: on a
  candidate net, coda-rate ↓ toward ~2.0% *without* sf-rate rising = genuine
  fix; both moving = redistribution. Cleaner cause-vs-symptom test than
  general Spearman.
- **The ~0.2pp asymmetry** is where WDL 0.26 + WRM might shave a little — a
  refinement, not a root-cause hunt.

Scale (Q2) remains the one open lever: the full ~436B harvest (GPU4) +
mirror metric will show whether coda-rate keeps dropping with corrective
volume or plateaus. The running gpu3 3-arm S200 test is a cheap preview
(6 stale-old-net blindspot files, LC0-labelled — labels valid, positions
are an older net's tail so directionally-not-perfectly-matched).

## 12. Correction to §11's "modest ceiling" — self-sampling makes this the primary lever

§11 read the symmetric *rate* (2.59 vs 2.36) as "modest ceiling / refinement."
That understates it, and Adam pushed back correctly (2026-07-08): the rate is
symmetric but the **game impact is asymmetric toward fixing OURS**, because of
adversarial self-sampling. Our search preferentially steers into positions
*we* mis-score, every game — so the fraction of *outcome-critical* positions
affected is far above the 2.59% base rate (275 LTC drawn games trace to it).
The damage is also *upstream*: the eval error makes us CHOOSE the phantom line
over a genuinely better one moves earlier; fixing the eval stops us walking
into the trap. "SF has it too" does not lower the value of fixing ours — ours
is what our own search drags us into.

**So this is the primary eval-side Elo lever, not a refinement.** Reframed
plan: fix the mis-scoring at the cause via corrective over-exposure on the
blindspot class (these positions are LEARNABLE — SF-static nails them on
identical data — so the cause is under-exposure/under-training, and SF's much
longer schedule is presumably why SF learned the class and we didn't). The
running corrective experiment + the GPU4-scale harvest test whether over-
exposure closes it; mirror metric (coda-rate ↓ without sf-rate ↑) is the
cause-vs-symptom / generalization readout. Dynamic contempt is a separate
lichess symptom-patch, NOT a fix for this bug.
