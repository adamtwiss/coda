# Conversion-failure study — "correct eval, didn't convert" (2026-06-29)

## TL;DR (for Hercules / training)

We played Coda (v8s3) 3,800 STC games vs 18 top-20 engines with **resign
adjudication off** so won positions play out, then SF-gated every "Coda reached
≥+2 for ≥3 moves but didn't win" position. Result: **52% were overscores** (peak
wasn't really winning → eval-flywheel / the T80 overscore-corpus you're already
building), and **48% were genuine conversion failures** (SF-confirmed winning,
drew/lost). The genuine half is the new, complementary signal.

Drilling the most vivid genuine cases — **Coda drawing KQ-v-K / KR-v-K / KBN-v-K**
— the cause is **NOT a net blindspot and NOT a bug**. The raw NNUE evaluates
KQ-v-K correctly at **+8.7** (and Coda mates KQ-v-K in 9–19 moves, KR-v-K in 10
from a clean start). The 0.00 you see in games is manufactured in **search** by
the rule50 eval damping (`score × (100−halfmove)/100`, search.rs:1257): once the
won material is reached with the 50-move clock already high, that factor → 0,
flattening even a +8.7 eval to draw-looking; the position then looks drawn → TM
under-searches → no progress → clock fills → 50-move draw. A self-fulfilling
spiral. Global softening of that damping was tried and **failed** (OB #2382).

The one mate Coda genuinely *can't* do is **KBN-v-K vs best defense** (the ~33-ply
drive-to-the-correct-corner technique) — beyond STC tactical search, with no eval
gradient to guide it. **Critically, this is NOT a training-data gap unique to us:
Stockfish's net is *also* flat** (~+10 across all KQ-v-K king positions, tested) —
neither net carries a mating gradient. SF converts anyway by (a) a **more
conversion-aware eval** (it scores KBN-v-K at +1.8, knowing it's hard; Coda says
+6.5) and (b) **deeper selective endgame search** (seldepth 42 vs our 34). And
per a prior cross-engine review, only Caissa uses internal mate recognizers — the
rest don't.

**Confirmed cause = flat eval + insufficient depth (NOT LMR — corrected
2026-06-29).** LMR is *already* disabled for ≤5-piece positions by the existing
`LMR_ENDGAME_PIECES_10X` gate (`tp10(49)=5`, skip LMR when `popcount ≤ 5`), so it
was never active in KBN-v-K. Deterministic fixed-node test (200k nodes/move)
confirms **baseline ≡ `NO_LMR` bit-identical** on KBN-v-K — an earlier *movetime*
test that suggested "NO_LMR fixes it" was non-determinism noise (the same config
gave MATE/DRAW/DRAW). The true cause: even at 200k nodes/move KBN-v-K is
**position-dependent** (draws from the symmetric-centre start, mates from others) —
the flat NNUE gives no gradient to find the ~33-ply W-maneuver, and it's beyond
reachable depth from the hard start. The rule50 damping then finishes losing games
by crushing the eval to 0 as the clock fills. No hidden post-NNUE filtering exists
(only `EVAL_SCALE_PCT` no-op + material scale + rule50 scale).

**So for training:** adding KQ-v-K-type data is *not* the fix (our net already
knows it; mate-scores are filtered for good reason; SF's net is also flat). The
fix is to give the eval a **progress gradient**, which is missing.
**Deployment-masked** for the tiny cases (CCRL + lichess both 5-man EGTB).

**Two eval/scaling fixes PROTOTYPED, both FAILED (2026-06-29).**
1. *Mop-up gradient* (lone-king-gated push-to-edge/corner + KBN bishop-corner
   bias): net-negative at every magnitude — fixed some starts, regressed others,
   slowed KQ/KR-v-K. Branch `zeus/endgame-mopup-eval` discarded.
2. *Rule50-damping exemption for forced-mate lone-king endgames* (keep the eval
   honest-winning so the search doesn't give up): also net-negative under clean
   methodology — fixed 4 KR-v-K cases at high halfmove but regressed 4 KQ-v-K
   cases (5→4 mates). Branch `zeus/endgame-no-rule50-damp` discarded.

**Why both fail — and the reliable path.** These mates are converted by *pure
tactical search on a flat eval*, so ANY eval perturbation (a gradient OR a
magnitude change) just reshuffles the search's pruning/ordering — helping some
positions, hurting others, netting neutral-to-negative. **The eval is not the
lever.** Coda *already* mates KQ/KR-v-K from normal/moderate clocks; the residual
failures are (a) high-halfmove arrivals where the 50-move budget is genuinely
near-exhausted, and (b) KBN-v-K vs best defense (33-ply technique beyond STC
depth). The only *reliable* "mate KX-v-K without EGTB" fixes are: a **hardcoded
mating-technique driver** for bare-king endgames (KQ/KR/KBB/KBN-v-K — the
Caissa approach, a bounded correct algorithm, not an eval nudge), or **more
endgame search depth**. Eval/scaling hacks are a dead end here.

**METHODOLOGY LESSON (load-bearing).** Single-game endgame conversion tests are
treacherous: (1) **movetime is non-deterministic** (same config → MATE/DRAW/DRAW),
and (2) **TT carries over between positions** unless you send `ucinewgame` per
position — both silently corrupted earlier conclusions in this very study (an
LMR "fix" and a mop-up "fail" that evaporated under clean testing). Always use
**fixed nodes + fresh `ucinewgame` per position + N>1**, and validate any real
change with an SPRT.

Remaining (non-elementary) levers unchanged: the **52%-OVERSCORE bucket** (Atlas
corpus) and **general 6–7+-men conversion** (LMR-in-bigger-endgames untested;
DTZ-labelled training data) carry the real competitive value. Full detail + the
167-candidate breakdown below.

**Goal.** Find where Coda reaches a winning position and fails to win it —
the bucket the T80 overscore-mining (Atlas) is structurally blind to, because
those positions aren't *misclassified labels*, they only manifest *dynamically*
in played-out games.

**Method.** Gauntlet: Coda (v8s3, `multi-v8-l132-s3-v3-swa.nnue`) vs 18 top-20
defenders (all ~3000 Elo), STC 10+0.1, **`-resign` removed** so wins play out,
draw-adjudication kept. 3800 games → `conversion_gauntlet.pgn`. Classifier
(`scripts/conversion_classifier.py`): find games where Coda's eval held
≥ +2.00 for ≥3 consecutive moves (sustained-peak — kills 1-ply search froth)
yet the game wasn't won; arbitrate the peak with (a) the opponent's in-PGN eval
[free], (b) SF depth-24 [gold]. Inspection tool: `scripts/conversion_inspect.py`.

## Headline (SF-gated, 167 sustained candidates of 3791 Coda games)

| bucket | n | share | lever |
|---|---|---|---|
| **OVERSCORE** (peak wasn't really winning) | 87 | 52% | eval flywheel / T80 corpus (Atlas) |
| **Genuine conversion failure** (SF says winning, drew/lost) | 80 | 48% | NEW — see below |

The 320 raw candidates → 167 after the sustained-peak filter (≈150 were
transient search-froth spikes, e.g. a move scored +2.6 for one ply then
self-corrected — those are NNUE-overscore + search optimism, not real positions).

## The key finding: a rule50-scaling × shallow-search death spiral (NOT a net blindspot)

Of the 80 genuine conversion failures, **26 are ≤6-men (Syzygy range)** and
**9 are literally KX-v-K** (lone enemy king). Coda could not mate a bare king:

```
KQ-v-K (Coda has Q+K):  Coda evals +1.22 -> +0.30 -> 0.00, shuffles, 50-move draw
                        meanwhile Starzix (the mated side) reads -M14, -M12
KR-v-K (x3), KBB-v-K (x2), KB+P, KB+N: same — drew by fifty-move rule
```

**The NNUE is NOT the problem.** Raw static NNUE on a clean KQ-v-K is **+8.73 to
+10.7** (hm-invariant) — the net evaluates elementary mates correctly. The 0.00
is manufactured downstream in search by the **rule50 eval damping**
(`apply_halfmove_scale`, search.rs:1257): `score * (100 - halfmove) / 100`. The
mechanism is a self-fulfilling spiral:

1. Net says +8.73; but the forced mate (mate-in-12, needs ~depth 24) is **not
   found** at the shallow depth the search reaches (in-game: depth 7, 72ms), so
   the backed-up score is the NNUE eval — which is **subject to scaling** (only
   `score.abs() >= MATE_SCORE-100`, i.e. *found* mates, are exempt).
2. As the 50-move clock climbs without progress, `(100-hm)/100 → 0` drags the
   +873cp toward 0 (hm90→+87, hm99→+8, hm100→0).
3. The ≈0 eval makes the won position *look drawn* → TM allocates minimal time →
   search stays shallow → never finds the mate that would earn the mate-exemption
   and reset the clock → hm climbs → step 2 worsens → 50-move draw.

So Coda's own 50-move eval-damping hides the win from the search, and the shallow
search that results can't find the mate that would override the damping. It is
the literal "pushes pieces into a draw" symptom — but the lever is search/scaling,
not the net. **Caveat:** the damping is *correct* for genuine fortresses (KNN-v-K
is up material yet drawn — rule50 damping rightly pulls it to 0). Eval magnitude
alone can't tell a real win (KQ-v-K) from a fortress (KNN-v-K); only a forced mate
(deeper search / TB) or a mate-driving heuristic distinguishes them.

## The overscore/underscore dichotomy

- **OVERSCORE (52%) is concentrated in COMPLEX positions** — only 3/87 are
  ≤6-men. Coda is over-optimistic in rich middlegames/imbalances (incl. the
  drawn-endgame *fortress* sub-class: KNN-v-K, R-v-N fortress, where Coda's
  material-anchored eval doesn't see the fortress).
- **UNDER-score conversion failures are concentrated in SIMPLE endgames** —
  the eval is too flat to drive a known win home.

Two opposite eval errors, cleanly separated by position complexity.

## Opponent-eval as a free arbitrator (cross-tab vs SF)

| opponent says | SF agrees | n | precision |
|---|---|---|---|
| SPLIT (overscore) | SF-overscore | 43 / 51 | **84%** |
| AGREE (winning)   | SF-winning   | 29 / 41 | **71%** |
| PARTIAL | mixed (43 win / 32 over) | 75 | ~57% |

All defenders being credible ~3000 engines, opponent-eval is a strong *cheap*
pre-sort — SPLIT predicts overscore 84%, AGREE predicts winning 71% — but SF
still corrects ~12% (flipped 8 SPLITs to winning, 12 AGREEs to overscore) and is
required to resolve PARTIAL. Verdict: opponent-eval triages for free; SF confirms
the close calls.

## Two things ruled out (2026-06-29)

- **No hard conversion bug for short mates.** From clean hm=0: KQ-v-K mates in
  9/19 moves, KR-v-K in 10 — Coda's search finds these tactically. The game-draws
  for these need an *already-advanced* 50-move clock when the won material is
  reached.
- **But the long *technique* mate fails.** KBN-v-K: mated a passive defender in
  39 moves (slow), but **DREW (50-move) against best defense** (lone king fleeing
  to the wrong-coloured corner). The ~33-ply W-maneuver is beyond STC tactical
  search, and the NNUE gives no gradient to guide it. Clearest proof the eval
  lacks endgame-driving technique. (KBN-v-K is rare + ≤5-men TB-masked, so its
  direct competitive value is ~nil — it is the *diagnostic*, not the prize.)
- **The net is flat, not blind.** Raw NNUE on KQ-v-K is ~+10 at *every* lone-king
  position (corner rated *lower* than centre) — it knows "up a queen" but gives
  the search **no mating gradient**. So conversion relies on tactical search
  finding the mate; at shallow STC depth that's slow (19 vs optimal ~10 moves),
  which loses the race against an advanced clock.
- **Global rule50 softening doesn't fix it** (OB #2382, HALFMOVE_SCALE_DENOM
  100→108, FAILED) — a global damping-rate knob can't fix a flat-eval problem.

## Deployment reality (TB masking)

**CCRL and lichess both use 5-man EGTB.** Every ≤5-men elementary mate
(KQ/KR/KBB/KBN-v-K) is therefore *masked in both competitive deployments*. The
bare-king cases are a diagnostic, not a direct competitive loss there. Real
competitive value: **6–7+-men won endgames** (outside 5-man TB), **bare-TB
testing** (OB SPRT + self-play, where these cost measured Elo), and general
won-position conversion efficiency.

## Levers (ranked by actionability)

1. **Mop-up / mate-driving eval term (search-side, SPRT-able) — most actionable.**
   The gradient test proves the NNUE lacks one. Blend a classic push-enemy-king-to-edge
   + bring-own-king-close term, tightly gated to overwhelming-material vs a
   lone/near-lone king. Restores the gradient → crisp conversion (~10 not ~19
   moves) → beats the clock. Validate: fewer moves-to-mate AND non-regression SPRT.
2. **Overscore (52%) → eval flywheel / T80 corpus** (Atlas), plus the
   drawn-endgame fortress sub-class.
3. **Complex no-progress drifts → eval *target*** (WDL blend / output-bucket
   resolution) — the general "make progress in won 7+-men endgames" case; hardest,
   longest-term, the one with real TB-uncovered competitive value.
4. **Datagen-augment endgames into training** — note KQ-v-K itself is NOT needed
   (net already +10) and mate-scores are filtered from our data anyway; value is
   in complex *non-mate* won endgames, if any.

## Caveats
- SPIKE/DRIFT sub-labels are contaminated on the mate cases (a decaying
  mate-ish eval reads as a "spike"); don't over-read that split. The robust cut
  is overscore-vs-underscore by SF + position complexity.
- Static-NNUE eval-vs-search decomposition deferred (batch tooling flaky;
  coda.rr exits mid-stream on a 167-position single input — needs small-batch).
- One box (Zeus), v8s3, STC; eval-centric so STC is appropriate.
