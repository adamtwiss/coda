# Eval Overrate & Blind-spot Findings (living document)

**Purpose.** A shared, growing log of positions where Coda's evaluation is
*wrong* — almost always **overrating** its own position — discovered by mining
real gauntlet games against the top-20 engines and cross-checking with
Stockfish (SF) as a trusted third party. This doc is for both humans (Adam) and
other Claude instances. The wandering-bishop work showed that a doc with
**concrete positions + diagrams + a named theme** is what makes a blind-spot
class fixable (there, mixing a few % of self-play-vs-SF into the training set —
unfiltered — measurably cut the overscoring of that class).

**Status:** started 2026-06-28. Pipeline is `scripts/game_analysis.py`
(taxonomy / mine / oracle / features). Tooling notes at the bottom.

> **One-line takeaway so far:** the overrate is *not* a single bug. It splits
> into three themes — (1) **illusory attacks** the threat features over-value,
> (2) **drawn-endgame blindness**, (3) **search-leaf inflation** over an honest
> static eval. The static NNUE is the dominant culprit only for theme (1).

---

## How the data is produced (methodology)

1. **Gauntlet.** Coda (12+0.12) vs 19 top-20 engines (10+0.1), ~950 games,
   eval-per-move recorded for both sides (STM-POV in PGN comments). Also run at
   1.2× and 2× Coda time to measure *time-elasticity* (see below).
2. **Mine** (`game_analysis.py mine`). Convert both evals to **Coda-POV**.
   `overrate = coda_self − opp_of_coda`. Keep only **draws/losses** with a
   **sustained** overrate (≥150cp over ≥6 plies) — a big eval that *won* is
   correct, not an overrate. This is the load-bearing filter: result + opponent
   disagreement together.
3. **Oracle** (`game_analysis.py oracle`). Re-score each candidate with SF
   (depth 24) as ground truth. `true_overrate = coda_self − sf_coda_pov`.
   - **Bucket B** (`true_overrate ≥ 150 & sf ≤ 250`): SF confirms Coda is
     genuinely overrating. **35 of 43 candidates (81%)** landed here — SF sides
     with the *opponent*, even weak ones (Quanticade, Starzix). Result split:
     **2 losses, 33 draws, 0 wins.**
   - **Bucket A** (the other 8): Coda was right and the *opponent* mis-evaluated,
     or it's a conversion/technique failure (see the KBN game below).
4. **Static-vs-search decomposition** (the key diagnostic). For each bucket-B
   FEN compare three numbers in Coda-POV:
   - **STATIC** = Coda's NNUE eval with **no search** (UCI `eval`).
   - **SEARCH** = Coda's in-game eval (`coda_self`, search at game TC).
   - **SF** = ground truth (depth 24+).
   On the **27 quiet** positions (where static eval is trustworthy):
   - mean **static − SF = −0.61** · mean **search − SF = +2.87**
   - **11 EVAL-blind** (static itself ≥1.5 over SF) · **16 SEARCH-inflate**
     (static ≈/below SF, search runs high).
   (8 more positions are tactical/in-check, where static eval is meaningless and
   excluded.)

**Reading that result.** Across the *same* mined positions the **static net is,
on average, slightly *below* SF** — so "threats make the static eval too high"
is **not** the systematic cause. It is real for a specific cluster (theme 1),
but the larger numeric gap is the **search** returning ~+2.9 over truth from
positions the root static rates honestly. That points at the search reaching
**leaves the NNUE overrates a few plies forward**, not a root-eval bug. (Next
experiment: walk Coda's PV to the leaf and static-eval it — see Open Questions.)

---

## Time-elasticity: which deficits are tactical vs strategic

From the 1× / 1.2× / 2× gauntlets (Coda_score = 100 − opp_score%, since each
opponent plays only Coda). How much does *more time* close the gap?

- **Tactical / depth-bound (time helps a lot — closeable by NPS, pruning,
  ordering):** Viridithas (52→72, +20), Reckless (31→50, +19), PlentyChess
  (44→62, +18), Clover (+16), Obsidian / Cinder (+14).
- **Strategic / eval-bound (time barely helps — this is where eval quality is
  the wall):** Alexandria (50→54, +4), Integral (53→58, +5), Stockfish (34→43,
  +9, partly ceiling).

**Convergent signal:** Integral shows up as *both* a top overrate-draw source
**and** low time-elasticity → it is the cleanest strategic-eval target. Most of
Coda's deficit-to-field, though, is tactical/depth (search-efficiency leverage),
not eval.

---

## The themes (with positions)

Three recurring classes. Diagrams are White-at-bottom; the score line is
Coda-POV (so "+" = Coda thinks it is better).

### Theme 1 — Illusory attacks (threat features over-value an attack that is only a draw)

The net's heavy threat features light up on a queen+piece battery near the enemy
king and read it as winning, when the attack is in fact a perpetual or fully
defensible — frequently while Coda is materially **equal or down**. This is the
cluster where the **static eval itself** is wrong, so it is the prime
**training-correction** target and the strongest support for Adam's
threat-feature hypothesis.


#### Stormphrax (move 40) — only a perpetual

`6k1/3r2b1/4rp2/1pp1nN1Q/8/1P4R1/2q3P1/5R1K w - - 4 40`

> Coda search **+7.06** · static **+2.86**, SF **+0.00**

```
  -----------------
8 |.|.|.|.|.|.|♚|.|
  -----------------
7 |.|.|.|♜|.|.|♝|.|
  -----------------
6 |.|.|.|.|♜|♟|.|.|
  -----------------
5 |.|♟|♟|.|♞|♘|.|♕|
  -----------------
4 |.|.|.|.|.|.|.|.|
  -----------------
3 |.|♙|.|.|.|.|♖|.|
  -----------------
2 |.|.|♛|.|.|.|♙|.|
  -----------------
1 |.|.|.|.|.|♖|.|♔|
  -----------------
   a b c d e f g h
```

SF PV: Qh6 Ree7 Qh5 Re6 (repetition). Coda is White and **a clean minor piece down** (Black has N+B for nothing). The threat features value the Qh5/Nf5 battery at +7; there is no breakthrough — it is a draw by perpetual.

#### Quanticade (move 27) — EVAL-blind: the *static* net says +5.46

`q4r2/5p2/2np1n1k/2p1pPNb/1pP5/1P1P2QP/6BK/5R2 w - - 1 27`

> Coda search **+5.58** · static **+5.46**, SF **+0.38**

```
  -----------------
8 |♛|.|.|.|.|♜|.|.|
  -----------------
7 |.|.|.|.|.|♟|.|.|
  -----------------
6 |.|.|♞|♟|.|♞|.|♚|
  -----------------
5 |.|.|♟|.|♟|♙|♘|♝|
  -----------------
4 |.|♟|♙|.|.|.|.|.|
  -----------------
3 |.|♙|.|♙|.|.|♕|♙|
  -----------------
2 |.|.|.|.|.|.|♗|♔|
  -----------------
1 |.|.|.|.|.|♖|.|.|
  -----------------
   a b c d e f g h
```

SF PV: Bxc6 Qxc6 Qe3 … ≈equal. The static net (no search) already returns +5.46 on a position SF calls +0.46. The Ng5 + Qg3 attacking shape reads as winning; Black is fine once the c6 knight is traded. This is the purest threat-overrate in the set.

### Theme 2 — Drawn-endgame blindness

Balanced rook endings, queen endings, and Q-vs-R / minor-piece technique. The
net overrates "I have an active piece / extra-looking structure" in positions
that are fundamentally drawn or very hard to convert. Mix of EVAL-blind
(PlentyChess static +4.02) and search-inflated. This is the same family as the
**KBN-vs-K** game below: the eval can even be *correct* while the failure is
*conversion*, or the eval overrates an objectively drawn ending.

#### PlentyChess (move 34) — EVAL-blind queen ending (static +4.02)

`8/4kp2/4b1p1/4Q3/1P1pP1P1/p2PqP2/P1r2NK1/5R2 w - - 3 34`

> Coda search **+3.15** · static **+4.02**, SF **+0.07**

```
  -----------------
8 |.|.|.|.|.|.|.|.|
  -----------------
7 |.|.|.|.|♚|♟|.|.|
  -----------------
6 |.|.|.|.|♝|.|♟|.|
  -----------------
5 |.|.|.|.|♕|.|.|.|
  -----------------
4 |.|♙|.|♟|♙|.|♙|.|
  -----------------
3 |♟|.|.|♙|♛|♙|.|.|
  -----------------
2 |♙|.|♜|.|.|♘|♔|.|
  -----------------
1 |.|.|.|.|.|♖|.|.|
  -----------------
   a b c d e f g h
```

SF PV: b5 g5 Qb8 … 0.00. Static +4.02 vs SF +0.07. Net overrates the centralised Qe5 + pawn mass and misses Black's a3 passer + active Qe3/Rc2 counterplay.

#### Viridithas (move 72) — balanced rook ending read as +3.9

`8/1R6/5ppK/3r1k1p/7P/5PP1/8/8 w - - 0 72`

> Coda search **+3.90** · static **+3.30**, SF **+0.18**

```
  -----------------
8 |.|.|.|.|.|.|.|.|
  -----------------
7 |.|♖|.|.|.|.|.|.|
  -----------------
6 |.|.|.|.|.|♟|♟|♔|
  -----------------
5 |.|.|.|♜|.|♚|.|♟|
  -----------------
4 |.|.|.|.|.|.|.|♙|
  -----------------
3 |.|.|.|.|.|♙|♙|.|
  -----------------
2 |.|.|.|.|.|.|.|.|
  -----------------
1 |.|.|.|.|.|.|.|.|
  -----------------
   a b c d e f g h
```

SF PV: Rb4 g5 … +0.18. Material dead equal (R+3P each). Textbook drawn rook ending; the net scores activity as +3.9 and does not grasp the drawing tendency of balanced rook endgames.

#### Clover (move 81) — Q-vs-R, right sign, magnitude overrate

`8/5k2/8/8/4R3/5P2/5K2/3q4 b - - 1 81`

> Coda search **+5.49** · SF **+1.82**

```
  -----------------
8 |.|.|.|.|.|.|.|.|
  -----------------
7 |.|.|.|.|.|♚|.|.|
  -----------------
6 |.|.|.|.|.|.|.|.|
  -----------------
5 |.|.|.|.|.|.|.|.|
  -----------------
4 |.|.|.|.|♖|.|.|.|
  -----------------
3 |.|.|.|.|.|♙|.|.|
  -----------------
2 |.|.|.|.|.|♔|.|.|
  -----------------
1 |.|.|.|♛|.|.|.|.|
  -----------------
   a b c d e f g h
```

SF PV: Qh1 Re2 … +1.82 (slow technical Q-vs-R win). Coda (Black) correctly sees it is winning, but +5.49 vs +1.82 overstates how easy the conversion is — and it was not converted (drawn).

### Theme 3 — Search-leaf inflation (honest static, optimistic search)

16 of 27 quiet bucket-B positions: the root **static eval is fine (≈ or below
SF)** but the **search returns +2 to +5**. The search is reaching leaves the
NNUE overrates a few plies forward and propagating that back. Still ultimately
an eval problem — but at the *leaf*, not the root — so the same training
correction should help; if it doesn't, it's a genuine search-optimism bug.

#### Integral (move 57) — drawn R+P ending, search inflates to +2.87

`8/8/7R/8/3p1K2/r6P/2k5/8 b - - 2 57`

> Coda search **+2.87** · static **-1.55**, SF **+0.00**

```
  -----------------
8 |.|.|.|.|.|.|.|.|
  -----------------
7 |.|.|.|.|.|.|.|.|
  -----------------
6 |.|.|.|.|.|.|.|♖|
  -----------------
5 |.|.|.|.|.|.|.|.|
  -----------------
4 |.|.|.|♟|.|♔|.|.|
  -----------------
3 |♜|.|.|.|.|.|.|♙|
  -----------------
2 |.|.|♚|.|.|.|.|.|
  -----------------
1 |.|.|.|.|.|.|.|.|
  -----------------
   a b c d e f g h
```

SF PV: d3 Rc6+ … 0.00. R+P vs R+P, dead drawn. STATIC is honest (−1.55); the SEARCH returns +2.87. The overrate is produced by the search, not the root eval — the leaf-inflation class.

---

## Egregious example — the +8 Integral draw (KBN-vs-K: eval was RIGHT)

The headline game that kicked this off (Coda scoring ~+8 vs Integral ~0, drawn)
turned out **not** to be an eval blind-spot. It reduced to **KBN-vs-K**, a
forced win. Coda's eval was *correct* (static **+10.83**, SF mate). The failure
was **conversion/technique**: Coda burned the 50-move count shuffling, then
played **Bd5?? Kxd5** hanging the bishop into KN-vs-K — a dead draw. Training
data will **not** fix this class; it is endgame technique (mate-distance /
tablebase-style knowledge / 50-move awareness in search). Logged here because it
is the canonical "big eval that drew" and shows why the **result + SF** filter
matters: without SF this looks like a −8 eval bug; with SF it is a conversion
bug. Keep these two classes separate.

---

## Actionable hypotheses & next steps

1. **Training correction set (theme 1 + drawn endgames).** Emit the bucket-B
   positions as **SF-rescored EPD/binpack** and mix into training (the
   wandering-bishop recipe: even a few % of self-play-vs-SF, unfiltered, moved
   the needle). Prioritise: *attack-fizzles-to-perpetual*, *balanced rook
   endings*, *Q-ending / Q-vs-R technique*. `game_analysis.py oracle` already
   produces SF-scored rows; add an EPD emitter.
2. **Leaf-eval walk (theme 3 root-cause).** For each search-inflate position,
   take Coda's PV to the leaf and static-eval the leaf vs SF(leaf). If leaf
   static ≫ SF(leaf) → still eval/training (one ply forward) → same fix. If leaf
   static ≈ SF(leaf) → genuine search-propagation bug. *Blocker:* Coda
   suppresses `info`/`pv` lines over UCI — need a PV dump path or instrument the
   search.
3. **Threat-feature probe (theme 1).** On the theme-1 statics (Stormphrax,
   Quanticade), check whether dampening/ablating threat features pulls the
   static eval toward SF. If yes, it localises the overrate to the threat block
   and motivates either a training fix or a threat-scaling change.
4. **Scale the mine.** This run was a 43-position proof-of-pipeline. Run the full
   950-game gauntlet, **focused on the strategic opponents** (Alexandria,
   Integral, Stockfish) where eval — not depth — is the wall.
5. **Drawn-endgame eval damping.** Separately consider whether 50-move / shuffle
   awareness or endgame eval scaling reduces theme-2 overrates (test on OB).

## Open questions

- Does the static net *systematically* overrate, or only on theme 1? (Current
  data: only theme 1; mean static−SF is slightly negative overall.)
- Is the search-leaf inflation an eval problem one ply forward, or a real search
  bug? (Leaf walk, item 2.)
- How much of the strategic gap to Alexandria/Integral is these blind-spots vs
  general eval noise?

---

## Tooling

- `scripts/game_analysis.py` — `taxonomy | mine | oracle | features`. Works in
  Coda-POV. PGN eval comments are STM-POV; python-chess strips the `{}` braces
  (regex must NOT require `{`). Mate → ±30000. `mine` filters draws/losses with
  sustained overrate; `oracle` adds SF ground truth + `true_overrate`.
- Decomposition scratch: `/tmp/static_decomp2.py`, `/tmp/enrich_decomp.py`
  (adds quietness/phase), `/tmp/sf_diag.py` (SF PV + diagram).
- Gotchas: `len(board.pieces(...))` not `chess.popcount(...)` on a SquareSet;
  Coda `eval` prints `NNUE evaluation +X.XX (white side)` — negate for Black to
  get Coda-POV.
- SF binary: `/home/adam/chess/engines/Stockfish/src/stockfish`.

## Changelog

- **2026-06-28** — Doc created. First wave: 43 mined candidates → 35 SF-confirmed
  overrates (33 draws / 2 losses) → static-vs-search decomposition (11 EVAL-blind,
  16 search-inflate, 8 tactical). Three themes named with diagrams. KBN Integral
  game classified as conversion (not eval).
