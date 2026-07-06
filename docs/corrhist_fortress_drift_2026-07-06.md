# Correction-History Fortress Drift (2026-07-06)

**Critical finding.** In dead-drawn / locked positions (opposite-coloured
bishops, blocked pawns, fortresses) Coda reports a confident **±0.45** eval
when the true value is **0.00**. The raw NNUE is *not* at fault — the culprit
is **correction history**, which self-reinforces into a phantom score. This
document records the diagnosis so it is not lost, and anchors the fix.

## How it surfaced

Two lichess games, played minutes apart (both drawn):

- `PYiXcgdg` (codabot = Black): from move ~67 onward Coda is a pawn (later two
  pawns) down in an OCB ending yet evaluates itself as *better*; declined the
  free `Bxa3` at move 67 "because it thought it was winning".
- `aIosNgFS` (coda_bot = White): a textbook **K+B+P vs K+B+P, opposite-coloured
  bishops, h3 vs h6 pawns blocked** — a dead draw — scored **+0.5 for White**.

## The investigation chain (each step narrowed it)

All searches 3 s, single thread, current `main` (v10 net), **no tablebases**
unless noted. Score is from the side-to-move's view.

### 1. Confirm the misread, six positions — magnitude is ~constant ±0.45

| Position | Truth | Coda search |
|---|---|---|
| G1 mv67 (down 1P, 7-man OCB) | draw | −44 |
| G1 mv94 (down 2P, 6-man) | draw | −44 |
| G1 mv100 (down 2P) | draw | −47 |
| G2 mv52 (OCB + h-pawns) | draw | −45 |
| **G2 mv54 (blocked h3/h6 OCB)** | **dead draw** | **+46** |
| G2 mv62 (blocked OCB) | dead draw | −45 |

The sign is essentially **noise** (whichever way a tiny initial fluctuation
points) — hence +0.46 in one game and −0.44 in the other. The *magnitude* is
suspiciously constant.

### 2. Tablebases are not the story

These are 6/7-man; the deployed set is 5-man (6-man is ~150 GB, impractical).
With 5-man TB active game-1's endings *do* resolve to cp 0 because search
simplifies into the tables — but game-2's blocked OCB **cannot** simplify
(OCB bishops never trade, blocked pawns), so it stays wrong regardless. TB is
a partial band-aid, not the cause.

### 3. It is NOT the net — the reference engines prove it is learnable/search

Same positions, no TB:

| Position | Coda | Stockfish | Reckless | Obsidian |
|---|---|---|---|---|
| G2 mv54 | **+46** | +1 | 0 | +2 |
| G2 mv62 | **−45** | 0 | 0 | 0 |
| G1 mv67 | **−44** | 0 | 0 | 0 |
| G1 mv94 | **−44** | 0 | −4 | 0 |

All three reference engines nail these as ~0. Coda is the lone outlier.

### 4. Static eval is fine — the divergence is in SEARCH

`eval` (static NNUE, no search):

| Position | SF static | Coda static | → SF search | → Coda search |
|---|---|---|---|---|
| G2 mv54 | +0.12 | −0.08 | ≈0 | **+46** |
| G2 mv62 | +0.13 | +0.11 | 0 | **−45** |
| G1 mv67 | −0.19 | +0.16 | 0 | **−44** |
| G1 mv94 | −0.24 | −0.26 | ≈0 | **−44** |

**Both nets are equally "blind" statically** (±0.1–0.27; G1mv94 is SF −0.24 vs
Coda −0.26, essentially identical). SF's *search* collapses to 0; Coda's search
**amplifies** — net says +0.16, search returns +0.44, i.e. it pushes *away* from
the draw. The bug lives between static eval and the backed-up score.

### 5. It is not rule50 decay

Halfmove-clock sweep on G2 mv62 (SF stays 0 at every clock value; Coda only
collapses at the very cliff):

| rule50 | Coda | SF |
|---|---|---|
| 2 | −45 | 0 |
| 40 | −42 | 0 |
| 80 | −46 | 0 |
| 98 | 0 | 0 |

Coda's existing `apply_halfmove_scale` is effectively **inert** here — the
search escapes it by making clock-resetting pawn pushes into net-rewarded
positions. SF is flat across the whole range, so its 0 is structural, not
rule50-driven.

### 6. The PVs — "piece pushing" vs "no progress"

G2 mv62:

- **SF PV** (0): `c1d2 c5e7 d2d3 e7f6 d3e4 f6d8 g4d1 …` — a pure piece shuffle,
  **no pawn moves**, held flat at 0. SF sees the shuffle makes no progress.
- **Coda PV** (−45): `c1d1 c5a3 … c4b5 b1a1 b5b6 h3h4 b6a5` — marches the black
  king up the board and slips in a clock-resetting `h3h4`, and its search
  rewards that phantom "progress".

### 7. The leaf of Coda's own PV is ~0 in BOTH engines

Position after Coda's 14-ply PV (`8/8/7p/k7/4B2P/b7/8/K7 w`): SF static **+0.12**,
Coda static **−0.03**. So the −0.45 is **in no single static eval** — the raw
net is ~0 everywhere sampled. Something between static eval and the backed-up
score manufactures it.

### 8. Root cause: CORRECTION HISTORY (ablation is decisive)

`NO_CORRECTION=1` (corrhist off), same positions:

| Position | baseline | NO_CORRECTION |
|---|---|---|
| G2 mv54 | +46 | **0** |
| G2 mv62 | −45 | **0** |
| G1 mv67 | −44 | **0** |
| G1 mv94 | −44 | **0** |

Corrhist off → exactly 0, matching SF/Reckless, on **all four**. `NO_FH_BLEND`
alone does nothing (−46), so it is specifically correction history.

Zeroing any **single** corrhist source (`CORR_W_PAWN/NP/CONT/TRANS = 0`) does
**not** fix it — the score just flips rail sign (−45 → +46). Only removing *all*
correction gives 0. So it is the corrhist **system** self-bootstrapping, not one
bad table.

## Mechanism — a positive-feedback instability

The gravity update (`update_corr_entry`, search.rs:1751) is

```
entry += bonus − entry·|bonus|/LIMIT ;  entry.clamp(±CORR_HIST_LIMIT)
```

Its fixed point for any **consistently-signed** error is the **rail** (±LIMIT),
*regardless of the error's magnitude*. A locked/no-progress position produces a
persistent micro-discrepancy between static and search; corrhist seeds it, every
source rails to its limit, and the summed weighted correction maxes out at
~±0.45cp (`corrected_eval`, search.rs:1743 — no clamp on the summed correction).

It is **self-exciting**: with corrhist off there is no seed error, so the search
returns 0. Stockfish runs the *same* gravity formula but does not drift, because
its static ≈ search ≈ 0 in fortresses gives it no seed to amplify. Coda's search
produces a small persistent seed (ultimately from corrhist itself) and rails.

Net: the raw NNUE is correct (~0), and the search *without* corrhist is correct
(0); corrhist — normally a strong +Elo feature — overfits its own search noise in
the low-signal regime and inverts an already-correct eval into a confident
phantom advantage.

## Why it matters

Not merely cosmetic. A confident phantom ±0.5 in dead positions can make Coda
decline safe repetitions / draws to "play for a win", and — more dangerously —
misjudge whether to *enter* a drawish/fortress line near the boundary, risking a
real loss against a strong opponent. The two source games drew, but the eval was
wrong throughout.

## Fix direction

Corrhist is load-bearing (+Elo overall) and ±0.45 is a perfectly *normal*
correction magnitude in live play, so a blunt magnitude clamp cannot separate
fortress-drift from legitimate correction. The fix must target the
**low-signal / no-progress regime specifically**, damping the *applied*
correction there while leaving normal (higher-material, in-progress) play
untouched, then SPRT for non-regression. Candidate signals: piece count
(material) and/or the no-progress clock. Tracked in the branch that follows this
doc; result logged to `experiments.md`.

## Reproduction

```
# fortress that stays broken regardless of clock / TB:
FEN="8/8/7p/2b5/6B1/7P/5k2/2K5 w - - 18 62"
printf 'position fen %s\ngo movetime 3000\n' "$FEN" ; sleep 3.3 ; echo quit   # → cp -45
NO_CORRECTION=1 <same>                                                          # → cp 0
```
