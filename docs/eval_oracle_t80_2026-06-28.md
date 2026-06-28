# T80 LC0-Oracle Eval Analysis — is the overrate visible at random? (2026-06-28)

**Companion to `docs/eval_overrate_findings.md`.** That doc mines *gauntlet
games* (positions Coda reaches in play) and finds a strong, systematic overrate.
This doc asks the complementary question on **random T80 positions**, using a
fast, SF-free method — and the answer reframes where the blind spot actually
lives.

## The key enabling fact: LC0 is a fair, shared oracle

**Coda and SF train on identical data** — the same T80/LC0 binpacks, copied
recipes. So the binpack's stored **LC0 score is a fair shared oracle for both
engines** — there is *no home-field bias* toward either (this is the basis of
`scripts/eval_quality.py`'s cross-engine ranking, and an earlier conversational
"Coda is just more T80-fit" caveat was wrong — both engines are equally fit to
this exact data).

Two consequences:
1. We can **rank** Coda's static eval against SF's (and the field) by agreement
   with the LC0 oracle — fast, fair, scalable.
2. We can **detect** eval overrate **without Stockfish at all**: the blind spot,
   if it shows as a static bias, is just "Coda's static overrates the LC0 label."
   Instant, and scalable to a billion positions. SF-search is only needed to
   *spot-validate the label* on >7-man positions (where LC0 is an MCTS estimate,
   not tablebase-exact).

The data file (`/training/sf/test80-2024-01-jan-2tb7p.min-v2.v6.binpack`) is
`2tb7p`-rescored: **positions with ≤7 men carry tablebase-exact WDL** — the
clean-oracle band where the signal is fully trustworthy.

## Result 1 — Coda's static eval is tied #1 with SF on the LC0 oracle

`coda eval-dist --csv` (gauntlet net, 20k quiet positions) → `eval_quality.py`
(drives SF + 10 engines via UCI `eval`, scores Spearman vs the LC0 blend oracle):

| Engine | Spearman vs LC0 | Engine | Spearman |
|---|---:|---|---:|
| **Coda** | **0.832** | Berserk | 0.789 |
| Stockfish | 0.830 | Tarnished | 0.786 |
| Obsidian | 0.814 | PlentyChess / Clover | 0.775 |
| Integral | 0.798 | Stormphrax | 0.773 |
| Halogen | 0.796 | Viridithas (d1*) | 0.761 |

Mean |static − LC0| over 20k: **Coda 157cp vs SF 200cp**. So on the T80
distribution our static eval is genuinely **top-tier** — Coda and SF are tied at
the top, clear of the field. This both confirms the framing ("eval 2nd-only-to-SF,
really tied") *and* explains why this metric cannot, by itself, expose the blind
spot: the oracle shares it.

## Result 2 — at RANDOM T80 positions the overrate is essentially symmetric

SF-free detector (`scripts/t80_overrate_scan.py`) over a **300k** quiet sample,
overrate = Coda static − LC0 ≥ 150cp, balanced band |LC0| ≤ 600:

| slice | over | under | asymmetry | net bias |
|---|---:|---:|---:|---:|
| ALL random T80 (n=276k) | 6.3% | 6.5% | 0.97× | −1.1cp |
| endgame ≤12 men (n=68k) | 6.7% | 6.6% | 1.01× | −0.6cp |
| **TB-exact ≤7 men (n=33k)** | **2.1%** | **1.8%** | **1.16×** | **+0.5cp** |

**On random positions Coda's static eval is well-calibrated to the oracle — not
systematically overrating.** Even in the tablebase-exact band the overrate is a
*mild* 1.16× asymmetry (real — 705 vs 607 is significant — but small), not a
gross bias. (An earlier 20k-sample read of "2:1" in this band was small-sample
noise: ~100 disagreements; the 300k band has ~1300.)

## Result 3 — the blind spot is PLAY-CONDITIONED, not a random-position bias

This reconciles the two docs. The gauntlet mine finds a **strong one-sided
overrate** (median search +136 over SF, fortress cluster); the random T80 scan
finds it **symmetric**. The difference is the **sampling frame**:

- **Random T80** samples the data distribution the net was *trained on* → the
  static eval is calibrated there (Result 2), and tied-#1 in quality (Result 1).
- **Gauntlet** samples positions the **search steers Coda into during play** —
  and the search preferentially walks toward the simplified fortress / drawn
  endgame positions the static *overrates*. The overrate is a property of the
  **tail the engine selects**, not of the average position.

The faint 1.16× echo in the TB-exact band is the same blind spot showing through
even at random — confirming the *direction* — but its small magnitude is the
point: **you cannot fix this by re-weighting random T80**, because at random
there is no one-sided error to correct.

## Implication for the fix

- **The right corrective data is play-conditioned, not random.** The incoming
  **~6M Coda-vs-SF games (~300M positions)** are exactly the positions Coda
  *reaches in play* — the frame where the overrate lives — with SF labels. That
  route is validated as the vehicle; random-T80 re-weighting is **not** a
  substitute for it.
- **The SF-free detector still earns its place** as a fast harvesting/validation
  filter: scan any corpus for `coda_static − lc0 ≥ thresh` to pull the
  overrate-direction tail (which carries correct LC0 labels for free), and to
  measure calibration drift on a new net cheaply. It is the billion-position-
  scalable funnel; it just isn't, alone, the blind-spot finder.
- **Deployment note:** the ≤7-man clean-oracle band is **Syzygy-covered at
  deployment** anyway (like the KBN case in the overrate doc), so the in-play
  target is the **>7-man fortresses** — where the LC0 label is MCTS, not exact,
  so SF-search spot-validation of the label is warranted before training toward
  it.

## Pipeline (SF-free detection, billion-scalable)

1. `coda eval-dist -i <binpack> --quiet-only --csv <out> -n <net>` — emits
   `fen, white_result, coda_static_white_cp, lc0_white_cp`. Fast (Coda static
   only); the scalable part.
2. `t80_overrate_scan.py <csv> --emit-tsv <cands.tsv> [--max-men 7]` — reports
   the overrate/underrate asymmetry by piece-count band and emits the
   overrate candidates as `fen<TAB>lc0_stm_cp` (the corrective label, already in
   the data).
3. `coda import-tsv -i <cands.tsv> --fen-col 0 --score-col 1 --repeat N -o
   oversample.binpack` — fold into fine-tuning (the over-sample knob).
4. SF-search (`sf_relabel.py`) **only** to spot-validate the label on a >7-man
   sample, not per-position.

## Tooling

- `scripts/t80_overrate_scan.py` — SF-free overrate-vs-LC0 detector + candidate
  emitter (this doc's Result 2/3).
- `scripts/eval_quality.py` — cross-engine static-eval ranking vs the LC0 oracle
  (Result 1). Needs `numpy`.
- `scripts/t80_sf_disagree.py` — SF-*search* arbiter for the Coda-vs-SF
  disagreement funnel (the slow, independent-arbiter leg; use on the funnel
  subset only).
- `coda eval-dist --csv` / `coda import-tsv` — the binpack→CSV and TSV→binpack
  legs (src/main.rs).

## Data + repro

- 20k CSV `/tmp/t80_evalq.csv`, 300k CSV `/tmp/t80_evalq_big.csv` (gauntlet net
  `nets/multi-v8-l132-s3-v3-swa.nnue`).
- `python3 scripts/t80_overrate_scan.py /tmp/t80_evalq_big.csv` reproduces
  Result 2. `python3 scripts/eval_quality.py /tmp/t80_evalq.csv` reproduces
  Result 1.
