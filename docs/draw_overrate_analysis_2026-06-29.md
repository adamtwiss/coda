# Why we draw: overrate-dominated, not conversion-dominated (2026-06-29)

Follow-up to `docs/loss_analysis_2026-04-28.md`, driven by the
`coda-1.2-time-boost.pgn` gauntlet (950 games, Coda + 1.2× time vs the local
top-20 pool, **no EGTB**: 213W-96L-641D, 67.5% draws). Question: of the games
Coda drew despite its own eval peaking clearly winning, how many are *real wins
thrown away* (a search/conversion problem) vs *eval mirages* (a training-data /
overrate problem)?

## Method

1. Mine the 950 games for **faded draws**: games that ended drawn where Coda's
   eval peaked **≥ +1.5** at some point. 151 such peaks.
2. Snapshot the peak FEN for each.
3. Adjudicate each peak with an oracle and classify real-win vs overrate.

Two oracles were used, and **they disagreed**, which is the main finding.

## Result: deep search reverses the static cut

| Oracle | Real win | Overrate / mirage | Ambiguous |
|---|---|---|---|
| **SF static eval** (patched evalfile, fast) | ~48% | ~34% | ~18% |
| **SF16 search to depth 24** | **8 (5%)** | **89 (59%)** | **54 (36%)** |

SF's *static* eval over-rated the same sharp tactical positions Coda did — so
the static cut's "real wins" were largely an overrate artifact shared between
the two static evaluators. Searching the positions to depth 24 collapses most
of them to drawn. **Deep search is the reliable signal; static adjudication is
not, for exactly the positions we care about.**

### Headline

The 67.5% draw rate is **overrate-dominated, not conversion-dominated.** At
depth 24, only ~5% of the faded-draw peaks are genuinely won. ~59% are
deep-confirmed eval mirages. This is a strong independent validation of the
eval-blindspot / corrective-net programme (task #82): the faded draws are
overwhelmingly the kind of position that work targets, *not* endgame-conversion
failures. Endgame-conversion search is **not** the frontier for these games.

## Artifacts

- **`testdata/coda_conversion.epd`** (8) — deep-SF-confirmed real wins Coda
  drew. `bm` from the SF16 d24 oracle. 5 are ≥6-men (genuine search targets);
  3 are ≤5-men (TB-trivial, kept for completeness). The honest version of "the
  64 conversion failures" — the static-cut's 64 collapsed to 8 under search.
- **`testdata/coda_overrate_gauntlet.epd`** (89) — deep-confirmed overrates
  (Coda peak ≥1.5 but SF d24 ≤ +50cp). Magnitude spread: 63 at 1.5–2.5, 23 at
  2.5–4, 3 at 4+. Oracle-confirmed addition to the overrate corpus.

## Material-prior (lichess stats.json) idea — recommendation: don't build it

`stats.json` (tablebase.lichess.ovh) is **perfect-play Syzygy WDL
position-counts** (1511 material signatures up to 7-men), *not* human games.
Grounded against the 114 conversion+overrate failures, the material prior
touches only ~8 of them — both halves are dominated by **>7-men** positions,
out of the table's reach. A Caissa-style internal node recogniser would address
a sliver of the problem. Recommendation: **no recogniser**; the material data
is more useful as an analysis lens. The only piece possibly worth a narrow
experiment is OCB/RvB eval draw-scaling, and that is speculative.

## Blindspot harvest (parallel work, task #82)

The corrective-net harvest completed for 6 monthly T80 files (Jan–Jun 2024):
150/80 blindspot filter (Coda eval vs LC0 ≥150cp, then SF-worse-by-80),
~318M coda-worse positions total in `/training/blindspot/`. The
overrate-dominated finding above argues this corpus is aimed at the right
target.
