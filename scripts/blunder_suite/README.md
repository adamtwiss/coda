# Blunder suite generation

Scripts to regenerate `testdata/coda_blunders.epd` from fresh lichess data.

## Prerequisites

- Stockfish at `/usr/games/stockfish` (for analysis)
- Coda binary at `/home/adam/code/coda/coda` built with v5 production net
- python3 with `chess` library

## Pipeline

**1. Fetch lichess games** (both bot accounts):

```bash
DIR=/tmp/coda_blunders
mkdir -p $DIR
curl -s -H "Accept: application/x-chess-pgn" \
  "https://lichess.org/api/games/user/coda_bot?max=200&rated=true&clocks=true&evals=true&perfType=blitz" \
  -o $DIR/coda_bot.pgn
curl -s -H "Accept: application/x-chess-pgn" \
  "https://lichess.org/api/games/user/codabot?max=200&rated=true&clocks=true&evals=true&perfType=blitz" \
  -o $DIR/codabot.pgn
cat $DIR/coda_bot.pgn $DIR/codabot.pgn > $DIR/all_games.pgn
```

**2. SF-analyse every Coda move** (~1.5h for 400 games at depth 16):

```bash
python3 scripts/blunder_suite/analyze_multi.py \
  --pgn $DIR/all_games.pgn \
  --out $DIR/blunders_combined.csv \
  --depth 16 --threads 4 --hash 256 \
  --our coda_bot codabot
```

**3. Categorize + emit EPD** (~10 min for fresh-Coda testing):

Edit `build_epd.py` to point at the new CSV if needed, then:

```bash
python3 scripts/blunder_suite/build_epd.py
```

Output:
- `testdata/coda_blunders.epd` — the suite (only eval_gap positions)
- `testdata/coda_blunders.README.md` — methodology + category counts

## Category definitions

For each SF-flagged blunder in the CSV, fresh Coda (at 3s movetime, Threads=4)
is run on the position:

- **state_pollution**: fresh Coda plays SF's best move. The game-time blunder
  came from TT/history contaminated by prior search state. Addressed by the
  F1 fix (external_stop separation) on `main`.
- **different_wrong**: fresh Coda plays something else (not SF best, not the
  game move). Eval weakness in a different dimension. Not in this suite.
- **eval_gap**: fresh Coda plays the same bad move. Only a better NNUE can
  fix. This is the suite.

## Suite scoring

The built-in Coda EPD runner handles both `bm` and `am` (since commit e88981c):

```bash
./coda epd testdata/coda_blunders.epd -t 3000 -n <your_net>.nnue
```

Pass = engine plays `bm` AND does not play `am`.

## Filter tuning

Defaults in `build_epd.py`:
- `MIN_CP_LOSS = 100` — only blunders (cp_loss >= 1 pawn)
- `SF_CP_BEFORE_LIMIT = 500` — position wasn't already lost/won by 5 pawns
- `MOVETIME = 3000` — fresh Coda has 3s per position
- `SF_DEPTH = 22` — final `bm` verification depth

Loosening any of these increases suite size but also includes weaker signals.
