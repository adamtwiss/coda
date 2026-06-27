---
name: lichess
description: Lichess game data for Coda analysis — downloading games, getting centisecond-precision clocks, parsing for TM/move pattern analysis. Invoke this skill when analysing any lichess game.
---

# Lichess Skill — Game Data for Coda Analysis

**Invoke before**: any `curl https://lichess.org/...` for game data, any
attempt to parse move times / clocks from lichess PGN, any analysis that
needs per-move time precision.

**The pattern these instructions prevent**: silently using integer-only
clock data when centisecond precision is available (rounding errors that
distort spike-ratio and easy-emit analyses), missing the right endpoint
for non-PGN access.

---

## 1. Game IDs and URLs

A lichess game URL is `https://lichess.org/<gameId>` (e.g. `Iq1IFE8x`).
Sometimes followed by `/white` or `/black` indicating the side to view
from — that's UI-only, the gameId is the same.

The API base for game export is `https://lichess.org/game/export/<gameId>`.

---

## 2. **PGN export — INTEGER SECONDS ONLY (avoid for TM analysis)**

```bash
curl -sL "https://lichess.org/game/export/<id>?clocks=1" \
    -H "Accept: application/x-chess-pgn" > game.pgn
```

Returns PGN with `{[%clk H:MM:SS]}` comments **rounded to integer
seconds**. A 0.5s spend appears as 0s or 1s. The integer rounding
distorts:
- Easy-emit floor analysis (p5/p10 ranges 0-1s collapse to 0s)
- Spike ratio (p95/p50) when values are small
- IQR/median (denominator small → ratio noisy)

**Don't use this for per-move TM analysis.** Use ndjson endpoint below.

---

## 3. **NDJSON export — CENTISECOND PRECISION (use this for TM analysis)**

```bash
curl -sL "https://lichess.org/game/export/<id>?clocks=true" \
    -H "Accept: application/x-ndjson" > game.json
```

Returns JSON with a top-level **`clocks` field — array of centiseconds**
(1cs = 10ms = 0.01s). Indexed by half-move (ply): index 0 = white's move
1, index 1 = black's move 1, index 2 = white's move 2, etc.

Example:
```python
import json
data = json.load(open('game.json'))
clocks_cs = data['clocks']
clocks_ms = [c * 10 for c in clocks_cs]
# Now compute spend = prev_clock_ms + inc_ms - cur_clock_ms
```

Other useful fields in the ndjson:
- `moves`: SAN string of all moves (e.g., `"e4 c5 Nc3 d6 ..."`)
- `players.{white,black}.user.name`: engine name
- `clock.initial`: starting time in centiseconds
- `clock.increment`: increment in centiseconds
- `speed` / `perf`: TC class (`bullet` / `blitz` / `rapid` / `classical`)
- `opening.{eco,name,ply}`: opening identification

**Note**: even if URL ends in `/white` or `/black`, the export needs only
the bare gameId — strip the color suffix.

---

## 4. Computing per-move spend from centisecond clocks

```python
INC_MS = clock_increment_centis * 10  # convert centis → ms
INITIAL_MS = clock_initial_centis * 10

prev_w = INITIAL_MS
prev_b = INITIAL_MS
white_spends, black_spends = [], []
for i in range(0, len(clocks_ms), 2):
    wc = clocks_ms[i] if i < len(clocks_ms) else None
    bc = clocks_ms[i+1] if i+1 < len(clocks_ms) else None
    if wc is not None:
        spend = prev_w + INC_MS - wc
        white_spends.append(spend); prev_w = wc
    if bc is not None:
        spend = prev_b + INC_MS - bc
        black_spends.append(spend); prev_b = bc
```

Spend can be slightly negative if the engine moved within the increment
grace; clamp to 0 if needed for downstream analysis.

---

## 5. Useful metrics for TM analysis

For each side, compute:
- **p5 / p10**: easy-emit floor (lower = more instant moves; top engines
  hit 0.08-0.20s, Coda currently 1.2-1.5s as of 2026-05-25)
- **p50**: median spend
- **p95 / max**: critical-move ceiling (top engines reach 25-35s at blitz
  TCs; Coda currently maxes ~10s with Phase 10h)
- **p95/p50**: "spike ratio" — sharper-is-better; top engines 5-6×, Coda
  currently 3-3.5×
- **IQR/p50**: dispersion proxy
- **Total time used**: sum of spends; should be ≤ initial + inc × moves

---

## 6. Pulling multiple games for a bot

Lichess provides a games-by-user endpoint:

```bash
# Last 50 games as ndjson (with clocks):
curl -sL "https://lichess.org/api/games/user/<botname>?max=50&clocks=true" \
    -H "Accept: application/x-ndjson" > games.ndjson
```

Each line is a separate game's JSON. Loop with:
```python
with open('games.ndjson') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        clocks_cs = data.get('clocks', [])
        # ... analyze
```

Useful for batch analysis across many games (e.g. confirming TM patterns
hold across a session, finding outlier games for inspection).

---

## 7. Coda bot accounts

- **codabot** — production / stable account
- **coda_bot** — experimental account (typically gets new features first
  for live A/B comparison)

Use these in `https://lichess.org/api/games/user/<name>?...` queries for
batch session analysis.

---

## 8. Common analysis recipes

### Compare two Coda variants via lichess A/B
```python
# Pull last N games from each bot
# Compute per-game metrics (spend distribution, clock-remaining curve)
# Compare medians of metrics across game sets
```

### Detect overspend pattern in a specific game
```python
# Get per-move spends
# Compute median, then count moves >2× median
# These are the "overspend" moves; look at the run-length
```

### Verify Phase 10h clip is firing
```python
# For each move, compute scale × soft (need to model this) vs hard/2 clip
# If clip would fire, log the move
# Compare across pre-10h and post-10h game samples
```

---

## 9. Common gotchas

- **Don't use integer PGN clocks** for spike-ratio / easy-emit analysis —
  rounding distortion is severe. Use the ndjson endpoint (§3).
- **Color suffix in URL is UI-only** (`/white`/`/black`) — strip it for
  the API.
- **Clock at index N is AFTER that ply was played** — so spend on move N
  is `prev_clock + inc - clocks[N]`, not the other way around.
- **First move's spend** may show as inc-equal (e.g. 2s on a 2+2 game)
  because the player gets inc before move 1 in some clock implementations.
- **Spend can be slightly negative** if move was made within increment
  grace window — clamp to 0 or take `max(0, spend)`.
- **Lichess rate limits**: don't hammer the API. For batch pulls, the
  user-games endpoint has stricter limits than per-game export.
