#!/usr/bin/env python3
"""Classify losses from cutechess PGN logs into horizon vs positional patterns.

For each game where Coda lost (or drew when expected to win), extract per-ply
eval scores from both engines and classify the loss mechanism:

- HORIZON: sharp eval cliff (one engine sees a tactic the other doesn't yet).
  Signature: opponent's eval shifts 100+cp at ply N, Coda's eval shifts
  similarly at ply N+1 to N+3. The opponent saw it earlier.

- POSITIONAL: gradual eval divergence with no single cliff. Both engines drift
  apart over 20+ plies, both initially "fine" but at different magnitudes.
  Signature: max single-ply delta < 50cp, but cumulative divergence > 200cp.

- MIXED: features of both, or eval data too sparse to classify cleanly.

Usage:
  ./classify_losses.py game1.pgn game2.pgn ...
  ./classify_losses.py --opponent Stockfish *.pgn
  ./classify_losses.py --csv-out losses.csv *.pgn

PGN format expected (cutechess default with eval logging enabled):
  1. e4 {+0.20/12 0.5s} e5 {-0.18/11 0.4s} 2. Nf3 {+0.25/13 0.5s} ...

Tags expected: [White], [Black], [Result].
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Eval comment pattern: {+0.20/12 0.5s} or {-1.50/15} or {+M5/20} (mate)
# Cutechess emits eval in pawns (not centipawns) by default; we convert.
EVAL_RE = re.compile(
    r"\{"
    r"([+-]?(?:M|\d+(?:\.\d+)?))"  # eval: number or M<n> for mate
    r"(?:/(\d+))?"                   # depth
    r"(?:\s+(?:[\d.]+s|book))?"     # time, book marker, etc.
    r"\}"
)
TAG_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
MOVE_RE = re.compile(r"(?:^|\s)\d+\.+\s*([^\s{]+)")


@dataclass
class GameMove:
    ply: int
    side_to_move: str  # "W" or "B"
    san: str
    eval_cp: int | None  # centipawns from side-to-move POV; None if missing
    depth: int | None
    is_mate: bool = False


@dataclass
class Game:
    white: str
    black: str
    result: str
    moves: list[GameMove] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def coda_color(self) -> str | None:
        if "Coda" in self.white:
            return "W"
        if "Coda" in self.black:
            return "B"
        return None

    @property
    def opponent(self) -> str:
        c = self.coda_color
        if c == "W":
            return self.black
        if c == "B":
            return self.white
        return "?"

    @property
    def coda_lost(self) -> bool:
        c = self.coda_color
        if c == "W":
            return self.result == "0-1"
        if c == "B":
            return self.result == "1-0"
        return False


def parse_eval(eval_str: str) -> tuple[int | None, bool]:
    """Parse cutechess eval string. Returns (centipawns, is_mate).

    "+0.50" -> (50, False)
    "-M5" -> (-30000, True)  # mate values get a sentinel
    """
    eval_str = eval_str.strip().replace("+", "")
    if "M" in eval_str:
        sign = -1 if eval_str.startswith("-") else 1
        return (sign * 30000, True)
    try:
        pawns = float(eval_str)
        return (int(pawns * 100), False)
    except ValueError:
        return (None, False)


def parse_pgn_file(path: Path) -> Iterator[Game]:
    """Parse a PGN file with possibly many games. Yields Game objects."""
    with path.open() as f:
        content = f.read()

    # Split games on blank line + [Event tag pattern (best-effort)
    games_raw = re.split(r"\n\n(?=\[Event)", content)

    for raw in games_raw:
        if not raw.strip():
            continue
        game = Game(white="", black="", result="*")
        for tag_m in TAG_RE.finditer(raw):
            game.headers[tag_m.group(1)] = tag_m.group(2)
        game.white = game.headers.get("White", "?")
        game.black = game.headers.get("Black", "?")
        game.result = game.headers.get("Result", "*")

        # Strip headers; keep movetext
        movetext = re.sub(r"\[[^\]]*\]\s*", "", raw, flags=re.MULTILINE)

        # Walk through movetext token by token, tracking ply
        ply = 0
        # Tokenise: move tokens or {...} eval comments
        tokens = re.findall(r"\{[^}]*\}|[^\s]+", movetext)
        last_san: str | None = None
        for tok in tokens:
            if tok.startswith("{"):
                # Eval comment for the prior move
                em = EVAL_RE.match(tok)
                if em and last_san is not None:
                    eval_cp, is_mate = parse_eval(em.group(1))
                    depth = int(em.group(2)) if em.group(2) else None
                    side_to_move = "W" if ply % 2 == 1 else "B"  # ply just incremented
                    game.moves.append(GameMove(
                        ply=ply,
                        side_to_move=side_to_move,  # whose move was just played
                        san=last_san,
                        eval_cp=eval_cp,
                        depth=depth,
                        is_mate=is_mate,
                    ))
                    last_san = None
            elif re.match(r"^\d+\.+$", tok):
                continue  # move number
            elif tok in ("1-0", "0-1", "1/2-1/2", "*"):
                continue  # result
            elif re.match(r"^[a-zA-Z]", tok):
                # Move SAN
                last_san = tok
                ply += 1
        yield game


def coda_eval_series(game: Game) -> list[tuple[int, int]]:
    """Return [(ply, coda_eval_cp_from_white_pov), ...] for moves where
    Coda was the side that just moved (so Coda's eval is the comment).
    Eval is normalised to white-POV centipawns."""
    series = []
    coda_color = game.coda_color
    if coda_color is None:
        return series
    for m in game.moves:
        if m.eval_cp is None:
            continue
        if m.side_to_move != coda_color:
            continue
        # cutechess eval is from side-to-move POV; flip if Coda is black
        cp = m.eval_cp if coda_color == "W" else -m.eval_cp
        series.append((m.ply, cp))
    return series


def opponent_eval_series(game: Game) -> list[tuple[int, int]]:
    """Same as coda_eval_series but for the opponent."""
    series = []
    coda_color = game.coda_color
    if coda_color is None:
        return series
    opp_color = "B" if coda_color == "W" else "W"
    for m in game.moves:
        if m.eval_cp is None or m.side_to_move != opp_color:
            continue
        cp = m.eval_cp if opp_color == "W" else -m.eval_cp
        series.append((m.ply, cp))
    return series


def classify(game: Game) -> dict:
    """Classify a Coda-lost game.

    Returns a dict with classification + diagnostic features.
    """
    coda_series = coda_eval_series(game)
    opp_series = opponent_eval_series(game)

    if len(coda_series) < 5 or len(opp_series) < 5:
        return {"class": "INSUFFICIENT_DATA", "n_coda_evals": len(coda_series),
                "n_opp_evals": len(opp_series)}

    # Build a unified per-ply view by interpolating between observations.
    # Cleaner: for each Coda eval at ply P, find opp eval at nearest ply <= P.
    opp_dict = dict(opp_series)
    opp_plies = sorted(opp_dict.keys())

    def opp_at_or_before(ply: int) -> int | None:
        """Most recent opponent eval at or before this ply."""
        candidates = [p for p in opp_plies if p <= ply]
        return opp_dict[candidates[-1]] if candidates else None

    # Per-Coda-move features:
    coda_deltas = []
    paired = []
    prev_cp = None
    for ply, cp in coda_series:
        if prev_cp is not None:
            coda_deltas.append((ply, cp - prev_cp))
        opp_cp = opp_at_or_before(ply)
        if opp_cp is not None:
            paired.append((ply, cp, opp_cp))
        prev_cp = cp

    # Same for opponent:
    opp_deltas = []
    prev_cp = None
    for ply, cp in opp_series:
        if prev_cp is not None:
            opp_deltas.append((ply, cp - prev_cp))
        prev_cp = cp

    # Feature: max single-step Coda eval drop (more negative = worse for Coda)
    # Coda is the side losing here; Coda's eval should drop (white-POV: white losing → eval falls; black losing → eval rises but Coda-POV is opposite)
    # Simpler: track abs deltas; the cliff is the biggest abs change in Coda's favour-to-disfavour direction.
    coda_color = game.coda_color
    coda_pov_evals = [(ply, cp if coda_color == "W" else -cp) for ply, cp in coda_series]
    opp_pov_evals = [(ply, cp if coda_color == "B" else -cp) for ply, cp in opp_series]

    coda_pov_deltas = [(coda_pov_evals[i][0], coda_pov_evals[i][1] - coda_pov_evals[i-1][1])
                       for i in range(1, len(coda_pov_evals))]
    opp_pov_deltas = [(opp_pov_evals[i][0], opp_pov_evals[i][1] - opp_pov_evals[i-1][1])
                      for i in range(1, len(opp_pov_evals))]

    # Cliff: most negative Coda delta from Coda's POV (we just realised something is bad)
    max_coda_drop_ply, max_coda_drop = (0, 0)
    if coda_pov_deltas:
        max_coda_drop_ply, max_coda_drop = min(coda_pov_deltas, key=lambda x: x[1])
    # Opponent eval shift (positive from opponent POV = good for them = bad for us)
    # opp_pov_eval is Coda-POV → opponent's eval inversion. opp_pov_delta > 0 means opp got happier
    # Cliff for opp = biggest drop from Coda-POV (opp's eval rose) → most negative Coda-POV delta
    max_opp_shift_ply, max_opp_shift = (0, 0)
    if opp_pov_deltas:
        max_opp_shift_ply, max_opp_shift = min(opp_pov_deltas, key=lambda x: x[1])

    # Mean opp-vs-coda eval gap over last 30% of game (positional drift indicator)
    n = len(paired)
    tail = paired[int(n * 0.7):] if n >= 10 else paired
    if tail:
        # Both already in white-POV (paired uses original cp). Convert to disagreement:
        # gap from Coda's POV: opp_cp_from_coda_pov - coda_cp_from_coda_pov
        gaps = []
        for ply, coda_cp, opp_cp in tail:
            coda_pov_coda = coda_cp if coda_color == "W" else -coda_cp
            coda_pov_opp = opp_cp if coda_color == "W" else -opp_cp
            # Disagreement: opp thinks position is X, Coda thinks Y. From a neutral observer:
            # both should converge to the truth. Persistent disagreement = positional.
            gaps.append(abs(coda_pov_opp - coda_pov_coda))
        mean_late_gap = sum(gaps) / len(gaps)
    else:
        mean_late_gap = 0

    # Lag detection: did opp's cliff come before Coda's?
    lag = None
    if abs(max_opp_shift) > 100 and abs(max_coda_drop) > 100:
        lag = max_coda_drop_ply - max_opp_shift_ply

    # Classification heuristic. Key insight: the OPPONENT in a horizon-class
    # loss usually does NOT have a cliff — their eval was already correctly
    # bad for us, because they saw it ahead. The signal is the persistent
    # disagreement (mean_late_gap), not paired cliffs.
    HORIZON_CLIFF = -150  # Coda eval drop in cp (single ply or 3-ply window)
    POSITIONAL_GAP = 100  # late-game disagreement in cp = opp saw what we didn't
    MUTUAL_OPP_CLIFF = -200  # if opp ALSO has a cliff this large, mutual tactic

    if mean_late_gap >= POSITIONAL_GAP:
        # Opponent's late-game eval significantly disagreed with ours — they
        # saw things we didn't. Distinguish "we eventually caught up via a
        # cliff" (HORIZON) from "we never had a cliff, just drifted lost"
        # (POSITIONAL).
        if max_coda_drop <= HORIZON_CLIFF:
            klass = "HORIZON"  # opp had a horizon advantage; Coda eventually caught up
        else:
            klass = "POSITIONAL"  # gradual drift, no Coda cliff; opp's eval was right all along
    else:
        # Late-game evals roughly agreed. Either nobody saw the loss until
        # very late (mutual blunder / forced-by-time), or the loss was
        # localized to a single tactical moment.
        if max_coda_drop <= HORIZON_CLIFF and max_opp_shift <= MUTUAL_OPP_CLIFF:
            klass = "MUTUAL_TACTIC"  # both engines cliffed at similar time
        elif max_coda_drop <= HORIZON_CLIFF:
            klass = "SELF_BLUNDER"  # Coda cliffed, opp didn't even know it was coming
        else:
            klass = "UNCLASSIFIED"

    return {
        "class": klass,
        "opponent": game.opponent,
        "result": game.result,
        "max_coda_drop_cp": max_coda_drop,
        "max_coda_drop_ply": max_coda_drop_ply,
        "max_opp_shift_cp": max_opp_shift,
        "max_opp_shift_ply": max_opp_shift_ply,
        "lag_plies": lag,
        "mean_late_gap_cp": int(mean_late_gap),
        "n_moves": len(coda_series),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pgn", nargs="+", help="PGN files to analyse")
    p.add_argument("--opponent", help="Filter to this opponent name (substring match)")
    p.add_argument("--csv-out", help="Write per-game results to CSV")
    p.add_argument("--include-draws", action="store_true",
                   help="Include drawn games (otherwise only Coda losses)")
    args = p.parse_args()

    rows = []
    for pgn_path in args.pgn:
        for game in parse_pgn_file(Path(pgn_path)):
            if game.coda_color is None:
                continue
            if not args.include_draws and not game.coda_lost:
                continue
            if args.opponent and args.opponent not in game.opponent:
                continue
            result = classify(game)
            result["white"] = game.white
            result["black"] = game.black
            rows.append(result)

    # Aggregate
    by_class: dict[str, int] = {}
    by_opp_class: dict[tuple[str, str], int] = {}
    for r in rows:
        by_class[r["class"]] = by_class.get(r["class"], 0) + 1
        key = (r.get("opponent", "?"), r["class"])
        by_opp_class[key] = by_opp_class.get(key, 0) + 1

    print(f"\n=== Classification of {len(rows)} losses ===")
    print(f"\nOverall:")
    for klass, n in sorted(by_class.items(), key=lambda x: -x[1]):
        pct = 100 * n / len(rows) if rows else 0
        print(f"  {klass:20s} {n:4d}  ({pct:5.1f}%)")

    print(f"\nBy opponent:")
    opps = sorted({k[0] for k in by_opp_class})
    classes = sorted({k[1] for k in by_opp_class})
    print(f"  {'Opponent':40s} " + " ".join(f"{c:14s}" for c in classes))
    for opp in opps:
        opp_total = sum(n for (o, _), n in by_opp_class.items() if o == opp)
        cells = []
        for c in classes:
            n = by_opp_class.get((opp, c), 0)
            pct = 100 * n / opp_total if opp_total else 0
            cells.append(f"{n:3d} ({pct:4.1f}%)")
        print(f"  {opp:40s} " + " ".join(f"{c:14s}" for c in cells))

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as f:
            if not rows:
                print(f"No rows to write to {args.csv_out}", file=sys.stderr)
                return 0
            keys = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
