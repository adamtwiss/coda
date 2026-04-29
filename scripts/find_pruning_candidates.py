#!/usr/bin/env python3
"""Find pruning-blind-spot candidates from SF-arbitrated CSV + source PGN.

Per Adam's framing (2026-04-28):
  - Very gradual erosion (small drops accumulating) = eval (NNUE) deficit
  - Moderate stepped (-50 to -100cp single drops) = pruning blind spots
  - Sudden drops (>-100cp single moves) = tactical/search blunders

The actionable subset for search-side improvement is the moderate-stepped
class: moments where Coda crossed below -100cp via a single -50 to -100cp
drop, where the SF bestmove differed from Coda's chosen move. These are
candidates for "we pruned the right move" investigation.

Output: JSONL with FEN-before, Coda's move, SF's best, eval before/after,
ply, opponent. Ready for blunder_ablation.py to probe each.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn


def walk_game_fens(game: chess.pgn.Game) -> dict[int, str]:
    """Return {ply: fen_before_move} for each move in the game."""
    fens = {}
    board = game.board()
    ply = 0
    node = game
    while node.variations:
        next_node = node.variation(0)
        ply += 1
        fens[ply] = board.fen()
        board.push(next_node.move)
        node = next_node
    return fens


def coda_color(game):
    if "Coda" in game.headers.get("White", ""):
        return chess.WHITE
    if "Coda" in game.headers.get("Black", ""):
        return chess.BLACK
    return None


def coda_lost(game):
    c = coda_color(game)
    if c is None:
        return False
    r = game.headers.get("Result", "*")
    return (c == chess.WHITE and r == "0-1") or (c == chess.BLACK and r == "1-0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", type=Path)
    ap.add_argument("csv", type=Path, help="SF-arbitrated CSV from sf_arbitrate_blunders.py")
    ap.add_argument("--out", type=Path, required=True, help="JSONL output")
    ap.add_argument("--mode", choices=["moderate_step", "sudden", "gradual"],
                    default="moderate_step",
                    help="Which crossing class: moderate_step=-50to-100 single drop crossing -100, "
                         "sudden=≤-100 single drop crossing -100, gradual=<-50 drop crossing -100")
    args = ap.parse_args()

    # Load CSV, group by (game_idx, opponent)
    rows = list(csv.DictReader(args.csv.open()))
    for r in rows:
        try:
            r["delta"] = int(r["delta_cp"]) if r["delta_cp"] not in ("", "None", None) else None
            r["ply"] = int(r["ply"])
            r["score_before"] = int(r["sf_score_before"]) if r["sf_score_before"] not in ("", "None", None) else None
            r["score_after"] = int(r["sf_score_after_codapov"]) if r["sf_score_after_codapov"] not in ("", "None", None) else None
        except (ValueError, KeyError):
            r["delta"] = None

    games = defaultdict(list)
    for r in rows:
        if r["delta"] is not None and r["score_after"] is not None:
            games[int(r["game_idx"])].append(r)
    for k, gmoves in games.items():
        gmoves.sort(key=lambda r: r["ply"])

    # For each game, find the FIRST move where score_after first crosses below -100.
    # Classify by single-move delta size at that crossing.
    candidates = []
    for game_idx, gmoves in games.items():
        for r in gmoves:
            if r["score_after"] >= -100:
                continue
            # First crossing
            d = r["delta"]
            if args.mode == "moderate_step":
                if not (-100 < d <= -50):
                    break
            elif args.mode == "sudden":
                if d > -100:
                    break
            elif args.mode == "gradual":
                if d <= -50:
                    break
            candidates.append({
                "game_idx": game_idx,
                "opponent": r["opponent"],
                "ply": r["ply"],
                # blunder_uci/san match blunder_ablation.py's expected schema
                "blunder_uci": r["coda_uci"],
                "blunder_san": r["coda_san"],
                "sf_best_uci": r["sf_best_before"],
                "score_before": r["score_before"],
                "score_after": r["score_after"],
                "delta_cp": d,
            })
            break

    print(f"Found {len(candidates)} candidates (mode={args.mode})")

    # Re-walk PGN to recover FENs for the candidate plies
    by_game = {c["game_idx"]: c for c in candidates}

    enriched = []
    with args.pgn.open() as f:
        gi = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            color = coda_color(game)
            if color is None:
                continue
            if not coda_lost(game):
                gi += 1
                continue
            if gi in by_game:
                fens = walk_game_fens(game)
                c = by_game[gi]
                fen = fens.get(c["ply"])
                if fen:
                    c["fen_before"] = fen
                    enriched.append(c)
            gi += 1

    print(f"Enriched {len(enriched)} candidates with FEN")

    with args.out.open("w") as f:
        for c in enriched:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {args.out}")

    # Print first 5 as preview
    print(f"\n=== First 5 candidates ===")
    for c in enriched[:5]:
        print(f"\n  Game {c['game_idx']} vs {c['opponent']} ply {c['ply']}")
        print(f"    FEN: {c['fen_before']}")
        print(f"    Coda played: {c['blunder_san']} ({c['blunder_uci']})")
        print(f"    SF best: {c['sf_best_uci']}")
        print(f"    Eval: {c['score_before']} → {c['score_after']} (Δ {c['delta_cp']})")


if __name__ == "__main__":
    main()
