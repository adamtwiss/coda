#!/usr/bin/env python3
"""Extract blunder positions from a cutechess PGN.

For each Coda-lost game classified as HORIZON (or any other class via
--include-class), identify the cliff ply — the move where Coda's eval
dropped most sharply from Coda's POV — and emit:

  - FEN at the position Coda was about to move from (parent of cliff ply)
  - The move Coda played (the suspected blunder, in UCI form)
  - The eval shift (cp from Coda POV, negative = Coda's view degraded)
  - Game metadata (opponent, ply, result)

Output is JSONL so the ablation harness can consume it directly.

Usage:
  ./extract_blunder_positions.py tough_rivals.pgn --out blunders.jsonl
  ./extract_blunder_positions.py tough_rivals.pgn --out blunders.jsonl --class HORIZON
"""

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

# python-chess strips the curly braces, so match the bare comment form.
EVAL_RE = re.compile(
    r"^\s*"
    r"([+-]?(?:M|\d+(?:\.\d+)?))"
    r"(?:/(\d+))?"
    r"(?:\s+(?:[\d.]+s|book))?"
)


def parse_eval_cp(eval_str: str) -> tuple[int | None, bool]:
    """Returns (cp, is_mate). cp is from side-to-move POV in centipawns."""
    eval_str = eval_str.strip().replace("+", "")
    if "M" in eval_str:
        sign = -1 if eval_str.startswith("-") else 1
        return sign * 30000, True
    try:
        return int(float(eval_str) * 100), False
    except ValueError:
        return None, False


def games_from_pgn(path: Path) -> Iterator[chess.pgn.Game]:
    with path.open() as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            yield game


def coda_color(game: chess.pgn.Game) -> chess.Color | None:
    if "Coda" in game.headers.get("White", ""):
        return chess.WHITE
    if "Coda" in game.headers.get("Black", ""):
        return chess.BLACK
    return None


def coda_lost(game: chess.pgn.Game) -> bool:
    c = coda_color(game)
    if c is None:
        return False
    result = game.headers.get("Result", "*")
    return (c == chess.WHITE and result == "0-1") or \
           (c == chess.BLACK and result == "1-0")


@dataclass
class CodaMove:
    ply: int  # 1-indexed half-move number after the move
    move: chess.Move
    san: str
    eval_cp_stm: int | None  # from side-to-move-after POV (cutechess default)
    eval_cp_white: int | None
    is_mate: bool
    fen_before: str  # FEN of the position Coda chose this move from


def coda_move_series(game: chess.pgn.Game) -> list[CodaMove]:
    """Walk the game; collect Coda's moves with the FEN at parent and eval."""
    color = coda_color(game)
    if color is None:
        return []
    out: list[CodaMove] = []
    board = game.board()
    node = game
    ply = 0
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        san = node.board().san(move)
        ply += 1
        # Whose turn was it BEFORE this move?
        side_moved = board.turn  # board.turn = next side to move = side that just moved? No: turn is who's TO MOVE
        # board.turn before push = side that's about to move
        if side_moved == color:
            fen_before = board.fen()
            comment = next_node.comment or ""
            em = EVAL_RE.search(comment)
            cp_stm = None
            is_mate = False
            if em:
                cp_stm, is_mate = parse_eval_cp(em.group(1))
            cp_white = None
            if cp_stm is not None:
                # cutechess emits eval from the POV of the side that JUST MOVED
                # (i.e., from the perspective of the engine that produced the
                # comment). Coda's eval is from Coda's POV.
                cp_white = cp_stm if color == chess.WHITE else -cp_stm
            out.append(CodaMove(
                ply=ply,
                move=move,
                san=san,
                eval_cp_stm=cp_stm,
                eval_cp_white=cp_white,
                is_mate=is_mate,
                fen_before=fen_before,
            ))
        board.push(move)
        node = next_node
    return out


def find_cliff(coda_moves: list[CodaMove]) -> int | None:
    """Return the index in coda_moves of the cliff move — the one where
    Coda's eval (from Coda's POV) dropped most sharply vs the prior move.

    Returns None if eval data is too sparse to identify a cliff.
    """
    # Coda POV: white-POV eval if Coda is white, else negated.
    # We already store eval_cp_white. Coda POV = sign-aligned with Coda's
    # winning probability: positive = Coda winning.
    # But we need Coda's own POV: if Coda is black, Coda POV = -white POV.
    # Easier: cutechess eval_cp_stm IS already from the engine's POV (Coda's
    # POV), so directly compute deltas on eval_cp_stm. Negative delta = bad
    # news for Coda.
    evals = [(i, m.eval_cp_stm) for i, m in enumerate(coda_moves) if m.eval_cp_stm is not None]
    if len(evals) < 3:
        return None
    # Mate scores can dominate; clamp to ±2000 for delta purposes
    clamped = [(i, max(-2000, min(2000, cp))) for i, cp in evals]
    deltas = [(clamped[k][0], clamped[k][1] - clamped[k - 1][1]) for k in range(1, len(clamped))]
    if not deltas:
        return None
    cliff_idx, cliff_delta = min(deltas, key=lambda x: x[1])
    # Only count if it's a real drop (>= 100cp)
    if cliff_delta > -100:
        return None
    return cliff_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="JSONL output path")
    ap.add_argument("--min-drop-cp", type=int, default=100,
                    help="Minimum eval drop in cp to count as a cliff (default 100)")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of blunders extracted")
    args = ap.parse_args()

    n_games = 0
    n_lost = 0
    n_blunders = 0
    rows = []

    for game in games_from_pgn(args.pgn):
        n_games += 1
        if not coda_lost(game):
            continue
        n_lost += 1
        coda_moves = coda_move_series(game)
        cliff_idx = find_cliff(coda_moves)
        if cliff_idx is None:
            continue
        cliff_move = coda_moves[cliff_idx]
        prev_eval = coda_moves[cliff_idx - 1].eval_cp_stm if cliff_idx > 0 else None
        delta = (cliff_move.eval_cp_stm - prev_eval) if prev_eval is not None else None
        rows.append({
            "opponent": game.headers.get("White") if "Coda" in game.headers.get("Black", "")
                                                else game.headers.get("Black"),
            "result": game.headers.get("Result"),
            "ply": cliff_move.ply,
            "fen_before": cliff_move.fen_before,
            "blunder_uci": cliff_move.move.uci(),
            "blunder_san": cliff_move.san,
            "eval_before_cp": prev_eval,
            "eval_after_cp": cliff_move.eval_cp_stm,
            "delta_cp": delta,
            "is_mate": cliff_move.is_mate,
            "n_coda_moves": len(coda_moves),
        })
        n_blunders += 1
        if args.limit and n_blunders >= args.limit:
            break

    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Parsed {n_games} games, {n_lost} Coda losses, {n_blunders} cliffs extracted")
    print(f"Wrote {len(rows)} positions to {args.out}")
    if rows:
        # Quick distribution
        deltas = sorted([r["delta_cp"] for r in rows if r["delta_cp"] is not None])
        if deltas:
            print(f"Cliff delta distribution: min={deltas[0]} median={deltas[len(deltas)//2]} max={deltas[-1]}")


if __name__ == "__main__":
    main()
