#!/usr/bin/env python3
"""SF-arbitrated blunder analysis on a cutechess PGN.

For each Coda move in each game, run Stockfish at fixed depth on:
  - the FEN BEFORE the move (gives SF's eval of the position)
  - the FEN AFTER the move (gives SF's eval after Coda's choice)

ΔSF = (SF eval at fen_after, sign-flipped to Coda POV) - (SF eval at fen_before)

Negative ΔSF = Coda's move lost ground per SF arbiter:
  - 0 to -50 cp:    accurate
  - -50 to -100:    inaccuracy
  - -100 to -200:   mistake
  - -200 cp or more: blunder

Output: CSV with one row per Coda move + summary statistics per opponent.

Usage:
  ./sf_arbitrate_blunders.py tough_rivals.pgn --out arbitrated.csv \\
    --depth 18 --workers 14 --losses-only
"""

import argparse
import csv
import json
import multiprocessing as mp
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn

DEFAULT_SF = "/home/adam/chess/engines/Stockfish/src/stockfish"
DEFAULT_DEPTH = 18
DEFAULT_HASH_MB = 128

INACCURACY_CP = 50
MISTAKE_CP = 100
BLUNDER_CP = 200


def tier_of(delta_cp: int) -> str:
    """Classify a Coda move by ΔSF magnitude (negative = bad)."""
    if delta_cp is None:
        return "UNKNOWN"
    if delta_cp >= -INACCURACY_CP:
        return "ACCURATE"
    if delta_cp >= -MISTAKE_CP:
        return "INACCURACY"
    if delta_cp >= -BLUNDER_CP:
        return "MISTAKE"
    return "BLUNDER"


class SFEngine:
    """Persistent Stockfish UCI subprocess with go-depth probe."""

    def __init__(self, sf_path: str, depth: int, hash_mb: int = DEFAULT_HASH_MB):
        self.depth = depth
        self.proc = subprocess.Popen(
            [sf_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until_token("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name Threads value 1")
        self._send("isready")
        self._read_until_token("readyok")

    def _send(self, cmd: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("SF stdin closed")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_line(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("SF stdout closed")
        return self.proc.stdout.readline()

    def _read_until_token(self, token: str) -> None:
        for _ in range(2000):
            line = self._read_line()
            if not line:
                raise RuntimeError(f"SF closed before {token!r}")
            if token in line:
                return
        raise RuntimeError(f"SF didn't emit {token!r} in 2000 lines")

    def eval_position(self, fen: str) -> tuple[int | None, str | None, int]:
        """Run SF on fen at fixed depth. Returns (cp_score_from_stm_pov, bestmove_uci, depth_reached).
        cp_score is from side-to-move's POV (positive = stm winning)."""
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        last_score = None
        last_depth = 0
        bestmove = None
        for _ in range(50000):
            line = self._read_line()
            if not line:
                raise RuntimeError("SF closed mid-search")
            line = line.strip()
            if line.startswith("info "):
                # Parse score (last seen wins)
                m = re.search(r"\bdepth (\d+)\b", line)
                if m:
                    d = int(m.group(1))
                    sm = re.search(r"score (cp|mate) (-?\d+)", line)
                    if sm:
                        last_depth = d
                        if sm.group(1) == "cp":
                            last_score = int(sm.group(2))
                        else:
                            mn = int(sm.group(2))
                            # Mate score → ±10000-ish, sign matches stm winning/losing
                            last_score = (10000 if mn > 0 else -10000)
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                return last_score, bestmove, last_depth
        raise RuntimeError("SF didn't produce bestmove within line cap")

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


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
    r = game.headers.get("Result", "*")
    return (c == chess.WHITE and r == "0-1") or (c == chess.BLACK and r == "1-0")


def coda_won(game: chess.pgn.Game) -> bool:
    c = coda_color(game)
    if c is None:
        return False
    r = game.headers.get("Result", "*")
    return (c == chess.WHITE and r == "1-0") or (c == chess.BLACK and r == "0-1")


@dataclass
class CodaMoveTask:
    game_idx: int
    opponent: str
    result: str
    coda_color_w: bool  # True = white, False = black
    ply: int
    fen_before: str
    fen_after: str
    coda_uci: str
    coda_san: str


def extract_tasks(pgn: Path, losses_only: bool, wins_only: bool = False) -> list[CodaMoveTask]:
    """Walk all Coda moves in PGN; emit (FEN-before, FEN-after, move) tasks."""
    tasks: list[CodaMoveTask] = []
    with pgn.open() as f:
        gi = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            color = coda_color(game)
            if color is None:
                continue
            if losses_only and not coda_lost(game):
                gi += 1
                continue
            if wins_only and not coda_won(game):
                gi += 1
                continue
            opponent = game.headers.get("White") if color == chess.BLACK else game.headers.get("Black")
            result = game.headers.get("Result", "*")
            board = game.board()
            ply = 0
            node = game
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                ply += 1
                if board.turn == color:
                    fen_before = board.fen()
                    san = node.board().san(move)
                    board.push(move)
                    fen_after = board.fen()
                    tasks.append(CodaMoveTask(
                        game_idx=gi,
                        opponent=opponent or "?",
                        result=result,
                        coda_color_w=(color == chess.WHITE),
                        ply=ply,
                        fen_before=fen_before,
                        fen_after=fen_after,
                        coda_uci=move.uci(),
                        coda_san=san,
                    ))
                else:
                    board.push(move)
                node = next_node
            gi += 1
    return tasks


def worker(task_chunk: list[CodaMoveTask], sf_path: str, depth: int) -> list[dict]:
    sf = SFEngine(sf_path, depth)
    rows = []
    try:
        for t in task_chunk:
            try:
                score_before, bm_before, _ = sf.eval_position(t.fen_before)
                score_after, _, _ = sf.eval_position(t.fen_after)
                # Both scores are from side-to-move POV at their respective FENs.
                # At fen_before, stm = Coda; score_before = Coda's-POV score (good for Coda = positive).
                # At fen_after, stm = opponent; score_after = opponent's-POV score (good for opponent = positive).
                # Convert score_after to Coda's POV: -score_after.
                if score_before is None or score_after is None:
                    delta = None
                    coda_pov_after = None
                else:
                    coda_pov_after = -score_after
                    delta = coda_pov_after - score_before
                rows.append({
                    "game_idx": t.game_idx,
                    "opponent": t.opponent,
                    "result": t.result,
                    "ply": t.ply,
                    "coda_color": "W" if t.coda_color_w else "B",
                    "coda_uci": t.coda_uci,
                    "coda_san": t.coda_san,
                    "sf_score_before": score_before,
                    "sf_best_before": bm_before,
                    "sf_score_after_codapov": coda_pov_after,
                    "delta_cp": delta,
                    "tier": tier_of(delta) if delta is not None else "UNKNOWN",
                })
            except Exception as e:
                rows.append({
                    "game_idx": t.game_idx,
                    "opponent": t.opponent,
                    "result": t.result,
                    "ply": t.ply,
                    "coda_uci": t.coda_uci,
                    "coda_san": t.coda_san,
                    "tier": "ERROR",
                    "delta_cp": None,
                    "sf_score_before": None,
                    "sf_best_before": None,
                    "sf_score_after_codapov": None,
                    "coda_color": "W" if t.coda_color_w else "B",
                    "error": str(e),
                })
    finally:
        sf.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sf", default=DEFAULT_SF, help="Stockfish binary path")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--losses-only", action="store_true",
                    help="Only analyse games where Coda lost")
    ap.add_argument("--wins-only", action="store_true",
                    help="Only analyse games where Coda won")
    ap.add_argument("--limit-games", type=int, default=0,
                    help="Cap to first N matching games (for quick pilot)")
    args = ap.parse_args()

    print(f"Extracting tasks from {args.pgn}...")
    tasks = extract_tasks(args.pgn, args.losses_only, args.wins_only)
    if args.limit_games:
        # Filter to first N unique game_idxs
        keep_games = set()
        kept = []
        for t in tasks:
            if t.game_idx in keep_games or len(keep_games) < args.limit_games:
                keep_games.add(t.game_idx)
                kept.append(t)
        tasks = kept

    print(f"  {len(tasks)} Coda-move tasks across {len({t.game_idx for t in tasks})} games")
    print(f"  SF depth {args.depth}, {args.workers} workers")

    # Chunk tasks across workers
    chunks: list[list[CodaMoveTask]] = [[] for _ in range(args.workers)]
    for i, t in enumerate(tasks):
        chunks[i % args.workers].append(t)

    t0 = time.time()
    rows: list[dict] = []
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(worker, [(chunk, args.sf, args.depth) for chunk in chunks])
        for r in results:
            rows.extend(r)
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min ({len(tasks)/elapsed:.1f} probes/s)")

    # Write CSV
    if rows:
        keys = list(rows[0].keys())
        for r in rows:
            for k in keys:
                r.setdefault(k, None)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}")

    # Summary: per-opponent tier counts
    print("\n=== Per-opponent move-quality breakdown (SF arbitrated) ===")
    by_opp: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_opp[r["opponent"]][r["tier"]] += 1
    print(f"  {'Opponent':<20s} {'N':>6s}  {'ACC':>6s} {'INACC':>6s} {'MIST':>6s} {'BLUND':>6s} {'ERR':>4s}")
    for opp in sorted(by_opp):
        c = by_opp[opp]
        n = sum(c.values())
        acc, inacc, mist, blund, err = c["ACCURATE"], c["INACCURACY"], c["MISTAKE"], c["BLUNDER"], c["ERROR"]
        print(f"  {opp:<20s} {n:>6d}  "
              f"{acc:>5d}({100*acc/n:>4.1f}%) "
              f"{inacc:>5d}({100*inacc/n:>4.1f}%) "
              f"{mist:>5d}({100*mist/n:>4.1f}%) "
              f"{blund:>5d}({100*blund/n:>4.1f}%) "
              f"{err:>4d}")
    # Also overall
    total = Counter()
    for c in by_opp.values():
        total.update(c)
    n = sum(total.values())
    if n:
        print(f"  {'OVERALL':<20s} {n:>6d}  "
              f"{total['ACCURATE']:>5d}({100*total['ACCURATE']/n:>4.1f}%) "
              f"{total['INACCURACY']:>5d}({100*total['INACCURACY']/n:>4.1f}%) "
              f"{total['MISTAKE']:>5d}({100*total['MISTAKE']/n:>4.1f}%) "
              f"{total['BLUNDER']:>5d}({100*total['BLUNDER']/n:>4.1f}%) "
              f"{total['ERROR']:>4d}")


if __name__ == "__main__":
    main()
