#!/usr/bin/env python3
"""Re-run Coda on each candidate position at the depth Coda actually
reached during the game (parsed from PGN comments).

PGN move comments have format `{eval/depth time}`, e.g. `{+0.95/16 1.8s}`.

For each candidate (FEN + ply + opponent), this:
  1. Walks the PGN to find the matching game
  2. Extracts the depth Coda reached at that ply from the PGN comment
  3. Runs ./coda at the same depth (fresh process, ucinewgame)
  4. Records bestmove + score
  5. Compares vs played + sf_best

Why: the original clean-hash test at fixed depth 14 conflates depth
deficit with state pollution. Using the played-at depth gives the
search apples-to-apples treatment minus only the warm-TT carry-over.

Output: CSV per candidate with played_depth, coda_clean_depth_match,
matches_sf, matches_played, score.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn


CODA_DEFAULT = Path("./coda")
EVAL_RE = re.compile(r"^\s*([+-]?(?:M\d*|\d+(?:\.\d+)?))(?:/(\d+))?(?:\s+(?:[\d.]+s|book))?")


def parse_eval_depth(comment: str) -> int | None:
    """Extract depth from an engine comment like '+0.95/16 1.8s' or '-0.39/18 1.7s'.
    Returns None if not parseable."""
    if not comment:
        return None
    m = EVAL_RE.match(comment.strip())
    if not m:
        return None
    if m.group(2):
        try:
            return int(m.group(2))
        except ValueError:
            return None
    return None


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


def walk_game_depths(game) -> dict[int, int]:
    """Return {ply: coda_depth_at_this_ply} for Coda's moves only."""
    color = coda_color(game)
    if color is None:
        return {}
    depths = {}
    board = game.board()
    ply = 0
    node = game
    while node.variations:
        next_node = node.variation(0)
        ply += 1
        if board.turn == color:
            d = parse_eval_depth(next_node.comment)
            if d:
                depths[ply] = d
        board.push(next_node.move)
        node = next_node
    return depths


def run_coda(coda_bin: Path, fen: str, depth: int) -> tuple[str | None, int | None]:
    """Run ./coda with ucinewgame at fixed depth. Returns (bestmove, score_cp)."""
    cmds = [
        "uci",
        "setoption name Hash value 64",
        "ucinewgame",
        f"position fen {fen}",
        f"go depth {depth}",
        "quit",
    ]
    proc = subprocess.Popen(
        [str(coda_bin.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True,
    )
    try:
        out, _ = proc.communicate(input="\n".join(cmds) + "\n", timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, None
    bestmove = None
    last_score = None
    for line in out.splitlines():
        m = re.match(r"^bestmove (\S+)", line)
        if m:
            bestmove = m.group(1)
            break
        m = re.match(r"^info .* score cp (-?\d+)", line)
        if m:
            last_score = int(m.group(1))
        m = re.match(r"^info .* score mate (-?\d+)", line)
        if m:
            mn = int(m.group(1))
            last_score = (30000 - abs(mn)) * (1 if mn > 0 else -1)
    return bestmove, last_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path, help="JSONL with fen_before + opponent + ply + blunder_uci + sf_best_uci")
    ap.add_argument("pgn", type=Path, help="Source PGN to extract played-at depth")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coda", type=Path, default=CODA_DEFAULT)
    ap.add_argument("--min-depth", type=int, default=10, help="Floor for probe depth")
    ap.add_argument("--max-depth", type=int, default=24, help="Cap for probe depth (don't run forever)")
    ap.add_argument("--extra-depth", type=int, default=0, help="Probe at played_depth + extra")
    args = ap.parse_args()

    # Load candidates by (game_idx, ply) — the game_idx in JSONL matches the
    # 0-based index of Coda-played games in the PGN walking order.
    cands = [json.loads(l) for l in args.candidates.open() if l.strip()]
    cands_by_game = {}
    for c in cands:
        cands_by_game.setdefault(int(c["game_idx"]), []).append(c)

    # Walk PGN — only count games where coda played (skip games with no
    # Coda involvement) so game_idx aligns with the find_pruning_candidates
    # output.
    enriched = []
    with args.pgn.open() as f:
        gi = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if coda_color(game) is None:
                continue
            if not coda_lost(game):
                gi += 1
                continue
            if gi in cands_by_game:
                depths = walk_game_depths(game)
                for c in cands_by_game[gi]:
                    pd = depths.get(c["ply"])
                    if pd:
                        c["played_depth"] = pd
                        enriched.append(c)
                    else:
                        c["played_depth"] = None
            gi += 1

    print(f"Found played-depth for {sum(1 for c in enriched if c.get('played_depth'))}/{len(cands)} candidates")

    rows = []
    for i, c in enumerate(enriched):
        pd = c.get("played_depth")
        if pd is None:
            continue
        probe_depth = max(args.min_depth, min(args.max_depth, pd + args.extra_depth))
        bestmove, score = run_coda(args.coda, c["fen_before"], probe_depth)
        matches_sf = (bestmove == c["sf_best_uci"])
        matches_played = (bestmove == c["blunder_uci"])
        rows.append({
            "idx": i,
            "opponent": c["opponent"],
            "ply": c["ply"],
            "played_depth": pd,
            "probe_depth": probe_depth,
            "fen": c["fen_before"],
            "played_uci": c["blunder_uci"],
            "sf_best_uci": c["sf_best_uci"],
            "coda_clean_uci": bestmove,
            "score_cp": score,
            "matches_sf": int(matches_sf),
            "matches_played": int(matches_played),
        })
        tag = "SF" if matches_sf else ("PLAYED" if matches_played else "OTHER")
        print(f"  [{i+1}/{len(enriched)}] vs {c['opponent']:<10} ply {c['ply']:>3}  "
              f"played_d={pd} probe_d={probe_depth} played={c['blunder_uci']} sf={c['sf_best_uci']} "
              f"clean={bestmove} {tag}")

    if rows:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    # Summary
    sf = sum(r["matches_sf"] for r in rows)
    played = sum(r["matches_played"] for r in rows)
    other = len(rows) - sf - played
    print(f"\n=== Verdict at played-depth + {args.extra_depth} ===")
    print(f"  Picks SF-best:       {sf}/{len(rows)} ({100*sf/len(rows):.0f}%)")
    print(f"  Picks the played:    {played}/{len(rows)} ({100*played/len(rows):.0f}%)")
    print(f"  Picks other:         {other}/{len(rows)} ({100*other/len(rows):.0f}%)")


if __name__ == "__main__":
    main()
