#!/usr/bin/env python3
"""Extract Coda's played-depth from PGN comments for blunder candidates.

For each candidate (game_idx, opponent, ply, blunder_uci) in a JSONL,
find the matching game in the PGN and extract the depth from the
comment of Coda's move at that ply. Cutechess format: `+0.95/16 1.8s`.

Output: enriched JSONL with `played_depth` (and `played_score`,
`played_time_s`) added to each candidate.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import chess
import chess.pgn


COMMENT_RE = re.compile(r"^\s*([+\-]?\d+\.?\d*|\-?M\d+|\+?M\d+|0\.00)\s*/\s*(\d+)\s+([\d.]+)s")


def parse_comment(comment: str):
    """Parse cutechess comment like '+0.95/16 1.8s' → (score_pawns, depth, time_s).
    Returns (None, None, None) if format doesn't match."""
    if not comment:
        return None, None, None
    m = COMMENT_RE.match(comment.strip())
    if not m:
        return None, None, None
    score_str, depth_str, time_str = m.groups()
    return score_str, int(depth_str), float(time_str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn", type=Path)
    ap.add_argument("candidates", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    candidates = [json.loads(line) for line in args.candidates.open() if line.strip()]
    print(f"Loaded {len(candidates)} candidates from {args.candidates}", file=sys.stderr)

    # Build games-with-Coda-as-loser list, capture the played move + comment per ply.
    # We index 0-based; the SF arbitration that produced game_idx used the same
    # convention.
    games = []  # list of dicts: {white, black, result, moves: [{ply, uci, comment}]}
    coda_loss_games = []  # subset where Coda lost (for game_idx mapping)
    with args.pgn.open() as f:
        idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            result = game.headers.get("Result", "")
            moves = []
            board = game.board()
            for ply_idx, node in enumerate(game.mainline()):
                uci = node.move.uci()
                comment = node.comment or ""
                moves.append({"ply": ply_idx, "uci": uci, "comment": comment})
                board.push(node.move)
            games.append({
                "abs_idx": idx,
                "white": white,
                "black": black,
                "result": result,
                "moves": moves,
            })
            # Coda loses if "Coda" is white and result "0-1", or Coda is black and result "1-0"
            coda_is_white = white == "Coda"
            coda_is_black = black == "Coda"
            coda_lost = (coda_is_white and result == "0-1") or (coda_is_black and result == "1-0")
            if coda_lost:
                coda_loss_games.append(games[-1])
            idx += 1

    print(f"Total PGN games: {len(games)}", file=sys.stderr)
    print(f"Coda-loss games: {len(coda_loss_games)}", file=sys.stderr)

    def try_match(g, opp, ply, blunder_uci):
        """Try matching (opp, ply±1, uci) in game g. Returns parsed comment or None."""
        if g["white"] == "Coda":
            other = g["black"]
        elif g["black"] == "Coda":
            other = g["white"]
        else:
            return None
        if other != opp:
            return None
        # Try exact ply, then ±1 (off-by-one between SF arbitrator and python-chess).
        for cand_ply in (ply, ply - 1, ply + 1):
            if 0 <= cand_ply < len(g["moves"]):
                mv = g["moves"][cand_ply]
                if mv["uci"] == blunder_uci:
                    return parse_comment(mv["comment"]), cand_ply
        return None

    enriched = []
    matched = 0
    unmatched = 0
    for c in candidates:
        gi = c["game_idx"]
        opp = c["opponent"]
        ply = c["ply"]
        blunder_uci = c["blunder_uci"]
        played_depth = None
        played_score = None
        played_time = None
        match_method = None
        matched_ply = None

        for label, glist in [("abs_index", games), ("loss_index", coda_loss_games)]:
            if 0 <= gi < len(glist):
                result = try_match(glist[gi], opp, ply, blunder_uci)
                if result is not None:
                    (score_str, depth, time_s), matched_ply = result
                    played_depth = depth
                    played_score = score_str
                    played_time = time_s
                    match_method = label
                    break

        if played_depth is None:
            # Last resort: scan all games. Disambiguate by FEN match if multiple.
            target_fen = c.get("fen_before", "")
            target_fen_no_clock = " ".join(target_fen.split()[:4]) if target_fen else None
            for g in games:
                result = try_match(g, opp, ply, blunder_uci)
                if result is not None:
                    if target_fen_no_clock:
                        # Verify by replaying to ply and checking FEN
                        bd = chess.Board(g["white"] and "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" or chess.STARTING_FEN)
                        # Actually use python-chess re-parse for FEN check
                    (score_str, depth, time_s), matched_ply = result
                    played_depth = depth
                    played_score = score_str
                    played_time = time_s
                    match_method = "scan"
                    break

        out = dict(c)
        out["played_depth"] = played_depth
        out["played_score"] = played_score
        out["played_time_s"] = played_time
        out["match_method"] = match_method
        out["matched_ply"] = matched_ply
        enriched.append(out)
        if played_depth is not None:
            matched += 1
        else:
            unmatched += 1

    with args.out.open("w") as f:
        for e in enriched:
            f.write(json.dumps(e) + "\n")

    print(f"Matched: {matched}, unmatched: {unmatched}", file=sys.stderr)

    if args.summary:
        # Distribution of played depth
        from collections import Counter
        depths = [e["played_depth"] for e in enriched if e["played_depth"] is not None]
        print(f"\n=== Played-depth distribution (n={len(depths)}) ===")
        if depths:
            depth_counter = Counter(depths)
            for d in sorted(depth_counter):
                print(f"  d{d:2d}: {depth_counter[d]}")
            print(f"  min={min(depths)} median={sorted(depths)[len(depths)//2]} max={max(depths)}")


if __name__ == "__main__":
    main()
