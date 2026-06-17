#!/usr/bin/env python3
"""
harvest_vs_sf.py — extract SF-labelled training positions from Coda-vs-Stockfish
games, FOR FREE (no rescore). In a Coda-vs-SF game, the positions where SF is to
move are exactly the positions Coda's own moves produced, and SF already scored
them during the game (the eval comment, STM-POV, WDL-calibrated, depth ~12-16).
We take (position-before-SF-move, SF-eval) -> directly feeds `coda import-tsv`.

Input : a PGN of Coda-vs-Stockfish games (with eval comments).
Output: TSV  fen<TAB>sf_cp   (sf_cp = STM-POV centipawns).
Skips: ply<min-ply (opening), mate-ish/non-float evals (import-tsv filters anyway).
"""
import argparse, re, sys
try:
    import chess, chess.pgn
except ImportError:
    sys.exit("needs python-chess")

EVAL_RE = re.compile(r'([+-]?\d+(?:\.\d+)?)/\d+')  # eval before /depth; matches OB compact & verbose, skips mate (M..) & book

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pgn', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-ply', type=int, default=16)
    a = ap.parse_args()
    games = pos = 0
    with open(a.pgn) as f, open(a.out, 'w') as out:
        out.write("fen\tsf_cp\n")
        while True:
            g = chess.pgn.read_game(f)
            if g is None: break
            games += 1
            w, b = g.headers.get("White",""), g.headers.get("Black","")
            # OB datagen names players "<engine>-<role/branch>" (e.g. "Stockfish-base"),
            # cutechess/local gauntlets use bare "Stockfish" — match either.
            sf_white = w.startswith("Stockfish"); sf_black = b.startswith("Stockfish")
            if not (sf_white or sf_black): continue
            board = g.board()
            for node in g.mainline():
                mover_is_sf = (board.turn == chess.WHITE and sf_white) or \
                              (board.turn == chess.BLACK and sf_black)
                ply = board.ply()
                if mover_is_sf and ply >= a.min_ply:
                    m = EVAL_RE.search(node.comment or "")
                    if m:
                        cp = int(round(float(m.group(1)) * 100))  # STM-POV (SF=mover)
                        out.write(f"{board.fen()}\t{cp}\n"); pos += 1
                board.push(node.move)
            if games % 200 == 0:
                print(f"  ...{games} games, {pos} positions", flush=True)
    print(f"done: {games} games -> {pos} SF-labelled positions -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
