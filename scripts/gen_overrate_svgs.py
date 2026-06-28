#!/usr/bin/env python3
"""Generate chess-diagram SVGs for docs/eval_overrate_findings.md.

Writes docs/img/overrate_*.svg using python-chess. Each diagram shows the
position; a green arrow marks SF's deep best move (the "what SF actually does
here" reference, matching the convention in
docs/threat_eval_asymmetry_2026-06-17.md). White at bottom for consistency.

Re-run after adding positions:  python3 scripts/gen_overrate_svgs.py
SVGs render natively on GitHub and GitLab (unlike the old ASCII boards).
"""
import os
import chess
import chess.svg

OUTDIR = os.path.join(os.path.dirname(__file__), os.pardir, "docs", "img")

# key, FEN, SF best move (UCI, green arrow) — None to omit the arrow
POSITIONS = [
    ("overrate_stormphrax_m40",
     "6k1/3r2b1/4rp2/1pp1nN1Q/8/1P4R1/2q3P1/5R1K w - - 4 40", "h5h6"),
    ("overrate_quanticade_m27",
     "q4r2/5p2/2np1n1k/2p1pPNb/1pP5/1P1P2QP/6BK/5R2 w - - 1 27", "g2c6"),
    ("overrate_plentychess_m34",
     "8/4kp2/4b1p1/4Q3/1P1pP1P1/p2PqP2/P1r2NK1/5R2 w - - 3 34", "b4b5"),
    ("overrate_viridithas_m72",
     "8/1R6/5ppK/3r1k1p/7P/5PP1/8/8 w - - 0 72", "b7b4"),
    ("overrate_clover_m81",
     "8/5k2/8/8/4R3/5P2/5K2/3q4 b - - 1 81", "d1h1"),
    ("overrate_integral_m57",
     "8/8/7R/8/3p1K2/r6P/2k5/8 b - - 2 57", "d4d3"),
    # KBN-v-K conversion study (Hobbes m74): objectively mate, Coda reads flat +6
    ("overrate_hobbes_kbn_m74",
     "8/6B1/8/8/2k2K2/8/8/N7 w - - 13 74", "f4e4"),
]

GREEN = "#15781b"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for key, fen, sf_uci in POSITIONS:
        board = chess.Board(fen)
        arrows = []
        if sf_uci:
            m = chess.Move.from_uci(sf_uci)
            arrows.append(chess.svg.Arrow(m.from_square, m.to_square, color=GREEN))
        svg = chess.svg.board(board, arrows=arrows, size=380,
                              coordinates=True)
        path = os.path.join(OUTDIR, key + ".svg")
        with open(path, "w") as f:
            f.write(svg)
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
