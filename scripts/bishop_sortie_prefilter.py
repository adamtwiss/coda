#!/usr/bin/env python3
"""
bishop_sortie_prefilter.py — CHEAP (no-engine) pre-filter over Coda-vs-SF
datagen PGNs to find candidate "wandering bishop" positions, before the
expensive deep-SF validation in overrate_corpus.py.

Selects Coda-to-move positions where Coda PLAYED a forward bishop sortie:
  - moving piece is a bishop,
  - it advances toward the enemy (to-rank ahead of from-rank for the mover),
  - it lands in / near enemy territory (white to rank>=4, black to rank<=5),
  - midgame (fullmove in [lo,hi]), not a deep endgame,
  - position before the move is not in check.
Optionally gate on Coda's own eval optimism (the in-PGN mover-POV eval on the
bishop move >= --opt-min) — the over-credit signature.

Captures (Bxf4) and checks (Bd4+) are KEPT but tagged (the doc splits the
class: captures are FT/material-driven, pure sorties are threat-driven).

Emits TSV: gid  ply  fullmove  tag  coda_eval  san  uci  fen_before
The fen_before + uci feed the deep-SF validator. Use --emit-epd-stub to also
write a scratch EPD with `am <coda_move>` (bm filled in by the SF pass).

Coda side is auto-detected from headers (the engine whose name contains
--coda-name, default 'Coda').
"""
import argparse, bz2, sys, glob, os, re
import chess, chess.pgn

EVAL_RE = re.compile(r'([+-]?\d+\.\d+)')

def coda_color(g, coda_name):
    w = g.headers.get('White', ''); b = g.headers.get('Black', '')
    if coda_name.lower() in w.lower(): return chess.WHITE
    if coda_name.lower() in b.lower(): return chess.BLACK
    return None

def mover_eval(comment):
    # fastchess comment like "+0.70/11 0.028s, n=15015, sd=18" — first float is
    # the mover-POV eval in pawns.
    m = EVAL_RE.search(comment or '')
    return float(m.group(1)) if m else None

def iter_pgns(paths):
    for p in paths:
        op = bz2.open(p, 'rt', errors='ignore') if p.endswith('.bz2') else open(p, errors='ignore')
        with op as f:
            while True:
                try:
                    g = chess.pgn.read_game(f)
                except Exception:
                    break
                if g is None: break
                yield p, g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', nargs='+', required=True, help='PGN(.bz2) globs')
    ap.add_argument('--coda-name', default='Coda')
    ap.add_argument('--max-games', type=int, default=0, help='0 = all')
    ap.add_argument('--fm-lo', type=int, default=8)
    ap.add_argument('--fm-hi', type=int, default=44)
    ap.add_argument('--min-pieces', type=int, default=14, help='skip deep endgames')
    ap.add_argument('--opt-min', type=float, default=0.30,
                    help="require Coda's own eval on the move >= this (pawns)")
    ap.add_argument('--opt-max', type=float, default=1.50,
                    help="...and <= this — balanced/over-credit zone, not a won game")
    ap.add_argument('--no-cap', action='store_true',
                    help='exclude captures (threat-driven quiet-sortie class only)')
    ap.add_argument('--deep-rank', type=int, default=4,
                    help='penetration: white to-rank-idx >= this (default 4 = rank5+); '
                         'black mirrored (<= 7-this)')
    ap.add_argument('--min-disagree', type=float, default=0.80,
                    help='require cev + SF-reply-eval >= this (both POV-positive = '
                         'both engines think THEY are better after the sortie). 0 disables.')
    ap.add_argument('--out', default='-')
    args = ap.parse_args()

    paths = []
    for gl in args.glob: paths.extend(sorted(glob.glob(gl)))
    out = sys.stdout if args.out == '-' else open(args.out, 'w')
    out.write("gid\tply\tfullmove\ttag\tcoda_eval\tdisagree\tsan\tuci\tfen_before\n")

    ng = 0; nhit = 0; by_san = {}
    for path, g in iter_pgns(paths):
        cc = coda_color(g, args.coda_name)
        if cc is None: continue
        ng += 1
        if args.max_games and ng > args.max_games: break
        board = g.board()
        node = g
        ply = 0
        while node.variations:
            node = node.variations[0]
            mv = node.move
            ply += 1
            mover = board.turn
            piece = board.piece_at(mv.from_square)
            is_coda = (mover == cc)
            # evaluate candidate BEFORE pushing
            if (is_coda and piece is not None and piece.piece_type == chess.BISHOP
                    and not board.is_check()
                    and args.fm_lo <= board.fullmove_number <= args.fm_hi
                    and chess.popcount(board.occupied) >= args.min_pieces):
                fr = chess.square_rank(mv.from_square); tr = chess.square_rank(mv.to_square)
                forward = (tr > fr) if mover == chess.WHITE else (tr < fr)
                deep = (tr >= args.deep_rank) if mover == chess.WHITE else (tr <= 7 - args.deep_rank)
                is_cap = board.is_capture(mv)
                if forward and deep and not (args.no_cap and is_cap):
                    cev = mover_eval(node.comment)
                    if cev is not None and args.opt_min <= cev <= args.opt_max:
                        # SF's reply eval (POV-positive = SF thinks SF is better)
                        sf_reply = mover_eval(node.variations[0].comment) if node.variations else None
                        disagree = (cev + sf_reply) if sf_reply is not None else None
                        if args.min_disagree > 0 and (disagree is None or disagree < args.min_disagree):
                            board.push(mv); continue
                        san = board.san(mv)
                        tag = 'cap' if is_cap else 'quiet'
                        if board.gives_check(mv): tag += '+chk'
                        fen_before = board.fen()
                        gid = f"{os.path.basename(path)}:{ply}"
                        dstr = f"{disagree:+.2f}" if disagree is not None else "NA"
                        out.write(f"{gid}\t{ply}\t{board.fullmove_number}\t{tag}\t"
                                  f"{cev:+.2f}\t{dstr}\t{san}\t{mv.uci()}\t{fen_before}\n")
                        nhit += 1
                        key = san.rstrip('+#')
                        by_san[key] = by_san.get(key, 0) + 1
            board.push(mv)
    if out is not sys.stdout: out.close()
    sys.stderr.write(f"\nscanned {ng} Coda games, {nhit} candidate sorties "
                     f"({100*nhit/max(1,ng):.2f} per 100 games)\n")
    top = sorted(by_san.items(), key=lambda kv: -kv[1])[:25]
    sys.stderr.write("top sortie SANs: " + ", ".join(f"{k}:{v}" for k, v in top) + "\n")

if __name__ == '__main__':
    main()
