#!/usr/bin/env python3
"""Diagnostic probe: structurally perturb Coda's worst eval-overrate positions
(king-safety blind spot) and measure whether the gap to a trusted reference
(SF static eval) shrinks. Not a fix — a "what is the net actually keying off"
probe, per Adam's request 2026-06-30.

gap(position) = coda_eval_stm - sf_eval_stm (signed, both NNUE-static, no
search). The heldout harvest already established SF static is materially
closer to LC0 truth on this population (that's how it was selected), so
gap's MAGNITUDE is a usable proxy for Coda-specific error even on synthetic
perturbed positions where we have no LC0 truth.
"""
import argparse
import random
import re
import subprocess
import sys

import chess

EVAL_RE = re.compile(r"NNUE evaluation\s+([+-]?[0-9]+\.[0-9]+)")


class Engine:
    def __init__(self, path, nnue=None):
        self.path = path
        self.nnue = nnue
        self._spawn()

    def _spawn(self):
        self.proc = subprocess.Popen(
            [self.path] + (["-n", self.nnue] if self.nnue else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._cmd("uci")
        self._drain_until("uciok")

    def _cmd(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _drain_until(self, marker):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if marker in line:
                break
        return lines

    def eval_stm_cp(self, board: chess.Board):
        if not is_sane(board):
            return None
        try:
            self._cmd(f"position fen {board.fen()}")
            self._cmd("eval")
            self._cmd("isready")
            lines = self._drain_until("readyok")
        except (BrokenPipeError, OSError):
            self._spawn()
            return None
        white_cp = None
        for line in lines:
            m = EVAL_RE.search(line)
            if m:
                white_cp = round(float(m.group(1)) * 100)
        if white_cp is None:
            # engine likely crashed on a malformed perturbed position; respawn for next call
            if self.proc.poll() is not None:
                self._spawn()
            return None
        return white_cp if board.turn == chess.WHITE else -white_cp

    def close(self):
        try:
            self._cmd("quit")
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def is_sane(board: chess.Board) -> bool:
    """Reject structurally-broken boards (adjacent kings, opponent-in-check)
    that could crash an engine's eval path or produce meaningless evals."""
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    if wk is None or bk is None:
        return False
    wf, wr = chess.square_file(wk), chess.square_rank(wk)
    bf, br = chess.square_file(bk), chess.square_rank(bk)
    if max(abs(wf - bf), abs(wr - br)) <= 1:
        return False
    try:
        if board.was_into_check():
            return False
    except Exception:
        return False
    return True


def king_danger_score(board: chess.Board, color):
    """Cheap king-safety heuristic: enemy attackers within Chebyshev dist 2
    of king, weighted by piece value, plus an uncastled-in-middlegame flag."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0, False
    enemy = not color
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    weight = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    danger = 0
    for sq, piece in board.piece_map().items():
        if piece.color != enemy or piece.piece_type == chess.KING:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if max(abs(f - kf), abs(r - kr)) <= 2:
            danger += weight.get(piece.piece_type, 0)
    non_pawn_material = sum(
        weight.get(p.piece_type, 0) for p in board.piece_map().values() if p.piece_type != chess.PAWN
    )
    home_rank = 0 if color == chess.WHITE else 7
    uncastled_center = kr == home_rank and kf in (3, 4) and non_pawn_material >= 16
    return danger, uncastled_center


def closest_enemy_piece(board: chess.Board, color, farthest=False):
    king_sq = board.king(color)
    if king_sq is None:
        return None
    enemy = not color
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    best_sq, best_dist = None, (-1 if farthest else 99)
    for sq, piece in board.piece_map().items():
        if piece.color != enemy or piece.piece_type in (chess.KING, chess.PAWN):
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        d = max(abs(f - kf), abs(r - kr))
        if (farthest and d > best_dist) or (not farthest and d < best_dist):
            best_dist, best_sq = d, sq
    return best_sq


def remove_random_piece(board: chess.Board, rng):
    candidates = [sq for sq, p in board.piece_map().items()
                  if p.piece_type not in (chess.KING, chess.PAWN)]
    if not candidates:
        return None
    sq = rng.choice(candidates)
    b = board.copy()
    b.remove_piece_at(sq)
    return b


def remove_piece_pair(board: chess.Board, piece_type):
    b = board.copy()
    removed_any = False
    for color in (chess.WHITE, chess.BLACK):
        squares = list(b.pieces(piece_type, color))
        if squares:
            b.remove_piece_at(squares[0])
            removed_any = True
    return b if removed_any else None


def remove_all_of_type(board: chess.Board, piece_type):
    b = board.copy()
    removed_any = False
    for color in (chess.WHITE, chess.BLACK):
        for sq in list(b.pieces(piece_type, color)):
            b.remove_piece_at(sq)
            removed_any = True
    return b if removed_any else None


def remove_closest_attacker(board: chess.Board, stm_color, farthest=False):
    sq = closest_enemy_piece(board, stm_color, farthest=farthest)
    if sq is None:
        return None
    b = board.copy()
    b.remove_piece_at(sq)
    return b


def relocate_king_to_corner(board: chess.Board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return None
    target = chess.G1 if color == chess.WHITE else chess.G8
    if king_sq == target:
        return None
    b = board.copy()
    occupant = b.piece_at(target)
    b.remove_piece_at(king_sq)
    if occupant is not None:
        b.remove_piece_at(target)
        if occupant.piece_type != chess.KING:
            b.set_piece_at(king_sq, occupant)
    b.set_piece_at(target, chess.Piece(chess.KING, color))
    b.castling_rights = chess.BB_EMPTY
    b.ep_square = None
    return b


def remove_balanced(board: chess.Board, piece_type):
    """Remove an EQUAL number of `piece_type` from both sides (material-neutral).
    Removes min(white_count, black_count) from each, preserving the original
    material balance — isolates the *positional* contribution of the piece type
    from the free-material shock that confounds single-side removal (the bug
    that made the first probe's sign flip between samples). Returns None if
    either side has zero of the type (nothing balanced to remove)."""
    b = board.copy()
    w = list(b.pieces(piece_type, chess.WHITE))
    k = list(b.pieces(piece_type, chess.BLACK))
    n = min(len(w), len(k))
    if n == 0:
        return None
    for sq in w[:n] + k[:n]:
        b.remove_piece_at(sq)
    b.ep_square = None
    return b if is_sane(b) else None


def jiggle_bishop(board: chess.Board, color, rng, two=False):
    """Move one of `color`'s bishops one (or two) squares diagonally to an
    empty, sane target — a 'slight move', not a removal. Probes how sensitive
    the eval is to bishop *placement* in a position we overscore."""
    bishops = list(board.pieces(chess.BISHOP, color))
    rng.shuffle(bishops)
    step = 2 if two else 1
    for sq in bishops:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        offsets = [(step, step), (step, -step), (-step, step), (-step, -step)]
        rng.shuffle(offsets)
        for df, dr in offsets:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                tsq = chess.square(nf, nr)
                # path must be clear for a real bishop slide (1 or 2 squares)
                if board.piece_at(tsq) is not None:
                    continue
                if two:
                    mid = chess.square(f + df // 2, r + dr // 2)
                    if board.piece_at(mid) is not None:
                        continue
                b = board.copy()
                b.remove_piece_at(sq)
                b.set_piece_at(tsq, chess.Piece(chess.BISHOP, color))
                b.ep_square = None
                if is_sane(b):
                    return b
    return None


_rng = random.Random(42)

PERTURBATIONS = {
    # --- material-NEUTRAL symmetric removal: isolates positional contribution ---
    "remove_bishops_BALANCED": lambda b, stm: remove_balanced(b, chess.BISHOP),
    "remove_knights_BALANCED_ctrl": lambda b, stm: remove_balanced(b, chess.KNIGHT),
    "remove_rooks_BALANCED_ctrl": lambda b, stm: remove_balanced(b, chess.ROOK),
    # --- slight bishop relocation: local sensitivity to bishop placement ---
    "jiggle_stm_bishop_1sq": lambda b, stm: jiggle_bishop(b, stm, _rng, two=False),
    "jiggle_stm_bishop_2sq": lambda b, stm: jiggle_bishop(b, stm, _rng, two=True),
    # --- original (material-CHANGING) perturbations, kept for contrast ---
    "remove_bishop_pair": lambda b, stm: remove_piece_pair(b, chess.BISHOP),
    "remove_all_bishops": lambda b, stm: remove_all_of_type(b, chess.BISHOP),
    "remove_knight_pair": lambda b, stm: remove_piece_pair(b, chess.KNIGHT),
    "remove_rook_pair": lambda b, stm: remove_piece_pair(b, chess.ROOK),
    "remove_queens": lambda b, stm: remove_all_of_type(b, chess.QUEEN),
    "remove_closest_attacker_on_stm_king": lambda b, stm: remove_closest_attacker(b, stm, farthest=False),
    "remove_farthest_enemy_piece_CONTROL": lambda b, stm: remove_closest_attacker(b, stm, farthest=True),
    "remove_random_piece_CONTROL": lambda b, stm: remove_random_piece(b, _rng),
    "castle_stm_king_to_corner": lambda b, stm: relocate_king_to_corner(b, stm),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="testdata/heldout_overrate_lc0_2023_06.tsv")
    ap.add_argument("--coda", default="./coda")
    ap.add_argument("--stockfish", default="/home/adam/chess/engines/Stockfish/src/stockfish")
    ap.add_argument("--sf-net", default=None, help="explicit SF nnue path; default = SF's own embedded net")
    ap.add_argument("-n", "--count", type=int, default=300)
    ap.add_argument("--king-safety-only", action="store_true",
                     help="restrict sample to positions matching the king-danger/uncastled heuristic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--err-min", type=float, default=0, help="exclude positions with |coda-lc0| below this (cp)")
    ap.add_argument("--err-max", type=float, default=1e9, help="exclude positions with |coda-lc0| above this (cp) — caps likely hanging-piece/tactical outliers")
    ap.add_argument("--min-fullmove", type=int, default=0, help="exclude very early game (e.g. 8 to skip opening noise)")
    args = ap.parse_args()

    rows = []
    with open(args.tsv) as f:
        for line in f:
            fen, lc0_cp = line.rstrip("\n").split("\t")
            rows.append((fen, int(lc0_cp)))

    coda = Engine(args.coda)
    sf = Engine(args.stockfish, args.sf_net)

    print(f"computing baseline coda/sf eval + |coda-lc0| for {len(rows)} positions...", file=sys.stderr)
    scored = []
    for fen, lc0_cp in rows:
        board = chess.Board(fen)
        coda_cp = coda.eval_stm_cp(board)
        if coda_cp is None:
            continue
        err_vs_lc0 = coda_cp - lc0_cp
        danger, uncastled = king_danger_score(board, board.turn)
        scored.append((fen, lc0_cp, coda_cp, err_vs_lc0, danger, uncastled))

    if args.king_safety_only:
        scored = [r for r in scored if r[4] >= 5 or r[5]]
        print(f"king-safety-filtered: {len(scored)} positions", file=sys.stderr)

    if args.min_fullmove:
        scored = [r for r in scored if int(r[0].split()[-1]) >= args.min_fullmove]
        print(f"after min-fullmove filter: {len(scored)} positions", file=sys.stderr)

    scored = [r for r in scored if args.err_min <= abs(r[3]) <= args.err_max]
    print(f"after err-band filter [{args.err_min},{args.err_max}]: {len(scored)} positions", file=sys.stderr)

    if args.err_max < 1e9:
        # bounded band requested -> representative random sample, not worst-N
        random.Random(args.seed).shuffle(scored)
    else:
        scored.sort(key=lambda r: -abs(r[3]))
    sample = scored[: args.count]
    print(f"sample = worst {len(sample)} by |coda-lc0| "
          f"(mean |err|={sum(abs(r[3]) for r in sample)/len(sample):.0f}cp)", file=sys.stderr)

    csv_rows = []
    agg = {name: [] for name in PERTURBATIONS}
    n_orig_gap = 0.0
    n_applicable = {name: 0 for name in PERTURBATIONS}

    for i, (fen, lc0_cp, coda_cp, err_vs_lc0, danger, uncastled) in enumerate(sample):
        board = chess.Board(fen)
        sf_cp = sf.eval_stm_cp(board)
        if sf_cp is None:
            continue
        gap_orig = coda_cp - sf_cp
        n_orig_gap += abs(gap_orig)

        row = dict(fen=fen, lc0=lc0_cp, coda=coda_cp, sf=sf_cp,
                   err_vs_lc0=err_vs_lc0, gap_orig=gap_orig,
                   king_danger=danger, uncastled=uncastled)

        for name, fn in PERTURBATIONS.items():
            pb = fn(board, board.turn)
            if pb is None or pb.king(chess.WHITE) is None or pb.king(chess.BLACK) is None:
                row[f"gap_{name}"] = None
                continue
            try:
                pfen = pb.fen()
            except Exception:
                row[f"gap_{name}"] = None
                continue
            coda_p = coda.eval_stm_cp(pb)
            sf_p = sf.eval_stm_cp(pb)
            if coda_p is None or sf_p is None:
                row[f"gap_{name}"] = None
                continue
            gap_p = coda_p - sf_p
            row[f"gap_{name}"] = gap_p
            agg[name].append((gap_orig, gap_p))
            n_applicable[name] += 1

        csv_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)}", file=sys.stderr)

    coda.close()
    sf.close()

    n = len(csv_rows)
    mean_gap_orig = n_orig_gap / n if n else 0
    print(f"\n=== baseline: mean |coda_eval - sf_eval| over {n} positions = {mean_gap_orig:.1f}cp ===\n")

    print(f"{'perturbation':38s} {'n':>5s} {'mean|gap_orig|':>15s} {'mean|gap_pert|':>15s} {'mean_delta':>11s} {'%_shrunk>=30cp':>15s}")
    for name in PERTURBATIONS:
        pairs = agg[name]
        if not pairs:
            print(f"{name:38s}  (not applicable to any sampled position)")
            continue
        m_orig = sum(abs(g0) for g0, _ in pairs) / len(pairs)
        m_pert = sum(abs(g1) for _, g1 in pairs) / len(pairs)
        deltas = [abs(g0) - abs(g1) for g0, g1 in pairs]
        mean_delta = sum(deltas) / len(deltas)
        pct_shrunk = 100 * sum(1 for d in deltas if d >= 30) / len(deltas)
        print(f"{name:38s} {len(pairs):5d} {m_orig:15.1f} {m_pert:15.1f} {mean_delta:11.1f} {pct_shrunk:14.1f}%")

    if args.csv:
        import csv as csvmod
        fields = ["fen", "lc0", "coda", "sf", "err_vs_lc0", "gap_orig", "king_danger", "uncastled"] + \
                 [f"gap_{name}" for name in PERTURBATIONS]
        with open(args.csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in csv_rows:
                w.writerow(row)
        print(f"\nwrote {len(csv_rows)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
