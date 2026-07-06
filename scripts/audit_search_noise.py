#!/usr/bin/env python3
"""Audit Coda for search-manufactured eval noise in low-signal positions.

Generalises the corrhist fortress-drift finding
(docs/corrhist_fortress_drift_2026-07-06.md): a search-time eval-shaping
feature inflating the score away from the raw net where the truth is ~0.

Method — divergence-from-oracle + ablation-attribution:
  1. Build a corpus of "should-be-~0" positions: generate low-material legal
     positions, keep those the ORACLE (Stockfish, no TB) scores near 0.
  2. Flag positions where Coda's SEARCH inflates past a threshold.
  3. Attribute each flagged case to a feature by re-running Coda with each
     NO_* ablation flag and seeing which one collapses the score toward 0.

No tablebases are used at eval time (they would mask the net/search opinion).

Usage:
  python3 scripts/audit_search_noise.py --n 400 --keep 120 --movetime 300
"""
import argparse, random, subprocess, sys, time
import chess

# Ablation flags to attribute blame to (the eval-shaping search features).
ABLATIONS = ["NO_CORRECTION", "NO_FH_BLEND", "NO_TT_CUTOFF", "NO_TT_NEARMISS"]

PIECE_POOL = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]


class Engine:
    """Persistent UCI session; one score per (position, go) with sync."""
    def __init__(self, cmd, options=None, env=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1, env=env)
        self._send("uci"); self._wait("uciok")
        for k, v in (options or {}).items():
            self._send(f"setoption name {k} value {v}")
        self._send("isready"); self._wait("readyok")

    def _send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, token):
        for line in self.p.stdout:
            if line.startswith(token):
                return line

    def score(self, fen, movetime):
        """Return cp score from side-to-move POV (mate -> +-100000)."""
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime}")
        last = None
        for line in self.p.stdout:
            if line.startswith("info") and " score " in line:
                last = line
            elif line.startswith("bestmove"):
                break
        if last is None:
            return None
        toks = last.split()
        i = toks.index("score")
        if toks[i + 1] == "cp":
            return int(toks[i + 2])
        if toks[i + 1] == "mate":
            m = int(toks[i + 2])
            return (100000 if m > 0 else -100000) - m
        return None

    def close(self):
        try: self._send("quit"); self.p.wait(timeout=3)
        except Exception: self.p.kill()


def random_lowmat(rng, min_pieces=4, max_pieces=7):
    """Random legal, non-check, non-terminal low-material position."""
    for _ in range(200):
        b = chess.Board.empty()
        sqs = rng.sample(range(64), 64)
        wk, bk = sqs[0], sqs[1]
        if chess.square_distance(wk, bk) <= 1:
            continue
        b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        n = rng.randint(min_pieces - 2, max_pieces - 2)
        used = 2
        for sq in sqs[2:]:
            if used >= 2 + n:
                break
            pt = rng.choice(PIECE_POOL)
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            b.set_piece_at(sq, chess.Piece(pt, rng.choice([chess.WHITE, chess.BLACK])))
            used += 1
        b.turn = rng.choice([chess.WHITE, chess.BLACK])
        if not b.is_valid():
            continue
        if b.is_check() or b.is_game_over():
            continue
        return b.fen()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coda", default="./coda")
    ap.add_argument("--sf", default="/home/adam/chess/engines/Stockfish-17.1/src/stockfish")
    ap.add_argument("--n", type=int, default=400, help="candidate positions to generate")
    ap.add_argument("--keep", type=int, default=120, help="target size of ~0 corpus")
    ap.add_argument("--oracle-band", type=int, default=25, help="|SF| <= band => 'should be 0'")
    ap.add_argument("--flag", type=int, default=40, help="|Coda| >= this => inflated")
    ap.add_argument("--movetime", type=int, default=300)
    ap.add_argument("--min-pieces", type=int, default=4)
    ap.add_argument("--max-pieces", type=int, default=7)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1. Oracle-filter a corpus of consensus-~0 positions.
    sf = Engine([args.sf], {"Threads": 1, "Hash": 64})
    corpus = []
    tried = 0
    print(f"# building corpus (oracle band |SF|<={args.oracle_band}cp, no TB)...", file=sys.stderr)
    while len(corpus) < args.keep and tried < args.n * 6:
        tried += 1
        fen = random_lowmat(rng, args.min_pieces, args.max_pieces)
        if fen is None:
            continue
        s = sf.score(fen, args.movetime)
        if s is not None and abs(s) <= args.oracle_band:
            corpus.append(fen)
            if len(corpus) % 20 == 0:
                print(f"#   {len(corpus)}/{args.keep} draws collected", file=sys.stderr)
    sf.close()
    print(f"# corpus: {len(corpus)} consensus-~0 positions from {tried} candidates", file=sys.stderr)

    # 2. Coda baseline (no TB) on the corpus; flag inflated.
    coda = Engine([args.coda], {"Hash": 64})
    base = {fen: coda.score(fen, args.movetime) for fen in corpus}
    flagged = [f for f, s in base.items() if s is not None and abs(s) >= args.flag]
    n = len([s for s in base.values() if s is not None])
    mean_abs = sum(abs(s) for s in base.values() if s is not None) / max(1, n)
    print(f"\n=== Coda on {n} consensus-~0 positions ===")
    print(f"mean |score| = {mean_abs:.1f}cp   inflated (|score|>={args.flag}): "
          f"{len(flagged)}/{n} ({100*len(flagged)//max(1,n)}%)")

    # 3. Ablation-attribution on flagged positions.
    if flagged:
        abl_engines = {name: Engine([args.coda], {"Hash": 64}, env={**__import__("os").environ, name: "1"})
                       for name in ABLATIONS}
        print(f"\n=== attribution on {len(flagged)} inflated positions "
              f"(fix = |score| drops below {args.flag}) ===")
        credit = {name: 0 for name in ABLATIONS}
        rows = []
        for fen in flagged:
            b = base[fen]
            abls = {name: abl_engines[name].score(fen, args.movetime) for name in ABLATIONS}
            for name, s in abls.items():
                if s is not None and abs(s) < args.flag:
                    credit[name] += 1
            rows.append((fen, b, abls))
        for name in ABLATIONS:
            print(f"  {name:16} fixes {credit[name]:3}/{len(flagged)}")
        print(f"\n# worst offenders (fen | base | ablation deltas):")
        for fen, b, abls in sorted(rows, key=lambda r: -abs(r[1]))[:12]:
            deltas = " ".join(f"{k.replace('NO_','-')}={v}" for k, v in abls.items())
            print(f"  {b:+5} | {deltas:50} | {fen}")
        for e in abl_engines.values():
            e.close()
    coda.close()


if __name__ == "__main__":
    main()
