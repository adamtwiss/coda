#!/usr/bin/env python3
"""Search-invariant harness for Coda.

Flushes out subtle search/TT correctness leaks that DON'T show up in SPRT (they
fire on rare positions, never moving the aggregate, but blunder on the game they
decide). The technique: assert invariants that MUST hold, whose violation
localizes a bug.

WHAT IS AND ISN'T A VALID INVARIANT (learned the hard way, 2026-07-12):
  Strict cross-config SCORE-equality is NOT a valid invariant for a modern
  fail-soft + aspiration + null-window engine. Two natural-looking ones were
  tried and REJECTED as cry-wolf:
    * "TT on vs TT off -> same score"  — false: a depth-based TT legitimately
      injects deeper transposition results (depth-leak), so TT-on sees tactics
      TT-off is horizon-blind to. Flagged 24/40 WAC positions, all benign.
    * "PVS on vs off (TT off) -> same score" — false: PVS is designed to
      re-search THROUGH the TT; run with TT off (a config the engine never uses)
      its fail-soft/aspiration interaction finds mates several plies late.
      Flagged 31/40, all artifacts of the unnatural ablation.
  Do not re-add score-transparency invariants. The invariants below hold in the
  REAL engine by definition of correctness, and target Coda's actual failure
  modes.

Invariants (real engine, full feature set, fixed depth):
  LEGAL : every PV move is legal from the root; bestmove == pv[0] and is legal.
          (Coda's illegal-PV = TT-collision bug class — the one that makes the
          lichess bot resign. CLAUDE.md: "critical, not cosmetic".)
  MATE  : a reported `mate N` must have a PV that delivers checkmate in exactly
          2N-1 plies (tests score_to/from_tt mate-ply adjustment + PV honesty).
  DET   : same position + config twice -> identical score, pv, bestmove, nodes.

Also retained (valid, structural, runs in pure alpha-beta):
  COLL  : pure A/B + TT, Hash=1 vs Hash=256 -> same score (torn-read / XOR-key
          collision handling; different eviction must not corrupt the result).

Usage:
  python3 scripts/invariant_harness.py [--engine ./coda] [--depth 12]
      [--epd testdata/wac.epd testdata/coda_blunders.epd testdata/arasan.epd]
      [--count 40] [--only LEGAL,MATE,DET,COLL] [--verbose]

Exit 1 if any invariant is violated.
"""
import argparse
import os
import re
import subprocess
import sys

try:
    import chess
except ImportError:
    print("needs python-chess (pip install chess)", file=sys.stderr)
    sys.exit(2)


class Engine:
    def __init__(self, path, env_extra=None, hash_mb=64, threads=1):
        env = dict(os.environ)
        if env_extra:
            env.update(env_extra)
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        self._cmd("uci"); self._wait("uciok")
        self._cmd(f"setoption name Threads value {threads}")
        self._cmd(f"setoption name Hash value {hash_mb}")
        self._cmd("isready"); self._wait("readyok")

    def _cmd(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, tok):
        for line in self.p.stdout:
            if line.strip().startswith(tok):
                return
        raise RuntimeError(f"engine died waiting for {tok!r}")

    def search(self, fen, depth):
        """Return (score, pv, bestmove). score=('cp'|'mate', int); pv=list[str]."""
        self._cmd("ucinewgame"); self._cmd("isready"); self._wait("readyok")
        self._cmd(f"position fen {fen}")
        self._cmd(f"go depth {depth}")
        score = nodes = bm = None
        pv = []
        for line in self.p.stdout:
            line = line.strip()
            if line.startswith("info") and " score " in line and "bound" not in line:
                m = re.search(r"score (cp|mate) (-?\d+)", line)
                if m:
                    score = (m.group(1), int(m.group(2)))
                mn = re.search(r" nodes (\d+)", line)
                if mn:
                    nodes = int(mn.group(1))
                mp = re.search(r" pv (.+)$", line)
                if mp:
                    pv = mp.group(1).split()
            elif line.startswith("bestmove"):
                bm = line.split()[1]
                break
        return score, pv, bm, nodes

    def quit(self):
        try:
            self._cmd("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# Pure alpha-beta + full TT, for the collision/size-invariance check.
PURE_TT_ON = {"DISABLE_ALL": "1", "ENABLE_TT_STORE": "1",
              "ENABLE_TT_CUTOFF": "1", "ENABLE_TT_NEARMISS": "1"}


def load_fens(paths, count):
    fens = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  (skip missing {path})", file=sys.stderr)
            continue
        n = 0
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            fen = " ".join(toks[:4])
            if len(toks) >= 6 and toks[4].isdigit() and toks[5].isdigit():
                fen += f" {toks[4]} {toks[5]}"
            else:
                fen += " 0 1"
            # keep only positions python-chess accepts (guards against junk EPD)
            try:
                chess.Board(fen)
            except Exception:
                continue
            fens.append((os.path.basename(path), fen))
            n += 1
            if n >= count:
                break
    return fens


def sstr(s):
    return f"{s[0]} {s[1]}" if s else "None"


def check_pv_legal(fen, pv):
    """Return (ok, bad_move, ply, board_after_full_legal_prefix)."""
    b = chess.Board(fen)
    for i, u in enumerate(pv):
        try:
            m = chess.Move.from_uci(u)
        except Exception:
            return False, u, i, b
        if m not in b.legal_moves:
            return False, u, i, b
        b.push(m)
    return True, None, None, b


def run(engine_path, depth, fens, only, verbose, threads=1):
    viol = []
    # DET only holds single-threaded (Lazy-SMP is nondeterministic); skip it if
    # threads>1. LEGAL/MATE must hold at any thread count.
    if threads > 1 and "DET" in only:
        only = [o for o in only if o != "DET"]
        print(f"(threads={threads}: dropping DET — SMP is nondeterministic by design)")
    real = Engine(engine_path, threads=threads) if ({"LEGAL", "MATE", "DET"} & set(only)) else None
    coll = None
    if "COLL" in only:
        coll = (Engine(engine_path, PURE_TT_ON, hash_mb=1),
                Engine(engine_path, PURE_TT_ON, hash_mb=256))

    n = {k: [0, 0] for k in ("LEGAL", "MATE", "DET", "COLL")}
    print(f"Engine: {engine_path}   depth={depth}   positions={len(fens)}\n")

    for src, fen in fens:
        if real is not None and ({"LEGAL", "MATE", "DET"} & set(only)):
            score, pv, bm, nodes = real.search(fen, depth)

            if "LEGAL" in only:
                n["LEGAL"][0] += 1
                ok, bad, ply, _ = check_pv_legal(fen, pv)
                bm_bad = bm not in (pv[0] if pv else None, None) and pv and bm != pv[0]
                if not ok:
                    n["LEGAL"][1] += 1
                    viol.append(("LEGAL illegal PV move", src, fen,
                                 f"pv[{ply}]={bad} illegal; pv={' '.join(pv)}  score={sstr(score)}"))
                elif threads == 1 and pv and bm and bm != pv[0]:
                    # Only valid single-threaded: under Lazy-SMP, info-pv lines
                    # from helper threads interleave on stdout, so the last
                    # printed PV need not be the one bestmove came from.
                    n["LEGAL"][1] += 1
                    viol.append(("LEGAL bestmove != pv[0]", src, fen,
                                 f"bestmove={bm} pv[0]={pv[0]}  score={sstr(score)}"))

            if "MATE" in only and score and score[0] == "mate":
                n["MATE"][0] += 1
                mate_n = score[1]
                ok, bad, ply, bend = check_pv_legal(fen, pv)
                if ok and bend.is_checkmate():
                    want = 2 * abs(mate_n) - 1
                    if len(pv) != want:
                        n["MATE"][1] += 1
                        viol.append(("MATE ply mismatch", src, fen,
                                     f"score=mate {mate_n} but PV mates in {len(pv)} plies (want {want}); pv={' '.join(pv)}"))
                elif ok and not bend.is_checkmate() and len(pv) >= 2 * abs(mate_n) - 1:
                    # PV is long enough to have shown the mate but doesn't -> suspect
                    n["MATE"][1] += 1
                    viol.append(("MATE PV not mating", src, fen,
                                 f"score=mate {mate_n}, PV len {len(pv)} does not end in checkmate; pv={' '.join(pv)}"))
                elif verbose and ok:
                    print(f"  MATE note {fen}  mate {mate_n}, PV truncated to {len(pv)} plies (can't verify)")

            if "DET" in only:
                n["DET"][0] += 1
                r2 = real.search(fen, depth)
                if (score, pv, bm, nodes) != r2:
                    n["DET"][1] += 1
                    viol.append(("DET nondeterministic", src, fen,
                                 f"run1 score={sstr(score)} nodes={nodes} bm={bm}; run2 score={sstr(r2[0])} nodes={r2[3]} bm={r2[2]}"))

        if "COLL" in only:
            a = coll[0].search(fen, depth)
            b = coll[1].search(fen, depth)
            n["COLL"][0] += 1
            if a[0] != b[0]:
                n["COLL"][1] += 1
                viol.append(("COLL hash-size score diverges", src, fen,
                             f"Hash=1 score={sstr(a[0])}; Hash=256 score={sstr(b[0])}"))

    for e in ([real] if real else []) + (list(coll) if coll else []):
        e.quit()

    print("=== Results ===")
    labels = {
        "LEGAL": "PV / bestmove legality (real engine)",
        "MATE": "mate-PV soundness (reported mate delivers mate)",
        "DET": "determinism (same config twice)",
        "COLL": "TT collision-safety (Hash=1 vs 256, pure A/B)",
    }
    for inv in ("LEGAL", "MATE", "DET", "COLL"):
        if inv not in only:
            continue
        chk, v = n[inv]
        status = "PASS" if v == 0 else f"*** {v} VIOLATIONS ***"
        print(f"  {inv:<5} {labels[inv]:<48} {chk} checked  ->  {status}")

    if viol:
        print("\n=== Violation detail ===")
        for name, src, fen, detail in viol:
            print(f"\n  [{name}]  ({src})\n    fen: {fen}\n    {detail}")
    return 1 if viol else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="./coda")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--epd", nargs="+", default=[
        "testdata/wac.epd", "testdata/coda_blunders.epd", "testdata/arasan.epd"])
    ap.add_argument("--count", type=int, default=40, help="positions per EPD")
    ap.add_argument("--only", default="LEGAL,MATE,DET,COLL")
    ap.add_argument("--threads", type=int, default=1,
                    help="engine Threads for LEGAL/MATE (SMP illegal-PV probe)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    only = [x.strip() for x in args.only.split(",") if x.strip()]
    fens = load_fens(args.epd, args.count)
    if not fens:
        print("no positions loaded", file=sys.stderr)
        return 2
    return run(args.engine, args.depth, fens, only, args.verbose, args.threads)


if __name__ == "__main__":
    sys.exit(main())
