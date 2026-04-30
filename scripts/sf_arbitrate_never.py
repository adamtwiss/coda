#!/usr/bin/env python3
"""SF-arbitrate NEVER-converged candidates at deep depths.

For each candidate (the cases where Coda never picks SF-best within
d24), run a fresh SF process on the FEN at depth 28 and depth 32.
Record SF's best move and score. Compare to the original blunder
delta to classify:
  - eval-blind: SF's deep-probe still says >100cp better than Coda's choice
    AND deep-probe agrees with the d18 SF-best from original arbitration
    → Coda's net misses something SF's net+search can see at any depth
  - ambiguous: SF's deep-probe flips to something else (different best move
    or significantly different evaluation) → original arbitration was
    optimistic; not a clean Coda eval-blind

Persistent SF process (avoid quit-before-bestmove pattern).
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


SCORE_CP_RE = re.compile(r"score (cp|mate) (-?\d+)")
BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)")
DEPTH_RE = re.compile(r" depth (\d+) ")


class SFEngine:
    def __init__(self, sf_bin: Path, hash_mb: int = 256, threads: int = 1):
        self.proc = subprocess.Popen(
            [str(sf_bin.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value {threads}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, c):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token, max_lines=2_000_000):
        last = None
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                return None, None
            if line.startswith("info") and " pv " in line:
                last = line
            if token in line:
                return line, last
        return None, last

    def probe(self, fen: str, depth: int):
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        bm_line, info_line = self._read_until("bestmove")
        if bm_line is None:
            return None, None, None
        bm_match = BESTMOVE_RE.match(bm_line.strip())
        bestmove = bm_match.group(1) if bm_match else None
        score_cp = None
        achieved_depth = None
        if info_line:
            sm = SCORE_CP_RE.search(info_line)
            if sm:
                kind, val = sm.group(1), int(sm.group(2))
                # Convert mate to large cp
                if kind == "mate":
                    score_cp = (32000 - abs(val)) * (1 if val > 0 else -1)
                else:
                    score_cp = val
            dm = DEPTH_RE.search(info_line)
            if dm:
                achieved_depth = int(dm.group(1))
        return bestmove, score_cp, achieved_depth

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("convergence_csv", type=Path,
                    help="convergence-depth probe CSV (NEVER cases extracted)")
    ap.add_argument("candidates_jsonl", type=Path,
                    help="original candidates JSONL (for FEN + sf_best_uci + delta)")
    ap.add_argument("--sf", type=Path,
                    default=Path("/home/adam/chess/engines/Stockfish/src/stockfish"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depths", default="28,32",
                    help="comma-separated depths to probe")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=256)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]

    # Load NEVER cases from convergence CSV
    never_idx = set()
    with args.convergence_csv.open() as f:
        for r in csv.DictReader(f):
            if r["convergence_depth"] == "":
                never_idx.add(int(r["idx"]))
    print(f"NEVER cases: {len(never_idx)}", file=sys.stderr)

    # Load matching candidates
    candidates = []
    with args.candidates_jsonl.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            c = json.loads(line)
            if i in never_idx:
                c["idx"] = i
                candidates.append(c)
    print(f"Probing {len(candidates)} NEVER candidates at depths {depths}", file=sys.stderr)

    sf = SFEngine(args.sf, hash_mb=args.hash, threads=args.threads)

    out_rows = []
    for c in candidates:
        idx = c["idx"]
        opp = c["opponent"]
        fen = c["fen_before"]
        played = c["blunder_uci"]
        sf_d18_best = c["sf_best_uci"]
        delta_cp = c.get("delta_cp", 0)
        score_before = c.get("score_before", 0)
        score_after = c.get("score_after", 0)

        row = {
            "idx": idx,
            "opponent": opp,
            "ply": c["ply"],
            "fen": fen,
            "played_uci": played,
            "sf_d18_best": sf_d18_best,
            "score_before_d18": score_before,
            "score_after_d18": score_after,
            "delta_d18": delta_cp,
        }
        for d in depths:
            t0 = time.time()
            bm, score, achieved_d = sf.probe(fen, d)
            elapsed = time.time() - t0
            row[f"sf_d{d}_best"] = bm
            row[f"sf_d{d}_score"] = score
            row[f"sf_d{d}_elapsed"] = round(elapsed, 1)
            row[f"sf_d{d}_achieved"] = achieved_d
            print(f"  [idx={idx:2d}] vs {opp:12s} ply {c['ply']:3d} d{d}: bm={bm} score={score} "
                  f"(elapsed {elapsed:.1f}s)", file=sys.stderr)

        # Classification
        d_high = depths[-1]
        sf_high_best = row.get(f"sf_d{d_high}_best")
        sf_high_score = row.get(f"sf_d{d_high}_score") or 0
        # SF score is from position-to-move's perspective; original delta is Coda's-eval-perspective.
        # For classification, just check whether SF's deep best move differs from d18 best,
        # and whether it's different from what Coda played.
        coda_picked_sf_best = (played == sf_high_best)
        sf_d18_d_high_agree = (sf_d18_best == sf_high_best)
        if coda_picked_sf_best:
            klass = "coda_now_picks_sf_deep_best"
        elif sf_d18_d_high_agree:
            klass = "eval_blind_durable"  # SF's d32 still says same move; Coda's net misses it
        else:
            klass = "ambiguous_sf_flipped"  # SF's d32 disagrees with d18; original arbitration optimistic
        row["class"] = klass
        out_rows.append(row)

    sf.quit()

    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {len(out_rows)} rows to {args.out}", file=sys.stderr)

    # Summary
    from collections import Counter
    klass_count = Counter(r["class"] for r in out_rows)
    print("\n=== NEVER-case classification ===")
    for k in ("eval_blind_durable", "ambiguous_sf_flipped", "coda_now_picks_sf_deep_best"):
        print(f"  {k}: {klass_count.get(k, 0)}/{len(out_rows)}")


if __name__ == "__main__":
    main()
