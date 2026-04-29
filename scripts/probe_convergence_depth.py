#!/usr/bin/env python3
"""For each candidate, find the smallest depth at which Coda picks SF-best.

If +1 ply suffices over the played-at depth → NPS-bound (more search
finds it cheaply).
If many plies needed → ordering/search/eval-bound (depth alone won't
close the gap economically).
If never within max-depth → eval-bound (depth doesn't help).

For each candidate:
  - Walk depths in [start_depth, ..., max_depth]
  - Probe Coda at that depth
  - Stop when bestmove == sf_best_uci

Output: CSV with idx, opponent, ply, played_depth, sf_best_uci,
played_uci, convergence_depth (None if never), final_bestmove,
final_score.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


CODA_DEFAULT = Path("./coda")


class CodaEngine:
    """Persistent ./coda process — reads to bestmove BEFORE sending next command.
    Sending 'quit' too early (via communicate) terminates search prematurely."""
    def __init__(self, coda_bin: Path, hash_mb: int = 64):
        self.proc = subprocess.Popen(
            [str(coda_bin.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, c):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token, max_lines=200000):
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                return None
            if token in line:
                return line
        return None

    def search(self, fen: str, depth: int) -> tuple[str | None, int | None]:
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last_score = None
        bestmove = None
        for _ in range(200000):
            line = self.proc.stdout.readline()
            if not line:
                return None, None
            line = line.strip()
            if line.startswith("info ") and "score" in line:
                m = re.search(r"score cp (-?\d+)", line)
                if m:
                    last_score = int(m.group(1))
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                return bestmove, last_score
        return bestmove, last_score

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def run_coda(coda_bin: Path, fen: str, depth: int, timeout_sec: int = 600) -> tuple[str | None, int | None]:
    """Wrapper for the old API — creates a fresh engine each call."""
    eng = CodaEngine(coda_bin)
    try:
        return eng.search(fen, depth)
    finally:
        eng.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coda", type=Path, default=CODA_DEFAULT)
    ap.add_argument("--depths", default="14,16,18,20,22,24,26,28",
                    help="Comma-separated depths to probe (in order)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]

    cands = [json.loads(l) for l in args.candidates.open() if l.strip()]
    if args.limit:
        cands = cands[: args.limit]

    rows = []
    for i, c in enumerate(cands):
        sf_best = c.get("sf_best_uci")
        played = c.get("blunder_uci")
        if not sf_best:
            continue
        fen = c["fen_before"]
        convergence_depth = None
        last_bm = None
        last_score = None
        per_depth_bms = []
        t_start = time.time()
        # Use a single persistent Coda process for all depths on this candidate
        # (ucinewgame between probes clears TT — same as fresh process)
        eng = CodaEngine(args.coda)
        try:
            for d in depths:
                bm, sc = eng.search(fen, d)
                per_depth_bms.append((d, bm))
                last_bm = bm
                last_score = sc
                if bm == sf_best:
                    convergence_depth = d
                    break
        finally:
            eng.close()
        elapsed = time.time() - t_start
        row = {
            "idx": i,
            "opponent": c.get("opponent"),
            "ply": c.get("ply"),
            "fen": fen,
            "played_uci": played,
            "sf_best_uci": sf_best,
            "convergence_depth": convergence_depth,
            "final_bestmove": last_bm,
            "final_score": last_score,
            "elapsed_sec": round(elapsed, 1),
            "depths_tried": ",".join(str(d) for d, _ in per_depth_bms),
            "bms_at_depths": ",".join(bm or "?" for _, bm in per_depth_bms),
        }
        rows.append(row)
        verdict = (f"converges at d{convergence_depth}" if convergence_depth
                   else f"NEVER within {depths[-1]}")
        print(f"  [{i+1}/{len(cands)}] vs {c['opponent']:<10} ply {c['ply']:>3}  "
              f"{verdict}  (elapsed {elapsed:.1f}s)  bms={[bm for _, bm in per_depth_bms]}")

    if rows:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    # Summary
    converged = [r for r in rows if r["convergence_depth"] is not None]
    never = [r for r in rows if r["convergence_depth"] is None]
    print(f"\n=== Convergence summary ===")
    print(f"  Converged within depth {depths[-1]}: {len(converged)}/{len(rows)}")
    print(f"  Never converged: {len(never)}/{len(rows)}")
    if converged:
        from collections import Counter
        depth_dist = Counter(r["convergence_depth"] for r in converged)
        print(f"\n  Convergence depth distribution:")
        for d in sorted(depth_dist):
            n = depth_dist[d]
            print(f"    d{d}: {n}/{len(converged)}")


if __name__ == "__main__":
    main()
