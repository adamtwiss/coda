#!/usr/bin/env python3
"""Disentangle TT-state pollution vs non-monotonic recovery.

The sudden-bucket convergence probe (probe_convergence_depth.py) found 14
"negative-deficit" cases where Coda fresh-TT picks SF-best at some depth
LOWER than the actual game-time played_depth. Two interpretations remain:

  (A) TT-state pollution: in-game TT was contaminated by prior moves; the
      same engine fresh-TT at played_depth would pick SF-best.
  (B) Non-monotonic recovery: the convergence probe picked SF-best at a
      shallow depth by noise; fresh-TT at played_depth actually picks the
      played-move (or some other move).

This script discriminates by running Coda fresh-TT at *exactly* played_depth
for each of the 14 cases and comparing to (sf_best, played).

Outputs CSV: idx, opp, ply, played_d, conv_d, fen, played, sf_best, fresh_d,
fresh_bestmove, fresh_score, classification.

Classifications:
  - tt_pollution_real     : fresh@played_d == sf_best  (B above is wrong, TT pollution real)
  - non_monotonic_noise   : fresh@played_d == played    (A above is wrong, methodology artifact)
  - third_move            : fresh@played_d != either    (something else; record but not actionable here)
"""
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path


CODA_DEFAULT = Path("./coda")


class CodaEngine:
    """Persistent ./coda process — reads to bestmove BEFORE sending next command."""
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

    def search(self, fen: str, depth: int):
        # ucinewgame clears TT — gives fresh-state probe
        self._send("ucinewgame")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        last_score = None
        for _ in range(2_000_000):
            line = self.proc.stdout.readline()
            if not line:
                return None, None
            line = line.strip()
            if line.startswith("info ") and "score" in line:
                m = re.search(r"score cp (-?\d+)", line)
                if m:
                    last_score = int(m.group(1))
            elif line.startswith("bestmove "):
                return line.split()[1], last_score
        return None, last_score

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path,
                    default=Path("/tmp/sudden_with_depth.jsonl"),
                    help="enriched candidates JSONL with played_depth")
    ap.add_argument("--convergence", type=Path,
                    default=Path("/tmp/convergence_sudden.csv"),
                    help="convergence-depth probe CSV")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coda", type=Path, default=CODA_DEFAULT)
    ap.add_argument("--hash", type=int, default=64)
    args = ap.parse_args()

    cands = [json.loads(l) for l in args.candidates.open() if l.strip()]
    conv = list(csv.DictReader(args.convergence.open()))

    # Negative-deficit set: convergence_depth < played_depth
    targets = []
    for r in conv:
        idx = int(r["idx"])
        if idx >= len(cands):
            continue
        cd_raw = r["convergence_depth"]
        if not cd_raw:
            continue
        cd = int(cd_raw)
        pd = cands[idx].get("played_depth")
        if pd is None or cd >= pd:
            continue
        targets.append({
            "idx": idx,
            "opponent": r["opponent"],
            "ply": int(r["ply"]),
            "fen": r["fen"],
            "played": r["played_uci"],
            "sf_best": r["sf_best_uci"],
            "conv_d": cd,
            "played_d": pd,
        })

    print(f"Probing {len(targets)} negative-deficit cases at played_depth (fresh TT)")

    eng = CodaEngine(args.coda, hash_mb=args.hash)
    rows = []
    try:
        for t in targets:
            t0 = time.time()
            bm, sc = eng.search(t["fen"], t["played_d"])
            elapsed = time.time() - t0

            if bm == t["sf_best"]:
                klass = "tt_pollution_real"
            elif bm == t["played"]:
                klass = "non_monotonic_noise"
            else:
                klass = "third_move"

            row = {
                **t,
                "fresh_bm": bm,
                "fresh_score": sc,
                "klass": klass,
                "elapsed": round(elapsed, 1),
            }
            rows.append(row)
            print(f"  [idx={t['idx']:>2d}] vs {t['opponent']:<12s} "
                  f"conv_d={t['conv_d']:>2d} played_d={t['played_d']:>2d}  "
                  f"played={t['played']} sf={t['sf_best']} fresh@d{t['played_d']}={bm} "
                  f"({klass})  [{elapsed:.1f}s]")
    finally:
        eng.close()

    fieldnames = ["idx", "opponent", "ply", "fen", "played", "sf_best",
                  "conv_d", "played_d", "fresh_bm", "fresh_score", "klass", "elapsed"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    from collections import Counter
    dist = Counter(r["klass"] for r in rows)
    print("\n=== Classification ===")
    for k, v in dist.most_common():
        print(f"  {k}: {v}/{len(rows)} ({100*v/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
