#!/usr/bin/env python3
"""For each clean-hash candidate, SF-eval Coda's clean-hash bestmove
to compare with SF-eval of the played move.

Inputs:
  - JSONL with fen_before + blunder_uci + sf_best_uci + score_before + score_after
    (the played-move SF data)
  - blunder_ablation CSV with bestmove for each candidate (the clean-hash data)

Output: CSV with columns:
  fen, played, sf_best, coda_clean,
  sf_score_before, sf_score_after_played, sf_score_after_clean,
  delta_played (= score_after_played - score_before),
  delta_clean (= score_after_clean - score_before),
  recovery_cp (= delta_clean - delta_played; positive = Coda's clean choice was better),
  category: matches_sf | recovers_>=50cp | partial_<50cp | same_or_worse

Verdict logic per candidate:
  - matches_sf: Coda's clean bestmove == sf_best_uci
  - recovers_50cp: clean choice gives ≥ +50cp improvement over played
  - partial: 0 < improvement < 50
  - same_or_worse: improvement ≤ 0
"""
import argparse
import csv
import json
import multiprocessing as mp
import re
import subprocess
from pathlib import Path

SF = "/home/adam/chess/engines/Stockfish/src/stockfish"


class SFEngine:
    """Persistent SF process — info lines flush correctly with readline + bufsize=1.
    The communicate() approach loses info lines."""
    def __init__(self, depth: int):
        self.depth = depth
        self.proc = subprocess.Popen(
            [SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name Hash value 128")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, c):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token, max_lines=5000):
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                return None
            if token in line:
                return line
        return None

    def score_after(self, fen: str, move_uci: str) -> int | None:
        self._send("ucinewgame")
        self._send(f"position fen {fen} moves {move_uci}")
        self._send(f"go depth {self.depth}")
        last_score = None
        for _ in range(20000):
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line.startswith("info "):
                sm = re.search(r"score (cp|mate) (-?\d+)", line)
                if sm:
                    if sm.group(1) == "cp":
                        last_score = int(sm.group(2))
                    else:
                        mn = int(sm.group(2))
                        last_score = 10000 if mn > 0 else -10000
            elif line.startswith("bestmove "):
                return last_score
        return None

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def chunk_worker(args):
    chunk, depth = args
    sf = SFEngine(depth)
    results = []
    try:
        for fen, move in chunk:
            try:
                results.append(sf.score_after(fen, move))
            except Exception:
                results.append(None)
    finally:
        sf.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path, help="JSONL from find_pruning_candidates.py")
    ap.add_argument("abl_csv", type=Path, help="CSV from blunder_ablation.py (baseline only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cands = [json.loads(l) for l in args.candidates.open() if l.strip()]
    by_fen = {c["fen_before"]: c for c in cands}

    abl_rows = list(csv.DictReader(args.abl_csv.open()))

    # Build (fen, move) tuples to evaluate via SF
    tasks = []
    for r in abl_rows:
        if r["bestmove"] in (None, "", "None"):
            continue
        c = by_fen.get(r["fen"])
        if not c:
            continue
        tasks.append((r["fen"], r["bestmove"], r["blunder_uci"], r["sf_best_uci"], c))

    print(f"Evaluating {len(tasks)} (fen, clean_bestmove) pairs at SF depth {args.depth}")

    # Chunk the tasks across workers so each worker keeps a persistent SF
    n_workers = args.workers
    chunks: list[list] = [[] for _ in range(n_workers)]
    for i, t in enumerate(tasks):
        chunks[i % n_workers].append((t[0], t[1]))

    with mp.Pool(n_workers) as pool:
        chunk_results = pool.map(chunk_worker, [(c, args.depth) for c in chunks])
    # Flatten back to original task order
    scores = [None] * len(tasks)
    for ci, chunk in enumerate(chunks):
        for j, _ in enumerate(chunk):
            scores[ci + j * n_workers] = chunk_results[ci][j]

    rows = []
    for (fen, clean_bm, played, sf_best, c), score_after_clean_stm in zip(tasks, scores):
        score_before = c["score_before"]  # Coda-side POV before its move
        score_after_played_codapov = c["score_after"]  # Already Coda-POV (sign-flipped at SF time)

        # score_after_clean_stm is from STM-POV at fen_after_clean.
        # STM at fen_after_clean is the OPPONENT (Coda just moved).
        # Coda-POV = -score_after_clean_stm
        score_after_clean_codapov = (
            -score_after_clean_stm if score_after_clean_stm is not None else None
        )

        delta_played = (
            score_after_played_codapov - score_before
            if (score_after_played_codapov is not None and score_before is not None)
            else None
        )
        delta_clean = (
            score_after_clean_codapov - score_before
            if (score_after_clean_codapov is not None and score_before is not None)
            else None
        )
        recovery = (
            delta_clean - delta_played
            if (delta_played is not None and delta_clean is not None)
            else None
        )

        if clean_bm == sf_best:
            cat = "matches_sf"
        elif recovery is None:
            cat = "no_data"
        elif recovery >= 50:
            cat = "recovers_>=50cp"
        elif recovery > 0:
            cat = "partial_<50cp"
        else:
            cat = "same_or_worse"

        rows.append({
            "fen": fen,
            "opponent": c.get("opponent"),
            "ply": c.get("ply"),
            "played": played,
            "sf_best": sf_best,
            "coda_clean": clean_bm,
            "sf_score_before": score_before,
            "sf_score_after_played_codapov": score_after_played_codapov,
            "sf_score_after_clean_codapov": score_after_clean_codapov,
            "delta_played": delta_played,
            "delta_clean": delta_clean,
            "recovery_cp": recovery,
            "category": cat,
        })

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")

    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    print(f"\n=== Recovery verdict ({len(rows)} candidates) ===")
    for cat in ("matches_sf", "recovers_>=50cp", "partial_<50cp", "same_or_worse", "no_data"):
        n = cat_counts.get(cat, 0)
        print(f"  {cat:<20} {n:3d}  ({100*n/len(rows):5.1f}%)")

    real_recoveries = sum(1 for r in rows
                           if r["category"] in ("matches_sf", "recovers_>=50cp"))
    print(f"\n=== REAL recoveries (matches_sf OR recovery ≥50cp): {real_recoveries}/{len(rows)} ({100*real_recoveries/len(rows):.0f}%) ===")
    print(f"   ↑ these positions are 'clean hash beats game-time choice per SF'")
    print(f"\n=== False positives (different but no SF improvement): {cat_counts.get('same_or_worse', 0) + cat_counts.get('partial_<50cp', 0)}/{len(rows)} ===")
    print(f"   ↑ these are 'just a different bad move; not real recovery'")


if __name__ == "__main__":
    main()
