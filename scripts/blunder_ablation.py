#!/usr/bin/env python3
"""Replay blunder positions with each pruning feature disabled in turn.

For each blunder (FEN + actual move played + depth), run ./coda with
various NO_X env vars set and record which move it picks. If the move
differs from the recorded blunder, the ablation is said to "recover"
the position — that pruning feature was responsible for missing the
right move at the original depth.

Output: CSV with one row per (blunder, ablation) pair, plus a summary
table of recovery rates per ablation.

Usage:
  ./blunder_ablation.py blunders.jsonl --depth 12 --out ablation.csv
  ./blunder_ablation.py blunders.jsonl --movetime 1000 --out ablation.csv
  ./blunder_ablation.py blunders.jsonl --depth 12 --abl NO_NMP NO_LMP --out a.csv
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ABLATIONS = [
    ("baseline", {}),  # control: same code, no env vars
    ("NO_NMP", {"NO_NMP": "1"}),
    ("NO_LMP", {"NO_LMP": "1"}),
    ("NO_RFP", {"NO_RFP": "1"}),
    ("NO_FUTILITY", {"NO_FUTILITY": "1"}),
    ("NO_SEE_PRUNE", {"NO_SEE_PRUNE": "1"}),
    ("NO_HIST_PRUNE", {"NO_HIST_PRUNE": "1"}),
    ("NO_BAD_NOISY", {"NO_BAD_NOISY": "1"}),
    ("NO_LMR", {"NO_LMR": "1"}),
    ("NO_PROBCUT", {"NO_PROBCUT": "1"}),
    # Combined
    ("NO_NMP_LMP", {"NO_NMP": "1", "NO_LMP": "1"}),
    ("NO_ALL_PRUNE", {
        "NO_NMP": "1", "NO_LMP": "1", "NO_RFP": "1",
        "NO_FUTILITY": "1", "NO_SEE_PRUNE": "1",
        "NO_HIST_PRUNE": "1", "NO_BAD_NOISY": "1",
    }),
]


def run_coda(coda_bin: Path, fen: str, depth: int | None, movetime_ms: int | None,
             env_extra: dict[str, str], hash_mb: int = 64) -> tuple[str | None, int | None]:
    """Run a single position via UCI on a fresh ./coda process. Returns (bestmove_uci, score_cp).

    CRITICAL: must read until 'bestmove' BEFORE sending 'quit'. Sending all
    cmds at once via communicate() with quit included makes Coda exit before
    search completes and produces premature bestmoves. Burnt 2026-04-29.
    """
    import os
    env = os.environ.copy()
    env.update(env_extra)
    proc = subprocess.Popen(
        [str(coda_bin.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env, text=True, bufsize=1,
    )
    def send(c):
        proc.stdin.write(c + "\n")
        proc.stdin.flush()
    def read_until(token, max_lines=200000):
        for _ in range(max_lines):
            line = proc.stdout.readline()
            if not line:
                return None
            if token in line:
                return line
        return None
    try:
        send("uci")
        read_until("uciok")
        send(f"setoption name Hash value {hash_mb}")
        send("isready")
        read_until("readyok")
        send("ucinewgame")
        send(f"position fen {fen}")
        if depth:
            send(f"go depth {depth}")
        else:
            send(f"go movetime {movetime_ms}")
        bestmove = None
        last_score = None
        for _ in range(200000):
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("info ") and "score" in line:
                m = re.search(r"score cp (-?\d+)", line)
                if m:
                    last_score = int(m.group(1))
                m = re.search(r"score mate (-?\d+)", line)
                if m:
                    mn = int(m.group(1))
                    last_score = (30000 - abs(mn)) * (1 if mn > 0 else -1)
            elif line.startswith("bestmove "):
                bestmove = line.split()[1]
                break
        send("quit")
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return bestmove, last_score
    except Exception:
        try: proc.kill()
        except Exception: pass
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blunders", type=Path, help="JSONL from extract_blunder_positions.py")
    ap.add_argument("--out", type=Path, required=True, help="CSV output")
    ap.add_argument("--coda", type=Path, default=Path("./coda"), help="Path to coda binary")
    ap.add_argument("--depth", type=int, help="Fixed-depth search (overrides movetime)")
    ap.add_argument("--movetime", type=int, default=2000, help="Movetime per position (ms)")
    ap.add_argument("--hash", type=int, default=64, help="Hash size MB per probe")
    ap.add_argument("--abl", nargs="+", help="Restrict to specific ablation names")
    ap.add_argument("--limit", type=int, default=0, help="Cap blunders to first N")
    args = ap.parse_args()

    blunders = []
    with args.blunders.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            blunders.append(json.loads(line))
    if args.limit:
        blunders = blunders[:args.limit]

    if args.abl:
        ablations = [(n, e) for n, e in ABLATIONS if n in args.abl]
    else:
        ablations = ABLATIONS

    print(f"Loaded {len(blunders)} blunder positions; running {len(ablations)} ablations each "
          f"({len(blunders) * len(ablations)} probes total)")

    rows = []
    for bi, blunder in enumerate(blunders):
        sf_best = blunder.get("sf_best_uci")  # may be None for old JSONLs
        for abl_name, env_extra in ablations:
            bestmove, score = run_coda(
                args.coda, blunder["fen_before"],
                args.depth, args.movetime, env_extra, args.hash,
            )
            differs = (bestmove is not None and bestmove != blunder["blunder_uci"])
            matches_sf = (sf_best is not None and bestmove == sf_best)
            rows.append({
                "blunder_idx": bi,
                "opponent": blunder["opponent"],
                "ply": blunder["ply"],
                "fen": blunder["fen_before"],
                "blunder_uci": blunder["blunder_uci"],
                "blunder_san": blunder["blunder_san"],
                "sf_best_uci": sf_best,
                "ablation": abl_name,
                "bestmove": bestmove,
                "score_cp": score,
                "differs_from_played": int(differs),
                "matches_sf_best": int(matches_sf),
            })
            tag = ""
            if matches_sf:
                tag = "MATCHES_SF"
            elif differs:
                tag = "DIFFERS"
            print(f"  [{bi+1}/{len(blunders)}] {abl_name:20s} bestmove={bestmove} "
                  f"(played={blunder['blunder_uci']} sf={sf_best}) {tag}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary — split clean-hash test (baseline) from feature ablations
    print(f"\n=== Per-ablation rates ({len(blunders)} candidates) ===")
    print(f"  {'ablation':<20} {'differs':>10} {'matches_sf':>12}")
    for abl_name, _ in ablations:
        relevant = [r for r in rows if r["ablation"] == abl_name]
        differs = sum(r["differs_from_played"] for r in relevant)
        matches = sum(r["matches_sf_best"] for r in relevant)
        n = len(relevant) or 1
        prefix = "**" if abl_name == "baseline" else "  "
        print(f"  {prefix}{abl_name:<18} {differs:3d}/{n:<3d} ({100*differs/n:5.1f}%) "
              f"{matches:3d}/{n:<3d} ({100*matches/n:5.1f}%)")
    print(f"\n  ** baseline = clean-hash test: if differs > 0, those are state-pollution candidates")
    print(f"  per-feature ablations: candidates where ablation X recovers are pruning-blind-spot signal for that feature")

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
