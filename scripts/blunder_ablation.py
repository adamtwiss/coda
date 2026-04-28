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
    """Run a single position via UCI on a fresh ./coda process. Returns (bestmove_uci, score_cp)."""
    import os
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [str(coda_bin.resolve())]
    cmds = [
        "uci",
        f"setoption name Hash value {hash_mb}",
        "ucinewgame",
        f"position fen {fen}",
        f"go depth {depth}" if depth else f"go movetime {movetime_ms}",
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, text=True,
    )
    try:
        out, _err = proc.communicate(input="\n".join(cmds) + "\nquit\n", timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, None

    bestmove = None
    last_score = None
    for line in out.splitlines():
        m = re.match(r"^bestmove (\S+)", line)
        if m:
            bestmove = m.group(1)
            break
        m = re.match(r"^info .* score cp (-?\d+)", line)
        if m:
            last_score = int(m.group(1))
        m = re.match(r"^info .* score mate (-?\d+)", line)
        if m:
            last_score = (30000 - abs(int(m.group(1)))) * (1 if int(m.group(1)) > 0 else -1)
    return bestmove, last_score


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
