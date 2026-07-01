#!/usr/bin/env python3
"""Natural-subpopulation comparison: does Coda's eval error vs LC0 ground
truth differ between positions that naturally have a bishop on the board
and positions that naturally don't, controlling for total material?

This is the non-synthetic follow-up to scripts/overscore_perturbation_probe.py.
That probe edited positions (removed pieces) and showed a real but unstable
signal confounded by the free-material-imbalance the edit itself introduces
(see docs/overscore_perturbation_probe_2026-06-30.md). This script instead
bins REAL game positions by piece-presence and material band and compares
Coda's eval error directly against LC0 ground truth (no SF proxy needed —
`coda eval-dist --csv` already emits the binpack's own LC0-calibrated score
per position).

Usage:
  ./coda eval-dist -i <binpack> -c 300000 --quiet-only \
      --csv /tmp/natural_sample.csv -n net-E6C62000.nnue
  python3 scripts/overscore_natural_subpop.py --csv /tmp/natural_sample.csv

Or let this script invoke eval-dist itself:
  python3 scripts/overscore_natural_subpop.py \
      --binpack /training/test80-jul2024/training-run1-test80-20240701-0017.no-db.binpack \
      --csv /tmp/natural_sample.csv --count 300000
"""
import argparse
import csv
import math
import statistics
import subprocess
import sys

import chess

# total-piece-count bands (incl. both kings), matching the existing
# docs/overrate_eval_investigation_2026-06-30.md taxonomy resolution
BANDS = [(2, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 25), (26, 29), (30, 32)]

PIECE_TYPES = {
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
}


def band_for(n):
    for lo, hi in BANDS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return None


def run_eval_dist(coda, binpack, nnue, count, csv_out, min_fullmove_note=True):
    cmd = [coda, "eval-dist", "-i", binpack, "-c", str(count), "--quiet-only", "--csv", csv_out]
    if nnue:
        cmd += ["-n", nnue]
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def mean_se(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0
    m = sum(values) / n
    if n < 2:
        return m, 0.0, n
    sd = statistics.stdev(values)
    return m, sd / math.sqrt(n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binpack", help="raw T80 binpack to harvest from (skipped if --csv already exists and --reuse)")
    ap.add_argument("--csv", required=True, help="eval-dist CSV path (read if exists+--reuse, else written here)")
    ap.add_argument("--reuse", action="store_true", help="skip eval-dist if --csv already exists")
    ap.add_argument("--coda", default="./coda")
    ap.add_argument("--nnue", default=None)
    ap.add_argument("--count", type=int, default=300000)
    ap.add_argument("--min-fullmove", type=int, default=0)
    ap.add_argument("--max-abs-lc0", type=float, default=0, help="restrict to |lc0_score_white_cp|<=N (0=off); matches the balanced-band filter used in the original overrate heldout set")
    ap.add_argument("--per-position-csv", default=None, help="optional: dump enriched per-position rows here")
    args = ap.parse_args()

    import os
    if not (args.reuse and os.path.exists(args.csv)):
        if not args.binpack:
            print("error: --csv doesn't exist yet and no --binpack given to generate it", file=sys.stderr)
            sys.exit(1)
        run_eval_dist(args.coda, args.binpack, args.nnue, args.count, args.csv)

    rows = []
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            fen = r["fen"]
            coda_cp = float(r["coda_eval_white_cp"])
            lc0_cp = float(r["lc0_score_white_cp"])
            if args.max_abs_lc0 > 0 and abs(lc0_cp) > args.max_abs_lc0:
                continue
            board = chess.Board(fen)
            if board.fullmove_number < args.min_fullmove:
                continue
            total_pieces = len(board.piece_map())
            band = band_for(total_pieces)
            if band is None:
                continue
            err = coda_cp - lc0_cp
            counts = {name: len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))
                      for name, pt in PIECE_TYPES.items()}
            rows.append(dict(fen=fen, total_pieces=total_pieces, band=band, err=err, abs_err=abs(err), **counts))

    print(f"loaded {len(rows)} positions (min_fullmove>={args.min_fullmove})\n", file=sys.stderr)

    if args.per_position_csv:
        with open(args.per_position_csv, "w", newline="") as f:
            fields = list(rows[0].keys()) if rows else []
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} enriched rows to {args.per_position_csv}", file=sys.stderr)

    overall_abs = [r["abs_err"] for r in rows]
    m, se, n = mean_se(overall_abs)
    print(f"=== overall: mean|err|={m:.1f} (se={se:.2f}, n={n}) ===\n")

    for piece_name in PIECE_TYPES:
        print(f"--- {piece_name}-presence, controlling for total-piece band ---")
        print(f"{'band':8s} {'has '+piece_name:>14s} {'no '+piece_name:>14s} {'delta(has-no)':>14s} {'n_has':>7s} {'n_no':>7s}")
        for lo, hi in BANDS:
            band = f"{lo}-{hi}"
            band_rows = [r for r in rows if r["band"] == band]
            has = [r["abs_err"] for r in band_rows if r[piece_name] >= 1]
            no = [r["abs_err"] for r in band_rows if r[piece_name] == 0]
            if len(has) < 20 or len(no) < 20:
                print(f"{band:8s} {'(n<20, skip)':>14s}")
                continue
            m_has, se_has, n_has = mean_se(has)
            m_no, se_no, n_no = mean_se(no)
            delta = m_has - m_no
            combined_se = math.sqrt(se_has**2 + se_no**2)
            sig = "***" if combined_se > 0 and abs(delta) >= 2 * combined_se else ""
            print(f"{band:8s} {m_has:14.1f} {m_no:14.1f} {delta:14.1f} {n_has:7d} {n_no:7d} {sig}")
        print()

    # signed-err direction check: does removing/lacking the piece type
    # correlate with over- or under-rating (not just |err| magnitude)?
    print("--- signed err (mean coda_white - lc0_white; +ve = Coda overrates white side), bishop only ---")
    print(f"{'band':8s} {'has bishop':>14s} {'no bishop':>14s} {'n_has':>7s} {'n_no':>7s}")
    for lo, hi in BANDS:
        band = f"{lo}-{hi}"
        band_rows = [r for r in rows if r["band"] == band]
        has = [r["err"] for r in band_rows if r["bishop"] >= 1]
        no = [r["err"] for r in band_rows if r["bishop"] == 0]
        if len(has) < 20 or len(no) < 20:
            continue
        m_has, _, n_has = mean_se(has)
        m_no, _, n_no = mean_se(no)
        print(f"{band:8s} {m_has:14.1f} {m_no:14.1f} {n_has:7d} {n_no:7d}")


if __name__ == "__main__":
    main()
