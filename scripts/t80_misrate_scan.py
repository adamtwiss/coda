#!/usr/bin/env python3
"""
t80_misrate_scan.py — find positions our static eval MISRATES vs the LC0 oracle.

Reframes the earlier "overrate" scan. We emit each position from the
side-to-move POV (and the data carries both colours), so the *signed* bias
(coda - lc0) is symmetric by construction — a near-zero net bias is expected,
not informative. The thing that actually matters is the MAGNITUDE of the
error, |coda_static - lc0|, on QUIET positions: places the model has learned
to value wrongly relative to the deep-MCTS ground truth, where it's not a
tactic the search would resolve.

Coda and SF train on IDENTICAL T80/LC0 data, so the binpack LC0 score is a
fair shared oracle. This script (stage 1, SF-free, billion-scalable) measures
the error-magnitude distribution and emits the high-|error| QUIET candidates
carrying their LC0 label, ready for over-sampled fine-tuning via
`coda import-tsv --fen-col 0 --score-col 1 --repeat N`.

The sharpest subset — "Coda AND SF static agree with each other but both miss
LC0" (a shared, learned eval error, provably not search-hidden tactics) — is
isolated in stage 2 by joining an SF *static* eval column (sf_static_eval.py)
onto these candidates.

Input: CSV from `coda eval-dist --csv`
       (fen, white_result, coda_eval_white_cp, lc0_score_white_cp).

Usage:
  python3 scripts/t80_misrate_scan.py /tmp/t80_evalq_big.csv
  python3 scripts/t80_misrate_scan.py /tmp/t80_evalq_big.csv \
      --thresh 200 --cap 600 --emit-tsv /tmp/misrate_cands.tsv [--max-men 7]
"""
import argparse
import csv
import statistics as st
import sys

try:
    import chess
except ImportError:
    sys.exit("needs python-chess (pip install chess)")


def load(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 4:
                continue
            try:
                rows.append((row[0], float(row[1]), int(row[2]), int(row[3])))
            except ValueError:
                continue
    return rows


def pctile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def report(rows, label, thresh, cap):
    band = [t for t in rows if abs(t[3]) <= cap]
    if not band:
        print(f"\n### {label}: (empty)")
        return
    err = [t[2] - t[3] for t in band]            # signed, white-POV
    aerr = [abs(e) for e in err]
    hi = [t for t in band if abs(t[2] - t[3]) >= thresh]
    over = sum(1 for t in hi if t[2] - t[3] > 0)  # demonstrate symmetry of SIGN
    under = len(hi) - over
    print(f"\n### {label}: band(|LC0|<={cap}) = {len(band)}")
    print(f"  |error| mean={st.mean(aerr):.0f}cp  median={st.median(aerr):.0f}  "
          f"p90={pctile(aerr,90)}  p95={pctile(aerr,95)}  p99={pctile(aerr,99)}")
    print(f"  signed net bias (coda-lc0): {st.mean(err):+.1f}cp  "
          f"(≈0 expected — both-POV symmetry, NOT the signal)")
    print(f"  MISRATE |error|>={thresh}cp: {len(hi):>6} = "
          f"{100*len(hi)/len(band):5.1f}%   "
          f"(of which over={over} / under={under} — confirms sign is symmetric)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="output of `coda eval-dist --csv`")
    ap.add_argument("--thresh", type=int, default=200,
                    help="|coda-lc0| misrate threshold in cp")
    ap.add_argument("--cap", type=int, default=600,
                    help="restrict to |LC0|<=cap (balanced, non-decided)")
    ap.add_argument("--emit-tsv", default=None,
                    help="write high-|error| candidates as fen<TAB>lc0_stm_cp "
                         "(the corrective LC0 label, already in the data)")
    ap.add_argument("--max-men", type=int, default=0,
                    help="for --emit-tsv: only emit <= this many men "
                         "(7 = TB-exact clean-oracle band; 0 = all)")
    a = ap.parse_args()

    rows = load(a.csv)
    print(f"Loaded {len(rows)} QUIET positions from {a.csv}")
    print("(signed bias is symmetric by construction — the metric is |error|.)")

    men = {t[0]: chess.popcount(chess.Board(t[0]).occupied) for t in rows}
    report(rows, "ALL quiet T80", a.thresh, a.cap)
    report([t for t in rows if men[t[0]] <= 12], "ENDGAME <=12 men", a.thresh, a.cap)
    report([t for t in rows if men[t[0]] <= 7],
           "TB-band <=7 men (LC0 = tablebase-exact)", a.thresh, a.cap)

    if a.emit_tsv:
        n = 0
        with open(a.emit_tsv, "w") as out:
            out.write("fen\tlc0_stm_cp\n")
            for fen, _wres, coda, lc0 in rows:
                if abs(lc0) > a.cap:
                    continue
                if abs(coda - lc0) < a.thresh:          # high |error| only
                    continue
                if a.max_men and men[fen] > a.max_men:
                    continue
                stm_white = chess.Board(fen).turn == chess.WHITE
                lc0_stm = lc0 if stm_white else -lc0
                out.write(f"{fen}\t{lc0_stm}\n")
                n += 1
        print(f"\nWrote {n} high-|error| quiet candidates -> {a.emit_tsv}")
        print(f"  stage 2 (isolate learned-eval errors, not tactics):")
        print(f"    python3 scripts/sf_static_eval.py {a.emit_tsv} "
              f"--out /tmp/misrate_sf.tsv   # adds SF static eval")
        print(f"  then keep rows where |coda-sf| is SMALL but |coda-lc0| is LARGE.")


if __name__ == "__main__":
    main()
