#!/usr/bin/env python3
"""
t80_overrate_scan.py — SF-free eval-overrate detector against the LC0 oracle.

Coda and SF train on IDENTICAL T80/LC0 data (same files, copied recipes), so the
binpack's LC0 score is a *fair, trusted shared oracle* — no home-field bias. That
means the eval blind spot is detectable without Stockfish at all: it is simply
the set of positions where Coda's static eval OVERRATES the LC0 label. This is
instant and scales to a billion positions (the SF-search leg, sf_relabel.py, is
only needed to spot-validate the label on >7-man positions where LC0 is an MCTS
estimate rather than tablebase-exact).

Input: the CSV from `coda eval-dist --csv`
       (columns: fen, white_result, coda_eval_white_cp, lc0_score_white_cp).

Reports the overrate/underrate asymmetry vs the LC0 oracle, split by piece count
(the `2tb7p` file is TB-EXACT for <=7 men — the clean-oracle band where the
signal is trustworthy). Optionally emits the overrate candidates as an
SF-free training-correction set carrying the existing LC0 label (STM-POV cp),
ready for `coda import-tsv --fen-col 0 --score-col 1 --repeat N`.

Usage:
  python3 scripts/t80_overrate_scan.py /tmp/t80_evalq_big.csv
  python3 scripts/t80_overrate_scan.py /tmp/t80_evalq_big.csv \
      --thresh 150 --max-men 7 --emit-tsv /tmp/overrate_oversample.tsv
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


def report(rows, label, thresh, cap):
    band = [t for t in rows if abs(t[3]) <= cap]
    if not band:
        print(f"\n### {label}: (empty)")
        return
    over = [t for t in band if t[2] - t[3] >= thresh]   # Coda > LC0 = overrate
    under = [t for t in band if t[3] - t[2] >= thresh]
    mover = st.median([t[2] - t[3] for t in over]) if over else 0
    ratio = (len(over) / len(under)) if under else float("inf")
    print(f"\n### {label}: band(|LC0|<={cap}) = {len(band)}")
    print(f"  Coda OVERrates LC0  >={thresh}cp: {len(over):>6} = "
          f"{100*len(over)/len(band):5.1f}%   median Δ={mover:+.0f}cp")
    print(f"  Coda UNDERrates LC0 >={thresh}cp: {len(under):>6} = "
          f"{100*len(under)/len(band):5.1f}%")
    print(f"  over/under asymmetry: {ratio:.2f}x   "
          f"net mean bias (coda-lc0): {st.mean([t[2]-t[3] for t in band]):+.1f}cp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="output of `coda eval-dist --csv`")
    ap.add_argument("--thresh", type=int, default=150,
                    help="overrate threshold in cp (white-POV)")
    ap.add_argument("--cap", type=int, default=600,
                    help="restrict to |LC0|<=cap (balanced positions)")
    ap.add_argument("--emit-tsv", default=None,
                    help="write overrate candidates as fen<TAB>lc0_stm_cp "
                         "(the corrective label, already in the data)")
    ap.add_argument("--max-men", type=int, default=0,
                    help="for --emit-tsv: only emit positions with <= this many "
                         "men (e.g. 7 = TB-exact clean-oracle band; 0 = all)")
    a = ap.parse_args()

    rows = load(a.csv)
    print(f"Loaded {len(rows)} positions from {a.csv}")

    men = {t[0]: chess.popcount(chess.Board(t[0]).occupied) for t in rows}
    report(rows, "ALL random T80", a.thresh, a.cap)
    report([t for t in rows if men[t[0]] <= 12], "ENDGAME <=12 men", a.thresh, a.cap)
    report([t for t in rows if men[t[0]] <= 7],
           "TB-band <=7 men (LC0 = tablebase-exact)", a.thresh, a.cap)

    if a.emit_tsv:
        n = 0
        with open(a.emit_tsv, "w") as out:
            out.write("fen\tlc0_stm_cp\n")  # import-tsv: --fen-col 0 --score-col 1
            for fen, _wres, coda, lc0 in rows:
                if abs(lc0) > a.cap:
                    continue
                if coda - lc0 < a.thresh:           # overrate candidates only
                    continue
                if a.max_men and men[fen] > a.max_men:
                    continue
                # LC0 is stored white-POV; import-tsv wants STM-POV cp.
                stm_white = chess.Board(fen).turn == chess.WHITE
                lc0_stm = lc0 if stm_white else -lc0
                out.write(f"{fen}\t{lc0_stm}\n")
                n += 1
        print(f"\nWrote {n} overrate candidates -> {a.emit_tsv}")
        print(f"  next: ./coda import-tsv -i {a.emit_tsv} -o oversample.binpack "
              f"--fen-col 0 --score-col 1 --repeat <N>")


if __name__ == "__main__":
    main()
