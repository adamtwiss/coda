#!/usr/bin/env python3
"""
sf_static_eval.py — stage 2 of the misrate hunt: where is Coda's STATIC eval
materially worse than SF's STATIC eval, relative to the LC0 oracle?

Target (Adam, 2026-06-28): positions where
  (1) |coda_static - lc0| is large                         — Coda misrates, and
  (2) |coda_static - lc0|  >>  |sf_static - lc0|            — and SF (same
      training data, ALSO static/no-search) gets materially closer.
(2) rules out "only deep search can value this" (SF static would miss it too),
isolating CODA-SPECIFIC learned eval errors: bad patterns we picked up that SF
did not, despite identical T80/LC0 training data.

SF eval is its raw NNUE static eval (the `eval` command, white-POV), NOT a
search — using SF *search* would defeat the "not a tactic" logic.

Scale calibration: Coda's cp is trained to the LC0 scale by construction; SF's
internal cp scale differs. We fit a robust per-engine linear map eval->lc0 on
the whole sampled set (least squares) BEFORE comparing residuals, so a pure
scale offset can't masquerade as error.

Input:  CSV from `coda eval-dist --csv`
        (fen, white_result, coda_eval_white_cp, lc0_score_white_cp).
Output: joined TSV (fen, white_result, coda, lc0, sf_static) + the analysis,
        and (--emit-tsv) the coda-worse-than-SF correction set with LC0 labels.

Usage:
  python3 scripts/sf_static_eval.py /tmp/t80_evalq_big.csv \
      --sample 60000 --workers 12 \
      --thresh 200 --margin 120 \
      --out /tmp/misrate_joined.tsv --emit-tsv /tmp/coda_worse_cands.tsv
"""
import argparse
import csv
import multiprocessing as mp
import re
import subprocess
import sys

try:
    import chess
except ImportError:
    sys.exit("needs python-chess (pip install chess)")

_NNUE_RE = re.compile(r"NNUE evaluation\s+([+-]?\d+\.\d+)\s+\(white side\)")
_proc = None


def _init(sf):
    global _proc
    _proc = subprocess.Popen([sf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    _proc.stdin.write("uci\n")
    _proc.stdin.flush()
    for line in _proc.stdout:                 # drain to uciok
        if line.startswith("uciok"):
            break


def _sf_static_white_cp(row):
    """(fen, wres, coda, lc0, sf_white_cp) or None."""
    fen, wres, coda, lc0 = row
    try:
        _proc.stdin.write(f"position fen {fen}\neval\n")
        _proc.stdin.flush()
        val = None
        for line in _proc.stdout:
            m = _NNUE_RE.search(line)
            if m:
                val = round(float(m.group(1)) * 100)
            if line.startswith("Final evaluation"):
                break                          # eval block done
        if val is None:
            return None
        return (fen, wres, coda, lc0, val)
    except Exception:
        return None


def linfit(xs, ys):
    """least-squares y = a*x + b -> (a, b)."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if den == 0:
        return 1.0, 0.0
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sf", default="/home/adam/chess/engines/Stockfish/src/stockfish")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=60000,
                    help="uniform stride subsample to SF-eval (0 = all)")
    ap.add_argument("--cap", type=int, default=600,
                    help="restrict analysis to |LC0|<=cap (balanced)")
    ap.add_argument("--thresh", type=int, default=200,
                    help="|coda-lc0| misrate threshold (calibrated cp)")
    ap.add_argument("--margin", type=int, default=120,
                    help="how much worse than SF to count as coda-specific: "
                         "|coda-lc0| - |sf-lc0| >= margin")
    ap.add_argument("--out", default="/tmp/misrate_joined.tsv")
    ap.add_argument("--emit-tsv", default=None)
    a = ap.parse_args()

    rows = []
    with open(a.csv) as f:
        r = csv.reader(f)
        next(r, None)
        for line in r:
            if len(line) < 4:
                continue
            try:
                rows.append((line[0], float(line[1]), int(line[2]), int(line[3])))
            except ValueError:
                continue
    if a.sample and a.sample < len(rows):
        stride = len(rows) / a.sample
        rows = [rows[int(i * stride)] for i in range(a.sample)]
    print(f"SF-static-evaluating {len(rows)} positions ({a.workers} workers)...",
          file=sys.stderr)

    scored = []
    with mp.Pool(a.workers, initializer=_init, initargs=(a.sf,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_sf_static_white_cp, rows,
                                                    chunksize=16)):
            if res:
                scored.append(res)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(rows)}", file=sys.stderr)

    with open(a.out, "w") as out:
        out.write("fen\twhite_result\tcoda\tlc0\tsf_static\n")
        for fen, wres, coda, lc0, sf in scored:
            out.write(f"{fen}\t{wres}\t{coda}\t{lc0}\t{sf}\n")

    # --- calibrate each engine's static eval to the LC0 scale ---
    band = [t for t in scored if abs(t[3]) <= a.cap]
    lc0s = [t[3] for t in band]
    ca, cb = linfit([t[2] for t in band], lc0s)
    sa, sb = linfit([t[4] for t in band], lc0s)
    men = {t[0]: chess.popcount(chess.Board(t[0]).occupied) for t in band}

    def cerr(t):  # calibrated |coda - lc0|
        return abs((ca * t[2] + cb) - t[3])

    def serr(t):  # calibrated |sf - lc0|
        return abs((sa * t[4] + sb) - t[3])

    n = len(band)
    hi = [t for t in band if cerr(t) >= a.thresh]
    coda_worse = [t for t in hi if cerr(t) - serr(t) >= a.margin]
    sf_worse = [t for t in hi if serr(t) - cerr(t) >= a.margin]
    both = [t for t in hi if abs(cerr(t) - serr(t)) < a.margin]

    print(f"\n=== Coda STATIC vs SF STATIC vs LC0 oracle ({n} balanced, "
          f"|LC0|<={a.cap}) ===")
    print(f"calibration  coda: lc0≈{ca:.3f}*coda{cb:+.0f}   "
          f"sf: lc0≈{sa:.3f}*sf{sb:+.0f}")
    import statistics as st
    print(f"mean |error| vs LC0 (calibrated):  coda={st.mean([cerr(t) for t in band]):.0f}cp"
          f"   sf={st.mean([serr(t) for t in band]):.0f}cp")
    print(f"\nCoda MISRATE (|coda-lc0|>={a.thresh}cp): {len(hi)} = {100*len(hi)/n:.1f}%")
    print(f"  -> CODA materially worse than SF (Δ>={a.margin}): {len(coda_worse)} "
          f"= {100*len(coda_worse)/max(1,n):.1f}% of all  "
          f"({100*len(coda_worse)/max(1,len(hi)):.0f}% of misrates)  <-- TARGET")
    print(f"  -> SF materially worse than Coda:               {len(sf_worse)} "
          f"({100*len(sf_worse)/max(1,len(hi)):.0f}% of misrates)")
    print(f"  -> both similar (shared NNUE blindspot):         {len(both)} "
          f"({100*len(both)/max(1,len(hi)):.0f}% of misrates)")

    coda_worse.sort(key=lambda t: cerr(t) - serr(t), reverse=True)
    print(f"\nTop coda-specific misrates (coda err / sf err / men):")
    for t in coda_worse[:18]:
        print(f"  cErr={cerr(t):4.0f}  sErr={serr(t):4.0f}  men={men[t[0]]:2d}  "
              f"coda={t[2]:+5d} sf={t[4]:+5d} lc0={t[3]:+5d}  {t[0]}")

    if a.emit_tsv:
        with open(a.emit_tsv, "w") as out:
            out.write("fen\tlc0_stm_cp\n")
            for t in coda_worse:
                stm_white = chess.Board(t[0]).turn == chess.WHITE
                out.write(f"{t[0]}\t{t[3] if stm_white else -t[3]}\n")
        print(f"\nWrote {len(coda_worse)} coda-worse-than-SF candidates -> {a.emit_tsv}")
        print(f"  -> coda import-tsv -i {a.emit_tsv} --fen-col 0 --score-col 1 "
              f"--repeat N -o oversample.binpack")


if __name__ == "__main__":
    main()
