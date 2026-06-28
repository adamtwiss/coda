#!/usr/bin/env python3
"""
t80_sf_disagree.py — the missing measurement for the eval-blindspot research.

Takes the CSV from `coda eval-dist --csv` (columns: fen, white_result,
coda_eval_white_cp, lc0_score_white_cp), adds an SF *deep-search* eval per FEN
(the trusted third party — not SF static, which shares the fortress/50mr blind
spot), and answers two questions:

  1. What PROPORTION of random T80 positions does Coda disagree with SF on
     materially (|coda - sf| >= --thresh cp, white-POV)?
  2. When we disagree, where does the dataset's own LC0 label sit?
       - "LC0 sides with SF"   (|lc0-sf| < |lc0-coda|)  -> the corrective signal
         is ALREADY in the data; the position is just rare. Cheap fix =
         mine + over-sample (`coda import-tsv --repeat N`). No new data needed.
       - "LC0 sides with Coda" (|lc0-coda| < |lc0-sf|)  -> the LABEL is also
         part of the blind spot (T80 self-play overrates it too). Need the
         SF-relabeled Coda-vs-SF set (`sf_relabel.py` -> `import-tsv`).

This split tells us whether re-weighting existing T80 fixes the blind spot or
whether the 300M Coda-vs-SF route is actually required.

Usage:
  python3 scripts/t80_sf_disagree.py /tmp/t80_evalq.csv \
      --sf /home/adam/chess/engines/Stockfish/src/stockfish \
      --depth 16 --workers 14 --sample 4000 --thresh 150 --out /tmp/t80_disagree.tsv
"""
import argparse
import csv
import multiprocessing as mp
import sys

try:
    import chess
    import chess.engine
except ImportError:
    sys.exit("needs python-chess (pip install chess)")

_eng = None
_depth = None


def _init(sf, depth):
    global _eng, _depth
    _eng = chess.engine.SimpleEngine.popen_uci(sf)
    _depth = depth


def _sf_white_cp(row):
    """Return (fen, white_result, coda, lc0, sf_white_cp) or None on failure."""
    fen, wres, coda, lc0 = row
    try:
        board = chess.Board(fen)
        info = _eng.analyse(board, chess.engine.Limit(depth=_depth))
        sc = info["score"].white()
        if sc.is_mate():
            # mate -> a large signed cp on the white-POV scale (cap for the
            # |SF|-balanced analysis; mates are not the subtle blind spot).
            cp = 30000 if sc.mate() > 0 else -30000
        else:
            cp = sc.score()
        return (fen, wres, coda, lc0, cp)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="output of `coda eval-dist --csv`")
    ap.add_argument("--sf", default="/home/adam/chess/engines/Stockfish/src/stockfish")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--sample", type=int, default=4000,
                    help="random subsample of the CSV to SF-score (0 = all)")
    ap.add_argument("--thresh", type=int, default=150,
                    help="material-disagreement threshold, cp (white-POV)")
    ap.add_argument("--sf-cap", type=int, default=600,
                    help="restrict the headline split to |SF| <= cap (balanced "
                         "positions; excludes genuinely won/lost)")
    ap.add_argument("--out", default="/tmp/t80_disagree.tsv")
    a = ap.parse_args()

    rows = []
    with open(a.csv) as f:
        r = csv.reader(f)
        next(r, None)  # header
        for line in r:
            if len(line) < 4:
                continue
            try:
                rows.append((line[0], float(line[1]), int(line[2]), int(line[3])))
            except ValueError:
                continue

    # Deterministic subsample (stride) — no Math.random, reproducible.
    if a.sample and a.sample < len(rows):
        stride = len(rows) / a.sample
        rows = [rows[int(i * stride)] for i in range(a.sample)]
    print(f"SF-scoring {len(rows)} positions at depth {a.depth} "
          f"({a.workers} workers)...", file=sys.stderr)

    scored = []
    with mp.Pool(a.workers, initializer=_init, initargs=(a.sf, a.depth)) as pool:
        for i, res in enumerate(pool.imap_unordered(_sf_white_cp, rows, chunksize=8)):
            if res:
                scored.append(res)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(rows)}", file=sys.stderr)

    with open(a.out, "w") as out:
        out.write("fen\twhite_result\tcoda\tlc0\tsf\n")
        for fen, wres, coda, lc0, sf in scored:
            out.write(f"{fen}\t{wres}\t{coda}\t{lc0}\t{sf}\n")

    # --- Analysis ---
    n = len(scored)
    # Material disagreement with SF (white-POV), within the balanced band.
    band = [(fen, wres, coda, lc0, sf) for (fen, wres, coda, lc0, sf) in scored
            if abs(sf) <= a.sf_cap]
    disagree = [t for t in band if abs(t[2] - t[4]) >= a.thresh]
    coda_over = [t for t in disagree if t[2] > t[4]]   # Coda > SF = overrate
    coda_under = [t for t in disagree if t[2] < t[4]]

    def side(t):
        _, _, coda, lc0, sf = t
        return "SF" if abs(lc0 - sf) < abs(lc0 - coda) else "CODA"

    lc0_with_sf = [t for t in disagree if side(t) == "SF"]
    lc0_with_coda = [t for t in disagree if side(t) == "CODA"]
    # Same split restricted to Coda-OVERRATE disagreements (the blind spot dir).
    over_with_sf = [t for t in coda_over if side(t) == "SF"]

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0

    print(f"\n=== T80 vs SF (depth {a.depth}) — {n} positions scored ===")
    print(f"Balanced band (|SF|<={a.sf_cap}): {len(band)} positions")
    print(f"\nMaterial disagreement (|coda-sf|>={a.thresh}cp): "
          f"{len(disagree)}/{len(band)} = {100*len(disagree)/max(1,len(band)):.1f}%")
    print(f"  Coda OVERrates  (coda>sf): {len(coda_over)} "
          f"(median coda-sf = {med([t[2]-t[4] for t in coda_over])}cp)")
    print(f"  Coda UNDERrates (coda<sf): {len(coda_under)} "
          f"(median coda-sf = {med([t[2]-t[4] for t in coda_under])}cp)")
    print(f"\nWhen we disagree, where is the LC0 label?")
    print(f"  LC0 sides with SF:   {len(lc0_with_sf)}/{len(disagree)} = "
          f"{100*len(lc0_with_sf)/max(1,len(disagree)):.1f}%  "
          f"(corrective signal ALREADY in T80 -> cheap over-sample fix)")
    print(f"  LC0 sides with Coda: {len(lc0_with_coda)}/{len(disagree)} = "
          f"{100*len(lc0_with_coda)/max(1,len(disagree)):.1f}%  "
          f"(label also wrong -> need SF-relabeled data)")
    print(f"\nWithin Coda-OVERRATE disagreements specifically:")
    print(f"  LC0 sides with SF: {len(over_with_sf)}/{len(coda_over)} = "
          f"{100*len(over_with_sf)/max(1,len(coda_over)):.1f}%")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
