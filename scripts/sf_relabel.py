#!/usr/bin/env python3
"""
sf_relabel.py — re-score harvested over-valuation candidates with Stockfish at a
FIXED depth, producing consistent WDL-calibrated cp labels (SF's normalized score
matches the T80/LC0 scale; see github.com/official-stockfish/WDL_model). The
in-game gauntlet evals are at wildly varying depths/engines and useless as labels;
this gives one engine, one depth, STM-POV cp — the corrective training target.

Input  TSV: fen<TAB>coda_ev<TAB>opp_ev<TAB>disagree  (from the gauntlet harvest)
Output TSV: fen<TAB>coda_ev<TAB>sf_cp   (sf_cp = STM-POV, WDL-calibrated)
-> feed to `coda import-tsv -i out.tsv --fen-col 0 --score-col 2`.

Incremental write (crash-safe) + parallel across workers. Skips |coda_ev|>max_coda
(mate-ish / not the subtle blindspot — Bullet would filter them anyway).

Usage:
  python3 scripts/sf_relabel.py --input /tmp/overrate_candidates.tsv \
      --output /tmp/overrate_labeled.tsv --depth 18 --workers 16 \
      --sf /home/adam/chess/engines/Stockfish/src/stockfish --max-coda 10 --min-disagree 0.75
"""
import argparse, chess, chess.engine, multiprocessing as mp, os, sys

_eng = None
def _init(sf):
    global _eng
    _eng = chess.engine.SimpleEngine.popen_uci(sf)

def _score(args):
    fen, coda_ev, depth = args
    try:
        b = chess.Board(fen)
        info = _eng.analyse(b, chess.engine.Limit(depth=depth))
        cp = info['score'].pov(b.turn).score(mate_score=30000)  # STM-POV, WDL-cal cp
        return f"{fen}\t{coda_ev}\t{cp}"
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--depth', type=int, default=18)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--sf', default='/home/adam/chess/engines/Stockfish/src/stockfish')
    ap.add_argument('--max-coda', type=float, default=10.0, help='skip |coda_ev| above (mate-ish)')
    ap.add_argument('--min-disagree', type=float, default=0.75)
    a = ap.parse_args()

    tasks = []
    with open(a.input) as f:
        next(f)  # header
        for line in f:
            c = line.rstrip('\n').split('\t')
            if len(c) < 4:
                continue
            try:
                cev, dis = float(c[1]), float(c[3])
            except ValueError:
                continue
            if abs(cev) > a.max_coda or dis < a.min_disagree:
                continue
            tasks.append((c[0], c[1], a.depth))
    print(f"relabeling {len(tasks)} candidates @ depth {a.depth} on {a.workers} workers", flush=True)

    done = 0
    # resume: skip fens already in output
    seen = set()
    if os.path.exists(a.output):
        for line in open(a.output):
            seen.add(line.split('\t', 1)[0])
        tasks = [t for t in tasks if t[0] not in seen]
        print(f"  resuming: {len(seen)} already done, {len(tasks)} remaining", flush=True)

    with open(a.output, 'a', buffering=1) as out, \
         mp.Pool(a.workers, initializer=_init, initargs=(a.sf,)) as pool:
        for res in pool.imap_unordered(_score, tasks, chunksize=64):
            if res:
                out.write(res + '\n')
            done += 1
            if done % 5000 == 0:
                print(f"  ...{done}/{len(tasks)}", flush=True)
    print(f"done: {len(seen)+done} labeled -> {a.output}", flush=True)

if __name__ == '__main__':
    main()
