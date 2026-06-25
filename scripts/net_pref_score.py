#!/usr/bin/env python3
"""
net_pref_score.py — mean net_pref over any am/bm EPD, per net, with N + SE and
pairwise deltas. net_pref(pos) = net's own eval (after am) - (after bm), mover
POV; >0 means the net over-credits the avoid-move (the sortie). Lower = better.
No Stockfish — pure static-eval comparison of nets on a fixed position set.

Usage: net_pref_score.py --epd corpus.epd net1.nnue net2.nnue ...
"""
import argparse, subprocess, re, statistics, sys, chess

NN_RE = re.compile(r"NNUE evaluation\s+([+-][0-9.]+)")

def ceval(coda, fen, net):
    o = subprocess.run([coda, "-n", net], input=f"position fen {fen}\neval\nquit\n",
                       capture_output=True, text=True, timeout=30).stdout
    m = NN_RE.findall(o)
    return float(m[0]) * 100 if m else None

def after(fen, san):
    b = chess.Board(fen)
    mv = next((m for m in b.legal_moves if b.san(m).rstrip('+#') == san.rstrip('+#')), None)
    if mv is None: return None, None
    t = b.turn; b.push(mv); return b.fen(), t

def npref(coda, fen, am, bm, net):
    fa, mv = after(fen, am); fb, _ = after(fen, bm)
    if fa is None or fb is None: return None
    va, vb = ceval(coda, fa, net), ceval(coda, fb, net)
    if va is None or vb is None: return None
    s = 1 if mv == chess.WHITE else -1
    return s * va - s * vb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epd', required=True)
    ap.add_argument('--coda', default='./coda')
    ap.add_argument('nets', nargs='+')
    args = ap.parse_args()

    rows = []
    for line in open(args.epd):
        line = line.strip()
        if not line or line.startswith('#'): continue
        fen = line.split(' bm ')[0].strip()
        if len(fen.split()) == 4: fen += " 0 1"
        bm = re.search(r' bm (\S+?);', line); am = re.search(r' am (\S+?);', line)
        if not (bm and am): continue
        rows.append((fen, am.group(1), bm.group(1)))
    print(f"corpus: {len(rows)} positions  ({args.epd})\n")

    results = {}
    for net in args.nets:
        vals = [npref(args.coda, f, a, b, net) for f, a, b in rows]
        vals = [v for v in vals if v is not None]
        mean = statistics.mean(vals); sd = statistics.pstdev(vals)
        se = sd / len(vals) ** 0.5
        results[net] = vals
        print(f"  {net:<34} n={len(vals):3d}  mean net_pref = {mean:+6.0f}cp  "
              f"SD={sd:4.0f}  SE={se:4.0f}")
    if len(args.nets) == 2:
        a, b = args.nets
        paired = [x - y for x, y in zip(results[a], results[b])]
        m = statistics.mean(paired); se = statistics.pstdev(paired) / len(paired) ** 0.5
        print(f"\n  paired delta ({a} - {b}): {m:+.0f}cp  SE={se:.0f}  "
              f"({'significant' if abs(m) > 2*se else 'within noise'})")

if __name__ == '__main__':
    main()
