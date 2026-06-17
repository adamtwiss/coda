#!/usr/bin/env python3
"""
blindspot_eval.py — one-command diagnostic for the bishop/rook-sortie eval
blindspot. Run a candidate net (e.g. a sortie-fine-tuned net) and report whether
the eval moved in the right direction on the overrate corpus, plus the general
eval-quality cost.

Metrics (all vs the prod baseline / SF):
  1. overrate.epd pass rate (coda epd) — did move choice improve (26.3% baseline)?
  2. sortie over-preference vs SF — for each `bm`/`am` pair, the mover-POV cp by
     which the net prefers the bad sortie (am) over the correct move (bm), minus
     SF's. Mean + per-position; "flips" = sorties the net now ranks below the
     correct move (the win condition). This is the direct "did eval move right".
  3. (optional, --binpack) general eval quality: Spearman of the net's static
     eval vs the LC0 oracle on quiet positions — the STRENGTH COST of the
     fine-tune (should stay near baseline ~0.85; a big drop = over-forgetting).

Usage:
  python3 scripts/blindspot_eval.py --net nets/candidate.nnue [--sf <sf>] \
      [--tc 800] [--binpack /training/sf/<file>.binpack]
"""
import argparse, subprocess, re, statistics, sys
try:
    import chess
except ImportError:
    sys.exit("needs python-chess (pip install chess)")

CODA = "./coda"
EPD = "testdata/overrate.epd"

def coda_eval(fen, net):
    o = subprocess.run([CODA, "-n", net], input=f"position fen {fen}\neval\nquit\n",
                       capture_output=True, text=True, timeout=20).stdout
    m = re.findall(r"NNUE evaluation\s+([+-][0-9.]+)", o)
    return float(m[0]) * 100 if m else None

def sf_eval(fen, sf):
    o = subprocess.run([sf], input=f"position fen {fen}\neval\nquit\n",
                       capture_output=True, text=True, timeout=20).stdout
    m = re.search(r"Final evaluation\s+([+-][0-9.]+)", o)
    return float(m.group(1)) * 100 if m else None

def mover_pref(fen, am, bm, evf):
    """mover-POV cp by which engine prefers am (sortie) over bm (correct)."""
    def after(san):
        b = chess.Board(fen)
        mv = next((m for m in b.legal_moves if b.san(m).rstrip('+#') == san.rstrip('+#')), None)
        if mv is None: return None, None
        mover = b.turn; b.push(mv); return b.fen(), mover
    fa, mv = after(am); fb, _ = after(bm)
    if fa is None or fb is None: return None
    va, vb = evf(fa), evf(fb)
    if va is None or vb is None: return None
    s = 1 if mv == chess.WHITE else -1
    return s * va - s * vb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--sf", default="/home/adam/chess/engines/Stockfish/src/stockfish")
    ap.add_argument("--tc", type=int, default=800, help="ms/pos for the epd suite")
    ap.add_argument("--binpack", default=None, help="optional: general eval-quality Spearman")
    args = ap.parse_args()

    print(f"=== blindspot diagnostic: {args.net} ===\n")

    # 1. overrate.epd pass rate
    r = subprocess.run([CODA, "epd", EPD, "-n", args.net, "-t", str(args.tc)],
                       capture_output=True, text=True).stdout
    m = re.search(r"Results:\s*(\d+)/(\d+)\s*passed\s*\(([0-9.]+)%\)", r)
    print(f"1. overrate.epd: {m.group(0) if m else '??'}  (baseline prod 26.3%)\n")

    # 2. sortie over-preference vs SF
    print("2. sortie over-preference (mover-POV cp, >0 = net over-prefers the bad move):")
    rows = []
    for line in open(EPD):
        line = line.strip()
        if not line or line.startswith('#'): continue
        fen = line.split(' bm ')[0].strip()
        if len(fen.split()) == 4: fen += " 0 1"
        bm = re.search(r' bm (\S+?);', line); am = re.search(r' am (\S+?);', line)
        if not (bm and am): continue
        bm, am = bm.group(1), am.group(1)
        cp = mover_pref(fen, am, bm, lambda f: coda_eval(f, args.net))
        sp = mover_pref(fen, am, bm, lambda f: sf_eval(f, args.sf))
        if cp is None or sp is None: continue
        rows.append((cp - sp, cp, sp, am, bm))
    rows.sort(reverse=True)
    flips = sum(1 for d, cp, sp, am, bm in rows if cp < 0)  # net now prefers the correct move
    fb = [r for r in rows if r[3] in ('Bxf4', 'Bf3', 'Bd6', 'Bd2', 'Bd4+', 'Bg5')]
    for d, cp, sp, am, bm in rows:
        tag = "  <- net prefers CORRECT move now" if cp < 0 else ""
        print(f"   {am:<5} vs {bm:<5}  net_pref={cp:+5.0f}  sf={sp:+5.0f}  delta={d:+5.0f}{tag}")
    print(f"\n   mean delta(net-SF)={statistics.mean(d for d,_,_,_,_ in rows):+.0f}cp  "
          f"sorties-now-correct={flips}/{len(rows)}")
    if fb:
        print(f"   forward-bishop subset (n={len(fb)}): mean net_pref="
              f"{statistics.mean(r[1] for r in fb):+.0f}cp  (prod ~+457, SF ~+65)\n")

    # 3. general eval-quality (strength cost)
    if args.binpack:
        print("3. general eval quality (strength cost) — Spearman vs LC0 oracle:")
        csv = "/tmp/blindspot_evalq.csv"
        subprocess.run([CODA, "eval-dist", "-i", args.binpack, "-c", "15000",
                        "--quiet-only", "--csv", csv, "-n", args.net],
                       capture_output=True, text=True)
        import numpy as np
        coda, orc = [], []
        for ln in open(csv):
            q = ln.rstrip().split(',')
            if len(q) >= 4 and not q[0].startswith('fen'):
                try: coda.append(float(q[2])); orc.append(float(q[3]))
                except ValueError: pass
        if coda:
            rx = np.argsort(np.argsort(coda)); ry = np.argsort(np.argsort(orc))
            sp = float(np.corrcoef(rx, ry)[0, 1])
            print(f"   Spearman={sp:.4f}  (prod ~0.853; a big drop = over-forgetting)")

if __name__ == "__main__":
    main()
