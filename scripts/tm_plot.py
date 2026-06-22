#!/usr/bin/env python3
"""Lichess-style time graphs from cutechess PGNs.
Comment format: {<eval>/<depth> <spent>s}. Coda's clock-remaining is reconstructed
from base - cumsum(spent) + ply*inc. Two modes:
  single game  -> per-move spend bars (white up / black down) + clock-remaining lines
  --agg        -> mean Coda spend-per-move + mean Coda clock-left, across all games;
                  optionally overlay a 2nd PGN (before/after).
Usage:
  tm_plot.py PGN --engine coda --game 0   --out g.png
  tm_plot.py PGN --engine coda --agg       --out agg.png  [--label A]
  tm_plot.py PGN_A --engine coda --agg --overlay PGN_B --labels before,after --out cmp.png
"""
import re, sys, argparse, statistics, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_games(path):
    if not glob.glob(path): return []
    blocks = re.split(r'\n\n(?=\[Event)', open(path).read())
    games = []
    for blk in blocks:
        W = re.search(r'\[White "([^"]+)"\]', blk); B = re.search(r'\[Black "([^"]+)"\]', blk)
        R = re.search(r'\[Result "([^"]+)"\]', blk); T = re.search(r'\[Termination "([^"]+)"\]', blk)
        tc = re.search(r'\[TimeControl "([0-9.]+)\+([0-9.]+)"\]', blk)
        if not (W and B and R and tc): continue
        mt = blk[blk.rfind(']')+1:]
        # one spend per ply, in order
        sp = [float(x) for x in re.findall(r'\{[^}]*?([0-9]+\.[0-9]+)s\b[^}]*?\}', mt)]
        games.append(dict(white=W.group(1), black=B.group(1), result=R.group(1),
                          term=(T.group(1) if T else ""),
                          base=float(tc.group(1)), inc=float(tc.group(2)), spent=sp))
    return games

def side_series(g, engine):
    """Return (spent_list, clkleft_list) for the engine's moves in game g."""
    if g['white'].startswith(engine): col = 0
    elif g['black'].startswith(engine): col = 1
    else: return None, None
    sub = g['spent'][col::2]
    clk = g['base']; left = []
    for s in sub:
        clk = clk - s + g['inc']; left.append(max(clk, 0))
    return sub, left

def plot_single(g, engine, out):
    ws, wl = side_series(g, engine if g['white'].startswith(engine) else g['white'][:0] or g['white'])
    # plot BOTH sides' spend; engine bars up, opponent down
    w_sub = g['spent'][0::2]; b_sub = g['spent'][1::2]
    def clkof(sub):
        c=g['base']; o=[]
        for s in sub: c=c-s+g['inc']; o.append(max(c,0))
        return o
    w_clk = clkof(w_sub); b_clk = clkof(b_sub)
    eng_white = g['white'].startswith(engine)
    fig, ax = plt.subplots(figsize=(12,5))
    xw = range(1, len(w_sub)+1); xb = range(1, len(b_sub)+1)
    ax.bar(xw, w_sub, width=0.9, color='#4a90d9' if eng_white else '#bbbbbb',
           label=f"{g['white']} spend")
    ax.bar(xb, [-v for v in b_sub], width=0.9, color='#4a90d9' if not eng_white else '#bbbbbb',
           label=f"{g['black']} spend")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel("time spent / move (s)")
    ax.set_xlabel("move number")
    # secondary axis: clock remaining
    ax2 = ax.twinx()
    ax2.plot(xw, w_clk, color='#1f5fa8' if eng_white else '#888', lw=1.6, label=f"{g['white']} clk")
    ax2.plot(xb, b_clk, color='#1f5fa8' if not eng_white else '#888', lw=1.6, ls='--', label=f"{g['black']} clk")
    ax2.set_ylabel("clock remaining (s)")
    ax2.set_ylim(0, g['base']*1.05)
    peak = max(w_sub if eng_white else b_sub)
    engclk = w_clk if eng_white else b_clk
    ax.set_title(f"{engine}: TC {g['base']:.0f}+{g['inc']:.1f}  result {g['result']} {g['term']}  "
                 f"| {engine} peak spend {peak:.1f}s, min clk {min(engclk):.1f}s")
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return peak, min(engclk)

def aggregate(path, engine):
    games = parse_games(path)
    by_mv_spend = {}; by_mv_clk = {}; peaks=[]; minclks=[]; n=0; base=inc=None
    early_spend=[]  # total spend moves 6-20 (the overspend window)
    for g in games:
        sub, left = side_series(g, engine)
        if not sub: continue
        n+=1; base=g['base']; inc=g['inc']
        peaks.append(max(sub)); minclks.append(min(left))
        early_spend.append(sum(sub[5:20]))
        for i,(s,l) in enumerate(zip(sub,left), start=1):
            by_mv_spend.setdefault(i,[]).append(s)
            by_mv_clk.setdefault(i,[]).append(l)
    return dict(n=n, base=base, inc=inc, by_mv_spend=by_mv_spend, by_mv_clk=by_mv_clk,
                peaks=peaks, minclks=minclks, early_spend=early_spend)

def plot_agg(aggs, labels, out):
    fig, (a1,a2) = plt.subplots(2,1, figsize=(12,8), sharex=True)
    colors=['#d9534f','#4a90d9','#5cb85c']
    for agg,lab,c in zip(aggs,labels,colors):
        mvs = sorted(agg['by_mv_spend'])[:80]
        ms = [statistics.mean(agg['by_mv_spend'][m]) for m in mvs]
        mc = [statistics.mean(agg['by_mv_clk'][m]) for m in mvs]
        es = statistics.mean(agg['early_spend']) if agg['early_spend'] else 0
        mk = statistics.mean(agg['minclks']) if agg['minclks'] else 0
        a1.plot(mvs, ms, color=c, lw=1.8,
                label=f"{lab} (n={agg['n']}, mv6-20 spend μ={es:.1f}s, min-clk μ={mk:.1f}s)")
        a2.plot(mvs, mc, color=c, lw=1.8, label=lab)
    base = aggs[0]['base']; inc=aggs[0]['inc']
    a1.set_ylabel(f"mean {engine} spend/move (s)"); a1.set_title(
        f"{engine} TIME SHAPE  TC {base:.0f}+{inc:.1f}  (top=spend/move, bottom=clock left)")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.set_ylabel("mean clock left (s)"); a2.set_xlabel("move number")
    a2.set_ylim(0, base*1.05); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--engine", default="coda")
    ap.add_argument("--game", type=int, default=None)
    ap.add_argument("--agg", action="store_true")
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--label", default="A")
    ap.add_argument("--out", default="/tmp/tmgaunt/tm.png")
    a = ap.parse_args()
    engine = a.engine
    if a.game is not None:
        games = [g for g in parse_games(a.pgn) if g['white'].startswith(engine) or g['black'].startswith(engine)]
        g = games[a.game]
        pk,mn = plot_single(g, engine, a.out)
        print(f"game {a.game}: peak spend {pk:.1f}s  min clk {mn:.1f}s -> {a.out}")
    elif a.agg:
        aggs=[aggregate(a.pgn, engine)]; labels=[a.label]
        if a.overlay:
            aggs.append(aggregate(a.overlay, engine)); labels=a.labels.split(",")
        plot_agg(aggs, labels, a.out)
        for agg,lab in zip(aggs,labels):
            es=statistics.mean(agg['early_spend']) if agg['early_spend'] else 0
            mk=statistics.mean(agg['minclks']) if agg['minclks'] else 0
            pk=statistics.mean(agg['peaks']) if agg['peaks'] else 0
            print(f"{lab}: n={agg['n']} mv6-20 spend μ={es:.1f}s  peak/move μ={pk:.1f}s  min-clk μ={mk:.1f}s")
        print(f"-> {a.out}")
