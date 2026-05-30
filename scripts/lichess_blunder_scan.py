#!/usr/bin/env python3
"""Lichess bug-mining: pull a bot's recent games, SF-arbitrate the losses/non-wins
in parallel, and classify each by the signature that points at the responsible
code area. Built from the 2026-05-30 regression hunt (found the TB drawn-root,
KQvK can't-mate, and no-inc-overspend bugs purely from lichess game analysis).

Per the bug-mining methodology (memory: feedback_lichess_bug_mining_priority):
deployment-only bugs (EGTB, clocks, ponder) are invisible to SPRT — lichess game
analysis is the only way to see them, and is currently the highest-Elo activity.

The KILLER signal is **spend@the-decisive-move**:
  - 0-200ms on a fat clock  -> TM/ponder fast-emit OR TB instant-emit bug
  - normal think but lost    -> eval/search blunder
combined with piece count (<=5 men = TB path; 6 = just-above-TB; more = search)
and result signature (won->draw repetition / won->loss flag / won->loss board).

Usage:
  python3 scripts/lichess_blunder_scan.py <botname> [--max 80] [--vs <opp>]
      [--depth 18] [--workers N] [--sf <path>] [--tb <path>] [--since-ms <epoch>]

Requires python-chess and a Stockfish binary. Read-only (network + SF); does not
touch the engine or repo. Prints a per-loss table; flags the likely bug class.
"""
import argparse, json, math, re, subprocess, sys, urllib.request
from multiprocessing import Pool

WIN_DROP_BLUNDER = 15.0  # win% points; = lichess 0.3 winningChances
DEFAULT_SF = None        # resolved from --sf or common paths

def winpct(cp):
    if cp is None: return None
    cp = max(-2000, min(2000, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)

def fetch_games(bot, mx, vs):
    url = f"https://lichess.org/api/games/user/{bot}?max={mx}&clocks=true&rated=true"
    if vs: url += f"&vs={vs}"
    req = urllib.request.Request(url, headers={"Accept": "application/x-ndjson"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]

class SF:
    def __init__(self, path, tb, depth):
        self.depth = depth
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._s("uci"); self._u("uciok")
        if tb: self._s(f"setoption name SyzygyPath value {tb}")
        self._s("setoption name Threads value 1"); self._s("isready"); self._u("readyok")
    def _s(self, x): self.p.stdin.write(x+"\n"); self.p.stdin.flush()
    def _u(self, tok):
        for _ in range(5000):
            l = self.p.stdout.readline()
            if not l or tok in l: return
    def ev(self, fen):
        self._s("ucinewgame"); self._s("position fen "+fen); self._s(f"go depth {self.depth}")
        last=None
        for _ in range(300000):
            l=self.p.stdout.readline()
            if not l: break
            if l.startswith("info "):
                m=re.search(r"score (cp|mate) (-?\d+)", l)
                if m: last=int(m.group(2)) if m.group(1)=="cp" else (30000 if int(m.group(2))>0 else -30000)
            elif l.startswith("bestmove"): return last
        return last

def analyze(args):
    import chess
    g, botset, sf_path, tb, depth = args
    sf = SF(sf_path, tb, depth)
    w=g['players']['white']['user']['name']; b=g['players']['black']['user']['name']
    cw = w.lower() in botset
    init=g['clock']['initial']*1000; inc=g['clock']['increment']*1000
    clk=[c*10 for c in g.get('clocks',[])]; moves=g['moves'].split()
    # per-ply bot spend
    prev_w=init; prev_b=init; spend={}
    for i,cs in enumerate(clk):
        if i%2==0: sp=prev_w+inc-cs; prev_w=cs; mine=cw
        else: sp=prev_b+inc-cs; prev_b=cs; mine=not cw
        if mine: spend[i]=sp
    board=chess.Board(); prev=None; worst=None; peak=-99999
    for i,san in enumerate(moves):
        is_bot=(board.turn==chess.WHITE)==cw
        try: board.push(board.parse_san(san))
        except Exception: break
        cp=sf.ev(board.fen());
        if cp is None: continue
        sw=(board.turn==chess.WHITE); botcp=(cp if sw else -cp) if cw else (-cp if sw else cp)
        peak=max(peak,botcp)
        if is_bot and prev is not None:
            wp=winpct(prev); wa=winpct(botcp)
            drop = (wp-wa) if (wp is not None and wa is not None) else 0
            men=chess.popcount(board.occupied)
            if worst is None or drop>worst['drop']:
                worst={'drop':drop,'ply':i,'san':san,'before':prev,'after':botcp,
                       'spend':spend.get(i,-1),'men':men}
        if is_bot: prev=botcp
    sf._s("quit")
    res=g.get('winner'); bot_res='WIN' if (res and (res=='white')==cw) else ('LOSS' if res else 'DRAW')
    return {'id':g['id'],'tc':f"{init//1000}+{inc//1000}",'col':'W' if cw else 'B',
            'result':bot_res,'status':g.get('status'),'peak':peak,'worst':worst}

def classify(r):
    wo=r['worst']
    if r['result']=='WIN' or wo is None: return ''
    sp=wo['spend']; men=wo['men']
    tags=[]
    if r['status']=='outoftime': tags.append('TIME-FORFEIT')
    if 0<=sp<200: tags.append('FAST-EMIT(<200ms)')
    if men<=5: tags.append('TB-RANGE(<=5)')
    elif men==6: tags.append('6-man')
    if r['peak']>=300 and r['result']!='WIN': tags.append('THREW-WIN')
    return ' '.join(tags) if tags else 'board-blunder'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("bot"); ap.add_argument("--max",type=int,default=80)
    ap.add_argument("--vs",default=None); ap.add_argument("--depth",type=int,default=18)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--sf",default=None); ap.add_argument("--tb",default=None)
    ap.add_argument("--losses-only",action="store_true",default=True)
    a=ap.parse_args()
    import os
    sf_path=a.sf
    if not sf_path:
        for c in ["/home/adam/chess/engines/Stockfish/src/stockfish",
                  os.path.expanduser("~/chess/engines/Stockfish/src/stockfish"),"stockfish"]:
            if os.path.exists(c) or c=="stockfish": sf_path=c; break
    tb=a.tb
    if not tb:
        for c in ["/tablebases", os.path.expanduser("~/chess/tablebases")]:
            if os.path.isdir(c): tb=c; break
    botset={a.bot.lower()}
    games=fetch_games(a.bot,a.max,a.vs)
    # only analyze non-wins (losses + draws) — that's where bugs live
    todo=[g for g in games if g.get('moves')]
    def is_nonwin(g):
        w=g['players']['white']['user']['name'].lower(); cw=w in botset
        res=g.get('winner'); return not (res and (res=='white')==cw)
    todo=[g for g in todo if is_nonwin(g)]
    print(f"{a.bot}: {len(games)} games pulled, analyzing {len(todo)} non-wins"
          f"{' vs '+a.vs if a.vs else ''} (SF d{a.depth}, {a.workers} workers, TB={tb})")
    with Pool(a.workers) as pool:
        res=pool.map(analyze,[(g,botset,sf_path,tb,a.depth) for g in todo])
    print(f"\n{'id':12}{'TC':>8}{'col':>4}{'res':>6}{'status':>11}{'peak':>7}"
          f"{'worstMove':>10}{'drop%':>6}{'spend':>8}{'men':>4}  CLASS")
    for r in sorted(res,key=lambda x:(x['result']!='LOSS', -(x['worst']['drop'] if x['worst'] else 0))):
        wo=r['worst']
        if wo is None:
            print(f"{r['id']:12}{r['tc']:>8}{r['col']:>4}{r['result']:>6}{r['status'] or '':>11}  (no data)"); continue
        print(f"{r['id']:12}{r['tc']:>8}{r['col']:>4}{r['result']:>6}{r['status'] or '':>11}"
              f"{r['peak']:>7}{wo['san']:>10}{wo['drop']:>6.0f}{int(wo['spend']):>7}ms{wo['men']:>4}  {classify(r)}")
    print("\nCLASS legend: TB-RANGE+FAST-EMIT=TB instant-emit bug; THREW-WIN+FAST-EMIT=ponder/TM fast blunder;")
    print("  TIME-FORFEIT=clock mgmt; 6-man=just-above-TB conversion; board-blunder=eval/search.")

if __name__=="__main__":
    main()
