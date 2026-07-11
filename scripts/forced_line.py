#!/usr/bin/env python3
"""Pruning-blindness vs eval-blindness discriminator (Adam, 2026-07-11).

For each sampled Coda-blindspot position (episode start with cached truth
V* from SF18@4M):
  1. SF18 8T@500k gives the truth PV (first 6 plies).
  2. Coda T=1 baseline: eval@150k -> d_pre = |eval - V*|.
  3. Walk Coda down the truth PV (search 100k at each of 6 nodes, TT warm).
  4. Return to root: re-eval@150k -> d_post.
Classification per position (only those with d_pre > 50):
  SNAP    d_post <= 50       -> knowledge was in eval; search was hiding it
                                (pruning/ordering-owned)
  PARTIAL d_post in (50, d_pre-25]
  STUBBORN otherwise         -> eval-owned (can't value it even when shown)
"""
import subprocess, csv, sys, time, statistics as st

S = "/tmp/claude-1001/-home-adam-code-coda/6a3a1099-ab50-4682-8fea-48bc7890ac15/scratchpad"
SF = ("/home/adam/chess/engines/Stockfish/src/stockfish", "/home/adam/chess/engines/Stockfish/src")
CODA = ("/home/adam/code/coda/coda", "/home/adam/code/coda")
N_SAMPLE = 300
WALK_PLIES = 6

class Eng:
    def __init__(self, path, cwd, threads, hash_mb):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, cwd=cwd)
        self.send("uci"); self.wait("uciok")
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_mb}")
        if 'coda' in path: self.send("setoption name OwnBook value false")
        self.ready()
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def wait(self, tok):
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("engine died")
            out.append(l)
            if l.startswith(tok): return out
    def ready(self): self.send("isready"); self.wait("readyok")
    def go(self, poscmd, nodes):
        self.send(poscmd); self.send(f"go nodes {nodes}")
        lines = self.wait("bestmove")
        cp, pv = None, []
        for l in reversed(lines[:-1]):
            t = l.split()
            if "score" in t and " pv " in l:
                i = t.index("score")
                cp = int(t[i+2]) if t[i+1] == "cp" else (10000 if int(t[i+2]) > 0 else -10000)
                pv = t[t.index("pv")+1:]
                break
        return cp, pv
    def newgame(self): self.send("ucinewgame"); self.ready()
    def quit(self):
        try: self.send("quit"); self.p.wait(timeout=5)
        except Exception: self.p.kill()

truth = {r['fen']: int(r['vstar']) for r in
         csv.DictReader(open(f"{S}/truth_cache.tsv"), delimiter='\t')}
fens = [f for f in truth if abs(truth[f]) <= 800]
import random
random.Random(7).shuffle(fens)
fens = fens[:N_SAMPLE]
print(f"{len(fens)} positions", flush=True)

sf = Eng(*SF, 8, 1024)
coda = Eng(*CODA, 1, 256)
res = []
t0 = time.time()
for i, fen in enumerate(fens):
    vstar = truth[fen]
    sf.newgame()
    _, pv = sf.go(f"position fen {fen}", 500_000)
    if len(pv) < 2: continue
    line = pv[:WALK_PLIES]
    coda.newgame()
    e0, _ = coda.go(f"position fen {fen}", 150_000)
    if e0 is None: continue
    # walk the truth line with TT retained
    for k in range(1, len(line) + 1):
        coda.go(f"position fen {fen} moves {' '.join(line[:k])}", 100_000)
    e1, _ = coda.go(f"position fen {fen}", 150_000)
    if e1 is None: continue
    res.append((abs(e0 - vstar), abs(e1 - vstar)))
    if (i+1) % 25 == 0: print(f"{i+1}/{len(fens)} {time.time()-t0:.0f}s", flush=True)
sf.quit(); coda.quit()

blind = [(a, b) for a, b in res if a > 50]
snap = sum(1 for a, b in blind if b <= 50)
part = sum(1 for a, b in blind if 50 < b <= a - 25)
stub = len(blind) - snap - part
print(f"\nn={len(res)} measured; blind at 150k (d_pre>50): {len(blind)}")
print(f"  SNAP     (search was hiding it): {snap} ({100*snap/max(1,len(blind)):.0f}%)")
print(f"  PARTIAL                        : {part} ({100*part/max(1,len(blind)):.0f}%)")
print(f"  STUBBORN (eval cannot see it)  : {stub} ({100*stub/max(1,len(blind)):.0f}%)")
print(f"  mean d_pre {st.mean(a for a,_ in blind):.0f} -> mean d_post {st.mean(b for _,b in blind):.0f}")
with open(f"{S}/forced_line_results.tsv", 'w') as f:
    f.write("d_pre\td_post\n")
    for a, b in res: f.write(f"{a}\t{b}\n")
