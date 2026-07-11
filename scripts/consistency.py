#!/usr/bin/env python3
"""Eval<->search consistency measurement (search input metric #1).

For each engine (coda, sf17, sf18) on two position sets (neutral controls,
blindspot episode-starts): static eval (UCI `eval`) + search value at
depths 4..14. Metrics:
  consistency(d) = |static - v(d)|      (search-friendliness of the eval)
  self_conv(d)   = |v(d) - v(14)|       (within-search convergence)
All values side-to-move cp, T=1, Hash=256, ucinewgame per position.
Output: consistency_results.tsv (engine, set, fen, static, v4..v14).
"""
import subprocess, csv, sys, time, re

S = "/tmp/claude-1001/-home-adam-code-coda/6a3a1099-ab50-4682-8fea-48bc7890ac15/scratchpad"
ENGINES = {
    'coda': ("/home/adam/code/coda/coda", "/home/adam/code/coda"),
    'sf17': ("/home/adam/chess/engines/Stockfish-17.1/src/stockfish", "/home/adam/chess/engines/Stockfish-17.1/src"),
    'sf18': ("/home/adam/chess/engines/Stockfish/src/stockfish", "/home/adam/chess/engines/Stockfish/src"),
}
DEPTHS = [4, 6, 8, 10, 12, 14]
N_PER_SET = 300
SF_EVAL = re.compile(r'Final evaluation\s+([+-]?\d+\.\d+)')

class Eng:
    def __init__(self, path, cwd):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, cwd=cwd)
        self.sf = 'stockfish' in path
        self.send("uci"); self.wait("uciok")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 256")
        if not self.sf: self.send("setoption name OwnBook value false")
        self.ready()
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def wait(self, tok):
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("dead")
            out.append(l)
            if l.startswith(tok): return out
    def ready(self): self.send("isready"); self.wait("readyok")
    def static_eval(self, fen, white_to_move):
        self.send(f"position fen {fen}")
        self.send("eval")
        if self.sf:
            # SF prints a board + "Final evaluation ... (white side)"; sync via isready
            self.send("isready")
            txt = ''.join(self.wait("readyok"))
            m = SF_EVAL.search(txt)
            if not m: return None
            cp_white = int(float(m.group(1)) * 100)
            return cp_white if white_to_move else -cp_white
        else:
            self.send("isready")
            for l in self.wait("readyok"):
                if 'eval cp' in l:
                    return int(l.split('eval cp')[1].split()[0])
            return None
    def value_at_depth(self, fen, d):
        self.send(f"position fen {fen}")
        self.send(f"go depth {d}")
        cp = None
        for l in reversed(self.wait("bestmove")[:-1]):
            t = l.split()
            if "score" in t:
                i = t.index("score")
                cp = int(t[i+2]) if t[i+1] == "cp" else (10000 if int(t[i+2]) > 0 else -10000)
                break
        return cp
    def newgame(self): self.send("ucinewgame"); self.ready()
    def quit(self):
        try: self.send("quit"); self.p.wait(timeout=5)
        except Exception: self.p.kill()

def load(path, n):
    rows = list(csv.DictReader(open(path), delimiter='\t'))
    seen, out = set(), []
    for r in rows:
        k = ' '.join(r['fen'].split()[:4])
        if k in seen: continue
        seen.add(k); out.append(r['fen'])
        if len(out) >= n: break
    return out

sets = {
    'blind': load(f"{S}/horizon_ep_starts.tsv", N_PER_SET),
    'neutral': load(f"{S}/control_candidates.tsv", N_PER_SET),
}
out = open(f"{S}/consistency_results.tsv", 'w')
out.write("engine\tset\tfen\tstatic\t" + '\t'.join(f"v{d}" for d in DEPTHS) + "\n")
t0 = time.time()
for name, (path, cwd) in ENGINES.items():
    eng = Eng(path, cwd)
    for setname, fens in sets.items():
        for i, fen in enumerate(fens):
            eng.newgame()
            wtm = fen.split()[1] == 'w'
            st_ = eng.static_eval(fen, wtm)
            vals = [eng.value_at_depth(fen, d) for d in DEPTHS]
            out.write(f"{name}\t{setname}\t{fen}\t{st_}\t" + '\t'.join(map(str, vals)) + "\n")
        out.flush()
        print(f"{name}/{setname} done {time.time()-t0:.0f}s", flush=True)
    eng.quit()
print("MEASUREMENT DONE", flush=True)

# report
import statistics as stt
from collections import defaultdict
rows = list(csv.DictReader(open(f"{S}/consistency_results.tsv"), delimiter='\t'))
print(f"\n{'engine':6} {'set':8} {'|static-v14|':>12} " + ' '.join(f"{'|v'+str(d)+'-v14|':>10}" for d in DEPTHS[:-1]))
for name in ENGINES:
    for setname in sets:
        sel = [r for r in rows if r['engine'] == name and r['set'] == setname
               and r['static'] != 'None' and all(r[f'v{d}'] != 'None' for d in DEPTHS)
               and abs(int(r['v14'])) < 800]
        if not sel: continue
        sc = stt.mean(abs(int(r['static']) - int(r['v14'])) for r in sel)
        cells = [stt.mean(abs(int(r[f'v{d}']) - int(r['v14'])) for r in sel) for d in DEPTHS[:-1]]
        print(f"{name:6} {setname:8} {sc:>12.1f} " + ' '.join(f"{c:>10.1f}" for c in cells))
