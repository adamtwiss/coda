#!/usr/bin/env python3
"""Eval-convergence-at-budget measurement (search input metric, tier 2).

Phase 1 (--truth): SF-dev 8T/4M nodes per episode-start position -> cached
truth values V* (side-to-move perspective), truth_cache.tsv.
Phase 2 (--measure ENGINE): run ENGINE at T=1 over the positions at several
node budgets; record eval@N. Appends to conv_results.tsv.
Phase 3 (--report): per engine x budget: mean |eval - V*|, median, and
%converged (within 50cp / 100cp of V*). Positions with |V*| > 800 dropped
(decided; convergence there is uninteresting).
"""
import subprocess, sys, csv, os, time

S = os.path.dirname(os.path.abspath(__file__))
SF = ("/home/adam/chess/engines/Stockfish/src/stockfish", "/home/adam/chess/engines/Stockfish/src")
SF17 = ("/home/adam/chess/engines/Stockfish-17.1/src/stockfish", "/home/adam/chess/engines/Stockfish-17.1/src")
CODA = ("/home/adam/code/coda/coda", "/home/adam/code/coda")
BUDGETS = [15000, 50000, 150000, 500000]

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
    def eval_at(self, fen, nodes):
        self.send("ucinewgame"); self.ready()
        self.send(f"position fen {fen}")
        self.send(f"go nodes {nodes}")
        cp = None
        for l in reversed(self.wait("bestmove")[:-1]):
            t = l.split()
            if "score" in t:
                i = t.index("score")
                cp = int(t[i+2]) if t[i+1] == "cp" else (10000 if int(t[i+2]) > 0 else -10000)
                break
        return cp
    def quit(self):
        try: self.send("quit"); self.p.wait(timeout=5)
        except Exception: self.p.kill()

def load_positions():
    rows = list(csv.DictReader(open(f"{S}/horizon_ep_starts.tsv"), delimiter='\t'))
    seen, out = set(), []
    for r in rows:
        key = ' '.join(r['fen'].split()[:4])
        if key in seen: continue
        seen.add(key); out.append(r['fen'])
    return out

def main():
    mode = sys.argv[1]
    fens = load_positions()
    print(f"{len(fens)} unique positions", flush=True)
    if mode == '--truth':
        eng = Eng(*SF, 8, 1024)
        t0 = time.time()
        with open(f"{S}/truth_cache.tsv", 'w') as f:
            f.write("fen\tvstar\n")
            for i, fen in enumerate(fens):
                v = eng.eval_at(fen, 4_000_000)
                f.write(f"{fen}\t{v}\n"); f.flush()
                if (i+1) % 50 == 0: print(f"truth {i+1}/{len(fens)} {time.time()-t0:.0f}s", flush=True)
        eng.quit()
    elif mode == '--measure':
        name = sys.argv[2]
        path, cwd = {'coda': CODA, 'sf17': SF17, 'sf18': SF}[name]
        eng = Eng(path, cwd, 1, 256)
        t0 = time.time()
        with open(f"{S}/conv_results.tsv", 'a') as f:
            for n in BUDGETS:
                for i, fen in enumerate(fens):
                    v = eng.eval_at(fen, n)
                    f.write(f"{name}\t{n}\t{fen}\t{v}\n")
                f.flush()
                print(f"{name}@{n} done {time.time()-t0:.0f}s", flush=True)
        eng.quit()
    elif mode == '--report':
        truth = {r['fen']: int(r['vstar']) for r in
                 csv.DictReader(open(f"{S}/truth_cache.tsv"), delimiter='\t')}
        import statistics as st
        from collections import defaultdict
        data = defaultdict(list)
        for line in open(f"{S}/conv_results.tsv"):
            name, n, fen, v = line.rstrip('\n').split('\t')
            vs = truth.get(fen)
            if vs is None or abs(vs) > 800 or v == 'None': continue
            data[(name, int(n))].append(abs(int(v) - vs))
        print(f"{'engine':7}{'nodes':>8}{'n':>6}{'mean|d|':>9}{'med|d|':>8}{'<=50cp':>8}{'<=100cp':>9}")
        for (name, n), d in sorted(data.items()):
            print(f"{name:7}{n:>8}{len(d):>6}{st.mean(d):>9.1f}{st.median(d):>8.0f}"
                  f"{100*sum(1 for x in d if x <= 50)/len(d):>7.1f}%"
                  f"{100*sum(1 for x in d if x <= 100)/len(d):>8.1f}%")

if __name__ == '__main__':
    main()
