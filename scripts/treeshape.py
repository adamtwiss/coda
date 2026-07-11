#!/usr/bin/env python3
"""Tree-shape comparison: Coda vs instrumented SF, TREESTATS at fixed nodes.

For each engine x {blind, neutral} x position: ucinewgame, position, go
nodes 150000, treestats; harness sums per-position dumps (both engines'
counters are per-search under this protocol). Reports per set:
  - node share by depth bucket (0=qs, 1-31 interior)
  - mean searched width by depth
  - first-move-cut rate by depth
  - pre-move-loop exit rate by depth (1 - width_cnt/nodes)
  - re-searches per 1M nodes (asp low/high, LMR)
"""
import subprocess, csv, time, sys
from collections import defaultdict

S = "/tmp/claude-1001/-home-adam-code-coda/6a3a1099-ab50-4682-8fea-48bc7890ac15/scratchpad"
ENGINES = {
    'coda': ("/home/adam/code/coda/coda", "/home/adam/code/coda"),
    'sf': ("/home/adam/chess/instr-stockfish/src/stockfish", "/home/adam/chess/instr-stockfish/src"),
}
NODES = 150000
N_PER_SET = 300

class Eng:
    def __init__(self, path, cwd):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, cwd=cwd)
        self.send("uci"); self.wait("uciok")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 256")
        if 'coda' in path: self.send("setoption name OwnBook value false")
        self.ready()
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def wait(self, tok):
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("dead")
            out.append(l.rstrip('\n'))
            if l.startswith(tok): return out
    def ready(self): self.send("isready"); self.wait("readyok")
    def stats_for(self, fen):
        self.send("ucinewgame"); self.ready()
        self.send(f"position fen {fen}")
        self.send(f"go nodes {NODES}")
        self.wait("bestmove")
        self.send("treestats")
        self.send("isready")
        lines = [l for l in self.wait("readyok") if l.startswith("TREESTATS")]
        out = {}
        for l in lines:
            parts = l.split()
            key = parts[1]
            out[key] = {}
            for kv in parts[2:]:
                if ':' in kv:
                    k, v = kv.split(':', 1)
                    out[key][k] = float(v)
        return out
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
# accumulators: agg[engine][set][key][bucket] -> weighted sums
agg = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
t0 = time.time()
for name, (path, cwd) in ENGINES.items():
    eng = Eng(path, cwd)
    for setname, fens in sets.items():
        for fen in fens:
            st = eng.stats_for(fen)
            a = agg[(name, setname)]
            for b, v in st.get('nodes_by_depth', {}).items():
                a['nodes'][int(b)] += v
            # width: recover sums via mean*[need counts]... instead use the
            # identity: mean_width lines lack counts, so re-derive from
            # first_move_cut_rate? No — accumulate mean weighted by cut count
            # is wrong. Simplest robust cross-engine aggregate: weight each
            # position's per-bucket mean width by that bucket's node count.
            for b, w in st.get('mean_width_by_depth', {}).items():
                n = st.get('nodes_by_depth', {}).get(b, 0)
                a['width_wsum'][int(b)] += w * n
                a['width_wcnt'][int(b)] += n
            for b, r in st.get('first_move_cut_rate', {}).items():
                n = st.get('nodes_by_depth', {}).get(b, 0)
                a['fmc_wsum'][int(b)] += r * n
                a['fmc_wcnt'][int(b)] += n
            for k, v in st.get('researches', {}).items():
                a['research'][k] += v
            for k, v in st.get('totals', {}).items():
                a['totals'][k] += v
        print(f"{name}/{setname} done {time.time()-t0:.0f}s", flush=True)
    eng.quit()

BUCKETS = [(0, 'qs'), (1, 'd1-3'), (4, 'd4-7'), (8, 'd8-11'), (12, 'd12-15'), (16, 'd16+')]
def bucket_of(b):
    if b == 0: return 'qs'
    if b <= 3: return 'd1-3'
    if b <= 7: return 'd4-7'
    if b <= 11: return 'd8-11'
    if b <= 15: return 'd12-15'
    return 'd16+'

for setname in sets:
    print(f"\n===== set: {setname} =====")
    print(f"{'':10}" + ''.join(f"{lab:>10}" for _, lab in BUCKETS) + f"{'asp/1M':>9}{'lmr/1M':>9}")
    for name in ENGINES:
        a = agg[(name, setname)]
        tot = sum(a['nodes'].values()) or 1
        share = defaultdict(float); wsum = defaultdict(float); wcnt = defaultdict(float)
        fsum = defaultdict(float); fcnt = defaultdict(float)
        for b, v in a['nodes'].items(): share[bucket_of(b)] += v
        for b in a['width_wsum']:
            wsum[bucket_of(b)] += a['width_wsum'][b]; wcnt[bucket_of(b)] += a['width_wcnt'][b]
        for b in a['fmc_wsum']:
            fsum[bucket_of(b)] += a['fmc_wsum'][b]; fcnt[bucket_of(b)] += a['fmc_wcnt'][b]
        asp = (a['research'].get('asp_fail_low', 0) + a['research'].get('asp_fail_high', 0)) / tot * 1e6
        lmr = a['research'].get('lmr', 0) / tot * 1e6
        print(f"{name:10}" + ''.join(f"{100*share[lab]/tot:>9.1f}%" for _, lab in BUCKETS) + f"{asp:>9.1f}{lmr:>9.0f}  <- node share")
        print(f"{'':10}" + ''.join(f"{(wsum[lab]/wcnt[lab]) if wcnt[lab] else 0:>10.2f}" for _, lab in BUCKETS) + f"{'':18}  <- mean width")
        print(f"{'':10}" + ''.join(f"{(fsum[lab]/fcnt[lab]) if fcnt[lab] else 0:>10.3f}" for _, lab in BUCKETS) + f"{'':18}  <- 1st-move-cut")
