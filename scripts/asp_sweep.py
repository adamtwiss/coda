#!/usr/bin/env python3
"""ASP_DELTA sweep: nodes-to-depth-12 (re-search cost) across window widths.

Same 48 bench FENs, T=1, ucinewgame per position. If windows are
half-width for our eval scale, wider deltas should REDUCE total nodes to
reach the same depth (fewer re-searches) up to the point where first-search
cost dominates. Also reports mean best-move agreement vs the delta=11
baseline as a sanity guard (same depth, same moves expected mostly).
"""
import subprocess, time, sys

S = "/tmp/claude-1001/-home-adam-code-coda/6a3a1099-ab50-4682-8fea-48bc7890ac15/scratchpad"
FENS = [l.strip() for l in open(f"{S}/bench_fens.txt") if l.strip()]
VALUES = [11, 16, 22, 30]
DEPTH = 12

def run(delta):
    p = subprocess.Popen(["/home/adam/code/coda/coda"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, cwd="/home/adam/code/coda")
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    def wait(tok):
        out = []
        while True:
            l = p.stdout.readline()
            if not l: raise RuntimeError("dead")
            out.append(l)
            if l.startswith(tok): return out
    send("uci"); wait("uciok")
    send("setoption name Threads value 1")
    send("setoption name Hash value 64")
    send("setoption name OwnBook value false")
    send(f"setoption name ASP_DELTA value {delta}")
    send("isready"); wait("readyok")
    total_nodes = 0; moves = []
    for fen in FENS:
        send("ucinewgame"); send("isready"); wait("readyok")
        send(f"position fen {fen}")
        send(f"go depth {DEPTH}")
        lines = wait("bestmove")
        moves.append(lines[-1].split()[1])
        nodes = 0
        for l in reversed(lines[:-1]):
            t = l.split()
            if "nodes" in t: nodes = int(t[t.index("nodes")+1]); break
        total_nodes += nodes
    send("quit"); p.wait(timeout=5)
    return total_nodes, moves

base_moves = None
print(f"{'ASP_DELTA':>9} {'nodes-to-d12':>13} {'vs d=11':>8} {'move-agree':>11}")
base_nodes = None
for d in VALUES:
    n, mv = run(d)
    if base_nodes is None: base_nodes, base_moves = n, mv
    agree = 100*sum(1 for a, b in zip(mv, base_moves) if a == b)/len(mv)
    print(f"{d:>9} {n:>13} {100*(n-base_nodes)/base_nodes:>+7.1f}% {agree:>10.0f}%", flush=True)
