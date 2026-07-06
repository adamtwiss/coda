#!/usr/bin/env python3
"""tree_shape_study.py — empirical tree-shape comparison across engines.

For each engine and each FEN, runs a single fixed-depth search (go depth D)
and parses the iterative-deepening info lines: nodes, time, seldepth at every
completed depth. No engine patching needed — Stage 1 of the tree-shape study
(2026-07-06, prompted by Atlas's measure-first list).

Outputs per engine:
  - median nodes-to-depth-D for D in a grid (the ply-density / EBF curve)
  - median EBF at depth (nodes_D / nodes_{D-1})
  - median seldepth - depth spread (selective spike depth)
And a cross-engine table ready for docs/.

Usage:
  python3 scripts/tree_shape_study.py --fens FILE --depth 18 \
      --engine Coda=./coda --engine SF=~/chess/engines/Stockfish/src/stockfish \
      --engine Reckless=~/chess/engines/Reckless/target/release/reckless \
      [--limit 150] [--hash 256] [--csv out.csv]

FEN file: one FEN per line (extra fields after the 6 FEN tokens ignored,
so .tsv/.epd files work).
"""
import argparse, json, os, statistics, subprocess, sys


def run_engine(path, fens, depth, hash_mb):
    """One persistent process; ucinewgame between positions (clean TT)."""
    p = subprocess.Popen([os.path.expanduser(path)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
    def wait_for(tok):
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"{path}: engine died")
            if line.startswith(tok):
                return line
    send("uci"); wait_for("uciok")
    send(f"setoption name Hash value {hash_mb}")
    send("isready"); wait_for("readyok")

    results = []  # per position: {depth: (nodes, seldepth, time_ms)}
    for i, fen in enumerate(fens):
        send("ucinewgame"); send("isready"); wait_for("readyok")
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        per_depth = {}
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"{path}: engine died mid-search")
            if line.startswith("bestmove"):
                break
            t = line.split()
            if line.startswith("info") and "depth" in t and "nodes" in t and "score" in t:
                try:
                    d = int(t[t.index("depth") + 1])
                    n = int(t[t.index("nodes") + 1])
                    sd = int(t[t.index("seldepth") + 1]) if "seldepth" in t else None
                    ms = int(t[t.index("time") + 1]) if "time" in t else None
                    # keep the LAST report per depth (aspiration re-searches)
                    per_depth[d] = (n, sd, ms)
                except (ValueError, IndexError):
                    pass
        results.append(per_depth)
        if (i + 1) % 25 == 0:
            print(f"  {os.path.basename(path)}: {i+1}/{len(fens)}", file=sys.stderr)
    send("quit"); p.wait(timeout=5)
    return results


def median_curve(results, depth):
    """median nodes/seldepth at each depth over positions that reached it."""
    out = {}
    for d in range(2, depth + 1):
        nodes = [r[d][0] for r in results if d in r]
        selds = [r[d][1] for r in results if d in r and r[d][1] is not None]
        if len(nodes) >= 5:
            out[d] = {
                "n_pos": len(nodes),
                "med_nodes": int(statistics.median(nodes)),
                "med_seldepth": statistics.median(selds) if selds else None,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", required=True)
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--hash", type=int, default=256)
    ap.add_argument("--engine", action="append", required=True,
                    help="NAME=PATH (repeatable)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    fens = []
    for ln in open(os.path.expanduser(a.fens)):
        ln = ln.strip().split("\t")[0]
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 4:
            fens.append(" ".join(parts[:6]) if len(parts) >= 6 else ln + " 0 1")
        if len(fens) >= a.limit:
            break
    print(f"{len(fens)} positions, depth {a.depth}, hash {a.hash}", file=sys.stderr)

    engines = [e.split("=", 1) for e in a.engine]
    curves = {}
    for name, path in engines:
        print(f"running {name} ({path})...", file=sys.stderr)
        res = run_engine(path, fens, a.depth, a.hash)
        curves[name] = median_curve(res, a.depth)

    # cross-engine table
    depths = sorted(set().union(*[set(c) for c in curves.values()]))
    hdr = f"{'D':>3} " + " ".join(f"{n+'-nodes':>12} {n+'-EBF':>7} {n+'-seld':>7}" for n, _ in engines)
    print("\n" + hdr)
    print("-" * len(hdr))
    for d in depths:
        row = f"{d:>3} "
        for name, _ in engines:
            c = curves[name].get(d)
            prev = curves[name].get(d - 1)
            if c:
                ebf = c["med_nodes"] / prev["med_nodes"] if prev else float("nan")
                sd = c["med_seldepth"]
                row += f"{c['med_nodes']:>12} {ebf:>7.2f} {(f'{sd:.0f}' if sd else '—'):>7}"
            else:
                row += f"{'—':>12} {'—':>7} {'—':>7}"
        print(row)

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("engine,depth,n_pos,med_nodes,med_seldepth\n")
            for name, _ in engines:
                for d, c in sorted(curves[name].items()):
                    f.write(f"{name},{d},{c['n_pos']},{c['med_nodes']},{c['med_seldepth']}\n")
        print(f"\ncsv: {a.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
