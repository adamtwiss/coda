#!/usr/bin/env python3
"""Parse TM-ceiling PGNs: Coda's score vs each opponent under CLOCK (TM active)
vs FIXED (TM disabled), and Δ = clock − fixed in Elo.

ΔElo > 0 → Coda's allocation TM beats that opponent's. ΔElo < 0 → opponent's
TM is better → Coda TM headroom = the deficit. Pooled Δ = relative standing.
"""
import glob
import math
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tm_ceiling"
HERO = sys.argv[2] if len(sys.argv) > 2 else "Coda"  # gauntlet engine name


def score_pgn(path):
    """Return (wins, draws, losses) for the gauntlet engine (HERO)."""
    w = d = l = 0
    white = result = None
    for line in open(path, errors="ignore"):
        if line.startswith("[White "):
            white = HERO in line
        elif line.startswith("[Result "):
            m = re.search(r'"([^"]+)"', line)
            res = m.group(1) if m else "*"
            if res == "1/2-1/2":
                d += 1
            elif res == "1-0":      # white won
                if white: w += 1
                else:     l += 1
            elif res == "0-1":      # black won
                if white: l += 1
                else:     w += 1
    return w, d, l


def elo(score, n):
    if n == 0 or score <= 0 or score >= 1:
        return float("nan"), float("nan")
    e = -400 * math.log10(1 / score - 1)
    # rough CI
    p = score
    se = math.sqrt(p * (1 - p) / n)
    lo = max(1e-6, p - 1.96 * se); hi = min(1 - 1e-6, p + 1.96 * se)
    ci = (-400 * math.log10(1 / hi - 1) - (-400 * math.log10(1 / lo - 1))) / 2
    return e, ci


def cond_stats(opp, cond):
    f = os.path.join(OUT, f"{cond}_{opp}.pgn")
    if not os.path.exists(f):
        return None
    w, d, l = score_pgn(f)
    n = w + d + l
    if n == 0:
        return None
    s = (w + 0.5 * d) / n
    e, ci = elo(s, n)
    return dict(n=n, w=w, d=d, l=l, score=s, elo=e, ci=ci)


def main():
    opps = sorted({os.path.basename(p).split("_", 1)[1][:-4]
                   for p in glob.glob(os.path.join(OUT, "*.pgn"))})
    print(f"{'Opponent':<12} {'CLOCK Elo':>14} {'FIXED Elo':>14} {'ΔElo (TM)':>11} {'N/cond':>7}")
    pooled_c = [0, 0, 0]; pooled_f = [0, 0, 0]
    for opp in opps:
        c = cond_stats(opp, "clock"); f = cond_stats(opp, "fixed")
        if not c or not f:
            print(f"{opp:<12}  (incomplete: clock={'y' if c else 'n'} fixed={'y' if f else 'n'})")
            continue
        pooled_c[0] += c['w']; pooled_c[1] += c['d']; pooled_c[2] += c['l']
        pooled_f[0] += f['w']; pooled_f[1] += f['d']; pooled_f[2] += f['l']
        dtm = c['elo'] - f['elo']
        print(f"{opp:<12} {c['elo']:>7.1f}±{c['ci']:<5.1f} {f['elo']:>7.1f}±{f['ci']:<5.1f} "
              f"{dtm:>+10.1f} {min(c['n'], f['n']):>7}")
    # pooled
    def pooled(p):
        n = sum(p); s = (p[0] + 0.5 * p[1]) / n if n else 0; e, ci = elo(s, n); return e, ci, n
    ec, cic, nc = pooled(pooled_c); ef, cif, nf = pooled(pooled_f)
    if nc and nf:
        print("-" * 64)
        print(f"{'POOLED':<12} {ec:>7.1f}±{cic:<5.1f} {ef:>7.1f}±{cif:<5.1f} {ec-ef:>+10.1f} {min(nc,nf):>7}")
        print(f"\nΔElo = Coda's allocation-TM value vs the field. >0 = our TM ≥ field; "
              f"<0 = TM headroom.\n(Allocation TM only — excludes ponder-leech defence.)")


if __name__ == "__main__":
    main()
