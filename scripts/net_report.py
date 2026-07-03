#!/usr/bin/env python3
"""net_report.py — one-command eval-quality report card for a net (or A/B of nets).

Answers "how good is this model's static eval?" across four lenses, all keyed
by FEN (never by line order — the UCI-scrape misalignment trap can't happen):

  A. GENERAL eval quality (broad quiet T80 sample, vs LC0 800-node truth):
       mean|err|, the long tail (p90/p99/max), and Spearman + Pearson of the
       net's eval vs the oracle. This is the strength-relevant number; prod-era
       nets sit ~Spearman 0.85 / Pearson 0.83.
  B. BLINDSPOT overscore (held-out overrate corpus — positions Coda is KNOWN
       to misscore): signed overscore (mean eval-lc0; +ve = rates too high) and
       mean|err|. prod-era nets ~207-224 mean|err| here (recipe-invariant band).
  C. LONG-TAIL on the blindspot set: p90/p99/max |err| — where blindspots live.
  D. WANDERING BISHOP (overrate.epd forward-bishop subset, n=6): mean net_pref
       = mover-POV cp by which the net prefers the bad bishop sortie over the
       correct move. Lower = better; <=0 means the net ranks the correct move
       first. Reference points (same metric, same subset): SF static ~+65;
       current prod nets ~+60-70. NO Stockfish, NO stamped corpus in the
       measurement — the net's own STATIC eval only.

Uses `coda eval-fens` (batch, FEN-keyed) and `coda eval-dist` (general CSV).
Builds ./coda if missing. Needs python-chess for section D.

Usage:
  python3 scripts/net_report.py nets/candidate.nnue [nets/baseline.nnue ...]
  python3 scripts/net_report.py --no-general nets/a.nnue        # skip A (fast)
  python3 scripts/net_report.py --general-n 40000 nets/a.nnue   # bigger general sample
"""
import argparse, subprocess, sys, os, tempfile

CODA = "./coda"
HELDOUT = "testdata/heldout_overrate_lc0_2023_06.tsv"
OVERRATE_EPD = "testdata/overrate.epd"
DEFAULT_GENERAL = "/training/sf/test80-2023-06-jun-2tb7p.min-v2.v6.binpack"
FWD_BISHOP = {"Bxf4", "Bf3", "Bd6", "Bd2", "Bd4+", "Bg5"}


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr[-1000:]}")
    return r.stdout


def eval_fens(net, fens):
    """Return {fen: eval_stm_cp} for a list of FENs via `coda eval-fens`."""
    fi = tempfile.NamedTemporaryFile("w", suffix=".fens", delete=False)
    fi.write("\n".join(fens) + "\n"); fi.close()
    fo = tempfile.NamedTemporaryFile("r", suffix=".evals", delete=False); fo.close()
    sh([CODA, "eval-fens", "-i", fi.name, "-o", fo.name, "-n", net])
    d = {}
    for ln in open(fo.name):
        f, v = ln.rstrip("\n").split("\t"); d[f] = int(v)
    os.unlink(fi.name); os.unlink(fo.name)
    return d


def pctl(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def spearman_pearson(x, y):
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return spear, pear


def general_quality(net, binpack, n):
    """Spearman/Pearson + error tail on a broad quiet sample, via eval-dist CSV."""
    csv = tempfile.NamedTemporaryFile("r", suffix=".csv", delete=False); csv.close()
    sh([CODA, "eval-dist", "-i", binpack, "-c", str(n), "--quiet-only",
        "--max-abs-lc0", "600", "--csv", csv.name, "-n", net])
    coda, lc0 = [], []
    for ln in open(csv.name):
        p = ln.rstrip("\n").split(",")
        if p[0] == "fen" or len(p) < 4:
            continue
        try:
            coda.append(int(p[2])); lc0.append(int(p[3]))
        except ValueError:
            pass
    os.unlink(csv.name)
    err = sorted(abs(c - l) for c, l in zip(coda, lc0))
    n = len(err)
    spear, pear = spearman_pearson(coda, lc0)
    return dict(n=n, mae=sum(err) / n, p90=pctl(err, .9), p99=pctl(err, .99),
                mx=err[-1], spear=spear, pear=pear)


def blindspot(net):
    truth = {}
    for ln in open(HELDOUT):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 2:
            truth[p[0]] = int(p[1])
    ev = eval_fens(net, list(truth))
    fens = [f for f in truth if f in ev]
    signed = [ev[f] - truth[f] for f in fens]
    absd = sorted(abs(x) for x in signed)
    n = len(fens)
    return dict(n=n, overscore=sum(signed) / n, mae=sum(absd) / n,
                p90=pctl(absd, .9), p99=pctl(absd, .99), mx=absd[-1])


def wandering_bishop(net):
    import chess, re
    # Build after-am / after-bm FENs for every epd line; batch-eval them all.
    # After either move it's the OPPONENT to move, so the STM-POV eval of the
    # resulting position is the opponent's POV; the mover's POV is its negation
    # for BOTH resulting positions. Hence net_pref (mover prefers am over bm) =
    # moverPOV(am) - moverPOV(bm) = (-ev[fa]) - (-ev[fb]) = ev[fb] - ev[fa].
    # No mover-color sign term (that was a white-POV-eval artifact; eval-fens is
    # STM-POV, verified white-up-a-queen +/-).
    cases = []
    need = set()
    for line in open(OVERRATE_EPD):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fen = line.split(" bm ")[0].strip()
        if len(fen.split()) == 4:
            fen += " 0 1"
        bm = re.search(r" bm (\S+?);", line); am = re.search(r" am (\S+?);", line)
        if not (bm and am):
            continue
        bm, am = bm.group(1), am.group(1)
        b = chess.Board(fen)
        def after(san):
            mv = next((m for m in b.legal_moves
                       if b.san(m).rstrip("+#") == san.rstrip("+#")), None)
            if mv is None:
                return None
            bb = chess.Board(fen); bb.push(mv)
            return bb.fen()
        fa = after(am); fb = after(bm)
        if fa is None or fb is None:
            continue
        need.add(fa); need.add(fb)
        cases.append((am, bm, fa, fb))
    ev = eval_fens(net, list(need))
    rows = []
    for am, bm, fa, fb in cases:
        if fa not in ev or fb not in ev:
            continue
        pref = ev[fb] - ev[fa]      # mover-POV cp preferring the bad am over bm
        rows.append((am, bm, pref))
    fb_rows = [r for r in rows if r[0] in FWD_BISHOP]
    fb_mean = sum(r[2] for r in fb_rows) / len(fb_rows) if fb_rows else float("nan")
    flips = sum(1 for _, _, p in rows if p < 0)
    return dict(n=len(rows), fb_n=len(fb_rows), fb_mean=fb_mean, flips=flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nets", nargs="+")
    ap.add_argument("--general-binpack", default=DEFAULT_GENERAL)
    ap.add_argument("--general-n", type=int, default=20000)
    ap.add_argument("--no-general", action="store_true",
                    help="skip section A (no binpack needed; fast)")
    a = ap.parse_args()

    if not os.path.exists(CODA):
        print("building ./coda (make) ...", file=sys.stderr)
        subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL)

    do_general = not a.no_general and os.path.exists(a.general_binpack)
    if not a.no_general and not do_general:
        print(f"note: general binpack {a.general_binpack} not found — skipping section A\n",
              file=sys.stderr)

    rows = []
    for net in a.nets:
        r = {"net": os.path.basename(net)}
        if do_general:
            r["gen"] = general_quality(net, a.general_binpack, a.general_n)
        r["bs"] = blindspot(net)
        try:
            r["wb"] = wandering_bishop(net)
        except ImportError:
            r["wb"] = None
        rows.append(r)

    def col(r, path, fmt, default="—"):
        cur = r
        for k in path.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                return default
        return fmt.format(cur)

    if do_general:
        print(f"\nA. GENERAL eval quality  (sample {rows[0]['gen']['n']} quiet, |lc0|<=600, "
              f"{os.path.basename(a.general_binpack)})")
        h = f"   {'net':<32} {'Spearman':>9} {'Pearson':>8} {'mean|err|':>10} {'p90':>5} {'p99':>5} {'max':>6}"
        print(h); print("   " + "-" * (len(h) - 3))
        for r in rows:
            print(f"   {r['net'][:32]:<32} {col(r,'gen.spear','{:.3f}'):>9} "
                  f"{col(r,'gen.pear','{:.3f}'):>8} {col(r,'gen.mae','{:.1f}'):>10} "
                  f"{col(r,'gen.p90','{:.0f}'):>5} {col(r,'gen.p99','{:.0f}'):>5} "
                  f"{col(r,'gen.mx','{:.0f}'):>6}")

    print(f"\nB/C. BLINDSPOT overscore + long tail  (heldout overrate set, "
          f"{rows[0]['bs']['n']} pos)")
    h = f"   {'net':<32} {'overscore':>10} {'mean|err|':>10} {'p90':>5} {'p99':>5} {'max':>6}"
    print(h); print("   " + "-" * (len(h) - 3))
    base_mae = None
    for r in rows:
        mae = r["bs"]["mae"]
        d = "" if base_mae is None else f"  ({mae-base_mae:+.1f})"
        if base_mae is None:
            base_mae = mae
        print(f"   {r['net'][:32]:<32} {r['bs']['overscore']:>+10.1f} {mae:>10.1f}{d} "
              f"{r['bs']['p90']:>5} {r['bs']['p99']:>5} {r['bs']['mx']:>6}")

    print(f"\nD. WANDERING BISHOP  (forward-bishop sortie subset; lower=better, "
          f"<=0 = ranks correct move first; SF static ~+65, prod nets ~+60-70)")
    h = f"   {'net':<32} {'fb_net_pref':>12} {'n':>3} {'sorties-now-correct':>20}"
    print(h); print("   " + "-" * (len(h) - 3))
    for r in rows:
        if r["wb"] is None:
            print(f"   {r['net'][:32]:<32}  (needs python-chess)")
            continue
        wb = r["wb"]
        flips = f"{wb['flips']}/{wb['n']}"
        print(f"   {r['net'][:32]:<32} {wb['fb_mean']:>+12.0f} {wb['fb_n']:>3} {flips:>20}")
    print()


if __name__ == "__main__":
    main()
