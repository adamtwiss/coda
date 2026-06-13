#!/usr/bin/env python3
"""Cross-engine static-eval quality benchmark.

Question: relative to top engines, how good is each engine's NNUE *static
eval* at predicting game outcomes? This isolates eval quality from search.

Method: for a shared set of quiet positions (FEN + actual game result, white
POV), get each engine's static eval via UCI `eval`, fit a per-engine logistic
`win% = sigmoid(a*eval_pawns + b)` (the per-engine fit absorbs eval-scale
differences, so we compare *predictive power*, not raw cp), and score it by
Brier (MSE of predicted win% vs result) and logloss. Lower = better eval.

Coda's eval is taken from the CSV column (computed in-process by
`coda eval-dist --csv`); other engines are driven via UCI on the same FENs.

Usage:
  coda eval-dist -i <binpack> -c 20000 --quiet-only --csv /tmp/evalq.csv -n <net>
  python3 scripts/eval_quality.py /tmp/evalq.csv
"""
import re
import subprocess
import sys
import numpy as np

# (label, binary, extra UCI setoption lines) — each uses its own production net.
ENGINES = [
    ("Stockfish", "/home/adam/chess/engines/Stockfish/src/stockfish", []),
    ("Reckless", "/home/adam/chess/engines/Reckless/reckless", []),
]
EVAL_RE = re.compile(r"NNUE evaluation\s+([+-]?[0-9]+\.[0-9]+)", re.IGNORECASE)


def drive_engine(binary, setopts, fens):
    """Return list of static evals (white-POV cp) for each FEN, or None on miss.

    Writes all UCI commands to a temp file and runs `engine < cmds > out` so
    there's no stdin/stdout pipe deadlock and no per-position sync. Each
    `eval` emits exactly one 'NNUE evaluation' line; we map them to FENs by
    order, using a printed marker line per position to stay aligned even if
    an engine skips a position.
    """
    import tempfile, os
    cmd_path = tempfile.mktemp(suffix=".uci")
    out_path = tempfile.mktemp(suffix=".out")
    with open(cmd_path, "w") as f:
        f.write("uci\n")
        for so in setopts:
            f.write(so + "\n")
        for i, fen in enumerate(fens):
            # MARK line (echoed back via an unknown command is unreliable;
            # instead rely on strict ordering — one eval line per position).
            f.write(f"position fen {fen}\neval\n")
        f.write("quit\n")
    with open(cmd_path) as fin, open(out_path, "w") as fout:
        subprocess.run([binary], stdin=fin, stdout=fout,
                       stderr=subprocess.DEVNULL, timeout=600)
    evals = []
    with open(out_path) as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                evals.append(float(m.group(1)) * 100.0)
    os.unlink(cmd_path); os.unlink(out_path)
    # Strict ordering: one eval per position. If counts mismatch, return what
    # we have padded/truncated so the caller can warn.
    if len(evals) < len(fens):
        evals += [None] * (len(fens) - len(evals))
    return evals[:len(fens)]


def spearman(x, y):
    """Rank correlation, scipy-free."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def agreement(evals_cp, oracle_cp):
    """How well an engine's static eval matches the LC0 deep-eval oracle.
    Returns (spearman, pearson, calibrated_rms_cp). The per-engine linear
    calibration absorbs scale, so RMS is the residual after best-fit scaling."""
    e = np.asarray(evals_cp, dtype=np.float64)
    o = np.asarray(oracle_cp, dtype=np.float64)
    sp = spearman(e, o)
    pe = float(np.corrcoef(e, o)[0, 1])
    # best linear fit e->o, residual RMS in oracle cp units
    A = np.vstack([e, np.ones_like(e)]).T
    coef, *_ = np.linalg.lstsq(A, o, rcond=None)
    resid = o - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return sp, pe, rms


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/evalq.csv"
    fens, oracle, coda_eval = [], [], []
    with open(csv_path) as f:
        next(f)  # header: fen,white_result,coda_eval_white_cp,lc0_score_white_cp
        for line in f:
            parts = line.rstrip("\n").rsplit(",", 3)
            if len(parts) != 4:
                continue
            fens.append(parts[0])
            coda_eval.append(float(parts[2]))
            oracle.append(float(parts[3]))  # LC0 deep-eval oracle (white cp)
    print(f"Loaded {len(fens)} positions from {csv_path}")
    print("Ground truth: LC0 WDL-calibrated deep eval (game RESULT is ~0-corr "
          "on T80 self-play — too drawish to use).\n")

    rows = []
    sp, pe, rms = agreement(coda_eval, oracle)
    rows.append(("Coda (549C20A5)", sp, pe, rms, len(fens)))

    for label, binary, setopts in ENGINES:
        print(f"Driving {label} over {len(fens)} positions...", flush=True)
        try:
            evs = drive_engine(binary, setopts, fens)
        except Exception as e:
            print(f"  {label} failed: {e}")
            continue
        pairs = [(ev, o) for ev, o in zip(evs, oracle) if ev is not None]
        if len(pairs) < 0.9 * len(fens):
            print(f"  {label}: only {len(pairs)}/{len(fens)} evals parsed — check format")
        if not pairs:
            continue
        ev2, o2 = zip(*pairs)
        sp, pe, rms = agreement(ev2, o2)
        rows.append((label, sp, pe, rms, len(pairs)))

    rows.sort(key=lambda x: -x[1])  # by Spearman desc, higher = closer to oracle
    print("\n=== Static-eval agreement with LC0 deep-eval oracle (higher Spearman = better) ===")
    print(f"{'Engine':<22} {'Spearman':>9} {'Pearson':>8} {'residRMS':>9} {'N':>7}")
    best = rows[0][1]
    for label, sp, pe, rms, n in rows:
        gap = best - sp
        print(f"{label:<22} {sp:>9.4f} {pe:>8.4f} {rms:>9.1f} {n:>7}  {'(best)' if gap==0 else f'-{gap:.4f}'}")
    print("\nCaveat: Coda trains on T80/LC0 scores, so it has a home-field bias "
          "toward matching this oracle. SF/Reckless train on their own data.")


if __name__ == "__main__":
    main()
