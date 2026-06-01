#!/usr/bin/env python3
"""Empirical training-run diagnostics for Coda NNUE trains.

Two things confound the raw train-loss curve we get out of Bullet:
  (1) per-batch loss samples a subset of games, so it is noisy and
      dominated by data-order, and
  (2) a low-LR cosine tail can't durably *raise* loss on fixed data, so a
      late rise is harder back-half data, not regression.

So we lean on TWO views that are immune to those:
  - smoothed train loss (rolling mean) for the macro trend / trough, and
  - per-checkpoint eval-distribution on a FIXED binpack (eval-dist), which
    is immune to data-order entirely and exposes eval-scale instability and
    tail-fattening (the overtraining signature).

eval-dist measures DISTRIBUTION, not strength: thin early tails partly
reflect undertraining, so this can flag instability/overtraining onset but
cannot rank checkpoint strength on its own. SPRT remains the arbiter.

Usage:
  # Sweep raw checkpoints -> results.csv (converts each, runs eval-dist):
  train_diag.py sweep \
      --ckpt-glob '/tmp/ckpt/q-*.bin' \
      --binpack /training/coda/selfplay-d8-2026-04-11.binpack \
      --count 80000 --out /tmp/ckpt/results.csv \
      --convert-flags '--pairwise --screlu --hidden 16 --hidden2 32 \
                       --int8l1 --threats 66864 --kb-layout reckless --hl-crelu'

  # Plot results.csv (+ optional smoothed loss log) -> PNG:
  train_diag.py plot --results /tmp/ckpt/results.csv \
      --loss /tmp/log.txt --out /tmp/ckpt/diag.png \
      --ref-rms 277 --ref-p99 755 --ref-p90 528 --ref-p50 97
"""
import argparse
import csv
import os
import re
import subprocess
import sys

CODA = os.environ.get("CODA", "/home/adam/code/coda/coda")
DEFAULT_CONVERT = ("--pairwise --screlu --hidden 16 --hidden2 32 --int8l1 "
                   "--threats 66864 --kb-layout reckless --hl-crelu")

# SB number is the trailing integer in the checkpoint stem, e.g. q-1600.bin -> 1600,
# q-1600swa.bin -> 1600 (swa suffix kept as a label distinction).
_SB_RE = re.compile(r"(\d+)(\D*)$")


def _sb_label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _SB_RE.search(stem)
    if not m:
        return (10**9, stem)
    sb = int(m.group(1))
    suffix = m.group(2) or ""
    label = f"{sb}{suffix}" if suffix else str(sb)
    return (sb, label)


def _grep(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def sweep(args):
    import glob
    paths = sorted(glob.glob(args.ckpt_glob), key=_sb_label)
    if not paths:
        sys.exit(f"no checkpoints match {args.ckpt_glob}")
    rows = []
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sb", "label", "mean", "absmean", "rms", "p50", "p90", "p99"])
        for p in paths:
            sb, label = _sb_label(p)
            nnue = os.path.join(os.path.dirname(p), f"net-{label}.nnue")
            if not (args.reuse_nnue and os.path.exists(nnue)):
                conv = [CODA, "convert-bullet", "-i", p, "-o", nnue] + args.convert_flags.split()
                subprocess.run(conv, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out = subprocess.run(
                [CODA, "eval-dist", "-i", args.binpack, "-c", str(args.count), "-n", nnue],
                capture_output=True, text=True).stdout
            row = [
                sb, label,
                _grep(r"Mean:\s+([+-]?[0-9.]+)", out),
                _grep(r"Abs mean:\s+([0-9.]+)", out),
                _grep(r"RMS:\s+([0-9.]+)", out),
                _grep(r"p50:\s+([+-]?[0-9]+)", out),
                _grep(r"p90:\s+([+-]?[0-9]+)", out),
                _grep(r"p99:\s+([+-]?[0-9]+)", out),
            ]
            w.writerow(row)
            f.flush()
            print(",".join(str(x) for x in row))
            rows.append(row)
    print(f"wrote {args.out} ({len(rows)} checkpoints)")


def _rolling(xs, ys, win):
    out = []
    half = win // 2
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sb, rms, absm, p50, p90, p99 = [], [], [], [], [], []
    with open(args.results) as f:
        for r in csv.DictReader(f):
            sb.append(float(r["sb"]))
            rms.append(float(r["rms"]))
            absm.append(float(r["absmean"]))
            p50.append(float(r["p50"]))
            p90.append(float(r["p90"]))
            p99.append(float(r["p99"]))

    npanel = 3 if args.loss else 2
    fig, ax = plt.subplots(npanel, 1, figsize=(10, 3.2 * npanel))
    if npanel == 2:
        ax = list(ax)

    # Panel 1: eval scale
    ax[0].plot(sb, rms, "o-", label="RMS", color="C0")
    ax[0].plot(sb, absm, "s-", label="abs mean", color="C1")
    if args.ref_rms:
        ax[0].axhline(args.ref_rms, ls="--", color="C0", alpha=0.5,
                      label=f"prod RMS {args.ref_rms}")
    ax[0].set_ylabel("eval scale (cp)")
    ax[0].set_title("Eval scale vs checkpoint (fixed binpack) — "
                    "oscillation = scale instability")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # Panel 2: percentile fan
    ax[1].plot(sb, p99, "o-", label="p99", color="C3")
    ax[1].plot(sb, p90, "s-", label="p90", color="C2")
    ax[1].plot(sb, p50, "^-", label="p50", color="C4")
    for ref, c, lab in ((args.ref_p99, "C3", "p99"),
                        (args.ref_p90, "C2", "p90"),
                        (args.ref_p50, "C4", "p50")):
        if ref:
            ax[1].axhline(ref, ls="--", color=c, alpha=0.5, label=f"prod {lab} {ref}")
    ax[1].set_ylabel("percentile (cp)")
    ax[1].set_xlabel("superbatch")
    ax[1].set_title("Eval percentile fan — tail fattening (p99 up, p50 flat) "
                    "= overtraining signature")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)

    # Panel 3 (optional): smoothed train loss
    if args.loss:
        x, y = [], []
        with open(args.loss) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        x.append(float(parts[0]))
                        y.append(float(parts[2]))
                    except ValueError:
                        continue
        r50 = _rolling(x, y, 50)
        r200 = _rolling(x, y, 200)
        ax[2].plot(x, y, color="0.8", lw=0.4, label="raw (per-batch)")
        ax[2].plot(x, r50, color="C0", lw=0.8, label="rolling-50")
        ax[2].plot(x, r200, color="C3", lw=1.5, label="rolling-200 (macro)")
        imin = min(range(len(r200)), key=lambda i: r200[i])
        ax[2].axvline(x[imin], ls=":", color="C3", alpha=0.6,
                      label=f"macro trough SB~{int(x[imin])}")
        ax[2].set_ylabel("train loss")
        ax[2].set_xlabel("superbatch")
        ax[2].set_title("Train loss (smoothed) — data-order-confounded; trough "
                        "is a hint, not proof of overtraining")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="convert checkpoints + run eval-dist -> results.csv")
    s.add_argument("--ckpt-glob", required=True)
    s.add_argument("--binpack", required=True)
    s.add_argument("--count", type=int, default=80000)
    s.add_argument("--out", required=True)
    s.add_argument("--convert-flags", default=DEFAULT_CONVERT)
    s.add_argument("--reuse-nnue", action="store_true",
                   help="skip convert if net-<label>.nnue already exists")
    s.set_defaults(func=sweep)

    p = sub.add_parser("plot", help="results.csv (+ loss log) -> diagnostic PNG")
    p.add_argument("--results", required=True)
    p.add_argument("--loss", help="optional train-log CSV: SB,batch,loss")
    p.add_argument("--out", required=True)
    p.add_argument("--ref-rms", type=float)
    p.add_argument("--ref-p99", type=float)
    p.add_argument("--ref-p90", type=float)
    p.add_argument("--ref-p50", type=float)
    p.set_defaults(func=plot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
