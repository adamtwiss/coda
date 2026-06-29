#!/usr/bin/env python3
"""
eval_compare_nets.py — mechanism metric for the blindspot corrective net.

Eval one or more .nnue nets over the oracle-labelled EPD sets and report, per
set, how far each net's STATIC eval sits from the SF16-d24 ground truth encoded
in the EPD `c0` comment. The blindspot hypothesis predicts the corrective net
(T80 + blindspot data) pulls the OVERRATE set down toward truth while keeping the
CONVERSION set (real wins) high.

Positions are oriented STM-favoured by construction (the `coda_peak` was an
STM-POV search value), so we compare the net's STM-POV static eval (cp) to the
`d24 <signed>cp` truth, which is also STM-POV.

Usage:
  python3 scripts/eval_compare_nets.py \
      --net baseline=/path/t80_only.nnue --net corrective=/path/t80_plus.nnue \
      --epd overrate=testdata/coda_overrate_gauntlet.epd \
      --epd conversion=testdata/coda_conversion.epd \
      [--coda ./coda] [--dump]
"""
import argparse, re, subprocess, sys
from statistics import mean

TRUTH_RE = re.compile(r"d24\s*([+-]?\d+)\s*cp", re.I)
EVAL_RE  = re.compile(r"NNUE evaluation\s+([+-]?\d+\.\d+)")


def parse_epd(path):
    """Return list of (fen, stm_is_white, truth_cp_stm)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        # FEN = first 6 space-separated fields; rest is the opcode tail.
        parts = line.split()
        if len(parts) < 6:
            continue
        fen = " ".join(parts[:6])
        stm_white = parts[1] == "w"
        m = TRUTH_RE.search(line)
        truth = int(m.group(1)) if m else None   # STM-POV cp (d24 was STM)
        out.append((fen, stm_white, truth))
    return out


def eval_net(coda, net, positions):
    """Feed all FENs through one coda `eval` session; return STM-POV cp list."""
    script = ["uci", "isready"]
    for fen, _, _ in positions:
        script.append(f"position fen {fen}")
        script.append("eval")
    script.append("quit")
    proc = subprocess.run(
        [coda, "--nnue", net],
        input="\n".join(script) + "\n",
        capture_output=True, text=True, timeout=600,
    )
    whites = [float(m.group(1)) for m in EVAL_RE.finditer(proc.stdout)]
    if len(whites) != len(positions):
        sys.exit(f"FATAL: net={net} got {len(whites)} evals for "
                 f"{len(positions)} positions (output desync). "
                 f"stderr tail:\n{proc.stderr[-400:]}")
    # white-POV pawns -> STM-POV cp
    stm_cp = []
    for (fen, stm_white, _), w in zip(positions, whites):
        cp = round(w * 100)
        stm_cp.append(cp if stm_white else -cp)
    return stm_cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", action="append", required=True,
                    help="label=path (repeatable)")
    ap.add_argument("--epd", action="append", required=True,
                    help="label=path (repeatable)")
    ap.add_argument("--coda", default="./coda")
    ap.add_argument("--dump", action="store_true",
                    help="per-position eval table")
    args = ap.parse_args()

    nets = [s.split("=", 1) for s in args.net]
    epds = [s.split("=", 1) for s in args.epd]

    for elabel, epath in epds:
        positions = parse_epd(epath)
        truths = [t for _, _, t in positions]
        has_truth = all(t is not None for t in truths)
        evals = {nlabel: eval_net(args.coda, npath, positions)
                 for nlabel, npath in nets}

        print(f"\n=== {elabel}  ({len(positions)} positions, "
              f"{epath}) ===")
        hdr = f"{'metric':<22}" + "".join(f"{nl:>14}" for nl, _ in nets)
        print(hdr)
        # mean STM eval (cp): overrate -> want LOWER; conversion -> want HIGH
        row = f"{'mean STM eval (cp)':<22}"
        for nl, _ in nets:
            row += f"{mean(evals[nl]):>14.0f}"
        print(row)
        if has_truth:
            print(f"{'mean truth (cp)':<22}" +
                  f"{mean(truths):>14.0f}" + " " * (14 * (len(nets) - 1)))
            row = f"{'mean |eval-truth| (cp)':<22}"
            for nl, _ in nets:
                row += f"{mean(abs(e - t) for e, t in zip(evals[nl], truths)):>14.0f}"
            print(row)
        if len(nets) == 2:
            (la, _), (lb, _) = nets
            d = mean(b - a for a, b in zip(evals[la], evals[lb]))
            print(f"\n  mean Δ ({lb} − {la}) STM eval: {d:+.0f} cp")
            if has_truth:
                ea = mean(abs(e - t) for e, t in zip(evals[la], truths))
                eb = mean(abs(e - t) for e, t in zip(evals[lb], truths))
                print(f"  mean |err|: {la}={ea:.0f}  {lb}={eb:.0f}  "
                      f"({'improved' if eb < ea else 'worse'} by {abs(eb-ea):.0f} cp)")

        if args.dump:
            print()
            for i, (fen, _, t) in enumerate(positions):
                vals = "".join(f"{evals[nl][i]:>14d}" for nl, _ in nets)
                print(f"  [{i:>3}] truth={str(t):>6}{vals}  {fen}")


if __name__ == "__main__":
    main()
