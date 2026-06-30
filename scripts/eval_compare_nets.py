#!/usr/bin/env python3
"""
eval_compare_nets.py — mechanism metric for the blindspot corrective net.

Eval one or more .nnue nets over a labelled test set and report, per set, how
far each net's STATIC eval sits from the ground-truth score. The blindspot
hypothesis predicts the corrective net (T80 + blindspot data) pulls the OVERRATE
positions down toward truth.

PRIMARY test set (`--tsv`): the held-out LC0-truth overrate set
`testdata/heldout_overrate_lc0_2023_06.tsv` — `fen<TAB>lc0_stm_cp`. Built as a
held-out twin (June 2023, NOT in Jan–Jun-2024 training) of the training harvest
filter: Coda static far from LC0 (>=150cp) AND SF static closer to LC0 (>=80cp),
deduped to <=1 position per game. Truth = LC0 800-node-MCTS score, STM-POV. This
matches what the corrective data trains toward and uses NO search anywhere.
See docs/blindspot_data_generation.md.

Legacy EPD sets (`--epd`) used an SF-searched-to-d24 oracle (the `d24 <cp>`
token). That oracle MISMATCHES the harvest's SF-*static* arbitrator: search
resolves tactics, so the EPD kept deep-tactical positions the static corrective
data cannot fix — contaminating the set. The contaminated
`coda_overrate_gauntlet.epd` was removed 2026-06-30; the `--epd`/parse_epd path
is retained only for ad-hoc d24-labelled sets.

Usage:
  python3 scripts/eval_compare_nets.py \
      --net ctrl=/path/t80_only.nnue --net mix=/path/t80_plus.nnue \
      --tsv heldout=testdata/heldout_overrate_lc0_2023_06.tsv \
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


def parse_tsv(path):
    """Held-out LC0-truth test set: `fen<TAB>truth_stm_cp` (truth already STM-POV,
    the LC0 score the corrective data was trained toward — NOT an SF-search oracle)."""
    out = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        fen, truth = line.split("\t")
        out.append((fen, fen.split()[1] == "w", int(truth)))
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
    ap.add_argument("--epd", action="append", default=[],
                    help="label=path EPD with d24 truth token (repeatable)")
    ap.add_argument("--tsv", action="append", default=[],
                    help="label=path  `fen<TAB>lc0_stm_cp` LC0-truth set (repeatable)")
    ap.add_argument("--coda", default="./coda")
    ap.add_argument("--dump", action="store_true",
                    help="per-position eval table")
    args = ap.parse_args()

    nets = [s.split("=", 1) for s in args.net]
    epds = [(*s.split("=", 1), parse_epd) for s in args.epd] + \
           [(*s.split("=", 1), parse_tsv) for s in args.tsv]
    if not epds:
        sys.exit("need at least one --epd or --tsv")

    for elabel, epath, parser in epds:
        positions = parser(epath)
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
