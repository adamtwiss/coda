#!/usr/bin/env python3
"""Generate an SPSA parameter file from the tunables!(...) block in src/search.rs.

Keeps SPSA tunes anchored to the current code defaults rather than a stale
static file. Intended usage:

    scripts/gen_tune_params.py > /tmp/params.txt
    scripts/ob_tune.py <branch> --params-file /tmp/params.txt --iterations 2500

Or inline:

    scripts/ob_tune.py <branch> \
        --params "$(scripts/gen_tune_params.py)" --iterations 2500

Each line: NAME, int, default, min, max, c_end, r_end
- c_end = max(0.5, (max - min) * 0.05) — 5% of the parameter range, floored.
- r_end = 0.002 (standard SPSA constant; override with --r-end).

Run from the repo root (or pass --file).
"""

import argparse
import re
import sys

TUNABLE_RE = re.compile(
    r'\(\s*([A-Z_][A-Z_0-9]*)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
)


def parse_tunables(path: str):
    with open(path) as f:
        text = f.read()
    m = re.search(r'tunables!\s*\(\s*(.*?)\s*\)\s*;', text, re.DOTALL)
    if not m:
        sys.exit(f'error: no tunables!(...) block found in {path}')
    body = m.group(1)
    # Strip // and /* */ comments so they don't feed the tuple regex.
    body = re.sub(r'//[^\n]*', '', body)
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.DOTALL)
    return TUNABLE_RE.findall(body)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--file', default='src/search.rs',
                   help='Path to source file containing tunables!(...) (default: src/search.rs)')
    p.add_argument('--r-end', type=float, default=0.002,
                   help='SPSA r_end constant (default: 0.002)')
    p.add_argument('--c-end-frac', type=float, default=0.05,
                   help='c_end as fraction of (max-min) (default: 0.05 = 5%%)')
    p.add_argument('--c-end-floor', type=float, default=0.5,
                   help='Minimum c_end regardless of range (default: 0.5)')
    args = p.parse_args()

    tunables = parse_tunables(args.file)
    if not tunables:
        sys.exit(f'error: tunables!(...) block in {args.file} parsed to zero entries')

    for name, default, lo, hi in tunables:
        lo_i, hi_i = int(lo), int(hi)
        if hi_i <= lo_i:
            sys.exit(f'error: {name} has non-positive range ({lo_i}..{hi_i})')
        c_end = max(args.c_end_floor, (hi_i - lo_i) * args.c_end_frac)
        print(f'{name}, int, {default}, {lo}, {hi}, {c_end:g}, {args.r_end}')


if __name__ == '__main__':
    main()
