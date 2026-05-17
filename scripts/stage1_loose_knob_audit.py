#!/usr/bin/env python3
"""Stage 1 loose-knob audit: perturb each candidate at default ± 3×c_end,
measure bench + bench-stats invariance.

Output: per-candidate × {low, high} delta on key search-behavior metrics.
A candidate whose bench-stats are all near-invariant under ±3×c_end is a
strong "noise — promote to non-core" signal. A candidate that moves
stats meaningfully is either at-optimum or load-bearing (keep in core).

Run from coda root after `make`.
"""

import re
import subprocess
import sys


# (name, default, c_end, range_lo, range_hi) — values from current main src/search.rs.
CANDIDATES = [
    ('DEXT_MARGIN_QUIET',     4,     4.0,    0,    100),
    ('NMP_UNDEFENDED_MAX_10X',17,   10.0,    0,     50),
    ('LMP_BASE',              9,    2.0,    1,     15),
    ('FUT_THREATS_MARGIN',   23,   10.0,    0,    200),
    ('QS_SEE_THRESHOLD',    -26,   10.0, -200,      0),
    ('FUT_BASE',             23,    9.0,   20,    200),
    ('HIST_PRUNE_MULT',   10410, 2475.0,  500,  50000),
    ('LMR_THREAT_DIV_10X',   38,   15.0,   10,     50),
    ('NMP_DEPTH_DIV_10X',    55,   15.0,   10,     60),
    ('NMP_EVAL_MAX_10X',     24,    5.0,   10,     60),
]


def parse_stats(output):
    """Pull key metrics from bench output. Returns dict."""
    s = {}
    for key, pat in [
        ('nodes',     r'Total nodes:\s*(\d+)'),
        ('ebf',       r'EBF \(depth 5\+\):\s*(\d+\.\d+)'),
        ('avg_pos',   r'avg cutoff pos (\d+\.\d+)'),
        ('avg_pos2',  r'avg pos² (\d+\.\d+)'),
        ('first_mv',  r'first-move (\d+\.\d+)%'),
        ('tt_cut',    r'TT cutoffs:\s+([\d.]+)/Kn'),
        ('rfp_cut',   r'RFP cutoffs:\s+([\d.]+)/Kn'),
        ('lmr_searches', r'LMR searches:\s+([\d.]+)/Kn'),
        ('lmp_prune', r'LMP prunes:\s+([\d.]+)/Kn'),
        ('hist_prune',r'Hist prune:\s+([\d.]+)/Kn'),
        ('see_prune', r'SEE prune:\s+([\d.]+)/Kn'),
        ('fut',       r'Futility:\s+([\d.]+)/Kn'),
        ('qs_pct',    r'QS nodes:\s+([\d.]+)%'),
    ]:
        m = re.search(pat, output)
        if m:
            s[key] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
    return s


def run_bench(set_args=None, depth=12):
    """Invoke ./coda bench with optional --set NAME=VALUE overrides."""
    cmd = ['./coda', 'bench', str(depth)]
    for kv in (set_args or []):
        cmd.extend(['--set', kv])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return res.stdout + res.stderr


def pct(new, base):
    if base == 0 or base is None or new is None:
        return 0.0
    return 100.0 * (new - base) / base


def fmt_pct(v, threshold_strong=2.0, threshold_weak=0.5):
    """Format % delta with markers — strong>thresh, weak<thresh."""
    if abs(v) >= threshold_strong:
        return f'**{v:+.1f}%**'
    if abs(v) >= threshold_weak:
        return f'{v:+.1f}%'
    return f'{v:+.2f}%'


def classify(deltas):
    """Decide: is this param invariant under perturbation?
    deltas: list of (metric_name, delta_pct) across both sides.

    Returns: 'INVARIANT' / 'WEAK' / 'STRONG' classification.
    """
    max_abs = max(abs(d) for _, d in deltas)
    n_strong = sum(1 for _, d in deltas if abs(d) >= 2.0)
    n_weak = sum(1 for _, d in deltas if abs(d) >= 0.5)
    if max_abs < 0.5:
        return 'INVARIANT'  # truly inert at ±3×c_end → loose-knob CANDIDATE
    if n_strong == 0 and n_weak <= 2:
        return 'WEAK'  # small response, marginal
    return 'STRONG'  # clear response — at-optimum or load-bearing


def main():
    print('=== Stage 1: Loose-Knob Perturbation Audit ===\n')
    print('Running baseline (no overrides)...')
    base_out = run_bench()
    base = parse_stats(base_out)
    print(f'  baseline: nodes={base["nodes"]:,}  ebf={base["ebf"]}  '
          f'first-mv={base["first_mv"]:.1f}%  tt_cut={base["tt_cut"]:.1f}/Kn  '
          f'lmr={base["lmr_searches"]:.1f}/Kn  fut={base["fut"]:.1f}/Kn\n')

    results = []
    for name, default, c_end, lo, hi in CANDIDATES:
        perturb = 3 * c_end
        low_val = max(lo, int(round(default - perturb)))
        high_val = min(hi, int(round(default + perturb)))

        print(f'[{name}] default={default}  ±3×c_end=[{low_val}, {high_val}]')

        low_stats = parse_stats(run_bench([f'{name}={low_val}']))
        high_stats = parse_stats(run_bench([f'{name}={high_val}']))

        deltas = []
        for key in ['nodes', 'ebf', 'first_mv', 'avg_pos2', 'tt_cut',
                    'rfp_cut', 'lmr_searches', 'hist_prune', 'fut', 'qs_pct']:
            d_lo = pct(low_stats.get(key), base.get(key))
            d_hi = pct(high_stats.get(key), base.get(key))
            deltas.append((f'{key}_lo', d_lo))
            deltas.append((f'{key}_hi', d_hi))

        cls = classify(deltas)
        # Pull the most informative subset for compact print
        nodes_lo = pct(low_stats.get('nodes'), base.get('nodes'))
        nodes_hi = pct(high_stats.get('nodes'), base.get('nodes'))
        tt_lo = pct(low_stats.get('tt_cut'), base.get('tt_cut'))
        tt_hi = pct(high_stats.get('tt_cut'), base.get('tt_cut'))
        lmr_lo = pct(low_stats.get('lmr_searches'), base.get('lmr_searches'))
        lmr_hi = pct(high_stats.get('lmr_searches'), base.get('lmr_searches'))
        fut_lo = pct(low_stats.get('fut'), base.get('fut'))
        fut_hi = pct(high_stats.get('fut'), base.get('fut'))
        fm_lo = (low_stats.get('first_mv', 0) - base.get('first_mv', 0))
        fm_hi = (high_stats.get('first_mv', 0) - base.get('first_mv', 0))

        print(f'    nodes  : low {nodes_lo:+.1f}%   high {nodes_hi:+.1f}%')
        print(f'    tt_cut : low {tt_lo:+.1f}%     high {tt_hi:+.1f}%')
        print(f'    lmr    : low {lmr_lo:+.1f}%     high {lmr_hi:+.1f}%')
        print(f'    fut    : low {fut_lo:+.1f}%     high {fut_hi:+.1f}%')
        print(f'    first  : low {fm_lo:+.2f}pp   high {fm_hi:+.2f}pp')
        print(f'    => {cls}\n')

        results.append((name, cls, nodes_lo, nodes_hi, tt_lo, tt_hi,
                        lmr_lo, lmr_hi, fut_lo, fut_hi, fm_lo, fm_hi))

    print('\n=== Summary ===')
    print(f'{"PARAM":<28} {"CLASS":<10} {"nodes Δ%":>20} {"tt Δ%":>16} {"lmr Δ%":>16}')
    print('-' * 100)
    for name, cls, n_lo, n_hi, tt_lo, tt_hi, lmr_lo, lmr_hi, *_ in results:
        print(f'{name:<28} {cls:<10} {n_lo:>+9.1f}/{n_hi:>+6.1f}  '
              f'{tt_lo:>+6.1f}/{tt_hi:>+5.1f}  {lmr_lo:>+6.1f}/{lmr_hi:>+5.1f}')

    print('\nINVARIANT  -> stage 2 SPRT candidate (likely loose, freeze at default)')
    print('WEAK       -> ambiguous, mild response; case by case')
    print('STRONG     -> at concave optimum OR load-bearing; KEEP in core')


if __name__ == '__main__':
    main()
