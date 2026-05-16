"""Cross-tune SPSA variance audit — identify loose knobs.

For each known recent full-sweep tune, pulls the digest from OB and
builds a per-parameter summary:

  - mean_abs_pct: average |movement %| across tunes — small = either
    well-converged near start, or flat gradient (need code analysis
    to distinguish).
  - sign_consistency: fraction of tunes moving in the SAME direction
    as the majority. 100% = consistent pull; ~50% = drifting around
    a stable point (or pure noise).
  - normalized_range_use: where in [min, max] do the final values
    cluster? If clustered, SPSA agrees on the location. If spread,
    SPSA disagrees.
  - mean_abs_normalized_movement: % of range traversed, accounting
    for the parameter's allowable span.

Output is a markdown-ish table sorted by ambiguity.
"""
import os
import sys
import csv
import urllib.request
import urllib.error
from collections import defaultdict
from statistics import mean, stdev

SERVER = os.environ.get('OB_SERVER', 'https://ob.atwiss.com')
USER = os.environ.get('OB_USER', 'claude')
PASSWORD = os.environ.get('OPENBENCH_PASSWORD', '')

# Recent full-sweep tunes on or near current trunk (descending recency).
# Older tunes are on different trunks; mixing all of them anyway is
# informative for cross-trunk variance — params that drift everywhere
# are robust loose-knob signals.
TUNE_IDS = [1250, 1247, 1228, 1117, 1070, 1071, 928, 882, 871, 870, 855]


def fetch_digest(tune_id):
    url = f"{SERVER}/api/spsa/{tune_id}/digest/"
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, SERVER, USER, PASSWORD)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler)
    try:
        with opener.open(url) as r:
            text = r.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print(f"  [WARN] tune #{tune_id}: HTTP {e.code}", file=sys.stderr)
        return None

    rows = list(csv.DictReader(text.splitlines()))
    params = {}
    for row in rows:
        name = row['Name']
        try:
            curr = float(row['Curr'])
            start = float(row['Start'])
            lo = float(row['Min'])
            hi = float(row['Max'])
        except (KeyError, ValueError):
            continue
        params[name] = (curr, start, lo, hi)
    return params


def main():
    if not PASSWORD:
        print("ERROR: OPENBENCH_PASSWORD env var required", file=sys.stderr)
        sys.exit(1)

    # collect all params across all tunes
    per_param = defaultdict(list)  # name -> list of (curr, start, lo, hi, tune_id)
    for tid in TUNE_IDS:
        print(f"Fetching #{tid}...", file=sys.stderr)
        d = fetch_digest(tid)
        if d is None:
            continue
        for name, vals in d.items():
            per_param[name].append((*vals, tid))

    print(f"\nParams seen: {len(per_param)}", file=sys.stderr)
    print(f"Tunes processed: {len({t for entries in per_param.values() for *_, t in entries})}", file=sys.stderr)

    summaries = []
    for name, entries in per_param.items():
        if len(entries) < 3:
            continue
        movements = []          # % movement from each tune's start
        normalized = []         # % of allowable range traversed
        signs = []              # +1 / -1 / 0
        currs = []              # final values
        for curr, start, lo, hi, _tid in entries:
            if start == 0:
                pct = 0.0
            else:
                pct = (curr - start) / abs(start) * 100.0
            movements.append(pct)
            currs.append(curr)
            span = max(hi - lo, 1e-9)
            normalized.append((curr - start) / span * 100.0)
            if abs(pct) < 0.5:
                signs.append(0)
            elif pct > 0:
                signs.append(1)
            else:
                signs.append(-1)

        mean_abs_pct = mean(abs(p) for p in movements)
        n = len(signs)
        non_zero_signs = [s for s in signs if s != 0]
        if non_zero_signs:
            pos = sum(1 for s in non_zero_signs if s > 0)
            sign_consistency = max(pos, len(non_zero_signs) - pos) / len(non_zero_signs)
        else:
            sign_consistency = 1.0  # everyone left it alone — clearly stable
        mean_abs_normalized = mean(abs(n) for n in normalized)
        curr_stdev_pct = (stdev(currs) / mean(currs) * 100.0) if len(currs) > 1 and mean(currs) != 0 else 0.0

        summaries.append({
            'name': name,
            'n_tunes': n,
            'mean_abs_pct': mean_abs_pct,
            'sign_consistency': sign_consistency,
            'mean_abs_normalized': mean_abs_normalized,
            'curr_stdev_pct': curr_stdev_pct,
            'currs': currs,
        })

    # Class hypotheses:
    #   STABLE-CONVERGED: low mean_abs_pct (<5%), low normalized (<5%), high sign_consistency (>=80%)
    #                     OR low curr_stdev_pct (<8%) — values cluster
    #   CONSISTENT-PULL: moderate-to-high mean_abs_pct (>=8%), high sign_consistency (>=80%)
    #                     — SPSA wants this moved
    #   DRIFTING-LOOSE: mean_abs_normalized < 8%, sign_consistency < 65% — moves around but small
    #   DISAGREEING: mean_abs_pct >= 8%, sign_consistency < 65% — moves a lot, both ways

    def classify(s):
        if s['mean_abs_normalized'] < 5 and s['curr_stdev_pct'] < 8:
            return 'STABLE-CONVERGED'
        if s['sign_consistency'] >= 0.8 and s['mean_abs_pct'] >= 8:
            return 'CONSISTENT-PULL'
        if s['mean_abs_normalized'] < 8 and s['sign_consistency'] < 0.65:
            return 'DRIFTING-LOOSE'
        if s['mean_abs_pct'] >= 8 and s['sign_consistency'] < 0.65:
            return 'DISAGREEING'
        return 'MIXED'

    for s in summaries:
        s['class'] = classify(s)

    # sort: drifting-loose first (highest audit priority), then disagreeing, then by ambiguity
    class_priority = {
        'DRIFTING-LOOSE': 0,
        'DISAGREEING': 1,
        'MIXED': 2,
        'CONSISTENT-PULL': 3,
        'STABLE-CONVERGED': 4,
    }
    summaries.sort(key=lambda s: (class_priority[s['class']], -s['mean_abs_pct']))

    print(f"\n{'PARAM':<32} {'N':>2}  {'|mean%|':>8}  {'sign%':>6}  {'|norm%|':>8}  {'curr_sd%':>8}  CLASS")
    print('-' * 100)
    for s in summaries:
        print(f"{s['name']:<32} {s['n_tunes']:>2}  "
              f"{s['mean_abs_pct']:>7.2f}%  "
              f"{s['sign_consistency']*100:>5.0f}%  "
              f"{s['mean_abs_normalized']:>7.2f}%  "
              f"{s['curr_stdev_pct']:>7.2f}%  "
              f"{s['class']}")

    print("\nCounts:")
    counts = defaultdict(int)
    for s in summaries:
        counts[s['class']] += 1
    for cls in ['DRIFTING-LOOSE', 'DISAGREEING', 'MIXED', 'CONSISTENT-PULL', 'STABLE-CONVERGED']:
        print(f"  {cls:<20}  {counts[cls]}")


if __name__ == '__main__':
    main()
