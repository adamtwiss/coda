#!/usr/bin/env python3
"""TM variation analyzer.

Parses cutechess PGN files (with `{score/depth time}` move-time comments)
and reports per-engine time-management variation metrics:

  - Per-game spend percentiles (p5/p25/p50/p75/p95)
  - Variation ratios: p95/p5 (heavy-tail), IQR/median (dispersion)
  - Spend-by-move-number profile (detects "spent it all early" pattern)
  - Side-by-side engine comparison

Output prioritises *comparable* metrics across engines so we can answer:
"Are top engines achieving more per-move spend variety than us, and is
it real per-position selection or just opening/endgame timing?"

Usage:
  python3 scripts/tm_variation_analyzer.py game.pgn [game2.pgn ...]
  python3 scripts/tm_variation_analyzer.py --label "30+0" game.pgn
  python3 scripts/tm_variation_analyzer.py --by-tc tc1.pgn tc2.pgn ...
"""

import argparse
import re
import sys
from collections import defaultdict
from statistics import mean, median

# Match a move's time comment, eg "{+0.21/15 0.51s}" or "{+0.21/15 0.51}"
# We accept comments with or without the trailing 's', and only extract
# the FINAL float-with-optional-'s' which is the spend.
TIME_RE = re.compile(r'([0-9]+\.[0-9]+)s\b')  # spend = number+'s' before word boundary (fixes 2026-05-25 bug that extracted scores)


def parse_pgn(path):
    """Yields one dict per game: {white, black, white_times, black_times}."""
    with open(path) as f:
        text = f.read()
    # Split into game blocks separated by blank lines after game body
    games = re.split(r'\n\s*\n(?=\[)', text)
    for g in games:
        if '[White ' not in g or '[Black ' not in g:
            continue
        white = re.search(r'\[White "([^"]+)"\]', g)
        black = re.search(r'\[Black "([^"]+)"\]', g)
        if not white or not black:
            continue
        # Extract body (after the headers): everything from the first
        # numbered move onward
        body_match = re.search(r'\n\n(.*)', g, re.DOTALL)
        if not body_match:
            continue
        body = body_match.group(1)
        # Strip comments-without-times so they don't confuse counting
        # Find every move's time comment, in order — white = even-indexed, black = odd-indexed
        # (cutechess emits a comment per move; index 0 = white's first, 1 = black's first, ...)
        times = [float(m.group(1)) for m in TIME_RE.finditer(body)]
        wt = times[0::2]
        bt = times[1::2]
        yield {
            'white': white.group(1),
            'black': black.group(1),
            'white_times': wt,
            'black_times': bt,
        }


def collect_per_engine(pgn_paths):
    """Returns {engine_name: [(game_idx, [move_times, ...]), ...]}."""
    engines = defaultdict(list)
    game_idx = 0
    for p in pgn_paths:
        for game in parse_pgn(p):
            engines[game['white']].append((game_idx, game['white_times']))
            engines[game['black']].append((game_idx, game['black_times']))
            game_idx += 1
    return engines


def percentiles(vs, ps):
    if not vs:
        return [0.0] * len(ps)
    s = sorted(vs)
    n = len(s)
    out = []
    for p in ps:
        idx = int(round((p / 100.0) * (n - 1)))
        out.append(s[idx])
    return out


def summarise_engine(name, games):
    """Compute variation metrics for one engine over many games.

    Per-game variation metrics are averaged across games to control for
    game-length variance.
    """
    # Per-game ratios (averaged) — within-game variety, what we care about
    p5_p95_ratios = []
    iqr_med_ratios = []
    p5_p50_ratios = []
    p50_p95_ratios = []
    all_times = []
    move_count_total = 0
    games_with_data = 0
    # Spend-by-move-bucket (1-10, 11-20, 21-30, 31-40, 41+)
    spend_by_bucket = defaultdict(list)
    for _, times in games:
        # Skip very short games (resignations) — TM patterns aren't meaningful
        if len(times) < 15:
            continue
        games_with_data += 1
        all_times.extend(times)
        move_count_total += len(times)
        p5, p25, p50, p75, p95 = percentiles(times, [5, 25, 50, 75, 95])
        if p5 > 0:
            p5_p95_ratios.append(p95 / p5)
            p5_p50_ratios.append(p50 / p5)
        if p50 > 0:
            iqr_med_ratios.append((p75 - p25) / p50)
            p50_p95_ratios.append(p95 / p50)
        for mi, t in enumerate(times, start=1):
            if mi <= 10:
                spend_by_bucket['1-10'].append(t)
            elif mi <= 20:
                spend_by_bucket['11-20'].append(t)
            elif mi <= 30:
                spend_by_bucket['21-30'].append(t)
            elif mi <= 40:
                spend_by_bucket['31-40'].append(t)
            else:
                spend_by_bucket['41+'].append(t)

    overall = percentiles(all_times, [5, 25, 50, 75, 95])
    return {
        'name': name,
        'games': games_with_data,
        'moves': move_count_total,
        # Average within-game variation ratios (NOT pooled — pooled mixes
        # across-game-length variance)
        'avg_p95_p5': mean(p5_p95_ratios) if p5_p95_ratios else 0.0,
        'avg_p50_p5': mean(p5_p50_ratios) if p5_p50_ratios else 0.0,
        'avg_p95_p50': mean(p50_p95_ratios) if p50_p95_ratios else 0.0,
        'avg_iqr_over_median': mean(iqr_med_ratios) if iqr_med_ratios else 0.0,
        # Pooled distribution (for printing the overall spend profile)
        'p5':  overall[0],
        'p25': overall[1],
        'p50': overall[2],
        'p75': overall[3],
        'p95': overall[4],
        'spend_by_bucket': {k: (median(v) if v else 0.0) for k, v in spend_by_bucket.items()},
        'spend_by_bucket_p95': {k: (percentiles(v, [95])[0] if v else 0.0) for k, v in spend_by_bucket.items()},
    }


def report(engines_summary, label=''):
    if label:
        print(f'\n========= {label} =========')
    engs = sorted(engines_summary.values(), key=lambda e: -e['avg_p95_p5'])
    print(f'\nPer-engine variation (sorted by within-game p95/p5):')
    print(f'  {"engine":<22} {"games":>6} {"moves":>6}'
          f' {"<p95/p5>":>9} {"<p95/p50>":>10} {"<p50/p5>":>9}'
          f' {"<IQR/p50>":>10}')
    print('  ' + '-' * 78)
    for e in engs:
        print(f'  {e["name"][:21]:<22} {e["games"]:>6} {e["moves"]:>6}'
              f' {e["avg_p95_p5"]:>9.2f} {e["avg_p95_p50"]:>10.2f}'
              f' {e["avg_p50_p5"]:>9.2f} {e["avg_iqr_over_median"]:>10.2f}')

    print(f'\nPooled spend distribution per engine (sec):')
    print(f'  {"engine":<22} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7}')
    print('  ' + '-' * 62)
    for e in engs:
        print(f'  {e["name"][:21]:<22} {e["p5"]:>7.2f} {e["p25"]:>7.2f}'
              f' {e["p50"]:>7.2f} {e["p75"]:>7.2f} {e["p95"]:>7.2f}')

    print(f'\nSpend by move-bucket (median sec / p95 sec):')
    print(f'  {"engine":<22} {"1-10":>14} {"11-20":>14} {"21-30":>14}'
          f' {"31-40":>14} {"41+":>14}')
    print('  ' + '-' * 110)
    for e in engs:
        buckets = ['1-10', '11-20', '21-30', '31-40', '41+']
        cells = []
        for b in buckets:
            med = e['spend_by_bucket'].get(b, 0.0)
            p95 = e['spend_by_bucket_p95'].get(b, 0.0)
            cells.append(f'{med:5.2f}/{p95:5.2f}')
        print(f'  {e["name"][:21]:<22}' + ''.join(f' {c:>14}' for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pgn', nargs='+')
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    engines_games = collect_per_engine(args.pgn)
    if not engines_games:
        print('No games parsed', file=sys.stderr)
        sys.exit(1)

    summary = {name: summarise_engine(name, games)
               for name, games in engines_games.items()}
    report(summary, args.label)


if __name__ == '__main__':
    main()
