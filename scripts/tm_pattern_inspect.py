#!/usr/bin/env python3
"""TM pattern inspector — quick visual check of per-move clock evolution.

Take a PGN with cutechess `{score/depth Ts}` comments. For each Coda variant
playing a target opponent, print:
  - Per-move spend medians by phase bucket (opening/mid/end)
  - Clock-remaining curve at key moves
  - Side-by-side compare of variants

Designed as the equivalent of inspecting lichess move-times graphs a few games
into a session. Run AFTER any TM mechanism change to confirm the change
actually changes the time-usage pattern, BEFORE committing CPU to gauntlets.

Usage:
  python3 scripts/tm_pattern_inspect.py <pgn> [--inc-ms 100] [--initial-ms 60000]
  python3 scripts/tm_pattern_inspect.py <pgn> --opponents Stockfish --variants Coda.p10g,Coda.main
"""
import argparse
import re
from collections import defaultdict
from statistics import median

SPEND_RE = re.compile(r'([0-9]+\.[0-9]+)s\b')  # Spend is number followed by 's' before word boundary


def parse_games(path):
    with open(path) as f:
        text = f.read()
    games = re.split(r'\n\s*\n(?=\[)', text)
    out = []
    for g in games:
        if '[White ' not in g or '[Black ' not in g:
            continue
        w = re.search(r'\[White "([^"]+)"\]', g)
        b = re.search(r'\[Black "([^"]+)"\]', g)
        if not w or not b:
            continue
        body = re.search(r'\n\n(.*)', g, re.DOTALL)
        if not body:
            continue
        spends = [float(m.group(1)) * 1000 for m in SPEND_RE.finditer(body.group(1))]
        term = re.search(r'\[Termination "([^"]+)"\]', g)
        result = re.search(r'\[Result "([^"]+)"\]', g)
        out.append({
            'white': w.group(1),
            'black': b.group(1),
            'w_spends': spends[0::2],
            'b_spends': spends[1::2],
            'termination': term.group(1) if term else '',
            'result': result.group(1) if result else '',
        })
    return out


def shape_report(games):
    """Pooled per-move spend distribution per engine: the TM 'spikiness'
    metric (p90/median) plus time-forfeit counts. Diagnostic for the
    spend-shape work (docs/tm_spikiness_experiment_2026-07-10.md):
    Coda baseline ~2.6 vs SF consensus ~3.4-3.7 at STC. Elo gates
    promotions — do not optimize this ratio for its own sake."""
    spends = defaultdict(list)
    n_games = defaultdict(int)
    forfeits = defaultdict(int)
    for g in games:
        spends[g['white']].extend(g['w_spends'])
        spends[g['black']].extend(g['b_spends'])
        n_games[g['white']] += 1
        n_games[g['black']] += 1
        if 'time' in g['termination'].lower():
            loser = g['black'] if g['result'] == '1-0' else \
                    g['white'] if g['result'] == '0-1' else None
            if loser:
                forfeits[loser] += 1
    print(f'  {"engine":<14}{"games":>6}{"moves":>8}{"mean":>8}{"med":>8}'
          f'{"p75":>8}{"p90":>8}{"p95":>8}{"max":>8}{"p90/med":>9}{"tforf":>6}')
    print('  ' + '-' * 91)
    for eng in sorted(spends, key=lambda e: -n_games[e]):
        s = sorted(spends[eng])
        if not s:
            continue
        q = lambda p: s[min(len(s) - 1, int(len(s) * p))] / 1000
        med = q(0.50)
        ratio = q(0.90) / med if med > 0 else float('nan')
        print(f'  {eng:<14}{n_games[eng]:>6}{len(s):>8}{sum(s)/len(s)/1000:>8.3f}'
              f'{med:>8.3f}{q(0.75):>8.3f}{q(0.90):>8.3f}{q(0.95):>8.3f}'
              f'{s[-1]/1000:>8.2f}{ratio:>9.2f}{forfeits[eng]:>6}')


def analyze(games, variants, opponents, initial_ms, inc_ms):
    per_variant_spends = defaultdict(lambda: defaultdict(list))  # variant -> move -> [spend_ms]
    per_variant_clock = defaultdict(lambda: defaultdict(list))   # variant -> move -> [clock_ms]
    games_filtered = defaultdict(int)
    for g in games:
        # Determine if this game involves a target variant vs a target opponent
        for variant in variants:
            if g['white'] == variant and g['black'] in opponents:
                eng, spends = variant, g['w_spends']
            elif g['black'] == variant and g['white'] in opponents:
                eng, spends = variant, g['b_spends']
            else:
                continue
            games_filtered[eng] += 1
            clock = initial_ms
            for i, s in enumerate(spends, start=1):
                per_variant_spends[eng][i].append(s)
                clock = max(0, clock - s + inc_ms)
                per_variant_clock[eng][i].append(clock)
    return per_variant_spends, per_variant_clock, games_filtered


def fmt_seconds(ms):
    return f"{ms/1000:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pgn')
    ap.add_argument('--variants', default='', help='comma-separated variant names (default: auto-detect Coda.*)')
    ap.add_argument('--opponents', default='', help='comma-separated opponent names (default: all non-variant)')
    ap.add_argument('--inc-ms', type=int, default=100, help='increment in ms (default 100 = 0.1s)')
    ap.add_argument('--initial-ms', type=int, default=60000, help='initial clock in ms (default 60000)')
    ap.add_argument('--shape', action='store_true',
                    help='print pooled spend-shape quantiles (p90/med spikiness metric) for ALL engines and exit')
    args = ap.parse_args()

    games = parse_games(args.pgn)
    if not games:
        print('No games parsed')
        return

    if args.shape:
        print(f'Spend shape over {len(games)} games ({args.pgn}):')
        shape_report(games)
        return

    # Auto-detect variants if not specified
    all_engines = set()
    for g in games:
        all_engines.add(g['white'])
        all_engines.add(g['black'])
    if args.variants:
        variants = args.variants.split(',')
    else:
        variants = [e for e in sorted(all_engines) if e.startswith('Coda')]
    if args.opponents:
        opponents = set(args.opponents.split(','))
    else:
        opponents = all_engines - set(variants)

    print(f'Variants: {variants}')
    print(f'Opponents: {sorted(opponents)}')
    print(f'TC: initial={args.initial_ms/1000:.1f}s + {args.inc_ms/1000:.2f}s')
    print()

    spends, clocks, n_games = analyze(games, variants, opponents, args.initial_ms, args.inc_ms)

    print(f'Games per variant vs target opponents: {dict(n_games)}\n')

    # Per-move-range spend medians
    ranges = [('open 1-5', 1, 5), ('open 6-10', 6, 10), ('early-mid 11-15', 11, 15),
              ('mid 16-20', 16, 20), ('mid 21-25', 21, 25), ('late-mid 26-30', 26, 30),
              ('end 31-40', 31, 40), ('end 41-60', 41, 60), ('end 61+', 61, 200)]
    print(f'  {"range":<18}' + ''.join(f'{v[:13]:>14}' for v in variants))
    print('  ' + '-' * (20 + 14 * len(variants)))
    for r_name, lo, hi in ranges:
        cells = []
        for v in variants:
            vals = []
            for m in range(lo, hi + 1):
                vals.extend(spends[v].get(m, []))
            if vals:
                cells.append(f'{median(vals)/1000:>10.3f}s')
            else:
                cells.append(f'{"--":>11}')
        print(f'  {r_name:<18}' + ''.join(f'{c:>14}' for c in cells))

    # Clock remaining curve
    print()
    print(f'  CLOCK REMAINING (median seconds):')
    print(f'  {"at move":<10}' + ''.join(f'{v[:13]:>14}' for v in variants))
    print('  ' + '-' * (12 + 14 * len(variants)))
    for m in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80]:
        cells = []
        for v in variants:
            vals = clocks[v].get(m, [])
            if vals:
                cells.append(f'{median(vals)/1000:>8.1f}s n={len(vals):<3}')
            else:
                cells.append('  --')
        print(f'  {m:<10}' + ''.join(f'{c:>14}' for c in cells))


if __name__ == '__main__':
    main()
