#!/usr/bin/env python3
"""Analyze time management profiles from PGN files or Lichess API.

Usage:
    python3 tm_profile.py game.pgn              # Analyze PGN file
    python3 tm_profile.py --lichess codabot 20   # Last 20 Lichess games

Reports per-game and aggregate time profiles:
- Phase averages (opening, middlegame, late, endgame)
- SD of time per move (higher = more adaptive)
- Time remaining at move 40
- Buffer at game end
"""

import sys
import re
import json

def parse_pgn_clocks(pgn_text):
    """Parse games from PGN text, extracting clock times.

    Supports two formats:
    - Lichess: [%clk 0:03:00]
    - Cutechess: {+1.17/22 8.3s} or {book} (time in seconds with 's' suffix)
    """
    games = []
    current_headers = {}
    current_moves = ""

    lines = pgn_text.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('['):
            m = re.match(r'\[(\w+)\s+"(.*)"\]', line)
            if m:
                current_headers[m.group(1)] = m.group(2)
        elif line and not line.startswith('['):
            current_moves += " " + line

        # End of game: empty line after moves, or last line
        if (not line and current_moves) or i == len(lines) - 1:
            if current_headers and current_moves:
                white_clocks = []
                black_clocks = []

                # Try Lichess format first: [%clk H:MM:SS]
                clk_matches = re.findall(r'\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]', current_moves)
                if clk_matches:
                    times = [int(h)*3600 + int(m)*60 + float(s) for h, m, s in clk_matches]
                    white_clocks = times[0::2]
                    black_clocks = times[1::2]
                else:
                    # Try cutechess format: {eval/depth Xs} — extract time per move
                    # Each annotation alternates white/black
                    annotations = re.findall(r'\{([^}]*)\}', current_moves)
                    tc_str = current_headers.get('TimeControl', '180+2')
                    tc_parts = tc_str.split('+')
                    base = int(tc_parts[0])
                    inc = int(tc_parts[1]) if len(tc_parts) > 1 else 0

                    # Build clock times from per-move times
                    w_time = base
                    b_time = base
                    is_white = True
                    for ann in annotations:
                        # Extract time: "8.3s" or "book" or "0.00s"
                        time_match = re.search(r'([\d.]+)s', ann)
                        if time_match:
                            used = float(time_match.group(1))
                        elif 'book' in ann:
                            used = 0.0
                        else:
                            continue

                        if is_white:
                            white_clocks.append(w_time)
                            w_time = w_time - used + inc
                        else:
                            black_clocks.append(b_time)
                            b_time = b_time - used + inc
                        is_white = not is_white

                if white_clocks and black_clocks:
                    games.append({
                        'headers': current_headers,
                        'white_clocks': white_clocks,
                        'black_clocks': black_clocks,
                    })
            current_headers = {}
            current_moves = ""

    return games


def parse_ndjson_clocks(ndjson_text, bot_name):
    """Parse games from Lichess NDJSON format."""
    games = []
    for line in ndjson_text.split('\n'):
        if not line.strip():
            continue
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue

        clocks = g.get('clocks', [])
        if not clocks:
            continue

        players = g['players']
        is_white = players['white']['user']['id'] == bot_name.lower()
        opp = players['black' if is_white else 'white']['user']['name']
        opp_rating = players['black' if is_white else 'white'].get('rating', 0)

        winner = g.get('winner')
        if winner == 'white':
            result = 'W' if is_white else 'L'
        elif winner == 'black':
            result = 'L' if is_white else 'W'
        else:
            result = 'D'

        tc = g.get('clock', {})
        base = tc.get('initial', 0)
        inc = tc.get('increment', 0)

        # Clocks in centiseconds
        our_clocks = [clocks[i] / 100.0 for i in range(0 if is_white else 1, len(clocks), 2)]
        opp_clocks = [clocks[i] / 100.0 for i in range(1 if is_white else 0, len(clocks), 2)]

        games.append({
            'opponent': opp,
            'opp_rating': opp_rating,
            'result': result,
            'base': base,
            'inc': inc,
            'our_clocks': our_clocks,
            'opp_clocks': opp_clocks,
            'is_white': is_white,
        })

    return games


def analyze_game(our_clocks, opp_clocks, base, inc, label=""):
    """Analyze time management for one game."""
    n = len(our_clocks)
    if n < 5:
        return None

    # Calculate time used per move
    used = []
    for i in range(1, n):
        time_used = our_clocks[i-1] - our_clocks[i] + inc
        used.append(max(0, time_used))

    opp_used = []
    for i in range(1, min(len(opp_clocks), n)):
        time_used = opp_clocks[i-1] - opp_clocks[i] + inc
        opp_used.append(max(0, time_used))

    # Phase breakdown
    def phase_avg(used_list, start, end):
        phase = used_list[max(0,start-1):min(len(used_list),end)]
        if not phase:
            return 0, 0
        avg = sum(phase) / len(phase)
        if len(phase) > 1:
            sd = (sum((x - avg)**2 for x in phase) / len(phase)) ** 0.5
        else:
            sd = 0
        return avg, sd

    opening_avg, opening_sd = phase_avg(used, 1, 10)
    middle_avg, middle_sd = phase_avg(used, 11, 30)
    late_avg, late_sd = phase_avg(used, 31, 45)
    endgame_avg, endgame_sd = phase_avg(used, 46, 999)

    overall_sd = 0
    if len(used) > 1:
        overall_avg = sum(used) / len(used)
        overall_sd = (sum((x - overall_avg)**2 for x in used) / len(used)) ** 0.5

    # Time remaining at key points
    remaining_40 = our_clocks[min(39, n-1)] if n > 39 else our_clocks[-1]
    remaining_end = our_clocks[-1]

    # Total available time
    total_available = base + n * inc
    total_used = total_available - remaining_end
    usage_pct = total_used / total_available * 100 if total_available > 0 else 0

    # Opponent comparison
    opp_remaining_40 = opp_clocks[min(39, len(opp_clocks)-1)] if len(opp_clocks) > 39 else (opp_clocks[-1] if opp_clocks else 0)
    opp_remaining_end = opp_clocks[-1] if opp_clocks else 0

    return {
        'label': label,
        'moves': n,
        'opening_avg': opening_avg,
        'middle_avg': middle_avg,
        'late_avg': late_avg,
        'endgame_avg': endgame_avg,
        'overall_sd': overall_sd,
        'middle_sd': middle_sd,
        'remaining_40': remaining_40,
        'remaining_end': remaining_end,
        'opp_remaining_40': opp_remaining_40,
        'opp_remaining_end': opp_remaining_end,
        'usage_pct': usage_pct,
        'total_available': total_available,
        'base': base,
        'inc': inc,
    }


def print_analysis(results):
    """Print formatted analysis."""
    if not results:
        print("No games to analyze")
        return

    print(f"\n{'='*80}")
    print(f"TIME MANAGEMENT PROFILE ({len(results)} games)")
    print(f"{'='*80}")

    # Group by TC
    by_tc = {}
    for r in results:
        tc = f"{r['base']}+{r['inc']}"
        by_tc.setdefault(tc, []).append(r)

    for tc, games in sorted(by_tc.items()):
        print(f"\n--- TC: {tc} ({len(games)} games) ---")
        print(f"{'Game':<30} {'Moves':>5} {'Open':>6} {'Mid':>6} {'Late':>6} {'End':>6} {'SD':>6} {'@40':>7} {'Final':>7} {'Opp@40':>7} {'Used%':>6}")
        print("-" * 115)

        totals = {'opening': [], 'middle': [], 'late': [], 'endgame': [], 'sd': [],
                  'r40': [], 'rend': [], 'or40': [], 'usage': []}

        for r in games:
            print(f"{r['label']:<30} {r['moves']:>5} {r['opening_avg']:>5.1f}s {r['middle_avg']:>5.1f}s "
                  f"{r['late_avg']:>5.1f}s {r['endgame_avg']:>5.1f}s {r['overall_sd']:>5.1f}s "
                  f"{r['remaining_40']:>6.0f}s {r['remaining_end']:>6.0f}s {r['opp_remaining_40']:>6.0f}s "
                  f"{r['usage_pct']:>5.1f}%")

            totals['opening'].append(r['opening_avg'])
            totals['middle'].append(r['middle_avg'])
            totals['late'].append(r['late_avg'])
            totals['endgame'].append(r['endgame_avg'])
            totals['sd'].append(r['overall_sd'])
            totals['r40'].append(r['remaining_40'])
            totals['rend'].append(r['remaining_end'])
            totals['or40'].append(r['opp_remaining_40'])
            totals['usage'].append(r['usage_pct'])

        if len(games) > 1:
            def avg(lst): return sum(lst)/len(lst) if lst else 0
            print("-" * 115)
            print(f"{'AVERAGE':<30} {'':>5} {avg(totals['opening']):>5.1f}s {avg(totals['middle']):>5.1f}s "
                  f"{avg(totals['late']):>5.1f}s {avg(totals['endgame']):>5.1f}s {avg(totals['sd']):>5.1f}s "
                  f"{avg(totals['r40']):>6.0f}s {avg(totals['rend']):>6.0f}s {avg(totals['or40']):>6.0f}s "
                  f"{avg(totals['usage']):>5.1f}%")

        # Target comparison
        inc = games[0]['inc']
        base = games[0]['base']
        if inc > 0:
            target_usage = 90
            target_r40 = base * 0.15
        else:
            target_usage = 77
            target_r40 = base * 0.30

        print(f"\n  Targets: usage {target_usage}%, time@40 <{target_r40:.0f}s")
        print(f"  Current: usage {avg(totals['usage']):.1f}%, time@40 {avg(totals['r40']):.0f}s")
        gap = target_usage - avg(totals['usage'])
        print(f"  Gap: {'+' if gap > 0 else ''}{gap:.1f}% usage to target")


def main():
    if len(sys.argv) < 2:
        print("Usage: tm_profile.py <pgn_file> | --lichess <username> [max_games]")
        sys.exit(1)

    if sys.argv[1] == '--lichess':
        import subprocess
        username = sys.argv[2] if len(sys.argv) > 2 else 'codabot'
        max_games = sys.argv[3] if len(sys.argv) > 3 else '20'
        r = subprocess.run([
            'curl', '-s',
            f'https://lichess.org/api/games/user/{username}?clocks=true&opening=true&max={max_games}',
            '-H', 'Accept: application/x-ndjson'
        ], capture_output=True, text=True, timeout=30)

        games = parse_ndjson_clocks(r.stdout, username)
        results = []
        for g in games:
            label = f"vs {g['opponent']} ({g['opp_rating']}) {g['result']}"
            r = analyze_game(g['our_clocks'], g['opp_clocks'], g['base'], g['inc'], label)
            if r:
                results.append(r)
        print_analysis(results)

    else:
        # PGN file
        pgn_file = sys.argv[1]
        engine_name = sys.argv[2] if len(sys.argv) > 2 else None

        with open(pgn_file) as f:
            pgn_text = f.read()

        games = parse_pgn_clocks(pgn_text)
        results = []
        for g in games:
            headers = g['headers']
            tc = headers.get('TimeControl', '180+2')
            tc_parts = tc.split('+')
            base = int(tc_parts[0])
            inc = int(tc_parts[1]) if len(tc_parts) > 1 else 0

            white = headers.get('White', '?')
            black = headers.get('Black', '?')
            result = headers.get('Result', '?')

            # Determine which side is "ours"
            if engine_name:
                is_ours_white = engine_name.lower() in white.lower()
            else:
                is_ours_white = True  # Default to white

            our_clocks = g['white_clocks'] if is_ours_white else g['black_clocks']
            opp_clocks = g['black_clocks'] if is_ours_white else g['white_clocks']

            opp_name = black if is_ours_white else white
            label = f"vs {opp_name} {result}"

            r = analyze_game(our_clocks, opp_clocks, base, inc, label)
            if r:
                results.append(r)

        print_analysis(results)


if __name__ == '__main__':
    main()
