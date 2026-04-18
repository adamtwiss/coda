#!/usr/bin/env python3
"""Score time management quality from a PGN file.

Works with both cutechess-cli output (has eval/depth/time in comments)
and lichess PGN exports (has [%clk] tags).

For each player we care about, computes:
- time_used_pct: % of total budget (start + all increments) actually used
- min_clock_pct: lowest clock seen at any point, as % of starting clock
- final_clock_pct: clock at end of game, as % of starting clock
- max_move_pct: biggest single-move time usage, as % of clock before that move
- short_move_frac: % of moves with <25% of expected-per-move time budget
  (proxy for "stockpiling via instant emits")
- forfeit: did the game end on time?

A game is flagged as:
- STOCKPILE if final_clock_pct > 40 AND short_move_frac > 0.3
- TIME_TROUBLE if min_clock_pct < 10
- CLEAN otherwise
- FORFEIT as a separate boolean

Usage:
  scripts/tm_score.py --pgn games.pgn --who Coda
  scripts/tm_score.py --pgn lichess.pgn --who coda_bot --format lichess
"""
import argparse, re, sys
from pathlib import Path
import chess, chess.pgn


def parse_tc(tc_str):
    """Parse "180+2" → (180000ms, 2000ms)."""
    m = re.match(r"(\d+)(?:\+(\d+))?", tc_str)
    if not m: return (None, None)
    base = int(m.group(1)) * 1000
    inc = int(m.group(2) or 0) * 1000
    return (base, inc)


def cutechess_think_time_ms(comment):
    """Cutechess format: '+0.48/17 1.5s' → 1500ms."""
    m = re.search(r"([+-]?\d+\.\d+|[+-]?\d+|mate)/(\d+)\s+([\d.]+)s", comment or "")
    if m:
        return int(float(m.group(3)) * 1000)
    return None


def lichess_clk_to_ms(clk_str):
    """'[%clk 0:01:38]' → 98000ms."""
    m = re.match(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", clk_str or "")
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (h * 3600 + mi * 60 + s) * 1000
    return None


def extract_think_times_lichess(moves, player_color, base_ms, inc_ms):
    """From lichess PGN, reconstruct per-move think times from clock deltas."""
    # Walk through player's moves only, subtract consecutive clocks.
    times = []
    clocks = [base_ms]  # before move 1
    prev_player_clk = base_ms
    for i, node in enumerate(moves):
        if i % 2 != (0 if player_color == chess.WHITE else 1):
            continue
        m = re.search(r"\[%clk ([\d:.]+)\]", node.comment or "")
        if not m:
            times.append(None)
            continue
        clk_after = lichess_clk_to_ms(m.group(1))
        if clk_after is None:
            times.append(None)
            continue
        # Think time = prev_clk + inc - clk_after (inc granted after move)
        think = prev_player_clk + inc_ms - clk_after
        times.append(max(0, think))
        clocks.append(clk_after)
        prev_player_clk = clk_after
    return times, clocks


def extract_think_times_cutechess(moves, player_color):
    """From cutechess PGN, read think time directly from each move comment."""
    times = []
    for i, node in enumerate(moves):
        if i % 2 != (0 if player_color == chess.WHITE else 1):
            continue
        t = cutechess_think_time_ms(node.comment)
        times.append(t)
    return times


def reconstruct_clock(times, base_ms, inc_ms):
    """Given per-move think times, simulate clock progression."""
    clock = [base_ms]
    for t in times:
        if t is None: t = 0
        new = clock[-1] - t + inc_ms
        clock.append(max(0, new))
    return clock


def analyse_game(game, who, fmt):
    w = game.headers.get("White", "?")
    b = game.headers.get("Black", "?")
    tc = game.headers.get("TimeControl", "0+0")
    term = game.headers.get("Termination", "")
    result = game.headers.get("Result", "*")

    if who != w and who != b: return None
    color = chess.WHITE if who == w else chess.BLACK

    base_ms, inc_ms = parse_tc(tc)
    if base_ms is None: return None

    moves = list(game.mainline())
    if fmt == "lichess":
        times, _ = extract_think_times_lichess(moves, color, base_ms, inc_ms)
    else:
        times = extract_think_times_cutechess(moves, color)
        # drop trailing Nones
        while times and times[-1] is None: times.pop()

    if not times: return None

    clock = reconstruct_clock(times, base_ms, inc_ms)
    # clock[i] = clock BEFORE move i; clock[-1] = final clock
    total_thought = sum(t or 0 for t in times)
    total_budget = base_ms + len([t for t in times if t is not None]) * inc_ms
    n_moves = len(times)
    avg_per_move_budget = total_budget / max(1, n_moves)

    # Clock stats
    min_clock_pct = 100 * min(clock) / base_ms
    final_clock_pct = 100 * clock[-1] / base_ms
    time_used_pct = 100 * total_thought / total_budget

    # Per-move stats
    move_pcts = []
    short_moves = 0
    for i, t in enumerate(times):
        if t is None: continue
        clk_before = clock[i]
        if clk_before > 0:
            move_pcts.append(100 * t / clk_before)
        # "Short move" heuristic: used <25% of per-move average budget
        if t < avg_per_move_budget * 0.25:
            short_moves += 1
    max_move_pct = max(move_pcts) if move_pcts else 0
    short_move_frac = short_moves / max(1, n_moves)

    forfeit = "time forfeit" in term.lower() or "time" in term.lower() and "forfeit" in term.lower()
    coda_lost = (color == chess.WHITE and result == "0-1") or \
                (color == chess.BLACK and result == "1-0")
    forfeit_on_us = forfeit and coda_lost

    # Classification
    tags = []
    if forfeit_on_us: tags.append("FORFEIT")
    if final_clock_pct > 40 and short_move_frac > 0.3: tags.append("STOCKPILE")
    if min_clock_pct < 10 and not forfeit_on_us: tags.append("TIME_TROUBLE")
    if not tags: tags.append("CLEAN")

    return {
        "white": w, "black": b, "result": result, "tc": tc, "termination": term,
        "our_color": "W" if color == chess.WHITE else "B",
        "n_moves": n_moves,
        "time_used_pct": time_used_pct,
        "min_clock_pct": min_clock_pct,
        "final_clock_pct": final_clock_pct,
        "max_move_pct": max_move_pct,
        "short_move_frac": short_move_frac * 100,
        "forfeit_on_us": forfeit_on_us,
        "tags": tags,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--who", required=True,
                    help="Player name to analyse (e.g., 'Coda' or 'coda_bot')")
    ap.add_argument("--format", choices=["cutechess", "lichess"], default="cutechess")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pgn = open(args.pgn)
    games = []
    while True:
        g = chess.pgn.read_game(pgn)
        if g is None: break
        games.append(g)

    results = [analyse_game(g, args.who, args.format) for g in games]
    results = [r for r in results if r is not None]

    if not results:
        print(f"No games found for player '{args.who}'", file=sys.stderr)
        sys.exit(1)

    # Per-game output
    if args.verbose:
        print(f"{'#':>3} {'tc':<8} {'result':<5} {'role':<4} "
              f"{'moves':>5} {'used%':>6} {'min%':>6} {'final%':>7} "
              f"{'max_move%':>10} {'short%':>7}  tags")
        for i, r in enumerate(results, 1):
            print(f"{i:>3} {r['tc']:<8} {r['result']:<5} {r['our_color']:<4} "
                  f"{r['n_moves']:>5} "
                  f"{r['time_used_pct']:6.1f} {r['min_clock_pct']:6.1f} "
                  f"{r['final_clock_pct']:7.1f} {r['max_move_pct']:10.1f} "
                  f"{r['short_move_frac']:7.1f}  {','.join(r['tags'])}")

    # Aggregate
    def avg(key): return sum(r[key] for r in results) / len(results)
    stock = sum(1 for r in results if "STOCKPILE" in r["tags"])
    trouble = sum(1 for r in results if "TIME_TROUBLE" in r["tags"])
    forfeit = sum(1 for r in results if "FORFEIT" in r["tags"])
    clean = sum(1 for r in results if r["tags"] == ["CLEAN"])

    # Score: higher is better. 100 = ideal.
    # Penalize stockpiling, time trouble, and forfeits.
    score_pct = 100 * clean / len(results)

    print()
    print(f"=== TM score for {args.who} across {len(results)} games ===")
    print(f"  avg time_used_pct:     {avg('time_used_pct'):5.1f}  (ideal: 70-90%)")
    print(f"  avg min_clock_pct:     {avg('min_clock_pct'):5.1f}  (ideal: >10)")
    print(f"  avg final_clock_pct:   {avg('final_clock_pct'):5.1f}  (ideal: <30)")
    print(f"  avg max_move_pct:      {avg('max_move_pct'):5.1f}  (ideal: <25)")
    print(f"  avg short_move_frac%:  {avg('short_move_frac'):5.1f}  (ideal: <30)")
    print()
    print(f"  Classification counts:")
    print(f"    FORFEIT:      {forfeit:3d} / {len(results)}")
    print(f"    STOCKPILE:    {stock:3d} / {len(results)}  (finished with >40% clock + many short moves)")
    print(f"    TIME_TROUBLE: {trouble:3d} / {len(results)}  (dipped below 10% at some point)")
    print(f"    CLEAN:        {clean:3d} / {len(results)}")
    print()
    print(f"  TM_SCORE: {score_pct:.1f}%  ({clean}/{len(results)} games clean)")


if __name__ == "__main__":
    main()
