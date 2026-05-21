#!/usr/bin/env python3
"""Per-engine clock-drain analysis for cutechess-cli PGN.

The "perfect decaying line" and "spends 14-24s on six consecutive
opening moves" failure modes from the Phase 3 lichess test are visible
in local cutechess games too — IF you reconstruct the clock trajectory.
This script does that without needing a lichess deploy + screenshot.

For each game it walks per-move spends, reconstructs the engine's clock
(start = base, +inc per move, −spent), and reports:
  - % of clock remaining at move 10/15/20/25/30 for each engine
  - Distribution of those values across games
  - "Overspend warning" if mean clock at move 20 < some threshold

Usage:
  python3 scripts/tm_clock_drain.py <pgn> [pgn ...]
  python3 scripts/tm_clock_drain.py --threshold-move20 30 <pgn>

Designed for the Phase 3 lichess overspend case: a game with Coda using
14-24s on 6 consecutive moves shows clock at move 15 = ~17% remaining
(vs the expected 70-80% if spending ~soft/move).
"""

import sys, re, argparse, math
from collections import defaultdict

def parse_time_spent(comment):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(s|ms)\b", comment)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1000.0 if m.group(2) == "ms" else v

def parse_tc(tc_str):
    """cutechess TC string '0/180+2' or '180+2' → (base_s, inc_s)."""
    s = tc_str
    if "/" in s:
        s = s.split("/", 1)[1]
    if "+" in s:
        base, inc = s.split("+", 1)
        return float(base), float(inc)
    return float(s), 0.0

def parse_games(pgn_text):
    games = re.split(r"\n\n(?=\[Event)", pgn_text)
    for g in games:
        if not g.strip():
            continue
        headers = {}
        for line in g.split("\n"):
            m = re.match(r'\[(\w+)\s+"(.*)"\]', line.strip())
            if m:
                headers[m.group(1)] = m.group(2)
        parts = g.split("\n\n", 1)
        movetext = parts[1] if len(parts) > 1 else ""
        moves = []
        for tok in re.finditer(r"(?:(\d+)\.(?:\.\.)?)?\s*(\S+?)\s*\{([^}]*)\}", movetext):
            san, comment = tok.group(2), tok.group(3)
            if san in ("1-0", "0-1", "1/2-1/2", "*"):
                continue
            spent = parse_time_spent(comment)
            color = "white" if len(moves) % 2 == 0 else "black"
            moves.append((color, spent))
        yield headers, moves

def quantile(xs_sorted, q):
    if not xs_sorted:
        return None
    pos = q * (len(xs_sorted) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs_sorted) - 1)
    f = pos - lo
    return xs_sorted[lo] * (1 - f) + xs_sorted[hi] * f

def reconstruct_clock(spends, base_s, inc_s):
    """spends: list of per-move spend times (seconds, None=skipped).
    Returns list of (full_move_num, clock_remaining_seconds) for each move.
    Engine starts with `base` seconds, gains `inc` after each move, loses
    `spent` for that move (matches cutechess accounting)."""
    clock = base_s
    out = []
    for i, spent in enumerate(spends):
        if spent is None:
            spent = 0.0
        # In cutechess: clock decreases by spent, then inc added after move.
        # Some servers add inc before, but cutechess uses spent-then-inc.
        clock = clock - spent + inc_s
        full_move = i + 1
        out.append((full_move, clock))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="+")
    ap.add_argument("--checkpoints", default="10,15,20,25,30,40",
                    help="comma-separated full-move numbers to report")
    ap.add_argument("--threshold-move20", type=float, default=30.0,
                    help="warn if mean clock at move 20 < this %% (default 30)")
    args = ap.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(",")]

    # engine -> checkpoint -> list of % remaining values across games
    engine_chk = defaultdict(lambda: defaultdict(list))
    # engine -> list of (move_count_at_which_clock_first_dropped_below_25%)
    engine_first_quarter = defaultdict(list)
    games_total = 0

    for pgn_path in args.pgns:
        with open(pgn_path) as f:
            pgn = f.read()
        for hdrs, moves in parse_games(pgn):
            white = hdrs.get("White")
            black = hdrs.get("Black")
            tc = hdrs.get("TimeControl", "")
            if not white or not black or not tc:
                continue
            try:
                base, inc = parse_tc(tc)
            except Exception:
                continue
            if base <= 0:
                continue
            games_total += 1

            # Split spends per engine
            w_spends = [s for (c, s) in moves if c == "white"]
            b_spends = [s for (c, s) in moves if c == "black"]

            for engine, spends in [(white, w_spends), (black, b_spends)]:
                clk_traj = reconstruct_clock(spends, base, inc)
                if not clk_traj:
                    continue
                # Record % remaining at each checkpoint
                for cp in checkpoints:
                    if cp <= len(clk_traj):
                        pct = 100.0 * clk_traj[cp - 1][1] / base
                        engine_chk[engine][cp].append(pct)
                # First move where clock fell below 25% of base
                for (mv, clk) in clk_traj:
                    if clk < 0.25 * base:
                        engine_first_quarter[engine].append(mv)
                        break
                else:
                    engine_first_quarter[engine].append(len(clk_traj) + 1)

    print(f"Games parsed: {games_total}")
    print()

    engines = sorted(engine_chk.keys(),
                     key=lambda e: (0 if "Coda" in e else 1, e))

    # Per-engine % clock remaining at each checkpoint (mean / p10)
    print("=== % clock remaining at full-move N (mean / p10 across games) ===")
    hdr = "{:<22}".format("engine") + "".join(f"  move{cp:>3}    ".format(cp=cp)
                                              for cp in checkpoints)
    print(hdr)
    print("-" * len(hdr))
    for e in engines:
        row = f"{e:<22}"
        for cp in checkpoints:
            xs = engine_chk[e][cp]
            if not xs:
                row += "       -    "
            else:
                m = sum(xs) / len(xs)
                xs_s = sorted(xs)
                p10 = quantile(xs_s, 0.10)
                row += f" {m:>5.1f}%/{p10:>4.1f}%"
        print(row)
    print()
    print("  Format:  mean% / p10% of starting clock remaining.")
    print("  Healthy engines hold ~70-90% at move 10, ~50-70% at move 20.")
    print()

    # Overspend warnings
    print("=== Early-overspend warnings ===")
    any_warn = False
    for e in engines:
        xs = engine_chk[e][20] if 20 in engine_chk[e] else []
        if xs:
            mean20 = sum(xs) / len(xs)
            if mean20 < args.threshold_move20:
                print(f"  ⚠️  {e}: only {mean20:.1f}% clock remaining at move 20 "
                      f"(threshold {args.threshold_move20:.0f}%) — overspending early")
                any_warn = True
    if not any_warn:
        print("  No engine below threshold.")
    print()

    # Distribution of "move at which clock first drops below 25%"
    print("=== First move at which clock dropped below 25% (mean / p10) ===")
    print("  Later = better (means engine kept reserves longer).")
    for e in engines:
        xs = engine_first_quarter[e]
        if not xs:
            continue
        xs_s = sorted(xs)
        m = sum(xs_s) / len(xs_s)
        p10 = quantile(xs_s, 0.10)
        print(f"  {e:<22}  mean={m:>5.1f}  p10={p10:>4.1f}  n={len(xs_s)}")

if __name__ == "__main__":
    main()
