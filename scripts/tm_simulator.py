#!/usr/bin/env python3
"""Pure-Python simulator of Coda's TM logic for fast regression checks.

Replicates the math from compute_tm_budgets + the dynamic-TM scale block.
The intent is to catch egregious mis-allocations (e.g., over-spend at
no-inc TCs that causes lichess time forfeits) WITHOUT running real chess
games. Each game simulates in milliseconds; we can sweep TCs.

This is NOT chess search — the model abstracts the search-progress
signals (stab, bmc, score_factor, nodes_factor) using a simple typical-
distribution model. The output is the SPEND DECISION TM would make
given that simulated search progress, which is what we want to
validate.

Usage:
  python3 scripts/tm_simulator.py 180 0       # 3+0
  python3 scripts/tm_simulator.py 60 1        # 1+1
  python3 scripts/tm_simulator.py 600 0       # 10+0
  python3 scripts/tm_simulator.py --all       # sweep common TCs
"""

import sys, argparse


def compute_tm_budgets(our_time, our_inc, movestogo, overhead, fullmove):
    """Port of src/search.rs compute_tm_budgets. Returns (soft, hard, soft_floor) in ms."""
    time_left = max(1, our_time - overhead)

    # Sudden-death no-inc gets tighter moves_left assumption.
    if movestogo > 0:
        moves_left = movestogo
    elif our_inc == 0:
        moves_left = 40
    else:
        moves_left = 25

    soft = time_left // moves_left + (our_inc * 4) // 5

    # Phase-aware soft scaling
    if movestogo == 0:
        if fullmove <= 5:    phase = 40
        elif fullmove <= 10: phase = 60
        elif fullmove <= 15: phase = 85
        else:                phase = 100
        soft = soft * phase // 100

    # Cap soft as fraction of remaining time
    if movestogo > 0:
        max_pct = max(30, min(90, 95 - movestogo * 5))
    else:
        max_pct = 50
    soft = min(soft, time_left * max_pct // 100)

    # Emergency for low time
    if time_left < 1000:
        emergency = time_left // 10
        if our_inc > 0 and our_inc < emergency:
            emergency = our_inc
        emergency = max(10, emergency)
        soft = min(soft, emergency)

    soft = max(soft, 10)
    base_soft = soft

    # Hard limit
    if movestogo > 0:
        hard = min(base_soft * 2, time_left * max(30, min(90, 95 - movestogo * 10)) // 100)
    else:
        hard = base_soft * 3

    # Universal hard cap for sudden-death
    if movestogo == 0:
        estimated_spm = time_left // 25 + our_inc
        if estimated_spm < 2000:    hard_mult10 = 20
        elif estimated_spm < 5000:  hard_mult10 = 25
        elif estimated_spm < 15000: hard_mult10 = 30
        else:                       hard_mult10 = 40

        # No-inc gets tighter pct caps
        if our_inc == 0:
            if estimated_spm < 2000:    max_single_pct = 4
            elif estimated_spm < 5000:  max_single_pct = 5
            elif estimated_spm < 15000: max_single_pct = 6
            else:                       max_single_pct = 8
        else:
            if estimated_spm < 2000:    max_single_pct = 8
            elif estimated_spm < 5000:  max_single_pct = 9
            elif estimated_spm < 15000: max_single_pct = 12
            else:                       max_single_pct = 15

        mult_cap = base_soft * hard_mult10 // 10
        pct_cap = time_left * max_single_pct // 100
        new_hard = min(mult_cap, pct_cap)

        # Minimum-reserve floor
        min_reserve = max(5 * our_inc, 2000)
        max_consumable = max(0, time_left - min_reserve)
        new_hard = min(new_hard, max_consumable)

        if hard > new_hard:
            hard = new_hard

        # Phase-aware hard scaling
        hard = hard * phase // 100

        # 3/4 absolute safety
        hard = min(hard, time_left * 3 // 4)

    if soft > hard:
        soft = hard

    # Floor: min(inc/4, soft/8) for Phase 8a, but main is min(inc/2)?
    # Actually main code is `(inc - overhead) / 2`. Let's use that.
    inc_minus_overhead = max(0, our_inc - overhead)
    soft_floor = min(inc_minus_overhead // 2, hard)

    return soft, hard, soft_floor


def simulate_game(base_ms, inc_ms, num_moves=60, scale_typical=1.0, scale_max=2.5, overhead=100, movestogo=0):
    """Simulate one side's clock through a game. scale_typical is the average
    dynamic TM multiplier; scale_max is the max on tactical moves (every ~6th).
    Returns dict with totals."""
    clock = base_ms
    spends = []
    timed_out_at = None
    for move in range(1, num_moves + 1):
        fullmove = move
        if clock <= 0:
            timed_out_at = move
            break
        soft, hard, floor = compute_tm_budgets(clock, inc_ms, movestogo, overhead, fullmove)

        # Choose a scale for this move: ~14% chance of "tactical" spike, else typical
        scale = scale_max if move % 7 == 0 else scale_typical
        adjusted_soft = int(soft * scale)
        adjusted_soft = max(adjusted_soft, floor)
        adjusted_soft = min(adjusted_soft, hard)

        # Spend the move
        spend = adjusted_soft
        clock -= spend
        if clock < 0:
            timed_out_at = move
            clock = 0
        clock += inc_ms
        spends.append(spend)

    return {
        "spends": spends,
        "final_clock": clock,
        "timed_out_at": timed_out_at,
        "total_spend": sum(spends),
        "moves_played": len(spends),
    }


def report_tc(base_ms, inc_ms, **kwargs):
    label = f"{base_ms//1000}+{inc_ms/1000:g}"
    res = simulate_game(base_ms, inc_ms, **kwargs)
    spends = res["spends"]
    if not spends:
        print(f"{label:>10}: (no moves)")
        return
    n = len(spends); spends_s = sorted(spends)
    avg = sum(spends) / n / 1000
    p10 = spends_s[n//10] / 1000
    p50 = spends_s[n//2] / 1000
    p90 = spends_s[9*n//10] / 1000
    max_spend = spends_s[-1] / 1000
    p90_p10 = p90 / p10 if p10 > 0 else float('inf')
    # Pacing check: average should be (base+inc*moves)/expected_moves
    expected_per_move = (base_ms + inc_ms * 60) / 60 / 1000
    pacing = avg / expected_per_move if expected_per_move > 0 else 0
    # Status
    forfeit = res["timed_out_at"] is not None
    over_paced = pacing > 1.3  # spending >30% more than sustainable
    under_paced = pacing < 0.5  # spending <50% — leaving time on clock
    flags = []
    if forfeit: flags.append(f"❌FORFEIT@m{res['timed_out_at']}")
    if over_paced and not forfeit: flags.append(f"⚠OVERPACED({pacing:.1f}x)")
    status = " ".join(flags) if flags else "✓"
    print(f"{label:>10}: avg={avg:5.2f}s p50={p50:5.2f}s p90={p90:5.2f}s max={max_spend:5.2f}s "
          f"spread(p90/p10)={p90_p10:4.1f}x pacing={pacing:.2f} final={res['final_clock']/1000:5.1f}s {status}")


def run_suite(num_moves=60, label=""):
    suite = [
        # Lichess/CCRL no-inc
        ( 60, 0,    "1+0 ultra-bullet"),
        (120, 0,    "2+0"),
        (180, 0,    "3+0 blitz (lichess regression case)"),
        (300, 0,    "5+0"),
        (600, 0,    "10+0"),
        # Lichess inc
        ( 60, 1,    "1+1 bullet"),
        (120, 1,    "2+1"),
        (180, 2,    "3+2"),
        (300, 3,    "5+3"),
        (600, 5,    "10+5 rapid"),
        (900,10,    "15+10"),
        # SPRT / dev TCs
        ( 10, 0.1,  "10+0.1 STC"),
        ( 40, 0.4,  "40+0.4 LTC"),
        ( 60, 0.6,  "60+0.6 CCRL Blitz"),
        ( 60, 1,    "60+1 (dup)"),
    ]
    if label: print(f"=== {label} ===")
    print(f"{'TC':>10}: {'(description)':<35}")
    print("-" * 110)
    for base, inc, desc in suite:
        if base == 60 and inc == 1 and "60+1" in desc and "(dup)" in desc:
            continue
        print(f"  {desc:<32}", end="")
        report_tc(base * 1000, int(inc * 1000), num_moves=num_moves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_s", nargs="?", type=int, help="base time in seconds")
    ap.add_argument("inc_s", nargs="?", type=float, default=None, help="increment in seconds")
    ap.add_argument("--all", action="store_true", help="run common TC suite")
    ap.add_argument("--num-moves", type=int, default=60)
    args = ap.parse_args()

    if args.all:
        run_suite(num_moves=args.num_moves,
                  label=f"TM regression suite ({args.num_moves} moves/side, typical scale=1.0, tactical=2.5×)")
    elif args.base_s is not None and args.inc_s is not None:
        report_tc(args.base_s * 1000, int(args.inc_s * 1000), num_moves=args.num_moves)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
