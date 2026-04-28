#!/usr/bin/env python3
"""Three diagnostic probes per candidate (Adam's framing 2026-04-28):

For each pruning-candidate JSONL row (FEN + played move + SF best),
this script answers:

  1. CLEAN-HASH ALTERNATIVE QUALITY: Coda picks bestmove on a clean
     hash at fixed depth. Run SF on the resulting position to score
     Coda's clean-hash choice. Three buckets:
       - "matches_sf"     bestmove == sf_best  (Coda's clean choice
                                                already agrees with SF)
       - "reasonable"     ΔSF for clean choice ≥ -50cp (better than
                          played move's ΔSF, but not SF-best)
       - "same_bug"       ΔSF for clean choice ≈ played-move's ΔSF
                          (Coda's clean choice is just as bad — the
                          played move wasn't a state-pollution
                          artifact, the issue is search/eval)

  2. CODA'S OPINION OF SF-BEST: use UCI `searchmoves <sf_best>` to
     restrict Coda's root-move list and probe its score for SF's
     bestmove. Compared to Coda's score for the played move:
       - score(SF_best) > score(played):  Coda likes SF's move more
                                          but didn't pick it →
                                          MOVE-ORDERING bug / pruning
                                          missed it on the way to root
       - score(SF_best) < score(played):  Coda's search RESOLVES SF's
                                          move to a worse value than
                                          the bad played move →
                                          EVAL/DEPTH bug
       - approximately equal: pruning culled the SF line before it
                              could refine; reads like ordering but
                              upstream

  3. PV-WALK DEPTH-OF-INSIGHT: walk down SF's PV from the candidate
     position. At step N, play SF's first N PV moves, then ask Coda
     for its bestmove at the resulting position. Find the smallest N
     where Coda agrees with SF on move N+1. That's how many plies of
     the refutation Coda needs handed to it before it "sees" the
     line. Reports the N for each candidate; histogram is the
     depth-of-insight distribution.

Output: CSV with one row per candidate.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import chess


SF_DEFAULT = "/home/adam/chess/engines/Stockfish/src/stockfish"
SF_PV_DEPTH = 20  # SF depth used to obtain the PV for diagnostic 3
SF_SCORE_DEPTH = 18  # SF depth used to score positions in diagnostic 1


def run_uci_engine(binary: str, fen: str, depth: int,
                   searchmoves: str | None = None,
                   hash_mb: int = 64,
                   threads: int = 1) -> tuple[str | None, int | None, list[str]]:
    """Run a fresh UCI engine on a position. Returns (bestmove, score_cp, pv).

    pv is the principal variation reported on the last info line.
    score_cp is from side-to-move's POV. Mate scores get clamped to ±10000.
    """
    cmds = [
        "uci",
        f"setoption name Hash value {hash_mb}",
        f"setoption name Threads value {threads}",
        "ucinewgame",
        f"position fen {fen}",
    ]
    if searchmoves:
        cmds.append(f"go depth {depth} searchmoves {searchmoves}")
    else:
        cmds.append(f"go depth {depth}")
    cmds.append("quit")
    proc = subprocess.Popen(
        [binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True,
    )
    out, _ = proc.communicate(input="\n".join(cmds) + "\n", timeout=300)
    bestmove = None
    last_score = None
    last_pv: list[str] = []
    for line in out.splitlines():
        if line.startswith("info "):
            sm = re.search(r"score (cp|mate) (-?\d+)", line)
            if sm:
                if sm.group(1) == "cp":
                    last_score = int(sm.group(2))
                else:
                    mn = int(sm.group(2))
                    last_score = 10000 if mn > 0 else -10000
            pm = re.search(r" pv (.+)$", line)
            if pm:
                last_pv = pm.group(1).split()
        elif line.startswith("bestmove "):
            bestmove = line.split()[1]
            break
    return bestmove, last_score, last_pv


def diagnostic_1_clean_hash(coda: str, fen: str, played_uci: str,
                             sf_best: str, depth: int) -> dict:
    """What does Coda pick on a clean hash?"""
    bestmove, score, pv = run_uci_engine(coda, fen, depth)
    return {
        "coda_clean_bestmove": bestmove,
        "coda_clean_score": score,
        "coda_clean_pv": " ".join(pv[:5]),
        "matches_played": int(bestmove == played_uci) if bestmove else 0,
        "matches_sf_best": int(bestmove == sf_best) if bestmove else 0,
    }


def diagnostic_2_score_alternatives(coda: str, fen: str, played_uci: str,
                                     sf_best: str, depth: int) -> dict:
    """How does Coda score SF's best vs the played move?"""
    _bm_played, score_played, pv_played = run_uci_engine(
        coda, fen, depth, searchmoves=played_uci)
    _bm_sf, score_sf, pv_sf = run_uci_engine(
        coda, fen, depth, searchmoves=sf_best)
    diff = None
    if score_played is not None and score_sf is not None:
        diff = score_sf - score_played  # positive = Coda prefers SF's move
    return {
        "coda_score_played": score_played,
        "coda_score_sf_best": score_sf,
        "coda_score_diff": diff,
        "coda_pv_after_sf_best": " ".join(pv_sf[:5]),
    }


def diagnostic_3_pv_walk(coda: str, sf: str, fen: str, sf_pv: list[str],
                          coda_depth: int) -> dict:
    """How many plies of SF's PV does Coda need before agreeing on the next?"""
    if len(sf_pv) < 2:
        return {"pv_walk_n": None, "pv_walk_msg": "SF PV < 2 plies"}
    board = chess.Board(fen)
    for n in range(0, min(len(sf_pv) - 1, 8)):
        # After playing sf_pv[0..n], does Coda pick sf_pv[n]?
        # n=0: at the starting fen, does Coda pick sf_pv[0]?
        # n=1: after sf_pv[0], does Coda pick sf_pv[1]?
        if n == 0:
            test_fen = fen
        else:
            test_board = chess.Board(fen)
            try:
                for i in range(n):
                    test_board.push(chess.Move.from_uci(sf_pv[i]))
                test_fen = test_board.fen()
            except Exception as e:
                return {"pv_walk_n": None, "pv_walk_msg": f"PV illegal at step {n}: {e}"}
        bestmove, _, _ = run_uci_engine(coda, test_fen, coda_depth)
        if bestmove == sf_pv[n]:
            return {"pv_walk_n": n, "pv_walk_msg": f"Coda agrees from step {n}"}
    return {"pv_walk_n": -1, "pv_walk_msg": "Coda doesn't agree within first 8 PV plies"}


def get_sf_pv(sf: str, fen: str, depth: int) -> list[str]:
    """Run SF at fen, return the full PV of bestmove."""
    _bm, _score, pv = run_uci_engine(sf, fen, depth, hash_mb=128)
    return pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path, help="JSONL with fen_before + blunder_uci + sf_best_uci")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coda", type=Path, default=Path("./coda"))
    ap.add_argument("--sf", default=SF_DEFAULT)
    ap.add_argument("--coda-depth", type=int, default=14, help="Coda probe depth")
    ap.add_argument("--sf-pv-depth", type=int, default=SF_PV_DEPTH, help="SF depth for diagnostic 3 PV")
    ap.add_argument("--limit", type=int, default=0, help="Limit to first N candidates")
    ap.add_argument("--skip-pv-walk", action="store_true", help="Skip diagnostic 3 (slowest)")
    args = ap.parse_args()

    cands = [json.loads(l) for l in args.candidates.open() if l.strip()]
    if args.limit:
        cands = cands[: args.limit]
    print(f"Probing {len(cands)} candidates at coda_depth={args.coda_depth}, sf_pv_depth={args.sf_pv_depth}")

    rows = []
    for i, c in enumerate(cands):
        fen = c["fen_before"]
        played = c["blunder_uci"]
        sf_best = c.get("sf_best_uci")
        if not sf_best:
            print(f"  [{i+1}] missing sf_best_uci, skip")
            continue

        d1 = diagnostic_1_clean_hash(str(args.coda.resolve()), fen, played, sf_best, args.coda_depth)
        d2 = diagnostic_2_score_alternatives(str(args.coda.resolve()), fen, played, sf_best, args.coda_depth)
        if args.skip_pv_walk:
            d3 = {"pv_walk_n": None, "pv_walk_msg": "skipped"}
            sf_pv = []
        else:
            sf_pv = get_sf_pv(args.sf, fen, args.sf_pv_depth)
            d3 = diagnostic_3_pv_walk(str(args.coda.resolve()), args.sf, fen, sf_pv, args.coda_depth)

        row = {
            "idx": i,
            "opponent": c.get("opponent"),
            "ply": c.get("ply"),
            "fen": fen,
            "played_uci": played,
            "sf_best": sf_best,
            "sf_pv": " ".join(sf_pv[:5]),
            "score_before": c.get("score_before"),
            "score_after": c.get("score_after"),
            "delta_cp": c.get("delta_cp"),
            **d1, **d2, **d3,
        }
        rows.append(row)
        print(f"  [{i+1}/{len(cands)}] played={played} sf_best={sf_best} "
              f"clean_pick={d1['coda_clean_bestmove']} "
              f"({'SF' if d1['matches_sf_best'] else ('PLAYED' if d1['matches_played'] else 'OTHER')}) "
              f"score_diff={d2['coda_score_diff']} pv_walk_n={d3['pv_walk_n']}")

    if rows:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")

    # Aggregate summary
    print(f"\n=== Diagnostic 1: Coda's clean-hash choice ===")
    n = len(rows) or 1
    matches_sf = sum(r["matches_sf_best"] for r in rows)
    matches_played = sum(r["matches_played"] for r in rows)
    other = n - matches_sf - matches_played
    print(f"  Picks SF-best:    {matches_sf}/{n} ({100*matches_sf/n:.0f}%)")
    print(f"  Picks the played: {matches_played}/{n} ({100*matches_played/n:.0f}%)  [same-bug — clean hash didn't help]")
    print(f"  Picks other:      {other}/{n} ({100*other/n:.0f}%)  [partial recovery, may be reasonable]")

    print(f"\n=== Diagnostic 2: Coda's eval gap (SF-best minus played) ===")
    diffs = [r["coda_score_diff"] for r in rows if r["coda_score_diff"] is not None]
    if diffs:
        prefers_sf = sum(1 for d in diffs if d > 50)
        prefers_played = sum(1 for d in diffs if d < -50)
        equal = len(diffs) - prefers_sf - prefers_played
        print(f"  Coda prefers SF (>+50cp):       {prefers_sf}/{len(diffs)}  [ordering bug — should pick SF]")
        print(f"  Coda prefers played (<-50cp):   {prefers_played}/{len(diffs)}  [eval/depth bug — Coda's search resolves SF-best to a worse value]")
        print(f"  Coda essentially equal (±50cp): {equal}/{len(diffs)}")

    print(f"\n=== Diagnostic 3: PV-walk depth-of-insight ===")
    walks = [r["pv_walk_n"] for r in rows if r["pv_walk_n"] is not None and r["pv_walk_n"] >= 0]
    if walks:
        agrees_at_n_zero = sum(1 for w in walks if w == 0)
        print(f"  N=0 (Coda agrees on bestmove):   {agrees_at_n_zero}/{len(walks)}  [no depth deficit; pruning bug]")
        for n in (1, 2, 3, 4, 5, 6, 7):
            count = sum(1 for w in walks if w == n)
            if count: print(f"  N={n}:                            {count}/{len(walks)}  [needs {n} ply{'s' if n != 1 else ''} of refutation handed]")
    never = sum(1 for r in rows if r["pv_walk_n"] == -1)
    if never:
        print(f"  Coda never agrees within 8 PV plies: {never}/{n}")


if __name__ == "__main__":
    main()
