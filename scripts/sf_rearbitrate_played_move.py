#!/usr/bin/env python3
"""Re-arbitrate sudden candidates by probing BOTH moves at high depth.

The original arbitration used SF d18 best. But Coda often searched deeper
than d18 in actual games (19% of sudden cases at >=18). For those, SF d18
isn't a strong-enough authority — its "best" might be weaker than what
Coda saw.

Right protocol: probe SF at high depth (default d24) from the position
AFTER each candidate move:
  - score_after_sf_best  (SF plays its picked-best from fen_before)
  - score_after_played   (Coda plays its actual move)

Both scores normalised to Coda's perspective. True delta = sf_best - played.
If true delta is small, the d18 "blunder" classification was noise.

Outputs CSV with delta_d18, delta_d24 (or whatever depth probed), and a
re-classification.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import chess

SCORE_CP_RE = re.compile(r"score (cp|mate) (-?\d+)")
BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)")


class SFEngine:
    def __init__(self, sf_bin: Path, hash_mb: int = 1024, threads: int = 8):
        self.proc = subprocess.Popen(
            [str(sf_bin.resolve())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value {threads}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, c):
        self.proc.stdin.write(c + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token, max_lines=2_000_000):
        last = None
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                return None, None
            if line.startswith("info") and " pv " in line:
                last = line
            if token in line:
                return line, last
        return None, last

    def probe(self, fen: str, depth: int):
        """Returns (bestmove, score_cp_from_stm) at given depth."""
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        bm_line, info_line = self._read_until("bestmove")
        if bm_line is None:
            return None, None
        bm_match = BESTMOVE_RE.match(bm_line.strip())
        bestmove = bm_match.group(1) if bm_match else None
        score_cp = None
        if info_line:
            sm = SCORE_CP_RE.search(info_line)
            if sm:
                kind, val = sm.group(1), int(sm.group(2))
                if kind == "mate":
                    score_cp = (32000 - abs(val)) * (1 if val > 0 else -1)
                else:
                    score_cp = val
        return bestmove, score_cp

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def make_fen_after(fen_before: str, uci: str):
    """Apply a UCI move to a FEN, return resulting FEN. None if illegal."""
    try:
        b = chess.Board(fen_before)
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves:
            return None
        b.push(mv)
        return b.fen()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates_jsonl", type=Path,
                    help="enriched candidates JSONL (with played_depth)")
    ap.add_argument("--sf", type=Path,
                    default=Path("/home/adam/chess/engines/Stockfish/src/stockfish"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=24,
                    help="SF probe depth")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--hash", type=int, default=1024)
    ap.add_argument("--min-played-depth", type=int, default=0,
                    help="only re-arbitrate cases with played_depth >= this")
    args = ap.parse_args()

    candidates = []
    with args.candidates_jsonl.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            c = json.loads(line)
            c["idx"] = i
            if (c.get("played_depth") or 0) >= args.min_played_depth:
                candidates.append(c)

    print(f"Re-arbitrating {len(candidates)} candidates at d{args.depth}", file=sys.stderr)

    sf = SFEngine(args.sf, hash_mb=args.hash, threads=args.threads)

    out_rows = []
    for c in candidates:
        idx = c["idx"]
        opp = c["opponent"]
        fen = c["fen_before"]
        played = c["blunder_uci"]
        sf_d18_best = c["sf_best_uci"]
        delta_d18 = c.get("delta_cp", 0)
        played_depth = c.get("played_depth")

        row = {
            "idx": idx,
            "opponent": opp,
            "ply": c["ply"],
            "fen": fen,
            "played_uci": played,
            "played_depth": played_depth,
            "sf_d18_best": sf_d18_best,
            "delta_d18": delta_d18,
        }

        # 1. Probe SF at high depth from fen_before to get its picked best
        t0 = time.time()
        sf_high_best, score_before = sf.probe(fen, args.depth)
        elapsed_a = time.time() - t0
        row[f"sf_d{args.depth}_best"] = sf_high_best
        row[f"score_before_d{args.depth}"] = score_before

        # Determine STM at fen_before — score_before is from STM perspective.
        b = chess.Board(fen)
        coda_to_move = b.turn  # True=white, False=black

        # 2. Score position after SF's high-depth best (from opponent perspective)
        fen_after_sf = make_fen_after(fen, sf_high_best) if sf_high_best else None
        score_after_sf = None
        if fen_after_sf:
            _, opp_score_after_sf = sf.probe(fen_after_sf, args.depth)
            if opp_score_after_sf is not None:
                # opp_score_after_sf is from opponent perspective; flip for Coda's
                score_after_sf = -opp_score_after_sf
        row[f"score_after_sf_d{args.depth}"] = score_after_sf

        # 3. Score position after Coda's played move (from opponent perspective)
        fen_after_played = make_fen_after(fen, played)
        score_after_played = None
        if fen_after_played:
            _, opp_score_after_played = sf.probe(fen_after_played, args.depth)
            if opp_score_after_played is not None:
                score_after_played = -opp_score_after_played
        row[f"score_after_played_d{args.depth}"] = score_after_played

        # 4. True delta at depth N: SF-best - played (both Coda perspective)
        if score_after_sf is not None and score_after_played is not None:
            delta_high = score_after_sf - score_after_played
        else:
            delta_high = None
        row[f"delta_d{args.depth}"] = delta_high

        # 5. Reclassify
        if delta_high is None:
            klass = "probe_failed"
        elif delta_high >= 100:
            klass = "blunder_confirmed"
        elif delta_high >= 50:
            klass = "modest_blunder"
        elif delta_high >= 20:
            klass = "small_edge"
        else:
            klass = "noise_d18_arbitration_was_optimistic"
        row["reclass"] = klass

        elapsed = time.time() - t0
        print(f"  [idx={idx:2d}] vs {opp:12s} pd={played_depth} d{args.depth}: "
              f"sf_best={sf_high_best} played={played} "
              f"Δ_d18={delta_d18:+5d} Δ_d{args.depth}={delta_high} "
              f"({klass}) [{elapsed:.0f}s]", file=sys.stderr)

        out_rows.append(row)

    sf.quit()

    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nWrote {len(out_rows)} rows to {args.out}", file=sys.stderr)

    from collections import Counter
    klass_count = Counter(r["reclass"] for r in out_rows)
    print("\n=== Re-classification summary ===")
    for k, v in klass_count.most_common():
        print(f"  {k}: {v}/{len(out_rows)}")


if __name__ == "__main__":
    main()
