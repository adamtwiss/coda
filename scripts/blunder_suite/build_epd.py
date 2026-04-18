#!/usr/bin/env python3
"""Build coda_blunders.epd from lichess blunder analysis.

Pipeline:
1. Load /tmp/coda_lichess/blunders_all.csv (cp_loss >= 100, already SF-flagged)
2. Secondary filter: |sf_cp_before| < 300 (position wasn't already lost)
3. For each candidate: run fresh Coda-v5 at movetime 5000ms. Categorize:
   - Fresh plays the same bad move → EVAL-GAP confirmed (include)
   - Fresh plays SF's best → state pollution (exclude from this suite)
   - Fresh plays something else → ambiguous (exclude for now, log)
4. For confirmed eval-gap positions: query SF at depth 22 for preferred move
   (we already have it in CSV but re-verify at deeper depth)
5. Emit EPD with bm/am/id/c0/c1

Output: testdata/coda_blunders.epd
"""
import csv, subprocess, threading, queue, time, re, sys, os, chess, chess.engine

CSV  = "/tmp/coda_blunders/blunders_combined.csv"
CODA = "/home/adam/code/coda/coda"
NET  = "/home/adam/code/coda/net-v5-768pw-consensus-w7-e800s800.nnue"
SF   = "/usr/games/stockfish"

MOVETIME = 3000   # 3s is enough for fresh Coda categorization
SF_DEPTH = 22
SF_CP_BEFORE_LIMIT = 500  # allow positions where Coda was down/up by up to 5 pawns
MIN_CP_LOSS = 100
OUT_EPD = "/home/adam/code/coda/testdata/coda_blunders.epd"
REPORT = "/home/adam/code/coda/testdata/coda_blunders.README.md"

class UCI:
    def __init__(self, cmd, net=None, threads=4):
        self.proc = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                     cwd="/home/adam/code/coda")
        self.q = queue.Queue()
        threading.Thread(target=self._r, daemon=True).start()
        self._s("uci"); self._w("uciok")
        if net: self._s(f"setoption name NNUEFile value {net}")
        self._s("setoption name Hash value 64")
        self._s(f"setoption name Threads value {threads}")
        self._s("setoption name OwnBook value false")
        self._s("isready"); self._w("readyok")
    def _r(self):
        for line in self.proc.stdout: self.q.put(line.strip())
    def _s(self, s): self.proc.stdin.write(s + "\n"); self.proc.stdin.flush()
    def _w(self, tok, tout=60):
        end = time.time() + tout
        tokens = tok.split("|")
        while time.time() < end:
            try:
                line = self.q.get(timeout=max(0.1, end-time.time()))
                if any(line.startswith(t) for t in tokens): return line
            except queue.Empty: continue
        raise TimeoutError(tok)
    def drain(self):
        while True:
            try: self.q.get_nowait()
            except queue.Empty: return
    def best_at_movetime(self, fen, mt):
        self.drain()
        self._s("ucinewgame"); self._s("isready"); self._w("readyok")
        self._s(f"position fen {fen}")
        self._s("isready"); self._w("readyok")
        self._s(f"go movetime {mt}")
        line = self._w("bestmove", tout=mt/1000+30)
        return line.split()[1]
    def quit(self):
        try: self._s("quit"); self.proc.wait(timeout=3)
        except Exception: self.proc.kill()

def main():
    os.makedirs(os.path.dirname(OUT_EPD), exist_ok=True)

    rows = list(csv.DictReader(open(CSV)))
    print(f"Loaded {len(rows)} rows from CSV.")

    # Primary filter
    candidates = [r for r in rows
                  if int(r["cp_loss"]) >= MIN_CP_LOSS
                  and abs(int(r["sf_cp"])) <= SF_CP_BEFORE_LIMIT]
    print(f"After filtering (cp_loss>={MIN_CP_LOSS}, |sf_cp|<={SF_CP_BEFORE_LIMIT}): "
          f"{len(candidates)} candidates")

    # Categorize via fresh Coda-v5
    print(f"\nRunning fresh Coda-v5 at {MOVETIME}ms on each candidate...")
    coda = UCI(CODA, net=NET)
    categorized = []
    for i, r in enumerate(candidates):
        try:
            bm = coda.best_at_movetime(r["fen"], MOVETIME)
        except Exception as e:
            print(f"  [{i+1}/{len(candidates)}] ERROR: {e}")
            continue
        r["fresh_coda_move"] = bm
        played = r["played"]
        sf_best = r["sf_best"]
        if bm == played:
            r["category"] = "eval_gap"
        elif bm == sf_best:
            r["category"] = "state_pollution"
        else:
            r["category"] = "different_wrong"
        categorized.append(r)
        print(f"  [{i+1:2d}/{len(candidates)}] {r['category']:<17} fen={r['fen'][:30]}... "
              f"played={played} fresh={bm} sf_best={sf_best}")
    coda.quit()

    # Counts
    from collections import Counter
    cats = Counter(r["category"] for r in categorized)
    print(f"\nCategory distribution:")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")

    eval_gap = [r for r in categorized if r["category"] == "eval_gap"]
    print(f"\nEval-gap candidates for EPD suite: {len(eval_gap)}")

    # Deep SF verify to confirm best move at depth 22
    print(f"\nVerifying SF best move at depth {SF_DEPTH} for each eval-gap position...")
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Hash": 256, "Threads": 4})
    final = []
    for i, r in enumerate(eval_gap):
        board = chess.Board(r["fen"])
        info = sf.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
        best_mv = info.get("pv", [None])[0]
        sf_score = info["score"].pov(board.turn)
        if best_mv is None: continue
        best_uci = best_mv.uci()
        best_san = board.san(best_mv)
        cp = sf_score.score(mate_score=30000)
        r["sf_best_d22"] = best_uci
        r["sf_best_d22_san"] = best_san
        r["sf_cp_d22"] = cp
        final.append(r)
        print(f"  [{i+1:2d}/{len(eval_gap)}] sf_best={best_uci} ({best_san}) cp={cp}")
    sf.quit()

    print(f"\nEPD positions to emit: {len(final)}")

    # Emit EPD
    # Get game URL from CSV if available; fall back to game_id
    def coda_uci_to_san(fen, uci):
        board = chess.Board(fen)
        # Find legal move matching UCI
        for mv in board.legal_moves:
            if mv.uci() == uci:
                return board.san(mv)
        return uci

    with open(OUT_EPD, "w") as f:
        f.write("# Coda blunder regression suite\n")
        f.write("# Built from real lichess games where Coda blundered (per SF) AND\n")
        f.write("# fresh Coda-v5 reproduces the bad move (confirmed eval-gap, not state).\n")
        f.write(f"# Generated {time.strftime('%Y-%m-%d')}. {len(final)} positions.\n\n")
        for r in final:
            fen = r["fen"]
            bm_san = r["sf_best_d22_san"]
            am_san = coda_uci_to_san(fen, r["played"])
            game_id = r.get("game_id", "").split("/")[-1] if "/" in r.get("game_id","") else r.get("game_id","")
            opp = r.get("opp", "?")
            cp_before = r["sf_cp"]
            cp_after  = r["our_cp"]
            cp_loss   = r["cp_loss"]
            f.write(
                f'{fen} bm {bm_san}; am {am_san}; '
                f'id "{game_id}.p{r["ply"]}"; '
                f'c0 "sf_before={cp_before} sf_after={cp_after} cp_loss={cp_loss}"; '
                f'c1 "opp={opp}";\n'
            )

    print(f"\n✓ Wrote {OUT_EPD}")

    # README
    with open(REPORT, "w") as f:
        f.write(f"""# Coda Blunder Regression Suite

Built {time.strftime('%Y-%m-%d')} from {len(rows)} lichess blunders.

## Construction

Pipeline (see `/tmp/coda_blunders/build_epd.py`):

1. Start from all Coda moves SF classified as blunders (cp_loss >= {MIN_CP_LOSS})
   from actual lichess games (user account coda_bot).
2. Filter to positions where Coda wasn't already lost/winning
   (|sf_cp_before| <= {SF_CP_BEFORE_LIMIT}).
3. Run fresh Coda-v5 ({NET}) at movetime {MOVETIME}ms on each candidate.
4. Categorize by what fresh Coda plays:
   - Same as game move → **eval_gap** (this suite)
   - SF's best → **state_pollution** (F1/TM territory, not this suite)
   - Different other move → **different_wrong** (excluded for now)
5. For each eval_gap position, re-query SF at depth {SF_DEPTH} to get a
   reliable `bm`.
6. Emit as EPD with `bm <sf_best>; am <played>; ...`.

## Category counts for this build

""")
        for cat, n in cats.most_common():
            f.write(f"- {cat}: {n}\n")
        f.write(f"\nFinal EPD size: {len(final)}\n")
        f.write(f"""

## How to use

Score with Coda's built-in EPD runner (scores `bm` + `am` since commit e88981c):

```
./coda epd testdata/coda_blunders.epd -t 3000 -n <your_net>.nnue
```

A position passes if engine plays the SF-preferred move AND does not play
the historical blunder move. Both conditions must hold.

## Purpose

Track eval-quality improvement across Coda net architectures. Each net
version should improve pass-rate on this suite. Position types represent
real failure modes observed in production play on lichess.

## Regeneration

When you've played more lichess games and have more blunder data:

```
# 1. Re-fetch recent games and re-run SF analysis
# (see /tmp/coda_lichess/analyze_blunders.py)
# 2. Regenerate suite
python3 /tmp/coda_blunders/build_epd.py
# 3. Review /tmp/coda_blunders/build_epd.log
```

The suite size will grow (or shrink if F1/v9 fixes existing categories).
""")
    print(f"✓ Wrote {REPORT}")

if __name__ == "__main__":
    main()
