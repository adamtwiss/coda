#!/usr/bin/env python3
"""Validate harvested candidate mistakes with two SF protocols.

For each candidate (fen, coda_move A):
  Protocol D (depth): SF fresh, go nodes N_BASE -> best B + eval;
      price A via searchmoves at N_PRICE; gap_d = eval(B) - eval(A).
      Then one deeper search at N_DEEP: gap_deep (re-priced).
  Protocol E (explore-and-return, Adam's lichess technique): after the
      baseline, WITHOUT ucinewgame, walk 3 plies down the A-line and the
      B-line (searching each node), then return to the root and re-search
      + re-price A with the warm TT. gap_e.

Verdict per protocol: CONFIRMED if gap >= 80cp, EXONERATED if gap <= 30cp,
SOFT otherwise. Report agreement matrix D-vs-E and the surviving suite.
"""
import subprocess, sys, csv, time

SF = "/home/adam/chess/engines/Stockfish/src/stockfish"
SFDIR = "/home/adam/chess/engines/Stockfish/src"
N_BASE, N_PRICE, N_WALK, N_DEEP = 2_000_000, 1_500_000, 600_000, 12_000_000
CONFIRM, EXON = 80, 30

class Engine:
    def __init__(self):
        self.p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, cwd=SFDIR)
        self.send("uci"); self.wait("uciok")
        self.send("setoption name Threads value 8")
        self.send("setoption name Hash value 1024")
        self.ready()
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def wait(self, tok):
        out = []
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("engine died")
            out.append(l)
            if l.startswith(tok): return out
    def ready(self): self.send("isready"); self.wait("readyok")
    def go(self, pos_cmd, nodes, searchmoves=None):
        self.send(pos_cmd)
        sm = f" searchmoves {searchmoves}" if searchmoves else ""
        self.send(f"go nodes {nodes}{sm}")
        lines = self.wait("bestmove")
        best = lines[-1].split()[1]
        cp = None
        for l in reversed(lines[:-1]):
            t = l.split()
            if "score" in t:
                i = t.index("score")
                if t[i+1] == "cp": cp = int(t[i+2])
                elif t[i+1] == "mate": cp = 10000 if int(t[i+2]) > 0 else -10000
                break
        return best, cp
    def new_game(self): self.send("ucinewgame"); self.ready()
    def quit(self):
        try: self.send("quit"); self.p.wait(timeout=5)
        except Exception: self.p.kill()

def pv_walk(eng, fen, first, plies=3):
    """Walk down a line from fen starting with `first`, searching each node
    (TT retained). Returns nothing — the point is warming the TT."""
    moves = [first]
    for _ in range(plies):
        best, cp = eng.go(f"position fen {fen} moves {' '.join(moves)}", N_WALK)
        if best in ("(none)", "0000"): break
        moves.append(best)

def verdict(gap):
    if gap is None: return "n/a"
    return "CONFIRMED" if gap >= CONFIRM else ("EXONERATED" if gap <= EXON else "SOFT")

rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
out = open(sys.argv[2] if len(sys.argv) > 2 else 'oracle_verdicts.tsv', 'w')
out.write("fen\tA\tB\tgap_d\tgap_deep\tgap_e\tvd\tvdeep\tve\n")
eng = Engine()
t0 = time.time()
agree = {}
for i, r in enumerate(rows):
    fen, A = r['fen'], r['played_uci']
    eng.new_game()
    pos = f"position fen {fen}"
    B, evB = eng.go(pos, N_BASE)
    if B == A:  # SF@2M already agrees with Coda — exonerated trivially
        out.write(f"{fen}\t{A}\t{B}\t0\t0\t0\tEXONERATED\tEXONERATED\tEXONERATED\n")
        continue
    _, evA = eng.go(pos, N_PRICE, searchmoves=A)
    gap_d = (evB - evA) if (evB is not None and evA is not None) else None
    # Protocol E: explore both lines, return (TT retained)
    pv_walk(eng, fen, A); pv_walk(eng, fen, B)
    B2, evB2 = eng.go(pos, N_BASE)
    _, evA2 = eng.go(pos, N_PRICE, searchmoves=A)
    gap_e = (evB2 - evA2) if (evB2 is not None and evA2 is not None) else None
    if B2 == A: gap_e = 0
    # Protocol D-deep: fresh, one big search + price
    eng.new_game()
    B3, evB3 = eng.go(pos, N_DEEP)
    if B3 == A:
        gap_deep = 0
    else:
        _, evA3 = eng.go(pos, N_DEEP // 2, searchmoves=A)
        gap_deep = (evB3 - evA3) if (evB3 is not None and evA3 is not None) else None
    vd, vdeep, ve = verdict(gap_d), verdict(gap_deep), verdict(gap_e)
    agree[(vdeep, ve)] = agree.get((vdeep, ve), 0) + 1
    out.write(f"{fen}\t{A}\t{B}\t{gap_d}\t{gap_deep}\t{gap_e}\t{vd}\t{vdeep}\t{ve}\n")
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{len(rows)} elapsed {time.time()-t0:.0f}s", flush=True)
eng.quit()
print("\nagreement (deep-verdict, explore-verdict): count")
for k, v in sorted(agree.items()):
    print(f"  {k}: {v}")
