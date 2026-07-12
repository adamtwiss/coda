#!/usr/bin/env python3
"""Phantom-overscore referee: for each game where Coda claimed sustained +2.0,
extract the claim position and have Stockfish re-evaluate it. Phantom = SF says
<= +0.5 (from Coda's side). Usage: phantom_referee.py <pgn> <coda_name> [max_games]"""
import sys, os, re, subprocess, time
# python-chess: pip install python-chess, or point PYCHESS_PATH at an extracted
# source tree (the package is pure Python).
if os.environ.get("PYCHESS_PATH"):
    sys.path.insert(0, os.environ["PYCHESS_PATH"])
import chess, chess.pgn

SF = os.environ.get("SF_PATH", "/home/adam/chess/engines/Stockfish/src/stockfish")
PGN, CODA = sys.argv[1], sys.argv[2]
MAXG = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
THRESH = 2.0
SF_DEPTH = 16

class Ref:
    def __init__(self):
        self.p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok")
        self.send("setoption name Threads value 4")
        self.send("setoption name Hash value 256")
        self.send("isready"); self.wait("readyok")
    def send(self, c): self.p.stdin.write(c + "\n"); self.p.stdin.flush()
    def wait(self, pref):
        while True:
            l = self.p.stdout.readline()
            if l.startswith(pref): return l
    def eval_fen(self, fen):
        self.send("ucinewgame"); self.send("isready"); self.wait("readyok")
        self.send(f"position fen {fen}")
        self.send(f"go depth {SF_DEPTH}")
        score = None
        while True:
            l = self.p.stdout.readline()
            m = re.search(r"score (cp|mate) (-?\d+)", l)
            if m:
                score = 100000 if (m.group(1) == "mate" and int(m.group(2)) > 0) else \
                        -100000 if m.group(1) == "mate" else int(m.group(2))
            if l.startswith("bestmove"): return score  # cp from side-to-move POV

ref = Ref()
stats = {"claims": 0, "phantom": 0, "real": 0, "mid": 0,
         "phantom_conv": [0,0,0], "real_conv": [0,0,0], "mid_conv": [0,0,0]}  # w/d/l
t0 = time.time()
with open(PGN) as f:
    n = 0
    while n < MAXG:
        game = chess.pgn.read_game(f)
        if game is None: break
        n += 1
        white = game.headers.get("White",""); res = game.headers.get("Result","*")
        if CODA not in (white, game.headers.get("Black","")) or res == "*": continue
        coda_white = (white == CODA)
        board = game.board()
        streak = 0; claim_fen = None
        for node in game.mainline():
            move = node.move
            mover_white = board.turn == chess.WHITE
            board.push(move)
            if mover_white != coda_white: continue
            c = node.comment or ""
            m = re.match(r"\s*([+-]?M?\d+(?:\.\d+)?)/", c)
            ev = None
            if m:
                tok = m.group(1)
                if tok.startswith("+M") or tok.startswith("M"): ev = 100.0
                elif tok.startswith("-M"): ev = -100.0
                else:
                    try: ev = float(tok)
                    except ValueError: pass
            if ev is not None and ev >= THRESH: streak += 1
            else: streak = 0
            if streak == 3 and claim_fen is None:
                claim_fen = board.fen()  # position after Coda's 3rd +2 move
        if claim_fen is None: continue
        stats["claims"] += 1
        sf_cp = ref.eval_fen(claim_fen)
        if sf_cp is None: continue
        # claim_fen is after Coda's move -> opponent to move -> flip sign for Coda POV
        sf_coda = -sf_cp
        outcome = 1 if res == "1/2-1/2" else (0 if (res == "1-0") == coda_white else 2)
        if sf_coda <= 50:
            stats["phantom"] += 1
            stats["phantom_conv"][outcome] += 1
        elif sf_coda >= 150:
            stats["real"] += 1
            stats["real_conv"][outcome] += 1
        else:
            stats["mid"] += 1
            stats["mid_conv"][outcome] += 1
c = stats["claims"]
print(f"{PGN.split('/')[-1]} ({CODA}): claims={c}  "
      f"PHANTOM(SF<=+0.5)={stats['phantom']} ({stats['phantom']*100//max(c,1)}%)  "
      f"mid={stats['mid']}  real(SF>=+1.5)={stats['real']}")
pw,pd,pl = stats["phantom_conv"]; rw,rd,rl = stats["real_conv"]; mw,md,ml = stats["mid_conv"]
print(f"  phantom W/D/L: {pw}/{pd}/{pl}   mid W/D/L: {mw}/{md}/{ml}   real W/D/L: {rw}/{rd}/{rl}")
print(f"  ({time.time()-t0:.0f}s)")
