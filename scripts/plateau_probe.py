#!/usr/bin/env python3
import sys, os, re, subprocess, time
# python-chess: pip install python-chess, or point PYCHESS_PATH at an extracted
# source tree (the package is pure Python).
if os.environ.get("PYCHESS_PATH"):
    sys.path.insert(0, os.environ["PYCHESS_PATH"])
import chess, chess.pgn
SF = os.environ.get("SF_PATH", "/home/adam/chess/engines/Stockfish/src/stockfish")
PGN, CODA = sys.argv[1], sys.argv[2]
class Ref:
    def __init__(self):
        self.p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok"); self.send("setoption name Threads value 4")
        self.send("setoption name Hash value 256"); self.send("isready"); self.wait("readyok")
    def send(self,c): self.p.stdin.write(c+"\n"); self.p.stdin.flush()
    def wait(self,pref):
        while True:
            l=self.p.stdout.readline()
            if l.startswith(pref): return l
    def eval_fen(self,fen):
        self.send("ucinewgame"); self.send("isready"); self.wait("readyok")
        self.send(f"position fen {fen}"); self.send("go depth 16")
        sc=None
        while True:
            l=self.p.stdout.readline()
            m=re.search(r"score (cp|mate) (-?\d+)",l)
            if m: sc = 100000 if (m.group(1)=="mate" and int(m.group(2))>0) else -100000 if m.group(1)=="mate" else int(m.group(2))
            if l.startswith("bestmove"): return sc
ref = Ref()
plateaus = phantom = soft = real = games = 0
with open(PGN) as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None: break
        if game.headers.get("Result","*") != "1/2-1/2": continue
        white = game.headers.get("White","")
        if CODA not in (white, game.headers.get("Black","")): continue
        games += 1
        coda_white = (white == CODA)
        board = game.board(); streak=0; mid_fen=None
        for node in game.mainline():
            mover_white = board.turn == chess.WHITE
            board.push(node.move)
            if mover_white != coda_white: continue
            m = re.match(r"\s*([+-]?M?\d+(?:\.\d+)?)/", node.comment or "")
            ev = None
            if m:
                tok = m.group(1)
                if "M" in tok: ev = 100.0 if not tok.startswith("-") else -100.0
                else:
                    try: ev = float(tok)
                    except ValueError: pass
            if ev is not None and ev >= 3.0:
                streak += 1
                if streak == 10 and mid_fen is None: mid_fen = board.fen()
            else: streak = 0
        if mid_fen:
            plateaus += 1
            sf = -ref.eval_fen(mid_fen)
            if sf <= 50: phantom += 1
            elif sf < 150: soft += 1
            else: real += 1
print(f"{PGN.split('/')[-1]} ({CODA}): drawn={games} plateaus(+3.0x10)={plateaus} PHANTOM={phantom} soft={soft} real-but-drawn={real}")
