#!/usr/bin/env python3
"""EGTB move-SELECTION fuzzer for Coda.

Generates draw/cursed-biased small endgames (2 kings + a pawn + 0-2 minor/rook
pieces), has the engine pick its move (go depth 1 -> triggers the TB root path)
with SyzygyPath set, and uses Stockfish (TB-backed) as ground truth. Flags any
position where the engine's chosen move DROPS the WDL class (win->draw/loss,
draw->loss) -- the failure mode of the 2026-05-30 drawn-root selection bug.

Usage:
  python3 scripts/tb_selection_fuzz.py [N] [seed] [engine1] [engine2]
    N         positions to test (default 400)
    seed      RNG seed (default 7)
    engine1/2 engine binaries to compare (default ./coda twice); pass two to
              A/B a fix vs baseline on the SAME position set.

Env:
  CODA_TB    Syzygy path (default /home/adam/chess/tablebases)
  CODA_SF    Stockfish binary (default ~/chess/engines/Stockfish/src/stockfish)

Requires python-chess. Validated it catches the KBPvKB drawn-root bug
(mainHEAD picks f1e2/loses; fixed build picks f1g2/holds).
"""
import os
"""EGTB draw/cursed-biased fuzzer. Generates positions likely to be DRAWN or
CURSED (few pieces incl. a pawn, often opposite bishops / wrong-rook-pawn),
finds where a given Coda binary picks a move that DROPS the WDL class.
Compares mainHEAD vs tbfix on the SAME position set."""
import subprocess, re, random, sys, chess
TB=os.environ.get("CODA_TB","/home/adam/chess/tablebases")
SF=os.environ.get("CODA_SF",os.path.expanduser("~/chess/engines/Stockfish/src/stockfish"))
N=int(sys.argv[1]) if len(sys.argv)>1 else 400
random.seed(int(sys.argv[2]) if len(sys.argv)>2 else 7)

def eng(cmd):
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    def s(x): p.stdin.write(x+'\n'); p.stdin.flush()
    s('uci')
    while True:
        l=p.stdout.readline()
        if not l or 'uciok' in l: break
    s(f'setoption name SyzygyPath value {TB}'); s('setoption name Threads value 1'); s('isready')
    while True:
        l=p.stdout.readline()
        if not l or 'readyok' in l: break
    return p,s
def mv_of(p,s,fen):
    s('ucinewgame'); s('position fen '+fen); s('go depth 1')
    for _ in range(100000):
        l=p.stdout.readline()
        if not l: return None
        if l.startswith('bestmove'): return l.split()[1]
def sfp(p,s,fen):
    s('ucinewgame'); s('position fen '+fen); s('go depth 16'); last=None
    for _ in range(200000):
        l=p.stdout.readline()
        if not l: return None
        if l.startswith('info '):
            m=re.search(r'score (cp|mate) (-?\d+)',l)
            if m: last=int(m.group(2)) if m.group(1)=='cp' else (30000 if int(m.group(2))>0 else -30000)
        elif l.startswith('bestmove'): return last
def cls(c): return None if c is None else (1 if c>=300 else (-1 if c<=-300 else 0))

ENG1=sys.argv[3] if len(sys.argv)>3 else './coda'
ENG2=sys.argv[4] if len(sys.argv)>4 else ENG1
mp,ms=eng([ENG1]); tp,ts=eng([ENG2])
sp,ss=eng([SF]); ss('setoption name Threads value 4'); ss('isready')
while True:
    l=sp.stdout.readline()
    if not l or 'readyok' in l: break

def gen():
    # bias: 2K + 1 pawn + 0-2 minor/rook pieces, pawn often near promotion
    b=chess.Board(None)
    sqs=random.sample(range(8,56),48)  # avoid rank1/8 for pawn safety
    wk,bk=sqs[0],sqs[1]
    while chess.square_distance(wk,bk)<2:
        random.shuffle(sqs); wk,bk=sqs[0],sqs[1]
    b.set_piece_at(wk,chess.Piece(chess.KING,chess.WHITE))
    b.set_piece_at(bk,chess.Piece(chess.KING,chess.BLACK))
    idx=2
    # one pawn (the draw/cursed driver)
    pcol=random.choice([chess.WHITE,chess.BLACK])
    b.set_piece_at(sqs[idx],chess.Piece(chess.PAWN,pcol)); idx+=1
    for _ in range(random.randint(0,2)):
        pt=random.choice([chess.BISHOP,chess.KNIGHT,chess.ROOK])
        b.set_piece_at(sqs[idx],chess.Piece(pt,random.choice([chess.WHITE,chess.BLACK]))); idx+=1
    b.turn=random.choice([chess.WHITE,chess.BLACK])
    return b

main_bugs=[]; tbfix_bugs=[]; tested=0; classes={}
for _ in range(N):
    b=gen()
    if not b.is_valid(): continue
    fen=b.fen()
    rc=cls(sfp(sp,ss,fen))
    if rc is None: continue
    classes[rc]=classes.get(rc,0)+1
    if rc==-1: continue  # already lost, skip
    for tag,(p,s),bucket in [('eng1',(mp,ms),main_bugs),('eng2',(tp,ts),tbfix_bugs)]:
        cm=mv_of(p,s,fen)
        if not cm: continue
        try:
            mv=chess.Move.from_uci(cm)
            if mv not in b.legal_moves: continue
            b2=b.copy(); b2.push(mv)
        except: continue
        if b2.is_checkmate(): continue
        ac=cls(sfp(sp,ss,b2.fen()))
        if ac is None: continue
        after=-ac  # opponent POV -> ours
        if (rc==1 and after<1) or (rc==0 and after<0):
            bucket.append((fen,cm,rc,after))
            if tag=='eng1': print(f"  MAIN BUG: {fen} {cm} {rc}->{after}")
            else: print(f"  *** TBFIX BUG (fix incomplete): {fen} {cm} {rc}->{after}")
    tested+=1
print(f"\ntested {tested}; root classes {classes}")
print(f"{ENG1} WDL-drops: {len(main_bugs)}   {ENG2} WDL-drops: {len(tbfix_bugs)}")
for p,s in [(mp,ms),(tp,ts),(sp,ss)]: s('quit')
