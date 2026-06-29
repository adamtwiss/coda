#!/usr/bin/env python3
"""
conversion_inspect.py — eyeball the conversion-failure pipeline using the
OPPONENT's in-PGN eval as a zero-cost arbitrator (no SF, no core contention
with a running gauntlet).

For each Coda winning-then-not-won game it prints Coda's peak eval AND the
opponent's eval of the same position (negated to Coda-POV). The two together
pre-sort:
  AGREE  (opp also thinks Coda is winning)  -> genuine conversion failure (#4)
  SPLIT  (opp thinks ~equal/better)         -> Coda OVERSCORE suspect (#1) [SF to confirm]

Modes:
  (default)  scan: one line per candidate, sorted by Coda peak.
  --game SUBSTR : replay one game (match on opponent name or round) move-by-move
                  with BOTH evals + SAN, marking the peak and the biggest drop.
"""
import argparse, re, sys, chess, chess.pgn

CODA="Coda"
EVAL_RE=re.compile(r'^\s*([+-]?M\d+|[+-]?\d+\.\d+|[+-]?\d+)\s*/\s*\d+')

def ev(comment):
    if not comment: return None
    m=EVAL_RE.match(comment.strip())
    if not m: return None
    t=m.group(1)
    if 'M' in t or 'm' in t: return (-30000 if t.startswith('-') else 30000)
    try: return int(round(float(t)*100))
    except: return None

def coda_col(g):
    if g.headers.get('White')==CODA: return chess.WHITE
    if g.headers.get('Black')==CODA: return chess.BLACK
    return None

def res_for(g,c):
    r=g.headers.get('Result','')
    if r=='1/2-1/2': return 'D'
    if r=='1-0': return 'W' if c==chess.WHITE else 'L'
    if r=='0-1': return 'W' if c==chess.BLACK else 'L'
    return '?'

def is_junk(g):
    return 'stall' in (g.headers.get('Termination','')+g.headers.get('Annotator','') or '').lower()

def walk(g):
    """yield (ply, board_before, mover, move, eval_mover_pov) for every move."""
    board=g.board(); node=g; ply=0
    while node.variations:
        nxt=node.variation(0); mv=nxt.move
        yield ply, board.copy(), board.turn, mv, ev(nxt.comment)
        board.push(mv); node=nxt; ply+=1

def candidates(games, col_filter=None):
    out=[]
    for g in games:
        c=coda_col(g)
        if c is None or is_junk(g): continue
        if res_for(g,c)=='W': continue
        steps=list(walk(g))
        coda=[(p,b,mv,e) for (p,b,t,mv,e) in steps if t==c and e is not None]
        opp ={p:(-e) for (p,b,t,mv,e) in steps if t!=c and e is not None}  # opp eval -> Coda POV
        elig=[(p,b,mv,e) for (p,b,mv,e) in coda if p>=16]
        if not elig: continue
        pk=max(elig,key=lambda x:x[3])
        if pk[3]<200: continue
        # opponent eval nearest the peak ply (their reply, else prior)
        oc=opp.get(pk[0]+1, opp.get(pk[0]-1))
        opp_name=g.headers.get('Black') if c==chess.WHITE else g.headers.get('White')
        out.append(dict(g=g,col=c,res=res_for(g,c),opp=opp_name,rnd=g.headers.get('Round','?'),
                        peak_ply=pk[0],peak_cp=pk[3],peak_fen=pk[1].fen(),opp_cp=oc,coda=coda))
    return out

def verdict(peak,oc):
    if oc is None: return 'NO-OPP'
    if oc>=0.5*peak and oc>=150: return 'AGREE(#4)'
    if oc<=80: return 'SPLIT(overscore?)'
    return 'PARTIAL'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pgn',nargs='+',required=True)
    ap.add_argument('--game',default=None,help='replay games whose opponent/round matches this substring')
    a=ap.parse_args()
    games=[]
    for p in a.pgn:
        with open(p) as f:
            while (g:=chess.pgn.read_game(f)) is not None: games.append(g)
    cand=candidates(games)
    if not a.game:
        print(f"# {len(games)} games, {len(cand)} winning-then-not-won candidates")
        print(f"{'res':3} {'opp':14} {'peakcp':>7} {'@ply':>5} {'oppcp':>7}  verdict           fen")
        for c in sorted(cand,key=lambda c:-c['peak_cp']):
            oc = '' if c['opp_cp'] is None else f"{c['opp_cp']:+d}"
            print(f"{c['res']:3} {c['opp'][:14]:14} {c['peak_cp']:+7d} {c['peak_ply']:5d} {oc:>7}  {verdict(c['peak_cp'],c['opp_cp']):17} {c['peak_fen']}")
        # quick tally
        from collections import Counter
        t=Counter(verdict(c['peak_cp'],c['opp_cp']) for c in cand)
        print("\n# opponent-arbitrated tally:", dict(t))
        return
    # replay mode
    for c in cand:
        label=f"{c['opp']} {c['rnd']}"
        if a.game.lower() not in label.lower(): continue
        g=c['g']; col=c['col']
        print(f"\n=== Coda ({'W' if col==chess.WHITE else 'B'}) vs {c['opp']} [{c['res']}] "
              f"peak {c['peak_cp']:+d}@ply{c['peak_ply']} opp {c['opp_cp']} verdict {verdict(c['peak_cp'],c['opp_cp'])} ===")
        steps=list(walk(g))
        # find biggest Coda-POV single-move drop after peak
        coda=[(p,e) for (p,b,t,mv,e) in steps if t==col and e is not None]
        drop=(0,None)
        for i in range(1,len(coda)):
            if coda[i-1][0]>=c['peak_ply']:
                d=coda[i-1][1]-coda[i][1]
                if d>drop[0]: drop=(d,coda[i-1][0])
        for p,b,t,mv,e in steps:
            if p< c['peak_ply']-6 or p> c['peak_ply']+40: continue
            san=b.san(mv); who='Coda' if t==col else 'opp '
            cp = '' if e is None else (f"{(e if t==col else -e):+d}")  # Coda POV
            mk=''
            if p==c['peak_ply']: mk=' <-- PEAK'
            if p==drop[1]: mk=f' <-- THROW (-{drop[0]}cp)'
            fm=b.fullmove_number
            print(f"  {fm:3}{'.' if t==chess.WHITE else '...'} {san:7} [{who} {cp:>6}]{mk}")

if __name__=='__main__':
    main()
