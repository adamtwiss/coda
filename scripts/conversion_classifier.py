#!/usr/bin/env python3
"""
conversion_classifier.py — classify Coda "winning-then-not-won" games into:
  A. OVERSCORE   — peak wasn't really winning (SF/opponent disagree). Sub-split
                   by static NNUE: 'eval-overscore' (static already high) vs
                   'search-overscore' (static modest, search inflated it).
  B. CONVERSION FAILURE (#4) — peak genuinely winning, yet drew/lost:
       SPIKE  single-ply throw   (#4a search-horizon)
       DRIFT  slow bleed         (#4b flat-gradient)
       STUCK  held eval, drew    (#4b no-progress)

Three arbitration signals, increasing cost:
  - opponent eval (free, in PGN; all defenders ~3000 so credible)
  - static NNUE eval at the peak (cheap; decomposes search vs eval)
  - SF deep at the peak (gold standard; gates the close calls)

Also emits a cross-tab of opponent-verdict x SF-bucket so we can see whether
opponent-eval alone is a good-enough arbitrator (avoiding SF in future).

Refinements baked in (from reading games 2026-06-29):
  - SUSTAINED peak: eval must hold >= win-thresh for >= --sustain consecutive
    Coda moves (kills 1-ply search-froth spikes like the Alexandria 25...b4).
"""
import argparse, re, sys, subprocess, chess, chess.pgn, chess.engine
from collections import Counter

CODA="Coda"
EVAL_RE=re.compile(r'^\s*([+-]?M\d+|[+-]?\d+\.\d+|[+-]?\d+)\s*/\s*\d+')
STAT_RE=re.compile(r'NNUE evaluation\s+(-?\d+\.\d+)')

def ev(c):
    if not c: return None
    m=EVAL_RE.match(c.strip())
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
    return 'stall' in (g.headers.get('Termination','')+(g.headers.get('Annotator','') or '')).lower()

def walk(g):
    b=g.board(); node=g; ply=0
    while node.variations:
        nxt=node.variation(0); mv=nxt.move
        yield ply,b.copy(),b.turn,mv,ev(nxt.comment)
        b.push(mv); node=nxt; ply+=1

def sustained_peak(coda, win, sustain, min_ply):
    """coda = [(ply,fen,cp,uci)]. Return (idx,row) of max cp inside the longest
    run of >=sustain consecutive entries all >= win (and ply>=min_ply); else None."""
    elig=[(i,r) for i,r in enumerate(coda) if r[0]>=min_ply]
    best=None; run=[]
    def flush(run):
        nonlocal best
        if len(run)>=sustain:
            top=max(run,key=lambda ir:ir[1][2])
            if best is None or top[1][2]>best[1][2]: best=top
    for i,r in elig:
        if r[2]>=win: run.append((i,r))
        else: flush(run); run=[]
    flush(run)
    return best

def classify_throw(coda, peak_i, spike):
    after=coda[peak_i:]
    if len(after)<2: return 'STUCK',None,None
    drops=[(after[k-1][2]-after[k][2], after[k-1]) for k in range(1,len(after))]
    md,row=max(drops,key=lambda x:x[0])
    if md>=spike: return 'SPIKE',md,row
    if after[-1][2]>=0.6*after[0][2]: return 'STUCK',None,None
    return 'DRIFT',None,None

def opp_verdict(peak,oc):
    if oc is None: return 'NO-OPP'
    if oc>=0.5*peak and oc>=150: return 'AGREE'
    if oc<=80: return 'SPLIT'
    return 'PARTIAL'

def static_batch(fens, coda_bin, net):
    if not fens: return []
    inp="".join(f"position fen {f}\neval\n" for f in fens)+"quit\n"
    try:
        out=subprocess.run([coda_bin,'-n',net],input=inp,capture_output=True,text=True,timeout=300).stdout
    except Exception as e:
        print(f"static batch failed: {e}",file=sys.stderr); return [None]*len(fens)
    vals=[int(round(float(x)*100)) for x in STAT_RE.findall(out)]   # white-side cp
    if len(vals)!=len(fens):
        print(f"static count mismatch {len(vals)}!={len(fens)}",file=sys.stderr)
        vals=vals+[None]*(len(fens)-len(vals))
    return vals

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pgn',nargs='+',required=True)
    ap.add_argument('--sf',default='/home/adam/chess/engines/Stockfish/src/stockfish')
    ap.add_argument('--no-sf',action='store_true')
    ap.add_argument('--coda',default='/home/adam/code/coda/coda.rr')
    ap.add_argument('--net',default='/home/adam/code/coda/multi-v8-l132-s3-v3-swa.nnue')
    ap.add_argument('--no-static',action='store_true')
    ap.add_argument('--win-thresh',type=int,default=200)
    ap.add_argument('--sustain',type=int,default=3,help='peak must hold >= this many consecutive Coda moves')
    ap.add_argument('--overscore-cut',type=int,default=120)
    ap.add_argument('--spike-drop',type=int,default=150)
    ap.add_argument('--sf-depth',type=int,default=24)
    ap.add_argument('--sf-hash',type=int,default=2048)
    ap.add_argument('--sf-threads',type=int,default=8)
    ap.add_argument('--sf-maxtime',type=float,default=15.0)
    ap.add_argument('--min-peak-ply',type=int,default=16)
    ap.add_argument('--out-tsv',default='/tmp/conversion.tsv')
    ap.add_argument('--out-md',default='/tmp/conversion.md')
    ap.add_argument('--out-epd',default='/tmp/conversion_spike.epd')
    a=ap.parse_args()

    cand=[]; ngame=ncoda=0
    for p in a.pgn:
        with open(p) as f:
            while (g:=chess.pgn.read_game(f)) is not None:
                ngame+=1
                col=coda_col(g)
                if col is None or is_junk(g): continue
                ncoda+=1
                if res_for(g,col)=='W': continue
                steps=list(walk(g))
                coda=[(pl,b.fen(),e,mv.uci()) for pl,b,t,mv,e in steps if t==col and e is not None]
                opp ={pl:(-e) for pl,b,t,mv,e in steps if t!=col and e is not None}
                sp=sustained_peak(coda,a.win_thresh,a.sustain,a.min_peak_ply)
                if sp is None: continue
                pi,(pply,pfen,pcp,_)=sp
                oc=opp.get(pply+1,opp.get(pply-1))
                opp_name=g.headers.get('Black') if col==chess.WHITE else g.headers.get('White')
                cand.append(dict(opp=opp_name,res=res_for(g,col),peak_ply=pply,peak_cp=pcp,
                                 peak_fen=pfen,peak_i=pi,coda=coda,opp_cp=oc,white=(col==chess.WHITE)))
    print(f"{ngame} games, {ncoda} Coda games, {len(cand)} SUSTAINED winning-then-not-won candidates",file=sys.stderr)

    # static NNUE (batch, one coda process)
    if not a.no_static:
        sv=static_batch([c['peak_fen'] for c in cand],a.coda,a.net)
        for c,v in zip(cand,sv):
            c['static_cp']= None if v is None else (v if c['white'] else -v)
    else:
        for c in cand: c['static_cp']=None

    # SF gate
    sf=None
    if not a.no_sf:
        sf=chess.engine.SimpleEngine.popen_uci(a.sf); sf.configure({'Threads':a.sf_threads,'Hash':a.sf_hash})
    for i,c in enumerate(cand):
        c['sf_cp']=None
        if sf is not None:
            try:
                b=chess.Board(c['peak_fen'])
                info=sf.analyse(b,chess.engine.Limit(depth=a.sf_depth,time=a.sf_maxtime))
                c['sf_cp']=info['score'].pov(b.turn).score(mate_score=30000)
            except Exception as e: print(f"  sf fail: {e}",file=sys.stderr)
        if (i+1)%25==0: print(f"  gated {i+1}/{len(cand)}",file=sys.stderr)
    if sf: sf.quit()

    # bucketing: SF is truth if present, else opponent
    for c in cand:
        truth = c['sf_cp'] if c['sf_cp'] is not None else c['opp_cp']
        c['ov']=opp_verdict(c['peak_cp'],c['opp_cp'])
        if truth is not None and truth < a.overscore_cut:
            # overscore: eval vs search origin
            if c['static_cp'] is not None and c['static_cp']>=a.overscore_cut:
                c['bucket']='OVERSCORE-eval'
            elif c['static_cp'] is not None:
                c['bucket']='OVERSCORE-search'
            else:
                c['bucket']='OVERSCORE'
        else:
            mode,md,row=classify_throw(c['coda'],c['peak_i'],a.spike_drop)
            c['bucket']='CONV-'+mode; c['_md']=md; c['_throw']=row

    by=Counter(c['bucket'] for c in cand)
    # cross-tab opp-verdict x (overscore? per SF)
    xt=Counter()
    for c in cand:
        if c['sf_cp'] is None: continue
        sf_over = 'SF-overscore' if c['sf_cp']<a.overscore_cut else 'SF-winning'
        xt[(c['ov'],sf_over)]+=1

    with open(a.out_tsv,'w') as f:
        cols=['bucket','res','opp','peak_cp','static_cp','opp_cp','sf_cp','peak_ply','peak_fen']
        f.write('\t'.join(cols)+'\n')
        for c in sorted(cand,key=lambda c:-c['peak_cp']):
            f.write('\t'.join(str(c.get(k,'')) for k in cols)+'\n')
    with open(a.out_md,'w') as f:
        f.write(f"# Conversion classification — {len(cand)} sustained winning-then-not-won (of {ncoda} Coda games)\n\n")
        f.write("| bucket | count |\n|---|---|\n")
        for k,n in by.most_common(): f.write(f"| {k} | {n} |\n")
        f.write("\n## opponent-verdict vs SF (does free opponent-eval predict SF?)\n\n")
        f.write("| opp says | SF says | n |\n|---|---|---|\n")
        for (ov,sfo),n in sorted(xt.items()): f.write(f"| {ov} | {sfo} | {n} |\n")
        f.write("\n## genuine conversion failures (CONV-*), peak desc\n\n")
        for c in sorted([c for c in cand if c['bucket'].startswith('CONV')],key=lambda c:-c['peak_cp']):
            sfs=f"{c['sf_cp']:+d}" if c['sf_cp'] is not None else 'na'
            st=f"{c['static_cp']:+d}" if c['static_cp'] is not None else 'na'
            f.write(f"- **{c['bucket']}** [{c['res']} v {c['opp']}] peak {c['peak_cp']:+d} (static {st}, opp {c['opp_cp']}, SF {sfs}) `{c['peak_fen']}`\n")
    with open(a.out_epd,'w') as f:
        f.write("# Conversion SPIKE avoid-move suite\n")
        for c in cand:
            if c['bucket']=='CONV-SPIKE' and c.get('_throw'):
                f4=' '.join(c['_throw'][1].split()[:4])
                f.write(f'{f4} am {c["_throw"][3]}; id "conv-{c["opp"]}-p{c["peak_ply"]}"; c0 "spike peak{c["peak_cp"]:+d}";\n')
    print(f"DONE: {dict(by)}")
    print(f"cross-tab: {dict(xt)}")

if __name__=='__main__':
    main()
