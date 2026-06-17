#!/usr/bin/env python3
"""
overrate_corpus.py — mine real Coda games for positions where Coda's
game-speed search prefers a move that deep Stockfish refutes: the engine's
systematic search/eval blindspots (the RpZ9LbYM move-35 Bd2 class —
under-seeing a quiet refutation despite reaching nominal depth).

Primary metric is MOVE-LOSS, not position optimism: Coda can be losing
overall yet still pick a move that throws another N cp. For each Coda-to-move
position (move range, not in check):
  1. Coda at game-realistic effort -> coda_move (cmove), coda_cp.
  2. SF deep on the position    -> sf_move, sf_cp (ground truth).
  3. if cmove != sf_move and the position isn't already a blowout:
       SF deep on Coda's move (root_moves) -> sf_codamove_cp.
       move_loss = sf_cp - sf_codamove_cp   (how much worse Coda's move is).
  Flag when move_loss >= --gap-thresh.  "quiet" when neither sf_move nor
  cmove is a capture/check and stm isn't in check (positional misjudgement,
  not a tactic Coda is mid-calculating). All cp from side-to-move POV.

Modes:
  scan  (default): --pgn ... --user ...   -> writes a rich TSV.
  merge: --build-from-tsv a.tsv b.tsv ... -> MD corpus + avoid-move EPD.

EPD line: `<fen4> bm <sf_move>; am <cmove>; id "..."; c0 "...";`
"""
import argparse, sys, chess, chess.pgn, chess.engine

# played_loss = SF(best) - SF(move the bot actually played)  -> real-game error
# coda_loss   = SF(best) - SF(current-Coda's preferred move) -> LIVE blindspot
# A position is flagged on max(played_loss, coda_loss). "live" when current
# Coda still prefers a refuted move; else "hist" (real-game error trunk fixed).
FIELDS = ['played_loss','coda_loss','live','quiet','res','gid','mv',
          'coda_cp','cmove','coda_uci','sf_cp','sfmove','sf_uci',
          'played','played_uci','fen']

def iter_games(paths):
    for p in paths:
        with open(p) as f:
            while (g := chess.pgn.read_game(f)) is not None:
                yield g

def coda_side(g, users):
    w, b = g.headers.get('White',''), g.headers.get('Black','')
    if w in users: return chess.WHITE
    if b in users: return chess.BLACK
    return None

def result_for(g, us_white):
    r = g.headers.get('Result','')
    if r == '1/2-1/2': return 'D'
    if r in ('1-0','0-1'): return 'W' if (r=='1-0')==us_white else 'L'
    return '?'

def cp_stm(info, board):
    return info['score'].pov(board.turn).score(mate_score=100000)

def scan(args):
    users = set(args.user)
    coda = chess.engine.SimpleEngine.popen_uci(args.coda)
    coda.configure({'Threads': args.coda_threads, 'Hash': 256})
    if args.coda_net: coda.configure({'NNUEFile': args.coda_net})
    sf = chess.engine.SimpleEngine.popen_uci(args.sf)
    sf.configure({'Threads': args.sf_threads, 'Hash': args.sf_hash})

    out = open(args.out_tsv, 'w')
    out.write('\t'.join(FIELDS) + '\n')
    def sf_move_cp(board, uci):
        try:
            mi = sf.analyse(board, chess.engine.Limit(depth=args.sf_depth), root_moves=[uci])
            return cp_stm(mi, board)
        except Exception:
            return None
    n_games = n_pos = n_flag = 0
    for g in iter_games(args.pgn):
        side = coda_side(g, users)
        if side is None: continue
        if args.results and result_for(g, side==chess.WHITE) not in args.results: continue
        n_games += 1
        if n_games > args.max_games: break
        res = result_for(g, side==chess.WHITE)
        gid = g.headers.get('Site','').rsplit('/',1)[-1]
        board = g.board()
        for mv in g.mainline_moves():
            movno = board.fullmove_number
            if board.turn == side and args.min_move <= movno <= args.max_move and not board.is_check():
                n_pos += 1
                fen = board.fen(); played = board.san(mv); played_mv = mv  # Move object
                try:
                    ci = coda.analyse(board, chess.engine.Limit(time=args.coda_movetime))
                    si = sf.analyse(board, chess.engine.Limit(depth=args.sf_depth))
                except Exception:
                    board.push(mv); continue
                ccp = cp_stm(ci, board); scp = cp_stm(si, board)
                coda_mv = ci['pv'][0] if ci.get('pv') else None      # Move
                sf_mv = si['pv'][0] if si.get('pv') else None        # Move
                if coda_mv is None or sf_mv is None: board.push(mv); continue
                cmove = board.san(coda_mv); sfmove = board.san(sf_mv)
                played_loss = coda_loss = ''
                if abs(scp) <= args.blowout:
                    if played_mv != sf_mv:
                        c = sf_move_cp(board, played_mv)
                        if c is not None: played_loss = scp - c
                    if coda_mv != sf_mv:
                        if coda_mv == played_mv and isinstance(played_loss, int):
                            coda_loss = played_loss
                        else:
                            cc = sf_move_cp(board, coda_mv)
                            if cc is not None: coda_loss = scp - cc
                quiet = int(not board.is_capture(sf_mv) and not board.gives_check(sf_mv))
                live = int(isinstance(coda_loss,int) and coda_loss >= args.gap_thresh)
                if (isinstance(played_loss,int) and played_loss>=args.gap_thresh) or live: n_flag += 1
                row = dict(played_loss=played_loss, coda_loss=coda_loss, live=live, quiet=quiet,
                           res=res, gid=gid, mv=movno, coda_cp=ccp, cmove=cmove, coda_uci=coda_mv.uci(),
                           sf_cp=scp, sfmove=sfmove, sf_uci=sf_mv.uci(), played=played, played_uci=played_mv.uci(), fen=fen)
                out.write('\t'.join(str(row[k]) for k in FIELDS) + '\n'); out.flush()
            board.push(mv)
        print(f"  {gid} ({res}): pos={n_pos} flagged={n_flag}", file=sys.stderr)
    out.close(); coda.quit(); sf.quit()
    print(f"SCAN DONE: {n_games} games, {n_pos} positions, {n_flag} flagged -> {args.out_tsv}")

def build(args):
    def toi(x):
        try: return int(x)
        except: return None
    rows = []
    for p in args.build_from_tsv:
        with open(p) as f:
            hdr = f.readline().rstrip('\n').split('\t')
            for ln in f:
                d = dict(zip(hdr, ln.rstrip('\n').split('\t')))
                d['played_loss']=toi(d.get('played_loss','')); d['coda_loss']=toi(d.get('coda_loss',''))
                for k in ('quiet','live','mv','coda_cp','sf_cp'): d[k]=toi(d.get(k,'')) or 0
                rows.append(d)
    def worst(r):  # the move-to-avoid for this row = whichever lost more
        # Focus on blindspots that mattered: skip positions where Coda was
        # already lost (best-defence-in-a-lost-position is low fix-value).
        if args.min_pos_cp is not None and r['sf_cp'] < args.min_pos_cp:
            return None
        pl, cl = r['played_loss'], r['coda_loss']
        cand=[]
        if cl is not None and cl>=args.gap_thresh and r['coda_uci']!=r['sf_uci']:
            cand.append((cl, r['cmove'], r['coda_uci'], 'live'))
        if pl is not None and pl>=args.gap_thresh and r['played_uci']!=r['sf_uci']:
            cand.append((pl, r['played'], r['played_uci'], 'played'))
        return max(cand) if cand else None
    flag=[]
    for r in rows:
        w = worst(r)
        if w and r['sfmove']!='?':
            r=dict(r); r['_loss'],r['_am'],r['_am_uci'],r['_kind']=w; flag.append(r)
    flag.sort(key=lambda r: -r['_loss'])
    seen=set(); uniq=[]
    for r in flag:
        f4=' '.join(r['fen'].split()[:4])
        if f4 in seen: continue
        seen.add(f4); uniq.append(r)
    nlive=sum(1 for r in uniq if r['live']); nq=sum(1 for r in uniq if r['quiet'])
    with open(args.out_md,'w') as f:
        f.write(f"# Coda overrate corpus — {len(uniq)} avoid-move positions "
                f"({nlive} LIVE current-Coda blindspots, {nq} quiet), move_loss>={args.gap_thresh}cp by SF\n\n")
        f.write("`am` = move to avoid (lost `move_loss` cp vs `bm` per SF deep). "
                "**LIVE** = current main still prefers a refuted move here; "
                "*played* = the bot's real-game move (trunk may already avoid it). "
                "quiet = SF best is non-capture/non-check.\n\n")
        for r in uniq:
            tag = '**LIVE**' if r['_kind']=='live' else 'played'
            q = 'quiet' if r['quiet'] else 'tactical'
            f.write(f"## −{r['_loss']}cp  [{r['res']} {q} {tag}] {r['gid']} move {r['mv']}\n"
                    f"- avoid **{r['_am']}** → play **{r['sfmove']}** (SF {r['sf_cp']:+d}); "
                    f"current Coda prefers {r['cmove']} ({r['coda_cp']:+d})\n"
                    f"- `{r['fen']}`  https://lichess.org/{r['gid']}\n\n")
    with open(args.out_epd,'w') as f:
        f.write(f"# Coda overrate-corpus avoid-move suite ({len(uniq)} pos, {nlive} LIVE). "
                "bm = play this; am = avoid (SF-refuted). LIVE = current main still errs.\n")
        for r in uniq:
            f4=' '.join(r['fen'].split()[:4])
            f.write(f'{f4} bm {r["sfmove"]}; am {r["_am"]}; id "ovr-{r["gid"]}-m{r["mv"]}"; '
                    f'c0 "{r["_kind"]} loss{r["_loss"]} sf{r["sf_cp"]:+d} {r["res"]}";\n')
    print(f"BUILD DONE: {len(uniq)} unique avoid-move ({nlive} LIVE, {nq} quiet) -> {args.out_md}, {args.out_epd}")
    for r in uniq[:18]:
        print(f"  −{r['_loss']:>4}cp [{r['res']} {r['_kind']:>6}] {r['gid']} m{r['mv']}: avoid {r['_am']} play {r['sfmove']}(sf{r['sf_cp']:+d})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pgn', nargs='+'); ap.add_argument('--user', nargs='+')
    ap.add_argument('--coda', default='./coda'); ap.add_argument('--sf', default='stockfish')
    ap.add_argument('--coda-net', default=None)
    ap.add_argument('--coda-movetime', type=float, default=0.6); ap.add_argument('--coda-threads', type=int, default=1)
    ap.add_argument('--sf-depth', type=int, default=22); ap.add_argument('--sf-threads', type=int, default=2)
    ap.add_argument('--sf-hash', type=int, default=512)
    ap.add_argument('--gap-thresh', type=int, default=150)
    ap.add_argument('--blowout', type=int, default=600, help='skip move-loss when |sf_cp|>this')
    ap.add_argument('--min-move', type=int, default=12); ap.add_argument('--max-move', type=int, default=55)
    ap.add_argument('--max-games', type=int, default=99999)
    ap.add_argument('--results', nargs='+', default=None, help='filter to results e.g. L D')
    ap.add_argument('--out-tsv', default='/tmp/overrate.tsv')
    ap.add_argument('--out-md', default='/tmp/overrate.md'); ap.add_argument('--out-epd', default='/tmp/overrate.epd')
    ap.add_argument('--build-from-tsv', nargs='+', default=None)
    ap.add_argument('--min-pos-cp', type=int, default=None,
                    help='build: only flag positions where SF pos-eval >= this (skip already-lost)')
    args = ap.parse_args()
    if args.build_from_tsv: build(args)
    else:
        assert args.pgn and args.user, "scan mode needs --pgn and --user"
        scan(args)

if __name__ == '__main__':
    main()
