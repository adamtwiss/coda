#!/usr/bin/env python3
"""
bishop_sortie_validate.py — deep-SF validation of pre-filtered bishop-sortie
candidates (from bishop_sortie_prefilter.py), the expensive stage of the
wandering-bishop corpus mine. Mirrors overrate_corpus.py's SF logic but only
runs on the cheap-pre-filtered candidate FENs (not every position).

For each candidate (fen_before, sortie_uci):
  1. SF deep -> sf_cp (stm POV), sf_bm.
  2. if sortie == sf_bm: drop (SF agrees -> not a blindspot).
  3. SF deep on the sortie (root_moves) -> sf_sortie_cp.
  4. move_loss = sf_cp - sf_sortie_cp  (how much worse the sortie is).
  5. keep if |sf_cp| <= blowout AND move_loss >= gap AND quiet (sf_bm not a
     capture/check; sortie already quiet from the pre-filter).
  6. (optional, --coda-net) net_pref gate: current net's OWN eval delta
     (after-sortie minus after-bm, mover POV) -> keep only sorties the CURRENT
     net still over-credits (net_pref >= --np-min). This is what makes each
     corpus position discriminate between nets.

Emits an avoid-move EPD compatible with testdata/overrate.epd:
  <fen4> bm <sf_bm>; am <sortie>; id "..."; c0 "ml=..;np=..;dis=..;src=..";
plus a rich TSV. Dedups near-identical FENs and caps positions per source game.
"""
import argparse, sys, subprocess, re, chess, chess.engine

def cp_stm(info, board):
    return info['score'].pov(board.turn).score(mate_score=100000)

NN_RE = re.compile(r"NNUE evaluation\s+([+-][0-9.]+)")

def coda_eval(coda_bin, net, fen):
    inp = f"position fen {fen}\neval\nquit\n"
    o = subprocess.run([coda_bin, "-n", net], input=inp, capture_output=True,
                       text=True, timeout=30).stdout
    m = NN_RE.findall(o)
    return float(m[0]) * 100 if m else None  # -> centipawns

def net_pref(coda_bin, net, board, sortie_mv, bm_mv):
    """current net's own eval: (after sortie) - (after bm), mover POV, cp."""
    mover = board.turn
    b1 = board.copy(); b1.push(sortie_mv); va = coda_eval(coda_bin, net, b1.fen())
    b2 = board.copy(); b2.push(bm_mv);     vb = coda_eval(coda_bin, net, b2.fen())
    if va is None or vb is None: return None
    s = 1 if mover == chess.WHITE else -1
    return s * va - s * vb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True, help='candidate TSV from prefilter')
    ap.add_argument('--sf', default='stockfish')
    ap.add_argument('--sf-depth', type=int, default=22)
    ap.add_argument('--sf-threads', type=int, default=2)
    ap.add_argument('--sf-maxtime', type=float, default=15.0)
    ap.add_argument('--sf-hash', type=int, default=512)
    ap.add_argument('--gap-thresh', type=int, default=150, help='min move_loss (cp)')
    ap.add_argument('--blowout', type=int, default=600, help='skip if |sf_cp|>this')
    ap.add_argument('--coda', default='./coda')
    ap.add_argument('--coda-net', default=None, help='enable net_pref gate with this .nnue')
    ap.add_argument('--np-min', type=float, default=0.0, help='min net_pref (cp) to keep')
    ap.add_argument('--cap-per-game', type=int, default=1)
    ap.add_argument('--max-cand', type=int, default=0, help='0 = all')
    ap.add_argument('--out-epd', default='/tmp/sortie_corpus.epd')
    ap.add_argument('--out-tsv', default='/tmp/sortie_corpus.tsv')
    args = ap.parse_args()

    rows = []
    with open(args.cand) as f:
        hdr = f.readline().rstrip('\n').split('\t')
        idx = {k: i for i, k in enumerate(hdr)}
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) < len(hdr): continue
            rows.append(p)
    if args.max_cand: rows = rows[:args.max_cand]

    sf = chess.engine.SimpleEngine.popen_uci(args.sf)
    sf.configure({'Threads': args.sf_threads, 'Hash': args.sf_hash})
    def sf_root_cp(board, mv):
        try:
            mi = sf.analyse(board, chess.engine.Limit(depth=args.sf_depth, time=args.sf_maxtime),
                            root_moves=[mv])
            return cp_stm(mi, board)
        except Exception:
            return None

    epd = open(args.out_epd, 'w'); tsv = open(args.out_tsv, 'w')
    tsv.write("idx\tmove_loss\tnet_pref\tdisagree\tsf_cp\tbm\tsortie\tgame\tfen\n")
    seen_fen = set(); per_game = {}; kept = 0
    for i, p in enumerate(rows):
        fen = p[idx['fen_before']]; sortie_uci = p[idx['uci']]
        gid = p[idx['gid']].split(':')[0]
        dis = p[idx['disagree']] if 'disagree' in idx else 'NA'
        key = ' '.join(fen.split()[:4])
        if key in seen_fen: continue
        if per_game.get(gid, 0) >= args.cap_per_game: continue
        board = chess.Board(fen)
        try:
            sortie_mv = chess.Move.from_uci(sortie_uci)
            si = sf.analyse(board, chess.engine.Limit(depth=args.sf_depth, time=args.sf_maxtime))
        except Exception:
            continue
        sf_cp = cp_stm(si, board); bm = si['pv'][0] if si.get('pv') else None
        if bm is None or sf_cp is None or abs(sf_cp) > args.blowout: continue
        if bm == sortie_mv: continue                      # SF agrees -> not a blindspot
        if board.is_capture(bm) or board.gives_check(bm): continue   # bm must be quiet
        s_cp = sf_root_cp(board, sortie_mv)
        if s_cp is None: continue
        move_loss = sf_cp - s_cp
        if move_loss < args.gap_thresh: continue
        np = None
        if args.coda_net:
            np = net_pref(args.coda, args.coda_net, board, sortie_mv, bm)
            if np is None or np < args.np_min: continue
        bm_san = board.san(bm); sortie_san = board.san(sortie_mv)
        fen4 = key
        c0 = f"ml={move_loss};np={'%.0f'%np if np is not None else 'NA'};dis={dis};src={gid}"
        epd.write(f'{fen4} bm {bm_san}; am {sortie_san}; id "sortie-{kept}"; c0 "{c0}";\n')
        tsv.write(f"{i}\t{move_loss}\t{'%.0f'%np if np is not None else 'NA'}\t{dis}\t"
                  f"{sf_cp}\t{bm_san}\t{sortie_san}\t{gid}\t{fen}\n")
        epd.flush(); tsv.flush()
        seen_fen.add(key); per_game[gid] = per_game.get(gid, 0) + 1; kept += 1
        if kept % 10 == 0:
            sys.stderr.write(f"  validated {i+1}/{len(rows)} candidates, kept {kept}\n")
    epd.close(); tsv.close(); sf.quit()
    print(f"VALIDATE DONE: {len(rows)} candidates -> {kept} kept -> {args.out_epd}")

if __name__ == '__main__':
    main()
