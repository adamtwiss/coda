#!/usr/bin/env python3
"""game_analysis.py — why is Coda losing / drawing winnable games?

Input: scored gauntlet PGNs where Coda plays every game vs a top-N opponent and
both engines annotate each move with `{eval/depth time}` (STM-POV). Coda is
always one of the two players.

Core idea
---------
Work in **Coda-POV** (how good Coda thinks *its own side* is doing) so that
"Coda overrates itself" is one signal regardless of color:

    coda_self    = Coda's eval, in Coda's POV
    opp_of_coda  = -(opponent's eval in opp POV)   # opp's view of Coda's position
    overrate     = coda_self - opp_of_coda          # >0: Coda more optimistic

The opponents are strong engines, so a *sustained* large `overrate` that ends in
a Coda draw/loss is a prime blindspot candidate. SF is then used as a trusted
3rd party (`oracle` stage) to confirm Coda is genuinely overrating before any
position is emitted as training-correction data.

Stages (subcommands)
--------------------
  taxonomy <pgn>                 classify every game by (eval trajectory x result)
  mine <pgn> --out cand.tsv      emit big sustained-disagreement positions
  oracle cand.tsv --out scored.tsv   SF 3rd-party score -> confirmed overrates
  features scored.tsv            common chess features of confirmed overrates

`mine` output feeds `oracle`; `oracle` output feeds `features` and is the
SF-scored training-correction set.
"""
import argparse
import re
import sys
import subprocess
from collections import defaultdict, Counter

import chess
import chess.pgn

MATE_CP = 30000
# python-chess strips the {} so node.comment is e.g. "+0.35/12 0.37s".
# Forms: "+0.35/12 0.37s", "-M3/30 0.1s", "0.00/22 0.11s".
EVAL_RE = re.compile(r'([+-]?)(M)?(\d+(?:\.\d+)?)/(\d+)\s+([0-9.]+)s')


def parse_comment(comment):
    """Return (cp_stm_pov, depth, time_s) or (None, None, None). STM-POV."""
    m = EVAL_RE.search(comment or "")
    if not m:
        return None, None, None
    sign, mate, num, depth, t = m.groups()
    val = float(num)
    if mate:
        cp = MATE_CP - int(val) * 100
        cp = -cp if sign == '-' else cp
    else:
        cp = int(round(val * 100))
        cp = -cp if sign == '-' else cp
    return cp, int(depth), float(t)


class Ply:
    __slots__ = ('ply', 'fullmove', 'mover_white', 'fen_before', 'san', 'uci',
                 'mover_cp', 'depth', 'time', 'is_capture', 'is_check')


def read_games(pgn_path):
    """Yield dicts: one per game with Coda-POV eval trajectories."""
    with open(pgn_path) as fh:
        while True:
            try:
                game = chess.pgn.read_game(fh)
            except Exception:
                continue
            if game is None:
                break
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            if white != "Coda" and black != "Coda":
                continue
            coda_white = (white == "Coda")
            opp = black if coda_white else white
            res = game.headers.get("Result", "*")
            # Coda-POV result: +1 win, 0 draw, -1 loss
            if res == "1/2-1/2":
                coda_res = 0
            elif res == "1-0":
                coda_res = 1 if coda_white else -1
            elif res == "0-1":
                coda_res = -1 if coda_white else 1
            else:
                continue  # unfinished

            board = game.board()
            plies = []
            for node in game.mainline():
                mv = node.move
                mover_white = board.turn == chess.WHITE
                cp, depth, t = parse_comment(node.comment)
                p = Ply()
                p.ply = len(plies)
                p.fullmove = board.fullmove_number
                p.mover_white = mover_white
                p.fen_before = board.fen()
                p.san = board.san(mv)
                p.uci = mv.uci()
                p.is_capture = board.is_capture(mv)
                p.mover_cp = cp  # STM-POV (i.e. the mover's own POV)
                p.depth = depth
                p.time = t
                board.push(mv)
                p.is_check = board.is_check()
                plies.append(p)

            # Build per-ply Coda-POV series via carry-forward of each engine's
            # most recent eval. mover_cp is the mover's-own-POV eval.
            coda_self_series = [None] * len(plies)   # Coda's POV of Coda
            opp_of_coda_series = [None] * len(plies)  # opp's POV of Coda (=-opp_self)
            last_coda = None
            last_opp_of_coda = None
            for i, p in enumerate(plies):
                mover_is_coda = (p.mover_white == coda_white)
                if p.mover_cp is not None:
                    if mover_is_coda:
                        last_coda = p.mover_cp  # already Coda-POV (Coda is mover)
                    else:
                        last_opp_of_coda = -p.mover_cp  # opp-POV -> Coda-POV
                coda_self_series[i] = last_coda
                opp_of_coda_series[i] = last_opp_of_coda

            yield {
                'white': white, 'black': black, 'opp': opp,
                'coda_white': coda_white, 'result': res, 'coda_res': coda_res,
                'plies': plies,
                'coda_self': coda_self_series,
                'opp_of_coda': opp_of_coda_series,
                'date': game.headers.get('Date', '?'),
                'round': game.headers.get('Round', '?'),
            }


def game_id(g):
    return f"{g['opp']}-r{g['round']}-{'W' if g['coda_white'] else 'B'}"


# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------
def classify_game(g, win_thresh=150, overrate_thresh=120, plateau_len=8,
                  step_cp=200, step_window=4):
    """Return (label, detail dict). Buckets prioritized worst-first."""
    cs = g['coda_self']
    oc = g['opp_of_coda']
    res = g['coda_res']
    n = len(g['plies'])
    # overrate where both evals known
    overr = [(cs[i] - oc[i]) if (cs[i] is not None and oc[i] is not None) else None
             for i in range(n)]
    valid = [(i, o) for i, o in enumerate(overr) if o is not None]

    # max sustained overrate: longest run where overrate >= overrate_thresh AND
    # Coda thinks it's winning (coda_self >= win_thresh)
    best_run = 0
    run = 0
    run_peak = 0
    cur_peak = 0
    for i in range(n):
        if overr[i] is not None and overr[i] >= overrate_thresh and cs[i] is not None and cs[i] >= win_thresh:
            run += 1
            cur_peak = max(cur_peak, overr[i])
            if run > best_run:
                best_run = run
                run_peak = cur_peak
        else:
            run = 0
            cur_peak = 0

    # peak Coda self-eval and final-stretch self-eval
    cs_known = [c for c in cs if c is not None]
    peak_self = max(cs_known) if cs_known else 0
    min_self = min(cs_known) if cs_known else 0

    # tactical step-change: an engine's self-view jumps >= step_cp within
    # step_window plies, the other lags. Detect who-saw-first.
    coda_jump_first = opp_jump_first = 0
    for i in range(step_window, n):
        if cs[i] is not None and cs[i - step_window] is not None:
            if cs[i] - cs[i - step_window] >= step_cp:
                # did opp already see it (its view of Coda rose too)?
                if oc[i] is not None and oc[i - step_window] is not None and \
                        (oc[i] - oc[i - step_window]) < step_cp / 2:
                    coda_jump_first += 1
        if oc[i] is not None and oc[i - step_window] is not None:
            if oc[i] - oc[i - step_window] >= step_cp:
                if cs[i] is not None and cs[i - step_window] is not None and \
                        (cs[i] - cs[i - step_window]) < step_cp / 2:
                    opp_jump_first += 1

    detail = {
        'best_overrate_run': best_run, 'run_peak_overrate': run_peak,
        'peak_self': peak_self, 'min_self': min_self,
        'coda_jump_first': coda_jump_first, 'opp_jump_first': opp_jump_first,
        'plies': n,
    }

    sustained = best_run >= plateau_len
    # Worst-first labeling
    if res == -1 and peak_self >= win_thresh and sustained:
        return 'overrate_then_lost', detail
    if res == 0 and sustained:
        return 'overrate_then_drew', detail   # had it, didn't convert
    if res == -1:
        # agreed loss (opp also thought Coda worse) vs surprise loss
        if sustained:
            return 'overrate_then_lost', detail
        return 'agreed_loss', detail
    if res == 0 and min_self <= -win_thresh:
        return 'salvaged_draw', detail        # thought losing, escaped
    if res == 1 and min_self <= -win_thresh:
        return 'won_from_behind', detail
    if res == 1:
        return 'clean_win', detail
    return 'quiet_draw', detail


def cmd_taxonomy(args):
    buckets = defaultdict(list)
    opp_overrate = defaultdict(lambda: [0, 0, 0])  # opp -> [n_overrate_draw/loss, n_games, sum_run]
    total = 0
    for g in read_games(args.pgn):
        total += 1
        label, d = classify_game(g, win_thresh=args.win_thresh,
                                  overrate_thresh=args.overrate_thresh,
                                  plateau_len=args.plateau_len)
        buckets[label].append((game_id(g), g, d))
        st = opp_overrate[g['opp']]
        st[1] += 1
        st[2] += d['best_overrate_run']
        if label in ('overrate_then_lost', 'overrate_then_drew'):
            st[0] += 1

    print(f"\n=== TAXONOMY: {total} games from {args.pgn} ===\n")
    order = ['overrate_then_lost', 'overrate_then_drew', 'agreed_loss',
             'salvaged_draw', 'won_from_behind', 'clean_win', 'quiet_draw']
    for label in order:
        rows = buckets.get(label, [])
        if not rows:
            continue
        print(f"[{label}]  {len(rows)} games")
        if label in ('overrate_then_lost', 'overrate_then_drew'):
            rows.sort(key=lambda r: (-r[2]['run_peak_overrate'], -r[2]['best_overrate_run']))
            for gid, g, d in rows[:args.show]:
                print(f"    {gid:32s} run={d['best_overrate_run']:2d} "
                      f"peakΔ={d['run_peak_overrate']/100:+.2f} "
                      f"peakSelf={d['peak_self']/100:+.2f} plies={d['plies']}")
        print()

    print("=== Per-opponent overrate-then-(draw|loss) rate ===")
    rank = sorted(opp_overrate.items(), key=lambda kv: -(kv[1][0] / max(1, kv[1][1])))
    print(f"  {'opponent':16s} {'bad':>4s} {'games':>5s} {'rate':>6s} {'avgRun':>7s}")
    for opp, (bad, games, sumrun) in rank:
        print(f"  {opp:16s} {bad:>4d} {games:>5d} {bad/max(1,games):>6.2f} {sumrun/max(1,games):>7.1f}")

    # step-change summary (effective-depth lever)
    jump = sum(1 for L in buckets.values() for _, _, d in L if d['opp_jump_first'] > d['coda_jump_first'])
    coda_first = sum(1 for L in buckets.values() for _, _, d in L if d['coda_jump_first'] > d['opp_jump_first'])
    print(f"\n=== Tactical step-change lead ===")
    print(f"  opp saw step-change first more often : {jump} games")
    print(f"  Coda saw step-change first more often: {coda_first} games")


# ---------------------------------------------------------------------------
# mine — emit big sustained-disagreement candidate positions
# ---------------------------------------------------------------------------
def cmd_mine(args):
    out = open(args.out, 'w')
    out.write("gid\tply\tfullmove\tcoda_res\tcoda_self\topp_of_coda\toverrate\tsan\tfen_before\n")
    n_cand = 0
    n_games = 0
    for g in read_games(args.pgn):
        # Result filter: a big eval that WON is a correct eval, not an overrate.
        # Default to draws/losses only (coda_res <= max_result).
        if g['coda_res'] > args.max_result:
            continue
        n_games += 1
        cs = g['coda_self']
        oc = g['opp_of_coda']
        plies = g['plies']
        gid = game_id(g)
        # find sustained overrate windows; emit the peak position in each window
        i = 0
        N = len(plies)
        while i < N:
            o = (cs[i] - oc[i]) if (cs[i] is not None and oc[i] is not None) else None
            if (o is not None and o >= args.min_overrate and cs[i] is not None
                    and args.min_self <= cs[i] < args.max_self):
                # extend window
                j = i
                peak_i = i
                peak_o = o
                while j < N:
                    oj = (cs[j] - oc[j]) if (cs[j] is not None and oc[j] is not None) else None
                    if oj is None or oj < args.min_overrate or cs[j] is None or cs[j] < args.min_self:
                        break
                    if oj > peak_o:
                        peak_o = oj
                        peak_i = j
                    j += 1
                run = j - i
                if run >= args.min_run:
                    # emit the position BEFORE the peak Coda move (a Coda move,
                    # so SF can judge Coda's own move/eval). Walk to a Coda ply.
                    pe = peak_i
                    while pe >= i and (plies[pe].mover_white != g['coda_white']):
                        pe -= 1
                    if pe < i:
                        pe = peak_i
                    p = plies[pe]
                    out.write(f"{gid}\t{p.ply}\t{p.fullmove}\t{g['coda_res']}\t"
                              f"{cs[pe]}\t{oc[pe]}\t{cs[pe]-oc[pe] if oc[pe] is not None else ''}\t"
                              f"{p.san}\t{p.fen_before}\n")
                    n_cand += 1
                i = j
            else:
                i += 1
    out.close()
    print(f"mined {n_cand} candidate positions from {n_games} games -> {args.out}")
    print(f"  filters: min_overrate={args.min_overrate}cp min_self={args.min_self}cp min_run={args.min_run}")


# ---------------------------------------------------------------------------
# oracle — SF trusted 3rd-party score
# ---------------------------------------------------------------------------
class SF:
    def __init__(self, path, threads=1, hashmb=256):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  universal_newlines=True, bufsize=1)
        self._cmd("uci", "uciok")
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hashmb}")
        self._cmd("isready", "readyok")

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _cmd(self, cmd, until):
        self.send(cmd)
        while True:
            line = self.p.stdout.readline()
            if not line or line.strip() == until:
                break

    def eval_fen(self, fen, depth):
        """Return (cp_white_pov, bestmove_uci). cp is STM-POV from SF then -> white."""
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        last_cp = None
        is_mate = False
        best = None
        stm_white = fen.split()[1] == 'w'
        while True:
            line = self.p.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("info ") and " score " in line and " pv " in line:
                toks = line.split()
                try:
                    si = toks.index("score")
                    kind = toks[si + 1]
                    val = int(toks[si + 2])
                    if kind == "cp":
                        last_cp = val
                        is_mate = False
                    elif kind == "mate":
                        last_cp = MATE_CP - abs(val) * 100
                        last_cp = -last_cp if val < 0 else last_cp
                        is_mate = True
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                best = line.split()[1]
                break
        if last_cp is None:
            return None, best
        # SF score is STM-POV -> convert to white-POV
        cp_white = last_cp if stm_white else -last_cp
        return cp_white, best

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


class Coda:
    """Clean Coda re-eval on a chosen net. Two trustworthy numbers per FEN:

      static  -- `eval` command, NNUE static eval (white-side), -> Coda-POV
      search  -- fixed-time `go movetime`, final info `score cp`, STM-POV (= Coda-POV)

    Replaces the in-game PGN eval (parse/POV/time gotchas) used by the old
    oracle. The net MUST be the one that played the gauntlet — pass it via -n.
    """
    EVAL_RE = re.compile(r'evaluation\s+([+-]?\d+(?:\.\d+)?)')

    def __init__(self, path, net, movetime=2000, no_threats=False):
        self.path = path
        self.net = net
        self.movetime = movetime
        self.no_threats = no_threats

    def _env(self):
        import os
        env = dict(os.environ)
        if self.no_threats:
            env['CODA_NO_THREAT_ACC'] = '1'
        return env

    def static_codapov(self, fen):
        """NNUE static eval in centipawns, Coda-POV (Coda = STM at fen).

        Uses the UCI `eval` command (white-side print). NOTE: UCI `go movetime`
        emits NO info lines in Coda, so search uses the epd path below instead.
        """
        script = ("uci\nsetoption name OwnBook value false\n"
                  f"position fen {fen}\neval\nquit\n")
        p = subprocess.run([self.path, '-n', self.net], input=script,
                           capture_output=True, text=True, env=self._env(),
                           timeout=120)
        m = self.EVAL_RE.search(p.stdout)
        if not m:
            return None
        cp_white = int(round(float(m.group(1)) * 100))
        stm_white = fen.split()[1] == 'w'
        return cp_white if stm_white else -cp_white

    def search_codapov(self, fen):
        """Fixed-time search score (cp), Coda-POV (STM-POV == Coda-POV here).

        Routed through `coda epd <file> -t <ms>` (the only path that prints
        per-iteration info lines). Returns the deepest-iteration score.
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile('w', suffix='.epd', delete=False) as tf:
            tf.write(fen + "\n")
            epd = tf.name
        try:
            p = subprocess.run([self.path, 'epd', epd, '--nnue', self.net,
                                '-t', str(self.movetime)],
                               capture_output=True, text=True, env=self._env(),
                               timeout=max(30, self.movetime // 1000 * 4 + 30))
        finally:
            os.unlink(epd)
        last = None
        for line in p.stdout.splitlines():
            if line.startswith("info ") and " score " in line:
                t = line.split()
                try:
                    si = t.index("score")
                    if t[si + 1] == "cp":
                        last = int(t[si + 2])
                    elif t[si + 1] == "mate":
                        v = int(t[si + 2])
                        last = (MATE_CP - abs(v) * 100) * (1 if v >= 0 else -1)
                except (ValueError, IndexError):
                    pass
        return last


def cmd_oracle(args):
    sf = SF(args.sf, threads=args.threads, hashmb=args.hash)
    rows = []
    with open(args.cand) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            rows.append(f)
    out = open(args.out, 'w')
    out.write("gid\tfullmove\tcoda_res\tcoda_self\topp_of_coda\tsf_coda_pov\t"
              "true_overrate\tsf_best\tsan\tfen_before\n")
    kept = 0
    for n, f in enumerate(rows):
        if args.max and n >= args.max:
            break
        fen = f[idx['fen_before']]
        coda_self = int(f[idx['coda_self']])
        oc = f[idx['opp_of_coda']]
        coda_white = fen.split()[1] == 'w'  # Coda is the mover at fen_before
        cp_white, best = sf.eval_fen(fen, args.depth)
        if cp_white is None:
            continue
        sf_coda_pov = cp_white if coda_white else -cp_white
        true_overrate = coda_self - sf_coda_pov
        # keep only positions SF confirms Coda overrates its own side
        if true_overrate >= args.min_true_overrate and abs(sf_coda_pov) <= args.sf_cap:
            out.write(f"{f[idx['gid']]}\t{f[idx['fullmove']]}\t{f[idx['coda_res']]}\t"
                      f"{coda_self}\t{oc}\t{sf_coda_pov}\t{true_overrate}\t{best}\t"
                      f"{f[idx['san']]}\t{fen}\n")
            kept += 1
        if (n + 1) % 25 == 0:
            print(f"  ...{n+1}/{len(rows)} scored, {kept} confirmed", file=sys.stderr)
    out.close()
    sf.quit()
    print(f"oracle: scored {min(len(rows), args.max or len(rows))} positions, "
          f"{kept} SF-confirmed overrates -> {args.out}")
    print(f"  filters: min_true_overrate={args.min_true_overrate}cp sf_cap={args.sf_cap}cp depth={args.depth}")


# ---------------------------------------------------------------------------
# features — common chess patterns of confirmed overrates
# ---------------------------------------------------------------------------
def extract_features(fen, coda_white):
    """Heuristic chess features of the position from Coda's POV."""
    b = chess.Board(fen)
    us = chess.WHITE if coda_white else chess.BLACK
    them = not us
    feats = []

    # piece counts / phase
    npieces = chess.popcount(b.occupied)
    if npieces <= 12:
        feats.append("phase:endgame")
    elif npieces <= 22:
        feats.append("phase:middlegame")
    else:
        feats.append("phase:opening")

    # our advanced minor on an exposed/forward square (the "sortie" pattern)
    for pt, name in ((chess.BISHOP, "bishop"), (chess.KNIGHT, "knight")):
        for sq in b.pieces(pt, us):
            rank = chess.square_rank(sq) if us == chess.WHITE else 7 - chess.square_rank(sq)
            if rank >= 4:
                # is it defended? is it attacked?
                attacked = b.is_attacked_by(them, sq)
                defended = b.is_attacked_by(us, sq)
                mob = len(list(b.attacks(sq)))
                tag = f"fwd_{name}_r{rank+1}"
                if attacked and not defended:
                    tag += "_hanging"
                elif attacked:
                    tag += "_contested"
                feats.append(tag)
                if mob <= 3:
                    feats.append(f"fwd_{name}_lowmob")

    # our queen far advanced
    for sq in b.pieces(chess.QUEEN, us):
        rank = chess.square_rank(sq) if us == chess.WHITE else 7 - chess.square_rank(sq)
        if rank >= 4:
            feats.append("fwd_queen")

    # passed-pawn-ish: our most advanced pawn
    pawns = b.pieces(chess.PAWN, us)
    if pawns:
        ranks = [chess.square_rank(s) if us == chess.WHITE else 7 - chess.square_rank(s) for s in pawns]
        if max(ranks) >= 5:
            feats.append("advanced_passer_cand")

    # king safety: is our king exposed (few pawn shield)?
    ksq = b.king(us)
    if ksq is not None:
        shield = 0
        kf = chess.square_file(ksq)
        for df in (-1, 0, 1):
            f = kf + df
            if 0 <= f <= 7:
                for s in b.pieces(chess.PAWN, us):
                    if chess.square_file(s) == f:
                        shield += 1
                        break
        if shield <= 1:
            feats.append("our_king_exposed")

    # material balance (Coda POV)
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    mat = 0
    for pt, v in vals.items():
        mat += v * (len(b.pieces(pt, us)) - len(b.pieces(pt, them)))
    if mat >= 1:
        feats.append("we_are_up_material")
    elif mat <= -1:
        feats.append("we_are_down_material")
    else:
        feats.append("material_equal")

    # opposite-colored bishops (drawish)
    wb = b.pieces(chess.BISHOP, chess.WHITE)
    bb = b.pieces(chess.BISHOP, chess.BLACK)
    if len(wb) == 1 and len(bb) == 1:
        wc = (chess.square_rank(list(wb)[0]) + chess.square_file(list(wb)[0])) % 2
        bc = (chess.square_rank(list(bb)[0]) + chess.square_file(list(bb)[0])) % 2
        if wc != bc:
            feats.append("opposite_bishops")

    return feats


def cmd_reeval(args):
    """Clean re-eval: replace the unreliable in-game PGN eval with fresh
    gauntlet-net Coda numbers (static + threats-off static + fixed-time search)
    and SF ground truth. Emits ALL candidates with every number so themeing can
    filter; `confirmed` marks search_overrate >= min AND |SF| <= cap."""
    sf = SF(args.sf, threads=args.threads, hashmb=args.hash)
    coda = Coda(args.coda, args.net, movetime=args.movetime)
    coda_noth = Coda(args.coda, args.net, movetime=args.movetime, no_threats=True)
    rows = []
    with open(args.cand) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= len(header):
                rows.append(f)
    out = open(args.out, 'w')
    out.write("gid\tfullmove\tcoda_res\tpgn_self\tsf_codapov\tcoda_static\t"
              "coda_static_noth\tcoda_search\tstatic_or\tstaticnoth_or\t"
              "search_or\tpgn_or\tconfirmed\tsf_best\tsan\tfen_before\n")
    kept = 0
    for n, f in enumerate(rows):
        if args.max and n >= args.max:
            break
        fen = f[idx['fen_before']]
        pgn_self = int(f[idx['coda_self']])
        coda_white = fen.split()[1] == 'w'
        cp_white, best = sf.eval_fen(fen, args.depth)
        if cp_white is None:
            continue
        sf_cp = cp_white if coda_white else -cp_white
        st = coda.static_codapov(fen)
        stn = coda_noth.static_codapov(fen)
        se = coda.search_codapov(fen)
        static_or = (st - sf_cp) if st is not None else None
        staticnoth_or = (stn - sf_cp) if stn is not None else None
        search_or = (se - sf_cp) if se is not None else None
        pgn_or = pgn_self - sf_cp
        confirmed = (search_or is not None and search_or >= args.min_search_overrate
                     and abs(sf_cp) <= args.sf_cap)
        if confirmed:
            kept += 1
        def s(x):
            return str(x) if x is not None else ''
        out.write(f"{f[idx['gid']]}\t{f[idx['fullmove']]}\t{f[idx['coda_res']]}\t"
                  f"{pgn_self}\t{sf_cp}\t{s(st)}\t{s(stn)}\t{s(se)}\t{s(static_or)}\t"
                  f"{s(staticnoth_or)}\t{s(search_or)}\t{pgn_or}\t{int(confirmed)}\t"
                  f"{best}\t{f[idx['san']]}\t{fen}\n")
        out.flush()
        if (n + 1) % 10 == 0:
            print(f"  ...{n+1}/{len(rows)} re-eval'd, {kept} confirmed", file=sys.stderr)
    out.close()
    sf.quit()
    print(f"reeval: scored {min(len(rows), args.max or len(rows))} positions, "
          f"{kept} search-confirmed overrates -> {args.out}")
    print(f"  net={args.net} movetime={args.movetime}ms sf_depth={args.depth} "
          f"min_search_overrate={args.min_search_overrate} sf_cap={args.sf_cap}")


def cmd_features(args):
    feat_count = Counter()
    res_count = Counter()
    n = 0
    with open(args.scored) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < len(header):
                continue
            n += 1
            fen = f[idx['fen_before']]
            coda_white = fen.split()[1] == 'w'
            for ft in extract_features(fen, coda_white):
                feat_count[ft] += 1
            res_count[int(f[idx['coda_res']])] += 1
    print(f"\n=== FEATURES of {n} SF-confirmed overrate positions ===\n")
    print(f"result of those games (Coda POV): "
          f"loss={res_count.get(-1,0)} draw={res_count.get(0,0)} win={res_count.get(1,0)}\n")
    print(f"  {'feature':28s} {'count':>6s} {'pct':>6s}")
    for ft, c in feat_count.most_common():
        print(f"  {ft:28s} {c:>6d} {100*c/max(1,n):>5.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('taxonomy', help='classify games by eval trajectory x result')
    t.add_argument('pgn')
    t.add_argument('--win-thresh', type=int, default=150, help='cp Coda must claim to "think winning"')
    t.add_argument('--overrate-thresh', type=int, default=120, help='cp gap for disagreement')
    t.add_argument('--plateau-len', type=int, default=8, help='plies of sustained overrate')
    t.add_argument('--show', type=int, default=15)
    t.set_defaults(func=cmd_taxonomy)

    m = sub.add_parser('mine', help='emit big sustained-disagreement positions')
    m.add_argument('pgn')
    m.add_argument('--out', required=True)
    m.add_argument('--min-overrate', type=int, default=150, help='cp Coda-vs-opp gap')
    m.add_argument('--min-self', type=int, default=120, help='cp Coda self-eval floor')
    m.add_argument('--max-self', type=int, default=3000,
                   help='cp Coda self-eval cap (excludes mate-range claims)')
    m.add_argument('--min-run', type=int, default=6, help='plies sustained')
    m.add_argument('--max-result', type=int, default=0,
                   help='emit games with coda_res <= this (0=draws/losses only; a big eval that WON is correct)')
    m.set_defaults(func=cmd_mine)

    o = sub.add_parser('oracle', help='SF 3rd-party score -> confirmed overrates')
    o.add_argument('cand')
    o.add_argument('--out', required=True)
    o.add_argument('--sf', default='/home/adam/chess/engines/Stockfish/src/stockfish')
    o.add_argument('--depth', type=int, default=20)
    o.add_argument('--threads', type=int, default=4)
    o.add_argument('--hash', type=int, default=256)
    o.add_argument('--min-true-overrate', type=int, default=150,
                   help='cp Coda overrates vs SF to confirm')
    o.add_argument('--sf-cap', type=int, default=600, help='|SF eval| cap (skip won/lost)')
    o.add_argument('--max', type=int, default=0, help='cap positions scored (0=all)')
    o.set_defaults(func=cmd_oracle)

    r = sub.add_parser('reeval', help='clean Coda re-eval (gauntlet net) + SF -> trustworthy overrates')
    r.add_argument('cand')
    r.add_argument('--out', required=True)
    r.add_argument('--net', required=True, help='gauntlet net that played the games')
    r.add_argument('--coda', default='/home/adam/code/coda/coda')
    r.add_argument('--sf', default='/home/adam/chess/engines/Stockfish/src/stockfish')
    r.add_argument('--depth', type=int, default=22, help='SF ground-truth depth')
    r.add_argument('--movetime', type=int, default=2000, help='Coda fixed-time search (ms)')
    r.add_argument('--threads', type=int, default=4, help='SF threads')
    r.add_argument('--hash', type=int, default=256, help='SF hash MB')
    r.add_argument('--min-search-overrate', type=int, default=150,
                   help='cp Coda SEARCH overrates vs SF to mark confirmed')
    r.add_argument('--sf-cap', type=int, default=600, help='|SF eval| cap (skip won/lost)')
    r.add_argument('--max', type=int, default=0, help='cap positions (0=all)')
    r.set_defaults(func=cmd_reeval)

    fe = sub.add_parser('features', help='common chess features of confirmed overrates')
    fe.add_argument('scored')
    fe.set_defaults(func=cmd_features)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
