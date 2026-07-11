#!/usr/bin/env python3
"""Horizon-lag detector v2 (per Adam 2026-07-11).

Signature: a modest divergence (|gap| >= DIV_CP ~ 40) between Coda's eval
and the opponent's (same-perspective, adjacent plies) that SUSTAINS with
constant sign for >= SUSTAIN consecutive Coda moves (filters the trivial
half-ply-lead alternation), then resolves. Attribution by who converged:

  SF-RIGHT (Coda lagged): Coda's eval ends within CONV_CP of the
      opponent's value at episode start, having moved >= MOVE_CP.
  CODA-RIGHT: the mirror (opponent converged to Coda's initial value).
  UNRESOLVED: neither (game ended, or both drifted).

The count asymmetry SF-RIGHT : CODA-RIGHT is itself a horizon-gap metric
(equal engines -> symmetric). Coda-lag episodes yield suite candidates
(Coda's decision points during the lag) + episode-start positions for the
eval-convergence-at-budget measurement.
"""
import re, sys
import chess, chess.pgn
from collections import Counter

DIV_CP = 40
SUSTAIN = 2      # consecutive Coda moves with same-sign gap
CONV_CP = 30     # "agrees with" tolerance
MOVE_CP = 35     # must have actually moved to count as converging
DECIDED = 400
RES_WINDOW = 4   # Coda moves after episode to find resolution

COMMENT_EVAL = re.compile(r'([+-]?\d+\.\d+|[+-]M\d+)/\d+')

def cp_of(comment):
    m = COMMENT_EVAL.search(comment or '')
    if not m: return None
    t = m.group(1)
    if 'M' in t: return 10000 if not t.startswith('-') else -10000
    return int(float(t) * 100)

def main(paths, coda_name='Coda'):
    counts = Counter()
    candidates = []   # (fen, uci, coda_cp, opp_cp, dur, src) during lag
    ep_starts = []    # (fen, coda_cp_start, opp_cp_start, resolved_cp) for horizon metric
    n_games = 0
    for path in paths:
        src = path.split('/')[-1]
        with open(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None: break
                hdr = game.headers
                w, b = hdr.get('White'), hdr.get('Black')
                if coda_name not in (w, b): continue
                n_games += 1
                coda_white = (w == coda_name)
                board = game.board()
                plies = []
                for node in game.mainline():
                    cp = cp_of(node.comment)
                    mover_is_coda = (board.turn == chess.WHITE) == coda_white
                    plies.append({
                        'fen': board.fen(), 'uci': node.move.uci(),
                        'cp': (cp if mover_is_coda else (-cp if cp is not None else None)),
                        'coda': mover_is_coda,
                    })
                    board.push(node.move)
                # per-Coda-move samples: (idx, coda_cp, opp_cp_next)
                S = []
                for i, p in enumerate(plies):
                    if p['coda'] and p['cp'] is not None and i + 1 < len(plies) \
                       and plies[i+1]['cp'] is not None:
                        S.append((i, p['cp'], plies[i+1]['cp']))
                if len(S) < 10: continue
                j = 1
                while j < len(S):
                    gap = S[j][1] - S[j][2]
                    prev_gap = S[j-1][1] - S[j-1][2]
                    if abs(gap) < DIV_CP or abs(prev_gap) >= DIV_CP \
                       or abs(S[j][1]) > DECIDED:
                        j += 1; continue
                    sign = 1 if gap > 0 else -1
                    k = j
                    while k < len(S):
                        g = S[k][1] - S[k][2]
                        if abs(g) < DIV_CP or (1 if g > 0 else -1) != sign: break
                        k += 1
                    dur = k - j
                    if dur < SUSTAIN:
                        j = k if k > j else j + 1; continue
                    coda0, opp0 = S[j][1], S[j][2]
                    # resolution scan
                    res = None
                    for t in range(k, min(len(S), k + RES_WINDOW)):
                        codat, oppt = S[t][1], S[t][2]
                        if abs(codat - oppt) < DIV_CP:
                            if abs(codat - opp0) <= CONV_CP and abs(codat - coda0) >= MOVE_CP:
                                res = 'SF_RIGHT'
                            elif abs(oppt - coda0) <= CONV_CP and abs(oppt - opp0) >= MOVE_CP:
                                res = 'CODA_RIGHT'
                            else:
                                res = 'MET_MIDDLE'
                            break
                    res = res or 'UNRESOLVED'
                    counts[(res, dur if dur < 5 else 5)] += 1
                    counts[res] += 1
                    if res == 'SF_RIGHT':
                        for t in range(j, min(k, j + 3)):
                            i = S[t][0]
                            candidates.append((plies[i]['fen'], plies[i]['uci'],
                                               S[t][1], S[t][2], dur, src))
                        ep_starts.append((plies[S[j][0]]['fen'], coda0, opp0, src))
                    j = k
    print(f"games: {n_games}")
    for r in ('SF_RIGHT', 'CODA_RIGHT', 'MET_MIDDLE', 'UNRESOLVED'):
        print(f"  {r:11}: {counts[r]}")
    sfr, cdr = counts['SF_RIGHT'], counts['CODA_RIGHT']
    if cdr: print(f"  asymmetry SF_RIGHT/CODA_RIGHT = {sfr/cdr:.2f}")
    with open('horizon_candidates.tsv', 'w') as f:
        f.write("fen\tplayed_uci\tcoda_cp\topp_cp\tdur\tsrc\n")
        for row in candidates: f.write('\t'.join(map(str, row)) + '\n')
    with open('horizon_ep_starts.tsv', 'w') as f:
        f.write("fen\tcoda_cp\topp_cp\tsrc\n")
        for row in ep_starts: f.write('\t'.join(map(str, row)) + '\n')
    print(f"lag-window candidates: {len(candidates)}; episode starts: {len(ep_starts)}")

if __name__ == '__main__':
    main(sys.argv[1:])
