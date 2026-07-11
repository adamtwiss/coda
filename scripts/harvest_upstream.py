#!/usr/bin/env python3
"""Pre-episode (upstream) decision-point harvester + control sampler.

For each SF_RIGHT lag episode (same detection as harvest_horizon), emit
Coda's decision points at offsets -1..-3 BEFORE the episode start (the
causal window: where deeper horizon would have steered elsewhere), plus a
CONTROL sample of Coda decisions from episode-free stretches of the same
games (>= 6 Coda moves away from any episode). Both classes get deep
pricing later; the comparison upstream-vs-control cp-loss distribution
tests the steering-leak hypothesis (Adam, 2026-07-11).
"""
import re, sys, random
import chess, chess.pgn

DIV_CP, SUSTAIN, DECIDED = 40, 2, 400
PRE_OFFSETS = (1, 2, 3)
CONTROL_PER_GAME = 1
COMMENT_EVAL = re.compile(r'([+-]?\d+\.\d+|[+-]M\d+)/\d+')
rng = random.Random(42)

def cp_of(c):
    m = COMMENT_EVAL.search(c or '')
    if not m: return None
    t = m.group(1)
    if 'M' in t: return 10000 if not t.startswith('-') else -10000
    return int(float(t) * 100)

def main(paths, coda_name='Coda'):
    upstream, control = [], []
    for path in paths:
        src = path.split('/')[-1]
        with open(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None: break
                w, b = game.headers.get('White'), game.headers.get('Black')
                if coda_name not in (w, b): continue
                coda_white = (w == coda_name)
                board = game.board()
                plies = []
                for node in game.mainline():
                    cp = cp_of(node.comment)
                    mover_is_coda = (board.turn == chess.WHITE) == coda_white
                    plies.append({'fen': board.fen(), 'uci': node.move.uci(),
                                  'cp': (cp if mover_is_coda else (-cp if cp is not None else None)),
                                  'coda': mover_is_coda})
                    board.push(node.move)
                S = [(i, p['cp'], plies[i+1]['cp']) for i, p in enumerate(plies)
                     if p['coda'] and p['cp'] is not None and i + 1 < len(plies)
                     and plies[i+1]['cp'] is not None]
                if len(S) < 10: continue
                # find episode-start indices (into S)
                ep_starts = []
                j = 1
                while j < len(S):
                    gap = S[j][1] - S[j][2]; prev = S[j-1][1] - S[j-1][2]
                    if abs(gap) < DIV_CP or abs(prev) >= DIV_CP or abs(S[j][1]) > DECIDED:
                        j += 1; continue
                    sign = 1 if gap > 0 else -1
                    k = j
                    while k < len(S):
                        g = S[k][1] - S[k][2]
                        if abs(g) < DIV_CP or (1 if g > 0 else -1) != sign: break
                        k += 1
                    if k - j >= SUSTAIN: ep_starts.append(j)
                    j = k
                if not ep_starts: continue
                in_ep = set()
                for js in ep_starts:
                    for t in range(max(0, js - 3), min(len(S), js + 8)):
                        in_ep.add(t)
                for js in ep_starts[:2]:  # cap per game
                    for off in PRE_OFFSETS:
                        t = js - off
                        if t < 1: break
                        i = S[t][0]
                        if abs(S[t][1]) > DECIDED: continue
                        upstream.append((plies[i]['fen'], plies[i]['uci'], S[t][1],
                                         off, src))
                free = [t for t in range(2, len(S)) if t not in in_ep
                        and abs(S[t][1]) <= DECIDED]
                for t in rng.sample(free, min(CONTROL_PER_GAME, len(free))):
                    i = S[t][0]
                    control.append((plies[i]['fen'], plies[i]['uci'], S[t][1], 0, src))
    with open('upstream_candidates.tsv', 'w') as f:
        f.write("fen\tplayed_uci\tcoda_cp\toffset\tsrc\n")
        for r in upstream: f.write('\t'.join(map(str, r)) + '\n')
    with open('control_candidates.tsv', 'w') as f:
        f.write("fen\tplayed_uci\tcoda_cp\toffset\tsrc\n")
        for r in control: f.write('\t'.join(map(str, r)) + '\n')
    print(f"upstream: {len(upstream)}  control: {len(control)}")

if __name__ == '__main__':
    main(sys.argv[1:])
