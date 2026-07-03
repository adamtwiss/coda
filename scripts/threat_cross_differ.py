#!/usr/bin/env python3
"""Empirical cross-implementation threat differ.

Verifies Coda INFERENCE (`coda dump-threats`, enumerate_threats) produces the
same threat-feature set as the REAL Bullet TRAINING encoder (map_features, run
via the `extra_stamp_probe::dump_file` test with the fixed extra[0]=stm stamp),
over a diverse corpus. This is the belt-and-suspenders the self-referential
`fuzz-threats` can't provide (its references both call Coda's own tables).

Per position, compares the UNION of both-perspective threat indices, which is
robust to the stm/ntm <-> POV=w/b ordering flip. A mismatch = a train/inference
feature-encoding divergence (like the extra[0] bug: coda 11095 vs bullet 11200).

Usage: python3 scripts/threat_cross_differ.py [N]   (default N=3000)
"""
import chess, random, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

random.seed(20260701)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
FENFILE = '/tmp/cross_differ_fens.txt'
CODA = './coda'
BULLET = '../bullet'


def mkfen(b):
    # placement + stm only; '-' castling/ep (irrelevant to threats, and Bullet
    # FromStr ignores them). Guarantees both parsers accept every position,
    # including chess960-derived piece placements.
    return f"{b.board_fen()} {'w' if b.turn else 'b'} - - 0 1"


fens = []
def playout(factory, kmin, kmax, count):
    for _ in range(count):
        b = factory()
        for _ in range(random.randint(kmin, kmax)):
            mv = list(b.legal_moves)
            if not mv:
                break
            b.push(random.choice(mv))
        fens.append(mkfen(b))


playout(lambda: chess.Board(), 4, 80, int(N * 0.65))                                  # standard, varied phase
playout(lambda: chess.Board.from_chess960_pos(random.randint(0, 959)), 4, 60, int(N * 0.20))  # 960 exotic placements
playout(lambda: chess.Board(), 70, 160, int(N * 0.15))                                # promotion-heavy endgames
fens = list(dict.fromkeys(fens))
open(FENFILE, 'w').write('\n'.join(fens) + '\n')
print(f"corpus: {len(fens)} unique FENs", file=sys.stderr)

# --- Bullet training encoder dump (batch, one process) ---
env = dict(os.environ, PROBE_FEN_FILE=FENFILE)
r = subprocess.run(['cargo', 'test', '--release', '-p', 'bullet_lib',
                    'extra_stamp_probe::dump_file', '--', '--ignored', '--nocapture'],
                   cwd=BULLET, env=env, capture_output=True, text=True)
bullet = {}
cur = None
for ln in r.stdout.splitlines():
    if ln.startswith('FEN '):
        cur = ln[4:]
    elif ln.startswith('IDX ') and cur is not None:
        bullet[cur] = set(int(x) for x in ln[4:].split())
        cur = None
    elif ln.startswith('PARSE_ERR'):
        cur = None
print(f"bullet dumped {len(bullet)} positions", file=sys.stderr)

# --- Coda inference dump (parallel spawns) ---
def coda_set(fen):
    out = subprocess.run([CODA, 'dump-threats', fen], capture_output=True, text=True).stdout
    s = set()
    for ln in out.splitlines():
        if ln.startswith('IDX '):
            s |= set(int(x) for x in ln[4:].split())
    return fen, s


mism = 0
checked = 0
examples = []
with ThreadPoolExecutor(max_workers=12) as ex:
    for fen, cset in ex.map(coda_set, fens):
        bset = bullet.get(fen)
        if bset is None:
            continue
        checked += 1
        if cset != bset:
            mism += 1
            if len(examples) < 8:
                examples.append((fen, sorted(cset - bset)[:15], sorted(bset - cset)[:15]))

print(f"\n=== checked {checked} positions, {mism} MISMATCHES ===")
for fen, co, bo in examples:
    print(f"MISMATCH {fen}\n  coda-only: {co}\n  bull-only: {bo}")
if mism == 0 and checked:
    print("OK — Coda inference == Bullet training encoder on every position.")
