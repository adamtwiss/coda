#!/usr/bin/env python3
"""Empirical cross-implementation PSQ (HalfKA) differ.

Companion to threat_cross_differ.py, which verified the THREAT half of the
v9 input encoding (0/12000). This verifies the PSQ half: Coda INFERENCE
(`coda dump-threats` PSQ lines, halfka_index_with + reckless kb10 tables)
vs the REAL Bullet TRAINING encoder (ChessBucketsMirrored inside
ChessBucketsWithThreats, run via the `extra_stamp_probe::dump_file` test).

The PSQ half has its own machinery the threat gate says nothing about:
king-bucket table expansion, e-file horizontal mirror, vertical flip +
color-swap for the non-STM perspective, piece-index ordering. None of it
was empirically cross-checked before this script existed.

Pairing is exact per perspective (not a union): Coda's POV=<stm> block must
equal Bullet's PSQ_STM set, and POV=<non-stm> must equal PSQ_NTM. This
catches perspective swaps a union comparison would hide.

Usage: python3 scripts/psq_cross_differ.py [N]   (default N=3000)
"""
import chess, random, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

random.seed(20260703)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
FENFILE = '/tmp/psq_differ_fens.txt'
CODA = './coda'
BULLET = '../bullet'


def mkfen(b):
    # placement + stm only; '-' castling/ep (irrelevant to PSQ features, and
    # Bullet FromStr ignores them). Guarantees both parsers accept every
    # position, including chess960-derived piece placements.
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
playout(lambda: chess.Board.from_chess960_pos(random.randint(0, 959)), 4, 60, int(N * 0.20))  # 960 exotic placements (kings on odd files/either wing)
playout(lambda: chess.Board(), 70, 160, int(N * 0.15))                                # promotion-heavy endgames
fens = list(dict.fromkeys(fens))
open(FENFILE, 'w').write('\n'.join(fens) + '\n')
print(f"corpus: {len(fens)} unique FENs", file=sys.stderr)

# --- Bullet training encoder dump (batch, one process) ---
env = dict(os.environ, PROBE_FEN_FILE=FENFILE)
r = subprocess.run(['cargo', 'test', '--release', '-p', 'bullet_lib',
                    'extra_stamp_probe::dump_file', '--', '--ignored', '--nocapture'],
                   cwd=BULLET, env=env, capture_output=True, text=True)
bullet = {}   # fen -> {'STM': set, 'NTM': set}
cur = None
for ln in r.stdout.splitlines():
    if ln.startswith('FEN '):
        cur = ln[4:]
        bullet[cur] = {}
    elif ln.startswith('PSQ_STM ') and cur is not None:
        bullet[cur]['STM'] = set(int(x) for x in ln[8:].split())
    elif ln.startswith('PSQ_NTM ') and cur is not None:
        bullet[cur]['NTM'] = set(int(x) for x in ln[8:].split())
    elif ln.startswith('PARSE_ERR'):
        bullet.pop(cur, None)
        cur = None
print(f"bullet dumped {len(bullet)} positions", file=sys.stderr)

# --- Coda inference dump (parallel spawns) ---
def coda_sets(fen):
    out = subprocess.run([CODA, 'dump-threats', fen], capture_output=True, text=True).stdout
    povs = {}   # 'w'/'b' -> set
    pov = None
    for ln in out.splitlines():
        if ln.startswith('# POV='):
            pov = ln[6]
        elif ln.startswith('PSQ ') and pov is not None:
            povs[pov] = set(int(x) for x in ln[4:].split())
    return fen, povs


mism = 0
checked = 0
examples = []
with ThreadPoolExecutor(max_workers=12) as ex:
    for fen, povs in ex.map(coda_sets, fens):
        bsets = bullet.get(fen)
        if not bsets or 'STM' not in bsets or 'NTM' not in bsets or len(povs) != 2:
            continue
        stm = fen.split()[1]           # 'w' or 'b'
        ntm = 'b' if stm == 'w' else 'w'
        checked += 1
        bad = []
        if povs[stm] != bsets['STM']:
            bad.append(('STM', sorted(povs[stm] - bsets['STM'])[:10], sorted(bsets['STM'] - povs[stm])[:10]))
        if povs[ntm] != bsets['NTM']:
            bad.append(('NTM', sorted(povs[ntm] - bsets['NTM'])[:10], sorted(bsets['NTM'] - povs[ntm])[:10]))
        if bad:
            mism += 1
            if len(examples) < 8:
                examples.append((fen, bad))

print(f"\n=== checked {checked} positions, {mism} MISMATCHES ===")
for fen, bad in examples:
    print(f"MISMATCH {fen}")
    for persp, co, bo in bad:
        print(f"  [{persp}] coda-only: {co}\n  [{persp}] bull-only: {bo}")
if mism == 0 and checked:
    print("OK — Coda inference PSQ == Bullet training encoder on every position.")
