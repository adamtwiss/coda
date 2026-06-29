#!/usr/bin/env python3
"""Submit a Coda-vs-Stockfish datagen workload to OpenBench via the API.

Replicates the 2062 datagen config: Coda (dev) generates the genfens openings
and plays; Stockfish (base, with its 1C000000 net) plays + provides labels; both
at fixed nodes N=15000; VERBOSE pgnout (eval+depth+nodes per move). Scaling is
DEV/250000 so the heterogeneous fleet normalises node budgets.

Usage:
    OPENBENCH_PASSWORD=... python3 scripts/ob_datagen.py                 # 3M games, defaults
    OPENBENCH_PASSWORD=... python3 scripts/ob_datagen.py --max-games 1000  # small test
    OPENBENCH_PASSWORD=... python3 scripts/ob_datagen.py --dev-bench 2944948 --priority 0

Env: OPENBENCH_SERVER (default https://ob.atwiss.com), OPENBENCH_USERNAME (claude),
OPENBENCH_PASSWORD (required).
"""
import argparse, os, re, subprocess, sys, requests

SERVER   = os.environ.get('OPENBENCH_SERVER',   'https://ob.atwiss.com')
USERNAME = os.environ.get('OPENBENCH_USERNAME', 'claude')
PASSWORD = os.environ.get('OPENBENCH_PASSWORD', '')

def git_bench(branch):
    for ref in [branch, f'origin/{branch}']:
        try:
            msg = subprocess.check_output(['git','log',ref,'-1','--format=%B'],
                                          stderr=subprocess.DEVNULL, text=True)
            m = re.search(r'Bench:\s*(\d+)', msg)
            if m: return int(m.group(1))
        except subprocess.CalledProcessError:
            continue
    return None

def main():
    p = argparse.ArgumentParser(description='Submit Coda-vs-SF datagen to OpenBench')
    p.add_argument('--max-games', type=int, default=3_000_000)
    p.add_argument('--dev-branch', default='main')
    p.add_argument('--dev-bench', type=int, default=None, help='Coda bench (default: parse Bench: from commit)')
    p.add_argument('--genfens', default='', help='extra genfens args (count/seed are added by OB)')
    p.add_argument('--priority', type=int, default=0)
    p.add_argument('--throughput', type=int, default=100)
    p.add_argument('--workload-size', type=int, default=32)
    p.add_argument('--scale-nps', type=int, default=250000)
    p.add_argument('--pgnout', default='VERBOSE', choices=['FALSE','COMPACT','VERBOSE'])
    p.add_argument('--book', default='UHO_Lichess_4852_v1.epd',
                   help='OB opening book for variety (default UHO; or noob_4moves.epd). '
                        'Pass NONE to rely on genfens only — AVOID: genfens openings recycle '
                        'and, with fixed-nodes deterministic play, the SAME opening+colour yields '
                        'IDENTICAL games, so ~50%% of the corpus ends up exact duplicates '
                        '(measured on run 2061). A big book gives far more distinct, non-recycled '
                        'openings. See the Datagen section of the ob skill.')
    p.add_argument('--server', default=SERVER); p.add_argument('--username', default=USERNAME)
    p.add_argument('--password', default=PASSWORD)
    a = p.parse_args()

    if not a.password:
        sys.exit('Error: OPENBENCH_PASSWORD required')
    if a.dev_bench is None:
        a.dev_bench = git_bench(a.dev_branch)
        if a.dev_bench is None:
            sys.exit(f'Error: could not parse Bench: from {a.dev_branch}; pass --dev-bench')

    s = requests.Session()
    s.get(f'{a.server}/login/'); csrf = s.cookies.get('csrftoken')
    r = s.post(f'{a.server}/login/', data={'username':a.username,'password':a.password,
               'csrfmiddlewaretoken':csrf}, headers={'Referer':f'{a.server}/login/'}, allow_redirects=False)
    if r.headers.get('Location','') != '/index/':
        sys.exit('Error: login failed')
    # refresh CSRF from the datagen form
    s.get(f'{a.server}/datagen/new/'); csrf = s.cookies.get('csrftoken')

    data = {
        'csrfmiddlewaretoken': csrf,
        # Dev = Coda (does the genfens + plays); embedded net (net.txt)
        'dev_repo': 'https://github.com/adamtwiss/coda', 'dev_engine': 'Coda',
        'dev_branch': a.dev_branch, 'dev_bench': str(a.dev_bench),
        'dev_options': 'Threads=1 Hash=64', 'dev_time_control': 'N=15000', 'dev_network': '',
        # Base = Stockfish ob_17.1 with its 1C000000 net (labels)
        'base_repo': 'https://github.com/AndyGrant/Stockfish', 'base_engine': 'Stockfish',
        'base_branch': 'ob_17.1', 'base_bench': '4644582',
        'base_options': 'Threads=1 Hash=64', 'base_time_control': 'N=15000', 'base_network': '1C000000',
        # Datagen
        'datagen_max_games': str(a.max_games), 'datagen_custom_genfens': a.genfens,
        'datagen_play_reverses': 'YES', 'book_name': a.book, 'upload_pgns': a.pgnout,
        # General + scaling + adjudication (2062 values)
        'priority': str(a.priority), 'throughput': str(a.throughput),
        'workload_size': str(a.workload_size),
        'scale_method': 'DEV', 'scale_nps': str(a.scale_nps),
        'syzygy_wdl': 'DISABLED', 'syzygy_adj': 'OPTIONAL',
        'win_adj': 'movecount=3 score=500', 'draw_adj': 'movenumber=20 movecount=10 score=10',
    }
    r = s.post(f'{a.server}/datagen/new/', data=data, headers={'Referer':f'{a.server}/datagen/new/'},
               allow_redirects=False)
    loc = r.headers.get('Location','')
    if '/index/' in loc:  # success redirects to the index (like tests); find the new id
        idx = s.get(f'{a.server}/index/')
        ids = sorted({int(i) for i in re.findall(r'/datagen/(\d+)/', idx.text)}, reverse=True)
        wid = ids[0] if ids else '?'
        print(f'Datagen submitted: workload {wid} — Coda({a.dev_branch} bench {a.dev_bench}) '
              f'vs Stockfish ob_17.1, {a.max_games} games @ N=15000, pgnout={a.pgnout}')
        print(f'  {a.server}/datagen/{wid}/')
        return
    # error: follow redirect back to the form to surface the message
    r2 = s.get(f'{a.server}{loc}' if loc else f'{a.server}/datagen/new/')
    for pat in [r'error-message.*?<pre>(.*?)</pre>', r'status-message.*?<pre>(.*?)</pre>']:
        for msg in re.findall(pat, r2.text, re.DOTALL):
            print(f'Error: {msg.strip()}')
    print(f'(submission did not redirect to /index/: {loc!r})')

if __name__ == '__main__':
    main()
