#!/usr/bin/env python3
"""Submit SPRT tests to OpenBench.

Usage:
    python3 ob_submit.py <dev_branch>                          # OB auto-detects bench from commit
    python3 ob_submit.py <dev_branch> --bounds '[-3, 3]'       # Custom SPRT bounds
    python3 ob_submit.py <dev_branch> --tc '40.0+0.4'          # Custom time control
    python3 ob_submit.py <dev_branch> --base-branch b98c0a1    # Explicit base commit
    python3 ob_submit.py <dev_branch> 1234567                  # Override dev bench (avoid if possible)

Best practice: let OB auto-detect bench from commit messages (Bench: NNNNNN).
Only pass explicit bench if OB fails to parse.

Environment variables:
    OPENBENCH_SERVER   (default: https://ob.atwiss.com)
    OPENBENCH_USERNAME (default: claude)
    OPENBENCH_PASSWORD (required)
"""

import argparse
import os
import requests

SERVER   = os.environ.get('OPENBENCH_SERVER',   'https://ob.atwiss.com')
USERNAME = os.environ.get('OPENBENCH_USERNAME', 'claude')
PASSWORD = os.environ.get('OPENBENCH_PASSWORD', '')

def submit_test(args):
    s = requests.Session()

    # Login
    s.get(f'{args.server}/login/')
    csrf = s.cookies.get('csrftoken')
    r = s.post(f'{args.server}/login/', data={
        'username': args.username,
        'password': args.password,
        'csrfmiddlewaretoken': csrf,
    }, headers={'Referer': f'{args.server}/login/'}, allow_redirects=False)

    if r.headers.get('Location', '') != '/index/':
        print('Error: login failed')
        return False

    # Build form data — let OB auto-detect bench unless explicitly overridden
    data = {
        'username': args.username,
        'password': args.password,
        'action': 'CREATE_TEST',

        'dev_repo':         args.repo,
        'dev_engine':       'Coda',
        'dev_branch':       args.dev_branch,
        'dev_bench':        str(args.dev_bench) if args.dev_bench else '',
        'dev_options':      args.options,
        'dev_time_control': args.tc,
        'dev_network':      '',

        'base_repo':         args.repo,
        'base_engine':       'Coda',
        'base_branch':       args.base_branch,
        'base_bench':        str(args.base_bench) if args.base_bench else '',
        'base_options':      args.options,
        'base_time_control': args.tc,
        'base_network':      '',

        'test_mode':       'SPRT',
        'test_bounds':     args.bounds,
        'test_confidence': '[0.05, 0.05]',
        'book_name':       '4moves_noob.epd',
        'priority':        str(args.priority),
        'throughput':       str(args.throughput),
        'workload_size':   '32',
        'scale_method':    'BASE',
        'scale_nps':       '1100000',
        'win_adj':         'movecount=3 score=500',
        'draw_adj':        'movenumber=20 movecount=10 score=10',
        'upload_pgns':     'FALSE',
        'syzygy_wdl':      'OPTIONAL',
        'syzygy_adj':      'OPTIONAL',
    }

    r = s.post(f'{args.server}/scripts/', data=data, allow_redirects=False)

    location = r.headers.get('Location', '')

    if '/index/' in location:
        bench_info = f' (bench {args.dev_bench})' if args.dev_bench else ''
        base_info = f' (bench {args.base_bench})' if args.base_bench else ''
        print(f'Test submitted: {args.dev_branch}{bench_info} vs {args.base_branch}{base_info}')
        print(f'Bounds: {args.bounds}, TC: {args.tc}')
        return True

    # Error — follow redirect to get message
    import re
    r2 = s.get(f'{args.server}{location}')
    for pat in [r'error-message.*?<pre>(.*?)</pre>', r'status-message.*?<pre>(.*?)</pre>']:
        for m in re.findall(pat, r2.text, re.DOTALL):
            print(f'Error: {m.strip()}')
    return False

def main():
    p = argparse.ArgumentParser(description='Submit SPRT test to OpenBench')
    p.add_argument('dev_branch', help='Dev branch name')
    p.add_argument('dev_bench', nargs='?', type=int, default=None, help='Dev bench (omit to let OB auto-detect)')
    p.add_argument('--base-branch', default='main', help='Base branch (default: main)')
    p.add_argument('--base-bench', type=int, default=None, help='Base bench (omit to let OB auto-detect)')
    p.add_argument('--bounds', default='[0.00, 5.00]', help='SPRT bounds (default: [0.00, 5.00])')
    p.add_argument('--tc', default='10.0+0.1', help='Time control (default: 10.0+0.1)')
    p.add_argument('--options', default='Threads=1 Hash=64', help='UCI options')
    p.add_argument('--priority', type=int, default=0, help='Priority (default: 0)')
    p.add_argument('--throughput', type=int, default=100, help='Throughput (default: 100)')
    p.add_argument('--repo', default='https://github.com/adamtwiss/coda', help='GitHub repo URL')
    p.add_argument('--server', default=SERVER, help=f'Server (default: {SERVER})')
    p.add_argument('--username', default=USERNAME, help='Username')
    p.add_argument('--password', default=PASSWORD, help='Password')
    args = p.parse_args()

    if not args.password:
        print('Error: password required. Set OPENBENCH_PASSWORD or use --password')
        return

    submit_test(args)

if __name__ == '__main__':
    main()
