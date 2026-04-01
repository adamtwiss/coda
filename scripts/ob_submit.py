#!/usr/bin/env python3
"""Submit SPRT tests to OpenBench via the scripts API.

Usage:
    python3 ob_submit.py <dev_branch> <dev_bench> [options]
    python3 ob_submit.py with-razoring 968352
    python3 ob_submit.py nmp-v1 1234567 --bounds '[0.00, 10.00]'

Environment variables (or use --flags):
    OPENBENCH_SERVER   (default: https://ob.atwiss.com)
    OPENBENCH_USERNAME (default: claude)
    OPENBENCH_PASSWORD (required)
"""

import argparse
import os
import re
import requests

SERVER   = os.environ.get('OPENBENCH_SERVER',   'https://ob.atwiss.com')
USERNAME = os.environ.get('OPENBENCH_USERNAME', 'claude')
PASSWORD = os.environ.get('OPENBENCH_PASSWORD', '')

def submit_test(args):
    s = requests.Session()

    # Step 1: Get CSRF token from login page
    s.get(f'{args.server}/login/')
    csrf = s.cookies.get('csrftoken')

    # Step 2: Login to get session
    r = s.post(f'{args.server}/login/', data={
        'username': args.username,
        'password': args.password,
        'csrfmiddlewaretoken': csrf,
    }, headers={'Referer': f'{args.server}/login/'}, allow_redirects=False)

    if r.headers.get('Location', '') != '/index/':
        print('Error: login failed')
        return False

    # Step 3: Submit test via scripts endpoint
    r = s.post(f'{args.server}/scripts/', data={
        'username': args.username,
        'password': args.password,
        'action': 'CREATE_TEST',

        'dev_repo':         args.repo,
        'dev_engine':       'Coda',
        'dev_branch':       args.dev_branch,
        'dev_bench':        str(args.dev_bench),
        'dev_options':      args.options,
        'dev_time_control': args.tc,
        'dev_network':      '',

        'base_repo':         args.repo,
        'base_engine':       'Coda',
        'base_branch':       args.base_branch,
        'base_bench':        str(args.base_bench),
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
    }, allow_redirects=False)

    location = r.headers.get('Location', '')

    # Success redirects to /index/
    if '/index/' in location:
        print(f'Test submitted: {args.dev_branch} (bench {args.dev_bench}) vs {args.base_branch} (bench {args.base_bench})')
        print(f'Bounds: {args.bounds}, TC: {args.tc}')
        return True

    # Error — follow redirect to get message
    r2 = s.get(f'{args.server}{location}')
    for pat in [r'error-message.*?<pre>(.*?)</pre>', r'status-message.*?<pre>(.*?)</pre>']:
        for m in re.findall(pat, r2.text, re.DOTALL):
            print(f'Error: {m.strip()}')
    return False

def main():
    p = argparse.ArgumentParser(description='Submit SPRT test to OpenBench')
    p.add_argument('dev_branch', help='Dev branch name')
    p.add_argument('dev_bench', type=int, help='Dev bench node count')
    p.add_argument('--base-branch', default='main', help='Base branch (default: main)')
    p.add_argument('--base-bench', type=int, default=1375565, help='Base bench (default: 1375565)')
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
