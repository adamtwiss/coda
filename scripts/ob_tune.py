#!/usr/bin/env python3
"""Submit SPSA tune to OpenBench via the web form.

Usage:
    python3 ob_tune.py <branch> --params "PARAM1, int, 100, 50, 200, 10.0, 0.002
    PARAM2, int, 50, 10, 100, 5.0, 0.002"

    python3 ob_tune.py <branch> --params-file params.txt

    python3 ob_tune.py <branch> --params-file params.txt --iterations 2500

    # Generate params from live src/search.rs defaults (preferred — avoids
    # starting the tune from stale hand-maintained defaults):
    python3 ob_tune.py <branch> \\
        --params "$(python3 scripts/gen_tune_params.py)" --iterations 2500

Environment variables:
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

def submit_tune(args):
    s = requests.Session()

    # Step 1: Get CSRF token from login page
    s.get(f'{args.server}/login/')
    csrf = s.cookies.get('csrftoken')

    # Step 2: Login
    r = s.post(f'{args.server}/login/', data={
        'username': args.username,
        'password': args.password,
        'csrfmiddlewaretoken': csrf,
    }, headers={'Referer': f'{args.server}/login/'}, allow_redirects=False)

    if r.headers.get('Location', '') != '/index/':
        print('Error: login failed')
        return False

    # Step 3: Get CSRF for tune form
    r = s.get(f'{args.server}/tune/new/')
    csrf = s.cookies.get('csrftoken')

    # Step 4: Load params
    if args.params_file:
        with open(args.params_file) as f:
            spsa_inputs = f.read().strip()
    else:
        spsa_inputs = args.params.strip()

    n_params = len([l for l in spsa_inputs.strip().split('\n') if l.strip()])
    print(f'Submitting SPSA tune: {args.branch} ({n_params} params, {args.iterations} iterations)')

    # Step 5: Submit tune
    r = s.post(f'{args.server}/tune/new/', data={
        'csrfmiddlewaretoken': csrf,
        'dev_engine':       'Coda',
        'dev_repo':         args.repo,
        'dev_branch':       args.branch,
        'dev_bench':        str(args.bench),
        'dev_network':      args.dev_network,
        'dev_options':      args.options,
        'dev_time_control': args.tc,

        'spsa_reporting_type':    'BATCHED',
        'spsa_distribution_type': 'SINGLE',
        'spsa_inputs':      spsa_inputs,
        'spsa_alpha':       '0.602',
        'spsa_gamma':       '0.101',
        'spsa_A_ratio':     '0.1',
        'spsa_iterations':  str(args.iterations),
        'spsa_pairs_per':   str(args.pairs_per),

        'book_name':    '4moves_noob.epd',
        'upload_pgns':  'FALSE',
        'priority':     str(args.priority),
        'throughput':   str(args.throughput),
        'win_adj':      'movecount=3 score=500',
        'draw_adj':     'movenumber=20 movecount=10 score=10',
        'scale_nps':    str(args.scale_nps),
        'scale_method': 'DEV',
        'syzygy_wdl':   'OPTIONAL',
        'syzygy_adj':   'OPTIONAL',
    }, headers={'Referer': f'{args.server}/tune/new/'}, allow_redirects=False)

    location = r.headers.get('Location', '')

    if '/tune/' in location and '/new/' not in location:
        print(f'Tune submitted: {args.branch}')
        print(f'URL: {args.server}{location}')
        return True

    if '/index/' in location:
        print(f'Tune submitted: {args.branch}')
        return True

    # Error — follow redirect
    r2 = s.get(f'{args.server}{location}')
    for pat in [r'error-message.*?<pre>(.*?)</pre>', r'status-message.*?<pre>(.*?)</pre>']:
        for m in re.findall(pat, r2.text, re.DOTALL):
            print(f'Error: {m.strip()}')

    # Also try finding any error text
    errors = re.findall(r'class="[^"]*error[^"]*"[^>]*>(.*?)<', r2.text)
    for e in errors[:3]:
        if e.strip():
            print(f'Error: {e.strip()}')

    return False

def main():
    p = argparse.ArgumentParser(description='Submit SPSA tune to OpenBench')
    p.add_argument('branch', help='Branch to tune')
    p.add_argument('bench', nargs='?', type=int, default=0, help='Bench value (0=auto-detect)')
    p.add_argument('--params', default='', help='SPSA params inline (newline-separated)')
    p.add_argument('--params-file', default='', help='File with SPSA params (one per line)')
    p.add_argument('--iterations', type=int, default=2500, help='SPSA iterations (default: 2500)')
    p.add_argument('--pairs-per', type=int, default=8, help='Game pairs per iteration (default: 8)')
    p.add_argument('--tc', default='10.0+0.1', help='Time control (default: 10.0+0.1)')
    p.add_argument('--options', default='Threads=1 Hash=64', help='UCI options')
    p.add_argument('--dev-network', default='', help='Dev network SHA256 hash (8 chars, from ob_upload_net.py)')
    p.add_argument('--priority', type=int, default=0, help='Priority (default: 0)')
    p.add_argument('--throughput', type=int, default=100, help='Throughput (default: 100)')
    p.add_argument('--scale-nps', type=int, default=None, help='Reference NPS for TC scaling. Auto-detects from branch name: 250000 for v9 branches (feature/threat-inputs, experiment/*, tune/v9-*, fix/threats-*), 500000 for main/v5. Override to force.')
    p.add_argument('--repo', default='https://github.com/adamtwiss/coda', help='GitHub repo URL')
    p.add_argument('--server', default=SERVER, help=f'Server (default: {SERVER})')
    p.add_argument('--username', default=USERNAME, help='Username')
    p.add_argument('--password', default=PASSWORD, help='Password')
    args = p.parse_args()

    # Auto-detect scale_nps from branch name (see ob_submit.py rationale).
    if args.scale_nps is None:
        v9_patterns = ('feature/threat-inputs', 'experiment/', 'tune/v9-', 'fix/threats-')
        is_v9 = any(args.branch.startswith(p) for p in v9_patterns)
        args.scale_nps = 250000 if is_v9 else 500000
        print(f'[auto] scale_nps={args.scale_nps} ({"v9 branch" if is_v9 else "v5/main"})')

    if not args.password:
        print('Error: password required. Set OPENBENCH_PASSWORD or use --password')
        return

    if not args.params and not args.params_file:
        print('Error: --params or --params-file required')
        return

    submit_tune(args)

if __name__ == '__main__':
    main()
