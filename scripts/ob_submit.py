#!/usr/bin/env python3
"""Submit SPRT tests to OpenBench.

Usage:
    python3 ob_submit.py <dev_branch>                          # OB auto-detects bench from commit
    python3 ob_submit.py <dev_branch> --bounds '[-3, 3]'       # Custom SPRT bounds
    python3 ob_submit.py <dev_branch> --tc '40.0+0.4'          # Custom time control
    python3 ob_submit.py <dev_branch> --base-branch b98c0a1    # Explicit base commit
    python3 ob_submit.py <dev_branch> 1234567                  # Override dev bench (avoid if possible)
    python3 ob_submit.py <dev_branch> --dev-network ABCD1234   # Custom NNUE net for dev
    python3 ob_submit.py <dev_branch> --base-network ABCD1234  # Custom NNUE net for base
    python3 ob_submit.py <dev_branch> \
        --dev-options 'Threads=1 Hash=64 HiddenActivation=crelu' # Custom UCI for dev only

Network hashes are the first 8 chars of SHA256 (uppercase). Use ob_upload_net.py to
upload nets and get their hashes, or check https://ob.atwiss.com/networks/.

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

def parse_bench_from_git(branch):
    """Parse Bench: NNNN from the latest commit message on a local or remote branch."""
    import subprocess, re
    for ref in [branch, f'origin/{branch}']:
        try:
            msg = subprocess.check_output(
                ['git', 'log', ref, '-1', '--format=%B'],
                stderr=subprocess.DEVNULL, text=True
            )
            m = re.search(r'Bench:\s*(\d+)', msg)
            if m:
                return int(m.group(1))
        except subprocess.CalledProcessError:
            continue
    return None

def submit_test(args):
    # Auto-detect bench from local git if not provided
    if not args.dev_bench:
        args.dev_bench = parse_bench_from_git(args.dev_branch)
        if not args.dev_bench:
            print(f'Error: could not parse bench from {args.dev_branch} commit message')
            return False
    if not args.base_bench:
        args.base_bench = parse_bench_from_git(args.base_branch)
        if not args.base_bench:
            print(f'Error: could not parse bench from {args.base_branch} commit message')
            return False

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
        'dev_bench':        str(args.dev_bench),
        'dev_options':      args.dev_options or args.options,
        'dev_time_control': args.tc,
        'dev_network':      args.dev_network,

        'base_repo':         args.repo,
        'base_engine':       'Coda',
        'base_branch':       args.base_branch,
        'base_bench':        str(args.base_bench),
        'base_options':      args.base_options or args.options,
        'base_time_control': args.tc,
        'base_network':      args.base_network,

        'test_mode':       'SPRT',
        'test_bounds':     args.bounds,
        'test_confidence': '[0.05, 0.05]',
        'book_name':       '4moves_noob.epd',
        'priority':        str(args.priority),
        'throughput':       str(args.throughput),
        'workload_size':   '32',
        'scale_method':    'BASE',
        'scale_nps':       str(args.scale_nps),
        'win_adj':         'movecount=3 score=500',
        'draw_adj':        'movenumber=20 movecount=10 score=10',
        'upload_pgns':     'FALSE',
        'syzygy_wdl':      'OPTIONAL',
        'syzygy_adj':      'OPTIONAL',
    }

    r = s.post(f'{args.server}/scripts/', data=data, allow_redirects=False)

    location = r.headers.get('Location', '')

    if '/index/' in location:
        print(f'Test submitted: {args.dev_branch} (bench {args.dev_bench}) vs {args.base_branch} (bench {args.base_bench})')
        print(f'Bounds: {args.bounds}, TC: {args.tc}')
        if args.dev_network:
            print(f'Dev network: {args.dev_network}')
        if args.base_network:
            print(f'Base network: {args.base_network}')
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
    p.add_argument('--options', default='Threads=1 Hash=64', help='UCI options (applied to both sides unless --dev-options/--base-options given)')
    p.add_argument('--dev-options', default='', help='UCI options for dev side only (overrides --options for dev). Example: "Threads=1 Hash=64 HiddenActivation=crelu"')
    p.add_argument('--base-options', default='', help='UCI options for base side only (overrides --options for base)')
    p.add_argument('--dev-network', default='', help='Dev network SHA256 hash (8 chars, from ob_upload_net.py)')
    p.add_argument('--base-network', default='', help='Base network SHA256 hash (8 chars, from ob_upload_net.py)')
    p.add_argument('--priority', type=int, default=0, help='Priority (default: 0)')
    p.add_argument('--throughput', type=int, default=100, help='Throughput (default: 100)')
    p.add_argument('--scale-nps', type=int, default=None, help='Reference NPS for TC scaling. Defaults to 250000 (v9, current main). Use --scale-nps 500000 for v5-only branches.')
    p.add_argument('--repo', default='https://github.com/adamtwiss/coda', help='GitHub repo URL')
    p.add_argument('--server', default=SERVER, help=f'Server (default: {SERVER})')
    p.add_argument('--username', default=USERNAME, help='Username')
    p.add_argument('--password', default=PASSWORD, help='Password')
    args = p.parse_args()

    # scale_nps default: 250000 (v9 is main since 2026-04-24 merge).
    # V9 runs at ~240-280K NPS per core (threat features add ~40% work).
    # V5 runs at ~500K+. Wrong scale_nps means games get wrong time budgets.
    # Explicit v5 branch work must pass --scale-nps 500000 (e.g. legacy
    # bullet_convert experiments, v5-only net comparisons).
    if args.scale_nps is None:
        args.scale_nps = 250000
    # Keep is_v9 True for the net-bench warning below — main is v9 now.
    # Net override warning (stale pre-merge check removed 2026-04-24).
    # OB builds with whatever net.txt points to, same as local make.
    # Pass --dev-network / --base-network only for explicit net-vs-net
    # comparisons or when testing a non-prod net.

    if not args.password:
        print('Error: password required. Set OPENBENCH_PASSWORD or use --password')
        return

    submit_test(args)

if __name__ == '__main__':
    main()
