#!/usr/bin/env python3
"""Read SPSA tune status and parameter values from OpenBench.

Usage:
    python3 ob_tune_status.py              # Show all active tunes
    python3 ob_tune_status.py 175          # Show specific tune by ID
    python3 ob_tune_status.py 175 --csv    # Output as CSV (for diffing)

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

def login(server, username, password):
    s = requests.Session()
    s.get(f'{server}/login/')
    csrf = s.cookies.get('csrftoken')
    s.post(f'{server}/login/', data={
        'username': username, 'password': password, 'csrfmiddlewaretoken': csrf,
    }, headers={'Referer': f'{server}/login/'}, allow_redirects=False)
    return s

def get_tune_ids(s, server):
    """Get all tune IDs from the index page."""
    r = s.get(f'{server}/index/')
    tunes = re.findall(r'/tune/(\d+)/', r.text)
    return sorted(set(tunes), key=int, reverse=True)

def get_tune_info(s, server, tid):
    """Get branch name and iteration progress for a tune."""
    r = s.get(f'{server}/tune/{tid}/')
    branch = re.search(r'Branch.*?<td[^>]*>(.*?)</td>', r.text, re.DOTALL)
    branch_name = branch.group(1).strip() if branch else '?'
    iters = re.search(r'Iterations.*?(\d+)\s*/\s*(\d+)', r.text, re.DOTALL)
    progress = f'{iters.group(1)}/{iters.group(2)}' if iters else '?'
    return branch_name, progress

def get_tune_digest(s, server, tid):
    """Get SPSA parameter values via the digest API."""
    r = s.get(f'{server}/api/spsa/{tid}/digest/')
    if r.status_code != 200:
        return None
    lines = r.text.strip().split('\n')
    if len(lines) < 2:
        return None
    # First line is header: Name,Curr,Start,Min,Max,C,C_end,R,R_end
    params = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 5:
            params.append({
                'name': parts[0],
                'curr': float(parts[1]),
                'start': float(parts[2]),
                'min': float(parts[3]),
                'max': float(parts[4]),
            })
    return params

def get_tune_outputs(s, server, tid):
    """Get SPSA output format (for re-submission)."""
    r = s.get(f'{server}/api/spsa/{tid}/outputs/')
    if r.status_code == 200:
        return r.text.strip()
    return None

def main():
    p = argparse.ArgumentParser(description='Read SPSA tune status from OpenBench')
    p.add_argument('tune_id', nargs='?', help='Specific tune ID (omit for all)')
    p.add_argument('--csv', action='store_true', help='Output as CSV')
    p.add_argument('--outputs', action='store_true', help='Print SPSA outputs (for re-submission)')
    p.add_argument('--compare', nargs='+', help='Compare multiple tune IDs side by side')
    p.add_argument('--server', default=SERVER)
    p.add_argument('--username', default=USERNAME)
    p.add_argument('--password', default=PASSWORD)
    args = p.parse_args()

    if not args.password:
        print('Error: password required. Set OPENBENCH_PASSWORD or use --password')
        return

    s = login(args.server, args.username, args.password)

    if args.compare:
        # Side-by-side comparison
        all_params = {}
        headers = []
        for tid in args.compare:
            branch, progress = get_tune_info(s, args.server, tid)
            headers.append(f'{branch} (#{tid}, {progress})')
            params = get_tune_digest(s, args.server, tid)
            if params:
                all_params[tid] = {p['name']: p for p in params}

        if not all_params:
            print('No tune data found')
            return

        # Get all param names from first tune
        first_tid = args.compare[0]
        if first_tid not in all_params:
            print(f'No data for tune #{first_tid}')
            return

        print(f'{"Param":25s}', end='')
        for h in headers:
            print(f'  {h:>15s}', end='')
        print()
        print('-' * (25 + 17 * len(headers)))

        for pname in [p['name'] for p in all_params[first_tid].values()]:
            print(f'{pname:25s}', end='')
            for tid in args.compare:
                if tid in all_params and pname in all_params[tid]:
                    val = all_params[tid][pname]['curr']
                    print(f'  {val:>15.1f}', end='')
                else:
                    print(f'  {"?":>15s}', end='')
            print()
        return

    if args.tune_id:
        tids = [args.tune_id]
    else:
        tids = get_tune_ids(s, args.server)

    for tid in tids:
        branch, progress = get_tune_info(s, args.server, tid)

        if args.outputs:
            outputs = get_tune_outputs(s, args.server, tid)
            print(f'=== #{tid} {branch} ({progress}) ===')
            if outputs:
                print(outputs)
            print()
            continue

        params = get_tune_digest(s, args.server, tid)

        if args.csv:
            if params:
                print(f'# Tune #{tid}: {branch} ({progress})')
                print('Name,Curr,Start,Min,Max')
                for p in params:
                    print(f"{p['name']},{p['curr']:.4f},{p['start']:.0f},{p['min']:.0f},{p['max']:.0f}")
            continue

        print(f'=== #{tid} {branch} ({progress}) ===')
        if params:
            for p in params:
                delta = p['curr'] - p['start']
                pct = delta / p['start'] * 100 if p['start'] != 0 else 0
                marker = ' ***' if abs(pct) > 5 else ' *' if abs(pct) > 2 else ''
                print(f"  {p['name']:25s}  {p['curr']:>10.1f}  (start: {p['start']:>8.0f}, {pct:>+6.1f}%){marker}")
        print()

if __name__ == '__main__':
    main()
