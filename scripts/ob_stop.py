#!/usr/bin/env python3
"""Stop an OpenBench test via the web UI API.

Usage:
    python3 ob_stop.py <test_id>
    python3 ob_stop.py 93

Environment variables (or use --flags):
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

def stop_test(args):
    s = requests.Session()

    # Step 1: GET login page for CSRF token
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

    # Step 3: POST to /test/<id>/STOP/ with CSRF and Referer
    csrf = s.cookies.get('csrftoken')
    r = s.post(f'{args.server}/test/{args.test_id}/STOP/', data={
        'csrfmiddlewaretoken': csrf,
    }, headers={
        'Referer': f'{args.server}/test/{args.test_id}/',
    }, allow_redirects=False)

    if r.headers.get('Location', '') == '/index/':
        print(f'Test #{args.test_id} stopped.')
        return True

    print(f'Error: unexpected response {r.status_code} {r.headers.get("Location", "")}')
    return False

def main():
    p = argparse.ArgumentParser(description='Stop an OpenBench test')
    p.add_argument('test_id', type=int, help='Test ID to stop')
    p.add_argument('--server', default=SERVER, help=f'Server (default: {SERVER})')
    p.add_argument('--username', default=USERNAME, help='Username')
    p.add_argument('--password', default=PASSWORD, help='Password')
    args = p.parse_args()

    if not args.password:
        print('Error: password required. Set OPENBENCH_PASSWORD or use --password')
        return

    stop_test(args)

if __name__ == '__main__':
    main()
