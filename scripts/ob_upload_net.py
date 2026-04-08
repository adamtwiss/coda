#!/usr/bin/env python3
"""Upload NNUE network to OpenBench.

Usage:
    python3 ob_upload_net.py <file> [--name NAME]

If --name is not provided, uses the filename without extension.

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

def upload_net(args):
    if not os.path.exists(args.file):
        print(f'Error: file not found: {args.file}')
        return False

    name = args.name or os.path.splitext(os.path.basename(args.file))[0]
    size_mb = os.path.getsize(args.file) / 1024 / 1024

    print(f'Uploading {name} ({size_mb:.1f} MB)...')

    s = requests.Session()
    with open(args.file, 'rb') as f:
        r = s.post(f'{args.server}/scripts/', data={
            'username': args.username,
            'password': args.password,
            'action': 'UPLOAD_NETWORK',
            'engine': 'Coda',
            'name': name,
        }, files={
            'netfile': (os.path.basename(args.file), f, 'application/octet-stream'),
        })

    if r.status_code == 200 and '/networks' in r.url:
        print(f'Uploaded: {name}')
        return True

    # Check for error messages
    import re
    errors = re.findall(r'error-message.*?<pre>(.*?)</pre>', r.text, re.DOTALL)
    status = re.findall(r'status-message.*?<pre>(.*?)</pre>', r.text, re.DOTALL)
    for e in errors + status:
        print(f'Error: {e.strip()}')

    if not errors and not status:
        print(f'Failed: HTTP {r.status_code}')

    return False

def main():
    p = argparse.ArgumentParser(description='Upload NNUE network to OpenBench')
    p.add_argument('file', help='Path to .nnue file')
    p.add_argument('--name', default='', help='Network name (default: filename without extension)')
    p.add_argument('--server', default=SERVER)
    p.add_argument('--username', default=USERNAME)
    p.add_argument('--password', default=PASSWORD)
    args = p.parse_args()

    if not args.password:
        print('Error: password required. Set OPENBENCH_PASSWORD or use --password')
        return

    upload_net(args)

if __name__ == '__main__':
    main()
