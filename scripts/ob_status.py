#!/usr/bin/env python3
"""Check OpenBench test status — shows active and recently finished tests."""

import os
import requests
import re

SERVER   = os.environ.get('OPENBENCH_SERVER',   'https://ob.atwiss.com')
USERNAME = os.environ.get('OPENBENCH_USERNAME', 'claude')
PASSWORD = os.environ.get('OPENBENCH_PASSWORD', '')

def get_status():
    if not PASSWORD:
        print('Error: password required. Set OPENBENCH_PASSWORD or use env var')
        return

    s = requests.Session()
    r = s.get(f'{SERVER}/login/')
    csrf = s.cookies.get('csrftoken')
    s.post(f'{SERVER}/login/', data={
        'username': USERNAME, 'password': PASSWORD, 'csrfmiddlewaretoken': csrf,
    }, headers={'Referer': f'{SERVER}/login/'}, allow_redirects=False)

    r = s.get(f'{SERVER}/index/')
    html = r.text

    # Split into Active and Finished sections
    active_start = html.find('>Active')
    finished_start = html.find('>Finished')

    if active_start < 0 and finished_start < 0:
        print('Error: could not parse index page sections')
        return

    # Handle missing sections (e.g. no active tests)
    if active_start < 0:
        active_html = ''
        finished_html = html[finished_start:] if finished_start >= 0 else ''
    elif finished_start < 0:
        active_html = html[active_start:]
        finished_html = ''
    else:
        active_html = html[active_start:finished_start]
        finished_html = html[finished_start:]

    def parse_tests(section_html):
        """Extract test IDs from a section of the index page."""
        return re.findall(r'/test/(\d+)/', section_html)

    active_ids = set(parse_tests(active_html))
    finished_ids = set(parse_tests(finished_html))

    # Also find tune IDs
    tune_ids = set(re.findall(r'/tune/(\d+)/', html))

    all_test_ids = sorted(active_ids | finished_ids, key=int)

    active_tests = []
    finished_tests = []

    for tid in all_test_ids:
        r2 = s.get(f'{SERVER}/test/{tid}/')
        t = r2.text

        elo_match = re.search(r'Elo\s*\|\s*([-\d.]+)\s*\+-\s*([\d.]+)', t)
        llr_match = re.search(r'LLR\s*\|\s*([-\d.]+)', t)
        games_match = re.search(r'Games\s*\|\s*N:\s*(\d+)', t)
        if not games_match:
            continue

        games = int(games_match.group(1))

        elo = float(elo_match.group(1)) if elo_match else 0
        elo_err = float(elo_match.group(2)) if elo_match else 0
        llr = float(llr_match.group(1)) if llr_match else 0

        branch_match = re.search(r'Dev Branch</td><td class="branch_name">([^<]+)', t)
        name = branch_match.group(1).strip() if branch_match else f'test-{tid}'
        if len(name) > 40: name = name[:40]

        status = ''
        if llr >= 2.94: status = 'H1 ✓'
        elif llr <= -2.94: status = 'H0 ✗'
        elif llr > 0: status = '→H1'
        else: status = '→H0'

        entry = (int(tid), name, elo, elo_err, games, llr, status)
        if tid in active_ids:
            active_tests.append(entry)
        else:
            finished_tests.append(entry)

    # Sort active by Elo descending, finished by Elo descending
    active_tests.sort(key=lambda x: x[2], reverse=True)
    finished_tests.sort(key=lambda x: x[2], reverse=True)

    header = f'{"ID":>4} {"Name":<42} {"Elo":>7} {"±":>6} {"Games":>7} {"LLR":>6} {"Status":<6}'
    sep = '-' * 82

    if active_tests:
        print(f'=== ACTIVE ({len(active_tests)} tests) ===')
        print(header)
        print(sep)
        for tid, name, elo, err, games, llr, status in active_tests:
            print(f'{tid:>4} {name:<42} {elo:>+7.1f} {err:>6.1f} {games:>7} {llr:>6.2f} {status}')
        print()

    if finished_tests:
        print(f'=== FINISHED ({len(finished_tests)} tests) ===')
        print(header)
        print(sep)
        for tid, name, elo, err, games, llr, status in finished_tests:
            print(f'{tid:>4} {name:<42} {elo:>+7.1f} {err:>6.1f} {games:>7} {llr:>6.2f} {status}')

if __name__ == '__main__':
    get_status()
