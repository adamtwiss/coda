#!/usr/bin/env python3
"""Check OpenBench test status."""

import requests
import re
import sys

SERVER = 'https://ob.atwiss.com'

def get_status():
    s = requests.Session()
    r = s.get(f'{SERVER}/login/')
    csrf = s.cookies.get('csrftoken')
    s.post(f'{SERVER}/login/', data={
        'username': 'claude', 'password': 'SE1APo1O413Gn', 'csrfmiddlewaretoken': csrf,
    }, headers={'Referer': f'{SERVER}/login/'}, allow_redirects=False)

    r = s.get(f'{SERVER}/index/')
    html = r.text

    # Parse test rows — look for test links and LLR/Elo data
    # Active tests have pattern: /test/N/ followed by branch name and stats
    tests = []

    # Find all test IDs and their data
    test_pattern = r'test/(\d+)/'
    test_ids = sorted(set(re.findall(test_pattern, html)), key=int)

    for tid in test_ids:
        r2 = s.get(f'{SERVER}/test/{tid}/')
        t = r2.text

        # Extract key data
        elo_match = re.search(r'Elo\s*\|\s*([-\d.]+)\s*\+-\s*([\d.]+)', t)
        llr_match = re.search(r'LLR\s*\|\s*([-\d.]+)', t)
        games_match = re.search(r'Games\s*\|\s*N:\s*(\d+)', t)
        if not games_match:
            continue

        games = int(games_match.group(1))
        if games == 0:
            continue

        elo = float(elo_match.group(1)) if elo_match else 0
        elo_err = float(elo_match.group(2)) if elo_match else 0
        llr = float(llr_match.group(1)) if llr_match else 0

        # Extract dev branch name from page
        branch_match = re.search(r'Dev Branch</td><td class="branch_name">([^<]+)', t)
        name = branch_match.group(1).strip() if branch_match else f'test-{tid}'
        if len(name) > 32: name = name[:32]

        status = ''
        if llr >= 2.94: status = 'H1 ✓'
        elif llr <= -2.94: status = 'H0 ✗'
        elif llr > 0: status = '→H1'
        else: status = '→H0'

        tests.append((int(tid), name, elo, elo_err, games, llr, status))

    # Sort by Elo descending
    tests.sort(key=lambda x: x[2], reverse=True)

    print(f'{"ID":>4} {"Name":<35} {"Elo":>7} {"±":>6} {"Games":>7} {"LLR":>6} {"Status":<6}')
    print('-' * 80)
    for tid, name, elo, err, games, llr, status in tests:
        print(f'{tid:>4} {name:<35} {elo:>+7.1f} {err:>6.1f} {games:>7} {llr:>6.2f} {status}')

if __name__ == '__main__':
    get_status()
