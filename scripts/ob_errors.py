#!/usr/bin/env python3
"""Fetch and parse the OpenBench /errors/ page into something readable.

The raw page stores each error's time as a bare Unix epoch in a
`<td class="timestamp">` cell, with the test id/branch and summary in
sibling cells. Grepping the summary strings (e.g. "Wrong Bench") in
isolation is the trap — a 26-hour-old error and a live one look
identical once the Date column is stripped off. This tool keeps the row
intact: it shows AGE + absolute time + test id + branch + summary, and
by default only surfaces errors inside a recent window so stale entries
don't masquerade as current.

Typical uses:
  scripts/ob_errors.py                  # errors in the last 6h (post-submit check)
  scripts/ob_errors.py --hours 1        # tighter window
  scripts/ob_errors.py --test 2106      # did THIS test error? (any age)
  scripts/ob_errors.py --user claude    # only my rows
  scripts/ob_errors.py --branch main    # only rows on a branch
  scripts/ob_errors.py --all --pages 3  # raw recent rows, no time filter

Exit code is 1 if any error matched the window/filters (so a caller can
gate on "are there fresh errors?"), else 0.
"""

import argparse
import html
import os
import re
import signal
import sys
import time

import requests

# Don't traceback when piped into `head`/`less` and the reader closes early.
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):
    pass  # non-Unix / non-main-thread

SERVER = os.environ.get('OPENBENCH_SERVER', 'https://ob.atwiss.com')

ROW_RE = re.compile(r'<tr>(?!<th)(.*?)</tr>', re.S)
CELL_RE = re.compile(r'<td[^>]*>(.*?)</td>', re.S)
TEST_RE = re.compile(r'/test/(\d+)/?"[^>]*>([^<]*)<')
EVENT_RE = re.compile(r'/event/(\d+)')


def strip_tags(s):
    return re.sub(r'<[^>]+>', '', s).strip()


def fmt_age(secs):
    secs = int(secs)
    if secs < 0:
        return '0s'
    if secs < 90:
        return f'{secs}s'
    if secs < 90 * 60:
        return f'{secs // 60}m'
    if secs < 48 * 3600:
        return f'{secs // 3600}h'
    return f'{secs // 86400}d'


def parse_page(html):
    """Return list of dicts for each error row on the page."""
    rows = []
    for raw in ROW_RE.findall(html):
        cells = CELL_RE.findall(raw)
        if len(cells) < 5:
            continue
        ts_txt = strip_tags(cells[0])
        if not ts_txt.isdigit():
            continue  # header / malformed row
        tm = TEST_RE.search(cells[3])
        ev = EVENT_RE.search(cells[5]) if len(cells) > 5 else None
        rows.append({
            'ts': int(ts_txt),
            'machine': strip_tags(cells[1]),
            'user': strip_tags(cells[2]),
            'test_id': tm.group(1) if tm else '?',
            'branch': tm.group(2).strip() if tm else strip_tags(cells[3]),
            'summary': strip_tags(cells[4]),
            'event': ev.group(1) if ev else None,
        })
    return rows


ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
PRE_RE = re.compile(r'<pre[^>]*>(.*?)</pre>', re.S)


def fetch_log(event_id, tail):
    """Print the build/run log for an /event/<id>/ page (the <pre> block).

    Build errors live at the END of cargo output, so show the last `tail`
    lines by default (tail=0 → whole log)."""
    url = f'{SERVER}/event/{event_id}/'
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f'fetch error on {url}: {e}', file=sys.stderr)
        return 2
    m = PRE_RE.search(r.text)
    if not m:
        print(f'no log <pre> block found at {url}', file=sys.stderr)
        return 2
    log = ANSI_RE.sub('', html.unescape(m.group(1))).strip('\n')
    lines = log.split('\n')
    print(f'=== {url}  ({len(lines)} lines) ===')
    if tail and len(lines) > tail:
        print(f'... [{len(lines) - tail} earlier lines hidden; --tail 0 for full] ...')
        lines = lines[-tail:]
    print('\n'.join(lines))
    return 0


def fetch(max_pages, cutoff=None):
    """Fetch error rows newest-first. With a cutoff, stop once a page already
    reaches past it (pages are time-ordered, so later pages are only older)."""
    out, sess = [], requests.Session()
    for p in range(1, max_pages + 1):
        url = f'{SERVER}/errors/' if p == 1 else f'{SERVER}/errors/{p}'
        try:
            r = sess.get(url, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f'fetch error on {url}: {e}', file=sys.stderr)
            break
        page_rows = parse_page(r.text)
        if not page_rows:
            break
        out.extend(page_rows)
        if cutoff is not None and min(row['ts'] for row in page_rows) < cutoff:
            break  # this page crosses the window; nothing newer beyond it
    return out


def main():
    ap = argparse.ArgumentParser(description='Readable OpenBench /errors view')
    ap.add_argument('--hours', type=float, default=6.0,
                    help='lookback window in hours (default 6; ignored with --all/--test)')
    ap.add_argument('--all', action='store_true', help='no time filter')
    ap.add_argument('--test', help='filter to a test id (implies no time filter)')
    ap.add_argument('--user', help='filter to a user (case-insensitive substring)')
    ap.add_argument('--branch', help='filter to a branch (case-insensitive substring)')
    ap.add_argument('--pages', type=int, default=0,
                    help='max pages to fetch (default: auto until older than window, cap 10)')
    ap.add_argument('--limit', type=int, default=40,
                    help='max rows to print (default 40; rest summarised)')
    ap.add_argument('--log', metavar='EVENT_ID',
                    help='instead of listing errors, print the build/run log '
                         'for an event id (shown as "log <id>" on build-fail rows)')
    ap.add_argument('--tail', type=int, default=80,
                    help='with --log, show the last N lines (default 80; 0 = full)')
    args = ap.parse_args()

    if args.log:
        return fetch_log(args.log, args.tail)

    now = int(time.time())
    time_filtered = not (args.all or args.test)
    cutoff = now - int(args.hours * 3600) if time_filtered else None

    if args.pages > 0:
        max_pages = args.pages
    elif time_filtered:
        max_pages = 10        # early-stops at the window via cutoff
    elif args.test:
        max_pages = 5         # search recent history for one test id
    else:                     # --all
        max_pages = 1
    rows = fetch(max_pages, cutoff)

    def keep(r):
        if cutoff is not None and r['ts'] < cutoff:
            return False
        if args.test and r['test_id'] != str(args.test):
            return False
        if args.user and args.user.lower() not in r['user'].lower():
            return False
        if args.branch and args.branch.lower() not in r['branch'].lower():
            return False
        return True

    hits = [r for r in rows if keep(r)]
    hits.sort(key=lambda r: r['ts'], reverse=True)

    scope = ('all' if args.all else
             f'test {args.test}' if args.test else
             f'last {args.hours:g}h')
    extra = []
    if args.user:
        extra.append(f'user~{args.user}')
    if args.branch:
        extra.append(f'branch~{args.branch}')
    scope += (' · ' + ' · '.join(extra)) if extra else ''
    print(f'OB /errors — {scope}  (now {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now))})\n')

    if hits:
        print(f'⚠ {len(hits)} error row(s):\n')
        print(f'  {"AGE":>4}  {"TIME (UTC)":<16}  {"TEST":>5}  {"USER":<7} {"BRANCH":<34} SUMMARY')
        for r in hits[:args.limit]:
            t = time.strftime('%Y-%m-%d %H:%M', time.gmtime(r['ts']))
            log_hint = f'   [log {r["event"]}]' if r['event'] else ''
            print(f'  {fmt_age(now - r["ts"]):>4}  {t:<16}  {r["test_id"]:>5}  '
                  f'{r["user"][:7]:<7} {r["branch"][:34]:<34} {r["summary"]}{log_hint}')
        if len(hits) > args.limit:
            print(f'  ... {len(hits) - args.limit} more (raise --limit)')
        if any(r['event'] for r in hits[:args.limit]):
            print('\n  view a build log:  scripts/ob_errors.py --log <id>')
    else:
        print(f'✓ no errors{"" if not (args.user or args.branch) else " matching"} in {scope}')

    # Always show the single most recent error overall, so an empty window
    # is obviously "stale page", not "tool fetched nothing".
    if rows:
        mr = max(rows, key=lambda r: r['ts'])
        t = time.strftime('%Y-%m-%d %H:%M', time.gmtime(mr['ts']))
        log_hint = f'   [log {mr["event"]}]' if mr['event'] else ''
        print(f'\nmost recent error overall: {fmt_age(now - mr["ts"])} ago  {t}  '
              f'test {mr["test_id"]}  {mr["user"]}  {mr["branch"]}')
        print(f'  {mr["summary"]}{log_hint}')

    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
