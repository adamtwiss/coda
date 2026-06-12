#!/usr/bin/env bash
# Scripted UCI deployment-correctness tests (TM audit 2026-06-13, Track A).
#
# Covers the [deploy]-class behaviors that OB SPRTs cannot see:
#   a1) go wtime 0 btime 0        -> bestmove within ~2s (no unbounded search)
#   a2) go wtime -50 btime -50    -> bestmove within ~2s (negative clock clamps)
#   b)  go movetime 0             -> immediate-ish bestmove
#   c)  go infinite               -> NO bestmove until `stop`, then prompt
#   d)  go ponder + ponderhit     -> bestmove respects the post-ponderhit budget
#   e)  go ponder + stop          -> prompt bestmove
#
# Usage: tests/uci_deploy_tests.sh [path-to-engine]   (default: ./coda)
# Requires GNU date (%3N). Exits non-zero on any failure.

set -u
ENGINE="${1:-./coda}"
if [ ! -x "$ENGINE" ]; then
    echo "FATAL: engine binary '$ENGINE' not found/executable (run 'make' first)" >&2
    exit 2
fi

TMP="$(mktemp -d /tmp/coda_uci_deploy.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
FAILURES=0

now_ms() { date +%s%3N; }

# mark <case> <label> — record a send-side timestamp
mark() { printf '%s %s\n' "$(now_ms)" "$2" >> "$TMP/$1.marks"; }

# get_mark <case> <label> -> ms timestamp (empty if missing)
get_mark() { awk -v l="$2" '$2==l{print $1; exit}' "$TMP/$1.marks" 2>/dev/null; }

# first_bestmove_ts <case> -> ms timestamp of first bestmove line (empty if none)
first_bestmove_ts() { awk '$2=="bestmove"{print $1; exit}' "$TMP/$1.out" 2>/dev/null; }

# run_case <case-name> <input-fn>: pipe timed input into the engine,
# timestamp every output line. Whole case capped at 30s.
run_case() {
    local name="$1" infn="$2"
    : > "$TMP/$name.marks"
    ( "$infn" "$name" ) | timeout 30 "$ENGINE" 2>/dev/null | \
        while IFS= read -r line; do
            printf '%s %s\n' "$(now_ms)" "$line"
        done > "$TMP/$name.out"
}

pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; FAILURES=$((FAILURES + 1)); }

# ---------------------------------------------------------------- case a1
in_a1() {
    echo "uci"; echo "isready"; echo "position startpos"
    mark "$1" go
    echo "go wtime 0 btime 0"
    sleep 3
    echo "quit"
}
check_fast_bestmove() {  # <case> <desc> <max_ms>
    local bm go
    bm=$(first_bestmove_ts "$1"); go=$(get_mark "$1" go)
    if [ -z "$bm" ]; then fail "$2 — no bestmove emitted (hang/unbounded search)"; return; fi
    local dt=$((bm - go))
    if [ "$dt" -le "$3" ]; then pass "$2 — bestmove after ${dt}ms (limit $3)"; else fail "$2 — bestmove took ${dt}ms (> $3)"; fi
}
run_case a1 in_a1
check_fast_bestmove a1 "(a1) go wtime 0 btime 0" 2000

# ---------------------------------------------------------------- case a2
in_a2() {
    echo "uci"; echo "isready"; echo "position startpos"
    mark "$1" go
    echo "go wtime -50 btime -50"
    sleep 3
    echo "quit"
}
run_case a2 in_a2
check_fast_bestmove a2 "(a2) go wtime -50 btime -50" 2000

# ---------------------------------------------------------------- case b
in_b() {
    echo "uci"; echo "isready"; echo "position startpos"
    mark "$1" go
    echo "go movetime 0"
    sleep 3
    echo "quit"
}
run_case b in_b
check_fast_bestmove b "(b) go movetime 0" 1500

# ---------------------------------------------------------------- case c
in_c() {
    echo "uci"; echo "isready"; echo "position startpos"
    mark "$1" go
    echo "go infinite"
    sleep 3
    mark "$1" stop
    echo "stop"
    sleep 2
    echo "quit"
}
run_case c in_c
bm=$(first_bestmove_ts c); go=$(get_mark c go); st=$(get_mark c stop)
if [ -z "$bm" ]; then
    fail "(c) go infinite — no bestmove after stop"
else
    dt_go=$((bm - go)); dt_stop=$((bm - st))
    # No bestmove before stop (allow 100ms pipe slop), prompt after stop.
    if [ "$dt_go" -lt 2900 ]; then
        fail "(c) go infinite — bestmove emitted ${dt_go}ms after go, BEFORE stop (UCI violation)"
    elif [ "$dt_stop" -gt 1000 ]; then
        fail "(c) go infinite — bestmove took ${dt_stop}ms after stop (> 1000)"
    else
        pass "(c) go infinite — silent for ${dt_go}ms, bestmove ${dt_stop}ms after stop"
    fi
fi

# ---------------------------------------------------------------- case c2
# Bare-kings position: search exhausts depth 64 (MAX_PLY/2) in a few ms,
# so without the A4-residual wait the engine emits bestmove immediately
# during `go infinite` (verified failing on pre-fix binary).
in_c2() {
    echo "uci"; echo "isready"
    echo "position fen k7/8/8/8/8/8/8/7K w - - 0 1"
    mark "$1" go
    echo "go infinite"
    sleep 3
    mark "$1" stop
    echo "stop"
    sleep 2
    echo "quit"
}
run_case c2 in_c2
bm=$(first_bestmove_ts c2); go=$(get_mark c2 go); st=$(get_mark c2 stop)
if [ -z "$bm" ]; then
    fail "(c2) go infinite (depth-exhausted) — no bestmove after stop"
else
    dt_go=$((bm - go)); dt_stop=$((bm - st))
    if [ "$dt_go" -lt 2900 ]; then
        fail "(c2) go infinite (depth-exhausted) — bestmove ${dt_go}ms after go, BEFORE stop (UCI violation)"
    elif [ "$dt_stop" -gt 1000 ]; then
        fail "(c2) go infinite (depth-exhausted) — bestmove took ${dt_stop}ms after stop (> 1000)"
    else
        pass "(c2) go infinite (depth-exhausted) — silent for ${dt_go}ms, bestmove ${dt_stop}ms after stop"
    fi
fi

# ---------------------------------------------------------------- case d
in_d() {
    echo "uci"; echo "isready"; echo "position startpos"
    echo "go ponder wtime 60000 btime 60000 winc 1000 binc 1000"
    sleep 0.4
    mark "$1" ponderhit
    echo "ponderhit"
    sleep 6
    echo "quit"
}
run_case d in_d
bm=$(first_bestmove_ts d); ph=$(get_mark d ponderhit)
if [ -z "$bm" ]; then
    fail "(d) go ponder + ponderhit — no bestmove (hang)"
else
    dt=$((bm - ph))
    # Budget at 60+1 fullmove-1: soft ~0.6s (phase-scaled), hard 27.6s.
    # Respecting the soft budget means well under ~3s; hitting hard
    # (the A1 stale-soft overspend) would be 27s+.
    if [ "$dt" -ge 30 ] && [ "$dt" -le 3000 ]; then
        pass "(d) ponderhit — bestmove ${dt}ms after ponderhit (within soft budget)"
    elif [ "$dt" -lt 30 ]; then
        fail "(d) ponderhit — instant emit ${dt}ms (floor/budget not honoured)"
    else
        fail "(d) ponderhit — bestmove took ${dt}ms (overspending toward hard window)"
    fi
fi

# ---------------------------------------------------------------- case e
in_e() {
    echo "uci"; echo "isready"; echo "position startpos"
    echo "go ponder wtime 60000 btime 60000 winc 1000 binc 1000"
    sleep 0.5
    mark "$1" stop
    echo "stop"
    sleep 2
    echo "quit"
}
run_case e in_e
bm=$(first_bestmove_ts e); st=$(get_mark e stop)
if [ -z "$bm" ]; then
    fail "(e) go ponder + stop — no bestmove (hang)"
else
    dt=$((bm - st))
    if [ "$dt" -le 1000 ]; then
        pass "(e) go ponder + stop — bestmove ${dt}ms after stop"
    else
        fail "(e) go ponder + stop — bestmove took ${dt}ms after stop (> 1000)"
    fi
fi

# ----------------------------------------------------------------
echo
if [ "$FAILURES" -eq 0 ]; then
    echo "ALL UCI DEPLOY TESTS PASSED"
    exit 0
else
    echo "$FAILURES TEST(S) FAILED"
    exit 1
fi
