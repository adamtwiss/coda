#!/bin/bash
# Log bench calibration metrics (NPS, EBF, FMC, evals/node, call-sites/node)
# for the current commit. Appends one row to docs/bench_calibration.csv.
#
# Usage:
#   ./scripts/log_bench_calibration.sh            # log current HEAD
#   ./scripts/log_bench_calibration.sh <commit>   # log specific commit
#
# Designed to be run after each merge to main (or manually for spot
# calibration), so we have a time-series of how trunk's tree shape
# evolves. Diffing CSV between rows highlights NPS regressions
# (Adam asked for periodic logging 2026-04-25).

set -e
cd "$(git rev-parse --show-toplevel)"

COMMIT="${1:-HEAD}"
SHA=$(git rev-parse --short "$COMMIT")
DATE=$(git show -s --format=%cI "$COMMIT")
SUBJECT=$(git show -s --format=%s "$COMMIT" | head -c 60 | tr ',' ';')

CSV=docs/bench_calibration.csv
if [ ! -f "$CSV" ]; then
    echo "sha,date,subject,nodes,nps,ebf,first_move_cut_pct,evals_per_node,call_sites_per_node" > "$CSV"
fi

# Build at the requested commit (if not HEAD). Use a temp checkout so
# we don't disturb the working tree.
if [ "$COMMIT" != "HEAD" ] && [ "$COMMIT" != "$(git rev-parse HEAD)" ]; then
    echo "ERROR: only HEAD supported for now (avoids working-tree thrash)" >&2
    echo "       checkout the commit yourself, then run with no args" >&2
    exit 1
fi

cargo build --release >/dev/null 2>&1

OUT=$(./target/release/coda bench 13 2>&1)
NODES_NPS=$(echo "$OUT" | grep -E "[0-9]+ nodes [0-9]+ nps$" | tail -1)
NODES=$(echo "$NODES_NPS" | awk '{print $1}')
NPS=$(echo "$NODES_NPS" | awk '{print $3}')
EBF=$(echo "$OUT" | grep "EBF (depth" | awk '{print $4}')
FMC=$(echo "$OUT" | grep -oE "first-move [0-9]+\.[0-9]+%" | head -1 | grep -oE "[0-9]+\.[0-9]+")
EVALS_PER=$(echo "$OUT" | grep "Evals / node:" | awk '{print $4}')
CALLS_PER=$(echo "$OUT" | grep "Call-sites / node:" | awk '{print $4}')

echo "$SHA,$DATE,$SUBJECT,$NODES,$NPS,$EBF,$FMC,$EVALS_PER,$CALLS_PER" >> "$CSV"
echo "Logged: $SHA  nodes=$NODES nps=$NPS ebf=$EBF fmc=$FMC% e/n=$EVALS_PER c/n=$CALLS_PER"
echo "  → $CSV"
