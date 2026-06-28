#!/usr/bin/env bash
# Fresh NNUE inference profiling driver.
# Usage: scripts/profile_inference.sh <net.nnue> [bench_depth]
# Requires: OB worker stopped (single-engine, full NPS). perf installed.
set -u
NET="${1:?need net path}"
DEPTH="${2:-13}"
OUT=/tmp/coda_prof
mkdir -p "$OUT"

echo "### net: $NET   depth: $DEPTH   cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
echo "### bench NPS (3 runs, override net):"
for i in 1 2 3; do ./coda bench "$DEPTH" -n "$NET" 2>/dev/null | tail -1; done

echo; echo "### perf stat (cycles / instructions / cache):"
perf stat -e task-clock,cycles,instructions,branches,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses \
  ./coda bench "$DEPTH" -n "$NET" >/dev/null 2>"$OUT/stat.txt"
cat "$OUT/stat.txt"

echo; echo "### perf record -> top symbols (self %):"
perf record -g -F 2999 -o "$OUT/perf.data" -- ./coda bench "$DEPTH" -n "$NET" >/dev/null 2>&1
perf report -i "$OUT/perf.data" --stdio --sort=overhead -g none 2>/dev/null \
  | grep -E '^\s+[0-9]' | head -35
echo
echo "### (full callgraph at $OUT/perf.data — perf report -i $OUT/perf.data)"
