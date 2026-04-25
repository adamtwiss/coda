#!/usr/bin/env bash
# NPS microbench harness for Coda vs Reckless.
#
# Builds both engines (AVX-512 + AVX-2 variants on x86_64; native on ARM),
# runs the eval-bench matrix on the same 8 bench positions, and prints a
# CSV-friendly summary. Intended to be run on multiple machines to check
# whether our 2× NPS gap is a platform-specific artefact or architectural.
#
# Usage:
#   # Prerequisites
#   #   - rustup with a stable toolchain on PATH
#   #   - ~/code/coda checked out, branch containing `coda eval-bench`
#   #   - ~/chess/engines/Reckless checked out, branch containing
#   #     `evalbench` (Coda-comparison hack in src/tools/eval_bench.rs)
#   #   - The production v9 net at
#   #     coda/net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue
#   #     (or `make -C ~/code/coda net` to fetch it)
#   ./scripts/nps_microbench.sh
#
# Results go to stdout and nps_microbench_<host>_<date>.csv. Share
# the CSV back in Slack / PR so we can build a per-arch matrix.
#
# This script does NOT touch the OpenBench worker — stop it yourself if
# you want clean CPU (`~/code/OpenBench/ob-worker.sh stop`).

set -euo pipefail

CODA_DIR=${CODA_DIR:-$HOME/code/coda}
RECKLESS_DIR=${RECKLESS_DIR:-$HOME/chess/engines/Reckless}
V9_NET=${V9_NET:-$CODA_DIR/net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue}

REPS_EVAL=${REPS_EVAL:-10000}
REPS_MAKEUNMAKE=${REPS_MAKEUNMAKE:-1000000}

HOST=$(hostname -s)
DATE=$(date +%Y-%m-%d)
OUT=${OUT:-nps_microbench_${HOST}_${DATE}.csv}

# Detect architecture and available ISA tiers.
ARCH=$(uname -m)
HAS_AVX512=0
HAS_VNNI=0
if [[ "$ARCH" == "x86_64" ]]; then
    if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then HAS_AVX512=1; fi
    if grep -qw avx512_vnni /proc/cpuinfo 2>/dev/null || \
       grep -qw avx_vnni /proc/cpuinfo 2>/dev/null; then HAS_VNNI=1; fi
fi

# Identify the CPU model for the report.
CPU_MODEL="unknown"
if [[ -f /proc/cpuinfo ]]; then
    CPU_MODEL=$(awk -F: '/^model name/{gsub(/^ +| +$/,"",$2); print $2; exit}' /proc/cpuinfo || true)
    if [[ -z "$CPU_MODEL" ]]; then
        # Apple Silicon / ARM — /proc/cpuinfo has "Model" or similar.
        CPU_MODEL=$(awk -F: '/^Hardware|^model name|^Model/{gsub(/^ +| +$/,"",$2); print $2; exit}' /proc/cpuinfo || true)
    fi
fi
[[ -z "$CPU_MODEL" ]] && CPU_MODEL=$(uname -p)

cat <<EOF
# NPS microbench — $HOST ($DATE)
# CPU: $CPU_MODEL
# Arch: $ARCH   AVX-512: $HAS_AVX512   VNNI: $HAS_VNNI
# Coda dir: $CODA_DIR
# Reckless dir: $RECKLESS_DIR
# V9 net: $V9_NET
# Reps: eval=$REPS_EVAL, make-unmake=$REPS_MAKEUNMAKE
EOF

if [[ ! -f "$V9_NET" ]]; then
    echo "ERROR: v9 net not found at $V9_NET"
    echo "Run: (cd $CODA_DIR && make net)"
    exit 1
fi

# ---- Build ----
echo ""
echo "==== Building Coda (native) ===="
(cd "$CODA_DIR" && cargo build --release 2>&1 | tail -2)
CODA_BIN="$CODA_DIR/target/release/coda"

echo ""
echo "==== Building Reckless (native, no-default-features to skip syzygy) ===="
(cd "$RECKLESS_DIR" && cargo build --release --no-default-features 2>&1 | tail -2)
RECKLESS_BIN="$RECKLESS_DIR/target/release/reckless"

# On x86_64, also produce an AVX-2-only Reckless build so we can test
# the lower ISA tier. Coda doesn't need a separate build — `--no-avx512`
# flips the runtime dispatch.
RECKLESS_AVX2_BIN=""
if [[ "$ARCH" == "x86_64" && "$HAS_AVX512" == "1" ]]; then
    echo ""
    echo "==== Building Reckless AVX-2-only ===="
    (cd "$RECKLESS_DIR" && RUSTFLAGS="-Ctarget-cpu=x86-64-v3" \
        cargo build --release --no-default-features --target-dir target-avx2 2>&1 | tail -2)
    RECKLESS_AVX2_BIN="$RECKLESS_DIR/target-avx2/release/reckless"
fi

# ---- Helpers ----
# Run Coda eval-bench and extract evals/sec.
# Args: $1=mode  $2=reps  $3=extra flags (e.g. --no-avx512)
coda_bench() {
    local mode=$1 reps=$2 extra=${3:-}
    "$CODA_BIN" eval-bench --mode "$mode" --reps "$reps" -n "$V9_NET" $extra 2>&1 \
        | awk -F'evals/sec=' '/evals\/sec=/{print $2}' | awk '{print $1}'
}

# Run Reckless evalbench and extract evals/sec.
# Args: $1=mode  $2=reps  $3=binary
reckless_bench() {
    local mode=$1 reps=$2 bin=$3
    printf 'evalbench %s %d\nquit\n' "$mode" "$reps" | "$bin" 2>&1 \
        | awk -F'evals/sec=' '/evals\/sec=/{print $2}' | awk '{print $1}'
}

# ---- Run the matrix ----
rows=()
rows+=("host,cpu,arch,engine,isa,mode,evals_per_sec")

echo ""
echo "==== Running Coda eval-bench (native / auto-detect) ===="
for mode in fresh refresh incremental make-unmake; do
    v=$(coda_bench "$mode" $REPS_EVAL)
    echo "coda   native    $mode: $v evals/sec"
    rows+=("$HOST,\"$CPU_MODEL\",$ARCH,coda,native,$mode,$v")
done

if [[ "$ARCH" == "x86_64" && "$HAS_AVX512" == "1" ]]; then
    echo ""
    echo "==== Running Coda eval-bench (AVX-2 forced) ===="
    for mode in fresh refresh incremental make-unmake; do
        v=$(coda_bench "$mode" $REPS_EVAL --no-avx512)
        echo "coda   avx2      $mode: $v evals/sec"
        rows+=("$HOST,\"$CPU_MODEL\",$ARCH,coda,avx2,$mode,$v")
    done
fi

echo ""
echo "==== Running Reckless eval-bench (native) ===="
for mode in fresh incremental make-unmake make-unmake-null; do
    v=$(reckless_bench "$mode" $REPS_EVAL "$RECKLESS_BIN")
    echo "reckless native    $mode: $v evals/sec"
    rows+=("$HOST,\"$CPU_MODEL\",$ARCH,reckless,native,$mode,$v")
done

if [[ -n "$RECKLESS_AVX2_BIN" ]]; then
    echo ""
    echo "==== Running Reckless eval-bench (AVX-2 build) ===="
    for mode in fresh incremental make-unmake make-unmake-null; do
        v=$(reckless_bench "$mode" $REPS_EVAL "$RECKLESS_AVX2_BIN")
        echo "reckless avx2      $mode: $v evals/sec"
        rows+=("$HOST,\"$CPU_MODEL\",$ARCH,reckless,avx2,$mode,$v")
    done
fi

# ---- Also run full search bench on Coda for evals/node counters ----
echo ""
echo "==== Coda search bench depth 13 (evals/node stats) ===="
"$CODA_BIN" bench 13 -n "$V9_NET" 2>&1 | awk '
    /^Total nodes:/                    { printf "total_nodes=%s\n", $3 }
    /^NNUE full rebuilds:/             { gsub(/[()%]/, "", $5); printf "full_rebuilds=%s pct=%s\n", $4, $5 }
    /^NNUE incremental:/               { gsub(/[()%]/, "", $4); printf "incremental=%s pct=%s\n", $3, $4 }
    /^TT static-eval hit:/             { gsub(/[()%]/, "", $6); printf "tt_eval_hits=%s pct=%s\n", $4, $6 }
    /^Already-computed:/               { printf "already_computed=%s\n", $2 }
    /^Evals \/ node:/                  { printf "evals_per_node=%s\n", $4 }
    /^Call-sites \/ node:/             { printf "call_sites_per_node=%s\n", $4 }
    /nps$/                              { printf "search_nps=%s\n", $(NF-1) }
'

# ---- Same for Reckless ----
echo ""
echo "==== Reckless search bench (nps only; dbg_hit eval counters require a separate instrumented build) ===="
printf 'bench\nquit\n' | "$RECKLESS_BIN" 2>&1 | awk '
    /^Bench:/  { printf "reckless_bench=%s nps=%s\n", $2, $4 }
'
# To collect per-perspective eval counters on a new host, uncomment the
# dbg_hit block in Reckless src/nnue.rs::evaluate (slots 0-6), rebuild,
# and re-run `bench`. This is a one-time measurement — the rates are
# search-behavioural, not per-host.

# ---- Write CSV ----
printf '%s\n' "${rows[@]}" > "$OUT"
echo ""
echo "CSV written to: $OUT"
echo ""
echo "Summary (compact):"
printf '%s\n' "${rows[@]}" | column -t -s','
