#!/usr/bin/env bash
# Triangular K=3 net-vs-net variance measurement.
#
# Convert 3 quantised.bin files to .nnue, bench each, upload to OB,
# and submit the 3 pairwise SPRTs to measure host-to-host training
# variance. With fixed seed (--seed 42) the goal is to know whether
# variance dropped below the prior ~15 Elo band.
#
# Usage:
#   triangular_net_variance.sh <hostA-quantised.bin> <hostB-quantised.bin> <hostC-quantised.bin>
#
# Requires: OPENBENCH_PASSWORD env var, OPENBENCH_USERNAME (default claude),
# coda binary built with `make`.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <hostA-quantised.bin> <hostB-quantised.bin> <hostC-quantised.bin>" >&2
    exit 1
fi

CODA=/home/adam/code/coda/coda
WORKDIR=$(mktemp -d -t triangular-net-XXXXXX)
echo "Workdir: $WORKDIR"

# Conversion flags must match training config exactly per
# project_convert_bullet_invocation.md — wrong flags silently corrupt nets.
CONVERT_FLAGS="--pairwise --screlu --hidden 16 --hidden2 32 --int8l1 --threats 66864 --kb-layout reckless"

declare -a LABELS=("A" "B" "C")
declare -a INPUTS=("$1" "$2" "$3")
declare -a NETPATHS=()
declare -a BENCHES=()
declare -a SHAS=()

# Step 1: Convert + bench each net.
for i in 0 1 2; do
    label="${LABELS[$i]}"
    input="${INPUTS[$i]}"
    out="$WORKDIR/net_${label}.nnue"
    echo "=== Converting host-${label}: $input ==="
    "$CODA" convert-bullet -i "$input" -o "$out" $CONVERT_FLAGS
    NETPATHS+=("$out")

    echo "=== Benching net_${label} ==="
    bench=$("$CODA" --nnue "$out" bench 2>&1 | tail -1 | awk '{print $1}')
    BENCHES+=("$bench")
    echo "net_${label} bench: $bench"

    echo "=== Uploading net_${label} to OB ==="
    upload_out=$(python3 /home/adam/code/coda/scripts/ob_upload_net.py "$out" 2>&1)
    echo "$upload_out"
    sha=$(echo "$upload_out" | grep -oE 'SHA256: [A-F0-9]{8}' | awk '{print $2}')
    if [ -z "$sha" ]; then
        echo "ERROR: failed to extract SHA from upload" >&2
        exit 1
    fi
    SHAS+=("$sha")
    echo "net_${label} sha: $sha"
done

# Step 2: Print bench + tree-shape comparison summary.
echo ""
echo "=== Bench summary ==="
for i in 0 1 2; do
    echo "net_${LABELS[$i]}: bench=${BENCHES[$i]}  sha=${SHAS[$i]}"
done

# Compute spread.
min_bench=$(printf '%s\n' "${BENCHES[@]}" | sort -n | head -1)
max_bench=$(printf '%s\n' "${BENCHES[@]}" | sort -n | tail -1)
spread_pct=$(python3 -c "print(f'{100.0 * ($max_bench - $min_bench) / $min_bench:.2f}%')")
echo "Bench spread: $min_bench .. $max_bench (${spread_pct})"

# Step 3: Submit 3 pairwise SPRTs. Bounds [-3, 3] = "are these the same or different?"
# Per feedback_net_vs_net_per_side_bench.md, dev_bench / base_bench come from per-side
# bench measurements, NOT trunk's net.txt bench.
echo ""
echo "=== Submitting triangular SPRTs ==="
declare -a PAIRS=("0:1" "0:2" "1:2")
for pair in "${PAIRS[@]}"; do
    di="${pair%:*}"; bi="${pair#*:}"
    dl="${LABELS[$di]}"; bl="${LABELS[$bi]}"
    echo "--- net_${dl} (${SHAS[$di]}) vs net_${bl} (${SHAS[$bi]}) ---"
    OPENBENCH_PASSWORD="$OPENBENCH_PASSWORD" python3 /home/adam/code/coda/scripts/ob_submit.py \
        main "${BENCHES[$di]}" \
        --base-bench "${BENCHES[$bi]}" \
        --dev-network "${SHAS[$di]}" \
        --base-network "${SHAS[$bi]}" \
        --bounds '[-3, 3]'
done

echo ""
echo "Done. Watch via: python3 scripts/ob_status.py"
echo "Workdir kept at: $WORKDIR"
