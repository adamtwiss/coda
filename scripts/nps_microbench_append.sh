#!/usr/bin/env bash
# Format the most recent nps_microbench CSV as a markdown block and
# append it to docs/nps_microbench_hostdata.md.
#
# Run this AFTER nps_microbench.sh. Looks for
# nps_microbench_${HOST}_*.csv in the current directory.
#
# Usage:
#   ./scripts/nps_microbench_append.sh
#   CSV=nps_microbench_foo_2026-04-23.csv ./scripts/nps_microbench_append.sh

set -euo pipefail

CODA_DIR=${CODA_DIR:-$HOME/code/coda}
DOC="$CODA_DIR/docs/nps_microbench_hostdata.md"
HOST=$(hostname -s)
DATE=$(date +%Y-%m-%d)

CSV=${CSV:-$(ls -t "$CODA_DIR"/nps_microbench_${HOST}_*.csv 2>/dev/null | head -1 || true)}
if [[ -z "$CSV" || ! -f "$CSV" ]]; then
    # Also check current directory
    CSV=$(ls -t "./nps_microbench_${HOST}_"*.csv 2>/dev/null | head -1 || true)
fi
if [[ -z "$CSV" || ! -f "$CSV" ]]; then
    echo "ERROR: no nps_microbench CSV found for host '$HOST'."
    echo "Run: ./scripts/nps_microbench.sh first."
    exit 1
fi

if [[ ! -f "$DOC" ]]; then
    echo "ERROR: doc $DOC not found — create it first or pull latest trunk."
    exit 1
fi

# Pull CPU model from CSV (quoted field 2).
CPU=$(awk -F, 'NR==2 {gsub(/"/,"",$2); print $2; exit}' "$CSV")

# Build a host block. Extract evals/sec per (engine, isa, mode).
get() {
    local engine=$1 isa=$2 mode=$3
    awk -F, -v e="$engine" -v i="$isa" -v m="$mode" \
        'NR>1 && $4==e && $5==i && $6==m {print $7; exit}' "$CSV"
}

fmt_int() {
    if [[ -z "$1" ]]; then echo "—"; else printf "%'d" "$1"; fi
}

ratio() {
    local a=$1 b=$2
    if [[ -z "$a" || -z "$b" || "$a" == "0" ]]; then
        echo "—"
    else
        awk -v a="$a" -v b="$b" 'BEGIN { printf "%.2f×", b/a }'
    fi
}

C_FRESH=$(get coda native fresh)
C_REFRESH=$(get coda native refresh)
C_INCR=$(get coda native incremental)
C_MM=$(get coda native make-unmake)
C_A2_REFRESH=$(get coda avx2 refresh)
C_A2_INCR=$(get coda avx2 incremental)
C_A2_MM=$(get coda avx2 make-unmake)

R_FRESH=$(get reckless native fresh)
R_INCR=$(get reckless native incremental)
R_MM=$(get reckless native make-unmake)
R_MM_NULL=$(get reckless native make-unmake-null)
R_A2_FRESH=$(get reckless avx2 fresh)
R_A2_INCR=$(get reckless avx2 incremental)
R_A2_MM=$(get reckless avx2 make-unmake)

# Construct the new section.
BLOCK_FILE=$(mktemp)
{
    echo ""
    echo "## $HOST — $DATE"
    echo ""
    echo "**CPU**: $CPU"
    echo ""
    echo "| Mode | Coda native | Reckless native | Ratio (R/C) |"
    echo "|---|---:|---:|---:|"
    printf "| fresh (scalar) | %s | — | — |\n" "$(fmt_int "$C_FRESH")"
    printf "| refresh / fresh (SIMD) | %s | %s | %s |\n" \
        "$(fmt_int "$C_REFRESH")" "$(fmt_int "$R_FRESH")" "$(ratio "$C_REFRESH" "$R_FRESH")"
    printf "| incremental | %s | %s | **%s** |\n" \
        "$(fmt_int "$C_INCR")" "$(fmt_int "$R_INCR")" "$(ratio "$C_INCR" "$R_INCR")"
    printf "| make-unmake (observer on) | %s | %s | %s |\n" \
        "$(fmt_int "$C_MM")" "$(fmt_int "$R_MM")" "$(ratio "$C_MM" "$R_MM")"
    printf "| make-unmake-null (no observer) | — | %s | — |\n" \
        "$(fmt_int "$R_MM_NULL")"
    if [[ -n "$C_A2_INCR" || -n "$R_A2_INCR" ]]; then
        echo ""
        echo "**AVX-2-forced** (runtime `--no-avx512` on Coda, separate build on Reckless):"
        echo ""
        echo "| Mode | Coda AVX-2 | Reckless AVX-2 | Ratio (R/C) |"
        echo "|---|---:|---:|---:|"
        printf "| refresh / fresh | %s | %s | %s |\n" \
            "$(fmt_int "$C_A2_REFRESH")" "$(fmt_int "$R_A2_FRESH")" "$(ratio "$C_A2_REFRESH" "$R_A2_FRESH")"
        printf "| incremental | %s | %s | **%s** |\n" \
            "$(fmt_int "$C_A2_INCR")" "$(fmt_int "$R_A2_INCR")" "$(ratio "$C_A2_INCR" "$R_A2_INCR")"
        printf "| make-unmake | %s | %s | %s |\n" \
            "$(fmt_int "$C_A2_MM")" "$(fmt_int "$R_A2_MM")" "$(ratio "$C_A2_MM" "$R_A2_MM")"
    fi
    echo ""
    echo "**Raw CSV**: \`$(basename "$CSV")\`"
    echo ""
    echo "---"
} > "$BLOCK_FILE"

# Append before the final sentinel comment line.
SENTINEL="<!-- Append new host blocks above this line."
TMPDOC=$(mktemp)
awk -v block_file="$BLOCK_FILE" -v sentinel="$SENTINEL" '
    index($0, sentinel) == 1 {
        while ((getline line < block_file) > 0) print line
        close(block_file)
    }
    { print }
' "$DOC" > "$TMPDOC"

mv "$TMPDOC" "$DOC"
rm "$BLOCK_FILE"

echo "Appended host block for '$HOST' to $DOC"
echo ""
echo "Preview:"
tail -40 "$DOC"
