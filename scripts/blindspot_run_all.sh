#!/bin/bash
# Blindspot pipeline over the remaining 5 T80 monthly files (Feb-Jun 2024).
# Per file: 32-way Coda harvest (150/80 prefilter) -> 32-way SF static eval ->
# calibrate+filter (coda-worse-by-80) -> import to binpack. Intermediates are
# deleted after each file so peak disk stays ~one-file's worth.
# Continue-on-error: a failed month is logged and skipped, the rest proceed.

NET=/home/adam/code/coda/net-035195DB.nnue
BIN=/home/adam/code/coda/target/release/coda
SF=/home/adam/chess/engines/sf-evalfile/stockfish
W=/tmp/blindspot
OUT=/training/blindspot
SRC=/training/sf
N=32
MINERR=150
MAXLC0=600

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
free_gb(){ df -BG / | awk 'NR==2{gsub("G","",$4); print $4}'; }

declare -A FILE=(
  [feb]="$SRC/test80-2024-02-feb-2tb7p.min-v2.v6.binpack"
  [mar]="$SRC/test80-2024-03-mar-2tb7p.min-v2.v6.binpack"
  [apr]="$SRC/test80-2024-04-apr-2tb7p.min-v2.v6.binpack"
  [may]="$SRC/test80-2024-05-may-2tb7p.min-v2.v6.binpack"
  [jun]="$SRC/test80-2024-06-jun-2tb7p.min-v2.v6.binpack"
)
ORDER="feb mar apr may jun"

mkdir -p "$OUT"
echo "[$(ts)] RUN_ALL start. order=$ORDER  free=$(free_gb)G"

for M in $ORDER; do
  F="${FILE[$M]}"
  echo "==================================================================="
  echo "[$(ts)] [$M] START  file=$F  free=$(free_gb)G"
  if [ ! -f "$F" ]; then echo "[$(ts)] [$M] MISSING FILE - skip"; continue; fi
  if [ -f "$OUT/t80_blindspot_150_80_${M}.binpack" ]; then
    echo "[$(ts)] [$M] output already exists - skip"; continue
  fi

  # --- stage 1: 32-way Coda harvest ---
  rm -f "$W"/shard_*.csv
  for k in $(seq 0 $((N-1))); do
    "$BIN" eval-dist --input "$F" -n "$NET" --shard "$k/$N" \
      --quiet-only --min-error $MINERR --max-abs-lc0 $MAXLC0 \
      --csv "$W/shard_$k.csv" > "$W/s1_$k.log" 2>&1 &
  done
  wait
  S1=$(cat "$W"/shard_*.csv 2>/dev/null | grep -vc '^fen,')
  echo "[$(ts)] [$M] stage1 done: $S1 candidates  free=$(free_gb)G"
  if [ "$S1" -eq 0 ]; then echo "[$(ts)] [$M] 0 stage1 rows - skip"; rm -f "$W"/shard_*.csv; continue; fi

  # --- stage 2: SF static eval (32-way) ---
  for k in $(seq 0 $((N-1))); do
    awk -F',' 'NR>1{print (NR-2)"\t"$1}' "$W/shard_$k.csv" > "$W/sf_in_$k.txt" &
  done; wait
  for k in $(seq 0 $((N-1))); do
    "$SF" evalfile "$W/sf_in_$k.txt" "$W/sf_out_$k.txt" > "$W/sf_$k.log" 2>&1 &
  done; wait

  # --- calibrate + coda-worse-by-80 filter (32-way) ---
  for k in $(seq 0 $((N-1))); do
    python3 "$W/filter_shard.py" "$k" > "$W/f_$k.log" 2>&1 &
  done; wait
  cat "$W"/cands_*.tsv > "$W/coda_worse_${M}.tsv"
  CAND=$(wc -l < "$W/coda_worse_${M}.tsv")
  echo "[$(ts)] [$M] stage2 done: $CAND coda-worse candidates  free=$(free_gb)G"

  # --- import to binpack ---
  { printf 'fen\tscore\n'; cat "$W/coda_worse_${M}.tsv"; } > "$W/import_${M}.tsv"
  "$BIN" import-tsv -i "$W/import_${M}.tsv" --fen-col 0 --score-col 1 \
    -o "$OUT/t80_blindspot_150_80_${M}.binpack" > "$W/import_${M}.log" 2>&1
  echo "[$(ts)] [$M] import: $(grep -o '[0-9]* entries written' "$W/import_${M}.log" | head -1)"

  # --- keep candidate tsv (gz), delete heavy scratch ---
  gzip -c "$W/coda_worse_${M}.tsv" > "$OUT/cands_${M}.tsv.gz"
  rm -f "$W"/shard_*.csv "$W"/sf_in_*.txt "$W"/sf_out_*.txt "$W"/cands_*.tsv \
        "$W/coda_worse_${M}.tsv" "$W/import_${M}.tsv"
  echo "[$(ts)] [$M] DONE  binpack=$(ls -lh "$OUT/t80_blindspot_150_80_${M}.binpack" | awk '{print $5}')  free=$(free_gb)G"
done

echo "[$(ts)] ALL_MONTHS_DONE"
ls -lh "$OUT"
