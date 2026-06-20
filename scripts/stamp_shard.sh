#!/usr/bin/env bash
# Stamp the datagen PGN corpus across N coda-stamp processes (CPU-bound, no SF —
# ~100x faster than the SF-rescore it replaces). Each shard -> its own binpack
# (the loader reads many). Coda-to-move positions get the filtered sentinel;
# SF-to-move keep their eval; chains preserved.
N=${N:-16}
OUT=${OUT:-/tmp/stamp_shards}
SRC=${SRC:-/tmp/dg2062x}
STAMP=/home/adam/code/coda/target/release/coda-stamp
mkdir -p "$OUT"
mapfile -t FILES < <(ls "$SRC"/*.pgn.bz2)
total=${#FILES[@]}
echo "$(date '+%F %T') stamping $total files across $N workers" | tee "$OUT/queue.log"
for ((k=0; k<N; k++)); do
  shard=()
  for ((i=k; i<total; i+=N)); do shard+=("${FILES[$i]}"); done
  (
    bzcat "${shard[@]}" 2>/dev/null \
      | "$STAMP" -i - -o "$OUT/shard_$k.binpack" --sentinel 32000 2>"$OUT/shard_$k.log"
    echo "$(date '+%F %T') shard $k DONE (exit $?)" >> "$OUT/queue.log"
  ) &
done
wait
echo "$(date '+%F %T') ALL STAMP SHARDS DONE" | tee -a "$OUT/queue.log"
