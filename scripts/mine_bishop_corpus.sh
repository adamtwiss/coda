#!/usr/bin/env bash
# Orchestrate the wandering-bishop corpus mine end to end:
#   parallel cheap pre-filter -> concat -> parallel deep-SF validate (net_pref
#   gated) -> merged overrate-format EPD. See bishop_sortie_{prefilter,validate}.py.
set -u
cd "$(dirname "$0")/.."
CODA=./coda
SF=${SF:-/tmp/sf_ob/src/stockfish}
NET=${NET:-net-035195DB.nnue}
PRE_FILES_2061=${PRE_FILES_2061:-150}
PRE_FILES_2062=${PRE_FILES_2062:-375}
PRE_FILES_2094=${PRE_FILES_2094:-375}
PRE_STREAMS=${PRE_STREAMS:-12}
SF_STREAMS=${SF_STREAMS:-6}
SF_THREADS=${SF_THREADS:-2}
SF_DEPTH=${SF_DEPTH:-22}
GAP=${GAP:-100}
NP_MIN=${NP_MIN:-0}
OUT=${OUT:-/tmp/bishop_mine}
rm -rf "$OUT"; mkdir -p "$OUT/pre" "$OUT/val"
ts(){ date '+%F %T'; }

echo "[$(ts)] STAGE 1: build file list + parallel pre-filter"
{ ls /tmp/dg2061x/*.pgn.bz2 2>/dev/null | head -$PRE_FILES_2061
  ls /tmp/dg2062x/*.pgn.bz2 2>/dev/null | head -$PRE_FILES_2062
  ls /tmp/dg2094x/*.pgn.bz2 2>/dev/null | head -$PRE_FILES_2094; } > "$OUT/files.txt"
nfiles=$(wc -l < "$OUT/files.txt")
echo "[$(ts)]   $nfiles PGN shards -> $PRE_STREAMS prefilter streams"
# round-robin split
awk -v n=$PRE_STREAMS -v o="$OUT/pre" 'NR>0{print > (o"/files_"((NR-1)%n)".txt")}' "$OUT/files.txt"
pids=()
for k in $(seq 0 $((PRE_STREAMS-1))); do
  [ -f "$OUT/pre/files_$k.txt" ] || continue
  ( python3 scripts/bishop_sortie_prefilter.py --glob $(cat "$OUT/pre/files_$k.txt") \
      --coda-name Coda --no-cap --opt-min 0.30 --opt-max 1.50 --deep-rank 4 \
      --min-disagree 0.80 --out "$OUT/pre/cand_$k.tsv" 2>"$OUT/pre/log_$k.txt" ) &
  pids+=($!)
done
wait "${pids[@]}"
# concat (one header)
head -1 "$OUT/pre/cand_0.tsv" > "$OUT/candidates.tsv"
for k in $(seq 0 $((PRE_STREAMS-1))); do
  [ -f "$OUT/pre/cand_$k.tsv" ] && tail -n +2 "$OUT/pre/cand_$k.tsv" >> "$OUT/candidates.tsv"
done
ncand=$(($(wc -l < "$OUT/candidates.tsv")-1))
echo "[$(ts)]   pre-filter done: $ncand candidates"

echo "[$(ts)] STAGE 2: shuffle + parallel deep-SF validate (net_pref gate, depth $SF_DEPTH, gap $GAP)"
hdr=$(head -1 "$OUT/candidates.tsv")
tail -n +2 "$OUT/candidates.tsv" | shuf > "$OUT/cand_shuf.tsv"
split -d -n r/$SF_STREAMS "$OUT/cand_shuf.tsv" "$OUT/val/shard_"
pids=()
for s in "$OUT"/val/shard_*; do
  ( { echo "$hdr"; cat "$s"; } > "$s.tsv"
    python3 scripts/bishop_sortie_validate.py --cand "$s.tsv" --sf "$SF" \
      --sf-depth $SF_DEPTH --sf-threads $SF_THREADS --gap-thresh $GAP \
      --coda "$CODA" --coda-net "$NET" --np-min $NP_MIN --cap-per-game 1 \
      --out-epd "$s.epd" --out-tsv "$s.out.tsv" >"$s.runlog" 2>&1 ) &
  pids+=($!)
done
wait "${pids[@]}"
# merge EPDs + renumber ids; dedup on fen4
: > "$OUT/corpus_raw.epd"
cat "$OUT"/val/shard_*.epd >> "$OUT/corpus_raw.epd" 2>/dev/null
awk '!seen[$1" "$2" "$3" "$4]++' "$OUT/corpus_raw.epd" \
  | awk 'BEGIN{i=0} {sub(/id "sortie-[0-9]+"/,"id \"sortie-"i"\""); i++; print}' \
  > "$OUT/corpus.epd"
nkept=$(grep -c . "$OUT/corpus.epd")
echo "[$(ts)] DONE: $nkept positions -> $OUT/corpus.epd"
echo "[$(ts)]   per-stream keep counts:"; for s in "$OUT"/val/shard_*.epd; do echo "    $(basename $s): $(grep -c . $s 2>/dev/null)"; done
