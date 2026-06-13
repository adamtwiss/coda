#!/usr/bin/env bash
# TM-ceiling measurement: how good is Coda's time management relative to the
# field? Run a Coda gauntlet vs near-strength top engines under two conditions:
#   CLOCK : tc=30+0.3      — TM active (engines allocate time per move)
#   FIXED : st=0.7         — every move gets exactly 0.7s, no allocation (~matched
#                            avg move time of the clock condition)
# For each opponent X, ΔCoda = score_Coda(CLOCK) − score_Coda(FIXED).
#   Δ > 0  → Coda's TM beats X's (Coda gains when both manage their clock)
#   Δ < 0  → X's TM is better → Coda has TM headroom = the deficit
# This isolates the ALLOCATION half of TM. It does NOT capture ponder-leech
# defence (non-ponder RR), which is a separate, deployment-only TM component.
set -u
cd "$(dirname "$0")/.."
CC=cutechess-cli
BOOK=/home/adam/chess/books/UHO_Lichess_4852_v1.epd
CODA="cmd=./coda proto=uci option.Threads=1 option.Hash=128 option.OwnBook=false"
OPP=(
  "Alexandria:cmd=/home/adam/chess/engines/Alexandria/Alexandria proto=uci option.Threads=1 option.Hash=128"
  "PlentyChess:cmd=/home/adam/chess/engines/PlentyChess/engine proto=uci option.Threads=1 option.Hash=128"
  "Berserk:cmd=/home/adam/chess/engines/berserk-13/src/berserk proto=uci option.Threads=1 option.Hash=128"
  "Obsidian:cmd=/home/adam/chess/engines/Obsidian/Obsidian proto=uci option.Threads=1 option.Hash=128"
)
ROUNDS=${1:-150}   # games/opponent/condition = ROUNDS*2 (both colours)
CONC=${2:-16}
OUT=/tmp/tm_ceiling_above
mkdir -p $OUT

run_match() {
  local cond="$1" name="$2" oeng="$3" tcargs="$4"
  echo "[$(date +%H:%M:%S)] $cond vs $name ..."
  $CC -recover -concurrency $CONC -rounds $ROUNDS -games 2 \
    -openings file=$BOOK format=epd order=random -repeat \
    -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=600 \
    $tcargs \
    -engine name=Coda dir=. $CODA \
    -engine name=$name $oeng \
    -pgnout $OUT/${cond}_${name}.pgn \
    2>&1 | tail -40 | grep -iE "Score of Coda|Elo difference" | tail -2 | sed "s/^/  [$cond vs $name] /"
}

# Interleave clock+fixed per opponent so a complete Δ lands early.
for entry in "${OPP[@]}"; do
  name="${entry%%:*}"; oeng="${entry#*:}"
  run_match clock "$name" "$oeng" "-each tc=30+0.3 timemargin=250"
  run_match fixed "$name" "$oeng" "-each st=0.7 timemargin=250"
  echo "--- $name complete: ΔCoda = clock_score − fixed_score (see PGNs) ---"
done
echo "=== DONE — parse with scripts/tm_ceiling_parse.py $OUT ==="
