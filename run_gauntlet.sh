#!/bin/bash
# Usage: ./run_gauntlet.sh <binary> <label>
BINARY=$1
LABEL=$2
NET=/home/adam/code/gochess/net-v5-120sb-sb120.nnue
OPENINGS=/home/adam/code/gochess/testdata/noob_3moves.epd

cutechess-cli \
  -tournament gauntlet \
  -engine name="Coda-${LABEL}" cmd="${BINARY}" proto=uci option.NNUEFile="${NET}" option.OwnBook=false option.Hash=64 \
  -engine name=Crafty cmd=./crafty dir=/home/adam/chess/engines/crafty proto=xboard \
  -engine name=Monolith cmd=/home/adam/chess/engines/Monolith/Source/Monolith proto=uci \
  -engine name=Rodent3 cmd=rodentIII proto=uci \
  -each tc=0/10+0.1 option.MoveOverhead=100 \
  -rounds 50 -concurrency 16 \
  -openings file="${OPENINGS}" format=epd order=random \
  -pgnout "gauntlet_${LABEL}.pgn" -recover -ratinginterval 10 \
  -draw movenumber=20 movecount=10 score=10 \
  -resign movecount=3 score=500 twosided=true 2>&1
