#!/usr/bin/env bash
# Conversion-failure gauntlet: Coda (v8s3) vs top-20 defenders, STC.
# Aimed at bullet #4 — "correct eval, failed conversion" (NOT overscore;
# Atlas's T80 mining covers overscore). RESIGN ADJUDICATION REMOVED so
# winning positions play out and conversion failures are observable.
# Draw adjudication KEPT (only cleans up genuinely-dead-equal games).
# Zeus 8C/16T -> concurrency 16. Prereq: OB worker stopped.
set -u
cd /home/adam/code/coda
E=/home/adam/chess/engines
NET=/home/adam/code/coda/multi-v8-l132-s3-v3-swa.nnue
OUT=/home/adam/code/coda/conversion_gauntlet.pgn

cutechess-cli \
  -tournament gauntlet \
  -engine name=Coda        cmd=/home/adam/code/coda/coda.rr proto=uci option.NNUEFile=$NET \
  -engine name=Reckless    cmd=$E/Reckless/reckless proto=uci \
  -engine name=Viridithas  cmd=$E/viridithas/Viridithas proto=uci \
  -engine name=Berserk     dir=$E/berserk/src cmd=./berserk proto=uci \
  -engine name=Obsidian    cmd=$E/Obsidian/Obsidian proto=uci \
  -engine name=Integral    cmd=$E/integral/integral proto=uci \
  -engine name=Stockfish   cmd=$E/Stockfish/src/stockfish proto=uci \
  -engine name=PlentyChess cmd=$E/PlentyChess/engine proto=uci \
  -engine name=Alexandria  cmd=$E/Alexandria/Alexandria proto=uci \
  -engine name=Hobbes      cmd=$E/hobbes-chess-engine/hobbes-chess-engine proto=uci \
  -engine name=Tarnished   cmd=$E/Tarnished/tarnished proto=uci \
  -engine name=Halogen     cmd=$E/Halogen/bin/Halogen-pgo proto=uci \
  -engine name=Astra       cmd=$E/Astra/astra proto=uci \
  -engine name=Starzix     cmd=$E/Starzix/starzix proto=uci \
  -engine name=Quanticade  cmd=$E/Quanticade/Quanticade proto=uci \
  -engine name=Motor       cmd=$E/motor/motor proto=uci \
  -engine name=Stormphrax  cmd=$E/Stormphrax/stormphrax-7.0.131-native proto=uci \
  -engine name=Clover      cmd=$E/CloverEngine/src/Clover.9.1 proto=uci \
  -engine name=Caissa      cmd=$E/Caissa/src/caissa proto=uci \
  -engine name=Cinder      cmd=$E/cinder/target/bin/cinder-v0.4.1-linux-native proto=uci \
  -each tc=0/10+0.1 option.Hash=256 \
  -rounds 200 -concurrency 16 \
  -openings file=/home/adam/chess/books/noob_4moves.epd format=epd order=random \
  -pgnout $OUT -recover -ratinginterval 20 \
  -draw movenumber=20 movecount=10 score=10
