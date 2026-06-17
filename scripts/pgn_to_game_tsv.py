#!/usr/bin/env python3
"""
pgn_to_game_tsv.py — convert Coda-vs-SF datagen PGNs to a per-position stream
for `coda import-game-tsv`. Emits EVERY position (no filtering — the trainer's
loader filters; keeping all positions in game order preserves the binpack's
chain compression) as:  fen <TAB> uci_move <TAB> score_cp <TAB> result <TAB> ply
  score_cp = the mover's in-game eval (STM-POV centipawns; mate capped ±30000)
  result   = game WDL from the side-to-move POV (+1 win / 0 draw / -1 loss)
  ply      = absolute ply (so the loader's ply>=16 filter works on opening depth)
Reads PGN from a file arg or stdin; writes the TSV stream to stdout.
"""
import sys, re, chess, chess.pgn

EVAL_RE = re.compile(r'([+-]?)(M)?(\d+(?:\.\d+)?)/')  # eval before /depth in the comment

def parse_eval(comment):
    m = EVAL_RE.search(comment or "")
    if not m: return None
    sign = -1 if m.group(1) == '-' else 1
    if m.group(2):  # mate score -> cap
        return sign * 30000
    return int(round(sign * float(m.group(3)) * 100))

def main():
    fin = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    out = sys.stdout
    games = pos = 0
    res_map = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
    while True:
        g = chess.pgn.read_game(fin)
        if g is None: break
        white_res = res_map.get(g.headers.get("Result", "*"))
        if white_res is None: continue          # unfinished/unknown result
        games += 1
        board = g.board()
        for node in g.mainline():
            cp = parse_eval(node.comment)
            if cp is not None:
                stm_white = board.turn == chess.WHITE
                result = white_res if stm_white else -white_res
                out.write(f"{board.fen()}\t{board.uci(node.move)}\t{cp}\t{result}\t{board.ply()}\n")
                pos += 1
            board.push(node.move)
        if games % 5000 == 0:
            print(f"  ...{games} games, {pos} positions", file=sys.stderr, flush=True)
    print(f"done: {games} games -> {pos} positions", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
