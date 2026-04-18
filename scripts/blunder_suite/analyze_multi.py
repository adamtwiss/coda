#!/usr/bin/env python3
"""Single-pass SF analyzer that handles multiple 'our' names (coda_bot + codabot).
Faster than running the existing analyzer twice.
Depth 16 for speed (matches earlier analysis that gave us the 45-blunder baseline).
"""
import argparse, csv, subprocess, sys, time, os
from pathlib import Path
import chess, chess.pgn, chess.engine

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pgn", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sf", default="/usr/games/stockfish")
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--hash", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--our", nargs="+", default=["coda_bot", "codabot"])
    p.add_argument("--blunder-cp", type=int, default=100)
    p.add_argument("--skip-opening", type=int, default=10)
    return p.parse_args()

def score_cp(info, board_stm):
    sc = info["score"].pov(board_stm)
    if sc.is_mate():
        m = sc.mate()
        return 30000 if m > 0 else -30000
    return sc.score()

def main():
    args = parse_args()
    sf = chess.engine.SimpleEngine.popen_uci(args.sf)
    sf.configure({"Hash": args.hash, "Threads": args.threads})

    rows = []
    game_id = 0
    t0 = time.time()

    with open(args.pgn) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None: break
            game_id += 1
            w = game.headers.get("White", "?")
            b = game.headers.get("Black", "?")
            our_name = None
            if w in args.our: our_name = w; our_color = chess.WHITE
            elif b in args.our: our_name = b; our_color = chess.BLACK
            if not our_name: continue
            opp = b if our_color == chess.WHITE else w
            result = game.headers.get("Result", "*")
            site = game.headers.get("Site", "")

            board = game.board()
            ply = 0
            for node in game.mainline():
                mv = node.move
                if board.turn == our_color and ply >= args.skip_opening:
                    info = sf.analyse(board, chess.engine.Limit(depth=args.depth))
                    sf_best = info.get("pv", [None])[0]
                    sf_cp = score_cp(info, our_color)
                    if sf_best is None or mv == sf_best:
                        pass
                    else:
                        board.push(mv)
                        info2 = sf.analyse(board, chess.engine.Limit(depth=args.depth))
                        our_cp = score_cp(info2, our_color)
                        board.pop()
                        cp_loss = sf_cp - our_cp
                        rows.append({
                            "game_id": game_id,
                            "site": site,
                            "our_account": our_name,
                            "opp": opp,
                            "ply": ply,
                            "our_color": "w" if our_color == chess.WHITE else "b",
                            "played": mv.uci(),
                            "sf_best": sf_best.uci(),
                            "sf_cp": sf_cp,
                            "our_cp": our_cp,
                            "cp_loss": cp_loss,
                            "fen": board.fen(),
                            "result": result,
                        })
                board.push(mv)
                ply += 1

            if game_id % 10 == 0:
                dt = time.time() - t0
                blunders = sum(1 for r in rows if r["cp_loss"] >= args.blunder_cp)
                print(f"[{dt:6.0f}s] game {game_id:3d}  rows={len(rows):4d}  blunders={blunders}",
                      flush=True)

    sf.quit()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    blunders = [r for r in rows if r["cp_loss"] >= args.blunder_cp]
    print(f"\nDone: {game_id} games, {len(rows)} moves analysed, {len(blunders)} blunders")

if __name__ == "__main__":
    main()
