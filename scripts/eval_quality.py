#!/usr/bin/env python3
"""Cross-engine static-eval quality benchmark.

Question: relative to top engines, how good is each engine's NNUE *static
eval* at predicting game outcomes? This isolates eval quality from search.

Method: for a shared set of quiet positions (FEN + actual game result, white
POV), get each engine's static eval via UCI `eval`, fit a per-engine logistic
`win% = sigmoid(a*eval_pawns + b)` (the per-engine fit absorbs eval-scale
differences, so we compare *predictive power*, not raw cp), and score it by
Brier (MSE of predicted win% vs result) and logloss. Lower = better eval.

Coda's eval is taken from the CSV column (computed in-process by
`coda eval-dist --csv`); other engines are driven via UCI on the same FENs.

Usage:
  coda eval-dist -i <binpack> -c 20000 --quiet-only --csv /tmp/evalq.csv -n <net>
  python3 scripts/eval_quality.py /tmp/evalq.csv
"""
import os
import re
import subprocess
import sys
import numpy as np

# (label, binary, eval-line regex, scale_to_cp, stm_pov). `stm_pov=True` means
# the engine reports its static eval from the SIDE-TO-MOVE perspective (positive
# = good for whoever is to move); we then negate when the FEN's side-to-move is
# black to convert to the white-POV the tool assumes. `stm_pov=False` means the
# engine already reports white-POV (no flip).
#
# POV/format were verified per engine with three probes: startpos (≈0),
# white-up-a-queen white-to-move (must be strongly +), and white-up-a-queen
# BLACK-to-move (strongly + for white-POV engines, strongly − for STM-POV
# engines). The four legacy engines below (SF, Reckless, Berserk, Obsidian)
# are white-POV; the engines added 2026-06-20 (PlentyChess, Caissa, Clover,
# Stormphrax, Halogen, Tarnished, Integral) are STM-POV and use the
# per-position sign flip. Viridithas has no usable static eval (see SKIPPED).
#
# Units: SF/Reckless/Stormphrax print pawns (×100 → cp); the rest print integer
# cp/units. The per-engine logistic/correlation fit absorbs scale, so only
# predictive power (Spearman/Pearson vs oracle) is compared.
ENGINES = [
    ("Stockfish",  "/home/adam/chess/engines/Stockfish/src/stockfish",
     r"NNUE evaluation\s+([+-]?[0-9]+\.[0-9]+)", 100.0, False),
    ("Reckless",   "/home/adam/chess/engines/Reckless/reckless",
     r"NNUE evaluation\s+([+-]?[0-9]+\.[0-9]+)", 100.0, False),
    ("Berserk",    "/home/adam/chess/engines/berserk-13/src/berserk",
     r"NNUE Score:\s*([+-]?[0-9]+)\s*cp", 1.0, False),
    ("Obsidian",   "/home/adam/chess/engines/Obsidian/Obsidian",
     r"Evaluation:\s*([+-]?[0-9]+)", 1.0, False),
    # --- added 2026-06-20 (all STM-POV; verified white-up-a-queen probes) ---
    # PlentyChess/Caissa/Clover print a bare signed integer cp on its own line;
    # the full-line-integer anchor avoids matching option/info lines (verified
    # exactly one match per `eval` in batch).
    ("PlentyChess", "/home/adam/chess/engines/PlentyChess/engine",
     r"^\s*([+-]?[0-9]+)\s*$", 1.0, True),
    ("Caissa",      "/home/adam/chess/engines/Caissa/src/caissa",
     r"^\s*([+-]?[0-9]+)\s*$", 1.0, True),
    ("Clover",      "/home/adam/chess/engines/CloverEngine/src/Clover.9.1",
     r"^\s*([+-]?[0-9]+)\s*$", 1.0, True),
    ("Stormphrax",  "/home/adam/chess/engines/Stormphrax/stormphrax-7.0.70-native",
     r"Static eval:\s*([+-]?[0-9]+(?:\.[0-9]+)?)", 100.0, True),
    ("Halogen",     "/home/adam/chess/engines/Halogen/bin/Halogen-pgo",
     r"Eval:\s*([+-]?[0-9]+)cp", 1.0, True),
    ("Tarnished",   "/home/adam/chess/engines/Tarnished/tarnished",
     r"Raw:\s*([+-]?[0-9]+)", 1.0, True),
    # --- added 2026-06-20 (lowercase dirs missed earlier) ---
    # Integral: UCI `eval` prints two lines per call — `info cp <N>` (raw) and
    # `info normalized cp <N>` (WDL-normalised). We take the RAW cp line; the
    # regex `info cp` does NOT match `info normalized cp`, so exactly one match
    # per eval (verified). STM-POV: white-up-a-queen black-to-move probed −566
    # (vs +687 white-to-move, +22 startpos) → side-to-move POV, flip=True. cp.
    ("Integral",    "/home/adam/chess/engines/integral/integral",
     r"info cp\s+([+-]?[0-9]+)", 1.0, True),
    # Viridithas — driven by the bespoke drive_viridithas() (NOT this table);
    #   added 2026-06-20. See VIRIDITHAS_BIN below and that function's docstring.
    #   It has no static-eval surface (`eval`/`print`/`d`/`go depth N` all silent
    #   in this dev build) but `go infinite`+stop at depth 1 IS a position-
    #   tracking, POV-verified near-static proxy. Marked "(d1*)" in output to
    #   flag it's a 1-ply search score, not pure static eval.
    # SKIPPED:
    #  Quanticade — no static `eval` command (eval/evaluate/static all silent;
    #    only `go depthN` yields a *search* score, not a static eval).
    #  Alexandria — binary NON-FUNCTIONAL in this 9.0.3 build (re-audited
    #    2026-06-20 with CORRECTED white-up-a-queen FENs — the earlier probe
    #    FENs had removed the WRONG queen). `position fen` IS parsed (the `d`
    #    board dump is correct), but BOTH eval surfaces are broken:
    #      - `eval` (`Raw eval:`): position-DEPENDENT but miscalibrated — startpos
    #        528 (≈5 pawns for a symmetric pos!), white-up-a-queen 354 (LESS than
    #        startpos, violates the must-exceed invariant), KvK 18. NNUE accum
    #        appears corrupt; unfixed by isready/ucinewgame sync.
    #      - `go depth N`/`go nodes N`/`go movetime N` all return immediately with
    #        `nodes 0 pv a8a8 score cp 0` (null move — search doesn't run).
    #      - only `go infinite`+stop runs a real search, and even THEN reports
    #        white-up-a-queen at score cp 5 @ depth 12 (vs startpos cp 0) — the
    #        eval is genuinely broken, not a POV/sign artifact. Unrecoverable.
    #  Hobbes — NO third-party Hobbes engine binary exists on this host
    #    (searched /home/adam/chess/engines and all of /home/adam). "Hobbes" in
    #    this repo refers only to a Coda *training recipe* (WDL-ramp schedule,
    #    nets/prod-hobbes-*.nnue, docs/engine-notes), not a runnable engine. The
    #    path the request named does not exist; nothing to drive.
]

# Viridithas binary, driven by the bespoke drive_viridithas() (depth-1 proxy).
VIRIDITHAS_BIN = "/home/adam/chess/engines/viridithas/Viridithas"


def drive_engine(binary, eval_re, scale, fens, stm_pov=False):
    """Return list of static evals (white-POV cp) for each FEN, or None on miss.

    Writes all UCI commands to a temp file and runs `engine < cmds > out` so
    there's no stdin/stdout pipe deadlock and no per-position sync. Each `eval`
    emits exactly one matching line; we map them to FENs by strict order.
    `eval_re` is the engine's eval-line regex (one capture group); `scale`
    converts its units to cp (×100 for pawn-printing engines, ×1 for cp).

    If `stm_pov` is True the engine reports from the side-to-move perspective,
    so we negate the eval for every FEN whose side-to-move (2nd FEN field) is
    black — converting to the white-POV the tool assumes. The negation is keyed
    to the FEN, mapped by strict eval order, so mixed-STM CSVs are handled.
    """
    import tempfile, os
    cmd_path = tempfile.mktemp(suffix=".uci")
    out_path = tempfile.mktemp(suffix=".out")
    with open(cmd_path, "w") as f:
        f.write("uci\n")
        for fen in fens:
            f.write(f"position fen {fen}\neval\n")
        f.write("quit\n")
    with open(cmd_path) as fin, open(out_path, "w") as fout:
        subprocess.run([binary], stdin=fin, stdout=fout,
                       stderr=subprocess.DEVNULL, timeout=900)
    # Per-FEN white-POV sign: +1 for white-to-move, -1 for black-to-move (only
    # applied when the engine is side-to-move POV). FEN STM is field index 1.
    signs = [(-1.0 if (stm_pov and fen.split()[1] == 'b') else 1.0)
             for fen in fens]
    evals = []
    with open(out_path, "rb") as f:  # rb: some engines emit NUL in board dumps
        for raw in f:
            line = raw.decode("utf-8", "replace")
            m = eval_re.search(line)
            if m:
                i = len(evals)
                sign = signs[i] if i < len(signs) else 1.0
                evals.append(float(m.group(1)) * scale * sign)
    os.unlink(cmd_path); os.unlink(out_path)
    # Strict ordering: one eval per position. If counts mismatch, return what
    # we have padded/truncated so the caller can warn.
    if len(evals) < len(fens):
        evals += [None] * (len(fens) - len(evals))
    return evals[:len(fens)]


def drive_viridithas(binary, fens):
    """Special driver for Viridithas 20.0.0-dev (white-POV cp list, depth-1 proxy).

    Viridithas has NO usable interactive static-eval surface: UCI `eval`,
    `print`, `d`, `static` are all silent (uciok only), and `go depth N` /
    `go nodes N` / `go movetime N` print NOTHING in this dev build (only
    `go infinite` emits info lines, interruptible by `stop`). So we use a
    *depth-1 search score* as a near-static proxy: per position, send
    `go infinite`, poll stdout until the first `info depth N ... score cp`
    line appears (depth 1, ~20 nodes — essentially the raw eval after a
    1-ply pass), then `stop`. This is NOT pure static eval — it is a 1-ply
    search score — but it is position-tracking and POV-verified.

    POV: STM-POV (verified white-up-a-queen probes: startpos +41,
    w-to-move +678, b-to-move −740) — flip sign when side-to-move is black
    to convert to white-POV. Mates map to ±30000 cp.

    A batched file driver can't do this (an immediately-following `stop`
    arrives before depth 1 completes → no output), and one process per FEN
    is wasteful, so this uses a single persistent process with non-blocking
    reads. ~120s for 20k positions, 0 misses measured 2026-06-20.
    """
    import os, fcntl, time
    p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    fd = p.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    def drain():
        out = ""
        try:
            while True:
                c = os.read(fd, 65536)
                if not c:
                    break
                out += c.decode("utf-8", "replace")
        except (BlockingIOError, TypeError):
            pass
        return out

    p.stdin.write("uci\nisready\n"); p.stdin.flush(); time.sleep(0.15); drain()
    cp_re = re.compile(r"info depth \d+ .*?score cp (-?\d+)")
    mate_re = re.compile(r"info depth \d+ .*?score mate (-?\d+)")
    any_re = re.compile(r"info depth \d+ .*score (cp|mate)")
    evals = []
    for fen in fens:
        p.stdin.write(f"position fen {fen}\ngo infinite\n"); p.stdin.flush()
        buf = ""
        deadline = time.time() + 0.6
        while time.time() < deadline:
            buf += drain()
            if any_re.search(buf):
                break
            time.sleep(0.002)
        p.stdin.write("stop\n"); p.stdin.flush(); time.sleep(0.003); buf += drain()
        m = cp_re.search(buf)
        mm = mate_re.search(buf)
        if m:
            cp = float(m.group(1))
        elif mm:
            cp = 30000.0 if int(mm.group(1)) > 0 else -30000.0
        else:
            evals.append(None)
            continue
        if fen.split()[1] == "b":
            cp = -cp  # STM-POV → white-POV
        evals.append(cp)
    p.stdin.write("quit\n"); p.stdin.flush()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()
    return evals


def spearman(x, y):
    """Rank correlation, scipy-free."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def agreement(evals_cp, oracle_cp):
    """How well an engine's static eval matches the LC0 deep-eval oracle.
    Returns (spearman, pearson, calibrated_rms_cp). The per-engine linear
    calibration absorbs scale, so RMS is the residual after best-fit scaling."""
    e = np.asarray(evals_cp, dtype=np.float64)
    o = np.asarray(oracle_cp, dtype=np.float64)
    sp = spearman(e, o)
    pe = float(np.corrcoef(e, o)[0, 1])
    # best linear fit e->o, residual RMS in oracle cp units
    A = np.vstack([e, np.ones_like(e)]).T
    coef, *_ = np.linalg.lstsq(A, o, rcond=None)
    resid = o - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return sp, pe, rms


# T80 score scale: ~100cp ≈ 75% WP (memory reference_t80_lc0_scoring_calibration)
# → sigmoid(100/K)=0.75 → K = 100/ln(3) ≈ 91.
WP_K = 91.0


def wp(cp):
    return 1.0 / (1.0 + np.exp(-np.asarray(cp, float) / WP_K))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/evalq.csv"
    fens, result, oracle, coda_eval = [], [], [], []
    with open(csv_path) as f:
        next(f)  # header: fen,white_result,coda_eval_white_cp,lc0_score_white_cp
        for line in f:
            parts = line.rstrip("\n").rsplit(",", 3)
            if len(parts) != 4:
                continue
            fens.append(parts[0])
            result.append(float(parts[1]))   # actual game result (white)
            coda_eval.append(float(parts[2]))
            oracle.append(float(parts[3]))    # LC0 deep-eval (white cp)
    result = np.asarray(result); oracle = np.asarray(oracle)
    print(f"Loaded {len(fens)} positions from {csv_path}\n")

    # Targets to score against. The net is trained on a WDL blend of search
    # score (LC0) and game result; that blended target is the most faithful.
    # Pure result is ~0-corr on T80 (too drawish); pure LC0 is the search-only
    # target; blends mix in real-outcome signal (Coda λ=0.20, SF λ=0.24).
    targets = {
        "LC0-only (λ=0)":  wp(oracle),
        "blend λ=0.20":    0.80 * wp(oracle) + 0.20 * result,
        "blend λ=0.24":    0.76 * wp(oracle) + 0.24 * result,
    }

    # Drive each engine once; cache its evals; score against every target.
    engine_evals = {"Coda (549C20A5)": np.asarray(coda_eval)}
    for label, binary, regex, scale, stm_pov in ENGINES:
        print(f"Driving {label} over {len(fens)} positions"
              f"{' (STM-POV→flip)' if stm_pov else ''}...", flush=True)
        try:
            evs = drive_engine(binary, re.compile(regex, re.IGNORECASE), scale,
                               fens, stm_pov=stm_pov)
        except Exception as e:
            print(f"  {label} failed: {e}")
            continue
        arr = np.array([np.nan if e is None else e for e in evs], float)
        if np.isnan(arr).mean() > 0.1:
            print(f"  {label}: {np.isnan(arr).mean():.0%} evals missing — check format")
        engine_evals[label] = arr

    # Viridithas uses a bespoke driver (depth-1 go-infinite proxy — see
    # drive_viridithas / the SKIPPED note). Drive it separately.
    if VIRIDITHAS_BIN and os.path.exists(VIRIDITHAS_BIN):
        print(f"Driving Viridithas over {len(fens)} positions "
              f"(depth-1 go-infinite proxy, STM-POV→flip)...", flush=True)
        try:
            evs = drive_viridithas(VIRIDITHAS_BIN, fens)
            arr = np.array([np.nan if e is None else e for e in evs], float)
            if np.isnan(arr).mean() > 0.1:
                print(f"  Viridithas: {np.isnan(arr).mean():.0%} evals missing")
            engine_evals["Viridithas (d1*)"] = arr
        except Exception as e:
            print(f"  Viridithas failed: {e}")

    for tname, tvals in targets.items():
        print(f"\n=== Static-eval agreement with target: {tname} (Spearman, higher=better) ===")
        print(f"{'Engine':<22} {'Spearman':>9} {'Pearson':>8} {'N':>7}")
        rows = []
        for label, ev in engine_evals.items():
            mask = ~np.isnan(ev)
            sp = spearman(ev[mask], tvals[mask])
            pe = float(np.corrcoef(ev[mask], tvals[mask])[0, 1])
            rows.append((label, sp, pe, int(mask.sum())))
        rows.sort(key=lambda x: -x[1])
        best = rows[0][1]
        for label, sp, pe, n in rows:
            gap = best - sp
            print(f"{label:<22} {sp:>9.4f} {pe:>8.4f} {n:>7}  {'(best)' if gap==0 else f'-{gap:.4f}'}")
    print("\nNote: Coda and SF train on the SAME T80/LC0 data — no home-field "
          "bias between them, so the Coda-vs-SF gap is a genuine eval-quality "
          "difference. (Reckless's training data may differ.)")


if __name__ == "__main__":
    main()
