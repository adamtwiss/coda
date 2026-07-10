#!/usr/bin/env python3
"""C2 repro: tm_cross_prev_score pollution via ponder-miss and analysis go's.

Drives the coda binary over UCI with real timing. Requires the C2_* debug
scaffolding (TMDebug-gated eprintln) built into target/release/coda.
"""
import subprocess, sys, threading, time, queue

BIN = "/tmp/claude-1001/-home-adam-code-coda/e50bbdde-8772-4ce8-8e12-6c709d0aa781/scratchpad/wt-c2/target/release/coda"
NET = "/home/adam/code/coda/net-E161C665.nnue"

class Engine:
    def __init__(self):
        self.p = subprocess.Popen([BIN, "--nnue", NET],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.out = queue.Queue()
        self.err_lines = []
        threading.Thread(target=self._pump, args=(self.p.stdout, self.out, "OUT"), daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    def _pump(self, stream, q, tag):
        for line in stream:
            q.put(line.rstrip())

    def _pump_err(self):
        for line in self.p.stderr:
            line = line.rstrip()
            self.err_lines.append(line)
            print(f"  [stderr] {line}", flush=True)

    def send(self, cmd):
        print(f">> {cmd}", flush=True)
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait_for(self, prefix, timeout=30):
        """Drain stdout until a line starting with prefix; return it."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.out.get(timeout=0.1)
            except queue.Empty:
                continue
            if line.startswith("info depth") and " pv " in line:
                # keep last score visible
                print(f"<< {line}", flush=True)
            elif line.startswith(prefix):
                print(f"<< {line}", flush=True)
                return line
            else:
                print(f"<< {line}", flush=True)
        raise TimeoutError(f"timeout waiting for {prefix}")

def main():
    e = Engine()
    e.send("uci"); e.wait_for("uciok")
    e.send("setoption name TMDebug value true")
    e.send("setoption name OwnBook value false")
    e.send("isready"); e.wait_for("readyok")

    print("\n=== SCENARIO A: ponder-miss pollution ===")
    e.send("ucinewgame")
    e.send("isready"); e.wait_for("readyok")
    # Real move 1 for white (move 3): position after 1.e4 e5, white to move.
    e.send("position startpos moves e2e4 e7e5")
    e.send("go wtime 60000 btime 60000 winc 1000 binc 1000")
    bm = e.wait_for("bestmove")
    b1 = bm.split()[1]
    print(f"--- engine played {b1} (S1 = score above, published as C2_PUBLISH) ---")
    # GUI pretends the game continued with our move g1f3 regardless (it almost
    # certainly IS g1f3); ponder on predicted reply d8h4?? which hangs the
    # queen to Nxh4 -> pondered score ~ +900 for us.
    e.send("position startpos moves e2e4 e7e5 g1f3 d8h4")
    e.send("go ponder wtime 59000 btime 59000 winc 1000 binc 1000")
    time.sleep(1.5)  # let the ponder search run
    # Opponent actually plays b8c6 -> PONDER MISS: GUI sends stop first.
    e.send("stop")
    e.wait_for("bestmove")
    print("--- ponder MISS: opponent played b8c6, not d8h4 ---")
    e.send("position startpos moves e2e4 e7e5 g1f3 b8c6")
    e.send("go wtime 58000 btime 58000 winc 1000 binc 1000")
    e.wait_for("bestmove")
    print("=== Scenario A done: check C2_CROSS cross_prev above — "
          "S1 (~+20..60) = correct, ~+800..1000 (queen-grab score) = POLLUTED ===")

    print("\n=== SCENARIO B: analysis-go pollution ===")
    e.send("ucinewgame")
    e.send("isready"); e.wait_for("readyok")
    e.send("position startpos moves e2e4 e7e5")
    e.send("go wtime 60000 btime 60000 winc 1000 binc 1000")
    e.wait_for("bestmove")
    print("--- S1 published; now an unrelated ANALYSIS go depth 12 on a "
          "queen-up position ---")
    # White up a full queen, roughly +900.
    e.send("position fen rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
    e.send("go depth 12")
    e.wait_for("bestmove")
    print("--- back to the game ---")
    e.send("position startpos moves e2e4 e7e5 g1f3 b8c6")
    e.send("go wtime 58000 btime 58000 winc 1000 binc 1000")
    e.wait_for("bestmove")
    print("=== Scenario B done: check C2_CROSS cross_prev — should show the "
          "analysis score (~+900..1300), i.e. POLLUTED ===")

    e.send("quit")
    time.sleep(0.3)
    e.p.terminate()

if __name__ == "__main__":
    main()
