# Coda Build Plan

Chess engine rewrite in Rust. Ported from GoChess with all accumulated knowledge.

## Timeline

### Phase 1: Board Mechanics (Day 1)
- Board representation (bitboards only — no Squares[64])
- Magic bitboards + PEXT runtime detection for sliding pieces
- Move generation (pseudo-legal + legality filter with pin mask)
- MakeMove / UnmakeMove (XOR-based, undo stack)
- Zobrist hashing (incremental)
- **Gate: Perft matches GoChess at depth 6+ on all test positions**
- Benchmark: target ~80-100M nodes/sec perft (vs GoChess ~30M)

### Phase 2: Search (Day 2)
- Transposition table (lockless, 5-slot buckets, 14-bit staticEval)
- Negamax + alpha-beta + PVS + aspiration windows
- Iterative deepening with time management
- All pruning: NMP (with eval >= beta guard), RFP, futility, razoring, LMR, ProbCut
- Move ordering: staged picker, TT move, good captures (MVV-LVA + captHist/16), killers, countermove, quiets (history), bad captures
- LMR doDeeper/doShallower
- History tables: main, capture, continuation (1-ply), pawn history
- Multi-source correction history (pawn + non-pawn per color + continuation)
- All search parameters from experiments.md (exact tuned values)
- **Gate: same move as GoChess on 20+ test positions at depth 12**

### Phase 3: NNUE (Day 3)
- Load v5/v6 .nnue files (1024 CReLU and SCReLU)
- HalfKA feature indexing with 16 king buckets + file mirroring
- Accumulator with incremental updates (lazy materialization)
- Finny table for king bucket changes
- Forward pass: CReLU and SCReLU with 8 output buckets
- SIMD via Rust intrinsics (AVX2 + runtime fallback)
- **Gate: same eval as GoChess on 100+ positions, bench node count matches**

### Phase 4: Integration (Day 4)
- UCI protocol
- bench subcommand (deterministic node count)
- Net loading from net.txt
- **Gate: plays complete games via cutechess-cli, no crashes**
- First SPRT against GoChess (target: within ±20 Elo)
- First rival engine gauntlet

## What We Port

### Exact values from GoChess (proven by experiment)
- All search parameters: NMP R=3+depth/3 div 200, RFP depth<=7, futility 60+d*60, LMR C=1.30/1.80, etc.
- History pruning threshold: -1500*depth
- QS delta: 240
- TT replacement: d > slotDepth-3
- Aspiration delta=15, contraction formulas
- All move ordering weights and scaling factors
- Correction history weights: pawn 512, nonpawn 204, cont 104, /1024

### Architecture decisions (learned from experience)
- **Bitboards only** — no dual representation
- **PEXT + magic hybrid** — runtime detect, PEXT on Intel/Zen3+, magic on Zen1
- **14-bit staticEval in TT** — not int8*4 (±508 clamping destroyed v7 evals)
- **d > slotDepth-3 replacement** — prevents shallow re-searches overwriting deep entries
- **Float for hidden layers** — no integer scale chain when v7 is added later
- **ExcludedMove on search stack** — enables clean SE design
- **cutNode tracking** — every top engine has this
- **NMP eval >= beta guard** — skips NMP in losing positions
- **TT bound tightening PV guard** — don't narrow windows at PV nodes
- **Capture LMR reads piece BEFORE MakeMove** — not after (was a bug in GoChess)
- **Evasion captHist /16** — match normal capture scoring scale

### What NOT to port
- Classical eval — Bullet-trained NNUE only
- Legacy .bin binpack format — use Stockfish .binpack directly
- v3/v4 net support — start with v5/v6 (1024 SCReLU is production)
- CPU NNUE trainer — we train on Bullet GPU
- Opening book builder — use external books
- The "retry candidate" experiments that were never proven
- Singular extensions — consistently harmful in our engine
- contHistPtr2 reads — table was never updated, reading zeros

### Testing methodology (from day 1)
- **Three-tier validation**: self-play fast fail → rival gauntlet → broad RR
- **One commit = one logical change** — never mix NPS and search changes
- **Perft on every build** — CI gate
- **Bench determinism** — same binary + same net = same node count
- Self-play can reject bug fixes that help cross-engine — always run Tier 2

## Key Gotchas (from GoChess experience)
- Move flag equality vs bitwise: check non-promotion flags with ==, not &
- TT mate scores: stored as score+ply, retrieved as score-ply
- En passant hash uses file only (8 keys), not full square
- NNUE incremental updates skip kings; king moves trigger full recompute
- Castling: rook captured on home square also loses rights
- NNUE v5/v6 hidden size auto-detected from file size; v7 stores in header
- SCReLU scale chain: keep v² at QA² through matmul, don't pre-divide
- Hidden→output activation is linear (no SCReLU before output buckets)
- Self-play SPRT gives contradictory signals to cross-engine testing

## NNUE Training (unchanged)
- Bullet GPU trainer (Rust, CUDA) on T80 binpack data
- 12-file 2023+2024 data for maximum diversity
- SCReLU activation, wdl=0.05, warmup LR schedule for hidden layers
- Score filter: unsigned_abs() < 10000 (consider tightening to 2000)
- Monitor neuron health at every checkpoint (SB20, SB50, SB100)
- int8 L1 (QA_L1=64) for hidden layer nets gives faster inference
