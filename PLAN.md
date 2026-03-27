# Coda Build Plan

Chess engine rewrite in Rust. Ported from GoChess with all accumulated knowledge.

## Completed (2026-03-27)

### Phase 1: Board Mechanics ✓
- Bitboards + mailbox, magic bitboards + PEXT, move generation, make/unmake, Zobrist
- Perft: 131M NPS, all 6 standard positions pass

### Phase 2: Search ✓
- Full negamax with 20+ pruning features, TT, move ordering, correction history
- PeSTO eval as classical fallback

### Phase 3: NNUE ✓
- v5/v6 format (CReLU + SCReLU), HalfKA 12288 features, 8 output buckets
- Lazy accumulator + Finny table + AVX2/AVX-512 SIMD + int8 quantization
- ~1.16M NPS with CReLU 1024

### Phase 4: Integration ✓
- UCI protocol (position, go, stop, ponder, setoption)
- Polyglot opening book support
- EPD test suite runner
- Time management with overhead and emergency mode
- 54 commits, ~6000 lines of Rust

## Current Strength
- ~-90 Elo vs Crafty/Rodent/ExChess/GreKo gauntlet
- Beats Crafty, competitive with Laser
- Roughly -250 to -300 from GoChess

## Next Steps

### Performance
- [ ] Lazy SMP (parallel search, shared TT)
- [ ] Async search thread (enables proper stop/ponderhit)
- [ ] SCReLU byte decomposition optimization

### Search Features
- [ ] Gives-check exemption in LMP/futility/LMR (needs post-make check)
- [ ] Pawn history in move ordering score_moves (currently only in history pruning)
- [ ] Eval instability detection (reduce LMR less in volatile positions)
- [ ] Dynamic time management (best-move stability, score stability)
- [ ] Alpha-reduce (disabled: needs better move ordering first)
- [ ] Failing heuristic (disabled: needs better move ordering first)

### NNUE
- [ ] v7 hidden layer support (1024→16→32→1×8 SCReLU)
- [ ] NEON SIMD for ARM64
- [ ] Contiguous accumulator allocation (cache-friendly)

### Infrastructure
- [ ] Syzygy tablebase support
- [ ] Multi-PV search
- [ ] UCI `info` with full PV, hashfull, tbhits
- [ ] fetch-net subcommand (download from GitHub releases)
