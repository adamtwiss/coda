# Move Ordering Analysis and Improvement Plan

## Current Baseline (2026-04-06)

```
Move ordering:  avg cutoff pos 1.91, avg pos² 16.6, first-move 75.0%
```

75% of beta cutoffs happen at the first move tried. The remaining 25% average around position 4.7, with some at position 8-10+ (high pos² of 16.6). Stockfish achieves 85-90% first-move cutoff rate.

### Impact of Different NNUE Models on Ordering

| Model | First-move % | Avg pos | Avg pos² | Notes |
|-------|-------------|---------|----------|-------|
| 768pw (production) | **75.0%** | 1.91 | 16.6 | Current production |
| 1024s-13f (blunders) | **76.0%** | 1.88 | 17.6 | Best first-move — diverse training helps |
| 1024s | 71.6% | 1.84 | 13.6 | Lower first-move but tight distribution |
| 1024c (oldest) | 72.6% | 2.34 | 30.7 | Worst late-miss tail |
| v7 1024h16x32s | **52.7%** | 2.09 | 11.0 | Broken — eval inconsistent between iterations |

**Key insight**: Model quality directly drives move ordering quality. The v7 model's 52.7% first-move rate explains much of its weakness — the eval flip-flops between iterations, so TT moves are wrong half the time.

## Cross-Engine Comparison (8 engines analyzed)

### Features We Have
- TT move → Good captures → Killers → Counter-move → Quiets → Bad captures
- 4D threat-aware main history (from_thr × to_thr × from × to)
- Pawn history [pawnHash%512][piece][to]
- 4 continuation history plies (1,2,4,6) at weights 3x/3x/1x/1x
- Capture scoring: MVV×16 + captHist
- Binary SEE threshold (≥0) for good/bad capture split

### Missing Features (prioritized by expected impact)

#### 1. Dynamic Capture SEE Threshold (HIGH — 1-line change)

**Gap**: We use binary SEE≥0. Every top engine uses `-captScore/N`:
- Stockfish: `-captScore/18`
- Obsidian/Alexandria: `-captScore/32`
- Viridithas: `-captScore/46 + 109`
- Berserk: `-captScore/2`

**Why it matters**: A capture with high MVV + positive captHist but slightly negative SEE (e.g. -50) gets thrown in the bad-captures bin. With a dynamic threshold, it stays in good captures — tried before killers/quiets.

**Implementation**: In movepicker, change `see_ge(board, m, 0)` to `see_ge(board, m, -score / 32)`.

**Measurability**: Run bench, check if first-move % improves.

#### 2. Quiet Check Bonus (HIGH — needs check detection)

**Gap**: Zero bonus for quiet checks. Three strong engines give massive bonuses:
- Stockfish: +16384 (requires SEE ≥ -75)
- Viridithas: +10000
- Reckless: +10000

**Why it matters**: Checking moves are disproportionately likely to cause cutoffs. Without a bonus, they're buried in quiet ordering by random history scores.

**Implementation**: Need pre-move check detection. Options:
- Precomputed `checking_squares` bitboard (Viridithas/Reckless approach) — per piece type, which squares give check to enemy king. Computed once at node start.
- Cheap knight+pawn direct check only (covers most common cases)
- Full gives_check with make/unmake (expensive, defeats purpose)

The `checking_squares` infrastructure would also benefit futility/LMP gives_check exemptions (currently removed due to pre-MakeMove migration).

**Measurability**: First-move % should improve noticeably.

#### 3. Piece-Type Threat Escape/Enter Bonuses (MEDIUM)

**Gap**: Our 4D history learns threat patterns over time, but explicit bonuses give correct ordering immediately:
- Queen escaping rook attack: +20K-32K
- Rook escaping minor attack: +14K-16K
- Minor escaping pawn attack: +8K-16K
- Moving piece INTO danger: corresponding malus

Used by: Obsidian, Berserk, Viridithas, Reckless, PlentyChess (5/8 engines).

**Implementation**: Requires computing `threats_by_pawn`, `threats_by_minor`, `threats_by_rook` bitboards (opponent's pawn/minor/rook attacks). Apply piece-type-specific bonus/malus in quiet scoring.

**Measurability**: Moderate impact on first-move %. Biggest impact in tactical positions.

#### 4. Good/Bad Quiet Split (MEDIUM)

**Gap**: Stockfish separates quiets into good (score > -14000) and bad. Good quiets tried before bad captures, bad quiets tried after. We try ALL quiets before bad captures.

**Why it matters**: Quiet moves with deeply negative history should be tried after losing captures — they're almost never the best move.

**Implementation**: In movepicker, stop yielding quiets when score drops below threshold. Add a "bad quiets" stage after bad captures.

#### 5. Continuation History Weights (LOW)

**Gap**: Our 3x/3x/1x/1x for plies 1/2/4/6 is unusual. Most engines use 1x/1x/1x/1x (equal weights). The 3x amplifies potentially noisy signals from recent plies.

**Worth testing**: 2x/2x/1x/1x or 1x/1x/1x/1x.

#### 6. Capture Scoring MVV Multiplier (LOW)

**Gap**: Our MVV×16 may drown out captHist signal. SF uses 7×PieceValue, giving more weight to capture history (a learned signal).

#### 7. Other Observations

- **SF removed killers and counter-moves entirely** — relies on history alone. We still use both. Worth testing removal if history tables are strong enough.
- **Low-ply history bonus** (SF): 8× history weight at ply 0, decaying with depth. Gives root moves better ordering.
- **Root history** (Alexandria): dedicated table for root moves, 4× weight.
- **Pawn history initialization** (PlentyChess): starts at -1000 (pessimistic prior for unknown moves).

## LMP Interaction

LMP_BASE has been pushed to 9 by every SPSA round (nearly disabling LMP). This is because:
1. Our 75% first-move rate means 25% of cutoffs are at move 5+
2. LMP prunes at move count > (BASE+d²)/(2-improving)
3. Some of those late cutoff moves are getting pruned before they're tried
4. SPSA compensates by raising BASE to let more moves through

**Fixing move ordering would fix LMP**: if first-move rate rises to 85%+, the remaining 15% are at position 2-3, and LMP at BASE=3-5 would be safe.

## Action Plan

### Phase 1: Quick wins (measurable via bench ordering stats)

1. **Dynamic capture SEE threshold** — 1-line change, measure first-move %.
2. **Cont hist weights 2x/2x/1x/1x → 1x/1x/1x/1x** — simple, measure impact.
3. **MVV multiplier 16→8** — simple, measure impact.

### Phase 2: Infrastructure + features

4. **Checking squares bitboard** — compute at node start, enables quiet check bonus AND future gives_check gates.
5. **Quiet check bonus** (+10000-16384) — needs checking squares.
6. **Piece-type threat bonuses** — needs opponent attack bitboards.

### Phase 3: Structural

7. **Good/bad quiet split** in movepicker.
8. **Test killer/counter-move removal** — if history is strong enough.
9. **Root history table** — dedicated ordering for root moves.

### Success Metric

Each change measured by bench `first-move %` and `avg pos²`. Target: first-move >80%, avg pos² <12. Then re-tune LMP via SPSA — should converge to a reasonable BASE (3-5) instead of 9.
