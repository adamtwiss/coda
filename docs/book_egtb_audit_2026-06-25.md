# Opening Book + Syzygy EGTB Audit — 2026-06-25

Three-agent read-only audit of two subsystems OpenBench can't test (OB has no
opening book and no tablebases). Files: `src/book.rs`, `src/polyglot_randoms.rs`,
`src/tb.rs`, `src/tb_cache.rs`, plus probe sites in `search.rs`/`uci.rs`. EGTB
compared against SF/Reckless/Obsidian/Berserk/Alexandria/PlentyChess (all have
Syzygy). HEAD ~e5e89ac.

## HEADLINE: both subsystems are correct and well-defended

No game-losing bug in either. The historically dangerous EGTB failure modes —
cursed/blessed-win scored as a win, a cache key ignoring rule50, KQvK
oscillation-to-draw, drawn-fortress throw, panic on missing table, torn reads
under SMP — are **all already handled and regression-tested** (evidence of prior
"C2"/"C8" audit fixes in the code). The book handles all four CLAUDE.md Polyglot
gotchas correctly. This audit is mostly a **validation**, not a bug harvest. The
actionable items are low-severity polish; none is worth a cutechess RR for Elo.

---

## OPENING BOOK (`book.rs`) — clean

All correct, verified against Coda's internal constants:
- **Polyglot hash** matches the standard construction exactly: piece keys
  (BlackPawn=0/WhitePawn=1/… ordering), castling keys 768-771, EP 772, turn 780,
  `Random64[0]=0x9D39247E33776D41`. Side-to-move key XOR'd only for white. ✓
- **EP-only-when-capturable** (the load-bearing rule) implemented correctly:
  only hashes the EP file when a side-to-move pawn is actually adjacent on the
  correct rank (`polyglot_has_ep_capture`). ✓
- **Move decode**: standard bit layout; castling king-to-rook → king-to-dest
  conversion correct for all four types and matches Coda's movegen king-dest
  squares; under-promotions distinguished via flag equality. ✓
- **Illegal-move safety**: every book move is matched against
  `generate_legal_moves` before play — a corrupt/foreign entry yields `None` and
  falls through to search. Cannot inject an illegal move. ✓
- **Robustness**: non-16-multiple file → Err; missing/corrupt → `info string Book
  load failed`, engine continues; out-of-book falls through cleanly; book move
  returned *before* time allocation (instant, burns no clock); skipped during
  ponder; read-only on the main thread (no SMP concern). ✓

**Notes (not defects):**
- Selection RNG seeds from `SystemTime` subsec_nanos — no determinism control,
  slight modulo bias, two probes in the same tick repeat. Only matters if we ever
  want reproducible book openings for testing → would add a seed / selection-policy
  UCI option (best-weight vs weighted-random). Feature gap, not a bug.
- Cosmetic: `book.rs:196,205` locals `rank4_mask`/`rank3_mask` are misnamed (hold
  RANK_5/RANK_4); values correct, names backwards. `&& f < 8` guard redundant.

---

## EGTB — SEARCH INTEGRATION & MOVE SELECTION (`tb.rs`, search/uci)

**Root strategy = Fathom-style "probe and play the DTZ-optimal move", NOT SF-style
rank-and-keep-searching.** When popcount ≤ max_pieces and the root probe succeeds,
Coda bypasses search and plays shakmaty's DTZ-optimal `best_move` directly (after
legality check), else falls through to normal search. Correct and instant for a
solved root; the only structural consequence is no in-search safety-net at the
root — covered by explicit child-WDL re-verification.

**Convergence ("knows it's winning but never converts") is correctly defended:**
- Interior: WDL win → `TB_WIN - ply`, loss → `-TB_WIN + ply`. The `-ply` term
  encodes "win, how soon" → progress pressure toward fastest conversion, identical
  to SF's `VALUE_TB - ss->ply`.
- Root winning tiebreak `pick_winning_tb_move`: `best_captured_value` starts at
  **0 not −1** so a non-capturing DTZ move is never overridden by an arbitrary
  win-preserving non-progressing move (the documented KQvK-draw fix); captures
  re-verify the child is still a definite loss (`child_wdl ≤ −19000`). Progress
  driven by DTZ. ✓
- Root drawn tiebreak `pick_drawn_tb_move`: re-probes each child, rejects any move
  handing the opponent a win (the KBPvKB draw-throw fix). ✓

**Gating**: cardinality checked at every call site ✓; interior castling gate ✓;
rule50 handled via shakmaty's halfmove-aware `AmbiguousWdl` + halfmove-keyed cache
(more capable than SF's `rule50==0` cut) ✓. Score constants cleanly separated from
mate (`TB_WIN=28800` < `MATE=29000`; deepest TB ≈28672 stays below the mate band);
`tb_floor` prevents an UPPER bound below TB ground-truth from poisoning the TT. ✓

### Actionable (all LOW severity)
| # | Finding | file:line | Type | Fix |
|---|---------|-----------|------|-----|
| F1 | Interior probes bump no `tbhits` counter (info lines under-report) | search.rs:3039 | Stat/cosmetic | add `stats.tb_hits += 1` on interior probe hit, thread into UCI info |
| F2 | Root probes lack the castling gate the interior probe has | tb.rs:99,132 | Robustness | add `if board.castling != 0 { return None; }` to `probe_root`/`probe_root_pv` for parity + guaranteed fall-through |
| F3 | No `SyzygyProbeDepth`; interior probe fires at every node incl depth-0 frontier | search.rs:3036 | NPS | **DONE 2026-06-26.** Added SF-style gate `pc<max \|\| depth>=tb_probe_depth` + UCI spin `SyzygyProbeDepth` (default 1). Local RR (5-man = deploy config, STC 10+0.1, **no adjudication** to force endgames, coda_dev gated vs coda_base always-probe): **+6.5 ±3.9 Elo, LOS 99.9%, N=7556**. Real local-only-measurable win (OB has no TB). See NET TAKEAWAY below. |
| F4 | Interior TB cutoff returns without a TT store | search.rs:3062-65 | Efficiency | optional TT store with bound+high depth so parent ordering sees the TB bound |

---

## EGTB — NEAR-EDGE CASES & TBHash CACHE (`tb.rs`, `tb_cache.rs`)

**The two dangerous classes are both CLEAN:**
1. **Cursed-win / blessed-loss**: `ambiguous_wdl_to_score` maps Win→+20000,
   Loss→−20000, CursedWin/MaybeWin→**+1**, BlessedLoss/MaybeLoss→**−1**, Draw→0.
   Interior search only promotes `|wdl|>1` to a mate-class bound; `±1`/`0` returned
   as the small score — a cursed win can never trigger a win cutoff or be
   over-pressed into the 50-move draw. Matches SF's `WDL_to_value` (cursed/blessed
   = DRAW±2). ✓
2. **Cache key incorporates rule50** (the subtle one): `effective_key(key,
   halfmove)` SplitMix-scrambles halfmove into both slot index and verification
   key; `probe_wdl` passes `board.halfmove` to probe+store; test
   `different_halfmove_misses` pins it. Same placement returns Win at hm=0 and
   CursedWin at hm=80 → cached separately, no cross-contamination. ✓

**Also clean:** no panic/unwrap on missing/malformed tables in any production path
(every `Err(_)` → `None` → fall back to NNUE search; malformed FEN → `.ok()?`);
atomics are correct for aarch64 (store value+key both Release, probe both Acquire,
full-64-bit XOR-key verification, `clear()` Relaxed is fine single-threaded);
`MaybeWin`/`MaybeLoss` DTZ-rounding collapsed to ±1 is conservative (never
over-claims); TBHash resize/clear on ucinewgame race-free.

### Actionable (cosmetic)
| # | Finding | file:line | Type | Fix |
|---|---------|-----------|------|-----|
| 5 | Root `dtz_to_wdl_score` is halfmove-UNAWARE — a cursed-by-clock root prints `score cp ±TB_WIN` and walks a multi-ply PV | tb.rs:195-214 | Cosmetic (display only) | gate on `\|DTZ\|+halfmove>100`; move choice already safe because `pick_winning_tb_move` re-probes children halfmove-aware |

**Minor strength notes (fine-as-is):** Coda declines the 6-man parent when only
5-man tables loaded (`popcount>max → None`), forgoing the one-capture-into-TB reach
that shakmaty's internal `MAX_PIECES+1` tolerance would give SF/Fathom — correct,
just slightly less reach. `MaybeWin` re-search (SF resolves DTZ rounding by
re-searching; Coda takes the conservative ±1) — acceptable at Coda's strength.

---

## NET TAKEAWAY & RECOMMENDATION

Both subsystems are in good shape — this audit found **no correctness bug worth a
test**. The sensible polish, if we touch anything, is a tiny cleanup bundle that
needs no Elo validation:
- **F2** (root castling-gate parity) + **Finding 5** (halfmove-aware
  `dtz_to_wdl_score`) — both one-liners, both make the root TB path strictly more
  correct/honest, zero strength risk.
- **F1** (tbhits accounting) — cosmetic, improves observability of TB usage.
- Book: optional **selection-policy / seed UCI option** if reproducible book
  openings are ever wanted for local RR testing (would also let us A/B
  best-weight vs weighted-random book play).

**F3 (probe-depth threshold) — IMPLEMENTED & MERGED 2026-06-26.** The one genuine
behavior/NPS change in this audit. Mechanism: at the maximum loaded piece count the
interior WDL probe now fires only when `depth >= SyzygyProbeDepth` (default 1) —
skipping the depth≤0 qsearch frontier, which is the most numerous, least-rewarding
probe layer. Below the max piece count it still always probes. The suppressed probes
are largely **redundant** (the same 5-man position is re-probed one ply up at
depth≥1), so the change removes a per-node FEN-roundtrip + table-decompression cost
at ~zero knowledge loss.

Could not be measured on OB (workers have no tablebases). Validated by local
cutechess RR on the **exact 5-man set codabot deploys** (7-man is 17 TB / 6-man is
150 GB — impractical; lichess runs 5-man), STC 10+0.1, **resign/draw adjudication
removed** so games grind into real endgames where the gate fires: `coda_dev` (gated)
vs `coda_base` (always-probe), bench-identical, **+6.5 ±3.9 Elo, LOS 99.9%,
N=7556**. The early reading (+9.6 @ N≈1500) regressed to a stable +6.5 band as the
CI tightened. No-adjudication makes this *more* deployment-representative than a
normal adjudicated RR, not less — real lichess endgames are played out and decided
on the clock (16% flag-falls in this RR), exactly where the endgame NPS saving
converts to Elo.

Follow-up under test: `SyzygyProbeDepth=1` only skips the qsearch frontier; our
5-man-everything economics differ from SF's 6/7-man default (where the gated top
layer is rarer/transient), so a higher value (skip 5-man probes up to depth 2–3) may
gain more before the lost knowledge bites — `gated-ProbeDepth=1` vs `=4` RR.
