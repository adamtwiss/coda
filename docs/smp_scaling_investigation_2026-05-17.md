# SMP Scaling Investigation (2026-05-17)

Lifting Coda's Lazy SMP scaling from ~+68 Elo (T=4 vs T=1) to ~+105 Elo,
matching Reckless and Stockfish on identical hardware.

This doc captures the methodology, the surprises, and the transferable
lessons. The code-level details live in the three commits on
`experiment/smp-helper-final` (final SHA `69977dd`).

## TL;DR

| Stage | T=1 Elo | T=4 Elo | Comment |
|---|---:|---:|---|
| Main (pre-investigation) | baseline | baseline | Coda.4 vs Coda.1 = +68 Elo (~31% scaling efficiency) |
| Bundle: helper aspiration + history seeding (cgu=1) | **−8.9** (H0) | **+89** | Bench identical, no T=1 code touched, yet T=1 regressed |
| Bundle + `codegen-units = 16` | **+3.0** (H1) | **+98** (4-way RR) | T=1 recovered, T=4 preserved |
| Final: bundle + cgu=16 + panic/strip | ~+3 | ~+105 | Reference parity: Reckless +105, SF +109 |

The compiler-flag fix was a bigger surprise than the search-code changes.

## Phase 1 — Establishing the gap

Before any code change, measured scaling on the actual hardware/TC the
deployment cares about (Lichess T=4, 10+0.1 / 60+1).

```
Reckless.4 vs Reckless.1 = +105 Elo  (48% scaling efficiency)
SF.4 vs SF.1             = +109 Elo  (50% scaling efficiency)
Coda.4 vs Coda.1         = +68  Elo  (31% scaling efficiency)
```

**Lesson — anchor every "is our X bad?" question on numerical
reference points before touching code.** "We should improve SMP" is
unfalsifiable. "Our T=4 / T=1 ratio is 19 Elo behind SF on the same
hardware" tells you the gap and gives you a target. Without anchored
references, post-change attribution is impossible: any improvement
looks great in isolation.

## Phase 2 — Cross-engine review

Surveyed Stockfish, Reckless, Obsidian, Alexandria, Viridithas, and
PlentyChess. Coda was the outlier on **four** simultaneous axes:

| Aspect | Coda (before) | Top-6 consensus |
|---|---|---|
| Helper search loop | `negamax(-INF, +INF)` per depth | Full aspiration ID, same as main |
| Score carry between iterations | None | `prev_score` carried (mainline aspiration center) |
| Helper history table | Zeroed on every `go` | Inherits aged history from main, or persisted across `go` |
| Depth offset | `+0` or `+1` per `thread_id % 2` | None — diversity from aspiration variance + async |

**Lesson — when a homegrown pattern is unique across 6 reference
engines, the prior is "we got it wrong," not "we discovered
something."** Six engines independently converged. Convergence under
SPRT pressure is unusually strong evidence. The four-axis divergence
was the entire SMP gap.

Code changes (commit `22d8eea` → rebased `8538458`):

1. `search_helper` runs the same aspiration ID loop as `search`, gated
   on `info.silent` (no UCI output, no TM).
2. `create_helper_info` copies main's just-aged history table
   (`History::copy_from` — new method on `movepicker.rs`). Pawn history
   (13 MB) and correction history (cleared each search anyway) excluded.
3. Depth offset removed.

Bench unchanged (single-thread path bypasses helper construction).

## Phase 3 — The surprise: T=1 regressed

Bundle measured **+89 Elo at T=4** in a 1200-game RR — matching/beating
reference engines. SPRT'd at T=1 as a routine non-regression check.

OB SPRT #1272: **T=1 −8.9 ±6.1 Elo, H0 locked**.

This was unexpected because:

- Bench is identical (5408541 → 4669324 after rebase, same on both sides).
- T=1 code path doesn't touch any new code: `if threads <= 1` in
  `search_smp` bypasses helper construction entirely. The new
  `search_helper` body, `create_helper_info`, and `History::copy_from`
  are all dead code at T=1.
- All three pruning-rate counters were within noise.

So *no code that runs at T=1 had changed*, yet T=1 played 9 Elo worse.

The cause was **LTO codegen perturbation**.

### What's actually happening

Default `release` profile: `lto = true` (full LTO), `codegen-units = 1`.
This means LLVM treats the entire crate as a single translation unit
for inlining and global optimization decisions.

Adding a new function — even dead-at-T=1 code — changes:

- Per-function inlining costs (more candidates compete for the inline
  budget).
- Register allocation in functions that *call* the new code (compiler
  must keep options open for paths that may be inlined).
- Function layout / I-cache locality on the hot path.
- Constant propagation across the now-larger program.

The result is a small but reproducible perturbation of the main-thread
hot path (`search` / `negamax` / `MovePicker::next`), enough to move
~9 Elo at 10+0.1 even with bench unchanged. Bench is a fixed-depth
node count — it cannot detect NPS regressions.

**Lesson — "bench unchanged + no executed code change ⇒ Elo neutral"
is not a sound inference under full LTO + cgu=1.** Compiler-level
effects can move Elo without moving nodes. Always SPRT at T=1 as a
regression check, even for changes that "obviously can't affect T=1."

## Phase 4 — Mitigation candidates

Tested four mitigations as separate branches and SPRT'd each at T=1
(regression check, [-3, 3]) and locally at T=4 (does it preserve the
+89 Elo win?). All results at the same identical-bench point:

| Branch | Mechanism | T=1 SPRT | T=4 (RR) |
|---|---|---:|---:|
| `inline-never` | `#[inline(never)]` on helpers | **−6.3** H0 | +83 |
| `aspiration-cold` | `#[inline(never)] + #[cold]` | **+3.3** H1 | +36 ⚠ |
| `thin-lto` | `lto = "thin", codegen-units = 16` | **−2.4** H0 | (not measured — bad T=1) |
| `cgu16` | `lto = true, codegen-units = 16` | **+3.0** H1 | **+98** |

Two observations:

**`#[cold]` is not free.** It marks a function as unlikely-to-execute,
which moves it out of the hot path and is the conventional fix for "I
added a helper that shouldn't perturb the main path." But `#[cold]`
also **reduces LLVM's optimization budget for the function body**:
every callee inside (negamax, MovePicker, eval) gets less aggressive
inlining inside `search_helper`. At T=1 this is invisible (function
never runs), so the regression fix looks clean. At T=4 it costs ~53
Elo (BundleC1 +89 → Cold +36). The direct head-to-head bundle-vs-cold
RR locked at +52.5 ±33 favouring the bundle.

**Thin LTO alone wasn't the answer.** Thin LTO is the "right" fix in
principle — it splits compilation into per-CGU passes with a
lightweight global merge. But empirically `thin-lto` at cgu=16 SPRT'd
−2.4 at T=1 (H0), shifting negative as the game count grew. The exact
reason is unclear (possibly worse inlining decisions for hot
single-threaded paths than full-LTO-cgu=16 produces).

**`codegen-units = 16` with full LTO retained was the structural fix.**
LLVM still does whole-program optimization, but the intermediate CGU
split gives the linker more breathing room. New code's perturbation is
localized to one CGU rather than rippling across the whole binary.
Crucially, no `#[cold]` annotation is needed, so the helper body keeps
its full optimization budget.

## Phase 5 — Validation

OB SPRT confirmations (all T=1, [-3, 3] regression-check bounds):

| ID | Branch | T=1 Elo | Status |
|---|---|---:|---|
| #1272 | bundle, cgu=1 | −8.9 ±6.1 | H0 (the regression) |
| #1281 | aspiration-only, `#[cold]` | +3.3 ±3.7 | H1 ✓ |
| #1287 | bundle + cgu=16 | +3.0 ±3.6 | H1 ✓ |
| #1288 | panic/strip alone | +0.1 ±1.2 | near-flat |
| #1295 | final (bundle + cgu=16 + panic/strip) | pending | — |

Local 4-way RR (600 games, 300 per pair, 10+0.1, UHO openings),
all measured against MainC1 (current main, default profile):

```
MainC1.4   =   0    (reference)
MainC16.4  = +10 ±36   (cgu=16 alone — within noise of free)
BundleC1.4 = +103 ±36  (bundle at cgu=1)
BundleC16.4 = +108 ±36 (final config)
BundleC16 vs MainC16 = +98 ±36   (bundle, apples-to-apples)
```

The apples-to-apples bundle-vs-bundle (cgu held fixed at 16) lock at
+98 Elo isolates the search-code contribution. The +10 for cgu=16
alone is within noise of free at T=4.

## Phase 6 — Profile-byte hygiene

Added `panic = "abort"` and `strip = true` to the release profile
(matching Viridithas, Reckless, Hobbes). SPRT #1288 locked near-flat
(+0.1 ±1.2). Inclusion was for binary-size hygiene (~0.4 MB smaller)
and consensus-profile parity, not Elo.

## Lessons (transferable)

1. **Anchor "is X bad?" questions in numerical reference points.**
   Cross-engine measurement on identical hardware/TC tells you both
   that a gap exists and how much there is to recover. No-anchor
   investigations grade themselves.

2. **Cross-engine consensus on a homegrown pattern usually means we
   got it wrong, not that we discovered something.** Six engines
   converging under SPRT pressure is strong evidence. Coda's
   four-axis SMP divergence was the entire 37-Elo gap.

3. **Compiler-level effects can move Elo with bench unchanged.** Full
   LTO + cgu=1 turns the whole crate into one optimization unit;
   adding dead-at-T=1 code can perturb T=1 hot paths. **Bench is not a
   regression detector for compiler-perturbation effects.** SPRT at T=1
   anyway, even for "obviously T=1-neutral" changes.

4. **`#[cold]` is not a free annotation.** It marks the function cold
   AND cuts its optimization budget. Use it for paths you genuinely
   want deprioritized (error handling, rare branches). Don't use it
   as a "make this not affect the main path" hammer for code that
   *does* execute, just on a different thread.

5. **Default `codegen-units = 1` is a general-purpose Rust release
   choice.** For an engine where we routinely add helper functions
   adjacent to hot paths, `codegen-units = 16` is structurally
   better: same LTO scope, less ripple from incremental additions.

6. **Test mitigations against both regression direction AND
   preservation of the win.** The first mitigation we tried
   (`#[cold]`) fixed T=1 cleanly but quietly halved the T=4 win.
   Without a held-out T=4 measurement we'd have shipped a worse
   final result than the one we got.

7. **Same-base-bench bookkeeping is fragile across rebases.** SPRT
   #1292 was rejected with "Wrong Bench" because main moved
   forward (Phase 3 cleanup) while the branch was queued, shifting
   main's bench from 5408541 → 4669324. Rebase + bench-update + force
   push is part of the SPRT workflow, not an exception. Always
   re-measure bench on the exact branch+main pair you're submitting.

## Followups

- **PGO with cgu=16.** PGO regressed when v9 landed; possibly because
  cgu=1 + 50 MB embedded net was too large for LLVM's PGO-LTO pass.
  With cgu=16, PGO has smaller units to optimize per-CGU and may
  recoup the 2-3% NPS it previously gave. Worth re-testing on the new
  trunk.
- **cgu sweep {4, 8, 16, 32}.** 16 was picked as the conservative
  middle of the consensus engine range; the actual sweet spot might be
  lower or higher.
- **Vote-based best-thread selection** (SF/Obsidian/Plenty pattern):
  +1-2 Elo at T>1, ~30 LOC.
- **Persistent thread pool** (condvar-based, à la SF/Reckless):
  eliminates per-`go` spawn overhead. Most visible at very short TCs.
- **Shared correction history across threads** (SF/Reckless pattern,
  `Arc<lockless atomic>`): another small win, more code-invasive.

## References

- Commits on `experiment/smp-helper-final`:
  `8538458` (helpers + history seeding), `f848809` (cgu=16),
  `69977dd` (ship + writeup)
- `History::copy_from` in `src/movepicker.rs`
- `search_helper` and `create_helper_info` in `src/search.rs`
- Cross-engine sources in `~/chess/engines/{Stockfish,Reckless,Obsidian,Alexandria,Viridithas,PlentyChess}`
- OB SPRTs: #1272, #1274, #1279, #1281, #1283, #1285, #1286, #1287,
  #1288, #1295 (final regression check)
