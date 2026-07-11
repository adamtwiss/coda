# SMP Cross-Thread Time-Management — Design (2026-07-05)

Scope of the external-search audit's remaining SMP lever (SMP F4). Coda is #3,
beaten only by Reckless (#2) and Stockfish (#1) — and cross-thread TM is present
in **exactly those two engines and neither of the two below us** (Obsidian #3,
Berserk #4). That distribution is the whole case: the engines we're chasing let
helper threads inform *when to stop*; Coda's helpers inform only *which move to
pick* (an end-of-search vote), contributing nothing to the during-search stop
decision. This scopes closing that gap.

## What the references actually do (two distinct designs)

**Stockfish (#1) — main-only decision, reads cross-thread instability.** Only
main runs TM, but each iteration it *sums `bestMoveChanges` across all threads*
and scales its soft budget up by
`bestMoveInstability = 1.088 + 2.315 · totBestMoveChanges / nThreads`
(`search.cpp:494-519`). Per-thread `bestMoveChanges` is a benign-race atomic,
incremented whenever a root move becomes best, reset by main each iteration.
More collective churn → more time. **No vote, no helper TM, one shared counter.**

**Reckless (#2) — distributed supermajority vote-to-stop.** *Every* thread runs
its own full TM soft-limit each iteration and casts a **retractable** vote into a
shared atomic counter; when a supermajority of threads have voted, the global
stop fires. A per-thread effort factor scales the soft-limit down as the
best-move node fraction rises. Hard deadline stays main-only. **Higher ceiling,
but helpers stop being "silent" — they must each compute a soft-limit — a real
departure from Coda's architecture.**

> [Reckless (AGPLv3) formula/source removed in the 2026-07-11 licence review — we do not reproduce AGPL-licensed code. The mechanism is described in prose.]

Obsidian and Berserk: main-only, no cross-thread signal — the same shape as
Coda today.

## Coda's starting position (favorable)

- Coda **already tracks `tm_best_move_changes`** (search.rs:2795) but it's been
  **diagnostic-only since Phase 13** (used only at :3160) — the upward
  instability multiplier it once drove was folded into the Viridithas stability
  table. The TM audit (2026-06-13, B3) explicitly flagged it "candidate for
  re-use as an SF/Reckless-style within-iteration instability factor."
- The TM factor product is assembled at **search.rs:3005** (stability ×
  fail-low × forced × subtree × score-trend) → `adjusted_soft` → break at
  :3055. A cross-thread factor slots in here as one more multiplicand.
- Helpers do **not** currently track best-move-changes (search_helper resets the
  field but never increments it — the increment lives only in main's ID loop).
- Shared-Arc pattern is established: `global_nodes`/`stop`/`ponderhit_*` are
  `Arc<Atomic…>` on `SearchInfo`, cloned into helpers, re-shared each `go` in
  `refresh_helper_common`. A cross-thread counter follows the same pattern.

## Recommendation: SF-style cross-thread instability factor (staged)

Port SF's mechanism, **not** Reckless's vote — it preserves silent helpers and
main-only TM (one shared atomic, no helper TM, no new stop path), which is the
low-risk change. Keep Reckless's supermajority vote as a documented higher-ceiling
follow-up only if the SF port validates and we want more.

### The factor
Fold into the multiplier at search.rs:3005:
```
bmc_instability = if n_threads > 1 { C1 + C2 · total_bmc / n_threads } else { 1.0 }
```
- `total_bmc` = summed best-move-changes across ALL threads (main + helpers)
  since the last main iteration.
- **Gated to `n_threads > 1`** → **T=1 is byte-for-byte unchanged** (no
  single-thread TM risk; the change is purely additive at T>1, and T=1 bench is
  unaffected → OB bench-gate untouched).
- Direction: collective churn (helpers still disagreeing) → factor > 1 → search
  longer. This is *complementary* to, not redundant with, main's stability table
  — that table reads only main's own best-move stability; the cross-thread term
  catches "main momentarily settled but the pool is still churning," which is
  genuinely new information. (The retune below rebalances any overlap.)
- Constants: **start at SF's `C1=1.088, C2=2.315`** but expose as tunables
  (`TM_BMC_INSTAB_BASE`, `TM_BMC_INSTAB_MULT`, fixed-point) — Coda's TM is
  Viridithas-shaped and separately calibrated, so these will want a retune
  (see below).

### Shared state + memory ordering
- Add `cross_thread_bmc: Arc<AtomicU32>` to `SearchInfo` (mirror `global_nodes`),
  re-shared in `refresh_helper_common`, reset to 0 at search start.
- **Publish (every thread, incl. main):** on a root best-move change,
  `cross_thread_bmc.fetch_add(1, Release)`. For helpers this means adding the
  best-move-change tracking to `search_helper`'s ID loop (it currently only
  resets the field). Once per root-move-improvement, as SF does.
- **Read + reset (main only, per ID iteration):** in the TM block, load with
  `Acquire`, compute the factor, then `store(0, Release)` to reset the window.
  The reset racing with helper increments is benign (an approximate count is
  fine for TM — SF relies on exactly this).
- ARM: the `Release`/`Acquire` pair is the one new reader-publish edge; matches
  the CLAUDE.md standard. No data-dependent load, so a benign race on the count
  is acceptable (documented).

### Guards
- `n_threads > 1` (above) and the existing `depth >= 4 && soft_limit > 0 &&
  !should_stop()` gate the whole TM block already applies.
- Clamp the factor into the existing no-inc / low-inc multiplier ceilings
  (search.rs:3017-3036) so it can't blow `adjusted_soft` past `hard`/`max_time`.
- Mate-score / single-legal-move paths already break before this block.

### Retune-on-branch
Adding an upward TM factor shifts the whole soft-budget balance — the stability
table and the inc-ceiling constants were tuned *without* it. So after the raw
port: **focused SPSA on the TM cluster** (the two new constants + stability
table + inc-ceilings) at LTC, then re-validate. Expect the raw port to be
~neutral-to-small and the retune to carry it (the classic Coda retune-on-branch
pattern; TM is high-leverage per §TM-class in CLAUDE.md).

## Validation plan

TM-class changes are partly OB-visible here (a during-search stop changes time
allocation → game outcomes at T>1), unlike ponder. So:
1. **OB Threads=4, Hash=256, `[0,3]`** — the cross-thread signal only exists at
   T>1; this is the primary measurable gate. (Raw port first; then the
   retune-on-branch values.)
2. **T=1 non-regression is free** — the `n_threads>1` gate makes T=1 identical;
   a quick `[-2,1]` T=1 SPRT (or just bench-identity) confirms no accidental
   single-thread change.
3. **Local ponder-enabled cross-engine RR** at deployment TC — the
   deployment-representative cross-check (lichess runs 4–8T + ponder), since OB
   can't exercise ponder and T=4 is a proxy for 4–8T. Use the standing TM-class
   methodology (per-move clock inspection first, then RR).
4. **Mechanism probe** before trusting Elo: instrument the factor's firing rate
   + the distribution of `total_bmc/n_threads`, confirm it actually varies
   across positions (not stuck at baseline) — the vote-override-probe pattern
   from P0.5.

## Higher-ceiling follow-up (only if the SF port wins)

Reckless's **supermajority retractable vote-to-stop** is the more aggressive mechanism and
Reckless is our closest peer. It requires making helpers TM-aware (each computes
its own soft-limit multiplier from its own `root_moves[0].nodes` + stability),
which is a substantial architecture change (silent-helper invariant broken).
Deferred: only worth it if the low-risk SF scaling validates AND leaves
measurable headroom. The mechanism is described above for when/if we build it.

## Effort / risk

- **SF port:** ~40-80 LOC (one Arc atomic + helper bmc tracking + the factor +
  two tunables) + a TM-cluster retune. Low architectural risk (preserves silent
  helpers). Main uncertainty is Elo magnitude — TM-class, needs the retune to
  express.
- **Files:** `src/search.rs` (SearchInfo field, `refresh_helper_common`,
  `search_helper` bmc tracking, the factor at :3005, two tunables in
  `tunables!`). No thread_pool.rs change needed (the atomic is on SearchInfo,
  shared like global_nodes).
