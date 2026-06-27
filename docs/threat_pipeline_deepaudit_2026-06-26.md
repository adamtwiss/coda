# v9 Threat Pipeline — Deep Correctness Audit (2026-06-26)

**Scope:** `src/threats.rs` (enumeration + delta generation), `src/threat_accum.rs`
(per-ply accumulator stack), and the threat→FT boundary in `src/nnue.rs`
(`forward_with_threats`). Read-only audit. Companion to
`docs/threat_features_design.md`, `docs/threat_accumulator_findings_2026-06-15.md`,
`docs/threat_eval_asymmetry_2026-06-17.md`,
`docs/threat_generation_pregating_plan_2026-06-17.md`.

## Headline

**The live incremental delta path is provably consistent with a full
from-scratch recompute, and it is among the best-verified subsystems in the
engine.** I found **no delta/recompute divergence** in the production path. The
two prior ~200-Elo parity bugs (x-ray re-indexing, `can_update` king-cross
ordering, both 2026-04-17) are fixed and now pinned by regression tests.

The findings below are **maintainability / test-coverage** issues, not live eval
corruption:

1. **MEDIUM (maintainability):** A complete *second* threat-accumulator
   implementation lives in `nnue.rs` (`recompute_threats_if_needed`,
   `recompute_threats_full`, `verify_threats`, the `self.threat` `AccDataStack`)
   and a dead `compute_move_deltas` in `threats.rs`. **None are on the
   production path** — production uses `threat_accum::ThreatStack`. The dead
   `nnue.rs` replay even carries a *weaker, admittedly-approximate* mirror rule.
   Drift / accidental resurrection risk.
2. **LOW (CI coverage gap):** The randomized parity test (`fuzz_random_games`)
   always materializes every ply (replay gap == 1) and never exercises
   pop/unmake or deep lazy replay (gap ≥ 2). The multi-ply lazy-replay path
   (≈9.5% of evals) + pop-restore are covered only by the **opt-in**
   `CODA_VERIFY_NNUE` runtime mode and a debug-only assert **capped at 20
   evals** — neither runs in `cargo test`/CI.

## Why the delta path is provably sound (how I verified)

The parity argument rests on three structural facts, then the test battery.

### 1. Generation and full-recompute share the *same* index function

`apply_threat_deltas` (threats.rs:2002-2011) computes every feature index via
`threat_index(...)` — the exact function `enumerate_threats` (the full-recompute
reference) uses (threats.rs:735, 788). Apply then **skips `idx < 0`**
(excluded/semi-excluded) and `idx >= num_threats`. Consequence: generation may
freely *over-emit* raw deltas (e.g. both directions of a same-type mutual
attack); the excluded direction is dropped identically in both paths. So the
delta path cannot disagree with full recompute on the *mapping* from a
(attacker, from, victim, to) relation to a feature — only on *which relations it
emits*. That narrows the audit surface to "does generation emit exactly the set
of changed relations," which the tests below pin down element-wise.

### 2. Same-type semi-exclusion is symmetric and double-count-free

`PiecePair::base` (threats.rs:463-466) folds `below = (attacking_sq <
attacked_sq)` into bit 30; the `& 0x80FFFFFF` mask makes exactly one direction
of a semi-excluded pair return a valid (positive) index and the other return
negative (skip). The kept index is identical regardless of which physical
direction is queried. So for a same-type mutual attack, enumerate fires one
feature and the delta path's two raw emissions collapse to the same single
add/sub — **no double count, no omission.** This is also the documented source
of the deliberate (non-bug) color asymmetry in
`threat_eval_asymmetry_2026-06-17.md`; training matches it (fuzzer 0/40000).

### 3. Lazy-replay mirror is *exact*, not approximate (live path)

`ThreatStack::update` (threat_accum.rs:297-353) computes the king mirror from the
*current* board and applies it to every ply in the replayed span. This is
correct because `can_update` (threat_accum.rs:260-293) walks the whole span and
returns `None` on the **first** ply whose move crosses the e-file for *this*
perspective (or whose deltas overflowed). Therefore an accepted span contains
**zero** mirror changes for the perspective being replayed → the perspective's
mirror is constant across the span → using the current mirror is exact. (Opponent
king crossings are irrelevant: a perspective's mirror depends only on its own
king.) The "net-zero crossing" escape — king leaves and returns across e4/e5 —
cannot slip through, because `can_update` bails on the first crossing it sees.

### 4. Incremental decomposition of compound moves is exact

board.rs `make_move` (board.rs:862-913) drives generation as a sequence of
single-square mutations on the *actual intermediate board*, each emitting deltas
against the live `occ`:
- **EP:** remove EP pawn → `push_threats_on_change(remove)`; then move the
  capturing pawn (board.rs:867-885).
- **Capture:** remove victim → change(remove) before the mover moves.
- **Promotion:** move pawn → on_move; then remove pawn / put promo piece at `to`
  as two `on_change` calls (board.rs:888-900).
- **Castling:** king `on_move`, then rook `on_move` — two real single-piece moves
  against true intermediate states (board.rs:903-913).
Each sub-step's delta is computed against the board as it actually is at that
instant, so the composition equals the net position change. The `occ_transit =
occ ^ (1<<to)` trick and the `& occ` candidate filters (threats.rs:1124,
1443-1444) are the only subtlety, and they are explained + covered by tests.

### 5. Push/pop lifecycle is balanced and corruption-free

`push` (before make), `absorb_deltas` (after successful make), `pop` (after
unmake or failed make) are emitted at every move site paired 1:1 with the NNUE
accumulator push/pop (search.rs:4199-4211 is representative). `pop` only
decrements the index (saturating, threat_accum.rs:184-187); parent entries are
never mutated by `update`/`refresh` (they only *read* parents and write
entries in `(ancestor, index]`), so a pruned/aborted subtree leaves the parent's
values and `accurate` flags intact. Null moves push `NO_MOVE` and do not absorb,
so their entry stays `mv == NO_MOVE` → replay copies the parent (correct).

### 6. SMP / aarch64

The accumulator stack is **per-thread** (each `SearchInfo`/helper owns its
`threat_stack`, search.rs:895, 1612). The only cross-thread threat state is
`net.threat_weights` (read-only after load) and `THREAT_TABLES` (an `OnceLock`,
threats.rs:488, with Acquire/Release publish noted at threats.rs:640-642). No
shared mutable threat accumulator exists, so the aarch64 `Relaxed`-ordering
class of bug cannot arise here. Clean.

### Test battery that backs the above

| Test / mode | Location | What it pins | In `cargo test`? |
|---|---|---|---|
| `incremental_tests` curated scenarios | threat_accum.rs:447+ | each delta code branch (x-ray reveal, capture-reveals-xray, EP-in-ray, promo, castle-through-xray) element-wise | yes |
| `fuzz_random_games` | threat_accum.rs:793 | 5 FENs × 20 games × ≤120 plies, incr vs refresh element-wise both POVs, **gap==1** | yes |
| `fuzz-threats [--postfix]` | main.rs:1235 | inference `enumerate_threats` vs Bullet trainer port — **0/40000** | manual CLI |
| `CODA_VERIFY_NNUE=1` | search.rs:1168-1186 | **real search**, incremental eval vs `force_recompute` per node (covers lazy replay + pop) | manual env |
| debug assert in `evaluate_nnue` | eval.rs:190-223 | ThreatStack vs full recompute — **first 20 evals only** | debug builds, weak |

## Ranked findings

### F1 — MEDIUM (maintainability): dead parallel threat-accumulator in nnue.rs + dead `compute_move_deltas`

**Evidence.**
- `recompute_threats_if_needed` (nnue.rs:4890), `recompute_threats_full`
  (nnue.rs:5021), `verify_threats` (nnue.rs:4984), and the `self.threat`
  `AccDataStack` machinery are called **only from nnue.rs test code**
  (nnue.rs:7164/7743/7749). Production eval goes through
  `threat_accum::ThreatStack` + `forward_with_threats` (eval.rs:226,
  nnue.rs:4191), which reads the *passed* `threat_stack`, never `self.threat`.
- `verify_threats` has **zero callers** anywhere.
- `threats::compute_move_deltas` (threats.rs:1629, ~330 lines) has **zero
  callers** — the live generator is `push_threats_on_move`/`on_change`.
- The dead nnue.rs replay even documents a *weaker* rule than the live one:
  "mirroring might be wrong … king moves are rare in the chain … acceptable"
  (nnue.rs:4951-4954). The live `ThreatStack::update` is *exact* (see §3). So a
  future dev who "consolidates onto the nnue.rs path" would silently downgrade
  correctness, and tests that exercise `recompute_threats_if_needed` are
  validating a non-production path.

**Severity rationale.** No live bug today (dead code), but it is a latent
foot-gun: two implementations of the most bug-prone subsystem, one
test-referenced and weaker, that can drift apart unreviewed.

**Fix.** Delete `compute_move_deltas`, `verify_threats`, and the nnue.rs
`recompute_threats_if_needed`/`recompute_threats_full`/`self.threat` path (or, if
any nnue.rs unit test genuinely needs an independent oracle, re-point those tests
at `threat_accum::ThreatStack` so the test oracle *is* the production code). If a
field/stack is removed from `NNUEAccumulator`, confirm `push`/`materialize` no
longer maintain it.

**Test plan.** Pure dead-code removal. `cargo test` + `cargo build --release`
(zero warnings). `./coda bench` node count and bench value **must be byte-for-byte
unchanged**. No SPRT needed (no compiled behavior change); if paranoid, a
`[-2, 1]` non-regression run. **Bench impact:** none expected (may slightly
shrink `NNUEAccumulator` if `self.threat` is dropped — verify NPS-neutral).

### F2 — LOW (CI coverage gap): lazy-replay (gap ≥ 2) and pop/unmake are not in `cargo test`

**Evidence.** `fuzz_random_games` (threat_accum.rs:842-900) calls
`ensure_computed` **after every single ply** and `refs.refresh` every ply, so the
incremental side always replays from the immediately-prior (gap == 1) ancestor.
It also plays strictly forward — no `pop`, no skipped materialization. But the
production path is *lazy*: nodes push without computing, get pruned and popped,
and a later eval replays across a multi-ply gap (the findings doc measures ≈9.5%
full-recompute, chains > 1). The exact-mirror argument in §3 and the
parent-immutability argument in §5 are therefore **only machine-checked by the
opt-in `CODA_VERIFY_NNUE` mode**, which nobody runs in CI, plus a debug assert
**capped at 20 evals** (eval.rs:194 `c < 20`) that real searches blow past in
microseconds and that most unit tests never trigger (the fuzzers compare
accumulator vectors directly, never calling `evaluate_nnue`).

**Severity rationale.** The path is *argued* correct and has held in production
for months, but the cheapest possible regression net for it is absent from CI. A
future refactor of `can_update`/`update`/lazy-replay could regress silently until
an OB SPRT mysteriously loses Elo.

**Fix (cheap, pure test — no SPRT, no bench impact).** Add a `cargo test` that
exercises the lazy + pop path. Two options, ideally both:
1. **Extend `fuzz_random_games`:** with some RNG probability, *skip*
   `ensure_computed` on a ply (forcing the next eval to replay a gap ≥ 2 span),
   and occasionally `make`/`unmake` a random legal move then `pop` before
   continuing — calling `ensure_computed` + element-wise compare vs a fresh
   `refresh` at the points where it *is* materialized. This drives variable
   replay gaps, intermediate (within-half) king moves, and pop-restore.
2. **A perft-style verify walk:** recurse to depth ~4 from a handful of FENs
   (incl. a castling-rich and an x-ray-rich position), at each node
   push/make/absorb, call `ensure_computed`, and assert it equals a from-scratch
   `refresh` for both POVs; unmake/pop on the way up. This is exactly what
   `CODA_VERIFY_NNUE` does in-search but as a deterministic, bounded unit test.

Either makes the §3/§5 invariants machine-checked in CI. Also consider lifting
the eval.rs:194 `c < 20` cap behind a `CODA_VERIFY_THREATS_ALL` env so a debug
bench fully checks every eval when desired.

### F3 — INFO: pre-gating, asymmetry, king-exclusion are all confirmed non-bugs

- **Pre-gating** (`do_z_finding` threats.rs:1328, `& occ` 2b filter
  threats.rs:1443): the gates only skip work that *provably* emits nothing
  (table reads of `ray_extension`/`between` ∩ `occ`); they do not drop live
  threats. Closed negative as an NPS lever in
  `threat_generation_pregating_plan_2026-06-17.md`; from a *correctness* view the
  gating is sound and the fuzzers exercise the gated branches.
- **Color asymmetry** (`threat_eval_asymmetry_2026-06-17.md`): a deliberate
  STM-invariance tradeoff in the physical-square semi-exclusion, required for
  clean incremental deltas. Train/inference agree 0/40000. Not a bug.
- **King-attacker features** still tracked (`king_threat_exclusion_2026-05-22.md`)
  — an open *recipe* question, not a correctness one.

## Bottom line

The production threat delta path is sound: generation and recompute share
`threat_index`, semi-exclusion is symmetric and double-count-free, the
lazy-replay mirror is exact via `can_update`'s crossing guard, compound moves
decompose against real intermediate boards, push/pop is balanced and
parent-immutable, and the threat state is per-thread (SMP-clean). The
actionable work is hygiene, not bug-fixing: **(F1)** delete the dead parallel
nnue.rs/`compute_move_deltas` implementations (bench-identical, no SPRT), and
**(F2)** put the lazy-replay + pop invariants under a `cargo test` so the one
production behavior currently checked only by an opt-in env mode is in CI.
