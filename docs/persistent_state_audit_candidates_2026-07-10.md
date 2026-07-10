# Persistent-state latent-bug candidates — audit targets (2026-07-10)

Motivated by the corrhist raw-baseline bug (`docs/corrhist_audit_2026-07-08.md`
finding #1): a persistent, self-updating table whose update rule had a wrong
fixed point (railed at the clamp instead of converging), invisible in self-play
(+17±15 cross-engine vs ~0 self-play), which survived the 2026-06-30
full-codebase review and the 2026-06-26 corrhist deepaudit. It fell only to a
**pathology-repro + fixed-point analysis of the update dynamics + cross-engine
formula diff** of one subsystem.

This doc ranks where else that bug class could be hiding. Four parallel
surveys (history tables; TT + tb_cache; NNUE/threat incremental state;
game/TM/SMP lifecycle state) inventoried every structure that persists beyond
a node — across iterations, `go` commands, ponder transitions, games, and
helper threads. Line refs are against `origin/main` @ `a24466d`.

**The bug-class signature used for ranking:**
1. state persists (errors accumulate/compound);
2. self-updating with feedback (its consumers influence its own updates);
3. a wrong fixed point / wrong write-key / wrong baseline is *possible* (needs
   dynamics analysis, not line-by-line review);
4. invisible to self-play SPRT (symmetric, or Threads=1-invisible, or
   ponder-only);
5. no detector watching.

---

## Tier 1 — full signature match, concrete suspect already in hand

### C1. Live threat-accumulator drift is invisible to the runtime verifier

`CODA_VERIFY_NNUE` recomputes only the **PSQ** accumulator
(`search.rs:1496-1511` → `acc.force_recompute`) and then re-evaluates with the
**same, possibly-drifted `threat_stack`** — a threat desync produces identical
values on both sides of the compare and *passes verification*. The only live
threat parity check is `#[cfg(debug_assertions)]`, capped at the **first 20
evals**, and only `eprintln!`s (`eval.rs:240-273`). The threat-pipeline
deepaudit (2026-06-26) itself flagged that the randomized parity fuzzer never
exercises pop/unmake or lazy replay gap ≥ 2.

Why top-ranked: the threat pipeline has already produced two ~200-Elo parity
bugs (2026-04-17); a rare-position drift (mirror-crossing edge, missed
absorb on some control-flow exit — see C1b) would silently corrupt eval for
the remainder of the search **and poison every corrhist/history update made
under the wrong eval** — a persistent-state bug that *launders itself through
the learned tables*. No detector exists.

Sub-suspects from the survey, in order:
- **C1a — replay mirror span**: `update`/`update_dual` derive `mirrored` once
  from the final king square and apply it to every replayed ply
  (`threat_accum.rs:301-302,363-366`); correctness hinges entirely on
  `can_update`'s e-file-crossing rejection (`:278-285`) exactly matching the
  apply-time flip. Any king-move bookkeeping path not flagged `moved_pt==KING`
  for the right color breaks the whole chain.
- **C1b — push/absorb pairing**: a ply left as `NO_MOVE`+empty deltas where a
  real move was made replays as copy-from-parent (`threat_accum.rs:331-334`),
  corrupting all descendants until the next refresh. Audit every
  `push … pop` site in search for a matching absorb on *every* exit path.
- **C1c — dead second implementation**: a complete legacy threat subsystem
  inside `NNUEAccumulator` (`nnue.rs:5059-5236`) with *weaker* king/mirror
  invariants, sharing `apply_threat_deltas` with the live path. Resurrection
  landmine; candidate for deletion.

**Action (cheap, high-value):** extend `CODA_VERIFY_NNUE` to also recompute
threats from scratch and compare; run it over a long bench + overnight fuzz +
real games. Extend the fuzzer to replay gaps ≥ 2 with pops. Then the C1a/C1b
code audit. This is detector-building first, audit second.

### C2. `tm_cross_prev_score` — written by the wrong searches, consumed as game state (NEW code)

The cross-move TM trend term persists `last_score` across moves
(`search.rs:990`, written unconditionally at `:3626-3628` whenever
`completed_depth>=1`, consumed at `:3503-3512`, reset **only** on `ucinewgame`
`uci.rs:361`). Three pollution paths:
- **Ponder miss**: the ponder search runs on the *predicted* position and
  overwrites the scalar; after a miss, the next real move's time allocation is
  trended against a **sibling position's** score. Never reset on miss.
- **Analysis `go`s** (movetime/depth/infinite) between game moves reseed it.
- **`position` jump without `ucinewgame`** (lichess-bot-shaped path) carries it
  into an unrelated game.

Exactly the corrhist shape: a persistent value whose write-site doesn't match
its consume-site's position identity. Ponder is deployment-critical and
OB-invisible → test via the TM methodology (mechanism inspect + ponder RR),
not SPRT alone.

### C3. TT sticky `tt_pv` — absorbing fixed point, no decay; generation gates nothing

- `tt_pv = is_pv || (tt_hit && tt_entry.tt_pv)` (`search.rs:3949`), re-stamped
  on essentially every store (`:5812`, ProbCut `:4701`). Feedback loop:
  tt_pv → LMR reduced (`:5228`) → searched deeper → stays PV → re-stamped.
  **No decay path** — the fixed point is absorbing-true, and the bit survives
  across moves and games (generation never gates it). Same dynamics question
  as corrhist railing: does the field (SF included) decay or lapse this bit
  anywhere we don't? Needs the cross-engine formula diff.
- **Generation is diagnostic-only**: cutoffs, static-eval reuse,
  window-narrowing, and near-miss all fire on cross-game entries
  (`search.rs:3938,6035` only count stats). A `position`-jump without
  `ucinewgame` trusts prior-game bounds/evals on any Zobrist match.
- Adjacent drift channel: **synthesized scores are re-stored as facts** —
  near-miss `tt_score ± 80` (`:4146-4153`), TT-damp (`:4133`), fail-high blend
  (`:5917`) all write blended values that later probes treat as ground truth.
  Also: `downgrade_50mr_mate` is applied at the main cutoff returns but **not**
  at the near-miss and TT-damp returns.

The 2026-06-19 TT audit was replacement/stub-focused; this is a different,
dynamics-level lens on the same structure.

---

## Tier 2 — feedback loops in learning tables, likely by-design, never dynamics-checked

### C4. TT-cutoff history bonus for a move that was never searched

`search.rs:4098-4122`: a TT lower-bound cutoff bonuses main/capture history
for `tt_move` **without searching it**. A stale/biased TT entry (which itself
persists across moves/games, see C3) keeps re-bonusing its move; the bonus
feeds ordering, LMR, and the futility strong-history exemption, making future
cutoffs on that move likelier. TT→history→search→TT loop with no fresh search
evidence in it. Cross-engine diff needed: who bonuses on TT cutoff, at what
scale, with what gates (Coda already down-weights via `TT_CUT_BONUS_PCT` —
verify the *gating* matches the field, not just the scale).

### C5. Cont-hist gravity decays toward a *foreign, moving* baseline

`update_cont_history_with_base` (`movepicker.rs:139-144`, used at
`search.rs:5506/5644/5690`): gravity divisor uses `base = cur_cont +
main_score/2`, so the cell's fixed point is `±MAX − main_score/2` — tied to a
*different persistent table that is itself drifting*, and main history enters
both the ordering score AND the cont-hist equilibrium (table coupling). This
is the documented Stormphrax T6 pattern, but it is the strongest structural
rhyme with "wrong baseline → shifted fixed point" in the codebase. Worth an
explicit convergence check + formula-level diff against Stormphrax/the field
(verify our transcription of T6 is exact, including which term carries the
gravity).

### C6. Beta-cutoff history writes are not gated on singular-exclusion searches

The whole beta-cutoff update block (`search.rs:5580-5764`) lacks an
`excluded_move` guard — corrhist updates have one, history updates don't.
Verification searches (reduced depth, one move artificially excluded) write
bonus/malus into persistent main/cont/pawn/capture history. Some engines do
update here, so this may be consensus — needs the cross-engine gate diff
before touching.

### C7. SMP pool worker history survives `ucinewgame` (+ the forgotten-table contract)

Independently flagged by two surveys. `ucinewgame` clears only the main
thread's tables (`uci.rs:357-362`); the process-global pool
(`thread_pool.rs:76`) is untouched, and `refresh_helper_per_go` **ages**
each worker's own main/capture/cont history in place (`search.rs:2128`)
rather than recopying from the just-cleared main. Previous game's move-ordering
history leaks into helpers across games. Threads=1 OB SPRT is structurally
blind to it; it affects lichess deployment directly.

Same area, structural: `refresh_helper_common` (`search.rs:2082-2108`) is the
spot where the "forgotten table" pattern recurs (trans_corr was once missed).
All six tables are accounted for **today**; recommend an invariant test that
fails when a new `SearchInfo` table field isn't classified
(copy/clear/age/share) in the refresh split.

---

## Tier 3 — lower priority / cheap monitors

- **C8. 50MR-blind TT bounds**: an entry stored at clock 10 is trusted at
  clock 80 (gate is `halfmove < 90`, `search.rs:3964,4046,6046,6229`); only
  *mate* scores get downgraded. Known engine-wide GHI class; Reckless's
  16-bucket hmc keying (corrhist audit #3) is the same idea applied to the TT
  question. Cheap probe: fortress/shuffling repro with TT on/off.
- **C9. Malus asymmetry from tried-move caps**: `quiets_tried[64]` /
  `captures_tried[32]` — moves beyond the cap get no malus while the cutoff
  move always gets its bonus; slow history inflation in wide nodes.
- **C10. aarch64 dotprod/i8mm L1 kernels have no scalar-parity test**
  (`nnue.rs:2274-2360`), and the splat threat enumerator is x86_64-only so ARM
  runs a different (scalar) threat path than what most testing exercises.
  Parity-test debt, first-class-platform policy says close it.
- **C11. Finny refresh silent 32-cap**: `refresh_accumulator` buffers cap at
  32 with silent truncation and no inaccurate-marking (`nnue.rs:5628-5631`) —
  can't overflow with legal chess today, but it's the same silent-drop pattern
  the DeltaVec `overflowed` flag was added to fix elsewhere. Add the flag or a
  debug assert.

## Surveyed and came back clean (don't re-audit without new evidence)

- **tb_cache**: halfmove is in the key (C2 fix), full 64-bit XOR verify,
  castling pre-declined, Release/Acquire pairing correct.
- **Repetition/key history**: fully rebuilt from `position … moves` each
  command; draw checks run before TT probe.
- **4D main-history threat keys**: read/write symmetric, recomputed fresh per
  node (post-C8fix); evasion picker uses the same key.
- **Mate/TB score TT round-trip**: ply adjustment symmetric at every store and
  probe site incl. qsearch; ambiguous TB scores never stored.
- **Helper seeding completeness**: all six persistent tables accounted for
  today (but see C7 for the contract).
- **Ponder atomics/budget state**: re-zeroed both before spawn and at search
  start; the double-ponderhit guard blocks the stale-read paths checked.

## Methodology note

What actually caught corrhist #1 (and should be the template for C1–C5):
1. a **runtime pathology repro** first (fortress FEN drifting under corrhist);
2. **fixed-point analysis** of the update dynamics, not line review;
3. **cross-engine formula diff** of the exact update rule against the top-6;
4. **set-once env A/B toggles** to isolate the mechanism empirically.

And the measurement lesson: eval-bias/persistent-state fixes **undersell in
self-play** (+17±15 cross-engine vs ~0 self-play for the corrhist fix). C2 and
C7 are additionally OB-invisible (ponder / Threads>1) — plan local cross-engine
RRs and mechanism probes, not just SPRT.
