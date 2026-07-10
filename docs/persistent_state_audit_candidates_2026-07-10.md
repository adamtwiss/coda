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

---

# Tier-1 investigation results (2026-07-10, same day)

Three parallel deep investigations on `origin/main` @ `a24466d`. Verdicts:

## C1 — threat verifier blind spot: claims TRUE, live pipeline CLEAN; gap closed

All three claims verified exactly as stated (verifier recomputed PSQ only and
re-evaluated with the same threat stack; live threat parity check was
debug-only/20-eval-capped/eprintln-only; fuzzer was forward-only with replay
gap 1 everywhere).

A detector was built (CODA_VERIFY_NNUE extended to recompute threat features
from scratch, both perspectives, into local buffers; `CODA_VERIFY_THREATS=panic`
escalates), validated by fault injection, and run with panic armed over
**~8.3M evals** (bench d12 + d16, WAC 201@500ms, adversarial
underpromotion/EP/castling/king-march/mirror-boundary searches): **zero
mismatches**. Code audit of the two flagged mechanisms:

- **push/absorb pairing — CLEAN.** Every threat push/make site absorbs
  immediately after make on every control-flow path; null-move NO_MOVE
  copy-forward is correct. One fragile no-push make/unmake window exists
  (`tt_cutoff_child_disagrees`) — safe today because nothing evals inside it;
  the detector would catch any future violation.
- **mirror-span — PROVEN CORRECT.** `can_update` validates exactly the replay
  span; any pov-king e-file crossing (incl. O-O-O; O-O correctly exempt)
  forces refresh, so deriving `mirrored` from the final king square is valid
  for every replayed ply. Per-perspective mirrors handled independently.

A new fuzzer (`fuzz_random_walk_with_pops_and_lazy_gaps`) now exercises pops,
null plies, and replay gaps up to 10 (2400+ gap>=2 events, coverage-asserted);
full suite 202 passed. **Detector + fuzzer pushed as
`test/threat-verifier-coverage` (bench-neutral 2085296, env-gated/test-only)**
— recommend merging as permanent CI/diagnostic coverage. C1 verdict:
**latent diagnostic gap, no live bug; gap permanently closed by the branch.**

## C2 — tm_cross_prev_score pollution: CONFIRMED live deployment bug, reproduced

All claims verified, plus two new findings:
- **TB scores are unguarded**: the Factor-5 guard uses `is_mate_score`
  (|s| >= 28872); TB scores (~28600-28800) slip under it. `is_decisive()`
  exists for exactly this and is unused here.
- **Book/TB-root moves skip search()** entirely, so the scalar goes stale by
  multiple plies (same position-identity family).

Empirical repro (worktree `wt-c2`, `scripts/c2_tm_cross_repro.py` preserved on
this branch): after pondering a queen-hanging predicted reply (score +1210)
and missing, the next real move's Factor-5 consumed +1210 instead of the true
previous score +46 — **pinned at the 1.55 ceiling for the entire search
(+55% opt budget)**. Analysis-`go` pollution reproduced identically.

Severity: OB-invisible by construction (no ponder, no interleaved analysis,
ucinewgame per game — CROSS_MOVE_TREND was tuned in a clean-scalar regime).
On lichess (~36% miss rate) roughly one move in three trends against a
sibling position, and the distortion correlates adversely (pondered score
diverges most exactly when the opponent avoids the prediction). Overspend-
biased (+55% cap vs -20% floor); opt-only, no forfeit vector.

**Recommended fix shape** (not implemented; rebases cleanly under FL-EXT):
1. Gate the *publish* — write `tm_cross_prev_score` only from clock-managed
   real-game searches (skip never-hit ponder and analysis go's). This
   preserves the true previous score across a miss (strictly better than
   resetting). Matches SF's `previousScore` discipline.
2. Widen both guards from `is_mate_score()` to `is_decisive()`.
3. Optional: sentinel-reset on the book/TB-root `continue` paths.
Validation per TM methodology: repro script above (mechanism), then
ponder-on cross-engine RR; SPRT non-regression only.

## C3 — TT dynamics cross-engine diff: tt_pv suspicion DROPPED; two new divergent suspects found

Full 6-engine formula extraction (SF dev-20260402, Reckless 0.10-dev,
Obsidian v16, Berserk 13, PlentyChess b-7.0.22, Alexandria 9). Ranked:

1. **Near-miss margin cutoff — DIVERGENT, strongest finding.** 0/6 references
   accept a shallower entry with a score margin; SF/Reckless/Obsidian demand
   one ply MORE on the fail-high side. Coda returns `tt_score -+ 80` from a
   depth-(d-1) entry and the parent stores the synthetic value as a
   full-depth bound — a depth-laundering ratchet no reference permits.
   Candidate: SPRT removing the near-miss branch (search.rs ~4139-4152).
2. **TT-damp blended cutoff return — DIVERGENT.** 6/6 return raw `tt_score`
   from the TT-cutoff path. The field's dampening lives at the fail-high
   STORE site (SF/Reckless/Plenty store the dampened value; Coda stores raw
   and dampens only the return at 5910-5918 — inverted placement).
   Candidate: return raw on LOWER-cutoff; optionally move FH blend pre-store.
3. **Mate/TB downgrade scope — DIVERGENT vs top-3.** The halfmove<90 cutoff
   gate is consensus (Coda even stricter: gates QS too, field doesn't). But
   SF/Reckless/Plenty downgrade at EVERY TT read (inside score_from_tt) and
   cover the TB band (`TB_WIN - v > 100 - hmc`); Coda downgrades mate-band
   only, at 3 return sites, so singular/ProbCut/window-narrowing consume
   potentially-false TB scores. Candidate: extend downgrade_50mr_mate with
   the TB clause + move into score_from_tt. (Note: the original C3 sub-claim
   that near-miss/TT-damp returns miss the downgrade was WRONG — both are
   !is_decisive-gated, mate scores can't reach them.)
4. **Sticky tt_pv — CONSENSUS, suspicion dropped.** 6/6 absorbing, 0/6 decay;
   SF/Reckless are stickier than Coda (fail-low parent propagation). Real
   residual deltas: (a) Coda's QS stores hardcode tt_pv=false — the only
   true->false lapse path in the survey (all six preserve the bit through
   qsearch); (b) SF/Reckless/Plenty condition the LMR tt_pv credit on entry
   quality (ttValue>alpha, ttDepth>=depth) where Coda's -1.00 ply is flat
   (Obsidian/Berserk/Alexandria flat too — MIXED); (c) 5/6 give PV entries a
   +2*pv-class store/replacement bonus; Coda and Berserk don't.
5. **Generation semantics — CONSENSUS, dropped.** 0/6 gate read-time trust on
   age; replacement-victim scoring only (Coda's depth-8*age matches
   SF/Obsidian exactly).
6. **ProbCut store — CONSENSUS, dropped.** depth-3/LOWER/sticky ttPv matches
   SF/Obsidian/Alexandria exactly.

## Consolidated next actions

| # | Action | Class | Test |
|---|--------|-------|------|
| 1 | Merge `test/threat-verifier-coverage` | test/diagnostic, bench-neutral | no SPRT needed |
| 2 | Fix tm_cross_prev_score publish gate + is_decisive | TM bug fix, ponder-class | repro + ponder RR + [-2,1] non-reg |
| 3 | Remove TT near-miss margin branch | divergent-from-field simplification | SPRT (removal = [0,3] on the removal or [-2,1] as cleanup) |
| 4 | Raw return on TT LOWER-cutoff (drop TT-damp) | divergent-from-field | SPRT |
| 5 | TB-band 50mr downgrade in score_from_tt | correctness, top-3 pattern | [-2,1] |
| 6 | QS stores: pass sticky tt_pv instead of false | one-liner, field-unanimous | [-1,2] candidate (complexity-free, validated mechanism) |
| 7 | Condition LMR tt_pv credit on entry quality | MIXED-consensus enrichment | [0,3] + tune |

Tier-2 candidates (C4-C7) not yet investigated.
