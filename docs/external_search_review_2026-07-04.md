# External / Driver Search Review — 2026-07-04

Companion to `docs/search_review_2026-07-01.md` (which audited the *interior*
search: move-loop pruning, LMR, history, TT slot mechanics). This pass covers
the **driver / root layer** — the part we'd given least attention: the
iterative-deepening loop, aspiration windows, root-move handling, PV/best-move
plumbing, the Lazy-SMP driver, TT interactions *across* iterations, and
UCI/ponder reporting.

Method: 6 parallel finder agents, one per cluster, each deep-reading Coda's
implementation and comparing against the strongest 6 engines (Stockfish,
Reckless, Obsidian, Berserk, PlentyChess, Alexandria),
checking `experiments.md` for priors. All findings below were **re-verified by
hand** before inclusion — the finders read code well but their prior-checking
was imperfect (see the false alarm in §TT).

---

## Headline: the external search is clean

**Zero correctness bugs across all six clusters.** SMP memory ordering is
correct (the ARM sweep held), the ponder/best-move pairing is hardened better
than anything else in Coda (triple defense), the aspiration loop is provably
window-safe and terminating, and TT generation/reuse/`ucinewgame` handling is
all correct. The one "alarming" finding (a TT age-weight regression) is a
**false alarm** — verified below.

This is a genuinely useful result: **the ~100-Elo gap to the top is not hiding
in the driver layer.** Most of it was already swept by the TM audits, the
2026-07-02 P0 wave (SMP thread-select, is_decisive), and this week's P1.4
bank-aborted + P2.1 hoist. What remains here is a short list of small P3 probes,
P4 cleanups, and two architecture investments (RootMoves array, persistent
thread pool) whose payoff is infrastructure/deployment, not this-week Elo.

---

## The false alarm (verified dead) — TT age-weight `*8`

The TT-driver finder's P2 headline claimed `slot_depth - age*8` (`tt.rs:498`,
commit `134d0dd`) shipped to prod "without a controlled test" and contradicts a
documented `-70 Elo` age-6 result. **Both premises fail on verification:**

- The `-70 Elo` age-6 result is dated **2026-03-12** — inside the *unreliable
  pre-2026-04-01 gauntlet era* (124 games, narrow 3-engine gauntlet; the exact
  false-result class the top of `experiments.md` warns about).
- The `*8` change **was** properly self-play SPRT'd later: **#842 STC +0.5/273K**
  (neutral, not a regression) and **#857 LTC +1.6 ±2.5 H1 ✓**, **merged
  2026-04-29** as the methodology-correct call (LTC is the right test for
  TT-pressure changes — `feedback_sprt_blind_to_long_game_effects`).

So age-8 is a validated, mildly-positive prod value. No action. (One real
sub-nit survives: the commit message's "matches Reckless" is factually wrong —
Reckless is 4/gen; SF is 8/gen. Doc-only.)

**Lesson carried forward:** treat every finder "novel/untested" tag as
unverified until the full prior chain (including post-2026-04-01 supersessions)
is checked by hand.

---

## Ranked — worth considering

### OB-visible, cheap probes (the only two fireable Elo candidates)

**1. TT overwrite margin 3 → 4** — P3, novel. `tt.rs:478`
`depth > slot_depth - 3` requires a new same-key entry to be within 2 ply of the
resident to overwrite; **all four peers use margin 4** (SF `d+2pv > depth8-4`,
Berserk `depth+4 > TTDepth`, Reckless `depth+4+2pv <= entry.depth`). Coda
preserves deep same-key entries strictly harder than every reference. Loosen to
`- 4`. Skip the SF/Reckless `+2*pv` half (adverse prior #692, PV-preservation
bonus H0'd). Test `[-1.5,1.5]` STC. Low magnitude, but genuinely untested.

**2. Asymmetric aspiration delta growth** — P3, novel. `search.rs:2509` (+`:2168`)
`delta += delta/2` is symmetric for both fail directions. **Reckless splits
them**: fail-low ≈1.21× (gentle — an unstable search shouldn't overshoot into a
huge window), fail-high ≈1.49× (fast — confirm the improvement quickly). This is
*distinct* from the symmetric-rate tests that H0'd (#83 1.33×, #2116 2.0×) —
none varied the two directions independently. Two fixed-point tunables, STC
`[0,3]`, bench-neutral. Low-med confidence (Reckless-only among the 6; Coda's
symmetric rate is already SPSA-blessed).

### Output-only, no Elo but useful to us

**3. Emit `lowerbound`/`upperbound` + intermediate progress info lines** — P3.
On an aspiration fail-high/low, SF and Reckless emit a bounded info line; Coda
prints one line per *completed* depth only. Output-only, bench-neutral —
**benefits our own TM PGN tooling** (the per-move-spend / mechanism probes) as
much as GUIs. No SPRT needed.

### Architecture investments (infrastructure, not this-week Elo)

**4. Persistent RootMoves array** — P2 infra. `search.rs` root handling. Coda has
no cross-iteration root-move list; ply-0 ordering is rebuilt each iteration from
the TT move + generic MovePicker, whereas 6/6 references keep a stable-sorted
per-move list (score/prev_score/avg/pv). ~150-250 LOC, cheap struct. Justify by
**consumers**, not direct Elo: enables the average-score window, MultiPV, SMP
best-thread tiebreak, and clean per-root-move effort accounting (currently the
node-fraction TM signal is a `from*64+to` hash — see nit below). Build only when
a consumer needs it.

**5. Persistent thread pool** — P2, **SMP-only (OB-invisible at Threads=1)**.
`search.rs:1918-1714`. Coda respawns helpers per `go` with a full history/corr/
NNUE alloc-and-copy, and **discards helper-learned history each move**. Reckless
keeps a persistent pool where per-thread history persists+ages — two separable
levers: per-move overhead *and* a genuine Lazy-SMP diversity source Coda lacks.
Only matters when codabot runs multi-threaded; validate via local multi-thread
RR (STC for overhead, LTC+ponder for diversity), not OB.

### P4 cleanups (bench-neutral, no SPRT — batch them)

- Extract the triplicated best-move-from-PV validation loop into a helper
  (`id_driver F3`).
- Extract the aspiration inner loop shared by `search()` and `search_helper()` —
  currently copy-pasted, must be edited in lockstep (`aspiration F5`).
- `root_move_nodes` keys `from*64+to` → **promotion collision** in the TM
  subtree factor; 1-line fix to include the promo flag (`id_driver F4` /
  `root F3`). Also accumulated before the stop check + ungated by
  `excluded_move[0]` during forced-move detection (minor TM-signal noise).
- `info.root_depth` missing from the per-search reset block (latent-harmless,
  same class as the documented `tm_max_time` A4 nit) (`id_driver F5`).
- `pv_len[ply] = 1 + child_len` can overstate vs `copy_len` near MAX_PLY (inert;
  one-word `1 + copy_len` restores the invariant) (`pv_ponder F3`).
- Correct commit `134d0dd`'s "matches Reckless" note (Reckless = 4/gen).

### Process

- Institutionalize a standing **local ponder RR + `PV_PONDER_BUG` grep** as a
  pre-merge gate (generalize the vote-override mechanism probe). The entire
  ponder-pairing stack is OB-invisible and has only ever regressed as lichess
  forfeits (oeZ7KRUt). Validate-don't-change (`pv_ponder F5`).

---

## Deferred / low / dead (recorded so they aren't re-audited)

- **Aspiration running-average center** — DEAD, #2503 H0 −0.4 (both `id_driver F1`
  and `root F4` re-flagged it; the aspiration finder correctly identified it as
  tested). The delta² running-mean half is covered by the same #2503.
- **asp_depth restore-to-full on fail-low** — likely DEAD: this is the
  #1935 (−0.9) / #1944 (−4.6) reset consensus-port. `id_driver F2` claims those
  had an outer-variable scoping bug making them an unclean test; unverified and
  low-value either way — Coda's monotonic `asp_depth--` is calibrated into the
  contraction shapes. Only revisit if the scoping-bug claim is confirmed in code.
- **Full-window snap on decisive fail-high** (Berserk/Plenty) — P3 deployment
  only (mate latency), OB can't adjudicate; #3213 already judged mates too rare
  at STC. Low.
- **Aspiration width (`root_delta`) into LMR** — cross-cluster; defer to a future
  LMR pass (retune-on-branch, tree-shape change).
- **Helper per-thread window jitter** (SF `threadIdx%8`) — SMP-only, not
  OB-measurable; cheap-to-add nicety, hard to justify fleet.
- **Thread-agreement soft-stop vote** (Reckless 65%) — P3 TM-class, SMP; needs a
  new Acquire/Release atomic + cross-engine RR.
- **TT generation-refresh on probed/kept entries** (Plenty/Berserk; SF/Reckless
  don't — 2/4 split) — P4 speculative; interacts with age-weight, easy wash.
- **Expose `TT_AGE_WEIGHT`/`TT_OVERWRITE_MARGIN` as tunables** — gated on a probe
  showing real sensitivity; loose-knob risk past the ~74-knob threshold. The
  age-weight is validated at 8, so its motivation is weak; only the margin
  (Item 1) might justify it.
- **MultiPV, `currmove`/`currmovenumber`** — analysis features, zero Elo; MultiPV
  is a RootMoves-array consumer if ever wanted.

---

## Honest assessment

The driver layer is in good shape — no bugs, and the obvious consensus ports
have almost all already H0'd (aspiration especially is a near-closed, 8-priors-
deep axis). The two OB-visible probes here (TT margin 3→4, asymmetric aspiration
growth) are low-magnitude long shots worth a cheap STC punt if fleet is idle,
but nobody should expect more than a fraction of an Elo. The genuine leverage
this audit *confirms* (by not finding it here) is elsewhere: the RootMoves-array
architecture if we want MultiPV/better-SMP/avg-window, the persistent thread pool
for multi-threaded deployment, and — the standing conclusion — net retraining +
TM/ponder deployment work, not another single-knob search pass. The most
valuable concrete output may be the **process gate** (ponder RR + PV_PONDER_BUG
probe): the one place a real, silent, OB-invisible regression can still ship.
