# Board-Mechanics Audit — 2026-06-13

Six-track parallel audit of the board layer: board.rs (make/unmake, struct,
FEN), zobrist.rs/zobrist_keys.rs + incremental keys, attacks.rs/bitboard.rs
(+ board.rs attack helpers), threats.rs/threat_accum.rs (v9 threat layer),
types.rs/see.rs/encoding discipline, plus a perf pass from search's view
and a verification-infrastructure meta-audit. Cross-engine references:
Stockfish, Reckless, Viridithas, Obsidian, Berserk + cozy-chess (Rust
technique only). EXCLUDED (covered by other recent audits): movegen.rs,
tt.rs, nnue.rs FT internals, movepicker internals.

Line numbers reference main @ time of audit. Two headline claims were
independently verified on this machine before filing (B1, B2).

## Profiling frame (sets expectations)

bench-12 profile: NNUE + threat machinery ~55% of cycles (ThreatStack::
update 27%, L1 pairwise 14.7%); the whole named board family ~9%
(see_ge 2.2%, make_move 1.4%, attackers_to 1.3%, is_legal 0.75%, pinned
0.67%). The full board-layer perf basket is worth ~1-3% NPS (~1-2 Elo),
NOT 10%. Correctness findings dominate this audit's expected value.

## B — Bug-class (verified or high-confidence)

### B1. SEE in-loop promotion handling is algebraically wrong — VERIFIED sign flip
`see.rs:93-109`, introduced as "C8 #40" (49a0c18, 2026-04-22, bench -5.2%).
The swap-algorithm update `balance = -balance - 1 - value(attacker)`
already encodes in-loop promotion EXACTLY with the plain pawn value: the
promoting side gains (Q-P) and risks Q, netting -P, and the negation next
iteration credits the recapturer +Q while charging the (Q-P) gain. The C8
change uses see_value(QUEEN) as the at-risk value WITHOUT crediting the
(Q-P) gain — the promoting side is ~1100cp too pessimistic, propagating
as ~+1100cp optimism for the opponent each subsequent iteration.

VERIFIED counterexample (the position experiments.md:13186 said couldn't
be constructed): FEN `3R2k1/4P3/3q4/3r4/8/8/8/6K1 b - - 0 1`, Qd6xd8.
Sequence QxR, exd8=Q, Rxd8: white loses R+P (740), black loses Q (1200)
→ true value -460. Engine output: `SEE from=43 to=59 val=640 ge0=true`.
Sign flip at threshold 0 and every SEE-pruning threshold in use. The
motif (back-rank piece defended by a promoting 7th-rank pawn, initiator
has a backup attacker) is COMMON: misclassified as a good capture in
movepicker ordering, passes the QS SEE gate, ProbCut SEE>=0, and escapes
SEE capture pruning. No reference engine special-cases in-loop promotion
(SF position.cpp:1467 `swap = PawnValue - swap`; Reckless plain
`attacker.value()`; Berserk same) — because the algebra makes
it unnecessary.
**Fix:** revert the hunk to `let effective_value = see_value(att_pt);`
+ add the two assertive FEN tests (-460 and +100 cases). Bench shifts
(~5%); SPRT [0,3]. SEE-cluster tunables were retuned on the broken
semantics since April → focused SEE retune on branch if vanilla SPRT
is flat.

### B2. setwise.rs cfg overlap breaks the build off the native-CPU path — VERIFIED
`setwise.rs:62-66` (`#[cfg(not(target_feature = "avx2"))]`) vs `:122-130`
(`#[cfg(target_arch = "x86_64")]`) — not mutually exclusive; x86_64
without AVX2 in the compile baseline defines `knight_attacks_setwise`
twice. VERIFIED: `RUSTFLAGS="" cargo check --release` → E0428. Masked
only by .cargo/config.toml's `-Ctarget-cpu=native` (and a user RUSTFLAGS
env OVERRIDES config.toml entirely). Any generic-target or pre-AVX2
x86_64 build cannot compile. The bishop/rook wrappers use the correct
exclusive pattern; the knight one is the odd one out.
**Fix:** change :62 to `#[cfg(not(target_arch = "x86_64"))]`. No-op
under native (verify bench identical); no SPRT.

### B3. set_fen accepts garbage EP squares → unmake-without-make reachable
`board.rs:423-433` (parse validates bounds only), `:92-114`
(ep_capture_available never checks the pushed pawn EXISTS). Corrupt FEN
`ep=e6` with no black pawn on e5 but a white pawn on d5: phantom ep_key
in hash; movegen generates the EP capture; make_move rejects it; and
generate_legal_moves' EP-verify path calls make_move/unmake_move WITHOUT
checking make's return — unmake pops an undo entry that was never pushed.
Unreachable from legal play; reachable from the UCI/FEN boundary
(lichess, cutechess, EPD files). SF/Berserk/Viridithas all sanitize EP
at parse.
**Fix:** sanitize in set_fen (correct rank + pushed pawn exists + both
squares empty, else clear); optionally add the pushed-pawn check to
ep_capture_available. cargo test + perft + new corrupt-FEN test. Bench
expected identical.

### B4. NNUE parity tests silently pass-as-no-op when no net file exists
nnue.rs:7496-7509 etc. — fuzz_psq_accumulator and both Finny consistency
tests `eprintln!("Skipping...")` and return OK when no .nnue is present.
VERIFIED on this machine: no net.nnue in tree → "4 passed in 0.06s",
all vacuous. This layer's historical bugs (AVX-512 pairwise threat drop,
aarch64 Finny no-op, king-bucket SMP race) are exactly the silent-eval
class these tests exist to catch. threat_accum tests build synthetic
weights and never skip — the pattern to copy.
**Fix:** synthetic in-memory net (~30-line test ctor); interim: panic
on skip when CODA_REQUIRE_NET/CI env set.

## S — Strength-relevant experiment candidates

### S1. non_pawn_key excludes the king — corrhist is blind to king placement
`board.rs:196-227` (`else if pt != KING`), `:461-471`. SF, Reckless, AND
Viridithas all include the king in nonPawnKey; Coda is the outlier. With
pawn-corr (no king), np-corr (no king), minor/major dropped, cont-corr
(last move only): NO corrhist source keys on king location — two
positions differing only in king placement share every corrhist bucket.
Secondary: the 2026-05-18 minor/major-corrhist drop (#1318 "strict
subsets of non_pawn_key, redundant") was conditioned on the king-less
construction — in SF/Viri the minor key is NOT a subset precisely
because king membership differs.
**Fix:** include KING in non_pawn_key (3 sites + set_fen loop). Main
hash unaffected (TT-safe); only corrhist bucketing changes. SPRT [0,3];
CORR_W_NP retune candidate. NOTE: coordinate with atlas/corrhist-overhaul
(in flight) — natural follow-on or addition to that branch after its
SPSA settles.

### S2. EP-hash legality refinement (pinned-EP repetition blindness)
`board.rs:929-931` + 3 mirror sites. EP hashed on geometric adjacency;
SF additionally requires the EP capture be LEGAL (pins/discovery) —
added to fix incorrect 3-fold handling. Coda mid-pack (Berserk/Obsidian
same as us; Reckless laxer). ~0 Elo; correctness-only [-3,3], bundle.

### S3. Leaf-dispatch restructuring (depth<=0 enters full negamax preamble)
Every depth-0 child runs ~380 lines of negamax preamble (draw checks,
MDP, attack/xray setup, TT probe) then QS REPEATS draw checks + TT
probe. SF/Reckless dispatch qsearch at the CALL SITE. Node-changing
(depth-0 negamax provides MDP/TB/cuckoo/nearmiss that QS lacks) →
[0,3] probe. CAUTION: #1947 (the threat-hoist H0) showed this preamble
work partially hides TT-probe latency — the conservative variant is
call-site dispatch with the preamble's cheap work retained.

## P — Perf basket (bundle as one bench-identical [-2,1] branch)

Realistic total: ~1-3% NPS. Items, by expected value:
1. **QS pinned/checkers dedup**: new_quiescence (movepicker.rs:332-333)
   recomputes both — the 2026-06-11 legality fix introduced it; #1923
   fixed the main picker but missed QS + probcut's QMovePicker
   (movepicker.rs:1244-1245). Pass the locals in. Bit-identical.
2. **ATTACK_TABLE static-mut-Vec → fixed array** (attacks.rs:66): every
   slider lookup pays Vec indirection + un-elidable bounds check. Size
   is compile-time constant (107,648 = SF's footprint). get_unchecked
   with documented invariant.
3. **Child checkers passed down**: parent computes child's checkers via
   gives_check (search.rs:3952), child recomputes at entry. KEEP the
   parent-side compute (hides child-TT prefetch latency per #1947);
   delete the child-side recompute by passing it down. The dead
   UndoInfo.checkers field (board.rs:23, written 0, never read) is the
   fossil of this fix — either use it or delete it (-8 bytes/entry).
4. **xray_blockers O(sliders×blockers) → classic 2-lookup x-ray form**
   (board.rs:614-661): per slider, attacks(occ), attacks(occ^blockers),
   between() per revealed victim. Bit-equivalence test vs current.
   Do NOT move its call-site position (#1947).
5. **Repetition scan starts at i=2** (search.rs:2751): repetition
   impossible until i=4 (SF starts at 4). One wasted load/compare per
   node. Also UndoInfo is 56B/record and scans touch one cache line per
   probe — hash-only side array deferred (low value).
6. **checking_sqs unification** (separate [0,3], NOT in the bundle):
   three pruning carve-outs call gives_direct_check per candidate while
   the movepicker builds SF-style checking_sqs per node 10 lines away.
   Pre-move vs post-move occupancy differs in one rare geometry → node-
   changing.
7. **Dead threat code deletion** (threats.rs:1296-1628 compute_move_deltas,
   332 lines, zero callers; nnue.rs legacy threat machinery ~200 lines +
   0.8-1 MB/thread allocated even for v5 nets, carries a WEAKER mirror-
   handling variant of an already-fixed bug class). Delete both; redirect
   3 tests to ThreatStack. Memory win ~12-16 MB at T=16.

aarch64 note: setwise sliders fall back to per-square magic loops on ARM
(no NEON Kogge-Stone) — M-series/Graviton lose the batched-attacks win.
Optional NEON port when ARM NPS work is scheduled.

## V — Verification-infrastructure gaps (the meta-audit)

Ranked by (bug-class severity × historical frequency — 402 fix-commits,
incl. 320 Elo of is_pseudo_legal holes, 6 threat-layer fixes, the C8
cross-repo feature mismatch that cost a net generation):

1. **B4 above** (vacuous NNUE parity tests) — worst.
2. **debug-build assert_consistent()** after make/unmake/null (mailbox↔
   bitboards, hash==recompute, aux keys, king counts) — SF pos_is_ok /
   Viridithas check_validity equivalent; Coda has NONE in production
   paths. Turns every debug bench/perft/test into a corruption oracle.
   Zero release cost.
3. **Multi-ply walkback fuzz**: BOTH parity fuzzers (threat + NNUE)
   materialize every ply → replay distance pinned at 1; production
   chains hit 8. The 2026-04-17 king-crossing bug lived exactly there.
   Fix: materialize ~1/3 of plies + random pop/re-push walk + null-move
   pushes (three gaps, one fuzzer rework).
4. **Training/inference threat-parity as cargo test**: the C8 mismatch
   class (cost a net generation) is guarded only by a manual CLI fuzzer.
   ~200-position enumerate_threats vs bullet_postfix_ref test, <1s.
5. **is_pseudo_legal exhaustive oracle** (cozy-chess pattern): all 65536
   encodings × ~20 FENs vs generated-move set. Current fuzz only mutates
   legal moves — TT collisions are uniform garbage, the unprobed region.
6. **gives_direct_check property test** (piggyback on existing fuzzer:
   claims-check ⇒ checkers!=0 after make) + curated castle/promo/EP FENs.
7. **TalkChess tricky perft set** (9 positions, EP-discovered-check,
   castle-rights-after-rook-capture etc.) — CPW-6 is decent but peers
   run deeper + FRC.
8. **Hostile-FEN table test** (pins down set_fen's implicit contract;
   pairs with B3).
9. **No CI** — nothing runs any of this automatically; interacts
   multiplicatively with B4 (CI must `make net` or use synthetic nets).
10. **Net-load validation**: loader never compares header
    num_threat_features against the engine's table (~66,864) — wrong-
    count net loads cleanly and silently drops high-index features. One
    if-statement. Also: refresh >256-feature overflow truncates eval
    with no actual fallback (threat_accum.rs:208-245; histogram label
    claims "[forced fallback]" — none exists); bump buffer to 512.

## Q — Quality / hygiene (batch, no SPRT)

- static mut CASTLE_MASK → const (board.rs:59-78); castle rook from/to
  mapping hand-duplicated 6× with the named consts DEAD (board.rs:55) —
  one helper; EP-recording restructure (record ep_square only when
  capturable → 4-site mirrored hash condition becomes unconditional —
  consensus structure, deletes a fuzzer-policed invariant).
- unmake corruption-bail should restore scalar state (free, board.rs:985).
- FEN nits: fullmove 0 accepted (u16 wrap), empty-FEN early-return leaves
  half-cleared board.
- SEE: king-illegality branch unreachable (see.rs:111-124, restructure
  Reckless-style or document); stale piece-value comments in see.rs:275 +
  CLAUDE.md (says P=100,N=320,B=330,R=500,Q=900; actual 100/420/420/640/
  1200); see_value_of binary-search range ±2000 < max possible 2300.
- types.rs: debug_asserts for promotion_piece_type (underflows on
  non-promo), piece_color (NO_PIECE→OOB), square_bb (64→silently a1).
  is_promotion accepts corrupt flags 8-15 (held only by is_pseudo_legal)
  — document or tighten.
- Zobrist: castle rights hashed as XOR of 4 per-right keys (linear over
  GF(2); 16 independent keys is consensus — harmless, fix only if ever
  regenerating); gen_zobrist2.go generator absent from repo; stale test
  mirrors (check_repetition_main missing plies_from_null cap + null-
  boundary break, stale line refs); CLAUDE.md pawn-hist says 8192,
  actual 512.
- ThreatStack::push(mv, moved_pt) params dead (all callers pass
  NO_MOVE) + stale moved_color trap — drop params or reset.
- init_bitboards stale comment; line()/ray_extension() zero unit tests;
  no slow-vs-fast magic parity property test.

## Verified clean (high-confidence good news)

- Zobrist key material: statistically clean (no zero/dup keys, balanced
  popcounts, ZERO 4-term GF(2) dependences over 305K pair-XORs, full
  rank in every low-bit window the small consumers use — pawn_hist &511,
  corrhist &16383, TT, cuckoo h1/h2). Incremental updates airtight, all
  three keys, all move shapes, fuzz-pinned.
- Castling-rights dual-mask bookkeeping consensus-exact (incl. rook
  captured on home square); make_move reject paths all run before any
  mutation; unmake restores every field (11-field fuzz).
- Cuckoo: byte-for-byte SF algorithm, 3668 entries asserted, key
  construction consistent.
- EP-hash 4-site mirror consistent today + regression-tested (B3/Q
  restructure removes the fragility).
- see_ge core: SF/Berserk consensus shape exactly (early exits, x-ray
  re-adds, EP victim, attribution) — B1 is the ONLY deviation.
- Move-flag ==-not-& discipline: 100% compliant across src/.
- pinned/checkers/attackers_to/between/line: algorithms correct,
  edge cases covered; attacks_by_color setwise is genuinely SF-class
  with proper parity tests.
- Magic/PEXT runtime dispatch (branch on static bool + Zen1/2 fast-PEXT
  CPUID exclusion) is BETTER than SF's compile-time scheme for a
  heterogeneous fleet; per-lookup cost near-zero. Table footprint = SF.
- Threat layer: can_update validate-before-accept correct; null-move
  handling correct; net-swap state handling correct; SIMD kernels
  scalar-parity-tested across widths; push/pop hardened; the 12,000-ply
  fuzzer + 15 curated scenarios is above-par armor (modulo the V3
  coverage-shape gap).
- Repetition-by-hash with no piece validation: universal practice
  (64-bit full-width, unlike TT's 32-bit) — not a gap.
- Board::clone() sites: not per-node; fine.

## Suggested execution order

1. **B1 SEE revert** + assertive tests → SPRT [0,3] (+ SEE-cluster
   retune if flat). Highest expected value in the audit.
2. **B2 cfg fix** (one line, build correctness, no SPRT) + **B3 FEN
   sanitization** + **B4 synthetic-net tests** as a correctness batch.
3. **V2 assert_consistent + V3 fuzzer rework + V4 parity test + V5
   oracle** — one verification branch, debug/test-only, no SPRT.
4. **S1 king-in-non_pawn_key** [0,3] (after corrhist-overhaul settles).
5. **P bundle** [-2,1] (items 1-5, 7), checking_sqs and leaf-dispatch
   as separate [0,3] probes.
6. **Q batch** opportunistically.
