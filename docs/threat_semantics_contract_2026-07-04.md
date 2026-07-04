# Coda Threat-Feature Semantics — Correctness Contract for a SIMD Reimplementation

Extracted 2026-07-04 from `main` @ ba6271c (plus branch commit 5096ac0 for the
splat parity test). All line numbers are from the current working tree unless
marked otherwise. An error in this document becomes silent eval corruption —
every claim below is cited to code.

Primary sources:
- `src/threats.rs` (2971 lines) — enumeration, index mapping, delta generation, apply kernels
- `src/board.rs` `make_move` (lines 844–1003) — delta generation call sites
- `src/threat_accum.rs` — the live consumer (ThreatStack)
- `docs/byteboard_splat_scoping_2026-05-03.md` — prior port attempt; the x-ray blocker
- `git show 5096ac0:src/threats_splat.rs` — abandoned AVX-512 splat + its parity test

---

## 1. The feature set: what is an ACTIVE threat feature?

### 1.1 The tuple

A threat feature is identified by the 4-tuple
**`(attacker_cp, from, victim_cp, to)`** where:

- `attacker_cp`, `victim_cp` ∈ 0..12 colored-piece indices:
  `0=WP 1=WN 2=WB 3=WR 4=WQ 5=WK 6=BP … 11=BK` (`colored_piece`,
  threats.rs:533–537).
- `from`, `to` are **physical board squares** (0..63, a1=0), pre-any-flip.
- The tuple is active iff the piece `attacker_cp` on `from` *threatens* the
  piece `victim_cp` on `to`, where "threatens" = **attacks directly OR attacks
  through exactly one blocker (x-ray depth 1)**. Ground truth is
  `enumerate_threats` (threats.rs:721–821).

Crucially, **the feature index does NOT distinguish direct from x-ray**
(there is no x-ray flag). The same index means "WR@a1 threatens WB@c1"
whether the attack is direct or through one blocker. This is the exact
property that killed the Reckless byteboard-splat port
(`docs/byteboard_splat_scoping_2026-05-03.md` lines 199–215): in Reckless the
index means "directly attacks" only; in Coda a piece interposing between a
slider and its target does NOT change the feature (it flips direct→x-ray,
same index), so a Reckless-style enumerator double-counts in Coda's model.

Total feature space: **~66,864** indices (threats.rs:8; asserted 60k–70k in
`test_init_threats`, threats.rs:2709–2716; matches the `--threats 66864`
convert-bullet flag).

### 1.2 Direct threats (enumerate_threats "section 1" analogue)

threats.rs:743–761. For every piece (both colors, all 6 types) compute
`piece_attacks_occ(pt, color, sq, occ)` (real occupancy; magic bitboards for
sliders) and emit a feature for every **occupied** attacked square whose
mailbox entry is a real piece (`victim_pt < 6`). All six piece types generate
direct threats, including kings and pawns. Victims of both colors (defence of
own pieces is a feature too), subject to the pair filter in §1.4.

### 1.3 X-ray threats — the exact rule

threats.rs:763–817 (enumerate); the incremental analogues are sections 1b /
2-Z / 2b of `push_threats_for_piece` (§3.3).

**Rule, precisely:**
- Only **sliders** (B, R, Q) generate x-ray threats
  (`if pt == BISHOP || pt == ROOK || pt == QUEEN`, threats.rs:766).
- For each **directly attacked occupied square** (i.e. each first-hit piece
  per ray — magic attacks stop at the first blocker, so there is at most one
  per ray direction), remove that blocker from occupancy and take the
  **closest newly-revealed occupied square** on the same ray
  (`revealed = attacks_through & !attacks & occ_without`, threats.rs:779;
  closest-in-ray-direction selection at 792–803). That revealed piece is the
  x-ray victim.
- **Exactly one blocker.** A slider attacking through TWO or more blockers is
  NOT a feature. In the incremental code this is explicit:
  `blockers_count != 1 → skip` (threats.rs:1505–1517, comment 1448–1452:
  "sliders with 2+ blockers are 2+ level x-rays not encoded in the feature
  set").
- **The blocker may be ANY piece — own or enemy, any type.** No filtering of
  the blocker whatsoever (threats.rs:769–777 removes whatever occupies the
  square). Example from the scoping doc: in the start position
  `(WR@a1 → WB@c1)` through own knight `WN@b1` is an active feature
  (byteboard_splat_scoping lines 199–203).
- **The victim may be either color**, same pair-filter as direct threats.
- The x-ray victim square always lies on the attacker's **empty-board** attack
  ray (needed for the index mapping, §1.5).
- A slider can hold **both** a direct feature (first piece on ray) and an
  x-ray feature (second piece on ray) simultaneously — two distinct features
  (different `to`).

Boundary hazard for a rewrite (threats.rs:784–803): at `blocker_sq == 63` the
naive `1u64 << (blocker_sq + 1)` is UB; current code guards it. The `revealed
== 0` check happens to filter that case but is documented as fragile.

### 1.4 The pair filter (excluded attacker×victim types)

`PIECE_INTERACTION_MAP` (threats.rs:450–457), rows = attacker type, cols =
victim type, -1 = excluded:

| attacker \ victim | P | N | B | R | Q | K |
|---|---|---|---|---|---|---|
| Pawn   | 0 | 1 | — | 2 | — | — |
| Knight | 0 | 1 | 2 | 3 | 4 | — |
| Bishop | 0 | 1 | 2 | 3 | — | — |
| Rook   | 0 | 1 | 2 | 3 | — | — |
| Queen  | 0 | 1 | 2 | 3 | 4 | — |
| King   | 0 | 1 | 2 | 3 | — | — |

So: **the king is never a victim**; queens are victims only of knights and
queens; pawns do not threaten bishops/queens/kings. Per-attacker victim-type
counts × 2 colors give `PIECE_TARGET_COUNT = [6, 10, 8, 8, 10, 8]`
(threats.rs:459–461).

Excluded tuples are dropped at index-expansion time: `threat_index` returns a
negative value and the apply/refresh paths skip it (threats.rs:707–710,
1732, 234). **The raw delta generator does NOT filter these** — it emits
tuples for all victim types and lets `threat_index` reject them.

### 1.5 The index mapping (`threat_index`, threats.rs:679–719)

```
threat_index(attacker_cp, from, victim_cp, to, mirrored, pov):
  1. POV remap:  attacking = (attacker_cp + 6) % 12 if pov==BLACK else attacker_cp
                 attacked  = (victim_cp  + 6) % 12 if pov==BLACK else victim_cp
     (table frame: slots 0–5 = POV's own pieces)
  2. pair  = piece_pair[attacking][attacked]     // precomputed base + flags
  3. base  = pair.base(from, to)                 // *** PHYSICAL squares — see §2 ***
     if base < 0 → excluded/semi-excluded, skip
  4. flip  = (7 * mirrored) ^ (56 * pov)         // horizontal ^ vertical
     from_f = from ^ flip;  to_f = to ^ flip
  5. index = base
           + piece_offset[attacking][from_f]              // cum. attack-square count below from_f
           + attack_index[attacking][from_f][to_f]        // rank of to_f in EMPTY-BOARD attack set of attacking@from_f
```

Precomputed tables (`init_threats`, threats.rs:586–672, published via
OnceLock):
- `piece_offset[cp][sq]` — cumulative empty-board attack-square popcount over
  all squares below `sq` for colored piece `cp`. Pawns on ranks 1/8 contribute
  no attack squares (threats.rs:606–609).
- `attack_index[cp][from][to]` — `popcount(below_mask(to) & empty_board_attacks(cp, from))`
  (threats.rs:650–660). Because x-ray victims lie on empty-board rays, every
  valid tuple has a well-defined slot. Queens use bishop|rook empty attacks.
- `base` per pair = `offset_table[attacking_cp] +
  (attacked_color * PIECE_TARGET_COUNT[pt]/2 + map) * piece_offset_total[attacking_cp]`
  (threats.rs:633–636) — i.e. within each attacker's block, victims are
  ordered [table-frame-white victims by map bucket][table-frame-black victims].

The mapping is **injective** over valid tuples: distinct (attacker_cp, from,
victim_cp, to) → distinct index. Two different physical tuples never collide
on one index for a given (pov, mirrored).

`mirrored` is true when the **perspective's own king** is on files e–h:
`(king_sq % 8) >= 4` (threat_accum.rs:211–212, 301–302, 363–366).

---

## 2. The semi-exclusion rule (Bullet semi-exclusion, physical-square)

Applies to **same-piece-type pairs**. Setup in `init_threats`
(threats.rs:638–644):

```rust
let semi_excluded = attacking_pt == attacked_pt
    && (enemy || attacking_pt != 0);  // pawn-pawn same color NOT semi-excluded
```

- Same-type, enemy color: semi-excluded (N–N, B–B, R–R, Q–Q, P–P enemy pairs).
- Same-type, same color, non-pawn: semi-excluded.
- Same-color pawn–pawn (a pawn defending a pawn): **NOT** semi-excluded —
  both orderings are distinct features (mutual attack is impossible for
  same-direction pawns, so no redundancy).
- King–king is fully excluded by the map anyway.

Mechanics (`PiecePair::base`, threats.rs:485–489): the base has bit 30 set
for semi-excluded pairs; `base(from, to)` adds `((from as u8) < (to as u8)) << 30`,
so a semi-excluded tuple with `from < to` carries into bit 31 and returns
negative → **the kept ordering is `attacker_sq >= victim_sq`** (matches
Reckless; comment threats.rs:484–487). Rationale: for symmetric same-type
attacks (if A attacks B, B attacks A — true for all non-pawn same-type pairs,
including in the x-ray sense), only one of the two mirror tuples is kept,
halving redundant features.

**The comparison uses PHYSICAL squares** — `pair.base(from, to)` is fed the
raw, un-flipped squares (threats.rs:702–707: "Semi-exclusion uses PHYSICAL
squares to match Bullet training"), and likewise in
`apply_threat_deltas_dual_body` (`pair_w.base(from, to)` / `pair_b.base(from,
to)` at threats.rs:1856, 1877 — note NOT `from_w/to_w`). Consequences:

- Both perspectives (and both mirror states) make the **same** keep/skip
  decision per tuple → the decision is STM-invariant and mirror-invariant,
  which is what keeps incremental deltas valid across POV/mirror expansion.
- It is deliberately **NOT mirror-symmetric**: a horizontally mirrored
  position can keep the *other* ordering of a same-type pair, which is the
  documented ~17–70cp color-eval asymmetry on bishop/rook same-type threats
  (threats.rs:836–841; `docs/threat_eval_asymmetry_2026-06-17.md`; memory
  "Threat color-asymmetry is NOT a bug"). **A reimplementation must NOT
  "fix" this** — train==inference is verified at 0 mismatches by
  `fuzz-threats --postfix` (threats.rs:974–976; 40,000 evals, both STMs,
  2026-06-17). The historical bf-frame variant that differed is retained
  only as `enumerate_threats_bullet_ref` (threats.rs:823–927) to
  characterize the OLD pre-C8fix bug.

Fully-excluded pairs (map == -1) have bit 31 set unconditionally and always
return negative.

---

## 3. The delta contract (make_move → RawThreatDelta records)

### 3.1 Generation call sites in `make_move` (board.rs:844–1003)

Gated on `board.generate_threat_deltas` (set true when the loaded net has
threats: search.rs:2113, 2189; also by datagen/tests). Sequence, in order:

1. `threat_deltas.clear()` at the top of every make_move (board.rs:912–913).
2. **EP capture**: `remove_piece(them, PAWN, cap_sq)` then
   `push_threats_on_change(…, them, PAWN, cap_sq, add=false)`
   (board.rs:916–922). Note `cap_sq != to`.
3. **Normal capture**: `remove_piece` then
   `push_threats_on_change(…, them, captured, to, add=false)`
   (board.rs:923–928).
4. **The mover** (always): `move_piece(us, pt, from, to)` then
   `push_threats_on_move(…, us, pt, from, to)` (board.rs:930–934).
5. **Promotion**: remove pawn at `to`, put promo piece at `to`, then TWO
   change calls: `(us, PAWN, to, add=false)` and `(us, promo_pt, to, add=true)`
   (board.rs:936–949).
6. **Castle**: rook `move_piece` then a second
   `push_threats_on_move(…, us, ROOK, rook_from, rook_to)` (board.rs:951–962).
7. **Null move**: `make_null_move` just clears `threat_deltas`
   (board.rs:1100–1101) — zero deltas; consumers copy the parent accumulator.

**Board state at call time (load-bearing for a rewrite):** every push call
runs on the **post-mutation** board (pieces_bb/colors/mailbox already
updated), with occupancy passed as `colors[0]|colors[1]` at that instant.
So during the capture-removal push the mover is still on `from`; during the
castle king-leg the rook is still on `rook_from`.

### 3.2 The two generator entry points

- `push_threats_on_move` (threats.rs:1130–1152): computes
  `occ_transit = occ ^ (1 << to)`. Since `occ` already has `from` cleared and
  `to` set (post-move_piece), **occ_transit has BOTH from and to empty** —
  the moving piece is absent from occupancy for both legs. Then:
  - `push_threats_for_piece(…, occ_transit, …, from, add=false)` (remove leg)
  - `push_threats_for_piece(…, occ_transit, …, to,   add=true)`  (add leg)
- `push_threats_on_change` (threats.rs:1154–1169): single
  `push_threats_for_piece(…, occ, …, square, add)` with occupancy as passed
  (for removals the piece is already out of occ; for the promotion-add the
  piece is in occ — harmless since attacks-from-square ignore the source
  square's own occupancy bit).

Because the moved piece sits in `pieces_bb`/`mailbox` at `to` while
`occ_transit` has `to` clear, all attacker-side scans mask candidates with
`& occ` to avoid phantom self-candidates — see the section-2b comment
(threats.rs:1460–1464) and regression test
`castling_queenside_phantom_xray_regression` (threat_accum.rs:734–747).

### 3.3 `push_threats_for_piece` — the 5 sections (threats.rs:1187–1633)

Semantics: emit every threat feature that appears/disappears **because the
piece `cp` appears (add=true) / disappears (add=false) on `square`**, given
the surrounding occupancy `occ`. Five sections:

**Section 1 — direct threats FROM this piece** (threats.rs:1211–1220).
`piece_attacks_occ(piece_type, color, square, occ) & occ`; for each occupied
target emit `(cp, square, victim, target, add)`.

**Section 1b — x-rays FROM this piece** (threats.rs:1229–1304), sliders
only. For each direct target (= per-ray first blocker), read
`ray_extension(square, blocker_sq) & occ` (precomputed table of squares
strictly beyond the blocker on the from→blocker ray, bitboard.rs:145–154);
the first occupant in ray direction (lowest bit if `square < blocker_sq`,
else highest) is the x-ray victim: emit `(cp, square, xvictim, xray_sq, add)`.

**Section 2 — sliders that see this square, + Z-level bookkeeping**
(threats.rs:1322–1425). `rook_attacks(square,occ)` / `bishop_attacks(square,occ)`
identify sliders directly attacking `square` (set-membership via
`pieces_bb[R|Q] & rook_att`, `pieces_bb[B|Q] & bishop_att`, masked `& occ`).
Per such slider S:
- **Z-delta** (only if the shared cull `do_z_finding` fires — some ray from
  `square` has an occupant beyond its first hit, threats.rs:1346–1350): let
  Y = first occupant past `square` on S's ray (`ray_extension(slider_sq,
  square) & occ`), Z = first occupant past Y (`ray_extension(slider_sq, y_sq)
  & occ`). Emit `(S_cp, slider_sq, Z_cp, z_sq, !add)` (threats.rs:1376–1411).
  Reason: when a piece APPEARS on `square`, S's feature for Y is unchanged
  (direct→x-ray, same index — the exact "same index" property of §1.1), but
  S's x-ray to Z (previously through Y as the single blocker) is LOST because
  x-ray depth is only 1; on disappearance Z is GAINED. Hence the inverted
  sign `!add`.
- **Direct delta**: `(S_cp, slider_sq, cp, square, add)` — S now
  attacks / no longer attacks the piece at `square` (threats.rs:1414–1417).
  (Skippable via the `CODA_NO_SLIDER_SEES` ablation env var, threats.rs:1171–1180
  — NOT bit-identical, test-only.)

**Section 2b — sliders whose X-RAY target is `square`** (threats.rs:1427–1580).
Candidates = rook/queen on `rook_attacks_empty(square)` rays plus bishop/queen
on `bishop_attacks_empty(square)` rays, masked `& occ` (threats.rs:1465–1467).
Per candidate S: count occupants strictly between S and `square`
(`between(s_sq, square) & occ`):
- count == 0 → S is a direct attacker; section 2's business; skip.
- count >= 2 → depth-2+ x-ray, not in the feature space; skip.
- count == 1 (the 2b case):
  - Emit `(S_cp, s_sq, cp, square, add)` — S's x-ray onto the
    appearing/disappearing piece (threats.rs:1536–1539).
  - **W-delta**: W = first occupant past `square` continuing away from S
    (`ray_extension(s_sq, square) & occ`); if present emit
    `(S_cp, s_sq, W_cp, w_sq, !add)` (threats.rs:1550–1567). Reason: when the
    piece appears on `square`, S's x-ray target on that ray shifts from W to
    `square` (S already had one blocker Y; the new piece becomes the depth-1
    x-ray victim and W drops to depth 2); on disappearance it shifts back.

**Section 3 — non-slider attackers of `square`** (threats.rs:1606–1620).
Reverse pawn/knight/king attack table lookups
(`pawn_attacks(WHITE, square)` selects black pawns and vice versa); per
attacker emit `(ns_cp, ns_sq, cp, square, add)`.

### 3.4 Per-move-shape delta inventory

`push_threats_for_piece` call count per move (each call = one full 5-section
pass on one square):

| move shape | calls | breakdown |
|---|---|---|
| quiet / double push | 2 | on_move(from−, to+) |
| capture | 3 | change(victim@to, −) + on_move |
| en passant | 3 | change(pawn@cap_sq, −) + on_move (cap_sq = to∓8) |
| castle | 4 | on_move(king) + on_move(rook) |
| promotion | 4 | on_move(pawn) + change(pawn@to,−) + change(promo@to,+) |
| promotion-capture | 5 | change(victim,−) + the promotion 4 |
| null move | 0 | deltas cleared only |

Double push has no special handling (no EP-square deltas; EP is a hash/state
concept, not a threat feature). There is NO special-casing of king moves in
generation — a king move emits normal raw-square deltas; mirror handling is
entirely the consumer's job (§4).

Promotion transient: the pawn's on_move add-leg emits attacker→pawn@to /
pawn@to→victim tuples that the subsequent `change(pawn@to, add=false)` emits
again with the opposite sign — guaranteed add+sub pairs that net to zero.
(A white pawn ON rank 8 has an empty `pawn_attacks` set so section 1 emits
nothing, but incoming-attacker sections do fire.)

### 3.5 Order and duplication guarantees

- **Order is NOT semantically significant.** The apply path partitions deltas
  into an adds array and a subs array and applies weight-row sums
  (threats.rs:1717–1749); i16 addition is commutative. Emission order is:
  victim-change → mover-from-leg → mover-to-leg → (promo changes | rook legs),
  and within a leg: section 1, 1b, 2 (per slider: Z-delta then direct),
  2b (per slider: square-emit then W-delta), 3. Preserve at will for cache
  reasons only.
- **The contract is NET-COUNT equality, not set equality.** For every feature
  index i (per pov/mirror expansion):
  `#adds(i) − #subs(i) == active_post(i) − active_pre(i) ∈ {−1, 0, +1}`,
  where `active_*` is the 0/1 enumerate_threats membership. The diagnostic
  tests explicitly net out add/sub pairs before comparing
  (threat_accum.rs:1170–1177).
- **Same feature as add+sub in one move: YES, routinely.** Promotions
  guarantee it (§3.4); slider-ray interactions produce it (e.g. capture of a
  piece on a slider's ray: removal leg emits `(S, Z, +)`, the mover's add-leg
  re-emits `(S, Z, −)`). Measured: ~3.8% of streamed weight rows are
  cancellable same-index add/sub pairs (threats.rs:35–39, 178).
- **Same feature twice as add (or twice as sub): does not occur by
  construction.** Within one `push_threats_for_piece` leg each section emits
  distinct tuples (one delta per slider per role; sections mutually exclusive
  per slider by blocker count 0/1/2+). Across legs, all repeated tuples appear
  with opposite signs (the second leg sees the effect of the first change in
  occ). A rewrite may rely on `|net| ≤ 1`; it must NOT emit a net of ±2.
- **No dedup is performed anywhere** — the apply kernels blindly sum rows, so
  any net-count error corrupts the accumulator silently.

### 3.6 Consumers

Live path (search): `SearchInfo::threat_stack` (`ThreatStack`,
threat_accum.rs). Protocol per node:
1. `threat_stack.push(…)` **before** `board.make_move` (search.rs:3985 etc.;
   threat_accum.rs:165–177 — resets accurate flags, clears the entry's delta).
2. `board.make_move(mv)` populates `board.threat_deltas`.
3. `threat_stack.absorb_deltas(&board)` copies `board.threat_deltas` into the
   ply entry (`DeltaVec::copy_from_slice`, capped at 128 with overflow flag)
   and records `mv` / `moved_pt` (mailbox at `to`, i.e. post-promotion type) /
   `moved_color` (threat_accum.rs:149–163; search.rs:3992, 4422, 5342, 5560).
4. On unmake: `threat_stack.pop()` (index decrement only — values above the
   index are stale-but-reusable ancestors for lazy replay).
5. At eval: `ensure_computed` (threat_accum.rs:417–444) → per-perspective
   `can_update` walk-back (§4) → `update`/`update_dual` replay or `refresh`.

A parallel copy of the same protocol exists inside `NNUEState`
(nnue.rs:4567–5087, `store_threat_deltas` at 4807, replay at 4894–4982) with
`threat_accurate` flags per stack entry; it follows identical semantics
(swap the board Vec instead of copying).

`board.threat_deltas` lifecycle: `Vec` (capacity 128, board.rs:131),
cleared at the top of make_move / make_null_move, appended during make_move,
**not touched by unmake_move** — it is only meaningful immediately after a
make_move and must be absorbed before the next make/unmake.

---

## 4. King moves, mirroring, and the full-refresh bail

- **Deltas are mirror-agnostic and POV-agnostic**: `RawThreatDelta` stores
  physical squares and absolute colored pieces. POV remap + mirror flip happen
  only at index-expansion time. In `apply_threat_deltas_dual_body`
  (threats.rs:1814–1911): `flip_w = 7 * mirrored_w`,
  `flip_b = (7 * mirrored_b) ^ 56` (threats.rs:1845–1846); white expansion
  uses `piece_pair[attacker][victim]` with `from^flip_w / to^flip_w`
  (1855–1862); black expansion first remaps piece colors via
  `flipped_colored_piece` (`[6,7,8,9,10,11,0,…,5]`, threats.rs:511–517) then
  flips squares with `flip_b` (1874–1883). Semi-exclusion (`pair.base(from,
  to)`) uses the UNflipped physical squares in both (1856, 1877). The
  single-perspective path routes through `threat_index` with the same math
  (threats.rs:1723–1731).
- **Incremental bail on king e-file crossing, per perspective**:
  `can_update(pov)` (threat_accum.rs:260–293) walks back from the current ply
  looking for an accurate ancestor, but validates the move at each
  intervening ply FIRST: if `moved_pt == KING && moved_color == pov` and
  `(from % 8 >= 4) != (to % 8 >= 4)` (the king crossed the d/e file
  boundary), return `None` → **full `refresh`** for that perspective
  (enumerate + `add_weight_rows`, threat_accum.rs:203–255). Reason: stored
  raw deltas below that ply would expand with the wrong `mirrored`. The
  check-order bug (accepting an accurate ancestor before validating the
  crossing at the current ply) was caught by the fuzzer on 2026-04-17
  (comment threat_accum.rs:264–272). Note O-O-O crosses (e1→c1: file 4→2);
  O-O does not (e1→g1: 4→6). The OTHER perspective replays incrementally
  through the same king move.
- Replay also bails (returns None → refresh) if any intervening ply's
  DeltaVec `overflowed()` (threat_accum.rs:275–277).
- `mirrored` at expansion time is computed from the CURRENT board's king of
  that perspective (threat_accum.rs:301–302, 363–366) — valid for the whole
  replayed span precisely because king-crossing plies force a refresh.
- Null-move / empty-delta plies copy the parent accumulator verbatim
  (threat_accum.rs:331–334, 381–385).

---

## 5. The oracle: how to verify a new enumerator

**Ground truth** is `enumerate_threats` (threats.rs:721–821): the active
feature SET of a position for a given (pov, mirrored). It emits each active
index exactly once (geometric uniqueness; direct and x-ray to distinct
squares).

Existing test layers, all reusable:

1. **Delta-level parity vs scalar (per single change)** — the pattern used by
   the abandoned splat port: `splat_change_parity_with_scalar` in
   `git show 5096ac0:src/threats_splat.rs` (tests module, ~line 516+). For 6
   FENs × every square × every applicable piece, run scalar
   `push_threats_on_change` and the SIMD version, sort both delta lists by
   raw u32 and require exact multiset equality
   (`assert_delta_sets_equal`, sorts + symmetric-difference dump). Phase A
   deliberately skipped slider focus pieces (sections 1b/2-Z not implemented
   in the splat). NOTE: exact-multiset is STRICTER than the semantic
   contract (§3.5 net-count); a new enumerator that legitimately emits a
   different but net-equivalent multiset should compare **net counts per
   tuple** instead — but matching the exact multiset is the easiest way to
   prove equivalence and inherit all downstream guarantees.
2. **Accumulator end-to-end parity** — `threat_accum.rs` `incremental_tests`
   (threat_accum.rs:447–1105+): `run_scenario` (523–584) drives two
   ThreatStacks over a move list, one incremental (absorb + ensure_computed)
   and one always-refreshed, comparing all H=768 channels element-wise per
   ply per perspective, using deterministic weights that make every
   (feature, channel) distinct (`make_weights`, 472–483 — a single-feature
   multiset divergence is guaranteed visible). 15 curated scenarios cover:
   quiet, captures ± x-ray, EP (`en_passant_capture`, 711–720), O-O
   (722–731), O-O-O phantom-x-ray regression (733–747), slider-reveals-slider
   (749–758), capture chains, promotion-capture (771–779). Plus
   `fuzz_random_games` (792–903): 5 start FENs × 20 deterministic-PRNG games
   × ≤120 plies, same incremental-vs-refresh assert per ply. **This suite,
   with the new generator wired into make_move, is the primary acceptance
   test.**
3. **Multiset-diff diagnostics** — ignored tests `dump_diff_*`
   (threat_accum.rs:921–1202) reconstruct `pre = post − adds + subs` and
   print missing/extra indices, netting add/sub pairs first
   (1146–1151, 1170–1177) — the debugging tool when layer 2 fires.
4. **Train/inference parity** — `coda fuzz-threats` CLI
   (main.rs:1346–1449): random positions, compares `enumerate_threats` index
   SET vs the Bullet reference walk (`--postfix` →
   `enumerate_threats_bullet_postfix_ref`, threats.rs:976–1099; expected
   **0 mismatches**). Only relevant if `enumerate_threats` or `threat_index`
   themselves are touched — a delta-generator rewrite that leaves them alone
   doesn't move this.
5. **Apply-kernel parity** — threats.rs tests mod (2481+):
   `apply_deltas_scalar_ref` vs AVX2/AVX-512/NEON kernels, and
   `add_weight_rows` equivalents. Untouched by an enumeration rewrite.

Recommended recipe for the SIMD port: (a) unit parity per
`push_threats_for_piece` call (exact multiset, all piece types incl.
sliders, add and remove, over the 6 parity FENs + kiwipete); (b) per-move
parity of `board.threat_deltas` (net-count per tuple) over the
fuzz_random_games move streams; (c) full layer-2 suite; (d) `cargo test`
green + bench-identical node count (the change must be bit-identical, so
SPRT class is `[-2, 1]` NPS-only per CLAUDE.md).

---

## 6. Data structures

### RawThreatDelta (threats.rs:1101–1126)

Packed u32, 4 bytes, layout identical to Reckless's ThreatDelta:

| bits | field |
|---|---|
| 0–7 | `attacker_cp` (0..12) |
| 8–15 | `from_sq` (0..64) |
| 16–23 | `victim_cp` (0..12) |
| 24–30 | `to_sq` (7 bits, 0..64) |
| 31 | `add` (1 = add, 0 = sub) |

`RawThreatDelta::ZERO = Self(0)`; constructor masks `to_sq & 0x7F`.

### MAX_THREAT_DELTAS and overflow (threats.rs:1102; threat_accum.rs:26–82)

`MAX_THREAT_DELTAS = 128` per ply. `board.threat_deltas` itself is an
**unbounded Vec** — generation never truncates. The cap is enforced at
absorb: `DeltaVec::push` beyond 128 sets `overflowed = true` and drops
(threat_accum.rs:51–59); `copy_from_slice` truncates to 128 and flags if
`src.len() > 128` (62–69). Overflow does NOT corrupt: `can_update` returns
`None` when any replay ply overflowed (threat_accum.rs:275–277) → **full
refresh** fallback. Cap hits are ~0% in practice (apply_stats CAP_HIT bucket,
threats.rs:33, 128, 153).

Refresh has its own cap: 256 active indices; overflow leaves
`accurate[p] = false` so the next read refreshes again rather than
compounding on a truncated baseline (threat_accum.rs:217–254 — the "C8 audit
LIKELY #18" fix).

### ThreatEntry / ThreatStack (threat_accum.rs:84–199)

`#[repr(C, align(64))]` per ply: `values: [[i16; MAX_FT_SIZE=1024]; 2]`
(per-perspective accumulators; prod v9 uses h=768), `accurate: [bool; 2]`,
`delta: DeltaVec`, `mv`, `moved_pt`, `moved_color`. Stack pre-allocated
MAX_PLY=256; `pop` saturates at 0.

---

## 7. Complexity inventory (what the scalar code computes; SIMD targets)

Given targets: `push_threats_for_piece` ≈ 5–6% of search cycles (delta
GENERATION), index expansion (the `threat_index` table chase inside
apply/dual) ≈ 3–6%.

### Per `push_threats_for_piece` call (any focus piece)

| section | work | notes |
|---|---|---|
| 1 direct-from | 1 attack lookup: magic (B/R), 2 magics (Q), table (P/N/K); popcount-loop over `attacks & occ` | per victim: 1 mailbox read + 1 white_bb test |
| 1b own-xray | sliders only; per direct target: 1 `ray_extension` table read (`[64][64]` u64) + lsb/msb pick + mailbox read | replaced per-blocker magic re-lookup (comment threats.rs:1229–1234) |
| 2 sliders-see | 2 magics from `square` (rook+bishop w/ occ) + 2 empty-board lookups (shared with 2b); per seeing-slider: mailbox read + (if `do_z_finding`) up to 2 `ray_extension` reads (Y then Z) + 1 direct emit | Z-cull: `occ & (empty_rays & !queen_att) != 0` gates all Z work (threats.rs:1346–1350) |
| 2b xray-to-square | candidates = `(R\|Q & ortho_empty \| B\|Q & diag_empty) & occ`; per candidate: 1 `between` table read + popcount; on exactly-1: 1 emit + 1 `ray_extension` (W) + mailbox | typically 0–4 candidates (comment threats.rs:1444–1446) |
| 3 non-sliders | 4 table lookups (2 pawn dirs, knight, king) masked into one loop | |

Calls per make_move: 2 (quiet) … 5 (promo-capture) — see §3.4 table.
Generated volume: ~10.6 deltas per push-pair on bench
(byteboard_splat_scoping lines 22–25); per-move generated histogram
available via `--features profile-threats` (`apply_stats::report`,
threats.rs:133–182; per-section cycle/emit/zero-emit counters in
`thr_stats`, threats.rs:245–418 — sections instrumented as
direct/own-xray/sliders/sliders-2b/nonsliders exactly matching §3.3).

### Per-delta expansion cost (apply side)

Per delta per perspective, `threat_index`/dual-body does 3 dependent table
loads: `piece_pair[12][12]` (u32), `piece_offset[12][64]` (i32),
`attack_index[12][64][64]` (u8, 48 KB — the "table-chase" target), plus the
POV/mirror XORs (threats.rs:700–718, 1844–1893). Then the weight-row apply
streams `hidden_size` i8s per index (register-tiled AVX2/AVX-512/NEON
kernels, threats.rs:1996–2199, 2249+; 8×ymm=128 i16/chunk AVX2, 16×zmm=512
i16/chunk AVX-512 — REG counts are tuned, see comments at 2020–2026 and
2125–2131 before changing).

Support tables from `bitboard.rs`: `ray_extension(from, blocker)` = squares
strictly beyond `blocker` on the from→blocker ray, 0 if unaligned/edge
(bitboard.rs:145–154, init at 156–170); `between(a, b)` excludes endpoints.
Both require `init_bitboards()` — a prior splat-test failure mode was
forgetting this (byteboard_splat_scoping lines 189–191).

### What the splat port established (do not re-learn)

- Reckless's byteboard splat (RAY_PERMUTATIONS mailbox permute +
  closest_on_rays + compress-store) is enumeration-compatible ONLY with a
  direct-attack feature space. Coda's space adds x-ray features at the SAME
  indices → a direct port double-counts (byteboard_splat_scoping lines
  175–248, "Option B: custom SIMD enumerator matching Coda's semantics" is
  the path this spec supports).
- ~600 LoC of Coda-encoded tables/primitives exist at
  `5096ac0:src/threats_splat.rs`: RAY_PERMUTATIONS (piece-encoding-
  independent, copied verbatim), RAY_ATTACKERS_MASK / RAY_SLIDERS_MASK,
  PIECE_TO_BIT_TABLE re-derived for Coda's colored_piece order,
  `mailbox_vector_avx512` bridging Coda's piece-type mailbox + white_bb into
  a colored-piece byte vector (+6 where NOT white; empty 6→12), and working
  `push_threats_on_change_avx512` / `push_threats_on_move_avx512` for the
  DIRECT-only subset. The missing SIMD pieces are exactly sections 1b, 2's
  Z-level, and 2b (second-hit-per-ray enumeration).
- Coda's `RawThreatDelta` bit layout == Reckless's `ThreatDelta`, so a SIMD
  enumerator can compress-store directly into the delta buffer.
- AVX-512 path needs VBMI(2) (`_mm512_permutexvar_epi8`,
  `_mm512_maskz_compress_epi8`) — Zen 4+/SPR+; AVX2 fallback is required for
  fleet coverage (Hercules/Atlas gained most from prior cache work).

---

## Appendix: invariants checklist for the rewrite

1. Feature = (attacker_cp, from, victim_cp, to), physical squares; direct and
   depth-1 x-ray share one index; depth ≥ 2 excluded; blocker any piece.
2. Pair filter per §1.4 may be applied at emission OR left to `threat_index`
   (current code leaves it to expansion — cheapest is to keep that).
3. Semi-exclusion: physical-square `attacker_sq >= victim_sq` keep-rule,
   same-type pairs except same-color pawns; decided at expansion, NOT at
   emission. Do not make it mirror-symmetric or bf-frame.
4. Emit raw physical tuples only; never pre-flip; never pre-index.
5. Net-count contract per index ∈ {−1,0,+1}; add+sub same-index pairs are
   legal and common; net ±2 is corruption.
6. Both push legs of a move run with the mover absent from occupancy
   (occ_transit) but present in pieces_bb/mailbox at `to` — every candidate
   scan must mask `& occ`.
7. King moves emit normal deltas; e-file crossing handled by consumer refresh;
   overflow (>128 at absorb) handled by consumer refresh; null move = 0 deltas.
8. `mailbox` stores piece TYPE (0..5, 6=empty); color from `colors[]`
   bitboards; NO_PIECE checks are `>= 6` everywhere in this path.
