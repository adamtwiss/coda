# TT hash-sensitivity investigation — why 64MB hurts Coda more than Reckless/SF

**Date:** 2026-04-27
**Anchor measurement:** 60+1 H2H 200-game gauntlet, hash 64→512MB with EGTB on
gave Coda **+35 Elo overall** (+43 vs SF, +27 vs Reckless). Same delta is
reportedly smaller for SF/Reckless when comparing across hash sizes — see
`feedback_sf_vs_reckless_gaps_are_different.md` for the 2×2 controlled
data and `feedback_sprt_blind_to_long_game_effects.md` for why SPRT
10+0.1 can't see this class.

**Question Adam asked:** *"why may we be worse hit than Reckless/SF with
a 64MB hash table?"*

This is a punch-list of structural differences ranked by suspected impact.
SPRT-able candidates are flagged with `[SPRT]`. Reading-only entries are
diagnostics.

---

## 1. QS in-check stores `-INFINITY` for static_eval `[SPRT]`

**Files:** `src/search.rs:3563`, plus the corresponding read-side at
`src/search.rs:3572`.

**Code:**
```rust
// Line 3563 — QS in-check store
info.tt.store(board.hash, -1, store_score, flag, best_move, -INFINITY, false);

// Line 3572 — when reading static_eval back
let raw_stand_pat = if tt_hit && tt_entry.static_eval > -(MATE_SCORE - 100) {
    tt_entry.static_eval                       // cache hit
} else {
    info.eval(board)                           // recompute
};
```

When the position is in check during QS, we have no stand-pat to store
so we sentinel `-INFINITY` for static_eval. The read-side gate
`> -(MATE_SCORE - 100)` then rejects this entry on every probe — meaning
**every revisit re-runs full NNUE inference**.

At 64MB TT pressure, the same in-check QS positions get visited many
times during a 60+1 game. Each visit pays the eval cost. At 512MB, the
"real" TT entries with valid `static_eval` survive longer, so even when
QS in-check entries get evicted, the parent positions still cache
their evals.

**Caveat:** in-check QS does not compute a static eval at all (we go
straight into capture-only search). Storing something useful here would
require either (a) computing a fresh eval at the in-check QS entry —
NPS cost — or (b) propagating a parent eval down. SF/Reckless behaviour
to be confirmed before SPRT.

**Test:** verify what SF/Reckless store at the equivalent QS in-check
TT save call. If they store something usable, port the pattern. SPRT
[0, 5] expecting larger delta as TT pressure rises (so the SPRT win at
10+0.1 is likely understated; deployment 1GB hash captures the rest).

## 2. Bucket density: 5 entries × 12B = 60B vs SF/Reckless 3 × 10B = 30B `[SPRT]`

**Files:** `src/tt.rs:13` (`BUCKET_SIZE = 5`), `:75-83` (struct layout).

**Layout:** Coda packs 5 entries × (8B data + 4B key) = 60B + 4B pad =
64B (one cache line). SF (`Stockfish/src/tt.cpp:140-147`) and Reckless
(`reckless/src/transposition.rs`) use 3 entries × 10B = 30B clusters,
which means **2 clusters per cache line**.

**Why this could matter under TT pressure:**

- Coda 64MB: ~1.05M buckets × 5 = 5.24M slots. SF/Reckless 64MB:
  ~2.1M clusters × 3 = 6.3M slots. **17% more slots per MB** for
  SF/Reckless before any other effects.
- More importantly, when a cluster scan misses, SF/Reckless's cache
  line carries a *neighbouring* cluster's 3 candidates. Coda's
  cache line is fully consumed by ONE bucket's 5 candidates. This
  doubles the effective "scan candidates per cache line" for SF/Reckless,
  improving locality during high-turnover search.
- Under TT pressure (small hash, deep search), the marginal cost of
  picking a worst-of-5 vs worst-of-3 also matters: 5-way replacement
  thrashes more entries per insertion.

**Test:** SPRT a `BUCKET_SIZE = 3` variant with 32B clusters (4-byte key
+ 6-byte data, or restructure to fit 30B + 2B pad). This is a
medium-sized refactor; expect retune-on-branch to be required.

**Risk:** PEXT/cache prefetch is currently tuned for 64B clusters via
`tt.rs:483 _mm_prefetch`. Restructuring would need to revalidate the
prefetch behaviour, since 32B clusters might want a different prefetch
pattern.

## 3. Replacement age weight: `* 4` vs SF's `* 8` `[SPRT]`

**Files:** `src/tt.rs:464-465`.

**Code:**
```rust
// Coda
let age = gen.wrapping_sub(slot_gen) as i32;
let slot_score = slot_depth - age * 4;
```

vs SF (`tt.cpp:80,122,237-238`): `relative_age` returns `age * 8`
(GENERATION_DELTA=8) and the replacement scoring is `depth - relative_age`,
i.e. effectively `depth - 8*age`.

**Mechanism:** under TT pressure, entries from prior moves are stale
(their depth was relevant for a *different* position's search tree).
Coda's lower age weight means deep-but-stale entries beat current
shallow entries longer, holding onto useless slots.

**Quick math:** depth-20 entry from 2 generations ago vs depth-15
fresh entry:
- SF replace score: `20 - 16 = 4` vs `15 - 0 = 15` → fresh wins.
- Coda: `20 - 8 = 12` vs `15 - 0 = 15` → fresh wins by less.

At 1 generation old: SF `20-8=12` vs `15-0=15` → fresh wins. Coda
`20-4=16` vs `15-0=15` → **stale wins**. The stale-vs-fresh
breakeven point is ~1 generation later for Coda.

**Test:** SPRT `age * 8` vs `age * 4`. Single-line change.

**Coupling:** interacts with #834 (PV-bonus same-key gate). Don't
SPRT both at once until #834 lands.

## Honourable mentions / not bugs

- **No separate eval cache:** none of SF/Reckless/Coda has one. Modern
  NNUE engines store eval in TT.
- **No separate pawn-eval hash:** same. Not relevant.
- **Continuation history table size:** Coda `cont_hist [13][64][13][64] *
  2 bytes` ≈ 1.36MB. Reckless similar. Not a differentiator under TT
  pressure.
- **Generation rollover (8-bit, 256 ucigos):** wraps fine. Reckless 5-bit
  (32) wraps faster but works. Not a bug.
- **Same-key replacement gate** (`depth > slot_depth - 3 || gen != slot_gen`):
  branch `experiment/tt-replacement-pv-bonus` (#834 SPRT'ing 2026-04-27)
  ports SF's `EXACT-always-wins + 2*pv` rule. Once it lands, the
  same-key path matches consensus.

---

## Suggested SPRT order

1. **#834 (already in flight):** EXACT-always-wins + 2×pv depth bonus
   on same-key gate. Wait for resolution.
2. **#1 above** (QS in-check static_eval): single-line change once
   we confirm SF/Reckless behaviour. Highest expected gain at 64MB.
3. **#3 above** (age weight 4→8): single-line, after #834 lands so
   the interaction is clean.
4. **#2 above** (bucket size 5→3): structural refactor. Largest
   uncertainty, biggest potential win. Last because it's the most
   work and carries retune-on-branch overhead.

## Cross-engine source references

- Coda TT: `/home/adam/code/coda/src/tt.rs`
- SF TT: `/home/adam/chess/engines/Stockfish/src/tt.{cpp,h}`
- Reckless TT: `/home/adam/chess/engines/reckless/src/transposition.rs`
  (path TBC — verify before reading)
- Obsidian, Berserk: `/home/adam/chess/engines/{Obsidian,Berserk}/`
  for cross-validation
