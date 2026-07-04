# Engine-Notes Experiments Plan (2026-07-04)

Actionable items surfaced by the four new engine-notes (`raphael.md`, `icarus.md`, `uralochka.md`, `tcheran.md`). Prior art in `experiments.md` grepped for each — items with hits are dropped or de-duped.

## Dropped up front (Coda-tested prior art)

- **2-ply / follow-up prev-move corrhist** (Icarus E1, Hobbes E2 in that note). H0'd twice: **#905 `prev-move-corrhist`** −0.6 ±2.0 / 20.8K, **#911 `tune-906-applied`** (focused retune) −2.0 ±3.6 / 7.3K. Two independent H0s at bounds `[0,3]`, both upper-CI well below elo1. Coda's regime specifically doesn't benefit — Icarus corroboration doesn't rescue it. **Do NOT retry.**
- **Major/minor Zobrist corrhist source** — Coda removed `CORR_W_MINOR/MAJOR` on 2026-05-19 (structural remove per `6a383c8`). Raphael E1 (`major_hash`) proposes essentially the same signal via a different key. Would need a genuinely new mechanism (e.g. per-file/rank-partitioned) not merely a different index base — otherwise re-treads dead ground. **Drop unless we can articulate why the new key adds signal beyond the removed one.**

## The queue (10 items, ranked, all with prior-art check = 0)

Bounds: standing default `[0, 3]` per CLAUDE.md. Retune-on-branch strategy noted per item. Order below is (a) low implementation cost first, then (b) magnitude × confidence.

### Batch A — one-liner NMP/RFP gate additions

Cheapest implementations. Each is 3-8 lines. No new tunables required for the initial SPRT; can retune-on-branch if H1-adjacent.

**E1 · TT-upper NMP skip** (Uralochka E1, `~/chess/engines/uralochka3/search.cpp` — pattern also in SF/Ethereal). Skip NMP when TT is UPPER-bound and stored score already fails low against beta. The TT is telling us "we already searched to some depth and got a fail-low against beta"; a null-move search's job is to argue the position is better than beta — the TT says the opposite. Coda's NMP gate at `src/search.rs:3830` has no TT-flag check.

Sketch (single new predicate on line 3837):
```rust
&& !(tt_hit && tt_flag == TT_FLAG_UPPER && tt_score < beta)
```

Branch: `experiment/nmp-tt-upper-skip`. SPRT `[0, 3]`. No new tunable initially. **+1 to +4 Elo** (consensus one-liner, precedent = many similar gates already landed).

**E2 · opp_worsening RFP term** (Raphael E2, `src/search/pruning.h`). Widen RFP margin when `static_eval + prev_static_eval` — a specific "opponent is worsening" signal built from two adjacent evals. Coda already has `improving`, `unstable`, `has_pawn_threats` widening RFP; this is a distinct signal keyed on trend rather than direction.

Sketch: at `src/search.rs:3738` add after the `improving` margin selection:
```rust
if ply >= 2 && ply_u >= 2 {
    let prev_static = info.static_evals[ply_u - 2];
    if prev_static != -INFINITY && static_eval + prev_static > tp(&RFP_OPP_WORSEN_THRESH) {
        margin += margin / 4;
    }
}
```

Add tunable `RFP_OPP_WORSEN_THRESH` (default 0; range −200..200; c_end 20). Branch: `experiment/rfp-opp-worsening`. SPRT `[0, 3]`. **+0 to +3 Elo**.

**E3 · RFP margin adjusted by TT-hit** (Uralochka E2 partial). When TT hit provides a stored bound, tighten or widen RFP margin depending on flag direction. Cheaper variant: `if tt_hit && tt_flag == TT_FLAG_LOWER { margin -= margin / 4; }` — TT lower bound confirms the direction, allow more aggressive RFP.

Branch: `experiment/rfp-tt-hit-margin`. SPRT `[0, 3]`. **+0 to +2 Elo, low confidence** (may interact with the existing RFP TT quiet guard at line 3729).

### Batch B — new correction-history source

**E4 · Threat-bitboard correction history** (Tcheran E1, `~/chess/engines/tcheran/tables.rs:292-318`). Add a corrhist table keyed on a hash of the threat bitboard (or `zobrist_of(threats)`), stm-partitioned, 16K slots to match `CORR_HIST_SIZE`. Coda has pawn/np/cont/trans corr but nothing keyed on the **attacker structure**. Reuses existing threat computation (Coda already maintains `enemy_attacks`).

Sketch: add `threat_bb_corr: [[i32; CORR_HIST_SIZE]; 2]` field on `SearchInfo`; add `CORR_W_THREAT_BB` tunable (default 60, range 0..200); wire into `correction_value`, `corrected_eval`, `update_correction_history`, and reset paths. Sub-hash: `simple_mix_64(enemy_attacks) & (CORR_HIST_SIZE - 1)`.

Branch: `experiment/threat-bb-corrhist`. SPRT `[0, 3]` at default weight first. If H0 near zero, retune weight (Coda pattern: focused SPSA on the one new weight + adjacent existing weights + `CORR_HIST_DIV`). **+1 to +3 Elo standalone, more with retune.** Bench delta expected (new source in eval-correction pipeline).

### Batch C — LMR/pruning axis tweaks

**E5 · LDSE — Low-Depth Singular Extension** (Raphael E3, `src/search/main.cpp` — also in some Berserk variants). Coda's SE currently gates on `depth >= SE_MIN_DEPTH_10X` (~depth 7-8). LDSE proposes a *zero-cost* extension at lower depth: when static eval + corrplexity coupling is well below alpha, treat the TT move as effectively singular without a re-search. Effectively: extend by 1 at depth 4-6 if `tt_hit && tt_flag != TT_FLAG_UPPER && (static_eval + coupling * corrplexity) <= alpha - margin`.

Sketch: net-new small block gated ahead of the SE re-search entry. Careful: only fires when SE would NOT have fired (below MIN_DEPTH). Bounded by a new `LDSE_MAX_MARGIN` tunable. Branch: `experiment/ldse`. SPRT `[0, 3]`. **+0 to +3 Elo, medium risk** (SE interactions are famously tricky).

**E6 · min(counter, follower) history-prune gate** (Uralochka E4). Currently Coda's history-pruning threshold is compared against `main_hist + cont_hist_1 + cont_hist_2 + cont_hist_4 + cont_hist_6 + pawn_hist`. Uralochka pattern: also require `min(counter_hist_score, follower_hist_score) < threshold` — a stricter gate that avoids pruning when EITHER of the two ply-based cont-histories signals the move is reasonable.

Sketch: additional guard in the history-pruning block. Cheap. Branch: `experiment/hist-prune-min-cont`. SPRT `[0, 3]`. **+0 to +2 Elo, speculative** (interacts with our recent hist-prune tuning).

**E7 · Threat-partitioned capture history** (Tcheran E2). Extend `capt_hist` from `[piece][to][captured]` to `[piece][to][captured][from_thr][to_thr]` — 4× memory. Coda's cap-hist is currently threat-agnostic; main-hist and cont-hist have threat axes but cap-hist doesn't.

Sketch: bump the last two dimensions on `capt_hist`, wire threat bits at write/read sites. **Retune-on-branch mandatory** — cap-hist scale interacts with LMR-cap-hist-div, cap-hist-mult, and cap-hist-base. Branch: `experiment/capt-hist-threat`. SPRT after focused SPSA on `CAP_HIST_MULT` + `CAP_HIST_BASE` + `LMR_CAP_HIST_DIV`. **+1 to +2 Elo**, cost = one retune round.

### Batch D — ProbCut / cont-hist gravity refinements

**E8 · TT-only ProbCut** (Icarus E3, `~/chess/engines/icarus/src/search.rs:273-286`). Coda's ProbCut at `src/search.rs:3936` runs a re-search at reduced depth to confirm the shallow-margin cutoff. Icarus proposes: **skip the re-search entirely when TT already stores a score that satisfies the ProbCut condition** (`tt_score >= probcut_beta && tt_flag == TT_FLAG_LOWER && tt_depth >= probcut_depth`). Return the TT score directly.

Sketch: at ProbCut entry, before the re-search loop, add a fast path that returns `probcut_beta` when the TT confirms. Cheap conditional; net-negative bench (fewer PC re-searches). Branch: `experiment/tt-only-probcut`. SPRT `[0, 3]`. **+1 to +3 Elo, high confidence** (matches SF's `if (ttData.value >= probCutBeta)` pattern).

**E9 · Cont-hist total-scaled gravity** (Icarus E2, `~/chess/engines/icarus/icarus-common/src/history/cont.rs:73-88`). Icarus's cont-hist gravity coefficient is the **summed cont-hist score at cutoff**, not the current-entry value. Novel: the "how confident the whole cont-hist chain was" scales the update magnitude on individual entries.

Sketch: at cont-hist update sites, compute `sum = cont_hist_1 + cont_hist_2 + cont_hist_4 + cont_hist_6` first, then use `sum` as the gravity modulator. Retune-on-branch: cont-hist weights likely shift. Branch: `experiment/conthist-total-scaled-gravity`. SPRT `[0, 3]`. **+0 to +3 Elo, moderate risk** (touches gravity math — history table shape).

### Batch E — de-dupe against Hobbes queue

**E10 · History-modulated good/bad tactical SEE split** (Tcheran E3, overlaps Hobbes E12). SEE threshold for capture "good vs bad" scaled by capt_hist entry: high-history captures pass with lower SEE bar. Since this is the same mechanism Hobbes E12 already flagged, **de-dupe** — track under the Hobbes queue, not create two branches. Included here only for cross-engine consensus (Reckless + Obsidian + Tcheran + Hobbes = 4-engine backing). **Move up Hobbes E12 priority.**

## Suggested execution batches

**Batch A (E1 + E2 + E3):** 3 branches × ~5 lines each. Total time to implement: ~1 hour. Total SPRT cost: 3 concurrent SPRTs at `[0, 3]`. Expected batch envelope: **+1 to +6 Elo**.

**Batch B (E4):** New corrhist source; ~50 lines + SearchInfo field. ~2 hours. One SPRT. **+1 to +3 Elo**.

**Batch C (E5 + E6 + E7):** Bigger touches, some need retune. E7 requires a focused SPSA before its SPRT. Sequence rather than parallelise. **+1 to +7 Elo total.**

**Batch D (E8 + E9):** ProbCut fast-path + cont-hist gravity. E8 is independent + cheap; E9 wants retune. **+1 to +6 Elo total.**

**Batch E:** Move E10 → Hobbes E12 slot with 4-engine backing.

**Recommended immediate action:** implement Batch A + E4 + E8 in parallel — 5 branches, all can go to the fleet at once, none require prior retunes. Total expected magnitude if all H1: **+4 to +15 Elo**. Cost: ~3-4 hours of implementation + concurrent SPRTs.

## Notes on interaction risks

- E2 + E3 both touch RFP margin math. Ideally test in isolation; if both land H1 individually, verify no double-counting by testing the union.
- E4 + E9 both touch corrhist-adjacent infrastructure. E4 adds a source; E9 changes gravity. If E4 lands, E9 should be re-baselined off the E4 trunk.
- E5 (LDSE) can misinteract with existing DEXT_CAP + hindsight-extension logic. Test with FEAT_EXTENSIONS toggle-off ablation first if surprising results.
- E7 (threat-partitioned cap-hist) is the only item that **must** SPSA-retune before initial SPRT, per the "cap-hist scale interacts with LMR-cap-hist-div" comment in `movepicker.rs`.

## Companion docs

- `engine-notes/raphael.md`, `engine-notes/icarus.md`, `engine-notes/uralochka.md`, `engine-notes/tcheran.md` — source of these items.
- `engine-notes/hobbes.md` — E10 lives here (E12 in the Hobbes note).
- `experiments.md` — grepped for prior art on every item.
- `CLAUDE.md` — bounds default `[0, 3]`, OB workflow.
