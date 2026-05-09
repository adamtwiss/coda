# Correctness audit + tunable deep-dive plan — 2026-05-09

Three-agent audit: code-review of last ~2 weeks of commits, SPSA-detuning
analysis across recent tunes, cross-engine consensus comparison vs SF /
Reckless / Viridithas / Obsidian / Berserk / PlentyChess.

**Goal**: surface the tens of Elo we suspect are sitting in sub-optimal
or buggy implementations — wrong gates, inverted STM, formulas that
don't scale right.

**Top-level finding**: three confirmed correctness bugs, one strongly
implicated cascade chain on NMP (gate-too-tight → MIN_DEPTH=7 →
EVAL_MAX=1), and one machinery-wide signal on continuation history
(write/read off-by-one + CONT_HIST_MULT pinned at floor for 5 tunes).
Many SPSA-detuned tunables are downstream symptoms of structural
divergences, not independent issues.

---

## Confirmed bugs (P1)

### Bug 1 — N6 promotion-imminent extension dead since merge

**Site**: `src/search.rs:3059-3060`

```rust
let on_seventh = (board.side_to_move == WHITE && to_rank == 6)
    || (board.side_to_move == BLACK && to_rank == 1);
```

This runs **after** `board.make_move(mv)` (line 3020). `make_move` flips
`board.side_to_move`. So when White pushes a pawn to rank 6, after
make_move `board.side_to_move == BLACK` — the condition `WHITE &&
to_rank==6` is never true. Symmetric for Black. **The promotion-imminent
extension has never fired since N6 merged in commit `8d18816`** (~3
weeks ago, +1.6 Elo measured at the time, which was actually the
tree-shape of "extension dead but code present").

**Fix**: branch `fix/n6-promotion-imminent-stm` (commit `b85e08f`,
2026-04-25) replaces `board.side_to_move` with the pre-loop `us`
captured at `search.rs:2151` area. Bench moves 788473 → 554401 (~30%
reduction once the extension actually fires).

**Status**: fix branch sat unmerged for 2 weeks. Trunk's tunable defaults
were calibrated under "extension dead" — merging needs SPRT `[-3, 3]`
followed by retune-on-branch.

### Bug 2 — cont_hist write/read off-by-one (`< 12` vs `< 13`)

**Sites**:
- Writes (exclude index 12): `src/search.rs:3299, 3396, 3438`;
  `src/movepicker.rs:259, 377`
- Reads (include index 12): `src/search.rs:2841, 2877, 2915`

`go_piece(p)` returns `p + 1` for piece indices 0..=11, so output is
1..=12. The cont_hist table is `[13][64][13][64]` — index 12 is valid.
Read paths use `p < 13`; write paths use `prior_piece < 12`. **King
moves' continuation history is permanently zero across all
perspectives** because the writes never land at index 12 but the reads
sample it.

SF, Reckless, Obsidian all include kings in continuation history. The
asymmetry only became visible when the new hist-prune diagnostic code
introduced `< 13` reads (commits `d103c8f`, `1805a86`).

**Fix**: change all write-path guards from `prior_piece < 12` to `<
13`. SPRT `[-3, 3]` — table cells at index 12 will accumulate real
signal for the first time.

### Bug 3 — `LMR_ENDGAME_PIECES = 4` despite documented override of 5

**Site**: `src/search.rs:235` declaration says `(LMR_ENDGAME_PIECES, 4,
4, 9, 1.5)`; tune-784 comment at `src/search.rs:79-82` says:

```
// Overrides applied to SPSA output:
//   LMR_ENDGAME_PIECES kept at 5 (SPSA drifted to 4 again; play-quality
//     load-bearing per feedback_play_quality_params_narrow_range)
```

Override was forgotten when tune-784 was applied; declaration says 4.

**Why it matters** (per Adam, 2026-05-09): Lichess play exposed clear
endgame blindspots — rook on an open board has many moves, so moves
were treated as "late" by LMR and tactical recoveries over-reduced.
Setting the gate to 5 closed these (SPRT slightly positive on self-play,
visibly improved play-quality on Lichess). At 4 the gate fires only for
≤4 pieces (K+P vs K only); at 5 it catches K+Q vs K, K+R vs K, etc.

SPSA repeatedly drifts this back to floor because self-play doesn't
generate enough rook-on-open-board endgames to see the gate's benefit;
the cost is more SPSA-visible than the benefit.

**Fix**: change `4` to `5`. SPRT `[-3, 3]`. See
`memory/project_lmr_endgame_pieces_play_quality.md` for the
canonical-anchor case.

---

## Cascade chain — NMP gate-too-tight (highest-leverage structural)

Strongest single story across all three audits. Three NMP tunables
SPSA-pinned at extremes vs cross-engine consensus:

| Tunable | Coda | Cross-engine consensus |
|---|---:|---|
| `NMP_MIN_DEPTH` | **7** | 3 (Vir), 3.78 (Plenty), none (SF/Reckless) |
| `NMP_EVAL_MAX` (cap on `(eval-beta)/EVAL_DIV` in R) | **1** | 4 (Vir/Obsidian/Plenty), unbounded (SF/Reckless) |
| `NMP_BASE_R` | 7 | 4-5 (most), 7 (SF) |
| `NMP_DEPTH_DIV` | 5 | 3 (most), 2.4 (Plenty) |

Combined: at `NMP_BASE_R=7, DEPTH_DIV=5, EVAL_MAX=1`, even when Coda
fires NMP at depth 7, R = 7 + 1 + 1 = 9, leaving null-move depth at -2
(clamped to 1). NMP only does meaningful work at depth ≥ 9.

But the root cause is the **gate**:

| Engine | NMP gate condition |
|---|---|
| Coda | `static_eval >= beta` (binary) |
| SF | `staticEval >= beta - 16*depth - 53*improving + 378` |
| Obsidian | equivalent shape with depth/improving margins |
| Reckless | `>= beta + (-8*depth + 116*tt_pv - 106*improvement/1024 - 20*(cutoff_count<2) + 304).max(0)` |

**Coda has the strictest gate of any engine compared.** SF/Obsidian
explicitly allow NMP when eval is *below* beta by a depth-scaled margin
(NMP is effective when eval is comfortably above beta; the depth term
acknowledges shallow NMP tolerates worse static evals).

**Hypothesis**: tight gate → NMP rarely fires at depth 2-6 → SPSA
raised `MIN_DEPTH` to 7 to avoid wasted attempts → at depth 7 the
eval-r contribution (`(eval-beta)/108`) saturates rarely, so SPSA
pinned `EVAL_MAX` at 1 to avoid noise. **One root cause, three symptom
tunables.**

**Plan**:
1. Add tunables `NMP_GATE_DEPTH_MUL, NMP_GATE_IMPROV_MUL,
   NMP_GATE_OFFSET` (SF defaults `16, 53, 378`)
2. Widen `NMP_MIN_DEPTH` lower bound to 2; widen `NMP_EVAL_MAX` to
   range `[1, 8]`
3. SPRT structural change at `[0, 5]`
4. SPSA retune NMP cluster (5-6 params, 1500 iter focused)
5. SPRT retuned vs trunk

**Decisive signal**: SPSA convergence on (`MIN_DEPTH≈3`,
`EVAL_MAX≈4`) confirms the cascade hypothesis. Expected payout +3-8
Elo if it holds.

---

## Cont-hist machinery audit

Pairs with Bug 2 above. Two independent signals say the cont-hist
tables are mis-shaped:

- **`CONT_HIST_MULT`** floor-pinned at 1 across 5 tunes (820/855/928/962/983)
  — at value=1 the four cont-hist plies [1,2,4,6] carry weights
  [1,1,1,1] vs SF/Obsidian/Berserk consensus [3,3,1,1] or Plenty's
  [2,1,1,½]
- **King writes missing** (Bug 2) means index-12 cells permanently zero

The prior x16 fixed-point split refactor (`(W1, W2, W4, W6)`) failed
SPSA convergence — `5a2df1d` notes the `[0,4]` int range with c_end 0.4
was too small for SPSA to find a gradient.

**Plan** (depends on Bug 2 landing first):
1. Land Bug 2 cont_hist write fix
2. Instrument cont_hist value distributions at offsets 1, 2, 4, 6 —
   already partially done (`hist_prune_ratio_buckets` at
   `search.rs:2873-2888`); add per-offset write-count + read-magnitude
   histogram
3. If offsets 4, 6 are systematically near-zero in writes despite reads
   sampling them: there's a bug in the write path specific to those
   offsets
4. Re-run x16 split tune with **wider c_end** (per `5a2df1d` lesson)
5. SPRT applied weights vs trunk

**Expected**: +3-5 Elo if writes at offsets 4, 6 are mechanically
broken; smaller if just calibration.

---

## DEXT permissiveness chain

`DEXT_MARGIN_BASE = +41` (Coda) is sign-flipped vs SF (-186) /
Reckless (-16). Coda has explicit comment that the Reckless port at -16
"exploded our bench +67% at #804" — so the +41 BASE compensates for
*missing suppressors* (cutoff_count, ttMoveHistory, ply-vs-rootDepth)
that Reckless / SF have.

**Plan**:
1. Add `ttMoveHistory` modulator (track per-move TT move's prior
   history at SE point) — pure addition, SPRT `[0, 3]`
2. If H1, drop `DEXT_MARGIN_BASE` to -16 and SPSA the cluster
3. Add SF-style `(ply > rootDepth) * 44` modulator if step 1 didn't
   suppress enough

**Expected**: +2-4 Elo across the chain.

---

## Decisive ablations (1-day SPRT each)

Floor-pinned tunables that are likely zero-Elo features:

| # | Ablation | Prior | Bounds |
|---|---|---|---|
| A1 | `NMP_UNDEFENDED_MAX = 0` | 25K-iter tune drove to 1.45 (~feature off); already queued in code comment at `search.rs:84-86` | `[-3, 3]` |
| A2 | `SE_KING_PRESSURE_MARGIN = 0` | 25K-iter tune drove to 0.22; #874 ablation showed -2.6 Elo (in noise) | `[-3, 3]` |
| A3 | `HIST_BONUS` shape (MULT=325 / OFFSET=22 / MAX=1752) | Cross-engine: SF MULT=119-135 / OFFSET=74-93 / MAX=1400-1529. Coda saturates at MAX by d=6, SF at d=10 — Coda loses depth discrimination above d=6. Biased-start SPSA at SF values | SPSA 1200 iter |
| A4 | NMP `r += 1 after capture` polarity | Code comment claims SF/Obsidian do this; cross-engine review says citation is wrong (Obsidian uses `ttMoveNoisy` of *this* node, not previous-move-was-capture). Re-read original SPRT history; if untested-port, SPRT removal | `[-3, 3]` |

---

## Cross-engine outliers (lower confidence — Tier 5)

Worth investigating but each likely +1-3 Elo and may overlap with the
chains above.

| # | Item | Coda vs consensus | First step |
|---|---|---|---|
| 5.1 | `SE_DEPTH=4` vs SF=6+ttPv, Reckless=5+ttPv; singular_beta margin 1×depth vs consensus 1.1×-2.3× | shallow + tight | Bump SE_DEPTH to 6+ttPv, SPRT `[0, 3]` |
| 5.2 | `SEE_QUIET_MULT=32` vs SF=25/Berserk=15; missing history modulation Reckless has | compensates by widening threshold | Add `SEE_QUIET_HIST_DIV` term, then SPSA |
| 5.3 | `LMP_BASE=9` vs consensus 3 — same formula, 3× more permissive at d=1-2 | bisect 9 → 5 → 3 | Two SPRTs `[-3, 3]` |
| 5.4 | `LMR_C_CAP < LMR_C_QUIET` (captures reduced more than quiets) — inverted vs Reckless | swap starting points | SPSA `(LMR_C_QUIET, LMR_C_CAP)` swapped |
| 5.5 | HINDSIGHT missing increase-on-eval-worsening branch (Reckless 2-way; Coda only reduces) | pure addition | Ablate Reckless's branch first per `feedback_ablate_source_before_port`; if load-bearing, port |
| 5.6 | `IIR_MIN_DEPTH=2` vs SF=6; floor-pinned. Add `prior_reduction <= 3` SF-gate | TT-hit-rate coupling | SPRT `[0, 3]` |
| 5.7 | `PROBCUT_MIN_DEPTH=2` vs SF/Reckless=4-5; floor-pinned | instrument fire rates first | dbg_hit then ablation |
| 5.8 | CORR_HIST per-source weight 2.9× weaker than Obsidian (CORR_HIST_GRAIN_T=11 over-divider) | Drop GRAIN_T to 1 and SPSA reset | Focused tune |

---

## Doc / cleanup

- **Stale comment block at `src/search.rs:60-86`** cites tune-deltas
  that don't match current values (e.g. `BASE_R 5→6` but actual is 7;
  `MIN_DEPTH 5→6` but actual is 7). Update or drop.
- CLAUDE.md was updated 2026-05-09 — TC-handicap sigmoid replaced with
  self-play sweep, in-flight numbers softened, stale +88 Elo framing on
  v9 low-LR tail corrected.

---

## Recommended execution order (by Elo / risk / dependency)

1. **Bugs 1-3 (P1)** — pure correctness, SPRT `[-3, 3]` each, sequentially
   (each invalidates next's calibration so retune-on-branch between
   if bench delta large)
2. **NMP cascade (Tier 1)** — single biggest expected payout (+3-8 Elo);
   needs structural change + SPSA retune
3. **Cont-hist machinery audit (Tier 2)** — depends on Bug 2 landing
   first
4. **Decisive ablations A1-A4** — run in parallel with Tier 2, low fleet
   cost
5. **DEXT chain (Tier 3)** — multi-step, +2-4 Elo
6. **Tier 5 cross-engine outliers** — as fleet capacity allows

**Don't queue hist-prune work** — Atlas owns it, model-specific per
Adam's instruction 2026-05-08.

---

## Methodology notes

- Each suggested experiment includes ablation-of-source (per
  `feedback_ablate_source_before_port`) before recommending a port.
- Per `feedback_consensus_patterns_dont_always_transfer`, this audit
  emphasises **structural divergences** (formula shape, gate polarity,
  missing modulator) over bare value imports.
- Bug 1 (N6) is the canonical example of a feature SPRT'd while dead —
  +1.6 Elo measured was tree-shape ablation, not the feature itself.
- LMR_ENDGAME_PIECES regression was play-quality-relevant on Lichess
  but invisible in self-play SPRT — see `feedback_play_quality_params_narrow_range`
  and the new `project_lmr_endgame_pieces_play_quality`.

---

## Source data

- Code review agent (~60 commits, 2 weeks)
- SPSA detuning analysis (tunes 820, 855, 928, 962, 983 raw float values
  via OB digest endpoints)
- Cross-engine comparison (SF, Reckless, Viridithas, Obsidian, Berserk,
  PlentyChess from `/home/adam/chess/engines/`)

Tune SPSA values were pulled directly from the OB API; engine values
were read from current sources of each engine.
