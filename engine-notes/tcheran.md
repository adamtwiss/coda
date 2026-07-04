# Tcheran Search/Eval Review (2026-04-19)

Rust engine by [@jgilchrist](https://github.com/jgilchrist). v12.0 CCRL Blitz
3663 / 40+15 3534 — weaker than Coda (~3700–3800 class). NNUE
`(768x8hm->1024)x2->8`, self-play data only, jw1912/bullet fork
(`README.md:1-92`, `NET.md:1-40`). Source at
`~/chess/engines/tcheran/engine/src/`. No feature-flag ablation, ~500 LoC/file.

Primary source of strength vs Coda: nothing structural. SPSA-tuned
(`params.rs:1-104`), simpler search than Coda, plain 1×1024 SCReLU NNUE with
8 output buckets and 8-input-bucket horizontal mirror — architecturally
*behind* Coda's v9 (`768pw + threats + 16→32 hidden`). Interesting ideas few
but crisp.

## Search architecture

- **Framework**. ID (`iterative_deepening.rs`) → aspiration
  (`aspiration.rs:26-68`) → negamax (`negamax.rs:20-597`) → QS
  (`quiescence.rs:16-193`). PVS zero-window + PV re-search
  (`negamax.rs:482-496`).
- **Aspiration**. `width=16`; on fail `width += width/2`
  (`aspiration.rs:22`); fail-low halves `beta` toward `alpha`
  (`aspiration.rs:57`); fail-high shrinks depth by 1 up to
  `aspiration_max_reduction=3` (`aspiration.rs:60-64`). Kicks in at
  `depth>=5`.
- **NMP**. `cut_node && eval>=beta && !zugzwang && parent!=null`
  (`negamax.rs:230-269`). `R = 5 + depth/5` — no eval-margin term
  (`params.rs:20-21`), **no verification search**.
- **RFP**. `depth<=5`, margin `111*d - 95*improving` (`negamax.rs:200-212`).
  Blend on return: `beta + (eval-beta)/3` (`negamax.rs:207-211`).
- **Razoring**. `depth<=5, eval+d*443<=alpha, |alpha|<2000` → qsearch
  (`negamax.rs:214-227`). Coda re-added razoring (`src/search.rs:130-135`).
- **Futility**. `lmr_depth = depth - LMR[depth][moves_tried]`; `eval + 496 +
  lmr_depth*148 <= alpha` → skip quiets (`negamax.rs:350-364`).
- **History pruning**. `lmr_depth<=4`, `(quiet_hist+conthist) < 86 +
  lmr_depth*-2787` (`negamax.rs:369-378`). Combined quiet+cont hist gate;
  Coda uses only main_hist.
- **SEE pruning**. `lmr_depth<=10`; quiet `lmr_depth² * -55`; capture
  `lmr_depth*-240 - tactical_hist/12` (`negamax.rs:380-398`).
- **LMP**. `(5 + lmr_depth²)/(1 + !improving)` (`negamax.rs:400-414`).
- **ProbCut**. **Absent.**
- **LMR**. Log table `base=0.24 + ln(depth)*ln(mc)/2.25` (`tables.rs:18-30`,
  `params.rs:48-49`). Modifiers via `DepthReduction::reduce_more_if /
  reduce_less_if` at fixed-point ×1024 (`negamax.rs:437-452`):
  - `+cut_node * 400/1024`
  - `+!is_pv * 940/1024`
  - `+(child.fail_highs > 2) * 760/1024`
  - `-in_check * 320/1024`
  - `-tt_pv * 1148/1024`
  Do-deeper re-search on fail high (`negamax.rs:471-481`). No do_shallower,
  no LMR fail-high re-search on capture, no history-modulated LMR.
- **Extensions**. Check ext (`negamax.rs:47-49`); singular
  (`negamax.rs:279-326`) with double if `!is_pv && s_score < se_beta - 18`,
  cap 5. Negative SE branches: multi-cut on `se_beta>=beta`; `!is_pv &&
  se_score>=beta → return se_score`; `tt_score>=beta → ext=-1`. No triple,
  no recapture, no hindsight.
- **IIR**. `depth>=3 && tt_entry.is_none()` (`negamax.rs:272-274`) — applies
  at any node type; Coda gates on `PV || cut_node`.
- **Correction history**. 5 sources (`tables.rs:287-350`): pawn(196),
  major(101), minor(168), non_pawn(88), **threat(149)** — threat key is
  `game.threats.as_u64()` (opponent-attacked BB), `tables.rs:315-318`.
  Divisor 2048. Skip on `in_check ∨ best_capture ∨ Lower&score≤eval ∨
  Upper&score≥eval` (`negamax.rs:576-582`).
- **SMP**. Shared-TT Lazy SMP; per-thread tables+NNUE+stack
  (`search.rs:69-100`).

## Move ordering

- **Stages**: BestMove → GenTacticals → GoodTacticals (history-adjusted
  SEE split) → Killer → GenQuiets → ScoreQuiets → Quiets → BadTacticals →
  Done (`move_picker.rs:14-25`).
- **Tactical scoring**: `SEE(captured) + SEE(promo) - SEE(pawn) +
  tactical_hist/8` (`move_picker.rs:193-208`).
- **Good/bad tactical split**: `see(mv, -entry.score/4)` — hist crosses
  moves into "good" at negative SEE (`move_picker.rs:94-101`). Coda uses
  fixed threshold.
- **Quiet scoring**: `quiet_hist + conthist + 1000*direct_check`
  (`move_picker.rs:211-215`). No pawn-hist / threat-hist / from-to axes.
- **Killer**: single slot per ply (`tables.rs:78-96`); no counter-move.

**History tables** (all `tables.rs`, i16 gravity):
- `QuietHistoryTable`: `[stm][from][to][from_thr][to_thr]`, MAX=8192
  (`tables.rs:111-143`). Matches Coda's main_hist shape.
- `TacticalHistoryTable`:
  `[piece][to][captured+1][from_thr][to_thr]` — **threat-partitioned**,
  MAX=8192 (`tables.rs:145-187`). Coda's cap_hist is `[piece][to][victim]`,
  threat-agnostic.
- `ContHistTable`: `[prev_piece][prev_to][piece][to]`, MAX=16384; plies =
  **[1, 2] only** (`tables.rs:189-285`). Coda has 4 plies (1,2,4,6).
- **Bonus**: `min(depth*factor - offset, max_bonus)` per table — same shape
  as Coda.

## NNUE / eval

Architecture: `(768x8hm->1024)x2->8` — 768 input features (12×64, no HalfKA
king-index), 8 input buckets, horizontal mirror, 1024 hidden per perspective,
plain concat (no pairwise) into 8 output buckets (`nnue.rs:14-83`). QA=255,
QB=64, `SCALE=267` (`nnue.rs:19-23`).

**Activation**: SCReLU (clamp → mul → weight → sum) at
`inference.rs:41-107`. SIMD paths for AVX-512, AVX2, NEON; 4-way unrolled.
**STM-first ordering** (`us` at `output_weights[..HIDDEN_SIZE]`,
`inference.rs:31-34`).

**Bucket layout**: 8 king-region buckets, mirrored across file 4
(`nnue.rs:36-88`). Coda uses HalfKA (12288 inputs, 16 king buckets).

**Training** (`NET.md`): self-play only via OB datagen; current net
`(768x8hm->1024)x2->8`, WDL≈0.4, **Ranger optimiser** for #20 (+3.5 Elo vs
AdamW). No filter/quiet augmentation, no transfer learning. Weakest link vs
Coda's v9 (1536 accum, threats, 16→32 hidden).

**Eval scaling**: `scale = 986 + material/24`, /1024 (`eval.rs:6-18`;
material = N+B+R+Q ×SEE values). Coda scales via `apply_halfmove_scale`
(`search.rs:1227`); Hobbes-note dropped this item.

## Notable / novel mechanisms

- **Threat-BB correction history**. `corr[stm][hash(threats_bb)]` weighted
  149/2048 = 5th corr source (`tables.rs:292-318`). Coda's 5 sources
  (pawn/np-w/np-b/cont/trans) do not include a raw attacker-BB key.
- **Threat-partitioned CAPTURE history**
  (`tables.rs:145-187`). 4× cap_hist memory; may capture "capturing a
  threatened piece" signal.
- **Fixed-point LMR modifier framework** (`types.rs:179-204`): modifiers
  compose as u32/1024 for sub-ply SPSA gradient.
- **Aspiration depth-reduction on fail-high** (`aspiration.rs:60-64`) —
  Coda already does this (see E4).

## Testable Experiments for Coda (ranked)

### E1. Threat-bitboard correction history source
- **Tcheran**: `tables.rs:292-318` — `threat` corrhist keyed on
  `game.threats.as_u64()` (bitboard of opponent-attacked squares) → hash →
  `CorrectionHistoryTable`, weight 149/2048.
- **Coda today**: `src/search.rs:1364-1438, 251-269` — 5 sources
  (pawn/np-white/np-black/cont/trans). No threat-BB-keyed source; the
  threat BB is only consumed as an ordering axis and NNUE feature. Coda
  already computes threats every make/unmake (`src/threats.rs`).
- **Prior art**: NONE. `experiments.md` grep for `threat.*corrhist` /
  `threat_correction` returns no dedicated test. Coda's minor/major corr
  were H0'd and dropped (`search.rs:254-259`) but those were **piece-set**
  keys, not threat-BB.
- **Sketch**: reuse `CorrectionHistoryTable` shape; key = `zobrist_of_bb`
  (e.g. `hash_u64(threats_bb) as usize % SIZE`); add `CORR_W_THREAT` weight
  tunable seeded at ≈75 (half Tcheran's, Coda uses more sources); update
  alongside existing sources.
- **Magnitude/risk**: +1–3 Elo. Low risk — additive corrhist source, gravity
  update. Retune CORR_W_* after.

### E2. Threat-partitioned capture history
- **Tcheran**: `tables.rs:145-187` — cap_hist keyed
  `[piece][to][captured+1][from_thr][to_thr]`, MAX=8192.
- **Coda today**: `src/movepicker.rs` capt_hist is `[piece][to][victim]`
  (see `src/movepicker.rs:42-54` for main-hist analogue; cap_hist mirrors
  structure without threats). Coda's caphist magnitude has been repeatedly
  tuned (`experiments.md` `caphist_retune_proposal_2026-04-19`).
- **Prior art**: NONE for threat-splitting cap_hist. `#2145
  movepicker-see-threshold-mvv H0 -2.5` was a different mechanism
  (`experiments.md:16215`).
- **Sketch**: expand cap_hist to 4× entries with `[from_thr][to_thr]`
  bucketing; probe with `game.threats.contains(from/to)`. Same update path,
  extra 2 indices.
- **Magnitude/risk**: +1–2 Elo plausible; risk medium (memory 4×, needs
  retune of cap_hist bonus/malus + SEE-cap-margin history divisor which
  Tcheran sets at 12 vs Coda's SEE_HIST_DIV equivalent).

### E3. History-modulated good/bad tactical split threshold
- **Tcheran**: `move_picker.rs:94-101` — tactical SEE threshold is
  `-entry.score / 4` where `entry.score` includes `see_value(captured) +
  tactical_hist/8`. High-hist captures cross into "good" pile with negative
  SEE; low-hist positives get demoted.
- **Coda today**: uses fixed `QS_SEE_THRESHOLD` for QS
  (`experiments.md:3935, 11103`) and BAD_NOISY_MARGIN
  (`src/search.rs`) for main search. Threshold does not read caphist.
- **Prior art**: this exact pattern is **Hobbes E-tier #12**
  (`engine-notes/hobbes.md:376-397`) — flagged there, not yet SPRT'd.
  Re-flagged here as consensus (Hobbes + Tcheran + Reckless + Obsidian).
  Not in `experiments.md`.
- **Sketch**: in movepicker good/bad split (`src/movepicker.rs` capture
  loop), compute `threshold = -caphist_score / MOVEPICK_SEE_DIV(≈8)`;
  clamp to `±Queen`; SEE-check against that threshold.
- **Magnitude/risk**: +2–5 Elo per Hobbes note; couples caphist magnitude
  into ordering. Risk medium — caphist scale sensitive; retune-on-branch.
  **Overlaps Hobbes E12** — de-duplicate before running.

### E4. Aspiration depth-reduction on fail-high (already in Coda)
- `aspiration.rs:60-64` vs `src/search.rs:2497-2503`. Confirmed clean.

## Confirmed-clean / Not-worth-porting

- **Material-phase eval scaling** (`eval.rs:6-18`). Coda already scales;
  Hobbes-note dropped item #17 (`hobbes.md:110-114`).
- **RFP dampening `beta+(eval-beta)/3`** (`negamax.rs:207-211`). Coda H0'd
  `(eval+beta)/2` at RFP (`experiments.md:1172-1175`).
- **Razoring**. Coda has it re-added (`src/search.rs:130-135`); structure
  identical.
- **child.fail_highs LMR bump** (`negamax.rs:441-444`). H0 #1157
  (`experiments.md:10124-10131`).
- **1000 × direct_check quiet bonus** — Coda has quiet_check_bonus.
- **SIMD SCReLU inference**. Coda's NNUE (v9 pairwise+hidden) is a superset.
- **Non-verified NMP** — SPSA overfit on Tcheran's search shape.
- **cut_node-gated NMP** — Coda already gates (`src/search.rs:3836`).
- **2-ply-only contHist** — Coda has 4 plies (1,2,4,6).

## Sources

- Search core: `~/chess/engines/tcheran/engine/src/engine/search/negamax.rs`,
  `search.rs`, `quiescence.rs`, `aspiration.rs`, `iterative_deepening.rs`,
  `time_control.rs`, `move_picker.rs`, `tables.rs`, `types.rs`.
- Params: `~/chess/engines/tcheran/engine/src/engine/params.rs:1-104`.
- Eval/NNUE: `~/chess/engines/tcheran/engine/src/engine/eval/eval.rs`,
  `nnue.rs`, `inference.rs`. Net catalog: `~/chess/engines/tcheran/NET.md`.
- TT: `~/chess/engines/tcheran/engine/src/engine/transposition_table.rs`.
- SEE: `~/chess/engines/tcheran/engine/src/engine/see.rs`.
- Threats bitboard source: `~/chess/engines/tcheran/engine/src/chess/game.rs:407-434`.
- Coda comparison: `/home/adam/code/coda/src/search.rs`,
  `movepicker.rs`, `threats.rs`; `/home/adam/code/coda/experiments.md`;
  `/home/adam/code/coda/engine-notes/hobbes.md`.
