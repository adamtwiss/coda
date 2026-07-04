# Uralochka3 Search/Eval Review (2026-04-19)

C++, long-running Russian engine by Alexander Kholupko. `~/chess/engines/uralochka3/`.
Not in the top-25 of the local RR; historical CCRL Blitz ~3550. Included because
of its long lineage predating the modern NNUE wave — carries older ideas some
engines have dropped. README: `README.md:1-46`; comments in Russian.

Single-file linear search (~1400 lines, `game.cpp`). No 4D history, no threats,
correction history is one pawn table, cont-hist only ply-1/2. Search-wise
**a generation behind Coda** — most items below are either already-in-Coda-and-
stronger, or legacy-shape variants worth flagging.

## Search architecture

- **PVS + aspiration** (`game.cpp:501-562`). Asymmetric widening: on fail-low
  `beta ← 0.48·alpha + 0.52·beta` and `alpha ← alpha − delta`; on fail-high
  `alpha ← 0.83·alpha + 0.17·beta`. The non-failing bound gravity-pulls toward
  the fail. Delta multiplier ≈ 1.22.
- **NMP** (`game.cpp:832-847`): `R = min(3.8, (eval−beta)/206) + 3.13 + d/2.4`.
  **Skipped when TT-upper says `tt_score < beta`** — Coda has no such gate.
  Returns raw `beta`, no verification.
- **RFP** (`game.cpp:822-826`): depth ≤ 13, `margin = 80·d − (improving? 80 :
  −9) − (tt_hit? −4 : 7)`. **TT-hit tightens the margin.** On cutoff, returns
  `0.27·eval + 0.73·beta` — a strong Hobbes-style beta blend.
- **ProbCut** (`game.cpp:849-876`): `probcut_beta = beta+116`, reduction of
  PROBCUT_DEPTH (≈7). No SEE gate on candidates.
- **Futility** (`game.cpp:967-978`, "as in Ethereal"): two tiers on lmr_depth.
  Tier 1: `eval + fut ≤ alpha AND hist < ~11.7k/6.0k`. Tier 2 unconditional at
  `+FUT_MARGIN_2 (~160)`.
- **LMP** (`game.cpp:963`): `d ≤ 8 && moves_all ≥ base + coef·d²/4.5`. **Also
  fires in PV** (`_prune_pv_moves_count`, default on) — unusual.
- **Cont-hist prune** (`game.cpp:979-982`): `min(hist_counter, hist_follower) <
  thresh` requires *both* histories bad. Coda uses summed cont-hist.
- **SEE pruning** (`game.cpp:986-988`): quiet linear `-66·d`, capture quadratic
  `-17·d²`, on raw depth. Coda is `d²` for both.
- **LMR** (`game.cpp:1055-1074`): base `ln(d)·ln(mc)/2.46`. Modifiers `−1 pv,
  +1 non-pv, +1 !improving, −1 killer, +1 king-move-in-check, +1 tt_move-was-
  capture, +2 cut_node, −clamp(hist/5176, −2, 2)`. No do_deeper, no tt_pv, no
  SE-margin axis.
- **Extensions** (`game.cpp:999-1036`): singular ext with margin `beta_cut =
  tt_val − depth` (**depth-scaled**, not fixed); double ext bounded by per-path
  `SINGULAR_EXTS≈8` counter. Multi-cut when `beta_cut ≥ beta`. Negative SE
  outcomes: `(pv?1:0)−2` if `tt_val≥beta`, `−2` cut_node, `−1` if `tt_val≤res`.
  SE gated on `tt_flag == BETA` only. +1 for check or promotion.
- **Correction history** (`game.cpp:1397-1401, 1160-1163`, `moves.h:29`): **one
  pawn table only**, `[color][hash&32767]`, gravity 1024, bonus `(best−eval)·
  d/8` clamped ±256. No non-pawn / minor / major / cont / move-keyed sources.
- **Eval outlier compression** (`evaluate.cpp:32-35`): `|eval|>2000 → ±2000 +
  (eval∓2000)/5`. 5× attenuation of large evals.
- **SMP**: Lazy SMP (`game.cpp:219-311`). Standard.
- **Legacy switches disabled in place**: razoring (`game.cpp:819`), alpha-
  pruning (`game.cpp:829`), classic IID (`game.cpp:880-894`). These shapes
  were tried and dropped.

## Move ordering

Stages (`moves.h:52-63`): HASH → GEN_KILLS → GOOD_KILLS → GEN_QUIET → KILLER1
→ KILLER2 → COUNTER → QUIET → BAD_KILLS. **Killers/counters still real**; Coda
dropped both (`movepicker.rs:6-7`, e28c78a).

- Butterfly `_history[color][from][to]` (`moves.h:26`) — no threat buckets.
- Continuation: TWO tables ply-1 ("counter") + ply-2 ("follower"), each
  `[piece][to][piece][to]`, weighted `0.90·counter + 0.80·follower`
  (`moves.cpp:174-181`). Coda already has ply-1/2/4/6.
- Dedicated counter-move table `_counter_moves[color][piece][to]` filling the
  COUNTER stage (`moves.cpp:104-107`).
- **No capture history.** Captures ordered by SEE only.
- History bonus **quadratic**: `A·d² + B·d + C` with separate pos/neg curves,
  clamped [0, MAX] (`moves.cpp:121-122`). Gravity: `node += 32·bonus −
  node·|bonus|/512` (`moves.cpp:769-772`). Coda uses linear.

## NNUE / eval

Architecture (`neural.h:8-31`, README): `HalfKA (16 king buckets × 12 × 64) ×
2 perspectives → 1280 + 6 PSQT → CReLU pairwise → 6·(32 SCReLU → 32 ReLU → 1)
+ PSQT bucket`. 6 PSQT outputs interpolated by piece-count stage
`max(0, (pieces−3)/5)` (SF-style). L1=1280 (bigger than Coda 768), NO threats,
NO horizontal mirror. Koivisto king-bucket table (`neural.cpp:39-52`).

PSQT delta bonus (`neural.cpp:778-781`): `psqt = (to_move[HIDDEN+stage] −
opponent[HIDDEN+stage])/2` added to network output — cheap material residual.

Full Ethereal-style Texel-tuned classical eval (`evaluate.cpp:40-105`) is
**compiled but disabled** (USE_NN flag). Historical relic.

## Notable / novel mechanisms

1. **NMP gated on TT-upper below beta** (`game.cpp:835`) — no Coda prior art.
2. **RFP margin adjusted by TT-hit** (`game.cpp:822-825`) — no Coda prior art.
3. **RFP fail-high blends 0.27·eval + 0.73·beta** (`game.cpp:826`) — same
   family as Coda's H1 NMP-return dampen (`experiments.md` #1673 +12.8).
4. **`min(counter, follower)` cont-hist prune gate** (`game.cpp:981`) —
   distinctive from summed variant.
5. **Asymmetric aspiration widening** (`game.cpp:539-552`).
6. **Eval 5× compression above ±2000cp** (`evaluate.cpp:32-35`).
7. **Depth-scaled SE margin `tt_val − depth`** (`game.cpp:1008`).
8. **LMP in PV nodes** (`game.cpp:957-963`).

## Testable Experiments for Coda

### E1. NMP gated on TT-upper bound

- **Uralochka**: `game.cpp:834-835` — skip NMP if `tt_hit && tt_flag==UPPER &&
  tt_score < beta`. TT already says fail-low; null won't validate fail-high.
- **Coda today**: `src/search.rs` NMP has no TT-bound guard.
- **Prior art**: grep `NMP.*ttalpha`, `null.*hash.*alpha`, `NMP.*ALPHA`,
  `NMP.*ttHit` in `experiments.md` → 0 hits.
- **Sketch**: one-line extension to NMP eligibility. No tunable.
- **Magnitude/risk**: +1-4 Elo. Consensus-shape idea (SF variants). Low risk.

### E2. RFP margin tightened on TT-hit

- **Uralochka**: `game.cpp:822-825` — `margin -= (tt_hit? 4 : -7)`.
- **Coda today**: `src/search.rs:3738` — no TT-hit component in the RFP margin
  (root-depth, knee, pawn-threat, unstable are the axes).
- **Prior art**: grep `RFP.*hashhit`, `BETA_HASHHIT`, `rfp.*tt.*hit` → 0.
- **Sketch**: `margin -= RFP_MARGIN_TT_HIT * tt_hit_bool` (~10cp). One tunable.
- **Magnitude/risk**: +1-3 Elo. Retune-on-branch expected (RFP margin already
  crowded).

### E3. RFP fail-high blends toward beta

- **Uralochka**: `game.cpp:826` — `return (1-0.73)·eval + 0.73·beta`.
- **Coda today**: RFP returns `static_eval - margin`. Coda already blends
  fail-high in main search and dampens NMP (`experiments.md` #1673 H1 +12.8).
- **Prior art**: grep `rfp.*blend`, `rfp.*lerp`, `RFP_RETURN` → 0. QS
  stand-pat blend was H0 (#2009 −11) but that's a clean eval; RFP static-null
  is a heuristic cutoff, closer to NMP-return.
- **Sketch**: `return (eval*1 + beta*3) / 4`, mirror NMP-dampen shape.
- **Magnitude/risk**: +1-3 Elo. Bundle with E2 as one RFP-shape SPRT.

### E4. `min(counter_hist, follower_hist)` cont-hist prune gate

- **Uralochka**: `game.cpp:979-982` — requires *both* ply-1 and ply-2 histories
  to signal bad.
- **Coda today**: `src/search.rs:4576-4586` — summed ply-1 + ply-2 + ply-4 +
  ply-6 for both ordering and pruning.
- **Prior art**: grep `min.*counter.*follower`, `min.*cont.*hist.*prune` → 0.
- **Sketch**: additional gate at the hist-prune site using ply-1/ply-2 min,
  new tunable, keep sum gate as fallback.
- **Magnitude/risk**: +0-2 Elo, retune-on-branch. Speculative — Coda's 4-ply
  cont-hist is richer than the 2 Uralochka gates on. Rank after E1-E3.

### E5. Asymmetric aspiration widening

- **Uralochka**: `game.cpp:539-552` — on fail, pull the non-failing bound
  toward the failing side by 0.17/0.52; delta multiplier ≈ 1.22.
- **Coda today**: standard exponential widening, other bound unchanged.
- **Prior art**: grep `aspiration.*shift`, `asymmetric aspiration` → 0.
- **Sketch**: on fail-low, `beta = (alpha + beta) / 2` (or tunable weight);
  mirror on fail-high.
- **Magnitude/risk**: +0-2 Elo. LTC-only effect (STC aspiration is a no-op).
  Low effort but LTC-testing overhead.

### E6. Post-NNUE outlier compression above ±2000cp

- **Uralochka**: `evaluate.cpp:32-35`.
- **Coda today**: EVAL_SCALE + halfmove + material-phase scaling, no absolute-
  magnitude compression.
- **Prior art**: grep `outlier compression`, `clamp.*2000`, `clip.*eval` → 0.
- **Sketch**: `if eval > 2000: eval = 2000 + (eval-2000)/K`, K=5. One tunable.
- **Magnitude/risk**: −1 to +1 Elo. Speculative. Do only after E1-E5 and if
  idle SPRT capacity. Sanity-check RMS first (CLAUDE.md EVAL_SCALE warning).
  Skip if unsure.

## Confirmed-clean / Not worth porting

- **Correction history**: 1 source (pawn). Coda has 5.
- **Continuation history plies**: (1,2) — Coda has (1,2,4,6).
- **LMR**: no do_deeper/shallower, no tt_pv gradient, no SE-margin axis.
- **Killers/counter-move stages**: Coda dropped both (SPRT-validated).
- **Depth-scaled SE margin**: SE is fragile in Coda (multiple H0s per CLAUDE.md);
  do not port SE variants until SE is confirmed healthy.
- **NNUE architecture (1280 + PSQT, 16 buckets, no threats)**: Coda v9's
  threats are a bigger win than accum width.
- **PSQT delta bonus**: redundant with Coda's output buckets.
- **PV LMP**: Coda's `!is_pv` gate is consensus; enabling is presumably weaker.
- **Classical Texel eval**: unused in Uralochka too; historical curiosity.

## Sources

- `~/chess/engines/uralochka3/README.md:1-46`
- `~/chess/engines/uralochka3/src/game.cpp:501-562` aspiration
- `~/chess/engines/uralochka3/src/game.cpp:822-847` RFP + NMP
- `~/chess/engines/uralochka3/src/game.cpp:849-905` ProbCut + IIR
- `~/chess/engines/uralochka3/src/game.cpp:963-988` LMP/futility/cont-hist/SEE prune
- `~/chess/engines/uralochka3/src/game.cpp:999-1036` SE + extensions
- `~/chess/engines/uralochka3/src/game.cpp:1055-1091` LMR + PVS re-search
- `~/chess/engines/uralochka3/src/game.cpp:1160-1163, 1397-1401` pawn corrhist
- `~/chess/engines/uralochka3/src/moves.cpp:82-311, 769-777` picker + history/corrhist gravity
- `~/chess/engines/uralochka3/src/moves.h:24-31` history table shapes
- `~/chess/engines/uralochka3/src/neural.h:8-31`, `neural.cpp:692-782` NNUE inference + PSQT delta
- `~/chess/engines/uralochka3/src/evaluate.cpp:32-38` outlier compression
- `~/chess/engines/uralochka3/src/tuning_params.cpp:280-368` SPSA defaults
- `/home/adam/code/coda/src/search.rs:3735-3767` Coda RFP (E2/E3 comparison)
- `/home/adam/code/coda/src/search.rs:4576-4586` Coda cont-hist read (E4)
- `/home/adam/code/coda/src/movepicker.rs:6-38` Coda picker + history shape
- `/home/adam/code/coda/experiments.md` #1673 NMP-return dampen H1 +12.8 (E3 precedent)
