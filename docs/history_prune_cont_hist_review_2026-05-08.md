# History Pruning & Continuation History Cross-Engine Review (2026-05-08)

Reviewed: Stockfish, Reckless, Obsidian, Viridithas, Alexandria, PlentyChess,
Berserk, Caissa, Stormphrax, Halogen — top 11 of the 38-engine RR plus
Reckless (top 2) and Halogen (top 10). Source paths in `~/chess/engines/`.

## Summary findings

**Coda is a structural outlier on both features.** Each anomaly is concrete
and directly explains one of the two reported symptoms:

1. **History pruning fires rarely** — three independent reasons compound:
   (a) `depth ≤ 3` raw-depth cap is the **tightest in the field** (peers
   use `lmrDepth ≤ 5-7` or no cap), (b) threshold magnitude `-12825*d` is
   **~2-3× tighter than median peer** (-4000 to -7500), (c) gates on
   `mv != tt_move` which most peers don't (TT move emitted first
   naturally). The conjunction makes the gate nearly-unreachable.

2. **CONT_HIST_MULT trains to floor = 1** — three structural mismatches:
   (a) write asymmetry `[bonus, bonus/2, bonus/2, bonus/2]` at {1,2,4,6}
   is **non-standard** (most peers write uniform `bonus`); (b) `cm` couples
   ply-1 and ply-2 reads at the same multiplier, but the signal asymmetry
   peers exploit is between **ply-1 and the rest** (Berserk `[2,2,1,1]`,
   PlentyChess `[2,1,1,½]`, Stormphrax `[1,1,½]`); (c) no engine outside
   Coda has a single shared multiplier across two adjacent offsets — they
   either use flat 1× (SF/Reckless/Obsidian/Alexandria/Halogen), or
   per-offset SPSA tunables (Viridithas/Caissa), or hardcoded shape
   (Berserk/PlentyChess/Stormphrax). SPSA→1 is the only way our shape
   can express "deeper plies less informative" given asymmetric writes.

## History Pruning — cross-engine table

| Engine | Depth gate | Threshold | History sources |
|---|---|---|---|
| Stockfish | **none** | `-4097*d` | cont1 + cont2 + pawn |
| Reckless | n/a (folded into futility/SEE/LMR thresholds) | hist as continuous adjustment | quiet + cont1 + cont2 |
| Obsidian | **none** | `-7471*d` | main + pawn + cont1 + cont2 + cont4 |
| Viridithas | `lmrDepth < 7` | `-3186*(d-1)` | main + cont1·37 + cont2·33 + cont4·13 |
| Alexandria | **none** | `-3753*d` | HH + cont1 + cont2 + cont4 |
| PlentyChess | `lmrDepth < 4.68` (also captures via separate factor) | `-67.96*d_plies` over amplified sum | quiet + 2·cont + pawn |
| Berserk | `lmrDepth < 5` | `-2788*(d-1)` | HH + cont1 + cont2 + cont4 |
| Caissa | `depth < 9` | `-234*d - 148*d²` | main + cont1 + cont2 + cont4 |
| Stormphrax | `lmrDepth ≤ 5` quiet, `depth ≤ 4` noisy (d²) | `-2314*d - 1157` | main + pieceTo + cont |
| Halogen | **none** | `-1650*d - 421` | pawn + threat + cont1 + cont2 |
| **Coda** | **`depth ≤ 3`** | **`-12825*d`** | main + cont1 + pawn |

**Coda is uniquely conservative on every axis.**

Notable patterns from peers:

- **5/10 engines have NO depth cap** (SF, Obsidian, Alexandria, Halogen,
  Reckless's analog). They self-gate via the threshold's depth scaling.
- **All engines using a depth cap use `lmrDepth`, not raw `depth`** — Coda's
  `raw_depth ≤ 3` is the only raw-depth gate in the set.
- **All thresholds in -2000 to -7500 range** for d=1 — Coda's -12825 is the
  outlier. Quadratic-in-depth is also a non-trivial second-order effect
  (Caissa, Stormphrax noisy variant).
- **TT-move exemption is rare**. Most engines order TT move first via the
  movepicker stage, so it never reaches hist-prune. Coda explicitly checks
  `mv != tt_move` which is structural redundancy.
- **Action on prune: peers use `skipQuiets = true`** (Stormphrax, Halogen,
  Alexandria, Obsidian) which sets a flag halting all subsequent quiets at
  this node, not just `continue` for the current move. This compounds
  pruning effectiveness within a single node.

## Continuation History — cross-engine table

Read weights are at move ordering / pruning unless noted.

| Engine | Read offsets & weights | Write offsets & magnitudes | R/W symmetric? |
|---|---|---|---|
| Stockfish | {1,2,3,4,6} flat 1× | {1,2,3,4,5,6} asymm: {1071,753,329,539,124,434}/1024 + dynamic consistency mul | **No** — flat reads, steeply tapered writes |
| Reckless | {1,2,4,6} flat 1× ordering; {1,2} pruning | {1,2,4,6} **uniform `bonus`** | Yes |
| Obsidian | {1,2,4,6} flat 1× ordering; {1,2,4} pruning | {1,2,4,6} `[1,1,½,½]` | Partial (writes taper deeper) |
| Viridithas | {1,2,4} weighted **37, 33, 13** | {1,2,4} per-offset SPSA-tuned independently | No (writes much larger on cont1) |
| Alexandria | {1,2,4,6} flat 1× ordering; {1,2,4} pruning | {1,2,4,6} **uniform `bonus`** | Yes (symmetric writes) |
| PlentyChess | {1,2,4,6} `[2,1,1,½]` | {1,2,3,4,6} `[1,1,¼,1,½]` | Both asymmetric, different shapes |
| Berserk | {1,2,4,6} `[2,2,1,1]` ordering; {1,2,4} pruning | {1,2,4,6} **uniform `bonus`** | No (reads weighted, writes flat) |
| Caissa | {1,2,4,6} `[1.0, 1.0, 0.55, 0.57]` | {1,2,3,4,6} per-offset SPSA-tuned | Yes (matched per-offset) |
| Stormphrax | {1,2,4} `[1, 1, ½]` | {1,2,4} **uniform `bonus`** | Partial (reads taper) |
| Halogen | {1,2} pruning, {1,2,4} ordering, flat 1× | {1,2,4} **uniform `bonus`** | Yes |
| **Coda** | **{1,2,4,6} `[cm, cm, 1, 1]`** | **{1,2,4,6} `[bonus, bonus/2, bonus/2, bonus/2]`** | **No, in opposite directions** |

**Coda's `[cm, cm, 1, 1]` × `[bonus, b/2, b/2, b/2]`** is unique:
- Reads weight ply-1 and ply-2 the SAME, weight ply-4 and ply-6 the SAME
- Writes weight ply-1 strongest, the other three EQUALLY half
- The asymmetry shapes don't match. The shape SPSA wants
  ("ply-1 strongest, taper") isn't expressible as a single `cm`.

**Other consistent peer patterns:**

- **(in_check, is_capture) bucketing of the cont-hist table** — Obsidian splits
  by `isCap` of parent move; Reckless splits by `(in_check, is_noisy)` of
  parent; Berserk splits by `isCap` of parent; PlentyChess splits by
  `(in_check, was_capture, stm)`. **Coda does not bucket.** This is a
  separate gap (4× table size for cleaner gradient signal).
- **Non-cutoff penalty patterns**:
  - SF: decay multiplier `*977/1024` per move iterated through (cumulative
    softening as you go down the list — Coda doesn't decay).
  - Reckless: malus has 3× larger linear coefficient than bonus
    (`352*d` vs `108*d`).
  - Berserk + Obsidian: post-LMR-research nudge — successful re-search
    fail-high adds `+statBonus`, fail-low adds `-statMalus`, all to
    cont-hist. **Coda only updates on beta cutoff**, missing this signal.
- **Bonus formula shape**: linear (Coda, SF, Obsidian, Reckless, Alexandria,
  Berserk, Stormphrax, Halogen) — universal. Caissa is `linear+quadratic`.
  Berserk has both: `min(1729, 4·d² + 164·d - 113)`.
- **Bonus depth boost**: SF and Obsidian boost `depth+1` when fail-high
  exceeds beta by margin (`bestScore > beta + StatBonusBoostAt`). Coda
  doesn't.

## Instrumentation across engines

Almost universal: **dbg_hit / dbg_stats / dbg_mean_of-style infrastructure
exists but is NOT shipped live in search code**. Pattern across SF, Reckless,
Obsidian, Alexandria, PlentyChess, Caissa: defined in misc.h / debug.h, used
ad-hoc during investigation, stripped before commit. Live production
counters are absent.

Coda is **already best-in-class on this axis** — bench prints actual
production counters (`History prunes: N`, hit rates, refresh causes, etc.).
We can read these at any time without re-instrumenting.

For this specific investigation, **add two more counters before any
experiment**:
- Distribution of `hist_prune_score` magnitudes at the gate site, bucketed
  (e.g. histogram of `score / -depth` ratio). Tells us how close current
  scores get to the threshold.
- Cont-hist read magnitude per offset, sampled at scoring loop entry.
  Tells us whether ply-2/4/6 entries are saturating, near-zero, or in the
  noise band.

## Specific hypotheses + ranked experiments

### Symptom 1 — history pruning fires rarely

Three independent compounding causes (rank ordered by likely leverage):

**Experiment A1: replace `depth ≤ 3` with `lmrDepth ≤ 5`**
- Modal peer choice (Berserk, Stormphrax). Roughly doubles the candidate
  pool. Coda's prior `improving && unstable` gate removal already opened
  ~50% of candidates per `cross_engine_comparison_2026-04-25.md` item #10
  — this completes the analogous broadening on the depth axis.
- File:line `src/search.rs:2771`. One-line change of `depth` → `lmr_d`.
- Risk: fires too often → bad pruning. Mitigated by retune of HIST_PRUNE_MULT.

**Experiment A2: lower `HIST_PRUNE_MULT` to peer-median**
- Current 12825 vs peer median ~4000-5000. Bring to ~5000-6000 and SPSA
  retune around new value.
- Likely needs to bundle with A1 since loosening threshold without
  loosening depth is mostly noise-amplification.

**Experiment A3: drop the `mv != tt_move` gate**
- Pure cleanup if TT move is ordered first (which it is in Coda's
  movepicker — verify with bench-eq SPRT).
- Modest leverage but removes structural redundancy.

**Experiment A4: `skipQuiets = true` action on hist-prune**
- Match Stormphrax/Halogen/Alexandria pattern. Once a quiet's history
  is below threshold, skip ALL subsequent quiets at this node (they're
  ordered worst-last, so anything after is at least as bad).
- `bool skip_quiets` flag, set on first prune, gates quiet stage in
  movepicker via `mp.skip_quiet = true`.

**Bundle**: A1 + A2 + A3 + A4 retune-on-branch. Estimate +3-8 Elo.

### Symptom 2 — CONT_HIST_MULT floors at 1

Two independent fixes; B1 first because it's the smallest change.

**Experiment B1: make writes symmetric (uniform `bonus` at all offsets)**
- Match Reckless, Berserk, Alexandria, Stormphrax (most common peer
  pattern). Replace `bonus/2` with `bonus` at offsets {2,4,6}.
- File:line `src/search.rs:3191, 3231, 3194` (cutoff bonus + non-cutoff
  malus). 4-line change in each block.
- Hypothesis: with symmetric writes, deeper offsets accumulate at full
  magnitude. SPSA will then find `CONT_HIST_MULT` > 1 useful again.
- Bench-test: should be approximately bench-neutral (same node count)
  unless a re-tune is applied.

**Experiment B2: per-offset read weights as separate SPSA params**
- Replace `let weights = [cm, cm, 1, 1]` with
  `let weights = [tp(&CH_W1), tp(&CH_W2), tp(&CH_W4), tp(&CH_W6)]`.
- 4 new SPSA-tunable parameters, each independently tunable in 1-128
  range with c_end ~10. Drops the structural mismatch entirely.
- Higher effort (param surface grows by 4) but matches Viridithas/Caissa
  highest-Elo-per-effort approach in this space.
- Skip if B1 alone fixes the symptom (CONT_HIST_MULT lifts off floor=1
  after B1 retune).

### Bundle order recommendation

1. **Add instrumentation** (hist_prune score histogram + per-offset
   cont-hist magnitudes). Run bench, capture numbers, document baseline.
2. **B1 alone**: write symmetry. SPRT [0, 3] for direct effect, then
   SPSA retune CONT_HIST_MULT specifically (1-param, ~800 iters).
3. **A1+A2+A4 bundle** with retune on branch. A3 ride-along if it
   bench-equals.
4. If `CONT_HIST_MULT` retune lands clearly above 1 after B1, leave it.
   If still floored, do **B2** (per-offset weights, drop the shared
   multiplier).
5. Optionally: **(in_check, is_capture) bucketing** of cont-hist table —
   significantly more invasive (4× table, every read/write site) but
   peer-consensus pattern across 4 engines. Defer until B1+B2 settles.

## Files referenced

Coda: `src/search.rs:2771-2790`, `src/search.rs:2998-3001`,
`src/search.rs:3185-3253`, `src/movepicker.rs:631-643`,
`src/movepicker.rs:854-867`.

Per-engine paths in agent reports above; key citations:
- Stockfish `search.cpp:1102-1110, 1853-1914`, `movepick.cpp:163-167`
- Reckless `search.rs:705-788, 1342-1349`, `movepick.rs:206-210`,
  `history.rs:166-180`
- Obsidian `search.cpp:1006-1009, 220-226`, `movepick.cpp:90-97`
- Viridithas `search.rs:1310, 1663-1697, 99-107`,
  `historytable.rs`, `movepicker.rs:198-219`
- Alexandria `search.cpp:722-725`, `history.cpp:68-82, 142-150, 193-208`
- PlentyChess `search.cpp:995-998, 125-127`, `history.cpp:113-185`
- Berserk `search.c:625-631`, `history.h:29-77`, `movepick.c:64-68`
- Caissa `Search.cpp:128-131, 1730-1762`, `MoveOrderer.cpp:148-166, 237-246`
- Stormphrax `search.cpp:932-955`, `history.h:75-260`
- Halogen `search.cpp:1024-1117, 1239-1249`, `data.cpp:102-181`
