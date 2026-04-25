# Research Threads — Data-Driven Proposals (2026-04-24)

Response to Hercules's five R-threads at the bottom of `next_ideas_2026-04-21.md`. Five parallel research agents, each surveying top-engine sources (Stockfish, Reckless, Obsidian, Viridithas, PlentyChess, Stormphrax, Alexandria, Ethereal, Halogen, Caissa, blackmarlin) plus Coda's own recent experiments / docs. Research-only — no implementation.

`experiments.md` remains the source of truth for resolved SPRTs.

## R1 — L1 sparsity: why λ=1e-4 didn't bite and three better paths

### Diagnosis

`docs/v9_sparsity_investigation_2026-04-19.md` reports λ=1e-4 probe achieved zero compaction beyond the 8.4% structural floor. Root cause is in Bullet's proximal operator:
`bullet/crates/acyclib/src/device/cpu/base.rs:311-322` applies `λ × lr` as the soft-threshold. At late-training LR ~5e-6 and λ=1e-4, effective threshold is 5e-10/step — well below Adam gradient-noise magnitudes of 1e-2. **λ=1e-4 was structurally doomed regardless of training length.**

### Cross-engine survey

Surveyed 10 engine trees for `l1_decay|l1_reg|proximal|soft_threshold|lasso` plus out-of-tree trainer references:

- **Zero engines do FT weight-row sparsity.** Bullet's proximal (in our fork) is the only in-tree implementation; mainline Bullet has none.
- "Sparse L1" in Stockfish/Reckless/PlentyChess/Obsidian/Alexandria refers to **activation sparsity** (NNZ indices from SCReLU/pairwise zeros), not weight sparsity. Different mechanism; doesn't prune FT rows.
- **Only positive L1-regularisation Elo result in the surveyed field:** Viridithas "sibilant" net at `viridithas/notes/networkhistory.txt:1844-1851`. L1 on L0 **output activations** (not weights), +5.9 ± 3.11 Elo LTC H1 (11,660 games). The preceding "flounce" (HardSwish6 to drop sparsity 70→50%) H0'd at −0.95. **Takeaway: structured FT weight-row sparsity is unreported territory.**

### Three ranked paths

| Path | Sparsity yield | Elo | Risk | GPU |
|---|---|---|---|---|
| **1. λ=1e-3 + decouple-λ-from-lr** | 15-35% above floor | −5 to −20 untuned; retune candidate | Moderate (may over-zero in late training) | 1× SB800 (4-8h) |
| **2. Late-only L1** at λ=5e-3 from SB640→SB800 | Similar yield; lower feature-learning disruption | Neutral to +3 | Low | Same |
| **3. Cold-row-only L1** (per-row λ mask from activation histogram) | 40-60% targeted | Likely neutral; prunes only what's already dead | Low; needs masked-proximal Bullet op (~20 LoC) | Same |

**Key 3-line Bullet change (angle 1):** decouple `l1_decay` from `lr` — use `λ · max(lr, 1e-4)` instead of `λ · lr`. Probably doubles effective sparsity at fixed λ. Currently the last 30% of training applies *no* effective regularisation.

### SB150 probe protocol

Do NOT commit to SB800 first. Run 1h SB150 probes per path. Decision rules:
1. Compacted rows ≥15% at SB150 (proves L1 is biting)
2. Eval RMS within ±20% of baseline 580 (no catastrophic scale drift)
3. Loss curve monotonic (no NaN, no sawtooth)

If any fails → abandon path. ~1h GPU per probe.

### Recommended priority

1. **λ=1e-3 + decouple** (cheapest, most falsifiable)
2. **Late-only L1** (fallback if path 1 regresses Elo)
3. **Cold-row-only L1** (best theoretical yield; most engineering)

Skip path (b) 768→640 alone (brute force). Skip path (c) L1-on-differential (implementation cost exceeds expected yield when factoriser is already on-disk).

**Full report:** agent output; ~800 words with citations to Viridithas networkhistory, Bullet proximal site, Coda sparsity investigation.

## R2 — LMP adaptive + NMP TT-capture: why direct ports H0'd

### LMP survey (11 engines)

All top engines split cleanly into **single-row** (`(B + d²)/(2-imp)`) and **two-row** (independent factors for improving vs not):

- **Single-row:** Coda (B=10), Stockfish (B=3), Obsidian (B=3), Stormphrax (B=3).
- **Two-row:** Viridithas, Ethereal, blackmarlin, Alexandria.
- **With history term:** PlentyChess, Stormphrax (blend `history` into margin).

**Coda's `LMP_BASE=10` is 3.3× the consensus.** Any "adaptive" formula ported from a B=3 engine over-prunes on the improving side once stacked onto Coda's already-high base. The `#708 LMP adaptive` regression (−6.7 Elo) is consistent with porting a two-row formula's `(factor0, factor1)` values into Coda's single-row shape — mathematically inconsistent.

### NMP TT-capture survey

No top engine does a **pure binary TT-capture skip**. The two that skip use multi-condition gates:
- **Reckless** (`search.rs:555-557`): `tt_bound == Lower AND capture AND victim ≥ Knight` (three stacked conditions).
- **Ethereal** (`search.c:515,518`): `ttHit AND ttBound == UPPER AND ttValue < beta` — orthogonal: "don't NMP when TT says fail-low".

Two engines use softer **R+=1**:
- **Obsidian** (`search.cpp:882`): `R += ttMoveNoisy`
- **Viridithas** (`search.rs:1098-1108`): `r += (tt_capture.is_some()) as i32`

Coda already does post-capture R++ at `search.rs:2177-2179`. The direct-port `#709 NMP TT-capture skip` (−7.8) stacked two aggressive signals; `#724 narrower gate` settled at +0.1 — mechanism correct but headroom exhausted.

### Three concrete proposals

| Branch | Change | SPSA | Bounds | Expected |
|---|---|---|---|---|
| **A: `experiment/lmp-adaptive-two-row`** | Replace single-row with two-row `(B_i + Q_i·d²/100)/D_i` indexed by improving. 6 new tunables. | 6-param, 2500 iters | [−5,5] with retune-on-branch | **+3 to +6 Elo** |
| **B: `experiment/nmp-ttscore-upper-skip`** | Ethereal-style: skip NMP when `ttHit && ttBound==UPPER && ttValue + NMP_TT_MARGIN < beta`. 1 new tunable. | 1-param after SPRT | [−5,5] | **+2 to +4 Elo** |
| **C: `experiment/nmp-ttnoisy-rplus`** | Obsidian/Viri-style: `r += tt_move.is_capture() as i32`. No new tunable. | None | [−3,5] | **+2 to +4 Elo, safest** |

Proposal C is the lowest-risk — no tunables, orthogonal to existing post-capture R++. Suggest trying C first; if it H0s, proposal A is the bigger-leverage bet with acknowledged retune cost.

## R3 — NPS levers beyond cache layout

### Top unexploited lever: input-chunk L1 permutation

Viridithas has a baked `REPERMUTE_INDICES: [usize; L1_SIZE / 2]` (`nnue/network.rs:187-258, 693-725`). At load time, all FT-neuron-indexed tensors are permuted so that frequently-active neurons cluster in low indices. Pairs are preserved at `i` and `i + L1_SIZE/2`.

**Direct targets the 28× L1-dcache-miss gap** Coda exhibits (15.72% vs Reckless 0.55% per `coda_vs_reckless_nps_2026-04-23.md:121`). When hot activations occupy contiguous lanes, zero-skip fires in fewer branches and the same weight rows stay hot across evals.

- **Expected NPS:** +8-15% (hypothesis — one-hour measurement can falsify).
- **Effort:** S (1-2 days). Viridithas provides a copy-pasteable template.
- **Bench-exact:** yes (same algebraic sum, same quantisation).
- **Compatible with factoriser:** orthogonal; combine freely.

### Ranked queue

| # | Lever | NPS | Effort | Bench-exact | Priority |
|---|---|---|---|---|---|
| 1 | Eval-only TT writeback (already scoped in NPS doc) | +5-8% | S | No | **Top** |
| 2 | **L1 permutation** (Viridithas pattern) | **+8-15%** (hyp) | S (1-2d) | Yes | **High** |
| 3 | VNNI compile-time dispatch (82 runtime-flag reads today; Reckless uses `cfg(target_feature)`) | +3-5% | M (2-3d) | Yes | High |
| 4 | Prefetch ThreatEntry at make-move | +1-3% | S (hours) | Yes | Medium |
| 5 | AVX-512 setwise movegen (Reckless has; Coda's movegen already ahead) | +1-2% | L (3-5d) | Risk | Medium |
| 6 | SF-style smallnet / lazy-NNUE (requires second training) | +4-8% | L (weeks) | — | Defer |

**1-hour measurement script for #2:** dump activations from 10K positions, compute permutation, apply at load time, re-run `coda bench`. If L1 miss rate drops from 15.7% → ~5%, confirm; otherwise the miss rate is elsewhere (ThreatEntry, Finny) and we've falsified cheaply.

**Binary distribution** for #3: OpenBench already builds with `-Ctarget-cpu=native` per-worker; compile-time dispatch is zero-friction. Only cost is dev-machine builds don't run on older CPUs — handled by a `--features portable` path if needed.

## R4 — Post-Item-7 retune timing

### Recommendation: WAIT for factor SB800

**Reasoning:**
- Eval-TT writeback shifted evals/node 0.677 → 0.581 (-14%). Raw eval value unchanged, but pruning-consuming nodes shifted composition. Bench changed → not bench-exact → tree-shape change, modest magnitude.
- Historical retune-on-branch wins compound over 3-4 merges: #490 (+7.4 after 4 features), #586 (+6.2 after knight-fork+more). Single-feature retunes typically land +2-5.
- Realistic band for Item-7-only retune: **+2 to +5 Elo** (not Hercules's +3-6).
- Factor SB800 is imminent (~overnight train) and changes tree shape substantially — will de-calibrate any tune done now. Combined retune post-factor typically banks +6-8 vs split's +3+3.

### Top 6 likely-drifted tunables (for when retune runs)
`RFP_MARGIN_IMP/NOIMP`, `FUT_BASE/FUT_PER_DEPTH`, `SEE_QUIET_MULT`, `LMR_HIST_DIV`, `NMP_EVAL_DIV`, `HIST_PRUNE_MULT`.

### Three candidate branches for post-factor-SB800
1. `tune/v9-post-factor-sb800-r1` — 18-param `scripts/tune_pruning_18.txt`, 2500 iters.
2. `tune/v9-post-factor-rfp-futility-see-focused` — focused 8-param (the six above + `FUT_HIST_DIV`, `SEE_CAP_MULT`), 1200 iters for first-pass.
3. `tune/v9-post-factor-r2-full` — full 45-param follow-up for cumulative compound gain.

## R5 — Audit SPECULATIVE triage + fuzzer expansion

32 SPECULATIVE items from `correctness_audit_2026-04-22.md` classified by reachability (L/T/D/Z) × blast radius (S/C/H/M). Full ranking table in agent output.

### Top 5 Lichess-reachable (fix now)

**Status updated 2026-04-25** — 3 of 5 already merged in the 2026-04-22
audit batch, 1 in flight, 1 closed-on-re-analysis. Don't re-queue these.

1. ✅ **SEE pawn-promotion recapture** (`see.rs:76-93`) — merged
   `b25366d` 2026-04-22 (#652 +1.8 Elo H1).
2. ❌ **`should_stop` 4096-node granularity** — re-tested on
   post-tune-750 trunk: `fix/should-stop-granularity` SPRT #757
   **H0 −2.4 / 6516g 2026-04-25**. Re-test didn't flip the original
   #674 H0; 4096 is correct at STC. Drop.
3. ✅ **Evasion capture-promotion scoring** — fixed via C8 audit #25 +
   #26 in 2026-04-22 batch (mvv_lva path adds promotion delta;
   `is_cap` checked before `is_promotion` so capture-promotions take the
   capture-MVV-LVA score path).
4. ⏭ **Forced-move path zeros `soft_floor`** — re-analysed 2026-04-25:
   for genuine 1-legal moves, 10ms is the intended TM behaviour
   (preserves stockpile time on the clock for next non-forced move). Not
   a bug. Closed.
5. ✅ **Repetition scan missing `plies_from_null` cap** — merged
   `402e366` 2026-04-22 (`fix/rep-detection-null`, H0 −0.3 but kept as
   confident-correctness).

### Next 10 for fuzzer sweep (tournament-reachable)

**Status updated 2026-04-25** during the empty-fleet correctness sprint:

- ✅ **Duplicate SEE source-of-truth** — verified clean; movepicker.rs
  uses MVV-LVA scoring (different function class), `see.rs::see_ge` is
  the single SEE truth source. Closed.
- ❌ **SE `singular_beta` mate-bypass range** — `fix/se-singular-beta-mate-clamp`,
  SPRT #761 [-3, 3] **H0 −1.8 / 14412g 2026-04-25**. Mate-distance
  clamp removed legitimate SE in mate-shaped positions. Drop.
- ✅ **Recapture extension `ply>0` guard** — **merged 2026-04-25**.
  `fix/recapture-ext-ply-guard`, SPRT #758 [-3, 3] **H1 +1.5 / 16606g**.
- ✅ **FH blending inside SE** — **merged 2026-04-25**.
  `fix/fh-blend-skip-in-se`, SPRT #759 [-3, 3] **H1 +1.7 / 14764g**.
- ❌ **LMR `do_shallower` cp margin** — actioned across three cycles.
  #673 at 30cp H0'd, #679 at 20cp H1'd +1.4 (merged), #762 at 10cp
  **H0 −2.5 / 10482g 2026-04-25**. 20cp is the optimum.
- ⏭ **`patch-net` NNUE magic** — defensive only, offline tool.
  Skipped (no Elo path).
- ✅ **TT/tb_cache aarch64 Release/Acquire** — **merged 2026-04-25**
  as part of ARM-as-first-class commitment. Branch
  `fix/aarch64-tt-tbcache-ordering`, SPRT #764 [-5, 5] non-regression
  **−0.1 ±1.9 / 24886g** confirming x86 cost ≈ 0. ARM correctness
  benefit fleet-untestable but required.
- ✅ **`threats.rs:456` sq=63** — **merged 2026-04-25**.
  `fix/threats-blocker-bounds`, SPRT #760 [-5, 5] **H1 +0.9 / 17094g**.
- ⏭ **SCReLU `>>8` vs `÷255` drift** — 0.8% drift, absorbed into SPSA tune.
  Cosmetic; would require recalibrating eval-aware tunables. Skipped.
- ⏭ **`sample-positions` filter alignment** — offline tool, not search-
  affecting. Skipped.

**Net for next-10 (final 2026-04-25):** 5 merged (recapture-ply, FH-skip-in-SE,
threats-bounds, aarch64-ordering, plus se-promo from earlier batch), 2 H0
(SE-mate-clamp, lmr-shallower-10), 3 skipped with rationale, 1 verified-clean.
Of the merged: +5.4 Elo total banked from this audit slice.

### Fuzzer expansion programme (5 categories)

| Cat | Scope | Example invariant | Cost |
|---|---|---|---|
| **A** Movegen / SEE | SEE consistency; ordering invariants; `is_pseudo_legal` multi-field mutation | SEE queen-promo-capture > plain capture; multi-field corruption on EP must be rejected | CI-gated, ~1M positions |
| **B** Time / search | Short-budget deadlines; rep-scan bounds; SE boundary | Wall-clock ≤ hard + 20ms; rep only within `min(halfmove, plies_from_null)`; no recapture ext at ply=0 | Nightly |
| **C** SIMD / NNUE | SIMD-tail vs scalar at odd sizes; SCReLU scale drift watchdog | Bit-identical across AVX2/AVX-512/NEON at `h ∈ {16, 32, ..., 1536}` | CI matrix |
| **D** Threats | Overflow at MAX_THREAT_ACTIVE; `push(NO_MOVE)` absorb invariant | No overflow on quiet-filtered positions; absorb within same node | One-shot |
| **E** TT hygiene | Stub-entry consumers (new #713 eval-only path); aarch64 reorder stress | No consumer reads `tt_score` with `tt_depth < 0` | CI + runtime asserts |

Existing fuzzers: `fuzz_is_pseudo_legal_rejects_corrupted`, `fuzz_incremental_vs_full_hash`, SIMD-vs-scalar. Extend A3 to mutate (from, to, flag) triples when `ep_square != NO_SQUARE` — this is the class that missed the CRITICAL C1 `is_pseudo_legal` EP hole.

### Fix-when-touched tail

16 items: dead code (`compute_move_deltas`, `binpack.rs`), unreachable branches (`fixup_move_flags` king-by-2, `promotion_piece_type` flags≥8), test-only (`cuckoo_entries_valid_moves`). No proactive SPRT cost.

## Cross-cutting observations

- **Multiple Hercules-proposed paths had structural flaws visible from cross-engine survey.** λ=1e-4 for sparsity (Bullet proximal coupled to lr, threshold 10⁹× below gradient noise). Direct-port LMP (single-row engine formula with two-row constants). Binary NMP TT-capture skip (no top engine does pure binary). In each case the reformulation is small (decouple λ, use two rows, use R+=1) and likely wins.
- **Highest-leverage lever not yet tried: L1 permutation** per Viridithas. Cheap to measure, likely biggest single NPS win, orthogonal to factoriser and sparsity work. One-hour measurement runs first.
- **R4 WAIT recommendation is the only "don't do it now" finding.** All other threads have concrete next steps.

## Queue for Hercules

Suggested order (parallel where possible):

1. **L1 permutation measurement** (1h) — falsifies or confirms R3's top lever before committing engineering.
2. **Proposal C: `experiment/nmp-ttnoisy-rplus`** — lowest-risk R2 proposal, ship if H1.
3. **λ=1e-3 + decouple-λ-from-lr** — R1 path 1, SB150 probe first.
4. **VNNI compile-time dispatch** — R3 item 3, independent work stream.
5. **SEE pawn-promotion inner loop + evasion capture-promo** — R5 top-5 low-risk correctness fixes.
6. **Proposal A: `experiment/lmp-adaptive-two-row`** — R2 bigger-leverage bet, run after trunk stabilises.
7. **Post-factor-SB800 retune** — R4, once factor SB800 lands.

## Companion docs

- `move_ordering_understanding_2026-04-19.md`, `signal_context_sweep_2026-04-19.md` — earlier pattern-analysis.
- `next_ideas_2026-04-21.md` — Hercules's R-threads this doc addresses.
- `peripheral_mechanisms_2026-04-22.md` — parallel non-pure-search idea catalogue.
- `correctness_audit_2026-04-22.md` — source of R5.
- `factoriser_design_2026-04-21.md` — R1 orthogonal path.
- `v9_sparsity_investigation_2026-04-19.md` — R1 baseline measurements.
- `coda_vs_reckless_nps_2026-04-23.md`, `nps_microbench_hostdata.md` — R3 baseline.
- `experiments.md` — source of truth for resolved SPRTs.
