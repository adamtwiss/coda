# Threat-accumulator & SF speed-gap findings (2026-06-15)

Consolidated measurements from the Coda-vs-SF speed investigation. Companion
to `coda_vs_sf_speed_2026-06-14.md` (the broader profile); this doc is the
hard-data ledger for the threat-accumulator deep-dive.

**Headline:** SF runs the *same* architecture (FT1024 + threats) on the *same*
hardware and is ~44% faster single-thread / ~93% faster under 16× contention.
That contended gap ≈ **~120 STC Elo** (at ~130-140 Elo/NPS-doubling) — the
**majority of the ~150 Elo gap to SF**, and pure implementation efficiency (no
eval-quality risk). The threat accumulator is the dominant slice: Coda ~31% of
cycles vs SF ~5.5%.

All measurements on Hercules (16C/16T, AVX2), OB worker stopped for NPS,
gating build = `experiment/xray-ablation`, nets as noted.

---

## 1. NPS decomposition — where the SF gap lives

Same gating binary, X-ray on/off via same-net `CODA_NO_XRAY` isolation, vs
`stockfish bench`:

| | Coda **+xray** | Coda **−xray** | Stockfish |
|---|---|---|---|
| single-thread (best) | 793,748 | 881,829 (**+11.1%**) | ~1,140,000 |
| 16× contended (agg) | ~3,406,000 | ~3,892,000 (**+14.3%**) | ~6,589,000 |

- **X-ray costs ~11% single / ~14% contended NPS** — real, measurable.
- **X-ray is only ~25% of the single-thread gap, ~15% contended.** With X-ray
  removed, SF is STILL +29% single / +69% contended faster on the SAME
  architecture. **~75-85% of the gap is non-X-ray implementation headroom.**
- Net-difference confound: prod net is **FT1024** (793k), the `gpu4-normal`
  baseline is **FT768** (963k). FT1024 vs FT768 ≈ −18-21% NPS. The X-ray cost
  (~11%) is consistent across both nets.
- SF contention retention is better too: ~36%/process under 16× vs Coda ~27-28%.

## 2. X-ray: cost vs value — KEEP IT (settled)

- **NPS cost:** ~11% single / ~14% contended ≈ **~3-5 Elo at deployment** (LTC
  NPS→Elo ~15-25/doubling), ~20 at STC.
- **Eval value:** SPRT **#2014 H0, −187.5 ±19.1 Elo** (no-X-ray net vs X-ray-on
  baseline, SB800, gating build). Larger than the −157 ±17 at S200 (#2008).
- **Verdict:** ~187 Elo eval gain for ~3-5 Elo NPS cost. One of the best trades
  in the engine. X-ray is a real, large, **Coda-specific** advantage (SF and
  Reckless emit NO X-ray features). Do not remove.

## 3. Threat-apply volume — the core finding

The apply **kernel is identical** between engines (load i8 weight row, widen
i8→i16 on load via `vpmovsxbw`/`vec_convert_8_16`, add/sub into i16
accumulator, register-tiled). So per-delta cost is the same; the gap is **how
many weight rows each streams**.

**Deltas/apply-call** (instrumented both engines, `bench`):

| | deltas/call | vs SF |
|---|---|---|
| **Stockfish** | **4.46** | 1.0× |
| Coda −X-ray (core model) | 7.31 | 1.64× |
| Coda +X-ray | 10.05 | 2.25× |

SF streams ~2.25× fewer rows → that *is* the ~8.7%-vs-1.5% SIMD-apply slice.

**Caching-immune deltas/MOVE** (counted at generation, once per `make_move`):

| | deltas/move | deltas/apply-call | lazy-replay inflation |
|---|---|---|---|
| Coda +X-ray | **8.43** | 10.05 | ~19% |
| Coda −X-ray (core) | ~6.1 (est.) | 7.31 | — |

The deltas/**call** metric is inflated ~19% by lazy-replay depth + eval-cache
materialization frequency (see §5). The architecture-pure number is
**8.43/move with X-ray, ~6.1/move core**. Even discounting the inflation,
Coda's core threat model is ~1.6× denser than SF's per move.

**Decomposition of Coda's excess vs SF:**
- X-ray: +2.74 deltas/call (kept, +187 Elo).
- **Core-model excess: +2.85** — Coda's *non-X-ray* threat model is genuinely
  denser than SF's. Two sources:
  1. **SF's `double_inc_update`** — cross-ply capture/recapture toggle
     cancellation that Coda lacks. **Bit-identical, recoverable.** (Coda's
     single-call same-index cancellation is only 3.80% — see §4 — but
     cross-*ply* cancellation is a different, likely larger mechanism, unmeasured.)
  2. Coda's threat feature *enumeration* encodes more relationship types
     (`push_threats_for_piece`: direct, own-xray, slider-sees, slider-xray,
     non-sliders) vs SF (direct + discovered). **Eval-architecture** — trimming
     needs retrain + SPRT (like the X-ray test), not bit-identical.

**Architecture vs training (Adam's Q):** deltas/move is set by the feature-set
*code* (which threats are enumerated), NOT training — training sets weights,
not which features toggle on a move. So the density is **fundamental
architecture**, inherited from the Reckless-style enumeration Coda is based on.
To confirm Coda≈Reckless and both>SF, instrument Reckless's apply (TODO).

## 4. Bit-identical micro-opts — ALL measured NEUTRAL (the dead ends)

Four scalar micro-opts, each bit-identical, each NPS-neutral — because each
targets a tiny ~2-3% slice, not the volume/eval cost:

| Opt | Branch | Result |
|---|---|---|
| X-ray zero-emit cull | `xray-zeroemit-cull` (23b7e05) | single +0.3%, contended +0.06% |
| 48KB→6KB attack_index table | `threat-index-compute` (6f1e329) | +0.08% contended |
| per-node cached `gives_check` | `check-info-cache` (9f81d84) | single ~0%, contended −0.17% |
| (Fix A: threat-row reorder) | #1993 | H0 −4.9 (regressed) |

- **Same-index add/sub cancellation = only 3.80%** of streamed rows (28.2M
  streamed, 1.07M cancellable). Not the lever; ~0.3% NPS if implemented.
- Lesson: scalar shaving doesn't move Coda's NPS — per-node cost is
  NNUE-eval + threat-volume dominated. The levers are **delta volume** and
  **eval cost**, not generation/index micro-opts.

## 5. Static-eval cache ablation — REFUTED (cache is +13.5% NPS)

Hypothesis (Adam): the TT static-eval cache (reuse `tt_entry.static_eval`,
skip the NNUE eval on a TT hit — `search.rs:3199`) helps by skipping evals but
*hinders* incremental updates (finny tables, threat apply) via deeper
lazy-replay gaps. Tested via `NO_TT_STATIC_EVAL=1` (flag 84906b8, default on).

| | cache ON | cache OFF | Δ |
|---|---|---|---|
| node count | 4,015,423 | 4,015,423 | **identical (bit-identical)** |
| apply calls | 3.62M | 4.89M | +35% materializations |
| deltas/call | 10.05 | 9.45 | −6% (thinner applies, as hypothesized) |
| total deltas streamed | 36.4M | 46.2M | **+27% MORE work** |
| **NPS** | **957,972** | **828,435** | **cache = +13.5% NPS** |

**Verdict: keep the cache.** The hypothesis is directionally right — the cache
*does* fatten individual applies (10.05 vs 9.45) and deepen lazy gaps — but
skipping 35% of *whole* evals (forward pass + both accumulators) dominates the
fatter-apply cost. Ablating is −13.5% NPS for zero behavior change.

## 6. ALL bit-identical threat-apply levers are small — the real conclusion

Every bit-identical route to making the threat apply SF-cheap has now been
measured and bounded small:

| Lever | Measured | Bound |
|---|---|---|
| Identical apply kernel | i8→i16 widen+add, both engines | no gap |
| Same-index cancellation | 3.80% of streamed rows | ~0.3% NPS |
| **`double_inc_update` (cross-ply)** | replay gap: **82.6% gap==1, only 17.4% gap>=2** (avg 1.27 plies) | ≤17.4%-of-materializations × cancelling-fraction = tiny |
| Re-streaming overhead | 11.0 streamed/move vs 8.43 generated = 1.30× | lazy-replay-inherent; SF has it too |

**So the threat-accumulator speed gap is NOT recoverable implementation waste.**
The Coda 7.31 vs SF 4.46 core deltas/call difference is **threat-model density**
— Coda enumerates more feature *types* (direct + own-xray + slider-sees +
slider-xray + non-sliders) than SF (direct + discovered). This is
eval-architecture (Reckless-derived), fixed by the feature-set *code*, not
training.

**Reframe of the "120 Elo recoverable" target:** most of Coda's ~31%-vs-5.5%
threat cost is **buying eval richness, not waste.** X-ray (~27% of the volume)
is worth **+187 Elo** for ~11% NPS — proven. The threat cost is largely a
deliberate, valuable eval choice. The free (bit-identical) speed recovery in the
threat accumulator is therefore *small*.

### The productive path — per-feature-type value/cost A/Bs
The X-ray test is the methodology: train a net with a threat feature *type*
removed, SPRT for the eval cost, compare to the type's NPS cost. X-ray passed
decisively (+187). **The other types are untested** — `slider-sees` (step 2,
~0.83 d/piece-push), `slider-xray` (2b), `own-xray` (1b). If any type costs
speed without buying comparable Elo (unlike X-ray), trimming it is a pure win
(faster + neutral Elo). This is the real lever on the threat accumulator —
NOT micro-opts.

### Speed levers OUTSIDE the threat accumulator (the non-threat ~part of the gap)
- **SF contention retention** (36% vs 27% per-process under 16×) — bandwidth/MLP;
  partly the FT-prefetch thread (#1994). Bigger under contention (deployment).
- FT/L1/L2 forward, search, movegen — roughly comparable to SF per the profile,
  but the non-threat slice is where remaining bit-identical wins would live.

## 7. Cross-ply cancellation (`double_inc_update`) — IMPLEMENTED, then DROPPED (net-negative on fleet)

**Outcome (the headline):** implemented as `recapture-combine`
(`experiment/recapture-combine`, branch parked) and SPRT'd as **#2015**. Despite
measuring **+0.5% single / +2.1% 16×-contended on Hercules**, it SPRT'd
**net-negative (~−2.9 ±2.5, converging H0 at ~21k games)**. DROPPED.
- **Why (the lesson): Hercules is the fleet's memory-bound outlier, so its NPS
  bench OVER-states bandwidth-saving opts.** The combine trades compute
  (sort/cancel) + cache (skipping intermediate materializations → siblings
  replay further back) for memory bandwidth. Memory-bound Hercules → win;
  non-memory-bound fleet majority → the compute+caching costs dominate →
  net-negative. See `memory/feedback_hercules_bench_overstates_bandwidth_opts.md`.
- The cross-ply lever (~8.8% of streamed rows) is **real but not cheaply
  capturable** — cancellation requires combining/skipping the middle ply, which
  costs sibling cache hits (structural tension), and it'd only pay on
  memory-bound hosts (deployment Zen5 is not one). My `move_to`-equality gate
  also skipped MORE intermediates than SF's narrow `threateningSqs` condition.
- **Process note:** I prematurely called it "net-negative/dropped" at N=4420
  (LLR −0.85, unresolved); Adam resumed it for more games — the early −4.7 was
  pessimistic noise; it settled to ~−2.9. Don't conclude from an unresolved SPRT.

The original measurement/design (still useful background) follows.

Precise measurement (gap>=2 replay spans, net-zero add/sub over the combined span):
- **28.96% cancellable** over gap>=2 spans (vs 3.80% per-ply).
- gap>=2 spans are **30.3% of all streamed rows**.
- → `double_inc_update` would cancel **~8.8% of ALL streamed threat rows**
  (~0.8% NPS single-thread, ~2% contended — bandwidth regime, the big gap).
  **Bit-identical, no Elo risk.**

**SF's design (the right scope):** SF's `double_inc_update`
(`nnue_accumulator.cpp:224/540`) is NOT general gap>=2 — it targets
**capture+recapture** specifically (trigger: `dp2.remove_sq` is a square the
middle move threatened). It combines middle+target diffs via `FusedUpdateData`,
cancels the captured piece's redundant toggles, applies once from the computed
ancestor, and skips the (transient) middle materialization. This sidesteps the
intermediate-caching tradeoff (the skipped middle is a transient capture seq).
- **Coda port:** in `ThreatStack::update` replay, detect ply N+1 capturing a
  piece on a square ply N's move affected; combine N,N+1 deltas; cancel the
  captured piece's toggles; apply once; skip middle. Bit-identical (verify bench
  node count + accumulator consistency vs refresh). **This is the priority lever.**

## 8. slider-sees ablation — INVALID experiment (do NOT train)

Attempted as a "per-feature-type" A/B. **It is not a feature type.** Bullet
enumerates threats attacker-centrically (each piece emits all its direct
attacks). The engine's step-2 "slider-sees" emits `(slider S → moved piece P)`
when **P** moves — the SAME feature S emits via its own step 1 `(S → P)` when
**S** moves. So slider-sees is the incremental-update path that maintains
incoming-slider-attack features *from the victim side*.
- Disabling step 2 doesn't weaken eval cleanly — it **corrupts** it (`(S→P)`
  maintained only on S-moves, stale on P-moves → move-order-dependent garbage).
- **No matching net can be trained** (Bullet has no slider-sees subset; it's a
  complete direct-attack feature). So the `CODA_NO_SLIDER_SEES` +3% NPS is the
  cost of *necessary* machinery, not a removable feature.
- The valid "are slider-attack threats worth it" experiment removes slider
  direct attacks ENTIRELY (step 1 + step 2, engine + Bullet) — big, likely very
  negative. Not a quick test. **S200 train NOT launched.**

## 9. L1=32 VNNI kernel gap (compute, not bandwidth — still LIVE)

A *compute* speed lever (not subject to §7's Hercules-mirage caveat). The fused
int8 dot is `VPDPBUSD` (1 instr vs AVX2's 2). Dispatch (`select_l1_kernel`,
nnue.rs:2452-2456):
- **Prod L1=16:** column-major `DenseAvx512Vnni` — optimal, input loaded once
  per chunk. On the Zen5 deploy box (Ryzen 9700X) we already run this. No gap.
- **L1=32:** falls back to **row-major** `RowMajorAvx512Vnni` (per-neuron, re-
  scans input per neuron) — we never wrote a column-major VPDPBUSD kernel for 32
  neurons. This is a real part of the ~10% L1=32 NPS tax — a *missing kernel*,
  not intrinsic.
- **Handoff:** `docs/l1_32_vnni_kernel_handoff_2026-06-15.md` — a Zeus (Zen5)
  Claude is writing the column-major L1=32 VPDPBUSD kernel. Measure on Zen5
  (invisible on AVX2-only Hercules). Directly informs the L1=32 go/no-go (tune
  #2017) by cutting the speed half of the tax.

## Current state & where the thread stands (2026-06-15)

**Reframe (Adam): eval is near-ceiling, so SPEED is the high-leverage lever** —
Coda's eval is #2 vs LC0 (Spearman 0.853, SF 0.861, Reckless 0.836); limited eval
headroom. So a free 5-10% speed is valuable in itself; the per-feature "is it
worth its cost" framing is secondary.

**What's settled:**
- Threat accumulator is the dominant SF gap (Coda ~31% cycles vs SF ~5.5%); most
  of it is **intrinsic eval richness we've validated as worth it** (X-ray = +187
  Elo, #2014). Not recoverable implementation.
- Bit-identical micro-opts (cull, table-shrink, gives_check): neutral (§4).
- Static-eval cache: keep (ablating = −13.5% NPS, §5).
- Cross-ply/recapture-combine: net-negative on fleet, DROPPED (§7).
- slider-sees: invalid experiment, not a feature type (§8).

**Live speed levers (ranked):**
1. **L1=32 VNNI kernel** (§9) — compute, deployment-relevant, Zeus working it.
   Only matters if L1=32 proves worth it (tune #2017).
2. **FT-prefetch / contention-retention (#1994)** — bandwidth; **re-validate on
   the fleet, NOT Hercules** (the recapture-combine lesson: Hercules over-states
   bandwidth opts). May be a mirage too.
3. Reckless apply instrumentation — diagnostic only (confirm Coda≈Reckless density).

**Bottom line:** the bit-identical *speed* recovery in the threat accumulator is
thin (recapture-combine was the best shot and didn't survive the fleet). The
remaining real speed is the L1=32 VNNI kernel (compute) and a fleet-validated
re-look at FT-prefetch. Beyond that, the SF NPS gap is largely our deliberate,
+187-Elo eval richness — not waste, and we should not trade it away.

## Methodology notes
- Delta counts are deterministic (no clean machine needed); NPS needs the OB
  worker stopped (Hercules is the fleet's memory-bound outlier).
- SF instrumentation: temporary counter in `nnue_accumulator.cpp` `apply()`,
  reverted + rebuilt clean after (binary is pristine for RR).
- `deltas/call` is caching/laziness-confounded; `deltas/move` (at absorb) is
  the architecture-pure metric. Both behind `--features profile-threats`.
