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

### TODO
- Instrument Reckless's apply to confirm Coda inherited its density (Coda≈Reckless≫SF).
- Per-type ablation A/Bs (slider-sees first — largest non-xray type).

## Methodology notes
- Delta counts are deterministic (no clean machine needed); NPS needs the OB
  worker stopped (Hercules is the fleet's memory-bound outlier).
- SF instrumentation: temporary counter in `nnue_accumulator.cpp` `apply()`,
  reverted + rebuilt clean after (binary is pristine for RR).
- `deltas/call` is caching/laziness-confounded; `deltas/move` (at absorb) is
  the architecture-pure metric. Both behind `--features profile-threats`.
