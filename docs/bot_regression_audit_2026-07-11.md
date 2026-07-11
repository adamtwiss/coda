# coda_bot bullet regression audit — deployment-regime window hunt (2026-07-11)

## Symptom

- **coda_bot** (Thor, bare metal, -t16×1 or -t8×2 — mix reverted without impact,
  ponder ON, bullet, Hash=2048, 5-man EGTB, MoveOverhead=100): peaked ~3085 on
  **June 21**, steady bleed to ~3005-3010 by July 10. Deploy lag 1-2 days →
  peak code ≈ main @ ~June 19-20.
- **codabot** (VPS, -t4, ponder ON, bullet, Hash=1024, 5-man EGTB,
  MoveOverhead=100→200): GAINED over the same window on the same code stream.
- **T1/STC/no-ponder cross-engine benchmark**: the same code gained **~+88**
  over the window (Coda.today +35 vs Coda.june −53 in one pool).
- **Constraint**: neither bot captured the +88 → the defect(s) scale
  monotonically with thread count and/or live in the ponder regime common to
  both. No hard T4-ok/T8-bad cliff required.
- **Throughput ruled out** (probe 2026-07-11): HEAD is FASTER than the Jun-19
  build at T=1/8/16, at 200ms and 2000ms movetimes; L1=32 net costs ~11-13%
  NPS uniformly across T (no bandwidth wall). It is a search-quality /
  time-usage regression, not speed.
- Thor is dedicated to the bot (no contention); host-level cause excluded.

## Method

Four parallel mechanism-level audit slices over the June 15-26 merge window
(+ TM/ponder changes through Jul 8, since the bleed spans them): TT-write
wave, TM/ponder, search-shape-under-SMP, perf/TB/config. All OB gating for
these merges was Threads=1 (occasionally T=4), no ponder, 10+0.1 — the
deployment regime (T8-16 × ponder × bullet) has never been SPRT'd.

## Ranked suspects

### Phase 1 — onset (deployed ~Jun 21-26; matches the peak/decline timing)

**S1. QS corrhist stand-pat → per-thread-corrected scores in the shared TT**
(`387a02d`, Jun 24, #2220 **+6.6 at T=1** — biggest win of the window).
The corrected stand-pat becomes the SCORE stored in shared-TT QS entries
(store sites search.rs ~6483/6623). Pre-merge, leaf scores derived from raw
net eval — thread-invariant. Post-merge every QS score embeds the writing
thread's private corrhist state; other threads consume them as cutoffs.
Foreign-state fraction = (T−1)/T: 0% @T1, 75% @T4, 94% @T16 — the observed
gradient. Ponder ~doubles per-position drift. References run per-thread
corrhist at high T fine — Coda-specific compounders: raw-training feedback
(live until Jul 8), all-node update volume (S2), T=1-calibrated CORR weights.

**S2. Corrhist all-node updates** (`dcb2dba`, Jun 20, +1.8 T=1 — deployed AT
the peak). ~Doubles corrhist training volume incl. loose fail-low bounds
(loosened further by `3e3ff0a` skip-remaining-quiets and later `399eb55`
LMP-first-order). Accelerates within-move divergence of 16 per-thread
tables → amplifies S1. Same era as the raw-training fixed-point bug
(corrhist audit 2026-07-08 #1): corrected leaf scores trained corrhist
toward themselves. Fires at T=1 too; threads/ponder scale it.

**S3. TT EXACT-override depth inversion** (`df3f54e`/a9a2019, Jun 20,
+5.6 T=1). Any EXACT store bypasses the same-key depth gate (tt.rs:477-495).
T=1: iterative deepening makes same-key EXACT stores depth-monotone — safe,
SF-equivalent. T>1: helpers restart full ID loops from depth 1 EVERY `go`
(search.rs ~2666-2720), so a lagging helper's depth-7 EXACT **downgrades**
main's depth-18 PV-backbone entry. Steady-state entry depth = "most recent
thread", not "best". Hash-INDEPENDENT (same-key, no table pressure needed —
survives the 2GB/1GB configs). Requires ≥2 threads by construction →
structurally invisible to every SPRT ever run. Downgraded entries also
become eviction bait (victim score `depth − 8·age`).

**S4. IIR-after-NMP + 58-param retune** (`856c89e`, Jun 25). Raw change was
−4.3; only the T=1/LTC retune made it +2.46. At T8-16, helper TT-prefill
makes `tt_move == NO_MOVE` rarer → IIR fires less → tree drifts toward the
−4.3 shape while carrying T=1-calibrated constants (RFP/LMR/LMP cluster).

**S5 (demoted). QS stand-pat fail-high store** (`0e18453`, Jun 24): +30-35%
TT write volume; every miss-store takes the unconditional replace-worst
path. First-order only at small Hash (64MB cycles ~1.5s at T16-bullet);
at the bots' 2048/1024MB the table cycles in ~46s/T16 — a per-game not
per-move effect. Kept as a mix-shift contributor, not a driver.

### Phase 2 — deepeners (deployed ~Jul 3-9; bleed continued/steepened)

**S6. Factor-6 `thread_bmc` ponder-window accumulation** (merged **Jul 5**,
`7b58a92`/9168676, OB #2576 T=4 Hash=256 10+0.1 NO ponder, +3.2). Defect:
slots are reset only at search start and read only when `soft_limit > 0` —
during `go ponder` soft_limit==0, so **cross-thread best-move-change churn
accumulates unread across the entire ponder**; the first post-ponderhit
read consumes the whole window: multiplier 1 + 2.315·(total/n) → typical
**3.3-8× on exactly the post-hit budget decision**, every non-instant hit.
SF decays its equivalent every iteration INCLUDING while pondering; Coda
does not. Normalized by n → hits T4 and T16 near-equally (matches both bots
underperforming), with a monotone-T kicker (shared-TT nondeterminism raises
flip rates; each main flip is also double-counted with the stability
table). Bounded ~2× by the P2 mid-iteration band on hit-moves; on MISS
moves (~36%) there is no soft band — at 120+1 hard is 27-40s and the 13×
factor ceiling is deliberately open at bullet inc-cover ratios.

**S7. FL-EXT soft-band suspension at bullet/high-T** (Jul 5-6, incl. v2/v3).
Post-hit root fail-low at depth≥10 suspends the soft band and inflates
×1.34-1.68 toward hard (27-40s at 2+1) — one re-think can eat 20-40% of a
bullet clock. Root fail-lows get MORE frequent at high T (helper TT
pollution destabilizes aspiration). Calibrated only at 10+0.1/T=1 vs SF's
tail stats.

**S8. Worker history survives ucinewgame** (Stage-2 pool diversity, Jul 4-5;
persistent-state audit C7): 15 stale workers at -t16 vs 3 at -t4. Post-onset
stacker.

**S9. C2 `tm_cross_prev_score` pollution** (Jul 8, **FIXED Jul 10**
`26f9829`): ponder-miss/analysis/TB-score pollution of the cross-move TM
trend, factor pinned at 1.55 → +55% opt post-miss. Measured +7±16 to fix
(968-game ponder gauntlet). Both bots, thread-independent. Already on main.

**S10. Hugepage threat-matrix allocation** (Jul 3) — only if Thor is
multi-NUMA (`numactl -H` to check): 65MB weights pinned to one node.

### Exonerated (mechanism-level, be-honest pass)

TB trio Jun 26 (no locks anywhere in probe path; interior probes REDUCED;
symmetric config both bots) · June TM commits incl. tm-inc-hard-cap (no-ops
at bullet: inc==0 path untouched, 120+1 ceiling 40s > any sane spend; the
double-merge was a revert+re-land, landed once) · finny-inline · weight
alignment · avx512-pack · threat dual-POV · ep-interpose · NMP verify
barrier · SEE pin filter · mobility-delta drop · evasion picker · probcut
TT seed (read-only, re-validated) · probcut decisive guard (volume ~0) ·
Thor concurrency mix (t16×1 vs t8×2 reverted by Adam without impact).

### Deployment-hygiene flags

- `TM_SUBTREE_BASE_100=130` was live on main part of Jul 10
  (`ddf603c`→reverted `9e049ec` same day) and measured **−48 vs SF while
  even in self-play** — verify neither bot shipped a build from that window.
- Bots track head within 1-2 days; both usually deployed together.

## Hermetic test ladder (cheap → expensive; lichess is NOT a measurement instrument)

1. **No-game probes (hours):**
   a. EXACT-downgrade counter (tt.rs: count same-key EXACT overwrites with
      `depth < slot_depth − 4`) on fixed positions at T=1/4/16 — expect ≈0
      at T1, ~linear in T if S3 is live.
   b. TMDebug factor decomposition at bullet+ponder (log already prints the
      inputs): distribution of the Factor-6 multiplier on ponderhit vs miss
      moves at T=4/16 — S6 shows up as a 3-8× spike cluster on hit moves.
2. **`NO_CORRECTION=1` self-play, T=8 bullet, ponder** (flag exists, zero
   code): correction is strongly positive at T=1 — if the flagged side
   equalizes/wins at T=8, the corrhist cluster (S1+S2) is confirmed.
3. **Factor-6 kill A/B without rebuild**: `setoption TM_BMC_INSTAB_MULT
   value 500` (UCI min; 4.6× attenuation) at bullet+ponder+T8; proper fix
   is a 3-line slot-reset when post-hit TM arms.
4. **One-line separators**: revert S1 (`stand_pat = scaled_stand_pat`) vs
   revert S2's upper branch vs env-flag S3's `flag_is_exact` — three
   binaries, T=8 bullet ponder RRs.
5. **Window bisect probes** if the above don't localize: Jun19 / Jun26 /
   Jul2 / Jul8 / HEAD at -t8 ponder bullet vs fixed pool.

## Meta-lesson

Same class as the C2/corrhist findings, one level up: **our entire
validation stack measures T=1/no-ponder/STC-LTC, and the deployment regime
(T8-16 × ponder × bullet) had four independent mechanisms land in three
weeks, none of which CAN fire in any test we ran.** The +88/T1 vs
−40/deployed divergence is the bill. Standing fix candidates: (a) a
periodic hermetic deployment-regime RR (T=8, ponder, bullet, fixed pool) as
a release gate for TM/SMP/TT-replacement changes; (b) SPRT-at-T=4 minimum
for any change whose mechanism touches shared state (TT replacement,
corrhist-into-TT, cross-thread TM inputs).
