# Closing the LTC-scaling gap: flex-formula + per-TC SPSA divergence

**Status: ACTIVE thread (started 2026-06-21).** This is the durable plan +
state. Value-level Viri-vs-Coda comparison lives in
`viridithas_scaling_2026-06-20.md`; this doc is the *methodology* and the
running cluster-by-cluster state.

## The problem (measured)

Coda is **stronger at STC but scales worse than both Viridithas and Obsidian**
as TC grows. Two of Adam's local scaling gauntlets (same engine entered at
1×/2×/4× TC, playing each other; Elo relative to the 6-entry pool):

**vs Viridithas (2026-06-21):**
| TC | Coda | Viri | Coda−Viri |
|----|------|------|-----------|
| 1× (10+0.1) | −71 | −115 | **+44 Coda** |
| 2× (20+0.2) | +6  | +31  | −25 Viri |
| 4× (40+0.4) | +72 | +81  | −9 (~tied) |

**vs Obsidian (2026-06-21):**
| TC | Coda | Obsidian | Obs−Coda |
|----|------|----------|----------|
| 1× | −113 | −69  | +44 |
| 2× | −6   | +47  | +53 |
| 4× | +34  | +111 | **+77** |

**Robust reads:**
- Coda is behind at **2×** vs *both* references → 20+0.2 is the cheap, robust
  measurement TC.
- Obsidian (a *stronger* engine, valid reference) shows the gap **widening
  monotonically** with TC (+44→+53→+77). The Viri "re-convergence at 4×" is an
  artifact of Viri being weaker than Coda at STC — do NOT conclude "4× is
  fine."
- **Symptom is draws, not losses.** Coda-4× draws 74.7% vs Obsidian-4× 67.3%;
  Coda's draw rate *rises* with TC. Coda isn't tactically beaten at long TC —
  it **draws winnable games**. That's a conversion / deep-resource-pruning
  signature, not a tactical-oversight one.

## The diagnosis

Coda's tunables are SPSA-tuned at STC, where the tuner inflates forward-pruning
aggression (pruned nodes → depth, which wins at STC). The same params, at the
depths LTC reaches, prune the long-horizon quiet resources needed to convert.
The fix splits three ways (see `viridithas_scaling_2026-06-20.md` for the
value-level evidence):

- **Bucket A — values + gate thresholds** (reachable by an LTC retune; Coda
  already exposes most gates as tunables: RFP_DEPTH, FUT_LMR_DEPTH, RAZOR_DEPTH,
  SE_DEPTH, NMP_BASE_R, futility magnitudes, CORR_W_NP).
- **Bucket B — formula shapes** (NOT reachable by any value; need code):
  SEE-quiet `−SEE_QUIET_MULT·lmr_d²` is hardcoded quadratic (Viri linear);
  aspiration `delta += delta/2` hardcoded ×1.5 (Viri adaptive); LMR integer ±1
  vs Viri fractional 1024-unit; history offset sign.
- **Bucket C — missing structure** (feature absent; need code): triple
  extensions; killer/refutation LMR reduce-less; corrhist major/minor + cont-14;
  optimism / material-scaled eval / eval-policy / adaptive probcut.

**Why pure LTC SPSA "didn't move much" (Adam's experience):** prior LTC tunes
were at 40+0.4 = the 4× point, where (vs Viri) the gap looked smallest; and
flat-value tuning can't reach Bucket B/C at all. Also Coda already over-prunes
RFP at great depth, which can **drown out** other changes' signal at LTC (watch
for this when a cluster's LTC tune looks flat — it may be RFP-masking, not
TC-stability).

## The methodology (Adam's plan, per-cluster loop)

For each related cluster (RFP, NMP, LMP, LMR, SEE, …):

1. **Make the formula flex like the best engines** — add the missing degrees of
   freedom (e.g. expose a hardcoded coefficient/exponent) so SPSA *can* reach
   the better shape. Only worth doing where Coda is genuinely rigid vs the
   references.
2. **Make the new knobs tunable**, **seeded at the cross-engine consensus**
   value (NOT Coda's current value).
3. **Focused SPSA at LTC and STC** from the *same* consensus seed, **~1000
   iters** (small param count → enough), Hash=256, vs the net in `net.txt`.
   Optionally a **VLTC** (e.g. 120+1.2) third anchor.
4. **Compare convergence across TCs.** The divergence IS the diagnostic:
   - Params that land **far apart** STC vs LTC → parameterise on `root_depth`
     (the two/three tunes hand you the anchors directly).
   - Params that **converge** → flat value is fine.
   - A param that **drifts back to Coda's old value** from the consensus seed →
     STOP and investigate *why* Coda's context differs from the reference
     (move ordering, a missing guard, an implementation bug) — the
     SPSA-reveals-a-bug pattern. Don't just accept the value.
5. **Repeat** on the next cluster.

This is steps (1)+(2) of the broader goal unified: **(1)** get the LTC-correct
*shape* first (this loop), **(2)** then auto-adjust by TC via `root_depth`
parameterisation — using the *measured* STC/LTC anchors, not guessed ones.

### Why root_depth (step 2 mechanism)
Coda already root-depth-relaxes RFP, LMR, ProbCut:
`margin += depth * (root_depth - ROOT_THRESH).max(0) * ROOT_COEF / 100` —
zero at STC, grows as the iteration depth (a clean, deterministic TC proxy:
~15-20 STC, 28-32 LTC, ~50 lichess bare-metal) increases. `root_depth` is
constant within an iteration, so adjusted params can be computed once per ID
iteration and cached — zero per-node cost, bench-stable, OB-safe. The earlier
root-depth attempts "never quite worked" most likely because they interpolated
toward a flat-retune anchor that wasn't a genuinely better shape — i.e. step 1
wasn't done first. Garbage anchor → garbage slope.

## Measurement frame
- **Fleet, not the local gauntlet.** The gauntlet is a *diagnostic* (±25
  Elo/arm — it found the gap) but can't resolve a 1-2 Elo tweak. OB SPRT/SPSA
  runs to significance.
- **Deficit TC = 20+0.2** for SPRTs (robust deficit in both references,
  cheaper → more games → resolution). **Validate winners at 40+0.4 too** (where
  the Obsidian gap is worst) so a 2×-only win doesn't hide a 4× regression.
- For formula/structure SPRTs that are too small to resolve solo, **bundle**
  several shape changes for a combined signal, then decompose (SF/Coda bundle
  pattern). Cross-check non-regression at STC (we're +44 ahead there — don't
  give it back).
- **Self-play SPRT captures these** (search-quality-at-depth, not a
  ponder-asymmetric TM effect — different from the TM inverted-methodology).

## Important calibration note (corrected 2026-06-21)
Do NOT claim these ports "amplify hugely at LTC." Coda's own alpha-raise port
was **+1.09 STC / +1.32 LTC (≈flat)** — a small TC-stable win, visible at STC.
(The +5.38 figure was Viridithas's measurement on *their* engine.) So the
LTC-shape changes have a *mechanistic* LTC argument but **unvalidated**
amplification for Coda; the alpha-raise data point is mildly cautionary that
ports may just be TC-stable. Validate in the gauntlet/fleet, don't assume.

## Cluster queue + current state

| cluster | rigidity to flex | priority | status |
|---|---|---|---|
| **LMP** | d² coef hardcoded =1; improving ÷2 hardcoded | shakedown (clean, override-confirmed TC-sensitive) | **IN FLIGHT** |
| **RFP** | over-prunes at depth AND masks other clusters' signal | **strong next** (double reason) | queued |
| **LMR** | integer ±1 vs fractional; missing per-condition muls | highest leverage, heaviest build | queued |
| **NMP / SEE** | NMP already flexible; SEE-quiet d² hardcoded (Bucket B) | later | queued |

### LMP cluster (first, 2026-06-21) — IN FLIGHT
- **Branch `experiment/lmp-flex`** (commit `8777f92`, Bench 2306785).
- **Flex applied** (src/search.rs): formula
  `(LMP_BASE + d²)/(2-improving)` → `(LMP_BASE*10 + LMP_QUAD_10X*d²)/((2-improving)*10)`.
  New tunable `LMP_QUAD_10X` (default 10 = coef 1.0) exposes the d² coefficient.
- **Consensus seed** (per the existing src comment: SF/Obsidian/Reckless use
  LMP_BASE=3): `LMP_BASE 6→3`, `LMP_QUAD_10X=10`, `LMP_DEPTH=8`. (Viri is
  `2.5 + 0.444·d²`, ≈ the same effective shape.)
- **Tunes**: **#2173 STC (10+0.1)** and **#2174 LTC (40+0.4)**, 3 params, 1000
  iters, Hash=256, dev-network 549C20A5. STC resolves first (LTC games 4×
  longer).
- **What to read**: (a) STC-vs-LTC convergence — expect `LMP_DEPTH` (gate) to
  diverge most (override had it 8→5) → root_depth candidate; (b) does
  `LMP_BASE` drift back from consensus 3 toward Coda's 6 → investigate why
  Coda's LMP context differs from SF's; (c) if the LTC tune is flat, suspect
  RFP-masking, not LMP TC-stability.

## See also
- `viridithas_scaling_2026-06-20.md` — value-level Viri-vs-Coda param tables,
  the 4-agent sweep, ranked portable candidates.
- The lichess deployment already hand-overrides LMP_BASE/LMP_DEPTH/SEE_QUIET_MULT/
  SEE_CAP_DEPTH/FUT_THREATS_MARGIN/HINDSIGHT/NMP_VERIFY_DEPTH/PROBCUT_MIN_DEPTH/
  NMP_BASE_R toward LTC values (SPRT'd +12 at 180+2). Those overrides are the
  *symptom* this thread aims to eliminate (auto-adjust instead of magic config),
  and a useful prior for which params are TC-sensitive.
