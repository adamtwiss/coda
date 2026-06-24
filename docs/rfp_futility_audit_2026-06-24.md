# RFP and Futility Pruning Audit — 2026-06-24

Three parallel agents: (1) RFP structural vs top 6, (2) futility vs top 6,
(3) quantitative bench measurement. HEAD at 654c354.

---

## QUANTITATIVE BASELINE

RFP fires at **301.1/Kn** — the single dominant pruning mechanism.
Compare: SEE 141.8/Kn, TT cutoffs 80/Kn, NMP 7.5/Kn, futility 13.2/Kn.
RFP alone accounts for more cutoffs than all other pruners combined.

RFP_AUDIT false-positive rates (null-verified — FP means NMP would disagree):
```
d=1: 472,556 audited | 42.4% FP rate  ← 69% of all volume
d=2:  92,582 audited | 45.4% FP rate
d=3:  56,047 audited | 42.2% FP rate
d=4:  33,141 audited | 40.0% FP rate
d=5:  17,417 audited | 38.2% FP rate
d=6:   8,028 audited | 38.1% FP rate
d=7:   3,606 audited | 36.7% FP rate
d=8:   1,123 audited | 29.4% FP rate  ← deep-knee activates here
d=9:     288 audited | 24.3% FP rate
d=10:     67 audited | 11.9% FP rate
```
**Key insight:** The "too strong at depth" hypothesis is REVERSED. Deep RFP
(d=8+) fires rarely AND has the lowest FP rates. Shallow RFP (d=1-3) has the
highest FP rates at 40-45% and 98% of the volume. The deep-knee quadratic
(+7.6 Elo H1) is doing its job correctly at depth.

---

## RFP: STRUCTURAL FINDINGS

### RFP-1. Shallow margins 35-55% below all 6 peers (HIGHEST PRIORITY)

At depths 4-8 — the dominant search depths at both STC and LTC — Coda's RFP
margins are far below every reference engine:

| Depth | Coda(STC) | Coda(LTC) | SF | Obsidian | Berserk | Reckless | Alexandria | Plenty |
|-------|-----------|-----------|-----|----------|---------|---------|-----------|--------|
| d=4   | 148       | 157       | 304 | 348      | 280     | 244     | 300       | 248    |
| d=6   | 222       | 236       | 456 | 522      | 420     | 472     | 450       | 450    |
| d=8   | 368       | 387       | 608 | 696      | 560     | 770     | 600       | 709    |

Coda's base multiplier is 37 (non-improving). Peers: SF 76, Obsidian 87,
Berserk 70, Alexandria 75, Reckless ~80+.

**Effect:** Coda fires RFP at d=6 when static_eval >= beta + 222. Peers require
beta + 420-522. Coda prunes more aggressively in the shallow zone where 42-45%
of cutoffs disagree with NMP verification.

**Fix direction:** Raise RFP_MARGIN_NOIMP from 37 toward 70-87 (peer range).
This LOOSENS shallow RFP (fewer wrong shallow cuts) while the deep-knee
quadratic handles the deep end independently. Needs SPSA retune after raise.

---

### RFP-2. Improving delta almost flat — structural divergence (MEDIUM)

Coda: improving margin = `depth * 33`, non-improving = `depth * 37`.
Delta = `depth * 4` → only 16cp at d=4, 32cp at d=8.

Peers use a **depth-flat constant** for the improving reduction:
- SF: -199cp flat (regardless of depth)
- Berserk: -118cp flat
- Obsidian: -87cp flat
- Alexandria: -61cp flat
- Reckless: -77cp flat

Result: Coda barely distinguishes improving from non-improving at shallow depths.
Peers make improving positions 2-6× harder to prune.

**Fix:** Change from `depth*(NOIMP-IMP)` to `depth*NOIMP_MULT - FLAT_IMP_CONST`.
Add `RFP_FLAT_IMP` tunable (default ~80, range 0-200). This decouples the
improving adjustment from the depth scaling. SPSA + SPRT [0,3].

---

### RFP-3. Missing TB/mate guard on static_eval (CORRECTNESS)

Every peer guards `static_eval.abs() < TB_WIN_BOUND` or equivalent before
applying RFP. Coda has no such guard. If NNUE sees near-mate (e.g. forced
mate in 3, eval ~29000), RFP could still fire and cut the node before the
mate continuation is searched.

The beta-side guard exists (`beta.abs() < MATE_SCORE - 100` in RFP_AUDIT
skip) but NOT the eval-side guard.

**Fix (one-liner):** Add `&& static_eval.abs() < MATE_SCORE - 200` to the
RFP gate. Bounds: `[-2, 1]`. Every peer has this.

---

### RFP-4. Depth cap at 17 — uniquely high (LOW)

Peer caps: Berserk 9, Obsidian/Alexandria 11, SF 14, PlentyChess 15.
Coda fires at depth 17; only Reckless has no cap (relies on quadratic).
At d=12+ Coda's margins exceed most peers' caps. Not a problem per the
FP-rate data (d=8+: 12-29% FP, low volume), but worth noting.

---

### RFP-5. Return value: fail-hard vs damped (LOW)

Coda: `(static_eval - margin + beta) / 2` (B3 blend fix applied ✓).
Peers: SF `(2*beta + eval)/3`, others `(eval+beta)/2` or similar.
All are now blended — no issue here.

---

## FUTILITY: FINDINGS

### FUT-1. `!bestMove` bonus missing (UNTESTED, medium priority)

SF adds +151 when `bestMove == NO_MOVE` (nothing has raised alpha yet at this
node). PlentyChess adds +113. Intuition: if the node hasn't found a good move
yet, futility candidates are less likely to save it → prune more aggressively.

**Fix:** Add `FUT_NO_BEST_MOVE` tunable (default 0). When `best_move == NO_MOVE`,
`futility_value += FUT_NO_BEST_MOVE`. SPRT [0,3].

### FUT-2. `eval >= alpha` bonus missing (UNTESTED, low priority)

SF adds +86 when `static_eval >= alpha`. Reckless adds +88 when `eval >= beta`.
Coda has neither. The `threats_adj` term partially captures this but differently.

### FUT-3. Futility itself is well-calibrated

At lmr_d=8: Coda margin 867cp, peers 449-1599cp. Coda sits mid-pack. The
FUT_BASE=75, FUT_PER_DEPTH=99 values are SPSA-tuned. History scope (main only)
was empirically validated — composite history gave H0 or worse in multiple tests.
The FUT_THREATS_MARGIN term is Coda-unique and SPSA-validated.

**No correctness bugs in futility.** The `!bestMove` and `eval>=alpha` bonuses
are the only untested structural additions.

---

## PRIORITY ORDER

| # | Finding | Type | Action | Bounds |
|---|---------|------|--------|--------|
| 1 | **RFP-1: raise NOIMP margins toward peers** | Structural | Raise RFP_MARGIN_NOIMP 37→75, SPSA retune, SPRT | [0,3] |
| 2 | **RFP-2: improving delta → flat constant** | Structural | Add RFP_FLAT_IMP tunable, SPSA, SPRT | [0,3] |
| 3 | **RFP-3: TB/mate guard on static_eval** | Correctness | One-liner + [-2,1] | [-2,1] |
| 4 | **FUT-1: !bestMove futility bonus** | Structural | Add FUT_NO_BEST_MOVE, SPRT | [0,3] |
| 5 | FUT-2: eval>=alpha futility bonus | Structural | Bundle with FUT-1 | [0,3] |
| 6 | RFP-4: depth cap | Low | Leave—FP data validates current behavior | — |

**Do NOT touch the deep-knee quadratic or depth 8+ behavior** — the FP-rate
data and the +7.6 Elo H1 both validate it. The problem is the shallow end.

The core insight: Coda's RFP base multiplier (~37-40) is roughly half the
peer consensus (~70-87). This makes shallow RFP aggressive (low threshold
to cut) while the deep-knee quadratic correctly restrains deep RFP. Raising
the base multiplier loosens shallow RFP (fewer wrong cuts in the d=1-3 zone
that has 42-45% FP rates) without touching deep behavior.
