# Peer-engine review synthesis — 2026-06-27 (ranks #8–14)

Cross-engine review of the **near-peer band below Coda** in the local RR
(Coda = #7): Integral (#8), Cinder (#9), Hobbes (#10), Viridithas (#11),
Clover (#12), Caissa (#13), Astra (#14). One agent per engine refreshed the
per-engine note against the *current* source and extracted candidate ideas,
each prior-art-checked against current Coda src + `experiments.md`.

**Framing caveat (load-bearing).** These engines are NOT stronger than Coda,
so their choices are *hypotheses*, not consensus authority (unlike the #1–6
reference set). An idea earns a slot here only if it is (a) genuinely absent
from current Coda, (b) not already H0'd, and (c) has a plausible mechanism.
**Convergence across ≥2 independent peers** is the strongest signal in this
band and is called out explicitly.

## Headline

The band is **largely exhausted** — most distinctive things these engines do,
Coda already does (often richer: 6-source corrhist, 4D threat history, 8 NNUE
buckets, multi-source TM) or has already H0'd. Every agent independently
reached this conclusion. The April notes were stale almost entirely on the
*Coda* side (Coda advanced past them), not the engine side. What survives is a
short, mostly-cheap queue plus one big NPS lever.

---

## Ranked testable-experiment queue

### Tier 1 — best signal (convergent and/or addresses a known Coda gap)

**T1. IIR on stale/shallow TT entries** — *convergent: Caissa + Hobbes.*
- Mechanism: Caissa fires IIR when `ttEntry.depth + 4 < depth` (`Search.cpp:1489`);
  Hobbes does `depth -= 1` at cut nodes with a *stale* TT move (`search.rs:380-385`).
- Coda today: IIR fires ONLY when `tt_move == NO_MOVE` (`search.rs:3747`) — a
  shallow/stale TT move suppresses IIR entirely.
- Prior art: NONE blocking. (#1214 se-excluded-guards touched IIR but is a
  different change; its IIR-only branch was bench-neutral.)
- Sketch: widen the IIR gate to also fire when the TT entry exists but
  `tt_depth + K < depth` (start K≈4, SF-style). One-line-ish in the IIR gate.
- Effort: **LOW**. Bounds `[0,3]`. Two independent near-peers do it → best
  first SPRT of the batch.

**T2. Quiet-history factoriser (shared baseline under the 4D buckets)** — *Hobbes.*
- Mechanism: Hobbes scores quiets as `factoriser + bucket[from_thr][to_thr]`
  (`history.rs:73-77,212-229`) — a shared, threat-bucket-*independent* component
  plus the bucketed delta.
- Coda today: `main_hist` is fully 4D threat-partitioned
  (`[from_thr][to_thr][from][to]`, `movepicker.rs:28`) with NO shared baseline,
  so each move's learning is fragmented across 4 threat buckets — a genuine
  Coda-specific structural weakness.
- Prior art: NONE. (The "factoriser" hits in `experiments.md` are all
  *NNUE-training* input factorisation — unrelated.)
- Sketch: add a `[from][to]` (or `[piece][to]`) factoriser table, update it
  alongside the 4D table, sum both at scoring time. Retune history-bonus shape.
- Effort: **MEDIUM** (hot path + retune). Highest-leverage *novel* mechanism in
  the batch.

**T3. Search-effort feedback into move ordering** — *convergent: Clover + Caissa.*
- Mechanism: Clover multiplies every history bonus/malus by `tried_count` (how
  many times the move was actually searched: 1 if LMR-only, 2–3 if re-searched)
  so confirmed moves dominate (`search.h:744-789`). Caissa adds a node-count
  ordering bonus `4096 * nodesSearched / nodesSum` from a persistent
  cross-iteration cache at ply<3 (`MoveOrderer.cpp:417-421`).
- Coda today: `history_bonus(depth)` is depth-only (`search.rs:4643,4909`);
  `root_move_nodes` is root-only / per-iteration / TM-only (`search.rs:825,2739`).
- Prior art: NONE for search-effort *ordering* (the `experiments.md`
  "search effort" hits are unrelated NMP/RFP notes).
- Sketch: two distinct variants — (a) Clover `tried_count` multiplier on
  history updates; (b) Caissa node-fraction ordering bonus at shallow ply. Test
  separately; (a) is the cleaner first cut.
- Effort: **MEDIUM**. Two independent peers feed search effort back into
  ordering → worth a careful try.

### Tier 2 — cheap singles, novel, lower expected magnitude

**T4. from/to single-square quiet histories** — *Hobbes.* Two `[2][64]` tables
added into quiet ordering (`history.rs:105-107,328-344`). Coda has no
single-square axis. No prior art. **LOW** effort — good cheap SPRT, natural to
bundle with T2.

**T5. Three-state `improving` (add a −1 "worsening" state)** — *Clover
`search.h:351-359`.* Coda's `improving` is binary; its LMP already uses the
`(2 − improving)` form (`search.rs:3989`) so a −1 state plugs straight in (and
into an LMR term). No prior art. **LOW** effort, smallest delta but cheapest.

**T6. RFP margin += parent-move history score** — *Integral `search.cc:731`*
(`margin += (stack-1)->history_score / kRevFutHistoryDiv`). Coda's RFP margin
has no parent-history term. No prior art (corr-complexity RFP is different and
H0'd). **LOW** effort — best folded into an RFP-margin SPSA cluster, not a
standalone SPRT.

**T7. Grandparent (ply-2) move-keyed correction-history axis** — *Hobbes
`correction.rs:59-61`.* A correction source keyed on `ss[ply-2].mv`. Distinct
from the just-H0'd #2317 paired-pairing (this is a separate single-keyed axis,
not a 2-D pairing). **LOW** effort but corrhist is heavily picked-over →
moderate confidence.

### Tier 3 — NPS / perf (separate track, not an Elo-mechanism)

**T8. Arity-specialized fused threat-accumulator apply** — *Cinder
`transformer.rs:136-143`* (branchless fused (sub,add) arms per dirty-piece
arity). Coda's threat-accumulator apply is the **named #1 NPS hotspot**
(~31% cycles vs SF ~5.5%). Audit `threat_accum.rs` / `nnue.rs` and specialize
common arities. No retrain. Bounds `[-2,1]` (non-regression / NPS). **MEDIUM**
effort, audit-first. Biggest single NPS lever found in the batch.

---

## Explicitly dropped (prior-art kills — do NOT re-propose)

- **Viridithas paired/2-move continuation correction (#424)** — overlaps the
  just-H0'd **#2317 paired-cont-corr** (−0.7). Dead.
- **Caissa threat-gated RFP improving discount** — threat-signal RFP/futility
  adjustments have H0'd ≥3× (`experiments.md` ~L1433: "threat signal only
  useful for LMR modulation"). Dead direction.
- **Cinder windowed multi-ply TT near-miss** — the 1-ply/80cp form is
  bracket-tested to death; multi-ply window is marginal upside at best.
- **Cinder graded/proportional singular extension** — hostile prior (SE
  positive-ext historically ~−30 Elo); would need retune-on-branch to even try.
- **Astra material-scaled eval** — subsumed by Coda's 8 NNUE output buckets on
  `(popcount−2)/4`. Redundant.
- **Integral cont-hist cross-ply-sum gravity base** — marginal; needs a
  gravity-divisor retune to even be neutral.
- **Clover eval-diff retroactive quiet bonus** — fail-low / PCM bonus family is
  closed in Coda (#1945/#1931/#1961/#866 H0).

## Recommended order of attack

1. **T1 (IIR stale-TT)** — cheapest, convergent, `[0,3]`. Fire first.
2. **T2 + T4 bundled** (factoriser + single-square histories) — the real
   structural bet on Coda's fragmented 4D history; retune-on-branch likely.
3. **T3a (Clover tried_count)** — convergent ordering signal.
4. **T5 / T6 / T7** — cheap fill-in SPRTs / fold T6 into the next RFP SPSA.
5. **T8** — schedule as a perf-audit task when an NPS push is wanted (separate
   from the Elo queue; no retrain, bankable if it speeds up).

Per-engine detail and full prior-art tables live in the refreshed
`engine-notes/<engine>.md` files (each carries a 2026-06-27 refresh section).
