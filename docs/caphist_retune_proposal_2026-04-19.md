# CapHist-Defender Rebase + Retune — Proposal

## Status as of 2026-04-21 — PRIORITY 1 execution item, not started

Branch `experiment/caphist-defender` still exists and is stale vs
trunk. Trunk has moved significantly since this doc was written —
confirmed-landed Elo since the Apr 19 diagnostic includes:

- **#502 B1 discovered-attack +52 Elo** (biggest single-feature win ever)
- **#553 SE-king-pressure +9.7**, **#542 unstable-ProbCut +6.7**,
  **#539 anythreat-NMP +6.0**, **#554 offense-bonus +5.7**,
  **#582 new v9 prod net +15.2**, **#583 LMR-endgame-gate +5.0**,
  **#578 tuned-knight-fork +5.2**, **#586 post-merge retune +6.2**,
  SMP trilogy bug fixes (+651 self-play at T=4)
- Plus structural changes: AVX-512/VNNI kernels, ponder-softfloor,
  EGTB hash.

**Impact on this proposal:** rebase WILL have merge conflicts in
movepicker.rs (knight-fork + offense bonuses merged there). But the
mechanism — 4D defensed-capture history lifting first-move-cut from
72% → 82% — is unchanged. Expected Elo gain is **still +8-15** per
the diagnostic; compensation pruning has only grown tighter with
post-merge retunes, giving more room to unwind.

**Next steps** (assigned to me / Hercules):
1. Checkout `experiment/caphist-defender`, rebase onto
   `feature/threat-inputs` @ 45ddaae. Resolve movepicker.rs conflicts.
2. Build + bench on reckless-crelu prod net (DAA4C54E).
3. Focused SPSA on 4 capture-coupled tunables (SEE_QUIET_MULT,
   SEE_CAP_MULT, BAD_NOISY_MARGIN, CAP_HIST_MULT) — 1000 iters.
4. Apply focused tune values, SPRT vs trunk at bounds [-3, 3].
5. If H1: merge, then full-tune the 64-param set for rebalancing.
6. If H0: inspect first-move-cut delta to confirm diagnostic still
   holds on current trunk (may have been absorbed by other merges).

**Recommendation** (original from Apr 19): rebase `experiment/caphist-defender` onto current trunk, then focused SPSA + full retune + SPRT. High-confidence ~+8-15 Elo candidate based on diagnostic evidence.

**Owner**: Hercules (implementation lane). This is a research-derived proposal with the specifics spelled out.

---

## 1. What we were investigating

Coda v9's **first-move-cut rate is 72.2%** vs v5's canonical **82.3%** — a 10pp move-ordering degradation between architectures. Same movepicker code runs both; only the eval differs. The question driving the investigation:

> What about v9's eval breaks the same ordering code that worked on v5, and what specifically fixes it?

Prior docs established that v9's non-linear eval produces a **flatter eval distribution per position** (median `gap_n` 1.06σ vs v5's 1.79σ) — the best move is less clearly separated from peers. Move ordering signals have less headroom.

What this doc adds: a **specific diagnostic of WHERE the 10pp gap lives**, and an already-built feature that may close it.

## 2. Key findings

### 2.1 The first-move-cut gap is concentrated in capture ordering at non-TT nodes

Instrumented `search.rs` to decompose first-move cutoffs by move type. Ran bench on v5 net and v9 net with the v9 trunk binary. Headline decomposition:

| Cutoff type | v5 net | v9 net | Δ |
|---|---|---|---|
| TT-move cut (first move is TT move, cuts) | 22.5% | 17.2% | −5.3 pp |
| **1st capture cut (no TT, MVV-best cap cuts)** | **41.7%** | **32.2%** | **−9.5 pp** |
| 1st quiet cut (no TT, no caps, 1st quiet cuts) | 15.8% | 23.0% | +7.2 pp |
| Cut at move 2-5 | 16.6% | 22.3% | +5.7 pp |
| Cut at move 6+ | 3.4% | 5.3% | +1.9 pp |

And crucially:

| Metric | v5 net | v9 net | Δ |
|---|---|---|---|
| **TT move available at cutoff** | 29.2% | 23.0% | −6.2 pp |
| **TT-cut rate WHEN available** | **77.1%** | **75.0%** | **−2.1 pp** (equivalent) |

**Implications**:

- **TT move reliability is NOT the problem.** When a TT move is available, it cuts at ~75-77% in both. Storing a TT move works equivalently well.
- **The biggest chunk of the gap (−9.5 pp) is in capture ordering.** Specifically: non-TT nodes where the first capture by MVV+captHist should cut but doesn't.
- **Secondary (+7.2 pp): more non-TT quiet-first-cut positions** — v9's tree is 62% larger and visits more quiet-only nodes.
- **Offense-bonus-style fixes (Reckless-port candidates) address the smaller +7.2 pp quiet shift, not the bigger −9.5 pp capture shift.**

### 2.2 SPSA has already partially compensated via tunable drift

Compared v5-tuned (main branch) vs v9-tuned (`feature/threat-inputs`) capture-related parameters. Same tunables, same binary, different SPSA-landed values:

| Tunable | v5 (main) | v9 (trunk) | Δ | Direction |
|---|---|---|---|---|
| SEE_QUIET_MULT | 18 | 45 | +150% | v9 prunes quiets much less aggressively |
| SEE_CAP_MULT | 120 | 146 | +22% | v9 prunes captures less aggressively |
| BAD_NOISY_MARGIN | 95 | 125 | +32% | v9 keeps more bad captures |
| CAP_HIST_MULT | 193 | 263 | +36% | v9 bigger captHist bonuses per depth |
| CAP_HIST_BASE | 36 | 15 | −58% | v9 less offset |

**SPSA has uniformly moved v9 parameters toward "prune less, lean harder on captHist."** This is the compensation tax for ordering weakness. It recovers some Elo but can't close the gap fully — we can see in bench that v9 still lags 10pp on first-move-cut and burns 62% more nodes.

This tells us something important: **the pruning parameters are coupled to ordering quality**. A better ordering signal would let SPSA tighten them back toward consensus values — capturing additional Elo via more efficient pruning on top of the direct ordering improvement.

### 2.3 `experiment/caphist-defender` already built the specific fix — and it works at the metric level

During investigation, found commit `8d0c1fc` on branch `experiment/caphist-defender`:

**What it does**: extends capture history from 3D `[piece][to][captured]` to 4D `[piece][to][captured][defended]`, where `defended = 1` iff the `to` square is attacked by any enemy piece.

**Why this specifically targets the diagnostic**: capturing a free hanging piece is categorically different from capturing a defended piece even when (piece, to, victim) are identical. Splitting them into separate captHist buckets lets the signal distinguish the two cases instead of averaging them.

**This is the same structure Reckless uses** (`noisy_history[piece][to][captured][to_threatened]`). Reckless has it landed. Coda built it but the SPRT failed without retune.

**Measured effects** (from commit message):

- **First-move-cut rose to 82.0%** — full v5 parity at the metric level
- **Bench dropped 22%** (2,642,436 → 2,061,895 nodes) — major tree efficiency gain
- EBF dropped 1.79 → 1.73 — search pays off per-depth with more per-node quality

**SPRT result** (#483): **H0 at −1.5 Elo after 14,520 games**. No retune was attempted.

## 3. Why SPRT-without-retune failed — and why we expect retune to fix it

The caphist-defender's bench dropped by 22%. That is a **massive tree-shape change**. The v9-trunk tunables (post-tune #490) are tuned against a 2.64M-node tree. On the new 2.06M-node tree:

- SEE_QUIET_MULT=45 is *too loose* (tree is already smaller; doesn't need extra leniency)
- CAP_HIST_MULT=263 is *too high* (captHist signal is now richer; doesn't need over-boosting)
- BAD_NOISY_MARGIN=125 is *too forgiving* (more bad captures surface where they shouldn't)

SPSA compensated for v9's bad ordering by loosening pruning and cranking captHist bonuses. With better ordering, that compensation becomes over-correction. The tuning landscape has shifted but the parameter values haven't. Net Elo is roughly flat or slightly negative.

This is the CLAUDE.md-documented **retune-on-branch pattern**:

> Some features are neutral without retuning but gain significant Elo when pruning parameters are recalibrated on their branch. Pattern: "big bench/node change but flat Elo → retune candidate"

Bench change: **−22%**. Elo change: **−1.5 (flat within noise)**. Classic signature.

## 4. Why this is worth investing in

- **The fix is already written.** No new algorithmic design; just execute rebase → retune → SPRT.
- **The ordering improvement is real and measured.** First-move-cut rising from 72% to 82% is the headline metric we've been chasing for weeks. This single change closes the full v5-v9 ordering gap.
- **Reckless has the same feature.** Reckless's capture ordering threat-indexed on `to_threatened` is architecturally the same. They have it landed because they tuned with it. We have it built; we've just skipped the tune step.
- **Expected Elo after retune: +8-15 Elo.** Math: v5's move-ordering advantage plus tunable unwinding. Direct ordering gain (v9→v5 first-move-cut parity) plausibly worth +5-8 Elo. Retune unwinding to less aggressive params plausibly adds another +3-7 Elo. Conservative estimate.
- **If it lands, it's a structural fix.** Unlike the per-signal sweep ideas (king-zone pressure etc.) which are pattern matches, this fix targets the mechanistic root cause of the v9 ordering gap as diagnosed. The first-move-cut 82.0% metric is evidence.

## 5. Staleness assessment

`caphist-defender` is dated 2026-04-18 but trunk has moved substantially since:

| Change landed since | Elo |
|---|---|
| B1 discovered-attack bonus (#502) — **movepicker changes** | +52.0 |
| Contempt=0 removal | +2.5 |
| Tune #490 post-merge retune | +7.4 |
| Main → trunk merge | — |

Total ~+62 Elo of search-side change on trunk since the caphist-defender SPRT. In particular:
- **Discovered-attack bonus modifies `movepicker.rs`** — merge conflict likely
- **Tune #490 re-SPSA'd** the 48 pruning params after those features — current tunable defaults differ from when caphist-defender was tested

**Answer: rebase is needed, not just retune.** The branch as-is has stale code and stale tunables relative to current trunk.

## 6. Recommendation — specific execution plan

### Step 1: Rebase `experiment/caphist-defender` onto current `feature/threat-inputs`

- Likely conflicts in `src/movepicker.rs` (discovered-attack-bonus touched quiet scoring) and possibly `src/history.rs` (captHist table dimension expansion).
- Preserve the caphist-defender semantics: add `defended` dim, compute `(enemy_attacks >> to) & 1` at scoring/update call sites, pass 0 in QS (per original commit's QS speed note).
- Bench the rebased code — expected: still ~2.0-2.1M nodes (huge drop from trunk's 3.56M), first-move-cut ~80-82%.
- If rebase bench numbers don't show the expected ordering improvement, investigate whether B1 absorbed part of the effect or there's a merge bug.

### Step 2: Focused SPSA on 4 capture-related tunables

Before investing a full 18-param tune, run a focused 1000-iter SPSA on the 4 parameters most directly coupled to capture behavior:

```
SEE_QUIET_MULT,    int, 45,  5,   80,  5, 0.002
SEE_CAP_MULT,      int, 146, 30,  200, 15, 0.002
BAD_NOISY_MARGIN,  int, 125, 30,  150, 10, 0.002
CAP_HIST_MULT,     int, 263, 50,  400, 30, 0.002
```

Submit with `--dev-network 6AEA210B --scale-nps 250000` per standard v9 SPSA flags.

**What to watch for**:
- `SEE_QUIET_MULT` drifting downward toward v5's 18 → compensation is unwinding, retune thesis confirmed
- `CAP_HIST_MULT` drifting downward toward v5's 193 → same direction, captHist no longer needs over-boosting
- All parameters near current defaults → retune thesis is wrong, feature genuinely doesn't help; abandon

Focused tune converges in ~4-6 hours. Fast confirmation of the thesis.

### Step 3: Full retune if focused tune points the right direction

If focused SPSA moves params toward v5-direction, proceed with full 18-param retune for 2500 iters. Standard `scripts/tune_pruning_18.txt` spec. This is the retune-on-branch pattern that landed +38.6 (tune #454) and +7.4 (tune #490). Expected: similar-scale gain since the structural change is comparable in impact.

### Step 4: SPRT tuned values vs current trunk

Apply focused + full tune outputs, submit SPRT `[0, 5]` bounds. Expected H1 at +8-15 Elo combined (direct ordering + tunable unwind).

### Step 5: If it lands, propagate

- Check: is QS capture ordering also using the defended bit via captHist lookup? The original commit passed `defended=0` in QS for speed. If H1, measure whether enabling the defended bit in QS (at cost of ~15 magic lookups per QS capture scoring) is net-positive. Likely a separate SPRT.
- Check: does Reckless's "factorizer + bucket" split (separate global piece-to score + threat-specific bucket) give additional Elo on top of the split dimension? Potential follow-up structural improvement.

## 7. What NOT to do

- **Don't SPRT caphist-defender as-is without retune.** That's what already happened with #483. Result will be H0 again.
- **Don't just apply the current trunk tunables to the rebased branch and SPRT.** Same problem — those tunables are calibrated for the old (worse-ordering) regime. They over-prune on the new (better-ordering) tree.
- **Don't conflate this with offense bonus / king-ring / other movepicker ideas.** Those address the +7.2 pp quiet-cut shift (smaller locus). This addresses the −9.5 pp capture-cut shift (bigger locus). Orthogonal, both worth testing, but prioritize by magnitude.

## 8. Risk assessment

**Scenario A** (most likely): focused SPSA pulls params toward v5-direction, full tune lands +8-15 Elo, SPRT H1. ~1-2 days of Hercules work. Very clean outcome.

**Scenario B**: focused SPSA doesn't move params much. This would mean the retune thesis is wrong — the −1.5 Elo from #483 is fundamentally because the feature doesn't help, not because the tune is stale. 4-6 hours sunk. Low cost.

**Scenario C**: focused SPSA moves params correctly, but full retune + SPRT only shows marginal gain (say +2-3 Elo). Still positive; worth merging; but less dramatic than hoped. Cost ~1 day, modest upside.

**Scenario D**: rebase has semantic issues, the feature breaks, noisy SPSA. Worth auditing carefully before the tune runs. Would burn ~a day before realizing.

Weighted expectation: ~+6-8 Elo. High enough to justify the work. The `experiment/caphist-defender` commit message's measurements (first-move-cut 82.0%, bench -22%) are the strongest evidence — if we can preserve those through rebase, the mechanical gain is already visible.

## 9. Companion docs

- `move_ordering_understanding_2026-04-19.md` — full investigation that produced the diagnostic
- `ordering_coupled_pruning_2026-04-19.md` — analysis of how pruning params compensate for ordering weakness
- `tunable_anomalies_2026-04-19.md` — cross-engine tunable comparison that flagged the drift pattern
- This doc — actionable proposal for caphist-defender retune

## Summary

**What**: rebase `experiment/caphist-defender` + focused SPSA + full retune + SPRT.

**Why**: the feature demonstrably closes the first-move-cut gap to v5 parity (82.0% vs v9 trunk's 72.2%) and drops bench 22%. It failed SPRT only because the pruning params were over-tuned for the worse-ordering regime and now over-prune on the better tree.

**Expected Elo**: +8-15 (direct ordering + tunable unwind), per the retune-on-branch pattern that has produced consistent gains in this project.

**Time budget**: 1-2 days of Hercules work. Focused SPSA (~4-6 hours) gives fast Go/No-Go before committing to full tune.

**Closes the deepest diagnostic finding from the move-ordering investigation.** Not a pattern-match; a root-cause fix.
