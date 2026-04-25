# Reckless vs Coda — instrumented pruning diff (2026-04-25)

Patched Reckless with `dbg_hit` counters at every pruning fire site,
ran bench at depth 12 against Coda's bench at depth 12. Reckless uses
51 positions, Coda uses 8 — but per-position node counts are
comparable (Reckless 67K avg, Coda 63K avg) so per-Kn rates are
directly comparable.

## Headline

| Metric | Coda d12 | Reckless d12 | Coda/Reckless |
|---|---:|---:|---:|
| Total nodes | 502 467 | 3 425 249 | — |
| First-move-cut rate | 81.7% | **82.85%** | -1.15pp |
| **NMP cutoff rate** | **30%** | **57%** | **0.53× (Reckless 1.9×)** |
| NMP cutoffs / Kn | 6.3 | 10.3 | 0.61× |
| RFP cutoffs / Kn | 408.5 | 372.3 | 1.10× |
| **LMP fires / Kn** | **573.9** | **20.4** | **28× more in Coda** |
| **Futility (incl BNFP) / Kn** | 707.0 | 138.2 | **5.1× more in Coda** |
| **SEE prunes / Kn** | 573.8 | 189.5 | **3.0× more in Coda** |

## Findings

### 1. Coda fires shallow-pruning gates 3-28× more often than Reckless

Across LMP, FP, BNFP, SEE — Coda's pruning gates fire dramatically more
often per node. This is the OPPOSITE of CLAUDE.md's prior framing
("Coda under-prunes vs Reckless"). The threshold values may be more
permissive in some cases, but the **firing rates** show Coda relies
heavily on shallow pruning.

The most extreme delta is **LMP: 28× more firings per Kn in Coda**.

### 2. Reckless's tree is naturally smaller via better move ordering

- First-move-cut: Reckless 82.85% vs Coda 81.7%
- NMP cutoff rate: Reckless **57%** vs Coda **30%**

When the first move cuts beta more often, fewer moves ever reach the
pruning gates downstream. Reckless's pruning gates fire less *because
they don't need to* — the tree is already shaped efficiently by
ordering and NMP.

Coda's higher LMP/FP/SEE rates are a SYMPTOM of weaker ordering,
not a sign that we need to prune harder.

### 3. The NMP cutoff-rate gap (30% vs 57%) is the highest-leverage finding

Same formula structure (Coda: `r = NMP_BASE_R + depth/NMP_DEPTH_DIV
+ capture_bonus + eval_margin/NMP_EVAL_DIV`; Reckless: `r = (5335 +
260*depth + 493*(eval-beta).clamp(0,1003)/128) / 1024`) gives roughly
the same R values at moderate depths. So the formula isn't the
problem.

**Material differences in the NMP gate:**

| | Coda | Reckless |
|---|---|---|
| `cut_node` requirement | NO | **YES** — only at expected fail-high nodes |
| Gate eval condition | `eval >= beta` | `eval >= beta + (negative-margin formula)` |
| Material gate | `stm_non_pawn != 0` | `material > 600` |
| TT-noisy guard | In flight (#768) | YES |
| Verify | `depth >= NMP_VERIFY_DEPTH=8` | `depth >= 16` |
| `cutoff_count` modifier | NO | YES |

Most likely culprits for the 2× efficiency gap:
- **`cut_node` requirement.** Reckless ONLY attempts NMP at nodes the
  search expects to fail-high. Those nodes have higher prior
  probability of beta cutoff → NMP succeeds more often. Coda fires
  NMP at all non-PV nodes meeting the conditions.
- **`cutoff_count` shaping.** Reckless tracks how many moves at the
  CHILD node have cut beta; this propagates back as a cheap
  search-quality signal. Coda doesn't track this.
- **Higher verify threshold (16 vs 8).** Reckless skips verify
  more aggressively, which doesn't help cutoff RATE per se but does
  reduce wasted verify-fails.

### 4. NMP fire depth distribution

Reckless NMP attempts mean depth: 7.0, max 19.
Reckless NMP cutoffs mean depth: 7.2, max 19.

Coda doesn't currently break out per-depth. This patch's
infrastructure could be ported back to Coda for symmetric measurement
of the same depths.

## Implications for the queue

Re-prioritising based on this:

1. **NMP cutoff rate gap is the highest-leverage open lever** at
   ~2× efficiency delta. Possible attacks:
   - Add `cut_node`-equivalent gate (only NMP at expected fail-high
     nodes — Coda has cut_node info via the search node-type
     parameter).
   - Add `cutoff_count` propagation from child node (cheap signal,
     used by Reckless in NMP, LMR, and several pruning gates).
   - The TT-noisy guard (#768) is the right direction, just one part
     of the gap.

2. **The "force more pruning" direction was correct in spirit but
   addresses symptoms.** Tune-750 worked because it tightened
   over-aggressive thresholds, but the deep fix is improving
   ordering so fewer moves reach pruning gates.

3. **Move ordering remains the single highest-value lever**. FMC
   gap is only 1.15pp but compounds across the tree. Every quiet
   move that moves UP one slot in the order saves whatever subtree
   it was about to expand.

4. **De-prioritise LMP/FP/SEE threshold work** unless we have
   strong prior. Coda's already firing those 3-28× more often —
   we're already pruning hard there.

## Methodology

Reckless instrumentation patch: `dbg_hit` calls at 8 sites in
`src/search.rs` (RFP fire, NMP attempt, NMP cutoff, LMP fire, FP
fire, BNFP fire, SEE quiet fire, SEE noisy fire, beta-cutoff,
first-move-cut). `dbg_stats` for depth distribution. Output via
existing `dbg_print()` at end of bench. Patch lives on Reckless
working tree; not committed (throwaway diagnostic).

Reckless bench: `~/chess/engines/Reckless/target/release/reckless
bench` (51 positions, depth 12).
Coda bench: `./target/release/coda bench 12` (8 positions, depth 12).

Same node-count regime per position; ratios meaningful.

## Next concrete experiment

`experiment/nmp-cut-node-gate-only`:

Add to Coda's NMP gate (around src/search.rs:2172):
```rust
let nmp_gate_cheap = depth >= tp(&NMP_MIN_DEPTH) && !in_check && ply > 0
    && stm_non_pawn != 0 && beta - alpha == 1
    && static_eval >= beta && !prev_was_null
    && beta.abs() < MATE_SCORE - 100
    && info.excluded_move[ply_u] == NO_MOVE
    && cut_node;  // NEW: only attempt NMP at expected fail-high nodes
```

`cut_node` is already passed through search; it's just a plumbing
addition. Bench would change (NMP fires less often → tree shape
shifts), so SPRT at `[0, 3]`. Downside if H0: NMP becomes
strictly less powerful in Coda's specific node distribution.
Upside if H1: meaningful step toward closing the 30% → 57% gap.
