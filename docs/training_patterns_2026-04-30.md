# NNUE Training Patterns — Cross-Engine Synthesis (2026-04-30)

Companion to `selfplay_data_strategy_2026-04-30.md` and `hobbes_selfplay_case_study_2026-04-30.md`. Where those docs focus on **data origin**, this doc is the broader **training-time hyperparameter** synthesis: WDL schedules, LR schedules, two-stage training, fen-skipping, DFRC blending, activation chains, opening filters, architecture co-evolution.

Sources: Hobbes (`~/chess/engines/Hobbes/network_history.txt`, 482 lines, 41 iterations), Viridithas (`~/chess/engines/viridithas/notes/networkhistory.txt`, the gold-standard public log spanning many years), Bullet trainer source (`~/code/bullet/crates/bullet_lib/`), and Coda's own training experience.

## Convention check — `wdl` / `blend` / `lambda`

Critical to nail down before reading anything else.

**Bullet (which Coda, Hobbes, Viridithas all use):** at `bullet_lib/src/value/loader.rs:244`,
```rust
result_target = blend * result + (1.0 - blend) * score
```
where `blend ≡ wdl ∈ [0, 1]`. **`wdl = 0` = pure eval target; `wdl = 1` = pure outcome target.**

So when Hobbes writes `wdl: constant(1)` for h-1, that's **pure outcome** (W/D/L only, no eval signal). When Viridithas writes `viri19: same as viri18, but with 90% WDL focus`, that's `wdl = 0.9` = **90% outcome / 10% eval**. Same convention.

**Other communities:** Lc0 uses `lambda` where `lambda = 0` is pure outcome and `lambda = 1` is pure eval — **opposite** of Bullet's `wdl`. SF/fishtest historically inconsistent. Always verify before importing a number from outside the Bullet ecosystem.

For the rest of this doc: **`wdl` = Bullet convention** = outcome weight.

## WDL — fixed blend

### What Coda has measured directly

For linear (fixed) WDL on Coda's own training:
- **V5 architecture: optimum `wdl = 0.07`**
- **V9 architecture: optimum `wdl = 0.15`**

(Pattern: as architecture richness increases, optimum WDL increases. Eval-quality improves → eval target less noisy → higher WDL signal can take more weight without polluting eval signal.)

### What other engines converge on (also fixed-blend)

| Engine | Final-stage WDL | Architecture stage | Source |
|---|---|---|---|
| Coda V5 | 0.07 | direct-FT 768pw | own experiments |
| Coda V9 | 0.15 | 768pw + threats + h16x32 | own experiments |
| Hobbes h-1 | 1.0 | 768→32 (random init) | network_history |
| Hobbes h-7 | 0.4 | 768→256 | h-7 |
| Hobbes h-11 → h-32 | schedule (see below) | progressively wider | many |
| Hobbes h-33+ | schedule, ending at 0.6 → 0.75 | (768x16hm→1280pw)x2→hidden | h-33 to h-41 |
| Viridithas — fixed-blend regime | 0.25 to 0.40 | various | viri19 lost 75 Elo at 0.9; viri47 lost 29 Elo at 0.2; sweet spot 0.25-0.40 |

**Cross-engine pattern**: optimum WDL roughly correlates with architecture maturity. Pure-RL engines (Hobbes, Viri at random-init phase) start near 1.0 and decay; deep-eval-T80 engines (Coda) sit lower. Direction of optimum: richer eval → higher WDL.

### Coda recommendation — fixed blend

`wdl = 0.15` is current V9 best. **Worth retesting if architecture changes** (new threat dim, different hidden chain, factor architecture revert). Direction of optimum movement should track architecture richness.

## WDL — schedule (alternative to fixed)

### The Hobbes ramp

Hobbes adopts a multi-segment schedule from h-11 onward:

```
WDL: 100sb constant(0.2), 700sb linear(0.2 → 0.4)        // h-11 to h-32
WDL: 100sb constant(0.2), 700sb linear(0.2 → 0.4),       // h-33 onward
     200sb constant(0.6)                                  // (s2 finetune)
WDL: 100sb constant(0.2), 700sb linear(0.2 → 0.6),       // h-38+
     200sb constant(0.75)                                  // (s2 finetune)
```

**Pattern: WDL weight *increases* as training progresses.** Hypothesis (Hobbes-style):
- Early: eval is undertrained, eval-loss signal is noisy; WDL ground-truth is more informative.
- Late: eval has converged on the average position; high WDL focuses residual capacity on game-outcome-relevant patterns (long-term strategic, conversion technique) that the per-position eval-loss can't capture cleanly.

Hobbes consistently reported gains generation-to-generation through h-11 → h-41 with this schedule.

### Viridithas's contrary evidence

Viridithas tested an explicit late-stage WDL ramp via the `basilisk.4` through `basilisk.8` finetunes — 200sb on top of `retrochron` at increasing fixed WDL:

| Net | Finetune WDL | fixed-nodes Elo | LTC Elo |
|---|---|---|---|
| basilisk.4 | 0.4 | +0.7 | **−8.8** |
| basilisk.5 | 0.5 | −0.3 | −7.4 |
| basilisk.6 | 0.6 | −4.85 | −5.5 |
| basilisk.7 | 0.7 | −13.95 | −5.8 |
| basilisk.8 | 0.8 | **−19.27** | **−10.7** |

All regressed. Earlier `everedge` (semiotic + WDL warmup to 0.5 over 160sb): −5.78 fixed-nodes.

### Reading

Engine-dependent. Hobbes's positive results spanning many iterations are real. Viridithas's negative results spanning many iterations are also real. **Mechanism difference is unknown** — could be architecture, could be data quality, could be that Hobbes's gradual ramp from low (0.2) gives a smoother schedule than Viri's step-jumps to 0.4-0.8.

### Coda recommendation — schedule

Worth one focused experiment: Coda's `wdl = 0.15` final → ramp `0.05 → 0.15` over the cosine-LR phase, then `0.20-0.30` constant for a low-LR tail. Tests Hobbes's hypothesis on Coda-specific architecture without committing to Viri's failed 0.6+ region. Test as a Bullet-config change once the variance issue lifts. Expected: **0 to +5 Elo**, but not a guaranteed win given the cross-engine split.

## LR schedule

### Cosine decay endpoints

Bullet cosine decay: `lr_initial → lr_final` over N superbatches. The endpoint value matters far more than the initial value.

Cross-engine endpoints, **ordered by magnitude** (Coda V9 is the lowest by ~10×):

| Engine / config | Final LR | × Coda V9 (2.43e-7) |
|---|---:|---:|
| **Coda V9 (current)** | **2.43e-7** | 1.0 |
| Hobbes stage-2 constant (h-29+) | 8.1e-7 | **3.3×** |
| Hobbes h-1 to h-7 cosine endpoint | 2.7e-6 | 11× |
| Bullet examples (jw1912's reference) | 2.4e-6 | 10× |
| Coda V5 | 5e-6 | 21× |
| **Hobbes h-8+ cosine endpoint** | **8.1e-6** | 33× |

**Per CLAUDE.md, Coda V5→V9 was a 20× drop in final LR (5e-6 → 2.43e-7), and the SB400→SB800 +88 Elo win is the load-bearing evidence** that V9's sparse threat features keep converging deep into the tail. So 2.43e-7 isn't arbitrary — it's measured to deliver real gain over higher endpoints.

But the optimum has not been bracketed. Coda has only the comparison `5e-6 → 2.43e-7` (positive). Intermediate points (1e-6, 5e-7, 1e-7) and below (5e-8) haven't been tested. Hobbes's measurement at h-8 (where they raised their endpoint *up* by 3× and gained +9.6 Elo) is at a much smaller architecture (768→256, no hidden chain) so doesn't directly transfer, but is a reminder that the optimum can sit higher than a "lower is always better" heuristic implies.

### Warmup duration

| Engine / config | Linear warmup duration |
|---|---|
| Bullet examples | ~5 SB |
| CLAUDE.md historical note | 5-10 SB |
| **Coda V9 measured optimum** | **30 SB** |
| Hobbes | not documented; likely Bullet default ~5 |

Coda's warm30 finding (warm40, warm50 H0'd; warm5 lost ~7 Elo per `project_v9_warmup`) is **3-6× longer than the cross-engine consensus**. Combined with the very-low endpoint, Coda V9 has unusually high sensitivity to both ends of the LR schedule — likely tied to the threat-feature head needing both a long ramp-up to converge cleanly AND a very low tail to refine on rare-feature gradients.

**Coda's V9 belief: low-LR tail is load-bearing** — `project_v9_low_lr_tail_critical.md`. SB400 → SB800 delivered +88 Elo on V9 (vs flat on V7). The threat features keep converging deep into the tail.

### The Hobbes 2.7e-6 → 8.1e-6 finding

This contradicts the Bullet-reference "2.4e-6" guidance and Coda's belief that "lower is better". Hobbes's measured +9.6 Elo from raising the endpoint LR from 2.7e-6 to 8.1e-6 is a clean datapoint. **Possible explanations:**
- Hobbes was at a smaller architecture during h-7/h-8 (768→256). Smaller nets need higher LR to converge in fixed superbatches.
- Hobbes's 5B-position dataset at h-7 was small enough that lower LR underutilised the data; at h-12+ they grew to billions of positions and the LR question may have re-balanced.
- Coda's "lower is better" was measured at 1024s+threats architecture on T80 data — may not generalise to all configs.

### Coda recommendation — LR

Current V9 belief (lower-than-V5's-5e-6) is worth retesting after factor revert. Specifically:
- Reproduce Hobbes's `cosine 0.001 → 8.1e-6` schedule on Coda V9.
- Compare against Coda's current schedule.
- If Hobbes-shape lands positive: also try Hobbes's 2-stage variant (8.1e-6 cosine end + 0.81e-6 constant tail).

This is a Bullet-config change; same compute as a normal training run. Worth one test under stable variance.

## Two-stage training

### The Hobbes pattern (since h-29)

```
Stage 1 ("d1"):                  Stage 2 ("d2"):
  data:   broad cumulative         data:   recent slice only
          (e.g. h6 → h27)                  (e.g. h23 → h27)
  sb:     800                      sb:     200
  lr:     cosine(0.001, 8.1e-6)    lr:     constant(0.81e-6)
  wdl:    100sb(0.2),              wdl:    constant(0.6 to 0.75)
          700sb linear(0.2 → 0.4)
```

**Conceptually: pretrain on broad data → fine-tune on fresh data at low LR.** Three things change at the stage boundary simultaneously: data slice (broader → recent), LR (cosine → constant low), WDL (ramping → high constant).

### Why this is more than just "low-LR tail"

Coda's existing belief is "long low-LR tail". Hobbes's two-stage adds two extra dimensions:
- **Different data at the tail** — they switch to recent-only games. Hypothesis: late training should refine on the freshest signal, not re-average over old games.
- **Higher WDL at the tail** — see WDL schedule above. Implies the tail is for outcome-pattern learning, not eval-pattern refinement.

### Coda recommendation — two-stage

Coda doesn't currently run two-stage training in Bullet. Adopting the Hobbes pattern is a **moderate Bullet-config change** (multi-segment trainer schedule, requires re-shuffling of input dataset between stages). Test on V9 after variance lifts. Expected: +2 to +6 Elo, low standalone risk, compounds with the Hobbes WDL-ramp + low-LR-tail patterns we already partially adopt.

## fen-skipping

### What it is

A training-time data subsampling switch in Bullet: each FEN has probability `p` of being skipped per epoch. Hobbes uses `0.5` standard, `0.75` (skip more aggressive) on h-35.

```
viriformat (fen skipping: 0.5)        // h-37+ standard
viriformat (fen skipping: 0.75)       // h-35 aggressive
viriformat (fen skipping: 0.01)       // h-17 trivial baseline
```

### Why it works

With ~23B positions and ~1000 superbatches × ~17M positions/SB, each FEN already only gets ~1 epoch on average. Random subsampling per epoch:
- **Halves per-epoch compute** (the obvious one).
- **Weak regularisation** — each position sees less reliable training signal across epochs, harder to overfit specific FENs.
- Reported as a **non-regression** in Hobbes (h-37 vs h-36) — same Elo, half the compute.

### When it doesn't apply

If your dataset is small enough that 100% per-epoch is already only 0.5-1× the training, dropping further hurts coverage. Coda's T80 corpus (10s of GB) at SB800 is closer to the "barely covered once" regime; aggressive fen-skipping would lose data more than gain regularisation.

### Coda recommendation — fen-skipping

**Defer until self-play data scale grows beyond 5B+ positions.** At current Coda T80 + threat-feature configurations, full coverage per epoch is the conservative choice. Revisit when data grows or when a Hobbes-style cumulative-data corpus accumulates.

## DFRC blend

### What it is

Double Fischer Random Chess (Chess960²) — both sides start with independently shuffled back ranks. 921,600 distinct opening positions. Forces the net to evaluate non-standard piece configurations.

Hobbes blends ~80/20 standard/DFRC since h-32. Viridithas, Stormphrax, PlentyChess all include DFRC in datagen. **Coda doesn't currently generate DFRC training data.**

### Why it works

- Forces the net to learn structural eval features rather than memorising standard-opening patterns ("rook on a1 because rooks usually start there" → "rook on a1 because of file/rank/safety reasons X, Y, Z").
- Diversity injection at scale — every starting position is unique.
- Hobbes h-32 (first DFRC retrain): **+159 Elo on DFRC test positions**. Standard-position regression was small (+6.21 Elo, basically a non-reg). So the blend is essentially free Elo on DFRC + neutral on standard.

### Coda recommendation — DFRC

If Coda commits to self-play (Recipe E or F from `selfplay_data_strategy_2026-04-30.md`), **DFRC blend at 20% is a near-free addition**. Implementation: extend `src/datagen.rs` to optionally produce DFRC-start games. Bullet handles DFRC FENs natively (its `viriformat` supports them).

If we stick with T80 only, DFRC isn't applicable (T80 is standard chess only).

## Opening eval-bound filter

Hobbes's `tools/datagen.rs:51-53`:
```rust
td.nnue.activate(&board);
if td.nnue.evaluate(&board).abs() > 1000 {
    return generate_random_opening(td, rng, random_moves, dfrc);
}
```

After random opening plies, **regenerate the opening if eval > 1000cp**. Throws away wildly imbalanced starts that would yield uninformative game data.

### Why it matters

Random plies frequently produce blunder positions where one side has already lost. Self-play games from those positions train the net on "mate-in-X conversion" patterns, not on "decision-making at near-equal evals". The latter is what makes a strong engine.

Coda's existing datagen at `src/datagen.rs:248-257` uses 4-9 random plies but **has no eval filter**. Adopting Hobbes's filter is one extra eval call per opening attempt — trivially cheap.

### Coda recommendation — eval-bound filter

**Adopt unconditionally** if/when Coda runs more self-play. Threshold: `1000cp` matches Hobbes; could be tightened to `800cp` for Coda's more decisive eval scale. Implementation: 5 lines in `src/datagen.rs`.

## Activation chain evolution (Hobbes h-39 → h-41)

| Iter | Activation chain | Note |
|---|---|---|
| h-1 to h-38 | `screlu` (FT only, no hidden) | scalar output, single SCReLU |
| h-39 | `screlu → crelu → crelu` | first hidden chain |
| h-40 | **`crelu+csrelu → crelu → crelu`** | dual activation at FT |
| h-41 | `crelu+csrelu → crelu → crelu` | (kept) |

### The dual-activation FT

`crelu+csrelu` means the FT output is concatenated CReLU and "custom-SReLU" (likely a scaled/saturated variant). Hobbes reports h-40 over h-39: **+9.5 Elo**. So the dual-activation pattern at FT (rather than SCReLU alone) is a measurable win.

### Coda's current activation

Coda V9 uses `SCReLU` at FT, then `CReLU` through hiddens (per CLAUDE.md hidden-layer activation findings). **Same as Hobbes h-39, not h-40.** The dual-activation FT (Hobbes h-40 pattern) isn't currently in Coda.

### Coda recommendation — activation

Worth a focused architecture experiment: **dual-activation FT** (`crelu + csrelu` concatenated, doubling FT output dim) on Coda V9. Bullet supports custom activation chains. Test budget: one full SB800 train + SPRT vs current V9.

This is a training-architecture experiment so blocked by the variance issue until factor reverts and we know our floor.

## Architecture co-evolution

### The Hobbes principle

Hobbes grew architecture in lockstep with data quality:
- Iter 1-7: small (768→32 to 768→256) on small data (33M to 5.5B)
- Iter 8-16: medium (768→256 to 1024 + 4 buckets) on growing data (1B to 1.7B)
- Iter 17-30: large + multiple buckets (8 → 10 buckets, 1280 FT)
- Iter 31-41: full architecture (16 buckets, pairwise, hidden chain) on 23B+ data

**Why this matters:** A small net trained on weak evals can't overfit the noise. A large net trained on weak evals overfits the noise in detail. **The Coda V5→V9 single-iteration experiment failed precisely this way** — V9 has the capacity to learn V5-eval-blindspots in detail.

### Cumulative data sourcing

Hobbes consistently writes "source: hobbes 6 → N-1" — using games from many recent generations, not just the latest net. The dataset accumulates rather than churning. Fights diversity collapse.

### Coda implication

Coda is at the architecture endpoint already (V9 = same shape as Hobbes h-41). If Coda goes self-play, we don't need the architecture-growth phase; we start at the endpoint. But the **cumulative-data principle** still applies: future self-play iterations should add to T80 + prior iterations, not replace.

This was already in the Hobbes case-study doc; restating here for completeness of the training-pattern picture.

## What Coda already has, what's missing

Compact view:

| Pattern | Coda status |
|---|---|
| WDL fixed blend (V5=0.07, V9=0.15) | ✅ measured |
| WDL ramp / schedule (Hobbes-style) | ❌ untested on Coda |
| Cosine LR endpoint (low) | ✅ believes-low; CLAUDE.md updated |
| Two-stage LR + data + WDL (Hobbes pattern) | ❌ not implemented |
| fen-skipping | N/A at current scale; defer |
| DFRC blend in datagen | ❌ not generated |
| Opening eval-bound filter | ❌ Coda has random-ply but no eval filter |
| Dual-activation FT | ❌ Coda is single-SCReLU FT (Hobbes h-39 shape) |
| Hidden activation = CReLU through chain | ✅ matches consensus |
| Cumulative-data principle for any future self-play | acknowledged in Hobbes doc |
| Quiet-position filter (ply ≥ 16, no checks/captures/tactical) | ✅ in place |
| Architecture co-evolution | N/A (Coda already at endpoint shape) |

Eight items in the "missing or untested" column. None are individually high-Elo (most are +1-6 each), but they cluster naturally — a single training-time-experiment branch could combine WDL ramp + 2-stage + opening-filter + dual-activation FT and SPRT once.

## Test plan (parked, blocked on factor variance)

When stable training A/B is restored, test in this order:

1. **WDL Hobbes-ramp on V9** (`0.05 → 0.15` cosine + `0.20-0.30` tail). Bullet-config change. Expected 0 to +5 Elo. Standalone test.
2. **Two-stage training on V9** (split last 200sb as constant-low-LR finetune). Bullet-config change. Expected +2 to +6 Elo. Standalone.
3. **LR endpoint at 8.1e-6** (Hobbes-style). Compare with current V9 endpoint. Expected ±2 Elo, mostly informational about whether Hobbes's "higher endpoint" finding generalises.
4. **Dual-activation FT** (Hobbes h-40 pattern). Architecture change. Expected +3 to +10. Standalone, requires re-train on retrained data.
5. **Opening eval-bound filter in datagen** (5-line addition). Required for any future self-play; expected neutral on T80-only training.

Most of (1)-(3) can be done together on a single training run as a "Hobbes-style schedule" SPRT. The test budget would be 1-2 SB800 runs + SPRTs.

## Cross-cutting observations

- **`wdl` convention discipline matters** — every cross-engine transfer of a number requires verifying the loss formula. Bullet ecosystem (Coda, Hobbes, Viri) is consistent at `wdl = outcome weight`; Lc0 and older communities flip it.
- **WDL ramp to higher in late training is engine-dependent.** Hobbes positive across 30+ generations; Viri negative across 5 finetune nets. Don't import the result blindly; test on Coda specifically.
- **Hobbes's measured +9.6 Elo from raising LR endpoint 2.7e-6 → 8.1e-6 contradicts CLAUDE.md's "lower is better" guidance.** Worth retesting on Coda V9.
- **Coda's V9 architecture endpoint matches Hobbes h-41.** Architecture isn't the bottleneck; the training-time hyperparameter schedule is the residual lever.
- **Eight training-pattern items in the "missing or untested" column.** Cumulative envelope ~+10-30 Elo if half land. None individually large; together meaningful.

## Sources

- `~/code/bullet/crates/bullet_lib/src/value/loader.rs:240-249` — Bullet WDL convention.
- `~/chess/engines/Hobbes/network_history.txt` — full schedule history per iteration.
- `~/chess/engines/Hobbes/src/tools/datagen.rs:51-53` — opening eval-bound filter.
- `~/chess/engines/viridithas/notes/networkhistory.txt` — `viri19`, `viri47`, `everedge`, `basilisk.4-.8`, `godsword`, `fixpoint`, `inflection` for WDL evidence.
- `~/code/coda/CLAUDE.md` — Coda's measured V5 0.07 / V9 0.15 WDL optima, low-LR-tail observation.

## Companion docs

- `selfplay_data_strategy_2026-04-30.md` — data origin (T80 vs self-play vs hybrid).
- `hobbes_selfplay_case_study_2026-04-30.md` — Hobbes's full bootstrap case.
- `factoriser_design_2026-04-21.md` — orthogonal architectural tweak.
- `cross_engine_comparison_2026-04-25.md` — search-side comparisons (no overlap with this doc's training scope).
- `experiments.md` — source of truth for resolved Coda training/SPRT results.
