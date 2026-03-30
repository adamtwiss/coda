# Coda TODO

Shared work queue across all Claude instances. Prioritised by expected impact.
Mark items DONE with date when completed. Move to experiments.md when tested.

## Search Improvements

### High Priority (likely +5-20 Elo each)

- [ ] **Retry: LMP depth 9** — +4 Elo at 1500g self-play, needs x-engine retest with current SE/search
- [ ] **Retry: FH Blend Depth Gate 4** — +1.9 self-play, SE changes fail-high patterns
- [ ] **Retry: History Divisor 4000** — +1-3 persistent, history profile changed by SE
- [ ] **Retry: Futility 50+d*50** — +1.9 self-play, tighter futility with stronger eval
- [x] **2026-03-30** 4-ply continuation history (+14 Elo gauntlet) — Viridithas/Obsidian/Reckless all use plies 1,2,4,6. We use 1-ply. (Source: engine reviews)
- [ ] **Correction history expansion (4→6)** — add minor piece and major piece correction tables. All top engines have 6. (Source: Viridithas/Reckless/Obsidian reviews)
- [ ] **Pin-aware SEE** — exclude pinned pieces from attacker set. Simple correctness fix. (Source: Reckless review)
- [ ] **GHI mitigation (50mr hash)** — XOR halfmove clock into TT key to prevent draw-related TT corruption. Cheap. (Source: Obsidian/Reckless reviews)
- [ ] **NMP cutNode restriction** — Obsidian restricts NMP to cut nodes only. Our NMP ablation showed only -18 Elo, suggesting it fires too broadly. (Source: Obsidian review, ablation data)

### Medium Priority (likely +3-10 Elo each)

- [ ] **Retry: NMP Divisor 170** — +2.7 at 3963g, eval scale changed with new nets
- [ ] **Retry: TT Near-Miss Margin 96** — +1.6, TT more accurate with better search
- [ ] **Retry: NMP Deep Reduction d>=14** — +0.6 but +4-9 early, deeper search from SE
- [ ] **Retry: Mate Distance Pruning** — +0.7, universal technique
- [ ] **Retry: LMP 4+d^2** — +2.3, LMP showed 0 in ablation so formula needs work
- [ ] **Retry: Threat-Aware SEE Quiet** — +0.8 but +12.6 early, sound idea
- [ ] **Complexity-aware LMR** — Obsidian uses correction magnitude to reduce LMR less in complex positions. (Source: Obsidian review)
- [ ] **Eval-dependent aspiration delta** — Reckless uses `13 + avg^2/23660`. Adapts window to position complexity. (Source: Reckless review)
- [ ] **TT cutoff retroactive history** — penalise opponent's last move in cont history on TT cutoff. (Source: Obsidian review)

### Lower Priority / Exploratory

- [ ] **Retry: LMP Failing /2** — +0.8, LMP needs recalibration generally
- [ ] **Retry: Opponent Material LMR <4** — +1.9, one-directional signal
- [ ] **Retry: History Pruning Depth 4** — -0.5 but +5.5 at 1141g
- [ ] **No killers / counter-move** — Reckless uses history alone. Radical but proven. Test removing killers and see if history compensates. (Source: Reckless review)
- [ ] **Alpha-raise LMR tracking** — more reduction after alpha has been raised. (Source: Reckless review)
- [ ] **Soft stop thread consensus** — 65% of SMP threads must agree on best move before stopping. (Source: Reckless review)

## NNUE / Model

### High Priority

- [ ] **Frozen-FT v7 training** — use convert-checkpoint to init v7 from best v5, freeze FT, train hidden layers. Bullet fork (adamtwiss/bullet) has freeze support. (Ready to test on GPU hosts)
- [ ] **Mix blunder/material data into training** — Titan generating ~86M positions/day. Mix 1-2% into T80 data for v7 training. (In progress)
- [ ] **NNZ sparse L1 matmul** — all top engines use this. Our test at H=1024 was slower, but need weight repermutation for NNZ locality. (Source: Viridithas/Reckless/Obsidian reviews)
- [ ] **Pairwise FT for v7** — all top engines use pairwise (768→1536→pw→768) not direct 1024. Train a pairwise FT net and add hidden layers. (Source: architecture consensus)

### Medium Priority

- [ ] **WDL sweep for 768pw** — w3/w5/w7 were flat in self-play for 1024s. Test for 768pw which showed different behaviour.
- [ ] **Dual activation (ReLU + x^2)** — Obsidian feeds both to L2, doubling information. (Source: Obsidian review)
- [ ] **Weight repermutation** — reorder FT neurons for NNZ locality. Viridithas does co-occurrence analysis. (Source: Viridithas review)
- [ ] **Wider L1 (32 instead of 16)** — doubles hidden capacity, ~20% NPS penalty. Check-net values suggest 16 neurons may be too narrow.
- [ ] **Longer warmup for v7** — try 10-15 SB warmup instead of 5. v7-e1200 showed degradation.

### Lower Priority

- [ ] **Embedded default net** — `include_bytes!()` for single-binary distribution
- [ ] **Makefile wrapper** — easy build for non-Rust users
- [ ] **check-net in Coda** — port from GoChess tuner
- [ ] **Material scaling of NNUE output** — Alexandria does this. Cheap post-processing. (Source: Alexandria review)
- [ ] **Factoriser regularization** — Alexandria adds factoriser matrix to king bucket weights. (Source: Alexandria review)

## Infrastructure

- [ ] **SPSA parameter tuning** — Reckless/Viridithas both use this. Automated tuning of all search constants.
- [ ] **Retry batch runner** — script to run the 15+ retry candidates from experiments.md as parallel SPRTs
- [ ] **Cross-engine gauntlet automation** — standard 3-engine test (Minic/Texel/Weiss) with 600g baseline
- [ ] **Coda on CCRL** — submit for official rating once stable

## Done

- [x] **2026-03-29** Singular extensions: multi-cut + negative extensions (+27 Elo)
- [x] **2026-03-30** Singular extensions: positive extension (+22 Elo additional, +49 total)
- [x] **2026-03-30** Lazy SMP with UCI stop handling
- [x] **2026-03-30** v7 NNUE support with int8 SIMD L1 matmul (673K NPS)
- [x] **2026-03-30** 768pw pairwise NNUE support with SIMD (1464K NPS)
- [x] **2026-03-30** TT prefetch (+3% NPS)
- [x] **2026-03-30** Datagen: multi-threaded selfplay + material removal → SF binpack
- [x] **2026-03-30** Bullet fork (adamtwiss/bullet) with freeze/unfreeze support
- [x] **2026-03-30** convert-bullet: Bullet quantised.bin → .nnue
- [x] **2026-03-30** convert-checkpoint: .nnue → Bullet checkpoint for transfer learning
- [x] **2026-03-29** is_pseudo_legal fixes: pawn direction, castling attacks (+320 Elo)
- [x] **2026-03-30** Drop LVA from capture scoring, raw captHist (+14 Elo gauntlet)
- [x] **2026-03-30** NMP R=4, depth>=4 (neutral, consensus alignment)
- [x] **2026-03-30** Same-bucket king NNUE optimization (+7 Elo SPRT)
- [x] **2026-03-30** Feature ablation study (16 features, 300g each)
- [x] **2026-03-30** Double extension tested — rejected (too aggressive at depth*3 margin)
