# Coda TODO

Shared work queue across all Claude instances. Prioritised by expected impact.
Mark items DONE with date when completed. Move to experiments.md when tested.

## Search Improvements

### High Priority — Untested or Conditions Changed

- [ ] **Store moved piece in search stack** — Currently ply-2 cont hist looks up piece via `board.piece_at(move_to(pm2))` which returns stale/wrong piece ~10-20% of the time (when ply-1 captured on that square). This makes ply-2 pruning catastrophic (-58 Elo) and degrades ply-2 LMR. Fix: add `moved_piece` field to per-ply search info, set before make_move. Enables correct ply-2+ cont hist for both LMR and pruning. Do when branch backlog is clear.
- [ ] **Node-based time management** — Concept proven (every top engine), but implementation failed twice (-31, -10). Needs: per-iteration node reset (done), correct parameters (Berserk 2.27/0.45 too aggressive for us), depth gate >= 6 (done). Try more conservative params or Alexandria-style (0.63 + notBest * 2.0).
- [ ] **Const attack tables + runtime PEXT reconciliation** — Const tables showed +1-2 Elo but conflicts with runtime AMD PEXT detection (+20% NPS). Need to make both coexist.

- [ ] **Retry: No killers / counter-move** — was -530 with old search, but we now have 4D threat-aware history which is the prerequisite SF/Reckless/Viridithas used when dropping killers. #1 retest priority.
- [ ] **SE margin tuning** — SE just added (2026-03-30) with GoChess params (depth>=8, margin=depth, verify=(depth-1)/2). Never tuned for Coda. Try margin=depth*2/3, depth gate 6 or 10, double-ext margin 20/25.
- [ ] **Weiss NMP skip after null move** — anti-chain NMP. From Batch C, never tested.
- [ ] **Reverse direction history in cont-hist** — extend Arasan reverse-hist idea to cont-hist tables. From Batch C, never tested.
- [ ] **Passed pawn push exemption from pruning** — From Batch C, never tested.

### Medium Priority — Retry Candidates (small positives, conditions changed)

- [ ] **Retry: NMP Deep Reduction d>=14** — +0.6 for 1657g, showed +4-9 early. Post-cuckoo NMP dynamics may differ.
- [ ] **Retry: LMP Failing /2** — +0.8 for 1693g, persistent +2-4 throughout.
- [ ] **Retry: History Pruning Depth 4** — -0.5 for 1321g, showed +5.5 at 1141g.
- [ ] **Retry: Threat-Aware SEE Quiet** — +0.8 for 1675g, showed +12.6 early. Post-4D history may change dynamics.
- [ ] **Retry: Opponent Material LMR <4** — +1.9 for 2710g, one-directional signal.
- [ ] **Soft stop thread consensus** — 65% of SMP threads must agree on best move before stopping. (Source: Reckless review)
- [ ] **Opponent easy captures for pruning gates** — gate RFP/NMP on whether opponent has easy captures. (Source: Berserk review)

### Lower Priority / Exploratory

- [ ] **CReLU vs SCReLU for v7** — Berserk uses CReLU specifically for better sparsity in sparse L1. SCReLU kills sparsity. Consider CReLU for pairwise+hidden arch. (Source: Berserk review)

### Done / Rejected (2026-03-30 and earlier)

- [x] **Cont-hist 4-ply reads AND writes at plies 1,2,4,6** — MERGED (+14 gauntlet). Writes at half bonus for plies 2/4/6. 6-ply is already included.
- [x] **Correction history expansion (4→6)** — MERGED (+11 combined gauntlet). Added minor_key + major_key.
- [x] **Threat-aware history (4D indexed)** — MERGED (+18 H2H, +16 cross-engine).
- [x] **Eval-dependent aspiration delta** — MERGED (+75 cross-engine).
- [x] **TT cutoff retroactive history** — MERGED (+19 raw Atlas combined with asp depth reduction).
- [x] **Complexity-aware LMR** — MERGED (+9 raw Atlas).
- [x] **Cuckoo cycle detection** — MERGED (+48 Elo, Hercules).
- [x] **Alpha-raise LMR tracking** — Already implemented (alpha_raised_count / 2 extra reduction).
- [x] **History aging (×0.80 per game)** — MERGED (+11 combined, Hercules).
- [x] **Pin-aware SEE** — REJECTED (-7.8 raw). Asymmetric pin mask biases SEE.
- [x] **GHI mitigation (50mr hash)** — REJECTED (-28.5 raw). TT fragmentation too costly.
- [x] **NMP cutNode restriction** — REJECTED (-23 raw). Too restrictive.
- [x] **LMP depth 9** — REJECTED multiple times. 3+d² at depth<=8 confirmed optimal.
- [x] **LMP 4+d²** — REJECTED (-4.9 raw after landscape shift). 3+d² confirmed.
- [x] **NMP Divisor 170** — REJECTED (-32 raw Atlas, -15.8 post-landscape). NMP 200 confirmed.
- [x] **TT Near-Miss Margin 96** — REJECTED (-23 raw Atlas). Margin 80 confirmed.
- [x] **Mate Distance Pruning** — REJECTED (dead flat, triggers too rarely at 10+0.1).
- [x] **FH Blend Depth Gate 4** — REJECTED (-10 raw Atlas).
- [x] **History Divisor 4000** — REJECTED (neutral at best, never gains).
- [x] **Futility 50+d*50** — MERGED on v5 after landscape shift (+10.2), then stable.
- [x] **Eval-depth bonus** — REJECTED (-11 raw, unchanged post-4D).
- [x] **SEE history gate** — REJECTED (-21 raw post-4D).

## NNUE / Model

### High Priority

- [ ] **Frozen-FT v7 training** — use convert-checkpoint to init v7 from best v5, freeze FT, train hidden layers. Bullet fork (adamtwiss/bullet) has freeze support. (Ready to test on GPU hosts)
- [ ] **Mix blunder/material data into training** — Titan generating ~86M positions/day. Mix 1-2% into T80 data for v7 training. (In progress)
- [ ] **NNZ sparse L1 matmul** — all top engines use this. Our test at H=1024 was slower, but need weight repermutation for NNZ locality. (Source: Viridithas/Reckless/Obsidian reviews)
- [ ] **Pairwise FT for v7** — all top engines use pairwise (768→1536→pw→768) not direct 1024. Train a pairwise FT net and add hidden layers. (Source: architecture consensus)

### Medium Priority

- [ ] **Threat accumulator** — Reckless/Stormphrax encode piece-attack relationships as a second NNUE input (60K+ features). This is why our history-based threat awareness fails — the signal needs to be in the eval. Requires Bullet training support. (Source: Reckless/Stormphrax reviews)
- [ ] **Dual net (big+small)** — SF selects 1024-wide or 128-wide net by material. Saves NPS in simple positions. (Source: SF review)
- [ ] **Layer stacks (material-bucketed)** — SF uses 8 sub-networks selected by material count. Different from output buckets. (Source: SF review)
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

## Done (2026-03-31)

- [x] **2026-03-31** Correction history 4→6 sources (+11 combined, Hercules gauntlet)
- [x] **2026-03-31** History aging ×0.80 instead of clear (+11 combined, Hercules gauntlet)
- [x] **2026-03-31** Atlas: Complexity-aware LMR + cont-hist deep writes (+9 raw, Atlas gauntlet)
- [x] **2026-03-31** Atlas: TT cutoff retroactive cont-hist penalty (+19 raw combined, Atlas gauntlet)
- [x] **2026-03-31** Atlas: Aspiration fail-high depth reduction (+19 raw combined, Atlas gauntlet)
- [x] **2026-03-31** Atlas: Cut node tracking (foundation for NMP/LMR/IIR gating)
- [x] **2026-03-31** Pairwise+v7 inference support in Coda (forward path + converter)
- [x] **2026-03-31** sample-positions: extract random positions from T80 binpack as EPD
- [x] **2026-03-31** 12 engine reviews (Stockfish, Reckless, Obsidian, Berserk, Caissa, Stormphrax, Viridithas, Koivisto, Seer, Quanticade, PlentyChess, Halogen)
- [x] **2026-03-31** v7 768pw training config for GPU hosts
- [x] **2026-03-31** 33-engine RR: Coda #18 (-14), +36 over GoChess
- [x] **2026-03-31** TM v2: extend time on best-move instability (+27 raw, testing)

## Rejected (2026-03-31)

- [x] GHI mitigation (50mr TT key bucketing) — -28.5 raw, TT fragmentation too costly
- [x] Pin-aware SEE (side-to-move only) — -7.8 raw, asymmetric pin mask biases SEE
- [x] TM v1 unified refactor — -17 raw, disrupted existing calibration
- [x] Atlas: Opponent move feedback + eval depth bonus — -17 raw combined
- [x] Atlas: Pawn hist in LMR — rejected
- [x] Atlas: Various retry candidates (NMP 170, FH blend, TT margin 96) — all negative
