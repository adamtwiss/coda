# Icarus Search/Eval Review (2026-04-19)

Icarus 1.1.0, Rust workspace (`icarus`, `icarus-board`, `icarus-common`),
edition 2024. Source at `~/chess/engines/icarus/`. Author @Sp00ph; README
credits Hobbes/Stormphrax/Viridithas/Cherry lineage, "no LLM-assisted code."
Rank unlisted; architecture places it in the 3700-3800 cluster. Strength =
clean fractional-depth (`DEPTH_SCALE=1024`) search + Bullet-trained
704→1024→16→32→1 SCReLU-pairwise net with 14 king buckets + NNZ-sparse int8
L1, in ~4.5K lines.

## Search architecture

Single `search<Node: NodeType>` in `src/search/search.rs`. **Typestate node
kinds** (`Root`/`PV`/`NonPV`, `:19-46`) — const-bool `ROOT`/`PV` and
`type Next: NodeType`, SF's `NT` template as Rust traits. Cutnode threaded
through negamax as a bool arg (`:64`).

- **Fractional depth** (`:17`): all gates in 1024ths — RFP `<6144`,
  NMP `>=3072`, singular `>=7168 (+1024 if tt_pv)`, LMR base 0.5 ply
  (`params.rs:116-131, 32-38`). Coda uses integer plies — biggest gap.
- **PVS + aspiration** (`searcher.rs:337-374`): initial 25 cp, widen
  ×doubling default; fail-low sets `beta = midpoint(alpha, beta)`.
- **NMP** (`:217-270`): R = `6144 + depth·128/640` (1024ths). Verify only at
  `depth > 14336`; below returns raw score. No `nmp_min_ply` machinery.
- **RFP** (`:203-215`): `score_est - 50·iD - 768·iD²/128 >= beta` — quadratic
  margin. Returns `midpoint` on cutoff (Coda H0 #2114).
- **LMP** (`:326-330`): `moves·1024 >= (4096 + 1024·lmr_depth²) >> !improving`
  on post-LMR fractional depth.
- **Futility** (`:334-342`): stage-terminating via `skip_quiets()`.
- **SEE pruning** split quiet/tactic (`:312-320, 353-356`), tactic past
  `YieldGoodNoisy`.
- **History pruning** (`:344-350`): `hist < -2000·lmr_depth/1024` →
  `skip_quiets(); continue`.
- **ProbCut** — **TT-only, no re-search** (`:273-286`). See E3.
- **Singular ext** (`:361-399`): positive +1024 (+1024 more if `!PV &&
  s_score+20<beta`), multi-cut midpoint, **negative ladder** -3072/-2048/
  -1024 (triple negext = Hobbes outlier).
- **LMR** (`params.rs:189-197`, `search.rs:427-433`): `base + LOG[d]·LOG[m]·
  div`, precomputed `LOG[]` since `f32::ln` isn't const. Post-loop
  `±1024` for PV/tt_pv/in_check/cutnode, continuous `-DEPTH_SCALE·
  quiet_hist/8192`.
- **No do-deeper/shallower**; single re-search on `score > alpha`.
- **Hindsight extension** (`:190-199`) — Coda H0 #866/#881.
- **Correction history** (`history/mod.rs:75-97`): 7 sources incl.
  **cont-2ply** (Coda lacks — E1). Cont-corr key
  `[stm][prev_piece][prev_to][cur_piece][cur_to]`.
- **SMP**: LazySMP, `Arc<GlobalCtx>`, `atomic_wait` parking.

## Move ordering

`src/search/move_picker.rs` (170 lines — half of Coda's).

- **Stages** (`:14-22`): `TTMove → GenNoisy → YieldGoodNoisy → GenQuiet →
  YieldQuiet → YieldBadNoisy`. Bad noisies deferred to last via in-place
  `bad_noisies` prefix (`:122-123`).
- **Noisy score** (`:97-103`): `tactic_hist/8 + victim*8` — MVV weighted 64×
  tactic hist.
- **Quiet score** (`:133-139`): `main_hist + cont(1,2,4) + 8000 *
  gives_direct_check`. No from/to axes, no pawn history.
- **`i64`-packed argmax** (`:63-73`): `(score<<32)|index` folded with `max`
  in a single pass — vectorises. Coda uses a plain scan.

### History dimensions

| Table | Shape | Notes |
|---|---|---|
| `MainHist` | `[stm][from][from_att][to][to_att]` | Threat bits are outer indices — 4 disjoint sub-tables per (stm, from, to). `main.rs:12`. |
| `TacticHist` | `[stm][piece][to]` | **No victim axis** (`tactic.rs:12-13`); Coda's `[piece][to][victim]` is richer. |
| `ContHist` odd | `[stm][prev_piece][prev_to][piece][to]` | 1-ply. |
| `ContHist` even | same | Shared for **both 2-ply and 4-ply** (`mod.rs:63-64, 120-131`). |
| `ContCorrHist` × 2 | `[stm][prev_piece][prev_to][cur_piece][cur_to]` | 1-ply + 2-ply. |

- Continuation plies 1,2,4 (no 6-ply); Coda has 1,2,4,6.
- Bonus/malus per-ply-class SPSA-tuned separately (`params.rs:67-86`).
- **Cont-hist bonus is `total`-scaled**: `apply_gravity(entry, total,
  amount)` where `total` = summed cont score at cutoff (`cont.rs:73-88`).
  High-total cutoffs decay siblings harder. Coda uses plain gravity. Note
  the `// FIXME: fixing overflow here loses elo.` at `:27` — intentional
  wrapping arithmetic.

## NNUE / eval

- **Architecture** (`nnue/network.rs:24-64`): 704 → 1024 (pairwise) → 16 →
  32 → 1. **704 input**: own king always maps to White in `Feature::idx`
  (`accumulator.rs:36-38`), collapsing 768 → 704. 14 king buckets, horizontal
  mirror.
- **NNZ-sparse int8 L1** (`inference.rs:74-135`): gather non-zero-chunk
  indices then matmul only over those chunks (unroll 4). Optional
  `count-nnz`/`count-act` cargo features for offline sparsity profiling.
  **Coda tried this in `sparse_l1.rs` — benched 1.8-2.4× slower** on our
  AVX2 path; skip.
- **Activation** (`inference.rs:32-70`): FT halves clamped `[0,255]`,
  `mulhi_shl7` pair-combined via `packus` which clamps negatives to 0 —
  only 2 of 4 halves need `max(_,0)`.
- **Accumulator**: lazy backward-scan + forward-chain via `add_sub` variants
  (`network.rs:221-277`) — same as Coda. **Refresh uses batched `add4`/
  `sub4`** (`accumulator.rs:160-194`); Coda's refresh is per-feature.
- **Finny cache** `[stm][mirror(2)][bucket(14)]` (`accumulator.rs:73-95`) —
  mirror as separate dimension, cleaner than Coda's fold.
- **Training**: self-play only, Bullet (jw1912), initial version from PeSTO
  PSTs. Net `glide-v22`, SHA-256-verified GH fetch. No threats, no transfer.
- **Material scaling** (`position.rs:81-95`): `eval * (25000 + Σ count ·
  piece_scale) / 32768`. Coda #529 per-piece tunable H0.

## Notable / novel mechanisms

Fractional depth `DEPTH_SCALE=1024`; NodeType typestate (`search.rs:19-46`);
NNZ-sparse int8 dpbusd L1 (`inference.rs:74-155`); batched `add4`/`sub4`
refresh (`accumulator.rs:160-194`); `packus` half-clamp elision
(`inference.rs:45-54`); 704 input via merged king plane;
`i64`-packed argmax movepick (`move_picker.rs:63-73`); cont-hist
total-scaled gravity with intentional wrap (`cont.rs:27, 73-88`);
RFP quadratic margin + return midpoint; sub-ply NMP verify gate with no
`nmp_min_ply`.

## Testable Experiments for Coda

Prior art grep-checked in `experiments.md`. Dropped: hindsight extension
(#866/#881 H0), RFP return-blend (#2114 H0 -0.4), 4-ply cont-hist already
present, sparse-L1 dpbusd (rejected in `src/sparse_l1.rs`), per-piece
material scaling (#529 H0), 1-ply cont-corr Coda already has, cutnode
propagation Coda already has, hindsight reduction Coda already has.

### E1. 2-ply continuation correction history (joined 5D key)

- **Icarus** (`history/contcorr.rs:6-9`, `mod.rs:93-94, 157-158`): two
  `ContCorrHist` tables keyed `[stm][prev_piece][prev_to][cur_piece][cur_to]`
  (1-ply, 2-ply), weights `corr_cont1_factor=64`, `corr_cont2_factor=64`.
- **Coda today** (`search.rs:1364-1438`): cont-corr is 1-ply only, `[piece]
  [to]` — no 2-ply, no `prev × cur` join. `trans_corr` partly covers 1-ply.
- **Prior art**: 0 hits for `2-ply corr` / `follow-up corr`. Same as Hobbes
  E1 — corroborated across two reviews.
- **Sketch**: add `contcorr_2ply: [[[[[i16;64];6];64];6];2]` (≈590 KB/side).
  Update in `update_correction_history`; add `CORR_W_CONT2`.
- **Magnitude/risk**: +2 to +4 Elo (SF/Berserk/Obsidian/Reckless/Hobbes/
  Icarus all have it). Low risk. **Highest-leverage item.**

### E2. Cont-hist gravity scaled by summed cont score at cutoff

- **Icarus** (`history/cont.rs:73-88`, `mod.rs:111`): gravity `(entry, total,
  amount)` with `total = cont(1)+cont(2)+cont(4)` at cutoff — decay uses
  `total·|amount|/MAX` not `entry·|amount|/MAX`. Loud cutoffs decay
  siblings harder.
- **Coda today**: plain `entry + amount - entry·|amount|/MAX`.
- **Prior art**: 0 hits. Retune-on-branch mandatory.
- **Magnitude/risk**: +0 to +3 Elo. One extra i32 arg; entangles axes.

### E3. TT-only ProbCut (no re-search)

- **Icarus** (`search.rs:273-286`): TT Lower/Exact + `score >= beta+375` +
  `tt_depth >= depth-2048` → return directly.
- **Coda today**: ProbCut runs reduced-depth + verification.
- **Prior art**: 0 hits. Add pre-loop TT-only shortcut with
  `PROBCUT_TT_MARGIN` / `_DEPTH_OFFSET`.
- **Magnitude/risk**: +0 to +2 Elo. Cheap, low risk.

### E4. Batched `add4`/`sub4` accumulator refresh kernels

- **Icarus** (`accumulator.rs:160-194`, `network.rs:144-158`): 4-add/4-sub
  chunks + remainder.
- **Coda today**: per-feature refresh. Flagged in Reckless (#793) and Hobbes
  (#22) reviews — Icarus is third consensus point.
- **Sketch**: AVX2/AVX-512 `acc_add4`/`acc_sub4`, call from
  `recompute_threats_full` and PSQ refresh.
- **Magnitude/risk**: 1-3% NPS. Low risk.

### E5. `i64`-packed argmax in movepicker

- **Icarus** (`move_picker.rs:63-73`): `(score<<32)|index` folded with `max`
  — single-pass, autovectorises. Coda uses a plain scan. Sub-1% NPS
  micro-opt.

### E6. Fractional depth (`DEPTH_SCALE=1024`) — horizon item

Icarus (`search.rs:17`), Stormphrax, Cherry all use fractional depth for
sub-ply SPSA nudges. Coda is integer. Structural refactor; log for roadmap,
not near-term SPRT.

## Confirmed-clean / Not-worth-porting

Cutnode propagation, hindsight reduction, 4-ply cont-hist, continuous
LMR-history divisor — Coda has these. Sparse L1 dpbusd — rejected
(`sparse_l1.rs`). RFP return-blend #2114, hindsight extension #866/#881,
per-piece material scaling #529 — Coda H0. Tactic history without victim
axis — Coda's `[piece][to][victim]` is richer. 704 king-plane merge —
net-training, not search.

## Sources

Icarus (`~/chess/engines/icarus/`):
- `src/search/search.rs`: `:17` DEPTH_SCALE, `:19-46` NodeType, `:190-199`
  hindsight-ext, `:203-215` RFP, `:217-270` NMP, `:273-286` TT-ProbCut,
  `:311-357` LMP/FP/hist-prune/SEE, `:361-399` SE, `:404-455` LMR+PVS.
- `src/search/params.rs:5-131, 157-197`; `src/search/searcher.rs:337-443`.
- `src/search/history/{mod,main,cont,tactic,corr,contcorr}.rs`; note
  `cont.rs:27` FIXME.
- `src/search/move_picker.rs:14-170, :63-73`.
- `src/search/transposition_table.rs:22-353` (3-slot cluster,
  SIMD key-idx :81-90, mmap+hugepage).
- `src/nnue/network.rs:24-64, :102-166, :221-277`;
  `src/nnue/accumulator.rs:12-95, 97-194`;
  `src/nnue/inference.rs:26-71, 74-155`.
- `src/position.rs:81-95`; `README.md:63-109`; `network.txt:1`.

Coda (`/home/adam/code/coda/`):
- `src/movepicker.rs:28, 910-928`;
  `src/search.rs:1364-1438, 193-202, 4595-4662`;
  `src/sparse_l1.rs:5-16`;
  `experiments.md:6274, 7428, 16199`.
