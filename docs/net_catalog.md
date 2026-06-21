# NNUE Net Catalog

Authoritative list of v9 nets, their OpenBench hashes, provenance, and status.
Update this file when you promote a new production net or retire an old one.

**Current v9 production:** `035195DB` — `net-035195DB.nnue`
(source file `multi-v6-l132-s5-swa.nnue`). **First L1=32 prod** (prior prods
were L1=16). Uploaded as an asset to the existing `v0.7.0-nets` release (no
version bump — parity-strength switch); referenced by `net.txt`. Promoted
2026-06-21 with SPSA core tune #2166 applied (73 non-TM params, re-applied
onto current main as `experiment/l132-prod-candidate`). Recipe: v6 multistage,
FT=1024 **L1=32**, kb10 reckless, crelu hidden, factoriser, wdl 0.20, mse-power
3.0, interleave, fs0.5, ply-skip f25, stage 5 = 1600SB SWA tail. SPRT **#2181
H1 +4.1 ±3.1** (L1=32 candidate+tune+net vs prior prod 549C20A5). All v9 SPRTs
against trunk should now pass `--dev-network 035195DB --base-network 035195DB`.

**Previous v9 production (L1=16, pre-L1=32 switch):** `549C20A5` — `net-549C20A5.nnue`
(source file `multi-v6-s5-swa.nnue`). Published as `v0.7.0-nets`;
referenced by `net.txt`. Promoted 2026-06-11 with SPSA tune #1896
applied (STC 5000-iter, NMP cluster reconciled to the post-#1903/#1906
merged structure). Recipe: v6 multistage (FT=1024, kb10 reckless,
crelu hidden, factoriser, wdl 0.20 from primer, mse-power 3.0,
interleave, fs0.5), stage 5 = 1600SB extended terminal with lower
initial LR, SWA tail. SPRTs: #1895 untuned +5.3 vs E4B66CE4; #1908
tune +3.1 vs untuned; #1909 bundle +7.9 vs prod (pre-merge trunk);
post-merge validation SPRT vs baseline/pre-v6s5 in flight. All v9
SPRTs against trunk should now pass
`--dev-network 549C20A5 --base-network 549C20A5`.

**Earlier v9 production:** `E4B66CE4` — `net-E4B66CE4.nnue` (source
`multi-v6-s4-swa.nnue`, `v0.6.0-nets`). First hash-named prod. Was prod
2026-06-09 → 2026-06-11. v6 multistage stage 4 (250/300/550/1000 SB
geometric ladder), SWA. Promoted after #1857/#1858-era H1s (+19.4 raw /
+12.4 swa vs E2773E50-era prod, swa won H2H by +1).

**Earlier v9 production:** `E2773E50` —
`net-v9-768th16x32-kb10-w15-e1200s1200-crelu-factor-xtradata-swa.nnue`.
Published as `v0.5.0-nets` release. Was prod 2026-05-30 → 2026-06-09.
Promoted after deployment-package SPRT #1645 H1'd at +13.1 ±5.2 Elo
vs prior prod 1EF1C3E5. Prod recipe (kb10, w15, crelu hidden, factoriser)
trained on extra data, SWA-averaged over SB1150-1200, schedule completed
to s1200. The +13 decomposes as: net swap alone +12.3 (#1630, untuned
swa-s1200 vs prod) plus +7.3 cumulative `--core` SPSA retune-on-net
(#1631 +4.0 → #1643/#1644 +3.3).

**Previous v9 production:** `1EF1C3E5` —
`net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue`
(`v0.4.0-nets`). Was prod from 2026-04-26 until 2026-05-30. Promoted
then after #789 H1 +4.9. Contains C8fix-1 only ("noisy threats"), NOT
C8fix-2 — the SB800 train (~30-40h) was already done before 62931d1
(Apr 25 20:15) landed, ~1h before the file's mtime (21:52).

**Earlier v9 production:** `DAA4C54E` —
`net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue`. Was prod from
v0.3.0-nets release until 2026-04-26.

**Current v5 production:** `net-v5-768pw-consensus-w7-e800s800.nnue`
(`v0.2.0-nets` release). Still the net.txt target for v5/main branch builds.

## Invariants

1. **Both sides of any SPRT must use the same net.** Different nets on each
   side measures net-vs-net, not search-vs-search.
2. **The net used in SPRT must match the net the tune ran against.** Tune
   values are net-specific.
3. **The `Bench:` line in the commit message must be measured with the net
   OB will actually build with** — not a local convenience net.

## v9 nets (current and recent)

| Hash | Layout | File | Status | Notes |
|---|---|---|---|---|
| `E2773E50` | kb10 | `net-v9-768th16x32-kb10-w15-e1200s1200-crelu-factor-xtradata-swa.nnue` | **PROD** (v0.5.0-nets) | SB1200 + factor + extra data + SWA (averaged SB1150-1200). Prod recipe (kb10, w15, crelu hidden). Promoted 2026-05-30 after deployment-package SPRT #1645 H1 +13.1 ±5.2 vs 1EF1C3E5. Net swap alone +12.3 untuned (#1630); +7.3 cumulative `--core` retune-on-net (#1631 +4.0, #1643/#1644 +3.3). Bench (embedded + tuned trunk) 4782984. Local: `nets/net-v9-768th16x32-kb10-w15-e1200s1200-crelu-factor-xtradata-swa.nnue` (= `gpu4-xtradata-swa-1150-1200-s1200.nnue`). |
| `1EF1C3E5` | kb10 | `net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue` | previous PROD (v0.4.0-nets, retired 2026-05-30) | SB800 + factor. Filename "C8fix" labels the **first** C8 fix only (a8e2c7d Apr 22). The "Complete C8 fix" (62931d1, Apr 25 20:15) was committed only 1h37m before file mtime (Apr 25 21:52); SB800 train (~30-40h) was already done — file mtime is the conversion timestamp. So this net has **C8fix-1 only ("noisy threats")**, NOT C8fix-2. Promoted 2026-04-26 after #789 H1 +4.9. Net swap alone +3.3 (#782); tune-784 retune-on-this +3.0 (#788); deployment package together +4.9 (#789). Local: `nets/net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue` (Apr 25 21:52). |
| `6C154331` | kb10 | `net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-xray-factor.nnue` | regression vs PROD; investigation open | SB800 + factor + **x-ray training**. Trained 2026-04-26 (GPU 2 promised batch). Tune #830 retune-on-net (5K iters, 80 params, 64 changed). Net-vs-net SPRT #836 vs PROD 1EF1C3E5: **H0 -10.7 ±6.3 / 3100g** at retuned state. Combined with #835 H1 +6.0 (pre-vs-post tune-830) implies the underlying net is ~15 Elo behind 1EF1C3E5 at equal-tune state. **Important framing (Adam 2026-04-27):** the C8fix-xray training change is a *committed correctness fix* to the training pipeline — future nets all use the corrected pipeline. This isn't an "adopt or reject" experiment; PROD 1EF1C3E5 is on borrowed time as the last net trained with the bug present. The -10.7 Elo is a diagnostic alarm, not a verdict. Open candidates for investigation: (a) which sub-change in the fix carries the regression — C8fix-2 alone vs adding x-ray feature labels to training data; (b) recipe re-search (LR tail, WDL, save-rate) at SB200 to find a recipe that recovers parity under the corrected pipeline; (c) widened-range SPSA on params tune-830 pinned at boundaries (HIST_PRUNE_MULT -20%, HIST_BONUS_OFFSET -57%). Local: `nets/net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-xray-factor.nnue` (Apr 27 13:31). Bench (with tune-830 outputs) 1505199. |
| `CC483681` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200-crelu-C8fix-factor.nnue` | C8fix-2 isolation test | **First net to actually contain C8fix-2.** SB200 + factor + complete C8-fix (both halves a8e2c7d + 62931d1). Trained Apr 26 ~01:15 (start) → Apr 26 08:31 (file mtime), well after the 62931d1 commit. Net-vs-net SPRT vs C0A97CF4 (#794, on tune-784 trunk) measures C8fix-2 contribution at SB200. Caveat: trunk tunables are calibrated against 1EF1C3E5 (noisy threats), so tunables fit the BASE here, not the DEV — SPRT result is a lower bound on C8fix-2 contribution. Local: `nets/net-v9-768th16x32-kb10-w15-e200s200-crelu-C8fix-factor.nnue` (Apr 26 08:31). Bench (post tune-784 main) 1502300. |
| `FF8C93DC` | kb10 | `net-v9-768th16x32-kb10-w15-e400s400-crelu-C8fix-factor.nnue` | intermediate | SB400 + factor. Apr 23 12:59 — predates 62931d1 (Apr 25), so C8fix-1 only. Filename label is C8fix-1. |
| `C0A97CF4` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200-factor.nnue` | C8fix-2 isolation base | SB200 + factor (trained Apr 22 06:37, before 62931d1 commit Apr 25 — has only C8fix-1). Used as base for #794. Same threat semantics as 1EF1C3E5, so tunables fit. Hidden activation: per training script default at the time (screlu-hidden — name lacks `-crelu` suffix). Bench (post tune-784 main) 1454351. |
| `80CB364B` | kb10 | `net-v9-nonfactor-sb400-warm30.nnue` | candidate | Non-factoriser SB400 warm30 on post-C8-fix-1 Bullet. Bench 3058198 on tuned trunk. First-move cut 76.9%, NMP cutoff rate 49%, EBF 1.74. Was under SPRT vs prod and vs C8-fix S200 (2026-04-23). |
| `1836917B` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200-crelu-C8fix.nnue` | tune baseline (no factor) | C8-fix S200 + crelu-hidden, **no factor**. The net #660 tune + #661 +8.25 H1 were validated against. Bench 2575054 on tuned trunk. |
| `DAA4C54E` | kb10 | `net-v9-768th16x32-kb10-w15-e800s800-crelu.nnue` (released) / `...reckless-crelu.nnue` (legacy name) | retired prod | v9 production from v0.3.0-nets through 2026-04-26. Trained from Bullet WITHOUT C8-fix-2 (a8e2c7d C8-fix-1 only). Tunes #585, #586, #599, #743/#747 all ran on this. Same content under both filenames. Replaced by 1EF1C3E5 (v0.4.0-nets). |
| `BFAC07B3` | kb10 | `net-v9-768th16x32-reckless-w15-e200s200-warm30.nnue` | promising | Best-of-warmup sweep (warm30, pre-C8). Lichess noted. |
| `BE5849B6` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200.nnue` | superseded | Earlier kb10 w15 at 200 SBs, s200. |
| `E06A42E8` | kb10 | `net-v9-768th16x32-kb10-lowlr-w15-e200s200.nnue` | experiment | Lower final LR variant. |
| `2B42E458` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200-creluHL.nnue` | experiment | CReLU on hidden layers. |
| `269374CB` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200-creluHL.nnue` | duplicate? | Same canonical name as `2B42E458`; different upload. Confirm content. |

## v9 group-lasso probes (kb10, structural-sparsity experiments)

Sweep results / decision-tree closure: see
`docs/group_lasso_runbook_2026-04-24.md` (Pareto frontier confirmed
at 1e-2; harder decay regresses) and `experiments.md` § "Group-lasso
decay sweep — Probes #2 and #3".

| Hash | File | Decay |
|---|---|---|
| `573854EF` | `net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-1e2.nnue` | 1e-2 (probe #1) |
| `7E9AEDD2` | `net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-3e2.nnue` | 3e-2 (probe #2) |
| `3D371C10` | `net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-5e2.nnue` | 5e-2 (probe #3) |
| `12232F49` | `net-v9-768th16x32-reckless-w15-e200s200-warm30-l1e-6.nnue` | element-wise 1e-6 (pre-group-lasso) |

## v9 nets (ladder / WDL sweep, kb10)

| Hash | File | Notes |
|---|---|---|
| `FAD123D2` | `net-v9-768th16x32-kb10-w0-e200s200.nnue` | WDL 0.00 |
| `4C361CD7` | `net-v9-768th16x32-kb10-w05-e200s200.nnue` | WDL 0.05 |
| `DC9282C6` | `net-v9-768th16x32-kb10-w10-e200s200.nnue` | WDL 0.10 |
| `B6DA1F20` | `net-v9-768th16x32-kb10-w20-e200s200.nnue` | WDL 0.20 |

## v9 nets (reckless warmup sweep)

| Hash | File | Warmup SBs |
|---|---|---|
| `F77E04F8` | `net-v9-768th16x32-reckless-w15-e200s200-warm5.nnue` | 5 |
| `BFAC07B3` | `net-v9-768th16x32-reckless-w15-e200s200-warm30.nnue` | 30 |
| `737DCEA2` | `net-v9-768th16x32-reckless-w15-e200s200-warm40.nnue` | 40 |
| `4AD468DA` | `net-v9-768th16x32-reckless-w15-e200s200-warm50.nnue` | 50 |

## OB purge (2026-04-21)

Purged 45 nets from OpenBench storage: pre-kb10 v9 xray/uniform, v9 pairwise
experiments, v7/v8 deprecated architectures, old v5 filter/selfplay research.
~2.9 GB freed. Historical test results preserved (Test.dev_network and
Test.base_network are CharField, not FK — no cascade delete on Network
removal; see OpenBench/models.py:110,119).

Local `.nnue` files under `nets/` were not touched. If any purged net is
ever needed again, re-upload via `scripts/ob_upload_net.py` with the local
file.

## Incoming (pending training)

- (none currently)

## Recently completed / under investigation

- **SB800 factor + x-ray** (`6C154331`, 2026-04-27): trained, retuned
  (#830 +6 Elo over default tunables on the new net), but net-vs-net
  vs PROD 1EF1C3E5 is -10.7 (#836). The C8fix-xray training change
  is a committed pipeline correctness fix, not an optional adoption —
  investigation needed to find a recipe / tune that recovers parity
  under the corrected pipeline. See entry in v9 table above.

## Catalog hygiene rules

When a new net arrives:
1. SHA256 it (`sha256sum nets/<file>.nnue | head -c 8`) and confirm with
   the OB upload result. Don't infer SHA from filename.
2. Add a row to the appropriate table with: hash, layout, file, status,
   notes (training source, what it's testing, mtime).
3. Note training-time confounds: factoriser? activation (crelu/screlu
   hidden)? C8-fix variant (1, 2, both)? warm-N? final-LR? Filename
   alone is not enough — verify via the training command or net.txt
   alternative.
4. When a net retires from "candidate" to "PROD", update the
   `Current v9 production` line at the top.

## Regenerating this catalog

```bash
for f in nets/net-v9-*.nnue; do
  hash=$(sha256sum "$f" | awk '{print substr($1,1,8)}' | tr 'a-f' 'A-F')
  size=$(stat -c%s "$f")
  layout=$([ $size -eq 63175829 ] && echo "kb10" || echo "kb16")
  echo "$hash  $layout  $(basename $f)"
done | sort
```

Size tells layout: kb10 reckless = 63,175,829 bytes; uniform kb16 = 70,253,715.
