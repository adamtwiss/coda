# NNUE Net Catalog

Authoritative list of v9 nets, their OpenBench hashes, provenance, and status.
Update this file when you promote a new production net or retire an old one.

**Current v9 production:** `1EF1C3E5` —
`net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue`.
Published as `v0.4.0-nets` release; referenced by `net.txt`. Promoted
2026-04-26 after deployment-package SPRT #789 H1'd at +4.9 Elo
(net swap alone +3.3 in #782, plus tune-784 retune for the rest).
**Caveat:** filename "C8fix" labels the FIRST C8 fix only (a8e2c7d).
The "Complete C8 fix" (62931d1) was committed Apr 25 20:15, only
~1h before the SB800 file's mtime (21:52) — SB800 train (~30-40h)
was already done. So 1EF1C3E5 contains C8fix-1 only ("noisy threats").
A future SB800 retrain on `bullet/feature/no-blocking-sync` (post-
62931d1) would actually include C8fix-2 — that's the next prod
candidate. All v9 SPRTs against trunk should pass
`--dev-network 1EF1C3E5 --base-network 1EF1C3E5`.

**Previous v9 production:** `DAA4C54E` —
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
| `1EF1C3E5` | kb10 | `net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue` | **PROD** (v0.4.0-nets) | SB800 + factor. Filename "C8fix" labels the **first** C8 fix only (a8e2c7d Apr 22). The "Complete C8 fix" (62931d1, Apr 25 20:15) was committed only 1h37m before file mtime (Apr 25 21:52); SB800 train (~30-40h) was already done — file mtime is the conversion timestamp. So this net has **C8fix-1 only ("noisy threats")**, NOT C8fix-2. Promoted 2026-04-26 after #789 H1 +4.9. Net swap alone +3.3 (#782); tune-784 retune-on-this +3.0 (#788); deployment package together +4.9 (#789). Local: `nets/net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue` (Apr 25 21:52). |
| `6C154331` | kb10 | `net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-xray-factor.nnue` | rejected | SB800 + factor + **x-ray training**. Trained 2026-04-26 (GPU 2 promised batch). Tune #830 retune-on-net (5K iters, 80 params, 64 changed). Net-vs-net SPRT #836 vs PROD 1EF1C3E5: **H0 -10.7 ±6.3 / 3100g** at retuned state. Combined with #835 +5.2 (pre-vs-post tune-830) implies the underlying net is ~15 Elo behind 1EF1C3E5 at equal-tune state. **Conclusion:** x-ray training does not currently translate to net Elo gain at SB800 / current recipe. Net-training thread should iterate (different recipe / longer SBs / alternative architecture). Local: `nets/net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-xray-factor.nnue` (Apr 27 13:31). Bench (with tune-830 outputs) 1505199. |
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

| Hash | File | Decay | Notes |
|---|---|---|---|
| `573854EF` | `net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-1e2.nnue` | 1e-2 | SB200 probe, 13.48% threat row sparsity, +11.22 Elo SPRT vs dense C8fix at 1100g (#1 probe — see project_group_lasso_acts_as_regularizer). |
| `7E9AEDD2` | `net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-3e2.nnue` | 3e-2 | SB200 probe, more sparsity, lost SPRT vs 1e-2 (-36 Elo). |
| `3D371C10` | `net-v9-768th16x32-kb10-w15-e200s200-crelu-grouplasso-5e2.nnue` | 5e-2 | SB200 probe, ~11 MB compact, dead at -156 Elo vs 1e-2. Decay too aggressive. |
| `12232F49` | `net-v9-768th16x32-reckless-w15-e200s200-warm30-l1e-6.nnue` | element-wise 1e-6 | Earlier element-wise L1 sparsity probe (not group-lasso). |

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

## Recently completed / rejected

- **SB800 factor + x-ray** (`6C154331`, 2026-04-27): trained, retuned
  (#830), SPRT'd as -10.7 vs PROD 1EF1C3E5 (#836). x-ray training
  doesn't carry net Elo at this SB / recipe. See main entry above.

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
