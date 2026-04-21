# NNUE Net Catalog

Authoritative list of v9 nets, their OpenBench hashes, provenance, and status.
Update this file when you promote a new production net or retire an old one.

**Current v9 production:** `DAA4C54E` — `net-v9-768th16x32-w15-e800s800-reckless-crelu`
(kb10 reckless layout, CReLU hidden, 800 SBs). All v9 SPRTs should pass
`--dev-network DAA4C54E --base-network DAA4C54E`.

**Current v5 production:** `net-v5-768pw-consensus-w7-e800s800.nnue` (from net.txt).

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
| `DAA4C54E` | kb10 | `net-v9-768th16x32-w15-e800s800-reckless-crelu.nnue` | **PROD** | Current v9 production. Tunes #585, #586, #599 all ran on this. |
| `BFAC07B3` | kb10 | `net-v9-768th16x32-reckless-w15-e200s200-warm30.nnue` | promising | Best-of-warmup sweep (warm30). Lichess noted. |
| `BE5849B6` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200.nnue` | superseded | Earlier kb10 w15 at 200 SBs, s200. |
| `E06A42E8` | kb10 | `net-v9-768th16x32-kb10-lowlr-w15-e200s200.nnue` | experiment | Lower final LR variant. |
| `2B42E458` | kb10 | `net-v9-768th16x32-kb10-w15-e200s200-creluHL.nnue` | experiment | CReLU on hidden layers. |

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

## v9 nets (uniform / old, retirement candidates)

These predate the kb10 move. No active SPRTs or tunes reference them.
Candidates for OB purge.

| Hash | File | Reason obsolete |
|---|---|---|
| `6AEA210B` | `net-v9-768th16x32-w15-e800s800-xray.nnue` | Uniform KB, pre-reckless era |
| `D3B8F1A6` | `net-v9-768th16x32-w15-e800s600-xray.nnue` | s600 snapshot, inferior to s800 |
| `7E75A3D1` | `net-v9-768th16x32-w15-e200s200-xray.nnue` | Early xray, superseded |
| `57A1D192` | `net-v9-768th16x32-w15-e200s200-xray-fixed.nnue` | Intermediate fix, superseded |
| `7A03FAC8` | `net-v9-768th16x32-w0-e200s200-xray-fixed.nnue` | w0 variant, uniform |
| `1BBB2097` | `net-v9-768th16x32-w0-e200s200-xray.nnue` | w0 variant, uniform |
| `4BBEF00B` | `net-v9-768th16x32-w15-e400s400.nnue` | 400 SBs, uniform, surpassed by e800 |
| `45768E09` | `net-v9-768th16x32-w0-e400s400-noxray.nnue` | no-xray experiment |
| `5F38DD17` | `net-v9-768th16x32-kbc-w15-e200s200.nnue` | consensus-KB layout, superseded |
| `F38AFE6A` | `net-v9-768th16x32-noplyfilter-w15-e200s200.nnue` | data-filter experiment |
| `CA16B950` | `net-v9-768th16x32-1xt80-w15-e200s200.nnue` | single-file data experiment |
| `CE090534` | `net-v9-768th16x32-selfplay-w15-e200s200.nnue` | self-play data experiment |
| `511CE6D3` | `net-v9-768th16x32-w0-e400s400-reckless-kb.nnue` | w0 reckless, early |

## v9 pairwise-threats (experimental, not in production path)

| Hash | File |
|---|---|
| `ECC87D46` | `net-v9-768pwth16x32-w0-e200s200.nnue` |
| `07D94E20` | `net-v9-768pwth16x32-w0-e400s400.nnue` |
| `FAE899ED` | `net-v9-768pwth16x32-w0-e800s700.nnue` |
| `07F6258E` | `net-v9-768pwth16x32-w0-e800s800.nnue` |
| `860E72FF` | `net-v9-768pwth16x32-w0-e800s400ss.nnue` |

## Incoming (pending training)

- **L1-sparse nets** (2026-04-21): GPU hosts running `--l1-decay 1e-7` and
  `--l1-decay 1e-6` on the kb10/reckless/warm30 config. ~4h. First runs are
  smoke tests for the L1+compact pipeline — not Elo tests.

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
