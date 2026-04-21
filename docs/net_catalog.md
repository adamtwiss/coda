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
