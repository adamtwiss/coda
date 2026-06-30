# Experiment log — blindspot data: dose / dilution series (2026-06-29/30)

**Question:** does mixing the 150/80 eval-blindspot corpus (~300M T80
positions Coda mis-scores, LC0 labels) into a normal T80 S200 run improve
the net — and does the gain survive **dilution** (the production stage-1
mix would be ~5%, not 50%)? Prior SFvCoda result: the gain held even
heavily diluted (it's the count of new targeted points that matters, not
the fraction). Testing whether blindspot behaves the same.

All runs: GPU4, current prod-shape S200 recipe (identical across arms so
they cross-compare). Net-vs-net SPRT on `main`, `[-1.5, 1.5]` STC. Only
`--dataset-dir` differs.

## CRITICAL: the trainer mixes by BYTES, not positions (verified in code)

The `--data-order interleave` path is a **size-weighted round-robin**
(`sfbinpack.rs`): each file's sampling weight = `scan_chunk_starts(path).len()`
∝ **file bytes**; streams are drawn ∝ byte-weight and emit 4096-pos blocks;
small files **wrap** (re-read from start) to keep filling their weight, with
the wrap count salting the fen-skip hash so a different subset passes each
pass. So **the fraction of positions the net trains on = the BYTE fraction**,
NOT the unique-position fraction. Because blindspot is un-chained (~33 B/pos
vs T80 ~3 B/pos, ~11× less dense), its byte-share ≫ its position-share, and
the 300M unique positions get **oversampled ~5–9×** (wrapped) to fill it.

| run | blindspot bytes | **EXPOSURE (net trains on)** | unique-pool fraction |
|---|---|---|---|
| 1 (1×T80, mar)  | 10G / 20G | **~50%** | ~10% (300M / ~3–4B) |
| 2 (4×T80, jan-apr) | 10G / 50G | **~20%** | ~2.5% |

So run 2 is a **2.5× exposure cut (50%→20%)**, not "2.5%". A production-like
~2.5–5% *exposure* needs blindspot at ~2.5–5% of BYTES → ~20× T80 (or less
blindspot byte-volume), not 4×. The GPU starvation in run 1 (~50% idle) is
the byte-weight tell — half the loader's bytes were the slow blindspot.

Shared recipe:
```
--superbatches 200 --warmup 20 --final-lr 1e-6 --seed 44 \
--swa-start-sb 180 --save-rate 200 --wdl 0.20 --fen-skip-prob 0.5 \
--kb-layout reckless --hidden-activation crelu --factoriser \
--ft-size 1024 --l1-size 32 --qat --data-order interleave --mse-power 3 \
--soft-early-ply 28 --soft-early-ply-floor 0.25 \
--val-data /workspace/all/test80-2023-11-nov-2tb7p.min-v2.v6.binpack
```

## Run 1 — 50% dose (the strong-form / pathological case)

- control `bs-ctrl-mar2024-s200` = mar-2024 T80 only.
- mix `bs-mix-mar2024-s200` = mar-2024 + 6 blindspot → ~50% blindspot.
- Note: the 50% un-chained mix data-starved the GPU (~2× wall-clock); the
  *result* is unaffected (fixed SB). Worst case for the loader; production
  dilution avoids it.

Nets (v10 FT1024/L1=32): control `5EC53278`, mix `CE196733`.

| metric | control | mix | delta |
|---|---|---|---|
| net_pref overrate.epd (lower=better) | +29cp | −24cp | mix −53cp (≈1 SE, within noise) |
| net_pref wandering_bishop | +102cp | +106cp | −4cp (flat) |
| **strength SPRT 2402** (mix vs ctrl) | — | — | **early +10 ±10.5, LLR ~0.6 →H1 (in progress)** |

## Run 2 — diluted to ~20% (4× T80 carrier)

- control `bs4x-ctrl-s200` = 4×T80 (jan-apr 2024, 40G).
- mix `bs4x-mix-s200` = 4×T80 + 6 blindspot (50G) → **exactly 20% blindspot**.
- Launched 2026-06-30 on GPU4 (control full-speed 3.3M pos/sec; mix mild
  starvation only at 20%).

| metric | control | mix | delta |
|---|---|---|---|
| strength SPRT (mix vs ctrl) | — | — | _pending_ |
| net_pref overrate / wandering | — | — | _pending_ |

## Read

If Run 1 (50%) is positive and Run 2 (20%) retains most of it → dilution-
robust, greenlights the diluted stage-1 mixing plan + Atlas's
extract-from-all-datasets pipeline. If the gain collapses with dilution →
blindspot differs from SFvCoda; concentration matters, rethink stage-1 dose.
