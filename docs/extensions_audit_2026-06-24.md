# Search Extensions Audit — Coda vs top-6 (2026-06-24)

Cross-engine audit of Coda's search-extension logic against the six
stronger reference engines (Stockfish, Reckless, Berserk, Obsidian,
PlentyChess, Alexandria). Method: extract each engine's full extension
scheme (singular/double/triple, multi-cut, negative extensions, check
ext, gating), compare to Coda's `src/search.rs` SE block.

## Coda's scheme (baseline)

Singular-extension-only. SE from `depth ≥ 4` (`SE_DEPTH_10X=40`), margin
`tt_score − depth − xray_bonus`, reduced depth `(depth−1)/2`, TT-slack 3,
`!Upper` bound. Double extension (PV/quiet/corr-aware margin) capped by a
`DEXT_CAP` propagated counter. Multi-cut returns `singular_score`
(`singular_beta` if decisive). Negative extensions −3 (tt≥β) / −2
(cutNode) / **−1 (all-node)**. **No** check/recapture/promotion
extension. **No** triple extension. **No** ply-based extension limiter.

## Correctness / quality

**No bugs found.** The core mechanics all match consensus exactly:
`singular_depth = (depth−1)/2` (6/6 unanimous), TT-slack 3 (5/6; Plenty
uses 5), `!Upper` bound, multi-cut decisive-score guard, DEXT_CAP counter
propagation. Code is clean and well-commented. Check-extension absence is
**correct** — 6/6 references also have no standalone check extension.

## Findings (what Coda doesn't do)

| Aspect | Coda | SF | Reck | Berserk | Obsid | Plenty | Alex | Verdict |
|---|---|---|---|---|---|---|---|---|
| **Triple extension** | ❌ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **6/6 gap** |
| **ttPv&&!PV margin widen** | ❌ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | **4/6 gap** |
| **Global ply ext-limiter** | ❌ | ✗ | ✗ | 2× | 2× | 2× | 2.5× | **4/6 gap** |
| `−1` all-node neg-ext | ✓ uniq | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **Coda-only** |
| singular_depth `(d−1)/2` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | unanimous |
| TT-slack 3 / `!Upper` / multicut | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Check / recapture / promo ext | none | none | none | none | none | none | none | ✓ unanimous |
| SE depth threshold | **4** | 6 | 5 | 6 | 5 | 6 | 6 | Coda lowest |

### Reference singular-margin coefficients (per depth)
SF ≈1.0 (2.2× on ttPv-nonPV); Reckless 1.0 (0.25 if Exact, +1.0 ttPv-nonPV);
Berserk 0.75; Obsidian 1.0; PlentyChess 1.0 (×2 ttPv-nonPV); Alexandria
0.625 (+1.0 ttPv-nonPV). **Coda 1.0** — mid/conservative, but **missing the
ttPv-nonPV widening** that 4/6 (incl. SF + Reckless) apply.

### Triple-extension forms
SF: discrete +1/+2/+3 via double+triple margins (corr/ttMoveHist/ply>root
aware). Reckless: same additive shape, PV/quiet/corr margins. Berserk:
+3 quiet-only at `sBeta−43`, `ss->de≤7` counter. Obsidian: +3 quiet-only at
`sBeta−121`. PlentyChess: continuous ramp 0→+3 (thresholds 6/41). Alexandria:
+3 at `sBeta−75`. **Common gate: triple is quiet-only in 3/6 (Berserk/Obsid/
Plenty); wider margin than double in all.**

## Experiments fired (all [0,3] STC, base main 654c354a = bench 2235673)

- **E1 `ext/triple-extension`** — triple builds on a double when a quiet TT
  move's fail margin clears `dext_margin + TEXT_MARGIN(80)`. The 6/6
  consensus gap; highest-value.
- **E2 `ext/se-ttpv-widen`** — `singular_beta −= depth·SE_TTPV_WIDEN/10`
  when `tt_pv && !is_pv` (default ×1.0 = +depth, Reckless/Alexandria value).
  4/6 consensus.
- **E3 `ext/ply-limiter`** — gate the whole SE block off when
  `ply·10 < SE_PLY_LIMIT_10X(25)·root_depth` (2.5×, Alexandria form). 4/6
  consensus.
- **E4 `ext/drop-allnode-negext`** — ablate the Coda-unique `−1` all-node
  negative extension (5/6 leave it 0).

If E1/E2 win, natural follow-ups: SE depth threshold 4→5/6 (Coda is the
lowest of all 7); SF-style `depth++` bump alongside double/triple
(SF/Plenty/Alex do a variant); capture-allowed triple if quiet-only proves
too narrow.

## Process note

First submission round (#2244–2247) was based off a stale local main
(3066543) — origin/main had advanced to 654c354a (#2229 LMR main-hist).
That produced "Wrong Bench: 2235673" errors (2235673 = current main's
bench) and confounded the dev-vs-base comparison (dev branches lacked
#2229). Fixed by pulling main, rebasing all four branches onto 654c354a,
re-benching, and resubmitting (#2248–2251). Lesson reinforced: always
`git pull origin main` + rebase branches before benching/submitting.
