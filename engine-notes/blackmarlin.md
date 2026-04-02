# Black Marlin Chess Engine - Crib Notes

Source: `~/chess/engines/blackmarlin/` (binary only, no source available for verification)
Author: Doruk Sekercioglu (Rust)
Rating: #13 in RR at +95 Elo (GoChess-era gauntlet; 203 above GoChess-v5)
NNUE: HalfKA with threats (28672->Nx2->1x8), SCReLU, 32 king buckets (file-mirrored), output buckets

**Note**: No source code available in the engines directory -- only binaries. Claims below are from the original GoChess-era review and cannot be re-verified against current source.

---

## 1. NNUE Architecture

### Threat Features (Unique)
BlackMarlin encodes threat information directly into NNUE inputs. The 7th "piece type" represents squares where pieces can be captured by weaker pieces (pawn attacks pieces, minor attacks majors, rook attacks queen). This is baked into the network, not a search heuristic. Incrementally updated when the threat map changes.

This is **unique among all reviewed engines** -- only BlackMarlin does this.

### Network
- 28672 input features (32 king buckets, mirrored, with threat features)
- SCReLU activation, output buckets by material count
- Quantization: FT_SCALE=255, SCALE=64, UNITS=400

Compare to Coda: Coda uses v5/v6/v7 architectures with 16 king buckets, 8 output buckets, SCReLU/CReLU, pairwise mul, Finny tables. BlackMarlin's threat features are the main differentiator.

---

## 2. Pruning Techniques

### RFP
- Margin: `71*d - 62*improving_with_no_threats`, d<=9
- `improving` conditioned on `nstm_threats.is_empty()` -- only improving when no opponent threats
- **Returns `(eval + beta) / 2`** (score blending at cutoff)
- Compare to Coda: Coda uses improving?70*d:100*d, d<=7. Coda returns raw eval. The threat-conditioned improving and score blending are both absent from Coda.

### Razoring
- `306 * depth`, d<=4. Much wider margins than Coda's 400+100*d.

### NMP
- `R = 4 + d*23/60 + (eval-beta)/204`, depth > 4
- **TT-adjusted eval**: uses bounded TT score instead of raw eval
- **Threat guard**: skips NMP at depth <= 7 when opponent has threats
- **Verification at depth >= 10**
- Compare to Coda: Coda uses R=3+d/3+(eval-beta)/200, verify at depth>=12. BlackMarlin has TT-adjusted eval and threat guard that Coda lacks.

### IIR
- Triggers on `depth >= 4 && (no_tt || tt_depth + 4 < depth)` -- also fires on shallow TT entries, not just missing ones.
- Compare to Coda: Coda triggers only when no TT move, depth >= 6.

### Futility
- `86 * lmr_depth`, d<=9, triggers skip_quiets (prunes ALL remaining quiets)
- Compare to Coda: Coda uses 60+lmrDepth*60 per-move (continue, not skip-all).

### LMP
- Formula: `2.97 + d^2`, /1.94 not improving
- Very similar to Coda's `3+d^2`.

### History Pruning
- `h_score < -(d^2 * 138/10)`, d<=6
- Guard: `(!good_capture || eval <= alpha)` -- allows HP on captures only when eval is bad
- Compare to Coda: Coda uses -1500*depth, d<=3. BlackMarlin's is quadratic and deeper.

### QSearch
- **3-capture limit** -- extremely aggressive, only searches up to 3 captures
- SEE >= 0 filter on all captures
- Good-delta SEE cutoff for strong captures

---

## 3. Extensions

### Singular Extensions
- Two tiers: depth >= 7 (full search) and depth < 7 (eval-based, no search)
- Margin: `ttScore - depth` (same as Coda)
- **Double extension** (+2), **Triple extension** (+3 for very singular quiets)
- **PV double extension** (+2)
- **Multi-cut**: return s_beta when s_beta >= beta
- **Negative extensions**: -2 for ttScore >= beta or cut_node
- **History bonus on singular move**: reinforces singular move for future ordering
- Compare to Coda: Coda has SE with multi-cut and -1 negative extension. No double/triple extension, no low-depth eval-based fallback.

### Check Extensions
- +1 when gives_check

---

## 4. LMR

- Single table: `0.50 + ln(d)*ln(m)/2.05`
- History-based: `-= h_score / 112`
- **Near-root bonus**: -1 when `ply <= (depth+ply)*2/5` (novel -- reduce less near root)
- **Threat-aware**: -1 for moves creating new threats
- +1 non-PV, +1 not-improving, -1 killer, +1 cut_node
- Compare to Coda: Coda uses split tables (cap C=1.80, quiet C=1.30). Coda has more LMR adjustments (failing, alpha-raised count, DoDeeper/DoShallower) but lacks near-root bonus and threat-aware reduction.

---

## 5. Move Ordering

Staged generation (TT -> good captures -> killers -> quiets -> bad captures).

### History Tables
- **Threat-indexed**: `[color][threatened][from][to]` -- doubles table by whether from-square is under attack
- Capture history also threat-indexed
- Counter-move history (ply 1) and follow-up (ply 2, ply 4)
- History amount: `depth + (eval <= alpha)` (eval-based depth bonus)
- **No counter-move table** (relies on continuation history)

Compare to Coda: Coda uses flat butterfly history (not threat-indexed), has counter-move table, continuation history at plies 1, 2, 4, 6.

### Correction History
- Pawn only, u16 hash. Much simpler than Coda's 5-table system.

---

## 6. Aggression System (Unique)

`aggression = 2 * non_pawn_count * clamp(root_eval, -200, 200) / 100`
- Positive when STM matches root STM, negative otherwise
- Effect: winning positions evaluated more optimistically (up to ~60cp in middlegame)
- Added to eval but NOT stored in TT
- Essentially a position-aware contempt that scales with material and advantage

---

## 7. Time Management

Triple-factor soft time scaling:
1. Move stability: 0.648 to 0.984 based on best-move consistency
2. Node factor: scaled by best-move node percentage (0.86x to 3.60x)
3. Eval factor: nearly constant (~1.67x) -- appears vestigial

Compare to Coda: Coda uses simpler time management with soft allocation, move overhead, and emergency mode.

---

## 8. TT

Single entry per index (no buckets), 12 bytes. No XOR verification. SSE prefetch.

Compare to Coda: Coda uses 5-slot buckets, XOR-verified, 64 bytes/bucket.

---

## 9. Parameter Comparison Table

| Feature | BlackMarlin | Coda |
|---------|-------------|------|
| RFP margin | 71*d - 62*imp_nothreats, d<=9 | improving?70*d:100*d, d<=7 |
| RFP return | (eval+beta)/2 | raw eval |
| Razoring | 306*d, d<=4 | 400+100*d, d<=2 |
| NMP reduction | 4+d*23/60+(eval-beta)/204 | 3+d/3+(eval-beta)/200 |
| NMP verification | depth >= 10 | depth >= 12 |
| NMP threat guard | Yes (depth<=7) | No |
| IIR trigger | depth>=4, shallow TT too | depth>=6, no TT move |
| Futility margin | 86*lmr_depth, d<=9 | 60+lmrDepth*60, d<=8 |
| LMR table | single, 0.50+ln*ln/2.05 | split cap/quiet C=1.80/1.30 |
| SE margin | ttScore - depth | ttScore - depth |
| SE double/triple ext | +2/+3 | No |
| SE multi-cut | Yes | Yes |
| SE neg ext | -2 | -1 |
| Cont-hist plies | 1, 2, 4 | 1, 2, 4, 6 |
| Correction history | pawn only | 5 tables |
| TT | single entry, 12B | 5-slot buckets, 64B |
| History tables | threat-indexed | flat butterfly |
| QS capture limit | 3 | Unlimited |

---

## 10. Ideas Worth Testing from BlackMarlin

### Still relevant for Coda:

1. **Threat-indexed history** -- `[color][from_threatened][from][to]`. Simple boolean index, 2x table size. The highest-consensus untested feature across all reviewed engines.

2. **NMP TT-adjusted eval** -- Use bounded TT score instead of raw eval for NMP. 3 lines, costs nothing.

3. **RFP score blending** -- Return `(eval+beta)/2` instead of raw eval on RFP cutoff. Note: GoChess-era test of "RFP score dampening" was -16.7 Elo, but the formula may have differed.

4. **NMP threat guard** -- Disable NMP at depth <= 7 when opponent has threats. Requires computing threats.

5. **Near-root LMR reduction** -- Reduce less when `ply <= (depth+ply)*2/5`. Novel, only in BlackMarlin.

6. **SE double/triple extension** -- +2/+3 for very singular moves. Coda only does +1.

7. **IIR on shallow TT** -- Also fire IIR when `tt_depth + 4 < depth`, not just missing TT.

8. **Low-depth singular extensions** -- Use raw eval at depth < 7 instead of searching. Much cheaper, extends SE to lower depths.

9. **Threat-aware LMR** -- Reduce less for moves that create new threats. Requires threat computation.

10. **QSearch 3-capture limit** -- Extremely aggressive but may gain NPS. Risky.

11. **Aggression/contempt system** -- Position-aware contempt scaling. Novel but could cause instability.

12. **RFP improving conditioned on no-threats** -- Only count as improving when opponent has no threats. Prevents false improving signal.

### IMPLEMENTED (remove from testing queue):

- **Multi-source correction history** -- Coda has 5 tables, far more than BlackMarlin's 1.
- **Continuation history ply 4** -- Coda has plies 1, 2, 4, 6.
- **SE multi-cut** -- Coda has this.
- **Aspiration fail-low contraction** -- Coda has this.
- **SCReLU / output buckets / Finny tables** -- Coda has all.
- **NMP verification** -- Coda has at depth >= 12.
- **Recapture extensions** -- Coda has this.
- **Counter-move heuristic** -- Coda has this (BlackMarlin doesn't).
- **Failing heuristic** -- Coda has this.
- **DoDeeper/DoShallower** -- Coda has this.
- **Alpha-raised count in LMR** -- Coda has this.
