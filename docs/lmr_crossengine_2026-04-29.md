# LMR Cross-Engine Survey, 2026-04-29

> **2026-04-30 update — single-feature LMR carve-out class CLOSED.**
> All five carve-outs derived from this doc were SPRT'd (some with
> retunes); all H0'd. See `experiments.md` 2026-04-30 entry for the
> full closure summary. Per-section status:
> - §3.5 anti-cascade — H0 #875 -1.4 Elo
> - §3.6 TT-bound informs LMR — H0 #876 -2.7 Elo
> - LMR threat-aware (Coda-original) — H0 #868 -0.8 Elo
> - LMR symmetric hindsight extension — H0 #866 +0.3 / H0 #881 bundle -3.7
> - LMR refutation bonus — H0 #867 -0.5 Elo
>
> The 2026-04-29 set-arithmetic finding (zero LMR-exclusive recovery
> cases) explained this in advance: single-feature LMR carve-outs cap
> at ~50% coverage and won't produce H1 at 10+0.1. **Path forward:
> multi-feature carve-out on a shared trigger** (NMP + LMR + RFP + FUT
> loosen simultaneously when major-threat creation fires). Distinct
> mechanism from the 5 single-feature attempts.

Per-feature ablation today showed disabling LMR alone recovers 53% of the
+3-+6 ply deficit candidates from our rivals-gauntlet loss subset — LMR is
the dominant choke point in moderate-stepped exploits. This is a survey of
how 9 strong engines gate LMR and adjust the reduction amount, and which
carve-outs Coda lacks.

Sources cited inline as `path:line`. All paths live under
`/home/adam/chess/engines/`.

---

## 1. Per-engine LMR matrix

### 1.1 Stockfish (`Stockfish/src/search.cpp`)

**Formula.** Reduction table indexed by depth alone:
`reductions[i] = int(2763 / 128.0 * std::log(i))` (search.cpp:622-623). The
runtime call is `reduction(improving, d, mn, delta) = reductions[d]*reductions[mn]
- delta*585/rootDelta + !improving*scale*206/512 + 1133`
(search.cpp:1756-1760). Stockfish stores `r` in 1024ths and divides at the
LMR call site (`r/1024`), allowing fine-grained adjustments.

**Skip conditions.**
- `depth >= 2 && moveCount > 1` (search.cpp:1251). No explicit
  in-check/promotion/endgame skip — captures and promos go through LMR with
  the same code path. Captures are reduced by the same table, but at PV nodes
  captures still get `+ PvNode` extension at the clamp.
- All else handled by the tens of additive `r += / r -=` knobs.

**Reduction-amount adjustments** (search.cpp:1062-1248, in order applied):
| Trigger | Effect (1024ths) |
|---|---:|
| `ttPv` | `r += 1013` (early), then `r -= 2819 + 973*PvNode + 905*ttBetter + (935 + 959*cutNode)*ttDeepEnough` (late) |
| Base offset | `r += 691` |
| Move count | `r -= 65 * moveCount` |
| Correction magnitude | `r -= |corrValue| / 25600` |
| Cut node | `r += 3611 + 985*!ttMove` |
| TT move is a capture | `r += 1054` |
| Child cutoff count | `r += 251 + 1124*(cutoff>2) + 1042*allNode` |
| First picked move (TT move) | `r -= 2239` |
| `statScore` (history+capt+contHist) | `r -= statScore * 428 / 4096` |
| Expected ALL node | `r += r * 273 / (256*depth + 260)` (multiplicative) |

**Re-search behavior** (search.cpp:1258-1281):
- LMR depth `d = clamp(newDepth - r/1024, 1, newDepth+2) + PvNode` —
  **PV nodes get a +1 extension**, not just less reduction. Negative `r`
  allowed up to `+2` plies extension.
- On fail-high: `doDeeper = (d < newDepth) && score > bestValue+48`,
  `doShallower = score < bestValue+9`. `newDepth += doDeeper - doShallower`.
  Re-search at adjusted `newDepth` if it differs from `d`.
- Post-LMR cont-hist update with bonus 1426.

**Hindsight** (search.cpp:773-777):
- `priorReduction >= 3 && !opponentWorsening` → `depth++`
- `priorReduction >= 2 && depth >= 2 && staticEval+(ss-1)->staticEval > 195` → `depth--`
  (matches Coda's existing pattern, but with **two thresholds**, not one.)

**Special.** SF's "step 18 full-depth when LMR is skipped" branch
(search.cpp:1283-1293) does its own reduction: `newDepth - (r > 4628) - (r > 5772 && newDepth > 2)`
— a 2-step ladder applied even when LMR didn't fire (e.g. first move at non-PV).

---

### 1.2 Reckless (`Reckless/src/search.rs`)

Strongest peer in the rivals pool that's similar to Coda's architecture.
Lots of unique knobs.

**Formula.** `r = 225 * (move_count.ilog2() * depth.ilog2())` for quiets,
`232 * ...` for captures (search.rs:817, 895). Stored in 1024ths like SF.

**Skip conditions.** `depth >= 2 && move_count >= 2` (search.rs:816). No
in-check/promo/endgame skip. **Two completely separate code paths**:
"LMR" (reduced search) and "FDS" (full-depth zero-window) — the FDS branch
also applies its own reductions at thresholds (search.rs:941: `reduced_depth = new_depth - (r >= 2864) - (r >= 5585)`).

**Reduction-amount adjustments** (search.rs:817-870):
| Trigger | Effect (1024ths) |
|---|---:|
| `move_count` | `r -= 68 * move_count` |
| `\|correctionValue\|` | `r -= 3297 * \|corr\| / 1024` (large weight!) |
| `alpha_raises` (count) | `r += 1306 * alpha_raises` |
| TT score `<= alpha` | `r += 546` |
| TT depth `< depth` | `r += 322` |
| Quiet | `r += 1806`, `r -= 166 * history / 1024` |
| Capture | `r += 1449`, `r -= 109 * history / 1024` |
| PV node | `r -= 424 + 433 * (beta-alpha) / root_delta` |
| `tt_pv` | `r -= 361`, `-= 636 * (tt_score>alpha)`, `-= 830 * (tt_depth>=depth)` |
| `!tt_pv && cut_node` | `r += 1818 + 2118 * tt_move.is_null()` |
| `!improving` | `r += clamp(430 - 263*improvement/128, ?, 1096)` |
| `in_check` | `r -= 1021` |
| Child cutoff_count > 2 | `r += 1515` |
| Singular margin | `r += clamp(512*(margin-160)/128, 0, 2048)` (BIG one) |
| Parent reduction much larger | `r += 129` (anti-cascade signal, search.rs:866) |
| Helper-thread bias | `r += helper_reduction_bias(td)` (Lazy SMP variation) |

**Re-search behavior** (search.rs:872-891):
- Reduced depth: `clamp(new_depth - r/1024, 1, new_depth + (move_count<=3) + 1) + 2*PvNode`
  — PV nodes get **+2 plies** extension, and first-3-move LMR can extend +1.
- On fail-high: `+(score > best+61) + (score > best+801) - (score < best+5+reduced_depth)`
  — **3-tier** shallower/deeper ladder.

**Hindsight** (search.rs:475-491):
- `priorReduction >= 2367 (~2.3 plies) && eval+prev_eval < 0` → `depth++`
- `!tt_pv && depth >= 2 && priorReduction > 0 && eval+prev_eval > 59` → `depth--`

**Special carve-outs Coda doesn't have:**
1. **Singular-margin LMR adjustment** (search.rs:861-864). When the previous
   singular search returned `singular_score`, the gap `tt_move_score - singular_score`
   directly increases reduction proportionally. This is a HEAD-OF-CLASS
   signal — moves close to a strong TT move get reduced harder.
2. **TT bound vs alpha gating** (search.rs:823-824). If TT score itself
   suggests this move can't beat alpha, reduce more.
3. **alpha_raises counter** (search.rs:821) — number of moves that have
   already raised alpha at this node feeds in linearly. (The more we've
   already improved, the less plausible later moves matter.)
4. **Anti-cascade reduction guard** (search.rs:866-868). When parent's
   reduction is much larger than this one's, bump by 129. Prevents reduction
   "valleys" in the tree.

---

### 1.3 Viridithas (`viridithas/src/search.rs`)

**Formula.** `lm_reduction_table[depth][played]` initialized as
`base + ld * lp / division` where `ld = ln(depth)`, `lp = ln(played)`,
`base=99`, `division=260` (search.rs:75-76, 2148). Stored in 1024ths.

**Skip conditions.** `depth > 2 && moves_made > 1 + ROOT` (search.rs:1451). No
explicit in-check/promo/endgame skip.

**Reduction-amount adjustments** (search.rs:1452-1471):
| Trigger | Effect (1024ths) |
|---|---:|
| Base offset | `r += lmr_base_offset` (226) |
| Non-PV | `r += 987` |
| `tt_pv` | `r -= 1289` |
| Cut node | `r += 1601` |
| `stat_score` | `r -= stat_score * 1024 / 17017` |
| Killer move | `r -= 775` (refutation reduces less) |
| `!improving` | `r += 613` |
| TT capture | `r += 999` |
| `in_check` | `r -= 1361` |
| Correction magnitude | `r -= \|corr\| * 448 / 16384` |

**Re-search behavior** (search.rs:1481-1516):
- `reduced_depth = clamp(new_depth - r, 0, new_depth + 1)` — allows
  +1 ply extension when r is negative.
- On fail-high (`r > 1`): `doDeeper = score > best + base + depth_margin*r`,
  `doShallower = score < best + new_depth`. Re-search at adjusted depth.
- **Distinct quirk** (search.rs:1509-1511): if score barely beat alpha
  (`score < best+16`), `new_depth -= 1` BEFORE the full-window re-search at
  PV nodes — one step shallower than the LMR re-search. Coda doesn't do this.
- **Cont-hist updates on LMR pass/fail** (search.rs:1506-1508):
  `update_cont_hist_single` after the re-search records the verdict for
  later move ordering.

**Hindsight** (search.rs:1041-1053):
- `priorReduction >= hindsight_ext_depth && static_eval + prev_static < 0` → `depth++`
- `depth >= 2 && priorReduction >= hindsight_red_depth && static_eval + prev > hindsight_red_eval` → `depth--`
  (gates restricted to `!ROOT && !PV && !in_check && !excluded`)

**Special.** Killer-move exemption (`m == killer` reduces less by ~775)
is a Viridithas/Alexandria-class signal Coda dropped when killers were
unified into the main 4D history.

---

### 1.4 Obsidian (`Obsidian/src/search.cpp`)

**Formula.** Standard `lmrTable[i][m] = dBase + log(i)*log(m)/dDiv`
(search.cpp:128). Plain integer, no 1024ths.

**Skip conditions** (search.cpp:1069). `depth >= 2 && seenMoves > 1 + 2*IsRoot`.
No in-check/promo/endgame skip — captures go through LMR.

**Reduction-amount adjustments** (search.cpp:1071-1088, in order):
| Trigger | Effect (plies) |
|---|---:|
| History | `R -= history / (isQuiet ? LmrQuietHistoryDiv : LmrCapHistoryDiv)` |
| Complexity | `R -= ss->complexity / LmrComplexityDiv` |
| Move gives check | `R -= 1` |
| `ttDepth >= depth` | `R -= 1` |
| `ttPV + IsPV` | `R -= ttPV + IsPV` (so up to -2 at PV+ttPV) |
| TT move noisy | `R += 1` |
| `!improving` | `R += 1` |
| Cut node | `R += 2 - ttPV` |

**Re-search** (search.cpp:1095-1106): `doDeeper = score > best + ZwsDeeperMargin + 2*newDepth`
(scales with depth!), `doShallower = score < best + ZwsShallowerMargin`.
Cont-hist updated post-LMR with `score-aware bonus/malus`.

**Hindsight.** None.

**Special.** Obsidian was the first to make `ss->complexity` (cont-hist
volatility) feed LMR. Coda already imports this pattern via
`LMR_COMPLEXITY_DIV`. Obsidian also has `IsPV` at the call site, separate
from `ttPV` — they double-deduct PV reductions.

---

### 1.5 Berserk (`berserk-13/src/search.c`)

**Formula.** `LMR[d][m] = log(d)*log(m) / 2.0385 + 0.2429` (search.c:52).
Plain integer.

**Skip conditions** (search.c:705). `depth > 1 && legalMoves > 1 && !(isPV && IsCap(move))`
— **Coda-relevant**: PV + capture skips LMR entirely. (Horsie has the same
gate.)

**Reduction-amount adjustments** (search.c:617-731):
| Trigger | Effect (plies) |
|---|---:|
| History | `R -= history / 8192` (early, before move) |
| TT move noisy | `R += 1` (early) |
| `!ttPv` | `R += 2` |
| `!improving` | `R += 1` |
| `killerOrCounter` | `R -= 2` (refutation big bump) |
| `board->checkers` (gives check) | `R -= 1` |
| Cut node | `R += 1 + !IsCap(move)` (more for quiets) |
| `ttDepth >= depth` | `R -= 1` |

**Re-search** (search.c:738-748). `doDeeper = score > best+69`, `doShallower = score < best+newDepth`.
Cont-hist updated with bonus.

**Hindsight.** None visible.

**Special.**
1. **Killer/counter exemption: -2** (search.c:715). Big magnitude — Berserk
   gives refutation moves a 2-ply boost. Coda has none.
2. **`isPV && IsCap` skip** (search.c:705). Captures at PV are exempt
   altogether, not just adjusted.

---

### 1.6 PlentyChess (`PlentyChess/src/search.cpp`)

**Formula.** Three reduction tables depending on move type, each with
its own base/divisor (search.cpp:202-204):
- 0: quiet `1.12 + log*log / 2.95`
- 1: noisy capture `-0.23 + log*log / 2.98`
- 2: "important capture" (tt-pv & cut-node-affecting) `-0.13 + log*log / 3.17`

Stored in 100ths.

**Skip conditions.** `moveCount > lmrMcBase + lmrMcPv*rootNode - hasTTMove && depth >= 322`
(search.cpp:1095) — depth is in centiplies, so `>= 3.22` plies. No
in-check/promo/endgame skip.

**Reduction-amount adjustments** (search.cpp:1096-1123, big and structured):
| Trigger | Effect |
|---|---:|
| Base offset | `+ lmrReductionOffset(important)` |
| Correction magnitude | `- |corr| / lmrCorrectionDivisor` |
| Gives check | `- lmrCheck(important)` |
| Cut node | `+ lmrCutnode` (257) |
| `ttPv` | `- lmrTtPv(important)` |
| `ttPv && tt_value <= alpha` | `+ lmrTtpvFaillow(...)` (anti-stale-tt) |
| Capture history | `- moveHistory * |moveHistory| / lmrCaptureHistoryDivisor` |
| Bad capture stage | `+= lmrImportantBadCaptureOffset * (stage == BAD_CAP)` |
| Quiet history | `- 100 * moveHistory / lmrQuietHistoryDivisor` |
| Quiet PV node | `- lmrQuietPvNodeOffset * pvNode` |
| Quiet improving | `- lmrQuietImproving * improving` |

**Re-search** (search.cpp:1136-1147):
- `reducedDepth = clamp(newDepth - reduction, 100, newDepth + lmrPotentialExtension) + lmrPvNodeExtension * pvNode`
  — PV nodes get a tunable extension boost; reduction can extend up to
  `lmrPotentialExtension` plies.
- After LMR, **TT-bound short-circuit** (search.cpp:1140): if
  `ttValue < alpha && ttDepth - lmrResearchSkipDepthOffset >= newDepth && ttFlag == UPPER_BOUND`,
  skip the re-search entirely. Coda doesn't have this.
- `lmrPassBonus` cont-hist update with verdict-aware bonus.

**Hindsight** (search.cpp:783-790):
- "Post-LMR depth adjustments": uses parent's `inLMR` + reduction +
  staticEval comparison to deduct depth. Variant of SF hindsight that
  fires only when **parent was actually in an LMR search**.

**Special carve-outs Coda doesn't have:**
1. **Three separate base tables** (quiet/noisy/important-capture). The
   "important capture" tier (`ttPv && capture && !cutNode`) is treated
   distinctly. PlentyChess literally adds `+5 plies` to newDepth for
   important captures (search.cpp:1090-1091).
2. **TT-fail-low aware reduction bump** (`ttPv && tt_value <= alpha`).
   Stale TT entries that thought this position was bad get reduced more
   so the search shifts attention.
3. **TT-upper-bound short-circuit on LMR re-search** (search.cpp:1140).
   Saves nodes when TT already proved the upper bound at sufficient depth.
4. **`moveHistory * |moveHistory|`** quadratic capture-history adjustment
   (search.cpp:1112). Quadratic in history sign. Aggressive when both
   tails matter.
5. **Cut-node move count adjustment in `newDepth`** (search.cpp:1087-1088):
   `if (cutNode && depth >= 600 && move != ttMove) newDepth -= 5;`
   — non-TT moves at deep cut nodes get a flat 5-cp depth penalty
   BEFORE LMR runs. (Different mechanism from cut-node LMR bump.)

---

### 1.7 Caissa (`Caissa/src/backend/Search.cpp`)

**Formula.** `lm_reduction_table[depth][movesPlayed]` built with separate
quiet and capture scales+biases (search.cpp:202-225). Stored in `LmrScale`
units; final divide by `LmrScale` at end.

**Skip conditions** (search.cpp:1872-1874):
`depth >= 1 && moveIndex > 1 && (!isPvNode || move.IsQuiet())` —
**captures at PV nodes are skipped entirely**, like Berserk and Horsie.

**Adjustments — quiet** (search.cpp:1876-1898):
| Trigger | Effect |
|---|---:|
| Non-PV | `+ LmrQuietNonPv` |
| TT capture | `+ LmrQuietTTCapture` |
| Killer/refutation | `- LmrQuietRefutation` |
| Stat score | `-= (statScore + 6877) / 240` (offset!) |
| Cut node | `+ LmrQuietCutNode` |
| `!improving` | `+ LmrQuietImproving` |
| Gives check | `- LmrQuietInCheck` |

**Adjustments — capture** (search.cpp:1900-1916):
| Trigger | Effect |
|---|---:|
| Winning capture (above threshold) | `- LmrCaptureWinning` |
| Bad capture (below threshold) | `+ LmrCaptureBad` |
| Cut node | `+ LmrCaptureCutNode` |
| `!improving` | `+ LmrCaptureImproving` |
| Gives check | `- LmrCaptureInCheck` |

**Common** (search.cpp:1919-1924):
- PV: `r -= LmrScale * depth / (1 + ply + depth)` — **ply-dependent**
  reduction. Lower-ply moves reduce less. Coda has nothing similar.
- TT high depth: `r -= LmrTTHighDepth`

**Re-search** (search.cpp:1953-1962). `doDeeper = score > best+LmrDeeperTreshold`
**+ depth-explosion guard** `(node->ply < 2*rootDepth)`. Coda lacks this.

**Hindsight.** None visible.

**Special.**
1. **Ply-tapered LMR**: `r -= scale * depth / (1 + ply + depth)`. Reduces
   less at low plies (closer to root), more at high plies. Strong fairness
   mechanism for early-tree decisions.
2. **Search-explosion guard on doDeeper**: only allows the +1 if `ply < 2*rootDepth`.
3. **Distinct winning-capture / bad-capture buckets**: explicit `moveScore`
   thresholds (`WinningCaptureValue`, `GoodCaptureValue`).

---

### 1.8 Halogen (`Halogen/src/search/search.cpp`)

**Formula.** Lookup `LMR_reduction[depth][seen_moves]`. Stored in
fixed-point with `LMR_SCALE`. Notable that Halogen passes `r` to the
search-helper, which separates **gating** from **adjustment** cleanly
(`late_move_reduction()` returns the int; `search_move()` consumes it).

**Skip conditions** (search.cpp:679). `depth >= 2 && seen_moves > 1`. No
in-check/promo/endgame skip.

**Reduction-amount adjustments** (search.cpp:588-615):
| Trigger | Effect |
|---|---:|
| PV | `- lmr_pv` |
| Cut node | `+ lmr_cut` |
| Improving | `- lmr_improving` |
| Loud (capture/promo) | `- lmr_loud` |
| History | `- (history * lmr_h).rescale<LMR_SCALE>()` |
| Base offset | `+ lmr_offset` |

Final clamp: `max(-pv_node, r)` — PV nodes can extend by 1 ply via negative r.

**Re-search** (search.cpp:679-695). One-step ladder: `bool reduce = score < best+lmr_shallower`,
re-search at `new_depth - reduce`. **Simpler than SF/Reckless.** No "do deeper".

**Hindsight** (search.cpp:923-927):
- `priorReduction >= lmr_hindsight_ext_depth && eval+prev_eval < lmr_hindsight_ext_margin` → `depth++`
- No corresponding decrement.

**Special.** Halogen is the cleanest minimal LMR — pretty much consensus,
nothing distinctive. Notable for being **simpler than Coda's** in the
adjustment cluster.

---

### 1.9 Tarnished (`Tarnished/src/search.cpp`)

Strongest member of the rivals pool (+56 ±24 vs Coda).

**Formula.** Separate quiet/noisy tables built from
`LMR_BASE + log(depth)*log(moves) / LMR_DIVISOR` (search.cpp:260-268).
Stored in 1024ths.

**Skip conditions** (search.cpp:663). `depth >= 3 && moveCount > 2 + root`.
Higher floor than most engines. (Coda's is `depth >= 3 && moves >= 3`.)

**Reduction-amount adjustments** — the distinctive thing here is the
**factorized table** (search.cpp:202-203, 666-685):

```cpp
uint32_t feature = isQuiet;
feature |= !isPV << 1;
feature |= improving << 2;
feature |= cutnode << 3;
feature |= ttPV << 4;
feature |= ttHit << 5;
feature |= (failHighs > 2) << 6;
feature |= (corrplexity > LMR_CORR_MARGIN()) << 7;
reduction += factoredLmrTable[feature];
reduction -= 1024 * historyScore / LMR_HIST_DIVISOR();
reduction /= 1024;
```

This is a **256-entry SPSA-tunable lookup table** indexed by an 8-bit
feature vector. Up to 3-way interactions between {isQuiet, isPV, improving,
cutnode, ttPV, ttHit, child_cutoffs, complexity}. (Comment explains the
"AGE idea": one-way + two-way + three-way table convolved into one.)

**Re-search** (search.cpp:697-705). doDeeper/doShallower like everyone else.

**Hindsight** (search.cpp:485): `priorReduction >= 4 && eval+prev <= 0` → `depth++`.
Single-sided.

**Special.** **Factorized LMR table** is unique among the surveyed engines
— exposes feature interactions for SPSA without hand-rolling
`if (a && b)` clauses. Tarnished gets +56 Elo over Coda; how much the
factor table contributes vs. their other choices is unclear, but it's
worth noting they're best-in-pool and use this.

---

### 1.10 Horsie (`Horsie/src/search.cpp`)

Tied for top of rivals pool (+58 ±23).

**Formula.** `LogarithmicReductionTable[depth][legalMoves]`, scaled by 128.

**Skip conditions** (search.cpp:627-629).
`depth >= 2 && legalMoves >= 2 && !(isPV && isCapture)`.
(Same gate as Berserk/Caissa: PV captures skip LMR entirely.)

**Reduction-amount adjustments** (search.cpp:631-643):
| Trigger | Effect (128ths) |
|---|---:|
| Cont-hist (ply-1, ply-2, ply-4!) + main | weighted sum into `histScore` |
| `!improving` | `+ LMRNotImpCoeff` |
| Cut node | `+ LMRCutNodeCoeff` |
| `ttPV` | `- LMRTTPVCoeff` |
| Killer move | `- LMRKillerCoeff` (refutation -1) |
| History | `- histScore / (capture ? LMRCaptureDiv : LMRQuietDiv)` |

Final divide by 128.

**Re-search** (search.cpp:645-664). `clamp(newDepth-R, 0, newDepth) + isPV` —
PV nodes get +1 extension. doDeeper with **4*newDepth** scaling
(`score > best + DeeperMargin + 4*newDepth`). Cont-hist updated with
verdict bonus.

**Hindsight.** None.

**Special.**
1. **Cont-hist at ply-1, ply-2, ply-4** for LMR adjustment (search.cpp:631-634).
   Ply-4 is unusual — most engines only use 1+2 (and Coda uses 1+2 with
   ply-2 weighted half). Stockfish uses 1+2+4+6 across the search but only
   1+2 in LMR adjustment.
2. **PV-capture skip** (gates same as Berserk/Caissa).
3. **`4*newDepth` doDeeper margin** is more aggressive than SF's flat `+48`.

---

### 1.11 Coda (`/home/adam/code/coda/src/search.rs`, current trunk)

For reference. Already detailed in the task description. Summary:

**Formula.** `r = ln(d) * ln(m) / c` with `c = LMR_C_QUIET/100` (1.32) for
quiets, `LMR_C_CAP/100` (1.03) for captures. Separate tables. Initialized
once at startup.

**Skip conditions.** `in_check`, `is_promo`, `is_endgame_skip` (popcount
threshold), `!FEAT_LMR`. Captures use a different table; quiet path needs
`depth >= 3 && moves >= 3`. Capture path needs `move_count > 1 && mv != tt_move && beta-alpha == 1` (zero-window only).

**Adjustments.** PV (-1), non-PV with mc>1 (+1), improving (-1), TT-noisy
(+1), opponent low-piece-count (+1), from-square attacked (-1), gives check
(-1), tt_pv (-1), main+cont(1)+cont(2)/2 + pawn_hist division, complexity
divisor, threat-count divisor, king-zone-pressure divisor.

**Capture LMR sub-path.** Capture-history bracketed (>2000 / <-2000),
gives-check (-1).

**Re-search.** doDeeper / doShallower with two thresholds against
`best_score + 60 + 10*reduction` and `best_score + 20`. Then PVS full-window
re-search if score in (alpha, beta).

**Hindsight.** Single-sided: `priorReduction >= 2 && eval_sum > thresh` → `depth -= 1`.
Coda is **missing the symmetric "+1 plies on parent over-reduced" branch**
that SF/Halogen/Reckless/Viridithas/Tarnished all have.

---

## 2. Comparison table

Coda baseline = current trunk. ✓ = present. ✗ = absent. ~ = partial/variant.

### 2.1 Skip conditions

| Skip | SF | Reckless | Virid | Obs | Berserk | Plenty | Caissa | Halogen | Tarnished | Horsie | **Coda** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `depth >= 2/3 && moves >= 2/3` | ✓ (≥2,>1) | ✓ (≥2,≥2) | ✓ (>2,>1) | ✓ (≥2,>1) | ✓ (>1,>1) | ✓ (≥3.2,>n) | ✓ (≥1,>1) | ✓ (≥2,>1) | ✓ (≥3,>2) | ✓ (≥2,≥2) | ✓ (≥3,≥3) |
| In-check skip (this side) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Promotion skip | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Endgame piece-count skip | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **PV + capture** skip entirely | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |

### 2.2 Reduction-amount adjustments

| Adjustment | SF | Reckless | Virid | Obs | Berserk | Plenty | Caissa | Halogen | Tarnished | Horsie | **Coda** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PV node | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (factor) | ✗ (call site) | ✓ |
| Cut node | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ (only via mc>1 proxy) |
| Improving | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TT move noisy | ✓ | (n/a) | ✓ | ✓ | ✓ | (n/a) | ✓ | (n/a) | (n/a) | (n/a) | ✓ |
| Gives check | (via cont) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | (via loud) | ✓ (factor) | ✗ | ✓ |
| `tt_pv` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Main history | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Cont-hist | ✓ (in stat) | ✓ | ✓ (in stat) | ~ | ~ | ~ | ✓ (in stat) | ~ | ✓ (factor) | ✓ (1+2+**4**) | ✓ (1+2) |
| Pawn history | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Capture history | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (×\|h\|) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Killer/refutation exemption | (TT-move) | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | (factor) | ✓ | ✗ |
| Correction-magnitude (complexity) | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ (factor) | ✗ | ✓ |
| TT-bound vs alpha | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ (faillow) | ✗ | ✗ | ✓ (ttHit) | ✗ | ✗ |
| TT depth ≥ depth | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ (`LmrTTHighDepth`) | ✗ | ✗ | ✗ | ✗ |
| Singular margin | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `alpha_raises` count | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Anti-cascade vs parent r | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Child cutoffCnt | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ (factor) | ✗ | ✗ |
| Move count | ✓ (`-65*mc`) | ✓ (`-68*mc`) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Ply-tapered | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Threat-count | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| King-zone pressure | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| From-square attacked | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Opponent low non-pawn material | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Important-capture tier (ttPV+!cutNode) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| All-node multiplicative scale | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Factorized lookup | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |

### 2.3 Re-search behavior

| Behavior | SF | Reckless | Virid | Obs | Berserk | Plenty | Caissa | Halogen | Tarnished | Horsie | **Coda** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| doDeeper | ✓ | ✓ (3-tier) | ✓ | ✓ | ✓ | ✓ | ✓ (+ply guard) | ✗ | ✓ | ✓ | ✓ |
| doShallower | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| PV gets +1/+2 plies via clamp | ✓ (+PvNode/+2) | ✓ (+2) | ✓ (+1) | ✓ (+1) | ✗ | ✓ (+ext) | ✗ | ✓ (-1 floor) | ✗ | ✓ (+1) | ✗ |
| TT-bound short-circuit on re-search | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Post-LMR cont-hist verdict update | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Skip-LMR FDS reduction ladder | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

### 2.4 Hindsight

| Hindsight | SF | Reckless | Virid | Obs | Berserk | Plenty | Caissa | Halogen | Tarnished | Horsie | **Coda** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Symmetric (extend AND shrink) | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| One-sided (shrink only) | (above) | (above) | (above) | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ |
| One-sided (extend only) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |

---

## 3. Carve-out candidates ranked by promise

Each entry: idea / engines using it / mechanism / effort / Coda fit / Elo prior.

### 3.1 (HIGH) Symmetric hindsight: extend depth when parent over-reduced and we just discovered a tactical position

- **Engines using:** SF (search.cpp:773-777), Halogen, Reckless, Viridithas,
  Tarnished. Five top engines.
- **Mechanism.** Coda already has the "shrink" half (`prior_red >= 2 && eval_sum > thresh → depth -= 1`).
  Missing the symmetric **extend** branch: when parent was reduced AND
  current node looks tactical (`eval + prev_eval < 0` or
  `!opponentWorsening`), give back a ply. Without it, LMR over-reduction
  is a one-way street — once we lose depth, we can't recover even when
  the position confirms the parent was wrong.
- **Effort.** Small. Add 4 lines next to the existing hindsight block at
  search.rs:2268-2280.
- **Coda fit.** Direct. We already track `info.reductions[ply_u-1]` and
  `info.static_evals[ply_u-1]`. The branch slots in cleanly.
- **Expected Elo.** **High** (3-8 Elo prior). Symmetric hindsight is one
  of the most-cited LMR additions of 2024-2025, has independent invention
  in Halogen/Stockfish, and addresses exactly the moderate-step
  exploit class — opponents constructing tactical pressure across plies
  Coda's LMR was suppressing.

### 3.2 (HIGH) Killer/refutation move LMR exemption (Berserk, Viridithas, Caissa, Horsie)

- **Engines using:** Berserk (`-2`!), Viridithas (`-775` ≈ -0.76 plies),
  Caissa (`- LmrQuietRefutation`), Horsie (`- LMRKillerCoeff`).
- **Mechanism.** Killer moves are by definition moves that caused beta
  cutoffs at sibling nodes — they're disproportionately likely to refute
  this node too. Reducing them at LMR cost equally to random late quiets
  loses the signal. Coda removed killers/counters as separate ordering
  layers when unifying into 4D history — but the **LMR adjustment is
  separable**, even if ordering uses the unified history.
- **Effort.** Medium. Coda doesn't track per-node killer separately
  anymore. Two paths:
  - (a) **Lite** — re-introduce a single 1-killer-per-ply `[Move; MAX_PLY]`
    array filled at fail-high, consult in LMR adjustment, no other use.
    Maybe 30 LOC.
  - (b) **History-threshold proxy** — treat moves with main-history above
    threshold as "refutation-class" and exempt. Already on the path; just
    pick a high cutoff (e.g. main_hist > 6000) and `reduction -= 1`.
    Smaller code change.
- **Coda fit.** Lite is clean; threshold variant rides existing
  4D history infra.
- **Expected Elo.** **Medium-High** (2-4 Elo prior). Killer exemption is
  one of the older LMR ideas (going back to ~Stockfish 8 era) and survived
  Stockfish's evolution into the unified `statScore`. Berserk's `-2`
  magnitude is large; even at half that, a +2-3 Elo prior is realistic.

### 3.3 (HIGH) PV node depth extension via clamp (allow negative reduction → +1/+2 plies)

- **Engines using:** SF (`+ PvNode`, allows up to +2), Reckless (`+2*PvNode`,
  +1 for first 3 moves), Viridithas (clamp to `new_depth+1`), Obsidian
  (`+1`), Plenty (`+ lmrPvNodeExtension`), Halogen (negative-r floor of `-pv_node`).
- **Mechanism.** Coda's clamp `if reduction > new_depth - 1 { reduction = new_depth - 1 }`
  caps reduction but **never extends**. When the reduction calculation
  comes out negative (history is great, gives_check, tt_pv, complexity,
  etc.), we just set it to 0. Top engines treat that as a +1 (or +2)
  extension. This biases the search toward depth at promising lines.
- **Effort.** Small. ~10 LOC change at search.rs:3004-3010 — change the
  floor and adjust the LMR depth computation:
  ```rust
  let pv_bonus = if beta - alpha > 1 { 1 } else { 0 };
  let max_ext = if move_count <= 3 { 1 } else { 0 } + pv_bonus;  // Reckless-shaped
  let lmr_depth = (new_depth - reduction).clamp(1, new_depth + max_ext);
  ```
- **Coda fit.** Clean. Tunable via a new `LMR_PV_EXT` parameter.
- **Expected Elo.** **Medium-High** (2-5 Elo prior). Universal. Coda's
  LMR is currently strictly downward-only — adding the upward direction
  costs almost nothing and addresses cliff-miss class directly.

### 3.4 (HIGH) Skip-LMR-altogether for PV-captures (Berserk, Caissa, Horsie)

- **Engines using:** Berserk (`!(isPV && IsCap)`), Caissa
  (`!isPvNode || move.IsQuiet()`), Horsie (same).
- **Mechanism.** Captures at PV nodes are usually critical (recaptures,
  exchanges, decisive captures). Reducing them is the highest-EV mistake
  in the LMR suite. Coda currently DOES reduce captures at PV via the
  capture branch, just at a different table — but **only at non-PV
  zero-window** (Coda's gate is `beta - alpha == 1`). Wait, let me re-check
  this... [from search.rs:3015-3017]: `if !in_check && is_cap && !is_promo
  && move_count > 1 && mv != tt_move && !is_endgame_skip && FEAT_LMR { if
  beta - alpha == 1 { reduction = lmr_cap_reduction(...); ... } }`. So
  Coda already gates capture LMR to zero-window (PV captures are exempt).
  This carve-out is **already present**. Crossing off.
- **Status.** ✓ Already in place. Move comparison table to mark `~`
  rather than ✗ for Coda on this row.

### 3.5 (CLOSED — H0 #875 -1.4 Elo / 25K games)

- **Engines using:** Reckless (search.rs:866-868). `if !PV && parent_red > my_red + 485 { my_red += 129; }`
- **Mechanism.** When the parent was reduced much more than this node's
  computed reduction, the parent's "this looks like an unimportant
  position" signal should propagate. Without this, parent and child make
  uncorrelated reduction decisions; with it, a "valley" (parent reduced
  hard, child reduced barely) bumps the child up so the search spends
  consistent effort along ZW lines.
- **Effort.** Small. Read parent reduction from `info.reductions[ply_u-1]`,
  add a `reduction += K` if it exceeds current by margin. ~5 LOC.
- **Coda fit.** Direct. Can use existing reduction stack.
- **Expected Elo.** **Low-Medium** (1-3 Elo prior). Reckless-only mechanism;
  no cross-engine consensus. Tune-on-branch likely required.
- **2026-04-30 outcome:** SPRT #875 H0 **-1.4 ±1.9 / 24878g** ✗.
  Reckless-only mechanism didn't transfer. Closed.

### 3.6 (CLOSED — H0 #876 -2.7 Elo / 15K games)

- **Engines using:** Reckless (`tt_score <= alpha → r += 546`,
  `tt_depth < depth → r += 322`), Plenty (`ttPv && tt_value <= alpha → r +=`),
  Tarnished (factor table includes `ttHit`).
- **Mechanism.** TT entries already encode "this position scored X with
  bound Y at depth Z." If the TT thinks this branch is below alpha, it's
  likely to be below alpha — reduce more. Coda doesn't currently use
  TT-bound or TT-score in LMR adjustment beyond `tt_pv`.
- **Effort.** Small (~10 LOC). Already have TT data at the call site.
- **Coda fit.** Clean.
- **Expected Elo.** **Medium** (1-3 Elo prior). Multi-engine consensus.
- **2026-04-30 outcome:** SPRT #876 H0 **-2.7 ±2.4 / 15014g** ✗.
  Multi-engine consensus didn't transfer — likely the existing
  `tt_pv` flag captures most of the signal and adding tt_bound ×
  tt_depth overlaps without adding new information. Closed.

### 3.7 (MEDIUM) Important-capture tier (PlentyChess)

- **Engines using:** Plenty alone. Three-tier table: quiet, normal capture,
  "important capture" (`ttPv && capture && !cutNode`).
- **Mechanism.** Captures that are also TT-PV in a zero-window expected-cut
  context (where the TT still thinks PV) are special — they're the closest
  thing to a "this is a critical capture" signal. Important captures get
  +5 cp `newDepth` AND a separate, less-aggressive reduction table.
- **Effort.** Medium. New tunable + alternate base-table init in
  `init_lmr()`. ~25 LOC. Possibly need to eat tunable budget for several
  knobs (`important_cap_base`, `important_cap_div`, `important_cap_offset`).
- **Coda fit.** OK. We have all the inputs. Just adds tunable surface.
- **Expected Elo.** **Low-Medium** (1-3 Elo prior). PlentyChess-only
  pattern; not yet validated cross-engine.

### 3.8 (MEDIUM) Move-count linear adjustment (`r -= K * move_count`, SF + Reckless)

- **Engines using:** SF (`-65 * moveCount`), Reckless (`-68 * move_count`).
- **Mechanism.** The `ln*ln` table is sub-linear in move_count — at
  moderate move counts (5-10) the marginal extra reduction per move is
  small. SF/Reckless found it pays to add a **linear** bonus reduction
  per move on top of the table. Coda's `LMR_C_QUIET=132` was tuned without
  this term, so the absolute base is calibrated for "pure ln*ln."
  Adding the linear term shifts the balance toward more pruning at high
  move counts.
- **Effort.** Small. One line. New tunable.
- **Coda fit.** Trivial. Will require LMR full retune.
- **Expected Elo.** **Low-Medium** (1-2 Elo prior). It's a tuning shape,
  not a structural change. Likely subsumed by retune.

### 3.9 (MEDIUM) Singular-margin LMR adjustment (Reckless)

- **Engines using:** Reckless alone (search.rs:861-864). When the singular
  search returned `singular_score`, the gap `tt_move_score - singular_score`
  → `r += clamp(512*(margin-160)/128, 0, 2048)`.
- **Mechanism.** A wide singular margin means the TT move is a clear
  outlier; everything else at this node is plausibly redundant. Reduce
  non-TT moves harder.
- **Effort.** Medium. Coda has SE but doesn't track `singular_score`
  past the SE block. Need to plumb it through ~3 functions. ~30 LOC.
- **Coda fit.** Possible. Some existing infrastructure.
- **Expected Elo.** **Medium** (1-3 Elo prior). Reckless-only but the
  mechanism is principled and parameterizes a real signal.

### 3.10 (MEDIUM) Ply-tapered LMR (Caissa)

- **Engines using:** Caissa alone (search.cpp:1921):
  `r -= LmrScale * depth / (1 + ply + depth)` at PV nodes.
- **Mechanism.** Reduce LESS at low ply. Decisions near root deserve more
  search; decisions deep in subtrees get the full LMR treatment. This is
  a fairness signal between root-distance and remaining depth.
- **Effort.** Small. ~5 LOC.
- **Coda fit.** Direct.
- **Expected Elo.** **Low-Medium** (1-2 Elo prior). Caissa-only, not yet
  cross-engine verified. But low cost.

### 3.11 (LOW-MEDIUM) Cont-hist ply-4 (Horsie)

- **Engines using:** Horsie alone uses ply-4. Stockfish uses ply-1+ply-2
  in LMR, ply-1+ply-2+ply-4+ply-6 elsewhere.
- **Mechanism.** A "longer-range" tactical motif signal. Two plies ago is
  still in the same maneuver; four plies ago is the previous "phase" of
  the move. Coda already has 1+2/2 weighting; adding ply-4 (probably
  scaled at /4) might catch motif-class info.
- **Effort.** Small. We already have `info.cont_hist` at multi-ply depth.
- **Coda fit.** Direct.
- **Expected Elo.** **Low** (1-2 Elo prior). Cont-hist depth varies
  across engines; SF's choice of 1+2 in LMR specifically (vs. 1+2+4+6 in
  movepicker) is a signal. Worth probing but not a guaranteed win.

### 3.12 (LOW-MEDIUM) Child cutoff count (SF, Reckless, Tarnished)

- **Engines using:** SF (`(ss+1)->cutoffCnt > 1`), Reckless
  (`stack[ply+1].cutoff_count > 2 → r += 1515`), Tarnished (factor: `failHighs > 2`).
- **Mechanism.** If the child node has been failing high a lot, the
  subtree is in a "many-cutoffs" regime — moves there will likely not
  raise alpha here. Reduce more.
- **Effort.** Medium. Need to instrument cutoff tracking on the search
  stack. Coda doesn't currently maintain a per-ply cutoff counter.
  ~30 LOC.
- **Coda fit.** OK. Search stack indexed by ply already.
- **Expected Elo.** **Medium** (1-2 Elo prior). Cross-engine consensus.

### 3.13 (LOW) Multiplicative all-node scale-up (SF only)

- **Engines using:** Stockfish alone (search.cpp:1247-1248):
  `if allNode { r += r * 273 / (256*depth + 260); }`
- **Mechanism.** Expected ALL-nodes (not cut, not PV) — reduce
  multiplicatively, not additively. Aggressively shrinks the dead-end
  search.
- **Effort.** Small.
- **Coda fit.** OK. We don't currently classify all-nodes explicitly
  (cut_node tag is binary). Need an `all_node` tag passed in (`!cut_node && !pv`).
- **Expected Elo.** **Low** (0-2 Elo prior). SF-only and recently added.

### 3.14 (LOW) `alpha_raises` running counter (Reckless only)

- **Engines using:** Reckless alone. `alpha_raises` tracks how many moves
  at this node have already pushed alpha; later moves get more reduction
  proportionally.
- **Mechanism.** Once alpha has been raised multiple times at a node, the
  remaining moves are improbable improvers.
- **Effort.** Small.
- **Coda fit.** OK.
- **Expected Elo.** **Low** (1 Elo prior). Reckless-only; small magnitude
  in their formula.

### 3.15 (RESEARCH) Factorized LMR table (Tarnished)

- **Engines using:** Tarnished alone. 8-bit feature vector → 256-entry
  SPSA-tuned table.
- **Mechanism.** Lets SPSA discover **interaction terms** between features
  (e.g. "improving AND cutnode AND !ttPV") rather than the engine designer
  hand-coding only the pairwise terms they noticed. Tarnished is +56 Elo
  over Coda; we don't know how much is the factor table specifically.
- **Effort.** **Large** (~150 LOC for the convolution machinery + table
  init + SPSA wiring + ~30 new tunables). Not a quick win.
- **Coda fit.** Possible but disruptive. Adds significant SPSA tunable
  budget.
- **Expected Elo.** **Unknown high-variance** (could be 0, could be 5+).
  Worth tracking but not the first carve-out to try.

---

## 4. Top 3 priority candidates

Combined criterion: high Elo prior × low effort × Coda fit.

### 4.1 Symmetric hindsight extension (§3.1)

Most leveraged. Coda already has half of the pattern; adding the symmetric
branch is 4 lines. Five top engines use it; SF and Halogen independently
re-invented it. Direct response to "moderate-stepped exploits" — when the
opponent's threat develops across plies our LMR was suppressing, the
symmetric extension restores depth on the recovery node.

```rust
// search.rs:2268-2280, replace existing block:
let prior_reduction = if ply_u >= 1 { info.reductions[ply_u - 1] } else { 0 };
if !in_check && ply >= 1 && depth >= tp(&HINDSIGHT_MIN_DEPTH) && ply_u >= 1
    && info.static_evals[ply_u - 1] > -(MATE_SCORE - 100)
    && static_eval > -INFINITY
    && FEAT_HINDSIGHT.load(Ordering::Relaxed)
{
    let eval_sum = info.static_evals[ply_u - 1] + static_eval;

    // Existing shrink branch
    if prior_reduction >= 2 && eval_sum > tp(&HINDSIGHT_THRESH) {
        depth -= 1;
    }

    // NEW: extend branch — parent was over-reduced and we just discovered
    // a tactical position (eval_sum < 0 ≈ "both sides think they're losing"
    // pattern that signals tactical density). Mirrors SF/Reckless/Halogen.
    if prior_reduction >= tp(&HINDSIGHT_EXT_MIN_RED) && eval_sum < tp(&HINDSIGHT_EXT_THRESH) {
        depth += 1;
    }
}
```

Tunables (new):
- `HINDSIGHT_EXT_MIN_RED` (default 3, range 2-5, c_end 0.5) — SF uses 3
- `HINDSIGHT_EXT_THRESH` (default 0, range -200..200, c_end 20) — SF uses
  effectively 0 via `!opponentWorsening`

SPRT bounds `[0, 3]` per CLAUDE.md default. Retune-on-branch with focused
hindsight cluster after if H1.

### 4.2 PV-node LMR depth extension (§3.3)

Cheap, universal. Lets the negative-reduction calculations actually reduce
search effort below depth-1 — they currently get clamped to 0 plies.

```rust
// search.rs:3004-3010, replace clamp:
// OLD:
//   if reduction < 0 { reduction = 0; }
//   if reduction > new_depth - 1 { reduction = new_depth - 1; }

// NEW: allow negative reduction (extension) at PV / first few moves.
let pv_bonus = if beta - alpha > 1 { 1 } else { 0 };
let early_bonus = if move_count <= tp(&LMR_EARLY_EXT_MC) { 1 } else { 0 };
let max_ext = pv_bonus + early_bonus;  // up to +2 plies
let min_r = -max_ext;
let max_r = new_depth - 1;
if reduction < min_r { reduction = min_r; }
if reduction > max_r { reduction = max_r; }
```

The `lmr_depth = new_depth - reduction` computation downstream is
unchanged; if reduction is negative, lmr_depth > new_depth and we naturally
search deeper.

Tunables:
- `LMR_EARLY_EXT_MC` (default 3, range 2-5, c_end 0.5) — Reckless uses 3.

SPRT `[0, 3]`. Reckless gets +2 PvNode extension; Coda + ~+1 first try is
conservative.

### 4.3 Killer/refutation LMR exemption (§3.2, threshold variant)

Easiest path: don't re-introduce a separate killer table; use main-history
threshold as proxy.

```rust
// search.rs:2978-2980, currently:
//   let hist_adj = hist_score / tp(&LMR_HIST_DIV);
//   reduction -= hist_adj;
// Add:
let main_only = info.history.main_score(from, to, enemy_attacks);
if main_only >= tp(&LMR_REFUTATION_THRESH) {
    reduction -= tp(&LMR_REFUTATION_BONUS);  // default 1
}
```

Tunables:
- `LMR_REFUTATION_THRESH` (default 6000, range 3000-12000, c_end 500)
- `LMR_REFUTATION_BONUS` (default 1, range 0-2, c_end 0.5)

SPRT `[0, 3]`. The simpler threshold-on-main-history is closer to what
Berserk's `killerOrCounter` test is doing in spirit (a high-signal move
that's not the TT move). If H0, fall back to the lite-killer-array variant
(§3.2 path-a) which is more direct but more LOC.

---

## 5. Re-search behavior survey

Coda's current pattern (search.rs:3066-3090):
```rust
// LMR fail-high: doDeeper / doShallower
let mut do_deeper_adj = 0;
if lmr_score > best_score + 60 + 10 * reduction { do_deeper_adj = 1; }
else if lmr_score < best_score + 20 { do_deeper_adj = -1; }
lmr_score = -negamax(... new_depth + do_deeper_adj ...);

// PVS: full window if score in (alpha, beta)
if lmr_score > alpha && lmr_score < beta { score = -negamax(... full window ...); }
```

Cross-engine pattern audit:

| Pattern | Engines |
|---|---|
| **Two-step** doDeeper(±1) + full-window if needed | Coda, SF, Obsidian, Berserk, Plenty, Caissa, Horsie, Tarnished, Viridithas |
| **Three-tier doDeeper** (+0/+1/+2 depending on margin) | Reckless (`+(score>best+61) + (score>best+801) - (score<best+5+reduced_d)`) |
| **One-step "shallower only"** | Halogen (just `bool reduce`, no doDeeper) |
| **TT-bound short-circuit** to skip re-search | PlentyChess (`ttDepth - offset >= newDepth && ttFlag = UPPER`) |
| **Search-explosion guard on doDeeper** | Caissa (`ply < 2*rootDepth`) |
| **Cont-hist verdict update post-LMR** | SF, Obsidian, Viridithas, Berserk, Plenty, Horsie |

Coda's two-step pattern is consensus. Two non-consensus features worth
considering:

1. **Cont-hist verdict update** (§ engine row above): after LMR re-search,
   write a bonus/penalty to cont-hist based on whether the move actually
   passed/failed. Coda doesn't do this. Six engines do. Effort: small;
   ~10 LOC. Elo prior: Low-Medium (1-3). Mechanism: turns the LMR re-search
   itself into ordering signal for future siblings.

2. **TT-bound short-circuit on re-search** (PlentyChess only). Skip the
   full-depth re-search entirely if the TT proves an upper-bound. Saves
   nodes. Elo prior: Low (0-2). Effort: small.

Reckless's three-tier doDeeper is more aggressive than Coda's two-step;
unclear if it converts to Elo independent of all the other Reckless-specific
knobs.

---

## 6. Move ordering — LMR sensitivity

LMR is move-order-sensitive: when the move that was reduced turns out to
be the best move, we eat a re-search penalty (and risk missing it
entirely if we picked the wrong reduction). Better ordering means LMR
bites less on critical moves.

Observations from this survey:

- **Killer/refutation as a separable LMR signal** (§3.2). Six engines
  treat killer/counter/refutation moves with explicit LMR exemption,
  separate from where they sit in the move loop. Coda unified killers
  into 4D history for ordering purposes; the LMR side of the trade went
  unmade. Threshold-on-main-history (§4.3) is the cheap fix.

- **SF, Reckless, Tarnished use child cutoffCnt in LMR** (§3.12). This
  is a kind of "the subtree is hot" signal. Different engines treat
  cutoffCnt as either an ordering signal (kill the rest of the moves
  faster) or an LMR signal (reduce them more). Both are valid; combining
  is rare. Coda does neither.

- **Reckless's anti-cascade reduction guard** (§3.5) is a unique
  parent-child reduction-correlation signal. Treats the reduction stack
  itself as ordering information.

- **Tarnished's factorized LMR table** (§3.15) is the most extreme
  example of "interaction-aware" LMR adjustment. They get +56 Elo over
  Coda; this is one of several mechanisms that could explain it.

- **Cont-hist depth choice for LMR varies**: Coda 1+2/2; Stockfish 1+2;
  Horsie 1+2+4. Stockfish uses 1+2+4+6 in movepicker but only 1+2 in LMR
  adjustment — they decided that LMR's signal-to-noise prefers shorter
  windows. Worth A/B against Horsie's longer.

The LMR-vs-ordering trade lives in three knobs: (a) what's exempt from
LMR entirely, (b) how strongly history feeds the reduction-amount, and
(c) what gets re-searched on fail-high. Coda has heavy knob-(b)
investment (multiple history sources combined into hist_score) but
relatively little knob-(a) investment beyond the macro skip conditions.
The candidates in §3.1-3.4 fix that.

---

## Appendix: line-cite quick index

- Stockfish call site: `Stockfish/src/search.cpp:1062-1281`
- Stockfish hindsight: `Stockfish/src/search.cpp:773-777`
- Reckless call site: `Reckless/src/search.rs:815-944`
- Reckless hindsight: `Reckless/src/search.rs:475-491`
- Viridithas call site: `viridithas/src/search.rs:1444-1517`
- Viridithas hindsight: `viridithas/src/search.rs:1041-1053`
- Obsidian call site: `Obsidian/src/search.cpp:1067-1107`
- Berserk call site: `berserk-13/src/search.c:617-748` (early R adj before
  make_move at 617-619, main bunch at 705-748)
- PlentyChess call site: `PlentyChess/src/search.cpp:1090-1148`
- PlentyChess post-LMR hindsight: `PlentyChess/src/search.cpp:783-790`
- Caissa call site: `Caissa/src/backend/Search.cpp:1870-1994`
- Halogen `late_move_reduction()`: `Halogen/src/search/search.cpp:587-616`
- Halogen call site: `Halogen/src/search/search.cpp:1151-1154`
- Halogen hindsight: `Halogen/src/search/search.cpp:923-927`
- Tarnished factorized table: `Tarnished/src/search.cpp:202-280`
- Tarnished call site: `Tarnished/src/search.cpp:660-705`
- Horsie call site: `Horsie/src/search.cpp:625-664`
- Coda call site: `coda/src/search.rs:2905-3098`
- Coda hindsight: `coda/src/search.rs:2261-2280`
- Coda LMR init: `coda/src/search.rs:1003-1022`
