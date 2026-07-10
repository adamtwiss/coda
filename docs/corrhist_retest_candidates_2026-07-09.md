# Corrhist re-test candidates (post-fix baseline) — 2026-07-09

**Why this doc.** The correction-history **root-cause fix** (Finding #1:
train the update on the *corrected residual*, not raw eval — `docs/corrhist_audit_2026-07-08.md`)
merged 2026-07-09, and `mat_damp` was dropped (residual subsumes it). Every
corrhist-related experiment that was run against the **old raw-baseline** corrhist
is now **systematically invalidated** — the subsystem it was measured against no
longer exists. Several of them came from a single audit wave, so they share one
root confound: one fix invalidates the batch. Re-running them is warranted rather
than treating the old H0s as settled ("failed experiments can succeed after
accumulated structural changes").

This is a **candidate list to merge with Adam's own list** — read-only/planning,
nothing fires yet.

## The two gates (read before adding anything)

1. **Structural-only.** The production LTC `--core` retune **#2664** (in flight)
   is already recalibrating *every* correction **tunable** against the fixed
   baseline (`CORR_HIST_DIV`, `CORR_ERR_DIV`, `CORR_W_*`, `CORR_UPDATE_WEIGHT_MAX`,
   `CORR_BONUS_CAP_DIV`). So prior **parameter-value** experiments (e.g. #2318
   `CORR_ERR_DIV 8→6`) do **not** need separate re-runs — the tune subsumes them.
   Only **structural** corrhist work (new sources/keys/topology/terms that #2664
   can't add or remove) belongs on this list.
2. **Fire after #2664 applies.** Re-test against the **fixed *and retuned***
   baseline (post-#2664-applied main), not fixed-but-untuned main — otherwise the
   re-tests carry a stale-downwind confound (a second version of the same bug).
   #2664 is moving the corrhist-downwind cluster hard (`CORR_HIST_DIV` −26%, the
   LMR history-consumption cluster), so this sequencing is load-bearing.

## HIGH — structural, high corrhist-interaction, likely flipped

| # | candidate | source | why it may flip now |
|---|---|---|---|
| H1 | **Paired 2-ply (+4-ply) continuation correction** | **Headline of BOTH audits**: `F1` in `docs/corrhist_deepaudit_2026-06-26.md` + `#5` in `docs/corrhist_audit_2026-07-08.md` | **The single top structural candidate — Coda is the sole 6/6 flat-1-ply outlier.** `cont_corr` is a flat `[piece][to]` (768 entries, massively aliased); every reference uses a paired 2-D (ss-2)+(ss-4) form. Per-ply data already exists (`info.moved_piece_stack` / `moved_to_stack`), so no new *board* plumbing — the work is a larger 4D table + subtable index + threading `ply` through `corrected_eval`/`correction_value` (~5 sites incl. qsearch + the corrhist unit tests kept green). **Retune-on-branch archetype** (deepaudit F1 plan): likely flat raw until `CORR_W_CONT`+`CORR_HIST_DIV` recalibrate — mirrors the cont-hist-malus precedent (flat raw → +6.5 retuned). Zeus deferred it as "not a tail-of-session change." |
| H2 | **Minor-piece correction — with a *correct* new key** | audit #6 + deepaudit `F2` | **Not a re-add — a genuinely new feature.** The #1318 H0 dropped a *broken* table aliased to `non_pawn_key`; Coda has never had a correct `minor_piece_key`. F2 = add a real incremental `minor_piece_key` Zobrist on the board (knights+bishops, ±king per SF) with make/unmake + recompute-verify parity, *then* a `minor_corr` source (`CORR_W_MINOR`). 3/6 references keep a proper minor key (SF wt 8620, Reckless, PlentyChess+major) — a split, MEDIUM conviction. Bigger change (board Zobrist). Do **after** H1, on the H1-retuned baseline. |
| H3 | **`trans_corr` ablation (`CORR_W_TRANS=0`)** | audit #4, `[-1.5,1.5]` | Coda-unique move-signature source (`hash ^ last.hash`), no peer precedent, most rail-exposed in shuffling positions. *Partly* probed by #2664 (moving `CORR_W_TRANS` −10%), but an explicit ablation against the fixed baseline is the clean read. If neutral/positive, drop it (fewer rails, less memory) — which then makes H2 the replacement. |
| H4 | **Futility `\|corr\|` uncertainty term (`FUT_CORR_MULT`)** | audit #2; branch `zeus/corr-uncertainty-pruning` | **Prime candidate — arguably the most affected of all.** It scales pruning by `\|corr\|`, and (a) the residual fix **changed the `\|corr\|` distribution itself** (no more railing to ±LIMIT), and (b) its interaction params are the exact ones #2664 is moving hardest (`FUT_LMR_DEPTH` −14%, `FUT_PER_DEPTH` −5.7%). A **preliminary** SPRT is running now, but against **fixed-but-untuned** main — a preliminary read, not the definitive one (it *is* the gate-#2 confound). Re-validate against the retuned baseline regardless of how the preliminary lands. (NB the LMR + DEXT/singular `\|corr\|` terms already exist and are *parameter*-only, so #2664 subsumes those — futility is the one genuine structural gap.) |

## HIGH — new / alternative correction SOURCES (H0'd against raw baseline)

A new source that added *rail-noise* under the buggy raw baseline could H0 for
that reason alone; the residual fix removes the railing, so these deserve a clean
re-test. Nearly the whole batch H0'd narrowly (−0.3 to −0.6) — consistent with
"good idea, killed by the buggy baseline it was measured against."

| # | candidate | source | note |
|---|---|---|---|
| H5 | **threat-bitboard source** (`CORR_W_THREAT_BB`) | `experiment/threat-bb-corrhist` (#2547) | New source keyed on `hash(enemy_attacks)`, Tcheran pattern — captures attacker-*structure* bias, orthogonal to pawn/np/cont/trans. Wired into all 3 sites. Rejected vs raw baseline. |
| H6 | **prev-move source** | `experiment/prev-move-corrhist` (#905, −0.6/H0; #911 stopped) | move-signature-ish source; overlaps the `trans_corr` question (H3). Re-test alongside the trans ablation. |
| H7 | **bad-capture / losing-caps training** | `codex/bad-capture-corrhist`, `experiment/corrhist-train-losing-caps` (#1402, −0.4/H0, Viri #420) | widen the update's training signal to (losing) captures. Structural update-gate change. |

## MEDIUM — correction update/mechanism + moderate interaction

| # | candidate | source | note |
|---|---|---|---|
| M1 | **multicut-corrhist** | #2263 (H0) | corrhist update on the multicut path; rare path, may behave differently under residual. |
| M2 | **king-in-non_pawn_key** | #2144 (H0, **bench-flip-confounded**) | a corrhist source-key change whose original H0 was already suspect (bench delta flipped +5% after later merges); a clean re-test against the fixed baseline is warranted on those grounds alone. |
| M3 | **eval-delta hist nudge / wall-pawn malus** | deep-audit wave 3 (2026-06-26) | lower confidence; batch with H1 — same invalidated audit provenance. |
| M4 | **is-check-conditioned cont-corr** | `experiment/contcorr-check` (#2304, −0.3/H0) | condition cont-corr on check context (references key on `[in_check]`). Subsumed if H1 (paired form with check flags) is done; otherwise a cheaper standalone. |
| M5 | **update during singular-exclusion** | `experiment/corrhist-update-during-se` (#1401, −1.6/H0) | audit lists Coda's *not*-updating-during-SE as "what Coda gets right," so this is lower-conviction — but its H0 was on the raw baseline; re-check only after the higher items, and expect to *confirm the current gating*. |

## LOW — cleanup / precision (near-neutral, `[-2,1]`)

| # | candidate | source | note |
|---|---|---|---|
| L1 | **Collapse two-stage division** (`CORR_HIST_DIV` × `CORR_HIST_GRAIN_T` → one divisor) | deepaudit `F3` | Coda two-stages the consumption divide (`/DIV` then `/GRAIN_T≈14`); every reference does a single divide. Collinear knobs + intermediate truncation throws away precision. Fold GRAIN_T into DIV, widen DIV range to absorb it. Near-bench-neutral; the durable win is precision/clarity, not SPSA-noise. (The catastrophic `corrhist-grain-t-1` #1205 −41 was a bad *config* of exactly this knob — F3 is the clean version.) Bundle with another small corrhist cleanup for SNR. |

## Truly handled — do NOT re-list

**Caveat (2026-07-09): "a test is in flight" ≠ "handled."** A corrhist test
running now against **fixed-but-untuned** main is a *preliminary* read and still
needs re-validation post-#2664 — that IS the gate-#2 confound. The futility
`|corr|` term (`zeus/corr-uncertainty-pruning`) is such a case and has been moved
to **H4** above, not here. Only the items below are genuinely settled:

- **50MR corrhist-index bucketing** (audit #3, Reckless pattern) — Zeus
  **deferred**: residual likely cures the drift, so test **only if** a
  high-material *locked* repro still shows drift on the retuned baseline. Cheap
  premise-check: corrhist-on vs `NO_CORRECTION=1` on such a repro.
- **All correction parameter-value tunes** (CORR_ERR_DIV, CORR_W_*,
  CORR_UPDATE_WEIGHT_MAX, CORR_BONUS_CAP) — **subsumed by #2664**.

## The "same-audit" batch

H1, M2/M3 (and the CORR_ERR_DIV param probe now subsumed by #2664) all originate
in the **deep-audit wave 3 (2026-06-26)** and were tested together against the raw
baseline — one root confound invalidates the batch, which is the core reason this
list exists.

## The #2500 killer datapoint — why this whole list is justified

**The single strongest reason to distrust every raw-baseline corrhist H0:** the
residual fix — now confirmed **+18 cross-engine, the biggest fix in some time** —
was *itself* rejected twice against the raw baseline:
- `#2453 fix/corrhist-residual` — −0.9 untuned, rejected.
- `#2500 fix/corrhist-residual` (retuned, #2492 `CORR_HIST_DIV` on-branch) — **−3.8
  ±3.1 H0**, rejected, with the recorded conclusion *"corrhist-residual genuinely
  doesn't fit Coda"* and *"the retune track record for rescuing an ≈flat feature
  is now clearly poor."*

**Both of those conclusions are now proven wrong.** The idea fit; the retune *can*
rescue. It failed for two reasons that apply to the *entire* raw-baseline corrhist
program: **(1) a bad/insufficient retune** (Zeus's #2648 got it right where #2492
didn't), and **(2) self-play SPRT can't see the value** (even Zeus's *good* retune
read +0.1 self-play vs +18 cross-engine). So the corrhist H0s aren't just
"measured against a buggy baseline" — they're triple-jeopardy: buggy baseline ×
possibly-wrong retune × self-play blindness. That is why re-testing the batch —
against the fixed **and** #2664-retuned baseline, and reading **cross-engine**, not
just self-play — is high-value rather than busywork.

## Resolved (context — do not re-open)

- **Finding #1 residual baseline** (#2453/#2500 → landed via Zeus's #2648) — see
  above; it *is* the fix, not a re-test candidate.
- **#2116 corrhist-allnode** (fail-low training) — H1, **merged**; correct as-is.
- **#1318 minor/major drop** — H1 on the *aliased* key; the re-test is F2/H2 (a
  new *correct* key), not a re-add of this.

## Status / assignments (2026-07-10, post-#2664-merge)

Retuned baseline is live (#2664 merged, +4.8 LTC). Re-tests fired against it:

| item | status |
|---|---|
| **H1** 2-ply/4-ply cont-corr | **Zeus — FIRED.** Implemented: flat 1-ply `[piece][to]` → paired 4D `[prev_piece][prev_to][cur_piece][cur_to]`, indexed by the last move (ply-1), subtable at ply-2 AND ply-4, summed; `ply` threaded through `corrected_eval`/`correction_value`/`update_correction_history` (incl. qsearch, clamped); `moved_piece_stack` (go_piece 1-12) for the older-ply pieces. Fortress + corrhist unit tests green (203/0). **Untuned SPRT #2675 `[0,3]`** (dev 2085296 vs main 2381675, −12% nodes). Per the archetype, expect flat untuned → **focused corrhist-cluster retune-on-branch is the definitive test**, then re-SPRT (and read cross-engine, per the #2500 lesson). |
| **H2** minor-piece-key corr | **Zeus** — queued: implement once H1 lands and test on the **H1-retuned** baseline. New feature (not a re-add): incremental `minor_piece_key` Zobrist on the board (knights+bishops, ±king per SF) with make/unmake + recompute-verify parity, then a `minor_corr` source (`CORR_W_MINOR`). MEDIUM conviction (3/6 refs keep a proper minor key; the #1318 H0 was a *broken* key aliased to `non_pawn_key`). |
| **H3** trans_corr ablation | **RESOLVED — keep.** Ablation #2670 −3.0 H0 (removing trans costs real Elo). |
| **H4** futility \|corr\| | **~neutral.** #2671 +0.6 →H0 (sub-midpoint, won't H1). Non-reg, not merge-worthy alone. |
| **H5** threat-bb source | **regresses untuned** (#2672 −3.5 H0). Adds a 6th source without a DIV rebalance → over-corrects; low prior (threats ~0 corr with eval error). Lean drop unless a CORR_W_THREAT_BB+DIV retune is wanted. |
| **H6** prev-move | **flagged superseded** by H1 (overlaps cont/trans; its follow-up term is a cruder 2-ply cont). Not re-implemented. |
| second-order-hist (Alice #8) | retune-on-branch #2673 done (LMR_HIST_DIV +16%); tuned SPRT #2674 in flight. |

**Read so far:** the residual fix + #2664 got the corrhist subsystem into good
shape — the re-tests mostly confirm it's well-calibrated (trans valuable, |corr|
-futility ~neutral, threat-bb not additive). The one real structural lever left
is **H1 (Zeus — fired, #2675)**. **H2** (minor-piece-key, needs a new board
Zobrist) is **Zeus's** next structural candidate, to be built and tested on the
H1-retuned baseline once H1 lands. So Zeus owns the two remaining structural
levers (H1, H2); Hercules's items (H3–H6, second-order-hist) are resolved or in
flight.

### Bottom-row (untouched) disposition — 2026-07-10 (Zeus)

Reviewed H7/M1/M2/M3/M4/M5/L1 against the well-calibrated post-#2664 baseline.
**Most fold into workstreams already owned, none needs a new independent track:**

| item | disposition |
|---|---|
| **M4** check-conditioned cont-corr | **Fold into H1.** It's an extra `[in_check]` dimension on the *paired* cont-corr (Reckless keys cont-corr on check). Add as an H1 follow-up *only if H1's base form lands* — don't gold-plate an unproven feature. |
| **M2** king-in-non_pawn_key | **Fold into H2.** Same question as H2 ("what pieces belong in a correction key"), same board-Zobrist machinery — bundle them. Its original #2144 H0 was **bench-flip-confounded** (invalid), so it earns a clean read anyway. Best Elo-EV of the leftovers. |
| **L1** collapse two-stage divide | **Worth it, low priority, `[-2,1]`.** Real precision loss (`/DIV` then `/GRAIN_T` truncates twice), but ~0 Elo (precision/clarity win) and it perturbs the just-tuned `DIV`/`GRAIN_T`. Do opportunistically, not while H1 is settling. |
| **H7** bad-capture training | **Skip.** Against reference consensus — SF/most engines deliberately gate captures OUT of the corrhist update; the −0.4 H0 is consistent with that, baseline-independent. |
| **M1** multicut-corrhist | **Skip.** Rare path → tiny effect, poor SNR. |
| **M3** eval-delta nudge / wall-pawn malus | **Skip.** Not correction history (main-hist/eval terms), vague, lowest confidence. |
| **M5** update-during-SE | **Skip.** Audit lists Coda's not-updating-during-SE as "what Coda gets right" (matches refs); a re-test would just re-confirm the H0. |

So Zeus's corrhist queue is: **H1** (in flight: untuned #2675 + retune #2676) → **H2 (+M2 bundled)** → **M4** (H1 extension, if H1 lands) → **L1** (cleanup). H7/M1/M3/M5 dropped.
