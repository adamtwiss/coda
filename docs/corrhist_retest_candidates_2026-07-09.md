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
| H1 | **2-ply (+4-ply) continuation correction** | audit #5; deep-audit "two-ply cont-corrhist" + "paired 2-D continuation correction" (`docs/corrhist_deepaudit_2026-06-26.md`) | Coda's `cont_corr` is 1-ply flat `[piece][to]`; SF/Reckless/Viri use stacked (ss-2)+(ss-4). Zeus **deferred** it (needs a 4D table + `ply` threaded through `corrected_eval`/`correction_value`, ~5 call sites incl. qsearch + the corrhist unit tests kept green). Highest-value structural item — the residual baseline gives a richer cont-corr a cleaner signal to sharpen. |
| H2 | **Minor-source correction re-add** | audit #6 | Coda dropped minor/major sources (2026-05-19) in favour of `trans`. SF keeps minor; Viri keeps minor+major. Peer-validated and more principled — the natural replacement experiment, especially paired with H3. |
| H3 | **`trans_corr` ablation (`CORR_W_TRANS=0`)** | audit #4, `[-1.5,1.5]` | Coda-unique move-signature source (`hash ^ last.hash`), no peer precedent, most rail-exposed in shuffling positions. *Partly* probed by #2664 (moving `CORR_W_TRANS` −10%), but an explicit ablation against the fixed baseline is the clean read. If neutral/positive, drop it (fewer rails, less memory) — which then makes H2 the replacement. |

## MEDIUM — structural, moderate interaction

| # | candidate | source | note |
|---|---|---|---|
| M1 | **multicut-corrhist** | #2263 (H0) | corrhist update on the multicut path; rare path, may behave differently under residual. |
| M2 | **king-in-non_pawn_key** | #2144 (H0, **bench-flip-confounded**) | a corrhist source-key change whose original H0 was already suspect (bench delta flipped +5% after later merges); a clean re-test against the fixed baseline is warranted on those grounds alone. |
| M3 | **eval-delta hist nudge / wall-pawn malus** | deep-audit wave 3 (2026-06-26) | lower confidence; batch with H1 — same invalidated audit provenance. |

## Already in flight / handled (do NOT re-list)

- **Futility `|corr|` uncertainty term** (audit #2, `FUT_CORR_MULT`) — Zeus's
  branch `zeus/corr-uncertainty-pruning`, SPRT `[0,3]` live. NB audit #2's
  "Coda uses `|corr|` nowhere" claim was **corrected**: LMR
  (`reduction -= complexity·LMR_SCALE/LMR_COMPLEXITY_DIV`) and DEXT/singular
  (`dext_margin -= DEXT_MARGIN_CORR·corr_abs/128`) already use it; **futility was
  the only genuine gap**.
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

## Resolved (context — do not re-open)

- **Finding #1 residual baseline** itself was rejected twice earlier as
  `fix/corrhist-residual` (#2453 −0.9 untuned; #2500 −3.8 with a *different*
  CORR_HIST_DIV retune). Zeus's version merged because the residual reframing +
  the #2648 retune got it right. The earlier rejections are **not** re-test
  candidates — they *are* the fix, now landed.
