# v9 → main merge plan (2026-04-19)

Planning doc for the eventual merge of `feature/threat-inputs` (v9 trunk)
into `main`. Captures the decision points, open questions, and
recommended sequencing.

## Status at time of writing

`feature/threat-inputs` has diverged from `main` substantially:

- **v9 NNUE architecture** — 768-width accumulator + 66,864 threat
  features + 16→32 hidden layers. `main` uses v5 (1536 accumulator,
  no threats, direct FT→output).
- **Search features added on v9 trunk that do not exist on main**:
  - 2b slider-iteration rewrite in threats.rs (+10.0 Elo)
  - ProbCut king-zone-pressure gate (+7.03)
  - LMR king-pressure modifier (+6.81)
  - Futility our_defenses widener (+7.00)
  - Full-attacks-history 4D indexing (+14.2)
  - King-zone-pressure NMP gate (+7.94)
  - Escape-capture bonuses (stratified form in flight)
  - 2b rewrite, xray_blockers helper, profile-threats counters
- **Tunables diverged** — all 60 pruning parameters have different
  values on v9 trunk (#489 retune + ongoing compound retunes).
- **Docs and tooling added**: signal_context_sweep, threat_ideas_plan,
  move_ordering_ideas, training_crelu_hidden, v9_nps_findings.
- **Self-play Elo**: v9 trunk > v5 main by ≥ +40 Elo as of 2026-04-19,
  with in-flight SPRTs (B1, A1 retest, threat-mag-LMR) likely adding
  another +10-30.

`main` has received in parallel:
- TM bug fixes (sleep-to-soft-floor, ponder fresh-search, dynamic-soft
  floor, ponderhit race) — combined ~+5-10 Elo (Atlas's work).
- tm_score.py and blunder_suite infrastructure.

These are now merged into v9 trunk (merge commit 80139f8).

## The core decision: tunable handling

When v9 merges to main, we need to answer: **what tunable values does
main ship with, given v5 and v9 nets have different optima?**

### Option 1 — "v9-as-default, v5-as-legacy" (RECOMMENDED)

Main ships with v9 trunk's tunable values. v5 users still play but at
slightly sub-optimal tuning.

**Pro:**
- Zero ongoing infrastructure cost.
- v9 is where Elo gains live; default-optimising for v9 gives the
  product its best face forward.
- Future SPSA tunes target v9 directly — no per-net-tag bookkeeping.
- Matches the "v5 is frozen legacy" framing (no new v5 nets incoming).

**Con:**
- v5 users running production v5 nets get sub-optimal pruning.
  Estimated cost: ~10-20 Elo vs v5-tuned-for-v5 baseline.
- One-time perception issue: "main got tuned for a different net".

**Mitigation:**
- Before merging, run one SPRT: v9 trunk vs current main, both using
  v5 production net. Confirms the Elo of v9-trunk-code with v5-net is
  still above current main (we expect it to be, because v5 search
  features also improved).
- If that SPRT H0's or regresses: fall back to option 2.

### Option 2 — Runtime per-net overrides

Main detects net architecture at NNUE load (presence of threat
features, FT size, hidden-layer dimensions) and sets tunable defaults
from a lookup table keyed on architecture signature.

**Pro:**
- Both v5 and v9 users get optimal tunings for their net.
- Extensible to future architectures (v10, v11, etc.).

**Con:**
- New infrastructure to maintain — every SPSA tune needs an
  architecture tag; every new net architecture needs a tuning entry.
- Two sets of SPSA values to keep current; one usually drifts stale.
- Runtime switching complicates SPSA on main (which net to tune against?).
- The "architecture signature → tuning" map is a growing surface
  area of behaviour hidden behind a single load-time decision.

**When to reach for it:**
- Only if option 1's v5 regression is demonstrably > 20 Elo, and
- v5 users are numerous enough that regression matters.

### Option 3 — Separate main-v5 and main-v9 branches

Two long-lived trunks, each independently tuned and merged for their
own net architecture.

**Pro:**
- Cleanest separation.
- No runtime logic.

**Con:**
- Every non-tunable fix (search bug, NPS improvement, UCI feature,
  etc.) has to be cherry-picked across both branches.
- Release engineering overhead: multiple builds to ship.
- We are already drowning in branch-maintenance; this makes it worse.

**Recommendation: do not pursue.** The cost of dual-branch maintenance
is permanent; the benefit (v5 users get optimal tunings) is bounded
and decaying as v5 nets age out.

## Recommended path: option 1 with a safety check

1. **Pre-merge SPRT**: test v9 trunk code + v5 production net vs main +
   v5 production net. Same net both sides; only search code differs.
   - If v9-code with v5-net is **positive (≥ +5)**: proceed with
     option 1. v9 code improvements more than offset tuning
     mismatch for v5 users.
   - If v9-code with v5-net is **flat (−3 to +3)**: option 1 still
     fine — v5 users neutral.
   - If v9-code with v5-net is **negative (≤ −5)**: reconsider.
     Either (a) pursue option 2, (b) improve v5-net tuning on v9 code
     before merging, (c) defer merge until v5 is deprecated.

2. **Net-loading default**: main's `net.txt` remains the v5 production
   net for backwards compatibility. v9 users override via UCI option
   `NNUEFile` or `make EVALFILE=...`.

3. **Documentation**: add a `docs/v9_migration_notes.md` explaining:
   - Tunables on main are v9-optimised.
   - v5 users experience ~10-20 Elo regression vs v5-tuned-for-v5
     (deprecated state; not fixable without ongoing maintenance).
   - v9 is the recommended net architecture going forward.

## Pre-merge checklist

Before opening the PR:

- [ ] Pre-merge SPRT result (above). Attach.
- [ ] All in-flight SPRTs on v9 trunk resolved (no dangling H1s that
      should land first).
- [ ] `feature/threat-inputs` current with `main` (✓ done 2026-04-19).
- [ ] Final 60-param post-merge retune on v9 trunk (currently landed
      via #489 → +7.38 H1). Run one more if another round of features
      lands before merge.
- [ ] Run `cargo test --release` — all passing (currently 125/125).
- [ ] `coda bench` with v9 prod net matches commit-message bench.
- [ ] `coda bench` with v5 production net also runs (verify legacy
      path works post-merge).
- [ ] README / docs updated to describe v9 as default architecture.
- [ ] `net.txt` flipped to v9 production net at merge (decided
      2026-04-19 — if main carries v9 tunings, the default net
      should be v9 for consistency; v5 nets still load via
      `-nnue` override).
- [ ] v7 training configs (`training/configs/v7_*.rs`) marked
      deprecated. v7 inference code in `src/nnue.rs` stays — v9's
      hidden-layer path (L1/L2 matmul, SCReLU/CReLU pack) is
      built on top of v7's infrastructure; removal would be
      regression surface for no benefit.

## Architectures supported post-merge

- **v5** (legacy) — backward compat only. v5 nets load and work.
  Not actively tuned against. Existing users with v5 nets get a
  small tune-mismatch regression (bounded; v5 nets don't change).
- **v9** (primary) — default architecture. `net.txt` points to a
  v9 production net. Tunables are v9-calibrated. All active
  development targets v9.
- **v7** (deprecated) — no new training runs. Existing v7 nets
  still load via the v7 code path in `src/nnue.rs`. Not a
  supported end-user configuration.

## Open questions for Adam

1. **Timing** — merge when the next ~2 weeks of in-flight work
   resolves, or after a specific Elo milestone (e.g., +100 over
   current main)?
2. **Release cadence** — is there a release cut point (e.g. Coda 2.0)
   that should coincide with the merge?
3. **Which specific v9 net becomes the `net.txt` default** at merge
   time? Current candidates: `net-v9-768th16x32-w15-e800s800-xray.nnue`
   (hash 6AEA210B) or a later CReLU-L1/L2-trained variant if the
   CReLU direction (validated +4 at s200 via #497) produces a
   stronger s800 net before merge.

## What the merge WON'T change

- Backwards compat: any existing binary+net combination continues to
  work. Main running a v5 net stays runnable; main running a v9 net
  with `HiddenActivation=screlu` works.
- UCI protocol: no change. UCI options for v9-specific features
  (`HiddenActivation`, `SparseL1`) default to non-v9-interfering
  values.
- Test suite: all existing tests continue to pass.
- Bench tooling: OB integration unchanged.

## Risks

- **Rebasing conflict surface.** v9 and main have diverged
  substantially. Future main updates will re-conflict. Recommendation:
  merge main → v9 weekly until the v9→main merge lands, to keep
  conflicts small.
- **Tunable drift during SPSA**. Tunes on v9 trunk during the
  pre-merge window will add more parameter divergence. Mitigate by
  (a) retune-on-branch before merge, (b) freeze v9 trunk tunables
  during the final merge prep.
- **Reviewer burden**. The eventual PR will be enormous. Consider
  pre-splitting into reviewable chunks: (1) threat-feature
  infrastructure, (2) v9 inference paths, (3) search-side threat
  consumers, (4) tunable retune, (5) docs.

## Companion docs

- `docs/threat_features_design.md` — v9 architecture original spec
- `docs/signal_context_sweep_2026-04-19.md` — signal × context matrix
- `docs/threat_ideas_plan_2026-04-19.md` — tiered threat experiment
  plan with live results
- `docs/move_ordering_ideas_2026-04-19.md` — ordering-specific catalogue
- `docs/training_crelu_hidden_2026-04-18.md` — clipped-ReLU L1/L2 training
