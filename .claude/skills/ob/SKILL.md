---
name: ob
description: OpenBench usage for Coda — bench measurement, SPRT submission, SPSA tune submission, stopping tests, reading results. Invoke this skill before any OB operation. Single source of truth for OB usage; supersedes scattered per-Claude memories.
---

# OB Skill — OpenBench Usage for Coda

**Invoke before**: any `ob_submit.py`, `ob_tune.py`, `ob_stop.py`,
`ob_status.py`, `ob_tune_status.py`, `ob_upload_net.py` call; before
deciding SPRT bounds; before benching for OB.

**The pattern these instructions prevent**: bench-mismatch errors,
wrong-bounds submissions, stopped-tests-that-didn't-actually-stop,
acted-on-early-N noise, missed `/errors/` followups.

---

## 1. Credentials & setup

- **Server**: `https://ob.atwiss.com`
- **Game runner**: **fastchess** (not cutechess). Ponder is NOT exposed
  as an SPRT configuration option — workers run engines without ponder
  regardless of the `Ponder` UCI option default. Behavioural changes
  gated on ponder (Phase 14 ponderhit floor, ponder-miss min-think
  floor) are inert under OB SPRT — only deployment (codabot lichess)
  exercises those paths. Use SPRT as a non-regression check for the
  non-ponder path; rely on deployment + cross-engine local RR for the
  actual gain measurement.
- **Usernames** (different roles, filter by user to see who owns what):
  - `adam` — human
  - `claude` — Hercules (implementation/fleet)
  - `atlas` — Atlas (tactical/live-play)
- **Password**: read ONLY from the `OPENBENCH_PASSWORD` environment variable.
  NEVER hardcode it here, in a script, or in any committed file, and NEVER paste
  it inline into a shell command — this is an admin OB account, and a leaked
  password is a remote-code-execution risk on the fleet workers. If
  `OPENBENCH_PASSWORD` is not set in your shell, STOP and ask the user to set it
  on this machine; do not guess, reconstruct, or inline it. All
  `scripts/ob_*.py` already read it from env
  (`os.environ.get('OPENBENCH_PASSWORD')`) — just run them with the env var
  present (e.g. it's exported in your shell profile).

**Per-Claude OB identity**: claims-of-test-ownership matter for `ob_stop.py`
and similar — you can only stop your own tests. Don't stop tests
owned by `adam` or `atlas` without explicit instruction.

---

## 2. Bench measurement — the most common failure mode

### 2.1. The Bench-for-OB ritual

OB workers run `make && ./coda bench` on the branch you submit. The
`Bench: <nodes>` you submit MUST match what they measure, or workers
reject as "Wrong Bench" and the test idles.

**Always:**
```bash
make                # builds ./coda with embedded prod net via net.txt
./coda bench        # reports the deterministic node count
```

**Never:**
- `cargo build --release && ./target/release/coda bench` — differs
  ~30% (no embedded net, different code paths)
- Reusing bench from a different branch — bench = f(code, net,
  tunables); changes on any axis change the number

### 2.2. The net-override case (CRITICAL — 2026-05-23 incident)

If your SPRT will use `--dev-network <SHA>` or `--base-network <SHA>`
to override the embedded net at play time, **OB's bench check uses the
OVERRIDE NET at bench time too** — not the embedded prod net.

So:

```bash
# Default case (no override, embedded prod net at play):
make && ./coda bench                              # this is your dev_bench

# Override case (--dev-network <SHA> at play):
make && ./coda bench -n /path/to/override-net.nnue  # this is your dev_bench
```

**Concrete example** (same binary, different nets):
- `./coda bench` → 4,963,169 (with embedded prod net)
- `./coda bench -n nets/fenskip-net.nnue` → 5,722,832 (with override net)

If you submit `dev_bench=4,963,169` with `--dev-network <fenskip-SHA>`,
OB benches with the fenskip net, gets 5,722,832, rejects as Wrong Bench.

**Paired-probe (both sides override)**: bench each side with its
respective net. dev_bench uses dev-network net; base_bench uses
base-network net.

**BENCH IS BINARY-SPECIFIC — re-bench on the CURRENT binary; don't reuse old
values (2026-06-22).** A net-override bench is only valid for the exact binary
it was measured on. When main moves (any search change — e.g. an LMP merge — or
a branch's tunables), the **same net benches to a different node count**. The
OB error line `[Coda-<BINHASH>-<NETHASH>] Wrong Bench: <N>` reports N **for that
BINHASH** — a value from a *different* binhash is useless to you.

Failure mode to avoid (I hit it): attempt-1 worker reported `Wrong Bench:
3430524` (binary `35E03F9F`); I reused `3430524` for attempt-2's binary
(`e6dac99`, post-LMP) where the correct value was `3770303` — then invented a
"Hercules ISA outlier / fleet-split" story to explain the mismatch. There was
no ISA issue: workers are consistent **within a binhash**; I was comparing
values across two binaries. (See memory
`feedback_dont_blame_fleet_rebench_current_binary`.)

**Workflow for net-vs-net `--dev/--base-network`:**
1. `git rev-parse HEAD` == `origin/main`, `make`, then `./coda bench -n <net>`
   for EACH side **right before submitting** — trust those fresh numbers.
2. If it Wrong-Benches, **check the error's `<BINHASH>` matches your local
   build** before anything else. If main moved, rebuild + rebench; do NOT paste
   a worker value from a stale binhash.
3. Don't reach for ISA/fleet explanations — that's ~99% a miscalc.

### 2.3. Branch-state awareness — pull before bench

**OB workers pull the latest of the specified branch at build time.**
If your local copy of the branch is stale, your bench number won't
match what OB measures.

**Required pre-bench discipline:**
```bash
git checkout <branch>
git pull --ff-only          # MUST do this before benching
make && ./coda bench        # now the bench reflects what OB will build
```

This applies to:
- **`main`** — other Claudes (Atlas/Titan/Zeus) push to main
  asynchronously. Always pull before benching main for an OB submission.
- **Shared experiment branches** — same hazard.
- **Your own feature branch** — less risk if only you push, but
  doesn't hurt to pull.

**Alternative: pin to a specific git commit SHA in the submission.**
`ob_submit.py` accepts a commit SHA in the `--base-branch` position
(and OB resolves dev_branch to a SHA on submit). Optically, branch
names are nicer in the OB UI than SHAs, so prefer branches when
the branch state is reliable; use SHAs when you need to lock the
exact build (e.g. running multiple SPRTs against the same dev state
while main may move under you).

After `git checkout <branch>`:
- **Always rebuild before benching**: `make && ./coda bench`. Stale
  binaries from a different branch return the wrong number.
- **Compare your branch base vs `origin/main`** if you want to be
  sure your branch isn't behind on a fast-moving trunk:
  ```bash
  git merge-base <branch> origin/main      # should match origin/main HEAD if rebased
  git rev-parse origin/main
  ```
- **If you get Wrong-Bench errors after submission**: rebase your
  branch against current `origin/main`, force-push, rebuild,
  re-bench, re-submit.

### 2.4. Bench delta as retune signal

A **>15-20% bench delta** from the baseline branch signals a real
tree-shape change. The trunk's tunables were calibrated for the
baseline's tree shape; the new branch's tunables may be miscalibrated.
**Queue a retune-on-branch SPSA before deciding on the change**.

(See §4 for retune-on-branch workflow.)

---

## 3. Submitting SPRTs

### 3.1. Standard SPRT command

```bash
OPENBENCH_PASSWORD=$PW python3 scripts/ob_submit.py <dev_branch> <dev_bench> \
  --base-bench <base_bench> \
  --bounds '<bounds>' \
  [--dev-network <SHA>] [--base-network <SHA>]
```

- `<dev_bench>` matches what `make && ./coda bench` returns on
  the dev branch (or `./coda bench -n <override-net>` if using
  net override)
- `<base_bench>` same for the base side (default: main)
- `--base-branch <name>` if base is not main (e.g. `mini-prod`
  for S200 paired probes)
- **Reference NPS / `--scale-nps`**: OB scales time budgets off a reference
  NPS, default **250000 (v9 prod)** — correct for current work, so omit it.
  Explicit **v5-only** work (legacy `bullet_convert` experiments, v5 net
  comparisons) MUST pass **`--scale-nps 500000`**. Wrong scale_nps → wrong
  time budgets: v9 code run at 500000 takes ~2× wall-clock, halving fleet
  throughput. (Same flag applies to `ob_tune.py`.)

### 3.2. Bounds policy — DO NOT improvise

**ALWAYS range-3 (width = H1 − H0 = 3). There is no range-6 option.**

| Change class | Bounds | Rationale |
|---|---|---|
| **DEFAULT — "does this feature/change help?"** | **`[0, 3]`** | Coda standing policy. Most ideas target +1-3 Elo. |
| Direction GENUINELY uncertain | `[-1.5, 1.5]` | Net-vs-net / alt-net compare, a correctness fix that could go either way. Centered, range-3. |
| Ship if not a meaningful regression | `[-2, 1]` | Bench-neutral refactor, NPS-only, ARM ordering, new tunables at default, correctness bundle. |

**Default is `[0, 3]`. Do NOT use range-6+ bounds** — `[-3, 3]`, `[-5, 5]`,
`[0, 5]`, `[0, 10]`, `[-3, 0]`, `[-10, 5]` all need far more games AND routinely
return without a meaningful signal (H0 on a real small effect, or never
separate). A true +5 effect H1s at `[0, 3]` faster than `[0, 5]`; a true +1.5
effect H0s at `[0, 5]` while H1ing at `[0, 3]`. Tightening is never wrong. For
uncertain-direction use `[-1.5, 1.5]` (NOT `[-3, 3]`); for non-regression use
`[-2, 1]`. (Matches CLAUDE.md §SPRT Testing Policy and memory
`feedback_sprt_bounds_range_3_never_minus3_3`.)

### 3.3. Paired-probe net-vs-net (mini-prod or main + override)

For S200 candidate testing or prod-replacement testing:

```bash
# S200 paired probe on mini-prod branch (both sides build mini-prod):
ob_submit.py mini-prod <dev_bench_with_dev_net> \
    --base-branch mini-prod --base-bench <base_bench_with_base_net> \
    --dev-network <CANDIDATE_SHA> --base-network <baby_prod_SHA> \
    --bounds '[0, 3]'

# Prod-replacement on main (both sides build main, different nets):
ob_submit.py main <dev_bench_with_dev_net> \
    --base-bench <base_bench_with_base_net> \
    --dev-network <CANDIDATE_SHA> --base-network <PROD_SHA> \
    --bounds '[0, 3]'
```

**Both benches MUST be measured with the corresponding net loaded
via `-n`**. See §2.2.

### 3.4. Priority — same for concurrent tests

OB workers all go to the highest priority test. If you submit two
tests with different `--priority` values, all workers pick the
higher-priority one and the lower one gets zero workers. **Use the
same `--priority` for concurrent tests** (default: 0). Vary priority
only when you explicitly want one test to drain workers first.

### 3.5. Always: 5-min /errors/ check after submit

Workers report Wrong-Bench and other errors **asynchronously** —
the test page shows "ACTIVE 0 games" and looks like it's just waiting.

**Use `scripts/ob_errors.py`, do NOT eyeball the raw page.** The HTML
stores each error's time as a bare Unix epoch in a `<td class="timestamp">`
cell; grepping the summary strings (`curl … | grep "Wrong Bench"`) strips
the Date/Test columns, so a 26-hour-old error and a live one look
identical. Claudes repeatedly mis-read stale errors as current this way.
The script keeps the row intact (AGE + time + test id + branch + summary)
and time-filters by default.

```bash
# 5 minutes after every ob_submit.py — clean window only:
python3 scripts/ob_errors.py                 # errors in the last 6h (default)
python3 scripts/ob_errors.py --test <id>     # did THIS test error? (any age)
python3 scripts/ob_errors.py --user claude   # only my rows
# exit code 1 if any error matched (gate scripts on it); 0 if clean.
```

Each row carries its full **Summary** (e.g. `[Coda-D13549A4] Wrong Bench:
2944948` — the second number is the bench OB measured, which is what you
need to fix a Wrong-Bench). Build-fail rows also carry a `[log <id>]`
hint; pull the actual cargo output (the real error is at the *tail*) with:

```bash
python3 scripts/ob_errors.py --log <event_id>            # last 80 lines
python3 scripts/ob_errors.py --log <event_id> --tail 0   # whole log
```

This is how you distinguish e.g. a generic "main build failed" from the
specific cause (bzip2-sys `cc`→`cargo` C-compile break, a missing
feature flag, a Rust compile error) without guessing.

The faster real-time signal is **game accrual**: if `ob_status.py` shows
the test gaining games, it cleared the bench gate — a Wrong-Bench never
produces games. Treat "0 games for >5 min AND a matching fresh row in
`ob_errors.py`" as the actual error condition.

If errors appear, **investigate and fix before letting the test sit**.
Common causes:
- Wrong Bench → see §2 for diagnosis
- Build failure → fix and force-push the branch
- Disconnects → may be transient worker issues; retry resubmit if persistent

### 3.6. Wrong-Bench is almost always submitter-side staleness, NOT a bad worker

When a Wrong-Bench summary shows a build hash like `[Coda-654C354A]`, that
8-char tag is the **git commit short-SHA OB built** — frequently it IS the
current `origin/main` HEAD. If multiple workers report the *same* wrong
bench across *different* branches, that is **not** a broken worker; it is the
**base** being built from a `main` that is ahead of your stale local `main`,
so the base-bench you declared (from your stale local build) doesn't match.
Don't blame a worker. Fix it submitter-side:

```bash
git fetch origin && git checkout main && git pull   # get true HEAD
make net && make && ./coda bench                     # the REAL base bench
```

**Base is PINNED at submit time; mid-run main drift does NOT confound a
running test.** OB records the base branch's commit when you submit; every
worker (including ones that join later) builds *that same pinned base
commit*, not whatever `origin/main` is now. So a test whose dev branch was
forked from the then-current main and submitted with the correct base bench
is fine for its whole life even if main advances afterwards — **do NOT stop
it for "base drift".** (Learned the hard way 2026-06-24: stopped a healthy
+1-Elo test for this non-reason.)

The genuinely-needs-rebase case is different: if your **local main was
already stale when you forked/benched** (origin had advanced), then (a) your
declared base bench is wrong → real Wrong-Bench rejection, and (b) your dev
branch is missing the commits main gained → the dev-vs-base delta is
confounded by those commits. Fix = pull main, **rebase the branch**, rebuild,
re-bench, force-push, resubmit. The tell for *which* case you're in: did
`origin/main` move **before** you forked (→ rebase) or **after** you
submitted (→ leave it running)?

---

## 4. Submitting SPSA tunes

### 4.1. Standard tune command

```bash
# Full-sweep (auto-derived from `./coda tune-spec`):
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune.py <branch> [bench] \
    [--no-core] [--iterations 2500] [--dev-network <SHA>]

# --core curated subset (default — usually what you want):
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune.py <branch> [bench] \
    --iterations 2500 [--dev-network <SHA>]

# Focused cluster (use --params or --params-file for hand-picked subset):
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune.py <branch> [bench] \
    --params-file scripts/tune_<cluster>.txt --iterations 1500
```

### 4.2. CRITICAL: --dev-network must match net.txt prod

Trunk tunables and the deployed prod net must stay calibrated
together. If you fire a trunk retune with `--dev-network` pointing
at a DIFFERENT net than what's in `net.txt`, the tune calibrates for
a net that's not deployed → silently detuned trunk.

**Before every tune submission**:
```bash
cat net.txt                          # check current prod net filename
# Confirm the SHA8 of that file matches the --dev-network you'll pass.
# If they don't match, DON'T SUBMIT — fix the mismatch first.
```

Exception: retune-on-branch for a candidate-net experiment (fenskip,
L1-widening, etc.) intentionally uses the CANDIDATE net SHA as
`--dev-network`. This is fine because you're calibrating tunables
for that specific net.

### 4.3. Iteration sizing

| Purpose | Iterations | Notes |
|---|---|---|
| Focused cluster (4-8 params) | 1000-1500 | High SNR per param |
| --core retune (53 params) | 2500 | Standard |
| Full-sweep (84 params) | 2500-3000 | Higher risk of loose-knob contamination — see §4.4 |
| Production lock-in | 10000+ | Only for stable trunk + net |

**Empirical**: 1000-1500 iter is the productive zone for most tunes
regardless of param count. Longer (5K+) only with explicit basin-
exploration motive.

### 4.4. Loose-knob contamination — the full-sweep hazard

Full-sweep tunes (`--no-core`) include all ~84 tunables. ~30+ of
them are "loose knobs" — gradient signal noise-dominated. Loose
knobs **don't just add noise; they flip gradient signs on adjacent
prune-direct params**, sometimes producing big-magnitude movements
that look like signal but aren't.

**Default to `--core`** (the 53-param curated subset). Only run
full-sweep when:
- You expect a meaningful basin shift (post-net-swap, post-major-refactor)
- You're willing to accept noise on the non-core params
- Iter budget is at least 2500

If full-sweep shows wild magnitudes on core params (e.g.
`CONT_HIST_MULT` drifting to 0, `IIR_MIN_DEPTH` jumping 4× from
default), **stop the tune** — that's contamination, not signal.
Continue burns fleet cycles.

### 4.5. UHO book is the default

`UHO_Lichess_4852_v1.epd` — produces ~2.5× more decisive games per
iteration than `4moves_noob.epd`. SPSA gradient signal proportionally
stronger.

`ob_tune.py` defaults to UHO since 2026-05-16. Don't override unless
reproducing pre-2026-05-16 SPSA outputs.

### 4.6. Stale tune outputs

SPSA outputs are calibrated against the trunk + net state at
submission time. After ANY meaningful change to trunk (new tunables,
demoted tunables, default-value shifts, search refactors), old tune
outputs are stale. Don't apply stale outputs verbatim — they
optimize for parameter relationships that no longer hold.

---

## 5. Stopping a test or tune

### 5.1. The URL gotcha

- SPRTs live at `/test/<id>/`
- Tunes live at `/tune/<id>/`
- The STOP endpoint must be uppercase: `/test/<id>/STOP/`
- `ob_stop.py` tries `/tune/` first then `/test/` (fixed 2026-05-22)

### 5.2. Always verify before stopping

```bash
# Confirm test ID + branch name + nets before stopping:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_status.py | grep <test_id>

# Confirm tune ID + branch:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune_status.py <id>

# THEN stop:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_stop.py <id>
# The script verifies the test went inactive; trust the verification.
```

Don't stop another user's tests without explicit instruction —
filter `ob_status.py` output by user if unsure.

### 5.3. When to stop early (vs let it resolve)

| Situation | Action |
|---|---|
| LLR hitting H1 boundary (≥ 2.94) | Test resolves itself; no action needed |
| LLR hitting H0 boundary (≤ -2.94) | Same |
| Early-N noise (CI ± large, < 500 games) | **Don't stop**; per `feedback_wait_500_games` |
| LLR trending H0 + CI upper bound below elo1 + N very large | **Stop** to save fleet (e.g. 95% CI upper < elo1 at 100k+ games) |
| Bench-errored | **Stop immediately**; resubmit with correct bench |
| Wrong direction at meaningful N (e.g. -30 ±5 at 5k games, [0, 3]) | OK to stop — won't recover, save cycles |

**Don't stop on LLR fade alone** — fade in either direction at low
N is just measurement noise. Wait for CI resolution or LLR boundary.

---

## 6. Reading results

### 6.1. Early-N drift caveat

Same-binary SPRT (main vs main) at N=554 produced +9.4 ±9.75 Elo —
just from opening-pair luck. Don't cite Elo numbers below N=500.
Don't construct mechanistic narratives from early-N point estimates.

```
N=500   → CI ≈ ±10 Elo (at 10+0.1)
N=10k   → CI ≈ ±2.5
N=30k   → CI ≈ ±1.5
N=60k   → CI ≈ ±1.0
```

At LTC 40+0.4, roughly double these CIs for same N.

### 6.2. Reading CIs

- A "trending H1" result with CI containing zero is **not evidence**.
- A result whose CI excludes zero AND is on the right side of elo1
  is the signal.
- Overlapping CIs across runs = "same distribution seen twice", NOT
  "earlier was noise".

### 6.3. H0 interpretation

- `H0 at [0, 3]` = "failed to prove ≥ +3 Elo", NOT "confirmed ≤ 0 Elo".
- `H0 at [-3, 3]` = "ruled out both extremes", a stronger result.
- Per change class:
  - **Parameter probe H0** → BISECT (try midpoint of the explored range)
  - **Structural port H0** → find asymmetry between port and reference engine
  - **Net comparison H0** → check bench-stats for conversion correctness; consider retune-on-branch

### 6.4. Worker variance

OB workers have a 50+ Elo spread on perf-sensitive changes (SIMD,
cache layout). For perf-sensitive SPRTs, pull per-worker breakdowns
via the OB UI. For pure tunable-value diffs, worker variance is
secondary; SPRT stopping-point bias matters more.

---

## 7. Applying tune outputs

```bash
# Pull outputs — prefer the RAW API (the `--outputs` CLI drops the _10X
# cluster and other params; the raw endpoint is complete):
curl -s "https://ob.atwiss.com/api/spsa/<tune_id>/outputs/" > /tmp/tune-N.txt

# Apply with the COMMITTED, self-validating tool (checkout the branch first):
git checkout -b experiment/<model-descriptor>-tuned
make                                   # tune-spec reads the live macro
python3 scripts/apply_tune.py /tmp/tune-N.txt   # --dry-run to preview

# Sanity:
make && ./coda bench    # confirm builds + get new bench number
# Commit + push + SPRT at [0, 3] vs main (or vs pre-tune trunk for retune-on-branch).
```

**ALWAYS use the committed `scripts/apply_tune.py` — never an ad-hoc
`/tmp` script.** Uncommitted apply scripts drift out of sync and have
silently mis-applied tunes (a hardcoded NMP-cluster skip-list once
detuned the trunk with no error; the SPRT then runs on wrong values
and the result is meaningless). `scripts/apply_tune.py` is
self-validating: it cross-checks every input param against the live
`./coda tune-spec` macro, requires exactly one match per param, and
HARD-ERRORS (non-zero exit, no write) on any unknown/renamed/ambiguous
param instead of skipping it. It replaces default values in the
`tunables!(...)` macro only — it does NOT modify macro structure. It
accepts both the raw `NAME, value` format and the full SPSA spec.

---

## 8. Quick command reference

```bash
# Bench-for-OB (default case) — ALWAYS pull first:
git pull --ff-only && make && ./coda bench

# Bench-for-OB (override-net case):
git pull --ff-only && make && ./coda bench -n /path/to/override-net.nnue

# Upload a net:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_upload_net.py /path/to/net.nnue

# Submit SPRT [0, 3] default bounds:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_submit.py <branch> <bench> \
    --base-bench <base> --bounds '[0, 3]'

# Submit paired-probe SPRT (S200 candidate vs baby-prod):
OPENBENCH_PASSWORD=$PW python3 scripts/ob_submit.py mini-prod <dev_bench> \
    --base-branch mini-prod --base-bench <base_bench> \
    --dev-network <CANDIDATE_SHA> --base-network <BASE_SHA> \
    --bounds '[0, 3]'

# Submit --core SPSA tune:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune.py <branch> [bench] \
    --iterations 2500 --dev-network <SHA-matching-net.txt>

# Status:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_status.py             # all SPRTs
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune_status.py        # all tunes
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune_status.py <id>   # specific tune
OPENBENCH_PASSWORD=$PW python3 scripts/ob_tune_status.py <id> --outputs  # tune outputs

# Stop:
OPENBENCH_PASSWORD=$PW python3 scripts/ob_stop.py <id>

# Errors (check 5min after every submit) — parses the epoch timestamps so
# stale rows don't read as current; do NOT grep the raw /errors/ HTML:
python3 scripts/ob_errors.py                  # last 6h (default)
python3 scripts/ob_errors.py --test <id>      # one test, any age
python3 scripts/ob_errors.py --hours 48 --limit 20   # wider window
python3 scripts/ob_errors.py --log <event_id>        # build log tail (real cause)
```

---

## 9. Common failure modes — quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Wrong Bench errors | Bench mismatch | §2 — rebuild via make; if using --dev-network, bench with `-n <override-net>` |
| Wrong Bench after trunk moved | Stale branch base | Rebase against `origin/main`, rebuild, re-bench, re-submit |
| Test active but 0 games | Worker errors | Check `/errors/` page — usually Wrong Bench or build failure |
| Tune outputs don't match expected scale | Stale post-trunk-change | Don't apply old tune outputs verbatim — re-run tune |
| Full-sweep tune wild movements | Loose-knob contamination | Stop tune, switch to `--core` |
| SPRT trending H1 then fading | Early-N drift | Don't act on N < 500 |
| Stop didn't actually stop | Wrong URL or test owned by another user | `ob_stop.py` should handle URL; check ownership |
| Disconnects on multiple workers | Worker-side connectivity | Usually transient; check `/errors/` for recurrence pattern |
| "Abandoned" games with PlyCount=0 | Engine panic at startup (rare) | Check worker stderr for Rust panic; e.g. div-by-zero or SIGPIPE — fix in code, not in workflow |

---

## 10. When to update this skill

When you discover a new OB gotcha or workflow change, update this
file directly. Do NOT save it as a per-Claude memory — local
memories don't propagate across Hercules/Atlas/Titan/Zeus. The
skill is the canonical reference.

For Coda-specific methodology that isn't strictly OB-mechanical
(e.g. retune-on-branch reasoning, post-merge SPRT discipline),
either:
- Update CLAUDE.md (durable, project-wide)
- Update this skill if it's an OB-operational detail
- Save as a memory ONLY for truly per-instance / per-session
  observations (rare)
