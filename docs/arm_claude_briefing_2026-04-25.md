# ARM Claude Briefing — first M5 pass on Coda

**You are**: Claude running on Adam's M5 MacBook (aarch64).
**I am**: Hercules — Claude on the x86 dev box / OpenBench fleet, who
wrote this note. Don't confuse our work.

**Why you're here**: As of 2026-04-25 ARM is a first-class supported
CPU family for Coda (commitment in `CLAUDE.md` preamble + memory file
`project_arm_first_class.md`). The full sweep tracker is
`docs/arm_correctness_2026-04-25.md`. I've audited atomics from x86
and queued items only an aarch64 box can validate. That's you.

## Ground rules

- **Branch for any code changes.** Don't push to `main`. Don't merge.
- **You can't SPRT** — fleet is x86. Use `cargo test`, perft, EPD,
  bench, and reasoning. Report findings; I'll handle SPRT validation
  on x86 if you make a code change.
- **If you find a bug, fix it on a branch and push the branch.**
  Then ping in a note for me to SPRT-validate (`[-5, 5]`
  non-regression at minimum) before merging.
- **Don't run training jobs.** GPU work belongs on cloud hosts.
- **Coordinate via the repo + memory** — if you save a memory note
  prefix it `arm_*` so I can tell ARM-side work apart from mine.

## Priority 1 — build & test sanity (15 min)

```bash
cd ~/code/coda  # or wherever you cloned
git fetch origin
git checkout main && git pull
cargo build --release           # plain build
cargo test --release            # full test suite
./target/release/coda perft-bench   # 6 perft positions
./target/release/coda bench 13      # search bench, print NPS
```

**Report back**: do all tests pass? Does the perft-bench print all
expected node counts? What's the bench NPS? Any compile warnings?

This validates that our x86-developed code at least *runs* on
aarch64 — and gives us a NPS calibration data point we don't have.

## Priority 2 — NEON SCReLU parity test (the durable fix from audit C4)

The audit (`docs/arm_correctness_2026-04-25.md` §NEON) noted that
commit `44baa95` once silently shipped a `>> 9` vs `>> 8` shift bug
in `neon_screlu_pack` that halved SCReLU activations on aarch64
v7 non-pairwise nets. No automated test would have caught it.

Your job: write a `#[cfg(target_arch = "aarch64")]` cargo test that
calls `unsafe neon_screlu_pack` and compares its output to the
scalar reference (loop in the function tail, lines around
`nnue.rs:1725-1729`) at production widths `h ∈ {16, 32, 64, 128,
256, 512, 1024, 1536}`. Bit-identical for the SIMD section, exact
match in the scalar tail. Same parity test for `neon_pairwise_pack`
(line ~1735) — that one was the one *intentionally* changed in the
same commit, so its scalar reference also lives in nnue.rs near it.

Put the test in `src/nnue.rs` under the existing `#[cfg(test)]
mod tests` block. Branch name suggestion: `arm/neon-parity-tests`.

**Report back**: do the tests pass at all widths? If they fail
anywhere, that's a real bug — describe it before fixing.

## Priority 3 — WAC EPD comparison vs x86

WAC (Win At Chess) is 201 tactical positions. Same net + same
search depth = same set of positions solved on x86 vs aarch64
**unless** there's a NEON eval bug or other aarch64-specific
divergence. This is the cheapest cross-arch correctness check we
have.

```bash
git checkout main && cargo build --release
./target/release/coda fetch-net  # downloads prod net to net.nnue
./target/release/coda epd testdata/wac.epd 1000 0 -nnue net.nnue
```

I'll run the same command on x86 and we compare. The output is
"solved N / 201" plus per-position move + time. Diff the
"solved-vs-not" set — any position solved on one arch but not the
other is a divergence worth investigating (likely NEON bug, but
could also just be NPS-driven horizon difference).

**Report back**: how many of 201 solved? Save the full output to
`docs/arm_wac_run_<date>.txt` and push the file so I can diff.

## Priority 4 — `make openbench`

```bash
git checkout main
make openbench      # OB-compatible build
```

We want to know whether an ARM OpenBench worker is feasible. If
`make openbench` produces a binary that can respond to `uci` /
`bench` correctly, we're good for the eventual ARM-OB-worker.

**Skip `make pgo`** — it doesn't work for v9 even on amd64, so
ARM is not the place to debug it.

**Report back**: did `make openbench` succeed, and does the
resulting binary pass `bench`?

## What to do with findings

- Bug found + you have a fix → branch, push, leave a note in
  `docs/arm_findings_<date>.md`. I'll SPRT on x86 and merge.
- Bug found, no fix yet → memory note `arm_<topic>.md` + a one-line
  pointer in `MEMORY.md`. I'll pick it up.
- Everything passes → memory note documenting the calibration data
  (bench NPS at T=1, perft pass, etc.) + tell me ARM is healthy.

## Scope reminder

This is a **first pass** — sanity + the one durable test (NEON
parity). Don't try to close every item in
`arm_correctness_2026-04-25.md` in one session. The full sweep is
multi-week.

If anything in this brief is unclear, ask before starting. The
context cost of asking is much lower than the cost of doing the
wrong test on the wrong branch.

— Hercules
