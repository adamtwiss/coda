# ARM (aarch64) as First-Class Supported CPU Family

**Commitment 2026-04-25 (Adam):** Coda needs to run well on ARM. Apple is
100% ARM, ARM is making inroads into the compute market beyond that, and
Adam's primary dev laptop is M5. ARM is now a first-class supported
target, not an afterthought.

This doc tracks the ARM-correctness sweep and remaining follow-up.

## Why this matters now

- **Lichess deployment** — codabot runs on x86 today, but moving to
  Graviton or similar would unlock cheaper compute for analysis games.
- **CCRL** — historically x86 but ARM machines (Apple Silicon) are
  increasingly used by testers. Wrong-result on ARM = bad CCRL games.
- **Dev experience** — Adam's M5, future M-series Macs, are the
  primary development machines. Slow or buggy on ARM = friction every
  day.
- **Future-proofing** — competitive engines need to target the chips
  developers actually use.

## What's done

### TT + tb_cache memory ordering (2026-04-25)

Branch: `fix/aarch64-tt-tbcache-ordering` (commit `54586b5`)
SPRT: #764 [-5, 5] non-regression

Fixed: both TT and tb_cache used `Ordering::Relaxed` for the data+key
XOR-validated lock-free pattern. On x86 this works because of strong
memory ordering. On ARM, Relaxed permits the writer's two stores to be
observed in either order on another core — torn reads escape the
XOR check at ~1-in-2³² rate per slot.

Now: stores use `Ordering::Release`, loads use `Ordering::Acquire`.
Probe-side load order flipped to load key first (Acquire), then data,
so the synchronization edge carries from key.store(Release) →
key.load(Acquire) → data.load.

x86 cost: ~zero. aarch64 cost: small per-load `dmb` barrier.

## What's queued

### Other shared atomics — needs audit

**`info.stop`** (AtomicBool, Relaxed). Writers: UCI thread on `stop`,
search worker on time-out. Readers: all search threads. Currently
Relaxed which is *probably* correct (no associated data dependency),
but worth verifying any "publish best_move then stop" patterns are
ordered correctly. Audit candidate.

**`info.global_nodes`** (AtomicU64, Relaxed). Writers: all search
threads via `fetch_add`. Readers: UCI for output. `fetch_add` provides
read-modify-write atomicity but no ordering. For node-count display,
Relaxed is fine. Confirmed clean.

**`info.ponderhit_time`** (AtomicU64, Relaxed). Writer: UCI on
ponderhit. Readers: search threads checking the budget. Read once per
~4096 nodes. The published value should propagate eventually; Relaxed
is correct because there's no associated data needing happens-before.

**`tt.generation`** (AtomicU8 or U16, Relaxed). Writer: main on
advance_generation. Readers: all storage paths. Stale reads cause
slightly suboptimal replacement but never corruption. Relaxed is
correct.

**Per-thread `SearchInfo.stats`** (atomic counters). Single-threaded
per `SearchInfo`, so ordering doesn't matter.

### NEON SIMD path for NNUE

The NEON SCReLU `>>9` vs `>>8` mismatch was already fixed (audit C4,
2026-04-22). Need to verify:

- [ ] NEON path is exercised by `cargo test` (CI catches AVX2-vs-NEON
      drift)
- [ ] NEON pack/unpack tail handling matches AVX2 at h ∈ {16, 32, 64,
      ..., 1536}
- [ ] CReLU and SCReLU path bit-identical to scalar reference at all
      production widths

### Build pipeline

- [ ] Verify `make` on aarch64 produces working binary
- [ ] Verify `make pgo` works (or document that PGO is x86-only)
- [ ] Verify `make openbench` produces correct OB-build on ARM (if we
      ever add ARM workers)

### Testing

- [ ] Run full test suite on M5 — `cargo test --release`
- [ ] Run perft suite on M5 — `coda perft-bench`
- [ ] Run WAC EPD on M5 — `coda epd testdata/wac.epd`
- [ ] Optional: add a CI job for aarch64 (cross-compile or QEMU)
- [ ] Lichess-watch on M5-deployed Coda to compare play quality vs x86

### OpenBench fleet

Adding even one ARM worker to the fleet would let us SPRT-detect ARM
correctness regressions. Currently impossible to verify ARM Elo on the
fleet. Consider if/when OB-on-ARM is worth setting up.

## Coding standard for new code

When adding shared atomics that back any reader-publish pattern:

- **Store ordering**: use `Release` on the "publish" atomic — the one
  whose update marks the data as ready.
- **Load ordering**: use `Acquire` on the "publish" atomic — readers
  that observe the published flag must see all data writes that
  happened-before the publish.
- **Order of stores/loads**: writer publishes after writing data;
  reader checks publish before reading data. Both sides use program
  order to chain the happens-before edge.
- **Counters / stats**: `Relaxed` is fine if there's no
  associated data dependency.
- **Stop flags**: `Relaxed` is fine if there's no "published-data
  comes with the stop" requirement.

When in doubt, use `Acquire/Release`. The x86 cost is negligible; ARM
correctness is real.

## Reference

- TT implementation: `src/tt.rs:370-462`
- TB cache implementation: `src/tb_cache.rs:97-145`
- Audit source: `docs/correctness_audit_2026-04-22.md` SPECULATIVE
  list (now resolved)
