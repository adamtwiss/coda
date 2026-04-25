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

### TT + tb_cache memory ordering (2026-04-25) — MERGED

Branch: `fix/aarch64-tt-tbcache-ordering` (commit `54586b5`)
SPRT #764 [-5, 5] non-regression: **−0.1 ±1.9 / 24886g** (LLR −0.40,
stopped at fade). x86 cost confirmed ≈ 0; merged to main 2026-04-25.

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

## Other shared atomics — audit done 2026-04-25

Audited all `Atomic*` uses outside `tt.rs` / `tb_cache.rs`. **All
remaining atomics are correct with `Relaxed`.** The TT/tb_cache
XOR-validated reader-publish pattern was the only one needing
`Acquire/Release`.

| Atomic | Site | Pattern | Verdict |
|---|---|---|---|
| `tt.data[i]`, `tt.keys[i]` | tt.rs | XOR-validated reader-publish | ✅ Acquire/Release on branch (SPRT #764) |
| `tb_cache` slot fields | tb_cache.rs | XOR-validated reader-publish | ✅ Acquire/Release on branch (SPRT #764) |
| `tt.generation` | tt.rs:362 | Replacement-policy hint | ✅ Relaxed — stale reads only degrade replacement quality, never corrupt |
| `info.stop` | search.rs, uci.rs | Termination flag, no data publish | ✅ Relaxed — best-move is conveyed via `thread::join`, not via shared memory ordered against `stop` |
| `info.global_nodes` | search.rs:579 | `fetch_add` counter | ✅ Relaxed — RMW atomicity is what matters, display tolerates stale |
| `info.ponderhit_time` | uci.rs:444, search.rs:589/1392/1643 | UCI publishes deadline, search reads | ✅ Relaxed — self-contained `u64`, no other data published with it |
| `info.ponder_depth` | search.rs:1483 | Search publishes depth scalar | ✅ Relaxed — self-contained, no companion data |
| `FEAT_*` AtomicBools | search.rs:211-232 | Init from env, read in search | ✅ Relaxed — init runs before `thread::spawn`, which carries happens-before |
| Tunable `AtomicI32`s | search.rs `tunables!` macro | UCI `setoption` publishes value, search reads | ✅ Relaxed — scalar reads, no associated state |
| `LMR_TABLE` / `LMR_TABLE_CAP` | search.rs:916-917 | `static mut` array rebuilt by `init_lmr` on `setoption LMR_C*` | ✅ Safe — `setoption` is single-threaded between games (UCI never overlaps a live search), and `thread::spawn` to start the next search establishes happens-before |
| `threat_profile.rs` static counters | threat_profile.rs:6-18 | Pure stats | ✅ Relaxed — sums tolerate any order |
| `datagen` counters + `line_idx` | datagen.rs | Counters and `fetch_add` work-claim | ✅ Relaxed — atomicity is the only requirement |
| `eval.rs DBG`, `search.rs VERIFY_*` | eval.rs:192, search.rs:634-635 | Debug counters | ✅ Relaxed |

**Methodology**: for each atomic, identified writers / readers / what
data (if any) the atomic gates. The only correctness-relevant case is
"writer publishes flag, reader observes flag then loads associated
data" — that pattern requires `Release/Acquire` on aarch64. Counters,
self-contained values, and one-shot init are all `Relaxed`-safe.

**Note on `LMR_TABLE` `static mut`**: technically a Rust data race
(non-atomic shared write), but practically safe because `setoption`
arrives between games during SPSA and tournaments — never concurrent
with a running search. If we ever change UCI to apply `setoption`
mid-search, this becomes a real race. Tracked as a watch item but not
fixed today.

### NEON SIMD path for NNUE

The NEON SCReLU `>>9` vs `>>8` mismatch was already fixed (audit C4,
2026-04-22). Background: SCReLU outputs `clamp(v, 0, 255)²` which
lives in `[0, 65025]`. To pack back into `u8` for the int8 L1 matmul,
we divide by ~255. The exact divide-by-255 is expensive in SIMD, so
we use `>> 8` (divide by 256) — error is ~0.4%, absorbed by training.
Commit `44baa95` accidentally bumped the NEON path to `>> 9` (divide
by 512) alongside an intentional `neon_pairwise_pack` change, halving
SCReLU activations on aarch64. Affected v7 non-pairwise SCReLU nets
only; no aarch64 SCReLU-vs-scalar regression test caught it. Now
fixed at `nnue.rs:1717`/`1727` with a comment cross-referencing the
x86 path. **Lesson queued**: add a `neon_screlu_pack` parity test
against the scalar tail at all production widths so a future shift
mismatch is caught at `cargo test`.

Need to verify:

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
