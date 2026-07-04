# Threat-pipeline Phase A: Coda-semantics SIMD threat enumerator (2026-07-04)

**Decision (Adam, 2026-07-04): GO on Option B** from
`docs/byteboard_splat_scoping_2026-05-03.md` — a custom SIMD threat-delta
enumerator that preserves Coda's x-ray threat semantics. Option C (retrain
to Reckless's direct-only threat space) is **rejected permanently**: x-ray
threats are worth 150+ Elo and are Coda's most powerful unique feature.
Any implementation that changes the feature space is wrong by definition.

## Fresh profile support (2026-07-04, current main incl. hugepage #2493 + fused-kernel #2495 merges)

perf record, bench 14/15, core-pinned, OB workers stopped:

| Component | titan (AVX2, Zen 1) | Zeus (AVX-512, Zen 5) |
|---|---|---|
| `apply_threat_indices` (row apply — NOT this campaign's target) | 19.5% | 10.8% |
| `push_threats_for_piece` (scalar enumeration — target) | 5.3% | 6.2% |
| `ThreatStack::update_dual` (expansion/table-chase — target) | 3.2% | 6.1% |
| `refresh` + `piece_attacks_occ` (partial target) | 2.4% | 1.9% |
| **Addressable slice** | **~9%** | **~13%** |
| Whole threat pipeline | ~30% | ~25% |

Key insight: post-hugepage/post-fused-kernel, the *scalar*
enumeration+expansion is a relatively BIGGER share on AVX-512 hosts
(Amdahl — their SIMD parts got fast). The fleet's future and the main
lichess account's host (identical to Zeus) are AVX-512, so Phase A
targets AVX-512/VBMI2 first. Prize if the slice halves: **~4.5-6.5% NPS
fleet-wide ≈ +3-4 Elo STC**.

Explicitly out of scope for this campaign: `apply_threat_indices` (the
10-20% row-apply share) — irreducible per-row streaming; the only lever
that shrinks it is threat-accumulator WIDTH narrowing (training-side,
next net cycle, preserves the feature space).

## The May 2026 blocker (what makes this Option B, not a port)

Reckless's byteboard-splat emits deltas for a DIRECT-ONLY threat space.
Coda's feature space is direct-OR-x-ray under one feature index: a
slider's attack through exactly one blocker is the SAME active feature
as a direct attack. Blocker removal/arrival therefore does NOT toggle
the feature in Coda's model but DOES in Reckless's — a direct port
double-counts (verified by parity test, May 2026). The enumerator must
be redesigned around Coda's semantics: multi-pass ray resolution
(first-hit AND second-hit per ray), not single-pass closest-blocker.

## Phase A plan

1. **Semantic contract** — precise spec of the scalar enumeration
   (feature set, x-ray rule, semi-exclusion, per-move-shape delta
   contract, king-crossing refresh conditions, RawThreatDelta format,
   oracle strategy) → `scratchpad/threat_semantics_spec.md`, to be
   committed here once reviewed. IN PROGRESS (agent).
2. **Scaffolding resurrection** — the 649-LoC `src/threats_splat.rs`
   from commit 5096ac0 (Coda-encoded ray tables, AVX-512 primitives,
   parity test) rebased onto current main, gated behind
   `#[cfg(any(test, feature = "splat-dev"))]`, parity test preserved as
   an #[ignore]d diagnostic with mismatch categorization. IN PROGRESS
   (agent, branch `nps/threat-splat-phase-a`).
3. **Design** — the x-ray-aware SIMD algorithm against the contract:
   per ray, resolve first AND second occupied square (two
   closest-on-ray passes, the second with the first hit masked out);
   a feature (slider, from, victim, to) is active iff victim is the
   first hit OR the second hit. Delta = symmetric difference of
   before/after active sets per changed square. Design doc before code.
4. **Implementation** — behind the parity oracle: no code advances
   until the fuzz parity vs scalar is exact over random playouts
   (target: 100k+ positions, all move shapes, plus the
   threat_accum incremental-vs-refresh suite).
5. **Validation** — perf -r 3 on BOTH Zeus and titan before any SPRT
   (three single-host micro-win failures this week — see the
   2026-07-04 session scorecard in experiments.md); STC SPRT `[0,3]`
   (structural NPS with a real expected gain), fallback `[-2,1]` if
   the measured NPS is small.

Keep the scalar path permanently: it is the parity oracle, the
non-AVX-512 fallback until Phase B (AVX2), and the king-crossing path.

## Constraints (hard)

- Feature-space semantics must be bit-identical — any divergence is
  silent eval corruption (May's parity harness is the gate).
- Production dispatch must fall back to scalar on non-VBMI2 hosts.
- Zero warnings; all existing threat tests green; bench identical
  until the enumerator is switched on (the enumeration is
  output-identical by construction — bench must stay identical even
  after switch-on).
