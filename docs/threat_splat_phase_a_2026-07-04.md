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

## Design (2026-07-04, after contract + scaffolding + section profile)

Per-section generation cycles (bench 12, profile-threats, Zeus):
direct 357 Mcy (32%), own-xray/1b 163 (15%), sliders/2 178 (16%),
sliders-2b 235 (21%), nonsliders 178 (16%). The splat's parity-proven
direct pipeline covers only ~48%; x-ray sections are ~52% with 74-76%
zero-emit rates (mostly wasted scans — ideal SIMD-culling prey). So:
full unified enumerator, not a hybrid.

**Key insight — everything falls out of ONE focus-square ray frame with
TWO hits per ray.** `board_to_rays(focus)` gives the mailbox permuted
into 8 rays × 8 positions. `closest_on_rays` gives the FIRST occupant
per ray (Y_d for ray direction d). Mask those out and rerun the carry
trick → SECOND occupant per ray (Z_d). With (Y_d, Z_d) for all 8 rays,
every section's emissions are direct reads:

- **§1 direct-from** (focus attacks): Y_d per aligned ray (+ pawn/
  knight/king tables) — already in the splat.
- **§1b own-x-ray** (focus is slider): x-ray victim on ray d = Z_d
  (when Y_d exists) — emit (focus, Z_d).
- **§2 sliders-see-focus**: S = Y_d where S is a slider aligned with d.
  Its Z-delta target = the piece the slider hits BEYOND focus =
  Y_{-d} (first hit on the opposite ray); the depth-2 piece it loses/
  gains = Z_{-d}. Emit (S, Z_{-d}, !add)... precisely: Y = Y_{-d},
  Z = Z_{-d} per the contract's §3.3-2 (Y first past focus, Z first
  past Y). Plus the direct emit (S, focus, add).
- **§2b x-ray-onto-focus**: S = Z_d where Z_d is a slider aligned with
  d AND Y_d (the single blocker) exists. Emit (S, focus, add) and the
  W-delta (S, Y_{-d}, !add) — W is the first hit continuing away.
- **§3 non-sliders**: reverse table lookups — already in the splat.

The scalar code's per-candidate ray_extension chases and between()
popcounts all collapse into the two closest_on_rays passes. One new
SIMD primitive needed: second-hit-per-ray (exclude first hits, rerun).

Emission sign rules carry over verbatim from the contract (§3.3): the
"same index" property makes Z/W deltas INVERTED sign (!add). The
call structure (push_threats_on_move/on_change legs, occ_transit
discipline, post-mutation board state) is inherited unchanged — this
replaces push_threats_for_piece's internals only.

Gate: exact-multiset parity vs scalar per change (the resurrected
diagnostic, now to be flipped from "categorize known gaps" to "assert
zero"), then net-count parity per move, then the full incremental
suite, then bench-identical.
