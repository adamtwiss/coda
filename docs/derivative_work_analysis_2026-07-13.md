# Derivative-work analysis and remediation (2026-07-13)

**Status:** engineering findings and Coda's good-faith view. This is not legal
advice and not a final legal determination; it is our honest technical
assessment of what was borrowed, what was not, under which licenses, and what we
changed. We welcome correction from the authors involved.

## Why this exists

After the 0.9.1 work, Andrew Grant publicly identified places where Coda
resembled other engines — principally **Reckless** and **Viridithas** — and
questioned whether Coda is a derivative work assembled from other engines. Coda's
intended license is GPL-3.0-or-later. We take both the licensing point and the
originality point seriously, and we went looking beyond the specific spots that
were reported. This document records what we found and our position; the
accompanying commits record the fixes.

## The governing principle

Copyright protects **creative expression, not ideas, algorithms, procedures, or
functional facts** (17 U.S.C. §102(b); the merger and scènes-à-faire doctrines).
For a chess engine that means:

- **Source text** (the specific code, names, structure, comments) is protectable
  expression. Copying it is the thing to avoid. We did not do this.
- **Search techniques and time-management algorithms** are ideas/procedures — not
  protectable. Re-implementing them is exactly what copyright permits.
- **Tuning constants and small functional tables** are facts dictated by their
  function. There are only so many ways to write a king-bucket table for a given
  symmetry, or a piece-value array like `[100, 300, 300, 500, 900]` — so they look
  the same across many engines by necessity, not by copying (merger doctrine).
  Individual numbers ("spend ~46% of the clock") carry no creative authorship.

Our test throughout: **was any expression copied, or only an idea / a functional
constant — and under which license was the source at the time we referenced it?**

## License landscape (authoritative, GitHub SPDX, verified 2026-07-13)

License applies to the **version referenced**, not to whatever the upstream is
today. This matters here:

| License at reference time | Engines Coda cites | Implication for us |
|---|---|---|
| **AGPL-3.0** | **Reckless** (AGPL when we referenced its KB layout, 2026-05) | Incompatible with GPLv3 redistribution — any Reckless-derived code/constants must be removed or independently reimplemented. |
| **MIT** (permissive, GPL-compatible) | **Viridithas** (MIT through v19/v20; relicensed to AGPL-3.0 **as of v21**, 2026-07-06 — after we referenced it); **Hobbes**, **Caissa** | GPL-compatible; obligation is attribution, not removal. |
| **No detected license** | **integral** | "All rights reserved" by default — treat as most restrictive; avoid. |
| **GPL-3.0** (compatible) | Stockfish, Obsidian, Alexandria, Berserk, PlentyChess, Halogen, Stormphrax, Clarity, Clover, Astra, Koivisto, Winter, Minic | Compatible with our GPLv3 intent. |

The vast majority of these citations are **attribution comments describing shared
techniques**, not copied code. The audit's job is to separate the two.

## 1. Time management (Viridithas — MIT at reference time)

### What we found
Coda's TM uses an opt/hard/max window model with a multiplicative factor product
— the structure **common to modern engines** (Stockfish, Reckless, Obsidian,
Hobbes, PlentyChess, Viridithas all run variants). Compared directly against
Viridithas `src/timemgmt.rs`:

- **No source text was copied.** Viridithas implements TM as a stateful
  `TimeManager` struct in a dedicated module driven by event callbacks; Coda
  inlines TM into the iterative-deepening loop with a separate `compute_tm_budgets`
  helper. Different names, module layout, and data flow. The Rust is Coda's own.
- **Two routines are parallel in *design*** (the window derivation and the
  factor-product). That shared design is an **algorithm**, not protected
  expression, and it is not unique to Viridithas.
- **Some numeric constants were initialised from Viridithas's published values**
  (window fractions, stability table, forced-move fractions, etc.). These are
  functional tuning facts.

### The license point
When Coda studied Viridithas and set these constants (TM redesign 2026-05-26), and
throughout our checkout (v19.0.1, 2026-06-20), **Viridithas was MIT-licensed**
(Copyright 2022-2025 Cosmo Bobak). Viridithas relicensed to **AGPL-3.0 as of v21**
on 2026-07-06 — *after* the versions we referenced. MIT is GPL-compatible and
irrevocable for the versions released under it; the later AGPL relicense governs
v21+ and does not reach back to the v19 we used. So **the TM is not an AGPL
problem**: it is MIT-origin (compatible with our GPLv3), and even the borrowed
values are most likely uncopyrightable functional facts.

### Also true
Coda had its own TM before this (a different 6-factor shape); the 2026-05-26
redesign moved to the standard decomposition after a cross-engine audit; and
roughly half the current TM code is Coda-original with its own test provenance
(no-inc caps, adaptive moves-to-go, phase-scaling, ponder bump, score-trend and
cross-thread factors, inc_cover ceiling).

### What made it look worse than it was
Coda's **own comments overstated the borrowing** — "ports Viridithas's TM window
structure", "verbatim from Viridithas". Those read as a *code* copy when the code
is a re-implementation and only some *constants* were shared, under MIT.

### Our view
Coda's TM is **not a derivative work of Viridithas in the copyright sense** (no
expression copied), and the constants that were shared came from **MIT-era
Viridithas** — compatible with our license, requiring at most attribution. We are
crediting Viridithas and, separately, retuning the constants on Coda so the
operating point is our own regardless.

## 2. King-bucket layout (Alexandria — GPL-3.0)

Coda's `CONSENSUS_BUCKETS` is byte-identical to Alexandria's `buckets[64]`.
Alexandria is **GPL-3.0**, compatible with our GPL-3.0-or-later intent — not an
AGPL problem. It is also the standard 16-bucket fine-near/coarse-far layout, and a
square→bucket table for a fixed symmetry is a textbook **merger** case: there are
only so many ways to write it, so matching values are expected without copying. We
corrected the attribution comment to cite Alexandria (GPL-3.0). No retrain is
needed — the shipped net already uses this GPL-compatible layout.

## 3. Reckless (AGPL — genuinely, at reference time)

Reckless *was* AGPL-3.0 when we referenced its kb10 king-bucket / output-bucket
layout (confirmed AGPL at 2026-05-31), so this was a real incompatibility and the
derived code was removed. The converter's remaining Reckless surface (an unused
enum variant, a CLI option, a `See Reckless/src/nnue.rs` source pointer) was
stripped; the loader retains only a numeric id used to *detect and reject* legacy
kb10 nets. Several NNUE SIMD/data-layout comments still say "Reckless pattern" —
these are idea attributions for re-implemented, functional tiling techniques, not
copied code; we are auditing them per-item and rewording any that overstate.

## What we changed (commits)
- `28a09dc` — stripped remaining Reckless surface from the converter; fixed the
  KB attribution to Alexandria (GPL-3.0).
- `a3c0914` — reworded all TM comments to match the code (re-implementation, not a
  port); bench-identical.
- `7a278ab` — exposed the (MIT-origin) TM constants as non-core tunables so a Coda
  SPSA can make the operating point our own; behavioral no-op at defaults.
- `a209817` / this doc — the analysis and remediation record.
- `275b86b` — README notice asking people not to rely on the license or
  redistribute until the cleanup is complete.

## Audit scope and next steps

We are not limiting remediation to the externally-reported spots.

**Planned audits (beyond what was reported):**
1. **Full pass against the AGPL-at-reference engines (Reckless).** Grep source and
   history for any code, constants, or table values that trace to Reckless, on the
   same principle applied to the KB layout and the converter leftovers.
2. **MIT engines (Viridithas post-v21, Hobbes, Caissa) and the unlicensed one
   (integral).** MIT borrowings are compatible but need attribution; an unlicensed
   repo (integral) should be treated as all-rights-reserved and avoided. Confirm we
   hold no verbatim expression from any of them.
3. **Review our own engine notes and research docs** (not just shipped source) for
   pasted snippets or material, the way we found the Reckless leftovers.

**Defensive measures going forward (to be encoded in CLAUDE.md):**
1. **Restrict the idea-reference set to GPL-3.0-compatible engines** and **exclude
   AGPL engines (Reckless, and Viridithas v21+) entirely** — their copyleft is
   incompatible with our distribution. Permissive-licensed engines (MIT/BSD) may be
   referenced with attribution.
2. **Ideas, never expression.** Reference engines are studied to learn the
   *technique*; Coda's implementation is written independently, and no code, comment
   text, or tuning constant is copied — even from license-compatible engines.
3. **Attribute techniques as general/cross-engine conventions**, not as ports of a
   specific engine's change, and keep provenance notes accurate (neither
   overstating nor concealing).

## Position

We believe Coda is **not a derivative work of the AGPL engines in the copyright
sense**: the remaining resemblances are to unprotected algorithms and functional
constants, the Reckless-derived code has been removed, and the Viridithas material
was taken under MIT (compatible), before that engine's later AGPL relicense. We are
nonetheless auditing further and hardening our process in good faith.

*If any author believes specific protectable expression remains, we want to hear
the specifics and will review and take appropriate steps to correct it.*
