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
  protectable. Re-implementing them is generally permitted.
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

## 1. Time management (Viridithas)

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

### The license point (narrow)
When Coda studied Viridithas and set these constants (TM redesign 2026-05-26), and
throughout our checkout (v19.0.1, 2026-06-20), **Viridithas was MIT-licensed**
(Copyright 2022-2025 Cosmo Bobak). It relicensed to **AGPL-3.0 as of v21** on
2026-07-06 — after the versions we referenced — and MIT stays in force for the
versions released under it. This rebuts one specific claim only: the borrowing was
**not an AGPL-copyleft violation**, because the source was MIT at the time. It does
**not** mean the borrowing didn't happen or doesn't matter, and MIT is not a
licence to pass someone else's design off as our own. We treat it as: not an AGPL
problem — still something to attribute and to make our own.

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
What we took from Viridithas was real but bounded: the **high-level shape of the
window/factor algorithm and a set of numeric tuning constants** — not source code,
comment text, or the surrounding implementation. In our view that sits on the
idea/fact side of the copyright line rather than being copied protectable
expression — but we will not lean on that to claim more originality than we are
owed. It was a deliberate borrowing, and the originality critique has a fair point.
Our response is to **attribute Viridithas plainly** and to **retune the constants
on Coda** so the operating point becomes ours, rather than to argue the label.

## 2. King-bucket layout (`CONSENSUS_BUCKETS`)

Coda's `CONSENSUS_BUCKETS` is the standard 16-bucket king layout — a symmetric 8×8
grid, fine-grained near the king and coarse far from it. This is the natural way to
bucket king squares for a mirrored HalfKA net, and many engines have converged on
the *exact* grid: the same table appears verbatim in at least Alexandria and
Tarnished, and the fine-near/coarse-far approach is near-universal. Coda named it
`CONSENSUS_BUCKETS` precisely because it is the common, shared choice. That its
values coincide with Alexandria's `buckets[64]` is a textbook **merger** case — for
a fixed symmetry there are only so many ways to write the table, so matching values
are expected without copying — and Alexandria is in any case **GPL-3.0**
(compatible, not AGPL). We corrected the comment to describe it as the standard
shared layout rather than implying a single source. No retrain is needed — the
shipped net already uses it.

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

On the specific **AGPL** claim: the Reckless-derived code has been removed, and the
Viridithas material was taken under MIT (before that engine's later AGPL relicense),
so we do not believe Coda carries AGPL-incompatible code — and we are auditing to
confirm it.

On the broader **originality** question we are deliberately not going to overclaim.
Coda did take a high-level algorithm outline and a set of constants from Viridithas
(under MIT), and studied other engines closely. We do not think any *copied
protectable expression* remains — but that is a narrower statement than "wholly
original", and we would rather earn the originality than assert it: by attributing
what we borrowed, making the borrowed constants our own, correcting the comments
that oversold the borrowing, and auditing for anything else.

*If any author believes specific protectable expression remains, we want to hear
the specifics and will review and take appropriate steps to correct it.*
