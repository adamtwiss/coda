# Derivative-work analysis and remediation (2026-07-13)

**Status:** engineering findings and Coda's good-faith view. This is not legal
advice, and not a final legal determination; it is our honest technical
assessment of what was borrowed, what was not, and what we changed. We welcome
correction from the authors involved.

## Why this exists

After the 0.9.1 work, Andrew Grant (Ethereal author) publicly identified places
where Coda resembled AGPLv3-licensed engines — principally **Reckless** and
**Viridithas** — and questioned whether Coda is a derivative work assembled from
other engines. Coda's intended license is GPL-3.0-or-later. AGPLv3 is a stricter
copyleft than GPLv3 and is *not* compatible with clean GPLv3 redistribution, so
any AGPL-derived material is a real problem that must be removed or relicensed.
We take both the licensing point and the originality point seriously. This
document records what we found and our position, and the accompanying commits
record the fixes.

The over-arching principle we're applying: **copyright protects creative
expression, not ideas, algorithms, procedures, or functional facts**
(17 U.S.C. §102(b); the merger and scènes-à-faire doctrines). A chess engine's
search *techniques* and *tuning constants* are largely in the unprotected
category; its *source text* (the specific code, names, structure, comments) is
where protectable expression lives. Our test throughout was therefore: **was any
expression copied, or only an idea / a functional constant?**

## 1. Time management (the main item)

### What we found

Coda's TM uses an opt/hard/max window model with a multiplicative factor product
(stability × fail-low × forced-move × subtree-size, plus Coda-only score-trend
and cross-thread factors). This is the structure **common to modern engines** —
Stockfish, Reckless, Obsidian, Hobbes, PlentyChess and Viridithas all run
variants of it.

Compared directly against Viridithas `src/timemgmt.rs` (v19.0.1):

- **No source text was copied.** Viridithas implements TM as a stateful
  `TimeManager` struct in a dedicated module, driven by event callbacks
  (`report_completed_depth`, `report_aspiration_fail`, `report_forced_move`).
  Coda inlines TM into the iterative-deepening loop in `search.rs` with a
  separate `compute_tm_budgets` helper. Different function names, different code
  organization, different data flow (Viridithas recomputes windows inside each
  callback and multiplies a fresh window; Coda computes budgets once and scales a
  `soft_limit` in the loop). Line-for-line, the Rust is Coda's own.
- **Two routines are parallel in *design*.** The window derivation
  (`max = clock·frac − overhead`; `hard = clock·frac` clamped to max;
  `opt = (clock/divisor + inc·frac)·frac` clamped to hard; movestogo clamps the
  divisor to `[2, N]`) and the factor-product mechanism map closely to
  Viridithas's two corresponding functions. That shared design is an **algorithm**,
  not protected expression, and it is not unique to Viridithas.
- **Some numeric constants were initialised from Viridithas's published values**
  when Coda adopted this structure (the window fractions 0.60 / 0.46 / 0.73 /
  0.94, default-moves-to-go 24, the stability table, the forced-move fractions
  0.386 / 0.627, the fail-low bonus, the subtree multiplier). Individual
  functional tuning constants are not copyrightable (there is no creative
  authorship in "spend ~46% of the clock as the abort ceiling"), and these are a
  general cross-engine convention.

Important context: **Coda had its own TM before this.** The earlier TM was a
different shape (6 factors compounding to ~30×, plus a separate hard×0.5 cap). A
2026-05-26 cross-engine audit found that outlier structure was clamping 65% of
iterations, so the redesign moved to the standard decomposition. Since then the
TM has accumulated substantial Coda-original work with its own test provenance:
the no-inc sudden-death caps, adaptive moves-to-go growth, phase-scaling, the
ponder bump, the cross-move score-trend and cross-thread-instability factors, and
the inc_cover ceiling. Roughly half the TM code is Coda-specific.

### What made it look worse than it was

Coda's **own comments overstated the borrowing.** They said the redesign
"ports Viridithas's TM window structure" and that constants were "verbatim from
Viridithas." Those were honest developer notes, but "port" and "verbatim" read as
a *code* copy when the code is a re-implementation and only some *constants* were
shared. The screenshots that circulated were of those comments, not of copied
code.

### Our view

In our assessment Coda's TM is **not a derivative work of Viridithas in the
copyright sense**: no protectable expression was copied. What is shared is (a) a
general TM algorithm used across the field, and (b) a set of functional tuning
constants that are not themselves copyrightable. The consciously-followed
*design* means this is a re-implementation informed by studying Viridithas, not
an independent reinvention — but re-implementing an unprotected algorithm is
exactly what copyright permits.

### What we changed

- **Reworded every TM comment** to describe the code accurately: a
  re-implementation of a cross-engine-standard structure, with Coda's own TM
  heritage and original layers, and one honest provenance note scoping the
  borrowing to the initial constants (commit `a3c0914`).
- **Exposed the borrowed constants as tunables** and are running a Coda TM-cluster
  SPSA so the operating point is derived on Coda's own search and net rather than
  inherited (commit `7a278ab`). This is the substantive originality fix: the
  values become ours, measured against Coda.

## 2. King-bucket layout (NNUE)

Coda's `CONSENSUS_BUCKETS` (the NNUE king-bucket layout) is byte-identical to
Alexandria's `buckets[64]`. Alexandria is **GPL-3.0**, which is compatible with
Coda's intended GPL-3.0-or-later license, so this is not an AGPL problem. It is
also the standard 16-bucket fine-near / coarse-far layout used across many
engines, and a square→bucket lookup table dictated by the symmetric design goal
is a weak copyright target (functional). We corrected the attribution comment to
cite Alexandria (GPL-3.0) accurately rather than "Alexandria/Viridithas"
(commit `28a09dc`). No net retrain is required: the shipped net already uses this
GPL-compatible layout.

## 3. Reckless items

- The earlier NNUE **king-bucket / output-bucket code** derived from Reckless's
  kb10 layout was removed in prior commits; the converter's remaining Reckless
  surface (an unused enum variant, a CLI option, and a `See Reckless/src/nnue.rs`
  source pointer) was stripped in `28a09dc`. The loader retains only a numeric id
  used to *detect and reject* legacy kb10 nets.
- Several NNUE SIMD/data-layout comments reference "Reckless pattern"
  (register-tiling, `activate_ft`-style pairwise packing, aligned accumulator
  layout). These are **idea attributions for re-implemented techniques**, not
  copied code; SIMD tiling strategies are functional and shared across the field.
  We are auditing these per-item and will reword any that overstate the
  relationship, on the same principle as the TM comments.

## 4. Position and next steps

- Coda's intended license is GPL-3.0-or-later. Until this cleanup is complete the
  README carries a notice asking people not to rely on the license or redistribute
  (commit `275b86b`).
- **Removed / in progress:** Reckless KB/output-bucket code (done); converter
  Reckless surface (done); TM comment accuracy (done); TM constants exposed for a
  Coda retune (done, tune pending); Reckless SIMD-comment audit (in progress).
- We believe the remaining resemblances are to **unprotected algorithms and
  functional constants**, not copied expression, and that Coda is therefore not a
  derivative work of the AGPL engines in the copyright sense. We are nonetheless
  remediating the originality concern in good faith by making the borrowed
  constants our own and by correcting comments that oversold the borrowing.

*If any author believes specific protectable expression remains, we want to hear
the specifics and will remove or relicense it.*
