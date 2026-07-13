# License analysis and remediation (2026-07-13)

**Status:** engineering findings and Coda's good-faith view. This is not legal
advice and not a final legal determination; it is our honest technical
assessment of what was borrowed, what was not, under which licenses, and what we
changed. **It is a living document:** it reflects our understanding as of the date
in the title, and we will update it as our audits continue or as new information
comes to light — from our own review or from anyone else. Later revisions are us
incorporating new facts, not shifting our position. We welcome correction from the
authors involved.

## Why this exists

In July 2026, Andrew Grant publicly identified places where Coda
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
  expression — copying it is the thing to avoid. Our audit found instances where
  it had happened (see §3–§4): a few small reproduced tables, some SIMD whose
  independence we could not fully guarantee, and one ported index construction.
  Erring on the side of caution, we removed or independently reimplemented each.
  Elsewhere the resemblances are to algorithms and functional constants, not
  copied source.
- **Search techniques and time-management algorithms** are ideas/procedures — not
  protectable. Re-implementing them is generally permitted.
- **Tuning constants and small functional tables** are facts dictated by their
  function. There are only so many ways to write a king-bucket table for a given
  symmetry, or a piece-value array like `[100, 300, 300, 500, 900]` — so they look
  the same across many engines by necessity, not by copying (merger doctrine). §2
  documents this concretely: Coda's king-bucket grid appears *verbatim* in four
  other independent engines (five including Coda). Individual numbers ("spend ~46%
  of the clock") carry no creative authorship.

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
Our response was to **attribute Viridithas plainly** and to **retune the constants
on Coda**, rather than to argue the label — which we have now done.

### The retune, and what it shows
We exposed all of these TM constants as tunable parameters and ran our own SPSA
over them on Coda's own search and net (tune #2738, LTC 40+0.4). It converged to
essentially the same values — ~40% bit-identical as integers, the rest within a few
percent. That convergence is the informative part: a tuning constant sitting at a
functional optimum is where any engine tuning for it ends up — the same reason
piece values (a rook ~500–550) or LMR curve constants look alike across engines.
The convergence shows these numbers are largely **scientific fact — a functional
operating point — not creative or artistic expression that could be copied**. The
constants are Coda's own tunable parameters, tuned and validated on Coda's own
search and net.

## 2. King-bucket layout (`CONSENSUS_BUCKETS`)

Coda's `CONSENSUS_BUCKETS` is the standard 16-bucket king layout — a symmetric 8×8
grid, fine-grained near the king and coarse far from it. This is the natural way to
bucket king squares for a mirrored HalfKA net, and many engines have converged on
the *exact* grid. A scan of our local engine collection (2026-07-13) found the same
table verbatim in at least **four** other engines — **Alexandria, Koivisto,
Stormphrax, and Tarnished** (Stormphrax in the mirrored 4-wide form) — and the same
fine-near/coarse-far principle with different granularity is near-universal (e.g.
Obsidian uses a 13-bucket variant, Arasan a 9-bucket one). Coda named it
`CONSENSUS_BUCKETS` precisely because it is the common, shared choice. That its
values coincide with those engines' tables is a textbook **merger** case — for
a fixed symmetry there are only so many ways to write it, so matching values
are expected without copying — and every engine we found carrying the exact grid is
**GPL-3.0** (compatible, not AGPL). We corrected the comment to describe it as the standard
shared layout rather than implying a single source. No retrain is needed — the
shipped net already uses it.

## 3. Reckless (AGPL at reference time — code already removed)

Reckless *was* AGPL-3.0 when we referenced its kb10 king-bucket / output-bucket
layout (confirmed AGPL at 2026-05-31), so anything derived from it was a genuine
incompatibility. We have already taken concrete action across several commits — not
just flagged it:

- **2026-07-11 (weekend licence pass, recorded in
  `docs/license_compliance_review_2026-07-11.md`):** erring on the side of caution,
  removed the SIMD optimisations whose independence from Reckless we could not fully
  guarantee — the threat-feature path reverted to Coda's own **pre-existing** scalar
  implementation (`ad4fc74`), and the setwise attack kernels were rewritten from
  scratch as clean-room AVX2 (`eb92686`, asserted byte-identical to Coda's own
  scalar oracle). Removed and redacted reproduced Reckless source from the research
  docs (`984b80b`), and did an attribution + reference-hygiene pass (`2baaf0c`,
  `c6c518b`). Cost ~3% NPS on affected x86 hosts, accepted for compliance.
- **2026-07-13:** removed the Reckless kb10 king-bucket and output-bucket layout
  tables from `nnue.rs`, replacing them with graceful detect-and-reject of legacy
  kb10 nets (`275c443`); the prod net promoted the same day uses the consensus
  (non-Reckless) layout (`ed2c1ec`). This session then stripped the converter's last
  Reckless surface — an unused enum variant, a CLI option, and a
  `See Reckless/src/nnue.rs` source pointer (`28a09dc`). The loader now keeps only a
  numeric id used to *detect and reject* legacy kb10 nets.
- **Comments:** the remaining "Reckless pattern" attribution comments on functional
  SIMD/data-layout techniques (idea attributions for independently written code, not
  copied code) were reworded to describe the mechanism — the same treatment as the
  TM comments (`4861dd0`, `68f8568`).

## 4. Threat features (index construction — closely modelled, now reimplemented)

This is the one place the audit found copied *expression* rather than a shared
algorithm or functional data, so we call it out plainly. The threat-feature concept
(piece P on square S attacks piece Q on square T) is shared with Stockfish and
Reckless, and Coda's threat enumeration and accumulator code are Coda's own. But the
threat feature-**index construction** in `threats.rs` was a close port of Reckless's
(AGPL) `threat_index.rs`: a helper struct and its bit-packing were byte-identical,
and the table construction followed it closely.

We reimplemented that construction in **Coda's own code** (`c2cf119`) — our own
structures, naming and control flow, not one engine's code swapped for another's —
preserving the exact feature→index *mapping*, which is a functional interface the
trained net and the Bullet training side depend on (so no retrain). It is verified
behavior-identical: the search benchmark is bit-identical and the threat fuzz tests
pass.

Two things sit apart from that rewrite. The **interaction map** — which
attacker×victim piece-pairs are tracked as features — is a small functional table:
a feature-set spec that defines the net's inputs, not creative expression. NNUE
threat inputs of this shape are a technique shared across strong engines (Stockfish's
`FullThreats`, Reckless, and others), and this map is a functional choice about which
pairs carry signal; it stays as-is because it *is* the net's input definition.
Separately, the internal delta encoding, which had matched Reckless's field order,
was given Coda's own bit layout (`cd170a4`).

The **Bullet training side** was audited the same way: its threat-index code was
already independent (no shared bit-packing; the enumeration is Coda-matched), so only
an over-attributing comment needed fixing, and a dead Reckless king-bucket layout
table in a training example — unused since Coda stopped supporting that net format —
was removed.

## What we changed (commits)
- `28a09dc` — stripped remaining Reckless surface from the converter.
- `b885c47` — corrected the KB-layout comment to describe it as a common pattern
  across many engines, rather than labelling it against any single source.
- `a3c0914` — reworded all TM comments to match the code (re-implementation, not a
  port); bench-identical.
- `7a278ab` — exposed the (MIT-origin) TM constants as SPSA-tunable parameters so a
  Coda tune can make the operating point our own; behavioral no-op at defaults.
- `c2cf119` — reimplemented the threat feature-index construction (§4) in Coda's
  own expression, preserving the exact mapping; bench bit-identical.
- `cd170a4` — gave the internal threat-delta encoding Coda's own bit layout.
- `4861dd0` / `68f8568` — right-sized AGPL-engine attribution comments across `src/`.
- `a209817` / this doc — the analysis and remediation record.
- `275b86b` — README notice asking people not to rely on the license or
  redistribute until the cleanup is complete.

## Repository structure

Coda began as a one-person hobby project, and we originally kept everything in a
single public repository — the engine, our per-engine study notes, research and
analysis docs, data-processing tooling, and the AI-assistant skills — and continued
that in the spirit of working in the open. Much of that internal material, though,
is hard to publish cleanly under GPLv3: the per-engine study notes in particular
reproduce substantial third-party engine source (across ~20 engines under various
licences) — other people's copyrighted code, not ours to redistribute under our own
licence — and the research docs and tooling carry similar entanglements.

So we moved the internal research and tooling — engine study notes,
research/analysis docs, the data-processing scripts, and the assistant skills — into
a **separate private repository**, leaving the public repository as the engine plus
these licensing docs. This is a licensing-hygiene and cleanliness step, done in the
open (recorded here and in the commit history, without rewriting history): it
removes the problem of redistributing others' code under our own licence, and keeps
the public repository clean and easy for others to build on.

## Git history

The remediation removes the AGPL-incompatible and reproduced material from the
current tree, but we have deliberately **not rewritten git history** — these are
ordinary removals, and the material still exists in older commits. We chose that
over erasure: while this audit is public, we would rather the whole process — what
was there, and exactly how it was removed or reimplemented — stay visible and
verifiable in the commit log than be quietly scrubbed. A history-cleaning pass
(e.g. `git filter-repo`) is an option we may take later, once the audit has
settled.

To be explicit until then: **older commits still contain the material described
here, so the git *history* is not GPLv3-clean.** The intended clean state is the
current tree once remediation is complete, and the redistribution caution in the
README applies to the history as well as the current tree.

## Audit scope and next steps

We are not limiting remediation to the externally-reported spots.

**Phasing.** The immediate priority is restoring clean licence compliance —
removing the AGPL-incompatible material so Coda stands cleanly as GPLv3. Once that
is settled, we will make a further pass to check we are not inappropriately
deriving anything from other (GPL-licensed) engines either — verifying attribution
is correct and rewriting anything that followed another engine too closely.

**Planned audits (beyond what was reported):**
1. **Full pass against the AGPL-at-reference engines (Reckless).** Grep source and
   history for any code, constants, or table values that trace to Reckless, on the
   same principle applied to the KB layout and the converter leftovers.
2. **MIT engines (Viridithas pre-v21, Hobbes, Caissa) and the unlicensed one
   (integral).** MIT borrowings are compatible but need attribution; an unlicensed
   repo (integral) should be treated as all-rights-reserved and avoided. Confirm we
   hold no verbatim expression from any of them.
3. **Review our own engine notes and research docs** (not just shipped source) for
   pasted snippets or material, the way we found the Reckless leftovers.
4. **GoChess** (the predecessor engine Coda was rewritten from): the same pass — its
   licence, and whether it contains any threats/NNUE or Reckless/Viridithas material.
   It predates the threats feature and the Bullet-trained NNUE work, so we expect it
   clean, but will confirm and record the result either way.

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
so we no longer believe Coda carries AGPL-incompatible code — and we are continuing
to audit to confirm this.

On the broader **originality** question we are deliberately not going to overclaim.
Coda did take a high-level algorithm outline and a set of constants from Viridithas
(under MIT), and studied other engines closely; and where the audit surfaced copied
expression — the reproduced tables and SIMD in the weekend pass, and the ported
threat-index construction — we removed or independently reimplemented it. We do not
think any *copied protectable expression* now remains, but that is a narrower (and
audited) statement than "wholly original", and we would rather earn the originality
than assert it: by attributing what we borrowed, making the borrowed constants our
own, correcting the comments that oversold the borrowing, and continuing to audit.

*If any author believes specific protectable expression remains, we want to hear
the specifics and will review and take appropriate steps to correct it.*
