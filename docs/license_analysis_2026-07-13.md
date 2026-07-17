# License analysis and remediation (2026-07-13)

**Status:** engineering findings and Coda's good-faith view. This is not legal
advice and not a final legal determination; it is our honest technical
assessment of what we took from other engines, what we did not, under which licenses, and what we
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
  independence we could not fully guarantee, and one closely-modelled index construction.
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
| **AGPL-3.0** | **Reckless** (AGPL when we referenced its KB layout, 2026-05); **Icarus**, **Raphael**, **Tcheran** (studied more recently) | Incompatible with GPLv3 redistribution. Reckless-derived material was removed or reimplemented; for Icarus/Raphael/Tcheran our review found no derived code in Coda's source. |
| **MIT** (permissive, GPL-compatible) | **Viridithas** (MIT through v20; relicensed to AGPL-3.0 **as of v21**, 2026-07-06 — after we referenced it), **Hobbes**, **Midnight** | GPL-compatible; obligation is attribution, not removal. |
| **WTFPL** (public-domain-equivalent, GPL-compatible) | **Starzix** | GPL-compatible; effectively no restrictions beyond honesty. |
| **GPL-3.0** (compatible) | Stockfish, Obsidian, Alexandria, Berserk, PlentyChess, Stormphrax, Clarity, Halogen, Seer, Cinder, Clover, Igel, Minic, Tucano, Weiss | Compatible with our GPLv3 intent. |

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
versions released under it. This rebuts the specific claim of an
**AGPL-copyleft violation**, because the source was MIT at the time. It does
**not** mean nothing was taken, or that it doesn't matter — and MIT is not a license
to pass someone else's design off as our own. What MIT does require is clear
attribution, which was already present and has now been further clarified; our use of
those MIT-era versions is consistent with it.

### Also true
Coda had its own TM before this (a different 6-factor shape); the 2026-05-26
redesign moved to the standard opt/hard/max window decomposition after a
cross-engine audit. A substantial part of the current TM is Coda-original, with its
own test provenance — no-inc caps, adaptive moves-to-go, phase-scaling, ponder bump,
score-trend and cross-thread factors, inc_cover ceiling — with no counterpart in the
engines we studied; the remainder is our own implementation of that standard
window/factor decomposition.

### What made it look worse than it was
Coda's **own comments overstated what was taken** — "ports Viridithas's TM window
structure", "verbatim from Viridithas". Those read as a *code* copy when the code
is a re-implementation and only some *constants* were shared, under MIT.

### Our view
What we took from Viridithas was real but bounded: the **high-level shape of the
window/factor algorithm and a set of numeric tuning constants** — not source code,
comment text, or the surrounding implementation. In our view that sits on the
idea/fact side of the copyright line rather than being copied protectable
expression — but we will not lean on that to claim more originality than we are
owed. It was deliberate, and the originality critique has a fair point.
Our response was to **attribute Viridithas plainly** — a clear reference, kept in the
code, that this part of the TM follows Viridithas's approach — and to **retune the
constants on Coda's own search and net** so the operating point is our own, rather than
to argue the label.

### The retune, and what it shows
We exposed all of these TM constants as tunable parameters and ran our own SPSA
over them on Coda's own search and net (tune #2738, LTC 40+0.4). It converged to
essentially the same values — ~40% bit-identical as integers, the rest within a few
percent. That convergence is the informative part: a tuning constant sitting at a
functional optimum is where any engine tuning for it ends up — the same reason
piece values (a rook ~500–550) or LMR curve constants look alike across engines.
The convergence shows these numbers are largely **scientific fact — a functional
operating point — not creative or artistic expression that could be copied**.
Following the tune, the constants are now Coda's own values, tuned and validated on
Coda's own search and net.

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

- **2026-07-11 (weekend license pass, recorded in
  `docs/license_compliance_review_2026-07-11.md`):** erring on the side of caution,
  removed the SIMD optimisations whose independence from Reckless we could not fully
  guarantee — the threat-feature path reverted to Coda's own **pre-existing** scalar
  implementation (`ad4fc74`), and the setwise attack kernels were rewritten in
  Coda's own code — a vectorisation of that scalar oracle plus the public-domain
  Kogge-Stone fill (`eb92686`, asserted byte-identical to the scalar oracle). Removed and redacted reproduced Reckless source from the research
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
threat feature-**index construction** in `threats.rs` was closely modelled on
Reckless's (AGPL) `threat_index.rs` — to the point that a helper struct and its
bit-packing were byte-identical, and the table construction followed it closely.

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

## 5. The other engines Coda references (originality, not licensing)

In addition to the AGPL concerns above, we extended the audit to the other engines
Coda references — variously GPL-, MIT-, and WTFPL-licensed, all compatible with Coda's
GPLv3 distribution. For the engines Coda draws on most substantially we did a detailed,
expression-level comparison against their source; the remaining references are lighter
— a named technique, or a constant cited in a cross-engine comparison. All these
licenses permit sharing a *technique* with credit, so this was not a compliance
question — it was about originality and the community's norms around attribution.

- **No material code overlap.** We found no evidence of copied code from any of these
  engines — no lifted functions, transliterated files, or engine-unique constants.
  Where Coda implements a shared technique, the code is our own: our own structure,
  naming, and tuned parameters.
- **Ideas and approaches learned from other engines, and credited.** Coda's search
  uses well-known techniques, many of them common across the strong open-source
  engines, and our source comments credit them.
- **Some small formulas and gates are necessarily similar.** A handful of functional
  formulas and pruning gates closely resemble those elsewhere; in several places we
  note the convergent choice explicitly ("engine X uses 3, engine Y uses 2"). These
  are functional facts rather than creative expression, and they generally recur
  across several engines rather than tracing to one.
- **Consistent with community norms.** In the places where Coda's code most resembles
  another engine's, the same pattern is typically used by multiple engines. We believe
  our usage is consistent with both these licenses and the community's conventions for
  crediting shared ideas.

The one substantive MIT relationship — the time-management constants Coda studied from
Viridithas (MIT at the time) — is documented in §1. Overall we found Coda's use of
these engines' ideas to be fully consistent with their licenses and appropriately
credited; the detailed, engine-by-engine review is kept in our internal notes.

The engines whose ideas and techniques we credit in Coda's own source — with thanks to
their authors — are, under the **GPL**: Stockfish, Berserk, Obsidian, Alexandria,
Stormphrax, Clarity, PlentyChess, Halogen, Seer, Cinder, Clover, Igel, Minic, Tucano
and Weiss; under the **MIT** license: Viridithas (its MIT-licensed versions through v20,
before its v21 relicense to AGPL; see §1), Hobbes and Midnight; and under the **WTFPL**:
Starzix. (We have looked at more engines than this over time; these are the ones we
actually cite as sources of ideas.)

## Third-party library licenses

Separate from the engine *ideas* credited in §5, Coda **links** a number of
third-party Rust libraries — and those carry real license obligations (ideas don't;
linked code does). We reviewed the full dependency tree: every dependency is under a
GPL-compatible license, and none is AGPL. Specifically:

- **GPL-3.0 libraries** — `shakmaty`, `shakmaty-syzygy` and `pgn-reader` (Syzygy
  tablebase probing and PGN handling), and `sfbinpack` (training-data format). Because
  Coda links these, the combined binary is conveyed under the GPL — which is Coda's own
  license anyway (`LICENSE`) — and corresponding source is available from this
  repository.
- **Permissive libraries** — the remaining crates are MIT / Apache-2.0 / Unlicense /
  Unicode-3.0 (e.g. `clap`, `libc`, `serde`, and the `shakmaty` family's dependencies).
  These are GPL-compatible; their copyright and permission notices are reproduced in
  [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
- **No dependency is AGPL** or otherwise incompatible with GPLv3 distribution.

What we were missing was an explicit third-party licenses file: the notices lived in
each crate's own source but were not collected for our binary distributions. We have
now added [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md), which reproduces
every dependency's copyright and permission notice. It is regenerated from the
dependency tree at release time (`cargo bundle-licenses`) and attached to each release,
so binary downloads now carry the third-party notices alongside the binary.

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
- `f31463a` — reworded engine-reference comments to credit techniques as shared
  conventions (concept-from-X, not port-of-X) and dropped stale/foreign source
  line-number citations; comment-only.
- **third-party licenses** — added `THIRD_PARTY_LICENSES.md` (the linked dependencies'
  notices) and wired its regeneration into the release build (see "Third-party library
  licenses").

## Repository structure

Coda began as a one-person hobby project, and we originally kept everything in a
single public repository — the engine, our per-engine study notes, research and
analysis docs, data-processing tooling, and the AI-assistant skills — and continued
that in the spirit of working in the open. Much of that internal material, though,
is hard to publish cleanly under GPLv3: the per-engine study notes in particular
contain snippets of code from multiple engines under several different licenses —
other people's copyrighted code, not ours to redistribute under our own license — and
the research docs and tooling carry similar entanglements.

So we moved the internal research and tooling — engine study notes,
research/analysis docs, the data-processing scripts, and the assistant skills — into
a **separate private repository**, leaving the public repository as the engine plus
these licensing docs. This is a licensing-hygiene and cleanliness step, done in the
open (recorded here and in the commit history, without rewriting history): it
removes the problem of redistributing others' code under our own license, and keeps
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

## Audit scope and status

We are not limiting remediation to the externally-reported spots.

**Phasing.** The immediate priority is restoring clean license *compliance* —
removing the AGPL-incompatible material so Coda stands cleanly as GPLv3. A second
pass covers **originality, not just compliance**: even where a resemblance is
license-compatible (e.g. GPL), we wanted to find and tidy any specific spots where
our implementation followed another engine too closely, and confirm attribution is
correct. We have now done this for the other engines Coda references — see §5.

**Further audits.** Beyond the externally-reported spots we set out to check several
other areas. These are now largely complete:

1. **AGPL engines (Reckless; Viridithas v21+) — done.** The Reckless-derived code has
   been removed or reimplemented (§3, §4), and our source review found nothing further
   tracing to Reckless or to Viridithas's post-relicense (v21+) source. (Older git
   history still contains the removed material — see "Git history".)
2. **The other engines Coda references (GPL, MIT, WTFPL) — done.** Reviewed in §5: no
   copied expression, and appropriately credited. Engines Coda does not reference — such
   as the unlicensed *integral* — are simply not used as a source.
3. **Internal research notes and docs — done.** Moved out of this public repository into
   a private one (see "Repository structure"), which removes the exposure of
   redistributing third-party source under our own license.
4. **GoChess** (the predecessor engine Coda was rewritten from) — **still to do.**
   GoChess is no longer actively developed — Coda superseded it — but we will run the
   same kind of review there, for the same patterns. It predates the material covered in
   items 1–3 above, so we expect it to be largely unaffected, but will confirm and record
   the result either way.

**Other AGPL engines reviewed (2026-07-14).** Community feedback (a GitHub issue)
prompted a review of **Icarus** (AGPL): we confirmed there is no Icarus-derived code in
Coda's source. We extended the same check to **Raphael** and **Tcheran** (also AGPL,
studied more recently) — none contributed any shipped code. Where we had noted ideas
from them, those experiments failed testing and were never merged. All three are now on
our AGPL-exclude list.

**Direct source-code reviews (2026-07-17).** As a further step, we have now done
direct source-code reviews of Coda's current source against the published source of
**every AGPL engine in question — Reckless, Icarus, Raphael, and Tcheran**. We found
**no copied protectable expression** in any case: the only overlaps are convergent Stockfish-lineage idioms and an
interop-mandated encoding (the threat-feature index, which Coda's own code
attributes to Stockfish, not to any of these engines), all written in Coda's
independent code. We did **not** include **Viridithas** in this review: the
Viridithas code Coda studied was under **MIT**, pre-dating that engine's v21 relicense to AGPL, so there
is no AGPL-Viridithas source for Coda to have derived from.

**On the "clean-room" wording (2026-07-17).** We had earlier described the SIMD
re-write as "clean-room." Following feedback that a single agent both analysing the
original and writing the replacement does not meet a strict clean-room (separate-teams)
standard, we have reworded it to state plainly what was done: the AVX2 kernels are a
vectorisation of Coda's own scalar code plus the public-domain Kogge-Stone algorithm.
The load-bearing point does not depend on the process label — **the result contains no
copied protectable expression, which we verified by direct comparison against the
engines' published source.** Non-infringement turns on whether the output copies
expression, not on what the process is called.

**Defensive measures going forward (to be encoded in CLAUDE.md):**
1. **Restrict the idea-reference set to GPL-3.0-compatible engines** and **exclude
   AGPL engines (Reckless, Viridithas v21+, Icarus, Raphael, Tcheran) entirely** — their copyleft is
   incompatible with our distribution. Permissive-licensed engines (MIT/BSD/WTFPL) may
   be referenced with attribution.
2. **Ideas, never expression.** Reference engines are studied to learn the
   *technique*; Coda's implementation is written independently, and no code, comment
   text, or tuning constant is copied — even from license-compatible engines.
3. **Attribute techniques as general/cross-engine conventions**, not as ports of a
   specific engine's change, and keep provenance notes accurate (neither
   overstating nor concealing).
4. **No AGPL engine source retained on our machines.** We have removed the source
   code of all AGPL-licensed reference engines from our development machines,
   deleted our internal crib-notes covering them, and use only their official
   **binary releases** for cross-engine testing (running a binary carries no
   license obligation). This should prevent any future possibility of AGPL-licensed code entering Coda.

## Position

Andrew Grant's original feedback had merit. He was right about several of his findings;
our own follow-up audit then surfaced further material that was not clearly
license-compatible, along with things we could do better. We are grateful to him for
raising it.

On the specific **AGPL** claim: the Reckless-derived code has now been removed or
reimplemented, and the Viridithas material was under MIT when we referenced it (before
that engine's later AGPL relicense), so we do not believe Coda carries
AGPL-incompatible code.

On the broader **originality** question: Coda has learned from the field and shares
techniques and patterns with other strong engines — as most top engines do, drawing on
a large body of shared, openly published ideas. A recurring finding of this audit was
just how many of the techniques we use are common across the top engines. Where the
audit did surface copied expression — in the NNUE threat-index construction and some
SIMD optimisations, based on AGPL code — we removed or independently reimplemented it,
and we do not believe any *copied protectable expression* now remains. We are
comfortable that we have credited the ideas we drew from MIT- and GPL-licensed engines
and that we are using them fairly.

*If any author — or anyone — believes specific protectable third-party expression
remains in Coda, please tell us the specifics — ideally by opening a
[GitHub issue](https://github.com/adamtwiss/coda/issues). We will review it promptly
and take the appropriate corrective step — removing the code where a license
requires it, or correcting the attribution where a permissive license allows reuse
with credit.*
