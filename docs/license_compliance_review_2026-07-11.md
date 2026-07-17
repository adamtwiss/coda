# Coda — Licence Compliance Review and Remediation Plan

*2026-07-11*

> **Status (2026-07-13): initial pass — superseded in part by the fuller audit.**
> This was Coda's *first* licence-compliance pass. A fuller audit followed and is
> recorded in [`license_analysis_2026-07-13.md`](license_analysis_2026-07-13.md),
> which is the current, authoritative record. Read this document with two things in
> mind:
>
> - **It was not exhaustive.** The fuller audit surfaced material this pass had
>   missed — most significantly the NNUE threat feature-**index construction**, which
>   turned out to be closely modelled on Reckless (AGPL) and has since been reimplemented
>   in Coda's own code. The "Reviewed and clean" list below reflects what this first
>   pass had checked, not a final all-clear.
> - **The plan below has since been carried out.** The remediation phases here
>   (independently re-writing the SIMD in Coda's own code, removing the Reckless bucket layouts, training a net on Coda's
>   own layout) are complete; the future-tense wording reflects the 2026-07-11
>   snapshot.
>
> We keep this document unedited as the record of the first pass.

## Purpose

Prompted by community discussion of licensing in agentically-developed engines,
this document records a review of Coda's code against a simple standard: while
ideas and algorithms are generally not copyrightable, our *implementations* of
them should be clean, licence-compatible and properly credited.

During the development of Coda, we have studied techniques from strong
open-source engines and implemented them for Coda's own architecture, validated
by testing; most of the codebase is independent implementation of shared ideas. This review focuses on
the places where third-party *code or data* was drawn in more closely than that
standard allows, and the steps being taken to put them right.

Particular attention goes to the one dependency whose licence is more restrictive
than Coda's own GPLv3: **Reckless (AGPLv3)**. AGPLv3 carries obligations —
including the network-use source-provision requirement of §13 — that attribution
alone does not satisfy. So where Coda leans too heavily on Reckless, the fix is to
re-derive independently or remove, not simply to attribute.

## Scope

Reviewed: Coda's search, evaluation, SEE, transposition table, move ordering,
NNUE inference (x86 AVX2 / AVX-512 / VNNI and ARM NEON), the accumulator and
threat machinery, and the training-side code in the Bullet fork
(`adamtwiss/bullet`, MIT). Coda's source was compared against the corresponding
source in Reckless and other referenced engines.

## Reviewed and clean

The review found the following to be independent implementations of standard
techniques — credited in comments where an idea came from a particular engine —
needing no change:

- NNUE inference SIMD — x86 (AVX2 / AVX-512 / VNNI) and ARM NEON, including the
  recent NEON dot-product kernels.
- Search, evaluation, SEE, transposition table and move ordering.
- The scalar threat-enumeration path, the threat accumulator, and the sparse-L1
  kernels.

## Items to put right

### 1. Reckless-influenced SIMD kernels (AGPLv3) — primary items

`src/threats_splat.rs` (AVX-512 threat-delta enumeration) and the AVX2 paths of
`src/setwise.rs` are too closely modelled on Reckless's vectorised SIMD to stand
as genuinely independent implementations.

**Remediation.** These will be removed and rewritten as clean, independent kernels
derived from Coda's own scalar reference (which is unaffected), with fuzz-tested
parity. In the interim the affected paths fall back to Coda's own scalar and
magic-bitboard code, so no Reckless-influenced SIMD remains in the distributed
source at any point. NNUE inference is unaffected and retains full SIMD.

### 2. Reckless-format bucket layouts (AGPLv3)

Coda's NNUE bucket-layout code is its own, built without reference to Reckless; it
supports several input king-bucket and output material-bucket *formats*, selected
as data. During experimentation one optional format in each set was populated with
Reckless's layout values. The affected tables are:

- Coda: `RECKLESS_BUCKETS_FLAT`, `RECKLESS_OUTPUT_BUCKETS_LAYOUT` (`src/nnue.rs`).
- Bullet fork: `RecklessBuckets` / `RECKLESS_LAYOUT`
  (`crates/bullet_lib/src/game/outputs.rs`) and the input layout in
  `examples/coda_v9_768_threats.rs`.

**Remediation.** These Reckless-format table options are removed; Coda's own
formats remain.
- The output-bucket format is unused by any released network and is removed now.
- The input king-bucket format is the one the current released network was trained
  with. Coda's next network trains on one of Coda's own formats; once it ships, the
  Reckless-format table is removed from both repositories.

### 3. Threat-feature interaction table — attribution (GPL-compatible)

The threat interaction-map table originates in a GPLv3 engine and is
GPL-compatible with Coda. Its attribution will be tidied, and where practical —
notably in Coda's Bullet fork — the table replaced with Coda's own computed values
so that no external table is carried there.

## Remediation phases

1. **Attribution and hygiene** *(now):* correct the threat-table and SEE
   attributions; tidy stray references that break Coda's own repository
   conventions.
2. **Remove unused code** *(now):* remove the unused Reckless output-bucket layout
   from both repositories.
3. **Clean the SIMD** *(days):* remove the Reckless-influenced `threats_splat.rs`
   and `setwise.rs` AVX2 kernels; fall back to Coda's own scalar / magic-bitboard
   paths. Result: no Reckless-influenced code in the source or the binary.
4. **Re-derive independent SIMD** *(follow-on):* restore SIMD performance with
   clean kernels written from Coda's own scalar reference.
5. **Network generation** *(with the next network):* train on Coda's own
   king-bucket layout, then remove the Reckless input layout from both
   repositories.

Each step lands in the public git history, so the remediation is as auditable as
this review.

## Standard applied

- Ideas and algorithms are studied and re-implemented; where an idea came from a
  particular engine, it is credited.
- Our implementations of those ideas should be clean, licence-compatible and
  credited. GPLv3 engines require attribution, which Coda provides and is
  completing where incomplete. The single AGPLv3 source (Reckless) requires more:
  code or data that leans on its implementation is re-derived independently or
  removed.
