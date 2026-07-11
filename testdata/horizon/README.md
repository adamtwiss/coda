# Horizon / blindspot corpora (harvested 2026-07-11)

Source: 1600 fixed-150k-node games Coda vs SF17/SF18-dev (Atlas, runA150),
via scripts/harvest_horizon.py + scripts/harvest_upstream.py. Methodology
and findings: docs/search_gap_decomposition_2026-07-11.md + experiments.md
(2026-07-10/11 entries).

- horizon_ep_starts.tsv — 1019 episode-start positions (sustained 40cp+
  divergence, resolved Coda-converged-to-opponent). Eval-convergence
  measurement targets. truth_cache.tsv = SF18-dev@4M values (side-to-move).
- horizon_candidates.tsv — 2712 Coda decision points inside lag windows.
  Oracle-validated sample: ~3% confirmed mistakes (moves are mostly sound;
  the class is assessment lag, not move error).
- strategic_corpus.tsv — 1779 positions from slow-resolving divergence
  episodes where the outcome sided against Coda: eval-blindspot training
  candidates (v10 track). STUBBORN share of the class <= 46% (forced-line).
- oracle_verdicts.tsv / consistency_results.tsv — validation + eval<->search
  consistency raw data (see experiments.md for the scale-normalization
  caveat: Coda's cp spread is ~2x SF's displayed scale).

Refresh policy: re-harvest per net/search generation; these stale as the
engine changes. Do not treat as a fixed benchmark — held-out discipline
and Elo gates per the design doc.
