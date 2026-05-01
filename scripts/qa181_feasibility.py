#!/usr/bin/env python3
"""QA=181 feasibility analysis for prod .nnue.

Reads the .nnue header + FT (PSQ) weights and threat weights, reports:
  - i16 weight distribution at QA=255
  - if rescaled to QA=181 (×181/255), what fraction would clip to i8 [-127, 127]?

The QA=181 NPS gain in viri qa-181 came from i8-storage of FT weights
(half the bandwidth of i16). Coda's prod net stores FT as i16 at QA=255;
to switch to i8 storage at QA=181 the rescaled values must fit in [-127, 127]
without significant clipping. This script quantifies that.

Usage: python3 scripts/qa181_feasibility.py <prod.nnue>
"""

import struct
import sys
from pathlib import Path


NNUE_MAGIC = 0x4E4E5545  # "NNUE" LE
PSQ_INPUTS_PER_BUCKET = 768


def read_header(data, offset):
    magic, version = struct.unpack_from("<II", data, offset)
    offset += 8
    if magic != NNUE_MAGIC:
        raise ValueError(f"bad magic: 0x{magic:X}")
    if version not in (7, 8, 9, 10):
        raise ValueError(f"only v7+ supported, got {version}")

    flags = data[offset]
    offset += 1
    use_screlu = bool(flags & 1)
    use_pairwise = bool(flags & 2)
    int8_l1 = bool(flags & 4)
    bucketed_hidden = bool(flags & 8)
    dual_l1 = bool(flags & 16)
    has_threats = bool(flags & 64)
    extended_kb = bool(flags & 128)

    ft_size, l1_size, l2_size = struct.unpack_from("<HHH", data, offset)
    offset += 6

    num_threats = 0
    if has_threats:
        (num_threats,) = struct.unpack_from("<I", data, offset)
        offset += 4

    if extended_kb:
        num_king_buckets = data[offset]
        kb_layout_id = data[offset + 1]
        offset += 2
    else:
        num_king_buckets = 16
        kb_layout_id = 0

    if version >= 10 and has_threats:
        offset += 1  # training_flags

    return {
        "version": version,
        "flags": flags,
        "ft_size": ft_size,
        "l1_size": l1_size,
        "l2_size": l2_size,
        "num_threats": num_threats,
        "num_king_buckets": num_king_buckets,
        "kb_layout_id": kb_layout_id,
        "screlu": use_screlu,
        "pairwise": use_pairwise,
        "int8_l1": int8_l1,
        "bucketed_hidden": bucketed_hidden,
        "dual_l1": dual_l1,
        "has_threats": has_threats,
        "header_end": offset,
    }


def histogram(label, values, qa_old=255, qa_new=181, dtype="i16"):
    """Report stats for an array of integer weights at QA=qa_old.

    If dtype=="i16", report what fraction would clip to i8 [-127, 127] after rescale.
    If dtype=="i8", report current clip rate at i8 (sanity check) + post-rescale clip.
    """
    n = len(values)
    if n == 0:
        return

    abs_vals = [abs(v) for v in values]
    abs_max = max(abs_vals)
    abs_max_rescaled = abs_max * qa_new / qa_old

    # Quantiles
    sorted_abs = sorted(abs_vals)
    q50 = sorted_abs[n // 2]
    q90 = sorted_abs[(n * 9) // 10]
    q99 = sorted_abs[(n * 99) // 100]
    q999 = sorted_abs[(n * 999) // 1000] if n >= 1000 else sorted_abs[-1]
    q9999 = sorted_abs[(n * 9999) // 10000] if n >= 10000 else sorted_abs[-1]

    # Rescale and check i8 clip rate
    rescale = qa_new / qa_old
    clipped_i8 = sum(1 for v in values if abs(round(v * rescale)) > 127)
    pct_i8 = 100.0 * clipped_i8 / n

    # Already-clipped (only if dtype=i8)
    clipped_now = sum(1 for v in values if abs(v) > 127)
    pct_now = 100.0 * clipped_now / n

    # Distribution shifts
    in_3sigma = sum(1 for v in values if abs(v) > qa_old / 2)  # > QA/2 = "large"
    pct_large = 100.0 * in_3sigma / n

    print(f"\n=== {label} (n={n:,}) ===")
    print(f"  dtype: {dtype}, scale: QA={qa_old}")
    print(f"  abs max: {abs_max} ({abs_max/qa_old:.3f} × QA)")
    print(f"  abs max @ QA=181 rescaled: {abs_max_rescaled:.1f}")
    print(f"  abs quantiles: q50={q50}, q90={q90}, q99={q99}, q999={q999}, q9999={q9999}")
    print(f"  >|QA/2|: {pct_large:.3f}%")
    if dtype == "i8":
        print(f"  CURRENT clip @ |127|: {clipped_now:,} ({pct_now:.4f}%)")
    print(f"  POST-rescale-to-181 clip @ |127| (i8 storage): {clipped_i8:,} ({pct_i8:.4f}%)")
    if pct_i8 < 0.01:
        verdict = "✓ trivial — i8 FT storage feasible"
    elif pct_i8 < 0.5:
        verdict = "≈ borderline — minor precision loss, likely OK"
    elif pct_i8 < 5.0:
        verdict = "⚠ moderate — likely Elo-negative without re-train"
    else:
        verdict = "✗ severe — i8 FT storage infeasible without weight clipping during training"
    print(f"  VERDICT: {verdict}")


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <prod.nnue>", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1])
    data = path.read_bytes()
    print(f"Loaded {path} ({len(data):,} bytes)")

    hdr = read_header(data, 0)
    print(f"\nHeader:")
    for k, v in hdr.items():
        if k != "header_end":
            print(f"  {k}: {v}")

    offset = hdr["header_end"]
    ft_size = hdr["ft_size"]
    kb_count = hdr["num_king_buckets"]
    psq_count = kb_count * PSQ_INPUTS_PER_BUCKET * ft_size

    # Read FT (PSQ) weights: i16 × psq_count
    ft_weights = list(struct.unpack_from(f"<{psq_count}h", data, offset))
    offset += psq_count * 2

    # Read FT biases: i16 × ft_size
    ft_biases = list(struct.unpack_from(f"<{ft_size}h", data, offset))
    offset += ft_size * 2

    # Read threat weights: i8 × (num_threats × ft_size)
    threat_count = hdr["num_threats"] * ft_size
    threat_weights = list(struct.unpack_from(f"<{threat_count}b", data, offset))
    offset += threat_count

    print(f"\nWeight blocks:")
    print(f"  FT (PSQ) weights: {psq_count:,} × i16")
    print(f"  FT biases: {ft_size} × i16")
    print(f"  threat weights: {threat_count:,} × i8")

    histogram("FT (PSQ) weights", ft_weights, dtype="i16")
    histogram("FT biases", ft_biases, dtype="i16")
    if threat_count > 0:
        histogram("threat weights", threat_weights, dtype="i8")

    # Summary verdict for the i8-FT-storage gating question
    n = len(ft_weights)
    ft_clip = sum(1 for v in ft_weights if abs(round(v * 181 / 255)) > 127)
    ft_clip_pct = 100.0 * ft_clip / n
    print(f"\n{'='*60}")
    print(f"SUMMARY: FT i16 → i8 at QA=181")
    print(f"  Clip rate: {ft_clip_pct:.4f}% of {n:,} weights")
    print(f"  This is the gating question for QA=181 NPS gain.")
    if ft_clip_pct < 0.1:
        print(f"  → FEASIBLE: re-quant from f32 checkpoint at QA=181 should work")
    elif ft_clip_pct < 1.0:
        print(f"  → MARGINAL: small clip rate, may need weight-clip-aware re-train")
    else:
        print(f"  → INFEASIBLE without re-train: too many weights would clip")


if __name__ == "__main__":
    main()
