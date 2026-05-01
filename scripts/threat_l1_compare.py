#!/usr/bin/env python3
"""Compare per-feature L1 norms of threat-weight matrices across replica nets.

For each .nnue file:
  1. Parse header to find threat-matrix offset.
  2. Read threat matrix (int8, [num_features × ft_size]).
  3. Compute per-row L1 norm (sum |w| over the ft_size dimension).

Then across the set of nets:
  - Compute per-row mean, stddev, coefficient-of-variation across replicas.
  - Sort rows by inter-replica variance.
  - Output: top-K most-variable rows, distribution of CV by row.

Hypothesis test: if the C8-fix's row-consolidation is amplifying seed
variance, then "same-type-pair semi-excluded" feature rows should show
systematically higher inter-replica variance than other rows.

Usage:
    python3 threat_l1_compare.py <net1.nnue> <net2.nnue> ...
"""
import argparse
import struct
import sys
from pathlib import Path

PSQ_INPUTS_PER_BUCKET = 768  # 12 piece types × 64 squares


def parse_header(data: bytes):
    """Return (version, flags, ft_size, l1_size, l2_size, threats, kb_count,
    body_offset)."""
    o = 0
    magic, = struct.unpack_from('<I', data, o); o += 4
    version, = struct.unpack_from('<I', data, o); o += 4
    flags = data[o]; o += 1
    if version not in (7, 8, 9, 10):
        raise ValueError(f"unsupported version {version}")
    use_pairwise = bool(flags & 2)
    has_threats  = bool(flags & 64)
    extended_kb  = bool(flags & 128)
    ft_size, = struct.unpack_from('<H', data, o); o += 2
    l1_size, = struct.unpack_from('<H', data, o); o += 2
    l2_size, = struct.unpack_from('<H', data, o); o += 2
    threats = 0
    if has_threats:
        threats, = struct.unpack_from('<I', data, o); o += 4
    kb_count = 16
    if extended_kb:
        kb_count = data[o]; o += 1
        _layout_id = data[o]; o += 1
    if version >= 10 and has_threats:
        _training_flags = data[o]; o += 1
    return {
        'magic': magic, 'version': version, 'flags': flags,
        'ft_size': ft_size, 'l1_size': l1_size, 'l2_size': l2_size,
        'threats': threats, 'kb_count': kb_count, 'body_offset': o,
        'use_pairwise': use_pairwise,
    }


def threat_matrix_offset_and_size(hdr):
    """Return (offset, size) of the threat int8 matrix in the file."""
    body = hdr['body_offset']
    ft_size = hdr['ft_size']
    kb_count = hdr['kb_count']
    threats = hdr['threats']
    psq_size = kb_count * PSQ_INPUTS_PER_BUCKET * ft_size * 2  # i16
    biases_size = ft_size * 2  # i16
    threat_offset = body + psq_size + biases_size
    threat_size = threats * ft_size  # i8 = 1 byte each
    return threat_offset, threat_size


def per_row_l1(data: bytes, offset: int, num_rows: int, ft_size: int):
    """Return list of L1 norms (sum |w|) per row, reading int8 weights."""
    import numpy as np
    arr = np.frombuffer(data, dtype=np.int8, offset=offset, count=num_rows * ft_size)
    arr = arr.reshape(num_rows, ft_size)
    return np.abs(arr.astype(np.int32)).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('nets', nargs='+', help='paths to .nnue files (replicas)')
    ap.add_argument('--top-k', type=int, default=20, help='top-K most-variable rows to print')
    ap.add_argument('--csv', type=Path, default=None, help='optional output CSV of all rows')
    args = ap.parse_args()

    import numpy as np

    rows_per_net = []
    labels = []
    for p in args.nets:
        path = Path(p)
        labels.append(path.stem.replace('net-v9-768th16x32-kb10-', '').replace('-e200s200', ''))
        data = path.read_bytes()
        hdr = parse_header(data)
        off, size = threat_matrix_offset_and_size(hdr)
        if off + size > len(data):
            raise ValueError(f"{p}: threat matrix exceeds file size (off={off} size={size} file={len(data)})")
        l1 = per_row_l1(data, off, hdr['threats'], hdr['ft_size'])
        rows_per_net.append(l1)
        print(f"{labels[-1]:50s} v{hdr['version']} threats={hdr['threats']} ft={hdr['ft_size']} "
              f"L1mean={l1.mean():.1f} L1std={l1.std():.1f} L1max={l1.max()} zero_rows={(l1==0).sum()}")

    M = np.stack(rows_per_net, axis=0)  # [n_nets, n_rows]
    n_nets, n_rows = M.shape
    row_mean = M.mean(axis=0)
    row_std = M.std(axis=0)
    row_cv = np.where(row_mean > 0, row_std / row_mean, 0.0)

    print(f"\n=== Cross-net stats over {n_rows} threat rows × {n_nets} nets ===")
    print(f"Row L1 mean: median={np.median(row_mean):.1f}, p90={np.percentile(row_mean, 90):.1f}, "
          f"max={row_mean.max():.1f}")
    print(f"Row L1 std:  median={np.median(row_std):.1f}, p90={np.percentile(row_std, 90):.1f}, "
          f"max={row_std.max():.1f}")
    print(f"Row CV:      median={np.median(row_cv):.3f}, p90={np.percentile(row_cv, 90):.3f}, "
          f"max={row_cv.max():.3f}")

    # Top-K most-variable rows (by std, gated to non-zero mean to avoid div-by-zero noise)
    nonzero_mask = row_mean > 0
    nz_rows = np.where(nonzero_mask)[0]
    print(f"\n=== Top {args.top_k} most-variable rows (by std, mean>0) — {nonzero_mask.sum()}/{n_rows} non-zero ===")
    sort_idx = nz_rows[np.argsort(-row_std[nz_rows])[:args.top_k]]
    hdr_str = f"{'row':>6s}  {'mean':>9s}  {'std':>9s}  {'cv':>6s}  " + "  ".join(f"{l:>9s}" for l in labels)
    print(hdr_str)
    for r in sort_idx:
        per_net = "  ".join(f"{int(M[i,r]):>9d}" for i in range(n_nets))
        print(f"{r:>6d}  {row_mean[r]:>9.1f}  {row_std[r]:>9.1f}  {row_cv[r]:>6.3f}  {per_net}")

    # Coefficient-of-variation distribution
    print(f"\n=== CV distribution buckets (rows with mean>0) ===")
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 100.0]
    cv_nz = row_cv[nonzero_mask]
    for i in range(len(bins) - 1):
        n = ((cv_nz >= bins[i]) & (cv_nz < bins[i+1])).sum()
        pct = 100.0 * n / len(cv_nz)
        print(f"  CV [{bins[i]:.2f}, {bins[i+1]:.2f}):  {n:6d} rows ({pct:5.2f}%)")

    if args.csv:
        with args.csv.open('w') as f:
            f.write('row,mean,std,cv,' + ','.join(labels) + '\n')
            for r in range(n_rows):
                f.write(f"{r},{row_mean[r]:.2f},{row_std[r]:.2f},{row_cv[r]:.4f},")
                f.write(','.join(str(int(M[i,r])) for i in range(n_nets)))
                f.write('\n')
        print(f"\nWrote per-row CSV to {args.csv}")


if __name__ == '__main__':
    main()
