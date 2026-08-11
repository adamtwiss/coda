//! FT pair reordering for L1 chunk-density reduction.
//!
//! The L1 matmul consumes the pairwise activations as 4-byte chunks, and a
//! chunk must be processed if *any* of its 4 bytes is non-zero. Measured on the
//! production net, only 25.84% of individual pairwise slots are live — the net
//! is already sparse — but the live slots are scattered essentially at random,
//! so `1-(1-0.2584)^4 = 69.75%` of chunks survive (69.32% measured). Grouping
//! slots that go quiet together drops that to ~49% with no change to the
//! function being computed.
//!
//! **Why this is output-preserving.** The pack pairs accumulator lane `i` with
//! lane `i + pw` and writes the product to output slot `i`. Reordering which
//! pair lands in which slot is a permutation of the hidden layer: apply `perm`
//! to the FT columns of both halves (PSQ weights, threat weights, biases) and
//! the same permutation to the L1 weight rows of both perspectives, and every
//! product in the L1 dot-product is preserved — only its position in the sum
//! moves. Integer addition is commutative, so the result is bit-identical.
//!
//! `perm[j] = k` means "new slot `j` holds what was slot `k`".

/// Parsed offsets of the blocks a reordering has to touch.
struct Layout {
    header_len: usize,
    hidden: usize,
    psq_inputs: usize,
    threats: usize,
    l1_cols: usize,
    /// byte offset of the L1 weight block ([hidden][l1_cols] i16)
    l1_off: usize,
}

fn rd_u16(d: &[u8], o: usize) -> usize { u16::from_le_bytes([d[o], d[o + 1]]) as usize }
fn rd_u32(d: &[u8], o: usize) -> usize {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize
}

fn parse(d: &[u8]) -> Result<Layout, String> {
    if d.len() < 16 { return Err("file too short".into()); }
    let version = rd_u32(d, 4);
    if !(7..=10).contains(&version) {
        return Err(format!("only v7-v10 nets can be reordered (got v{version})"));
    }
    let flags = d[8];
    let pairwise = flags & 2 != 0;
    let bucketed = flags & 8 != 0;
    let has_threats = flags & 64 != 0;
    let extended_kb = flags & 128 != 0;
    if !pairwise {
        return Err("net is not pairwise; slot reordering does not apply".into());
    }
    let hidden = rd_u16(d, 9);
    let l1_size = rd_u16(d, 11);
    let _l2 = rd_u16(d, 13);
    let mut o = 15;
    let threats = if has_threats { let t = rd_u32(d, o); o += 4; t } else { 0 };
    let kb = if extended_kb { let k = d[o] as usize; o += 2; k } else { 16 };
    if version >= 10 { o += 1; } // training_flags
    let header_len = o;

    let psq_inputs = kb * 768;
    let l1_cols = if bucketed { 8 * l1_size } else { l1_size };
    // PSQ weights (i16) + biases (i16) + threat weights (i8)
    let l1_off = header_len + psq_inputs * hidden * 2 + hidden * 2 + threats * hidden;
    if l1_off + hidden * l1_cols * 2 > d.len() {
        return Err(format!(
            "computed L1 offset {l1_off} + block overruns file ({} bytes) — header parse disagrees \
             with the loader; refusing to write a corrupt net", d.len()));
    }
    Ok(Layout { header_len, hidden, psq_inputs, threats, l1_cols, l1_off })
}

/// Apply `perm` (length `hidden/2`) to a net, returning the reordered bytes.
pub fn permute(data: &[u8], perm: &[usize]) -> Result<Vec<u8>, String> {
    let l = parse(data)?;
    let pw = l.hidden / 2;
    if perm.len() != pw {
        return Err(format!("perm has {} entries, net needs {pw}", perm.len()));
    }
    let mut seen = vec![false; pw];
    for &p in perm {
        if p >= pw { return Err(format!("perm entry {p} out of range 0..{pw}")); }
        if seen[p] { return Err(format!("perm entry {p} repeated — not a permutation")); }
        seen[p] = true;
    }

    let mut out = data.to_vec();

    // FT columns: rows of `hidden` i16 values, both halves moved together.
    let mut o = l.header_len;
    for _row in 0..l.psq_inputs {
        for j in 0..pw {
            for (dst, src) in [(j, perm[j]), (j + pw, perm[j] + pw)] {
                let (d0, s0) = (o + dst * 2, o + src * 2);
                out[d0] = data[s0];
                out[d0 + 1] = data[s0 + 1];
            }
        }
        o += l.hidden * 2;
    }

    // FT biases (one row of `hidden` i16).
    for j in 0..pw {
        for (dst, src) in [(j, perm[j]), (j + pw, perm[j] + pw)] {
            let (d0, s0) = (o + dst * 2, o + src * 2);
            out[d0] = data[s0];
            out[d0 + 1] = data[s0 + 1];
        }
    }
    o += l.hidden * 2;

    // Threat weights: rows of `hidden` i8.
    for _row in 0..l.threats {
        for j in 0..pw {
            out[o + j] = data[o + perm[j]];
            out[o + j + pw] = data[o + perm[j] + pw];
        }
        o += l.hidden;
    }
    debug_assert_eq!(o, l.l1_off);

    // L1 weights: [hidden][l1_cols] i16, row index = perspective*pw + slot.
    let rb = l.l1_cols * 2; // row bytes
    for j in 0..pw {
        for (dst, src) in [(j, perm[j]), (j + pw, perm[j] + pw)] {
            let (d0, s0) = (l.l1_off + dst * rb, l.l1_off + src * rb);
            out[d0..d0 + rb].copy_from_slice(&data[s0..s0 + rb]);
        }
    }

    Ok(out)
}

/// Read a permutation from a text file (one index per line, or whitespace separated).
pub fn read_perm(path: &str) -> Result<Vec<usize>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    s.split_whitespace()
        .map(|t| t.parse::<usize>().map_err(|e| format!("bad perm entry {t:?}: {e}")))
        .collect()
}

pub fn describe(data: &[u8]) -> Result<String, String> {
    let l = parse(data)?;
    Ok(format!(
        "header {} B, hidden {} (pw {}), psq inputs {}, threats {}, L1 {}x{} at offset {}",
        l.header_len, l.hidden, l.hidden / 2, l.psq_inputs, l.threats, l.hidden, l.l1_cols, l.l1_off))
}
