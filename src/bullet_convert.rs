/// Convert Bullet quantised.bin to .nnue format.
///
/// Supports:
///   v5/v6: (InputSize × H)×2 → output (direct, pairwise, SCReLU)
///   v7: (InputSize × H)×2 → L1 → [L2 →] output (hidden layers)

use std::io::Write;

const NNUE_INPUT_SIZE: usize = 12288;
const NNUE_OUTPUT_BUCKETS: usize = 8;
const NNUE_MAGIC: u32 = 0x4E4E5545; // "NNUE" LE

fn read_i16_le(data: &[u8], offset: usize) -> i16 {
    i16::from_le_bytes([data[offset], data[offset + 1]])
}

fn read_i32_le(data: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

fn read_f32_le(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]])
}

fn write_u32_le(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_u16_le(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_i16_le(buf: &mut Vec<u8>, v: i16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn write_i32_le(buf: &mut Vec<u8>, v: i32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Strip Bullet footer if present ("bulletbulletbull" at end).
fn strip_footer(data: &[u8]) -> usize {
    let len = data.len();
    if len >= 32 && &data[len - 32..len - 16] == b"bulletbulletbull" {
        len - 32
    } else {
        len
    }
}

/// Convert a Bullet quantised.bin to v5/v6 .nnue format (no hidden layers).
pub fn convert_v5(
    input_path: &str,
    output_path: &str,
    use_screlu: bool,
    use_pairwise: bool,
    src_output_buckets: usize,
) -> Result<(), String> {
    let data = std::fs::read(input_path).map_err(|e| format!("read {}: {}", input_path, e))?;
    let data_len = strip_footer(&data);

    // Infer hidden size using source bucket count (may differ from NNUE_OUTPUT_BUCKETS=8)
    let ob = src_output_buckets;
    let out_mul = if use_pairwise { ob } else { ob * 2 };
    let bytes_per_neuron = NNUE_INPUT_SIZE * 2 + 2 + out_mul * 2;
    let bias_bytes = ob * 4;
    let h = (data_len - bias_bytes) / bytes_per_neuron;
    let output_width = if use_pairwise { h } else { 2 * h };

    let expected = NNUE_INPUT_SIZE * h * 2 + h * 2 + output_width * ob * 2 + ob * 4;
    println!("Input: {} bytes, hidden size: {}, src buckets: {}, expected: {} bytes", data.len(), h, ob, expected);
    if data_len < expected {
        return Err(format!("file too small: got {}, need {}", data_len, expected));
    }

    let mut offset = 0;

    // Read l0w: [InputSize][H] i16
    let mut input_weights = vec![0i16; NNUE_INPUT_SIZE * h];
    for i in 0..NNUE_INPUT_SIZE * h {
        input_weights[i] = read_i16_le(&data, offset);
        offset += 2;
    }

    // Read l0b: [H] i16
    let mut input_biases = vec![0i16; h];
    for i in 0..h {
        input_biases[i] = read_i16_le(&data, offset);
        offset += 2;
    }

    // Read output weights: [outputWidth][src_buckets] i16
    let mut out_w_src = vec![vec![0i16; ob]; output_width];
    for i in 0..output_width {
        for b in 0..ob {
            out_w_src[i][b] = read_i16_le(&data, offset);
            offset += 2;
        }
    }

    // Read output bias: [src_buckets] i32
    let mut out_bias_src = vec![0i32; ob];
    for b in 0..ob {
        out_bias_src[b] = read_i32_le(&data, offset);
        offset += 4;
    }

    // Pad/map to NNUE_OUTPUT_BUCKETS (8) for inference compatibility.
    // Each inference bucket maps to the nearest source bucket.
    let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * output_width];
    let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
    for b in 0..NNUE_OUTPUT_BUCKETS {
        let src_b = b * ob / NNUE_OUTPUT_BUCKETS; // map 8 buckets to ob buckets
        for i in 0..output_width {
            output_weights[b * output_width + i] = out_w_src[i][src_b];
        }
        output_bias[b] = out_bias_src[src_b];
    }

    println!("Parsed {} bytes of {}", offset, data.len());

    // Write .nnue
    let mut buf = Vec::new();
    write_u32_le(&mut buf, NNUE_MAGIC);

    let version: u32 = if use_screlu || use_pairwise { 6 } else { 5 };
    write_u32_le(&mut buf, version);

    if version == 6 {
        let mut flags = 0u8;
        if use_screlu { flags |= 1; }
        if use_pairwise { flags |= 2; }
        buf.push(flags);
    }

    for &w in &input_weights { write_i16_le(&mut buf, w); }
    for &b in &input_biases { write_i16_le(&mut buf, b); }
    for &w in &output_weights { write_i16_le(&mut buf, w); }
    for &b in &output_bias { write_i32_le(&mut buf, b); }

    std::fs::File::create(output_path)
        .and_then(|mut f| f.write_all(&buf))
        .map_err(|e| format!("write {}: {}", output_path, e))?;

    let activation = if use_pairwise { "pairwise" } else if use_screlu { "SCReLU" } else { "CReLU" };
    println!("Saved {} ({} bytes, v{} {} {})", output_path, buf.len(), version, activation, h);
    Ok(())
}

/// Convert a Bullet quantised.bin to v7 .nnue format (with hidden layers).
pub fn convert_v7(
    input_path: &str,
    output_path: &str,
    use_screlu: bool,
    use_pairwise: bool,
    l1_size: usize,
    l2_size: usize,
    int8_l1: bool,
    bucketed_hidden: bool,
    ft_size_override: usize,
    int16_hidden: bool,
) -> Result<(), String> {
    let data = std::fs::read(input_path).map_err(|e| format!("read {}: {}", input_path, e))?;
    let data_len = strip_footer(&data);

    let l1w_bytes_per = if int8_l1 { 1 } else { 2 };
    // Bucketed: Bullet bakes output buckets into hidden layer dimensions
    // Unbucketed: hidden layers are shared, only output is bucketed
    let bl1 = if bucketed_hidden { NNUE_OUTPUT_BUCKETS * l1_size } else { l1_size };
    let bl2 = if bucketed_hidden { NNUE_OUTPUT_BUCKETS * l2_size } else { l2_size };
    // i16 hidden: all hidden layer weights/biases are i16 (GoChess-style)
    // f32 hidden (default): l1b is f32, l2/l3 are f32
    let hbytes = if int16_hidden { 2 } else { 4 }; // bytes per hidden weight/bias element
    let l1b_bytes = bl1 * hbytes;
    let l2_bytes = if l2_size > 0 { l1_size * bl2 * hbytes + bl2 * hbytes } else { 0 };
    let out_input = if l2_size > 0 { l2_size } else { l1_size };
    // Output: i16 weights at QB + i32 biases for i16 mode, f32 for f32 mode
    let out_bytes = if int16_hidden {
        out_input * NNUE_OUTPUT_BUCKETS * 2 + NNUE_OUTPUT_BUCKETS * 4 // i16 weights + i32 biases
    } else {
        out_input * NNUE_OUTPUT_BUCKETS * 4 + NNUE_OUTPUT_BUCKETS * 4 // f32
    };
    let fixed_bytes = l1b_bytes + l2_bytes + out_bytes;
    // Pairwise: L1 input is H (after CReLU+pairwise+concat = ft_size)
    // Direct: L1 input is 2*H
    let l1_mul = if use_pairwise { 1 } else { 2 };
    let bytes_per_neuron = NNUE_INPUT_SIZE * 2 + 2 + l1_mul * bl1 * l1w_bytes_per;
    let h = if ft_size_override > 0 {
        ft_size_override
    } else if data_len < fixed_bytes {
        return Err(format!("file too small: {} bytes, need at least {} for headers", data_len, fixed_bytes));
    } else {
        (data_len - fixed_bytes) / bytes_per_neuron
    };

    println!("Input: {} bytes, FT={} L1={}{} L2={} (bucketed: {}x{}, {}x{})",
        data.len(), h, l1_size, if int8_l1 { "(i8)" } else { "" }, l2_size, bl1, bl2,
        NNUE_OUTPUT_BUCKETS, l1_size);

    // Verify size
    let l1_input = l1_mul * h;
    let expected = NNUE_INPUT_SIZE * h * 2 + h * 2 + l1_input * bl1 * l1w_bytes_per
        + l1b_bytes + l2_bytes + out_bytes;
    if expected != data_len {
        return Err(format!("Size mismatch: expected {} bytes for FT={}, got {}", expected, h, data_len));
    }

    let mut offset = 0;

    // l0w: [InputSize][H] i16
    let mut input_weights = vec![0i16; NNUE_INPUT_SIZE * h];
    for i in 0..NNUE_INPUT_SIZE * h {
        input_weights[i] = read_i16_le(&data, offset);
        offset += 2;
    }

    // l0b: [H] i16
    let mut input_biases = vec![0i16; h];
    for i in 0..h {
        input_biases[i] = read_i16_le(&data, offset);
        offset += 2;
    }

    // l1w: [l1_input][bl1] — i8 or i16
    // Training configs must NOT use .transpose() on L1/L2/L3 save format.
    // The .nnue loader handles transposition for SIMD internally.
    let mut l1_weights = vec![0i16; l1_input * bl1];
    if int8_l1 {
        for i in 0..l1_input * bl1 {
            l1_weights[i] = data[offset] as i8 as i16;
            offset += 1;
        }
    } else {
        for i in 0..l1_input * bl1 {
            l1_weights[i] = read_i16_le(&data, offset);
            offset += 2;
        }
    }

    // Quantization scales for f32 → int conversion
    // l1w is already quantised (i8@64 or i16@255 by Bullet)
    let qa_l1 = if int8_l1 { 64.0f32 } else { 255.0 };
    let qb = 64.0f32;

    // l1b: [BUCKETS*L1]
    let mut l1_biases = vec![0i16; bl1];
    if int16_hidden {
        // Already quantised as i16 at QA=255
        for i in 0..bl1 {
            l1_biases[i] = read_i16_le(&data, offset);
            offset += 2;
        }
    } else {
        // f32 → i16 scaled by QA_L1
        for i in 0..bl1 {
            let f = read_f32_le(&data, offset);
            l1_biases[i] = (f * qa_l1).round() as i16;
            offset += 4;
        }
    }

    // L2 weights and biases
    let mut l2_weights = vec![0i16; l1_size * bl2];
    let mut l2_biases = vec![0i16; bl2];
    if l2_size > 0 {
        if int16_hidden {
            // Already quantised as i16 at QA=255
            for i in 0..l1_size * bl2 {
                l2_weights[i] = read_i16_le(&data, offset);
                offset += 2;
            }
            for i in 0..bl2 {
                l2_biases[i] = read_i16_le(&data, offset);
                offset += 2;
            }
        } else {
            // f32 → i16 scaled by QA_L1
            for i in 0..l1_size * bl2 {
                let f = read_f32_le(&data, offset);
                l2_weights[i] = (f * qa_l1).round() as i16;
                offset += 4;
            }
            for i in 0..bl2 {
                let f = read_f32_le(&data, offset);
                l2_biases[i] = (f * qa_l1).round() as i16;
                offset += 4;
            }
        }
    }

    // Output weights: [out_input][BUCKETS] → transpose to [BUCKETS][out_input]
    let mut out_w_raw = vec![[0i16; NNUE_OUTPUT_BUCKETS]; out_input];
    if int16_hidden {
        // Already quantised as i16 at QB=64
        for i in 0..out_input {
            for b in 0..NNUE_OUTPUT_BUCKETS {
                out_w_raw[i][b] = read_i16_le(&data, offset);
                offset += 2;
            }
        }
    } else {
        for i in 0..out_input {
            for b in 0..NNUE_OUTPUT_BUCKETS {
                let f = read_f32_le(&data, offset);
                out_w_raw[i][b] = (f * qb).round() as i16;
                offset += 4;
            }
        }
    }
    let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * out_input];
    for b in 0..NNUE_OUTPUT_BUCKETS {
        for i in 0..out_input {
            output_weights[b * out_input + i] = out_w_raw[i][b];
        }
    }

    // Output bias: [BUCKETS]
    let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
    if int16_hidden {
        // Already quantised as i32 at QA*QB
        for b in 0..NNUE_OUTPUT_BUCKETS {
            output_bias[b] = read_i32_le(&data, offset);
            offset += 4;
        }
    } else {
        for b in 0..NNUE_OUTPUT_BUCKETS {
            let f = read_f32_le(&data, offset);
            output_bias[b] = (f * qa_l1 * qb).round() as i32;
            offset += 4;
        }
    }

    println!("Parsed {} bytes of {} (FT={})", offset, data.len(), h);

    // Write v7 .nnue
    let mut buf = Vec::new();
    write_u32_le(&mut buf, NNUE_MAGIC);
    write_u32_le(&mut buf, 7); // v7

    let mut flags = 0u8;
    if use_screlu { flags |= 1; }
    if use_pairwise { flags |= 2; }
    if int8_l1 { flags |= 4; }
    if bucketed_hidden { flags |= 8; } // bit 3 = bucketed hidden layers
    buf.push(flags);
    write_u16_le(&mut buf, h as u16);       // FT size
    write_u16_le(&mut buf, l1_size as u16); // per-bucket L1 size
    write_u16_le(&mut buf, l2_size as u16); // per-bucket L2 size

    // Write weights — hidden layers have bucketed dimensions
    for &w in &input_weights { write_i16_le(&mut buf, w); }
    for &b in &input_biases { write_i16_le(&mut buf, b); }
    for &w in &l1_weights { write_i16_le(&mut buf, w); } // [L1_input][BUCKETS*L1]
    for &b in &l1_biases { write_i16_le(&mut buf, b); }   // [BUCKETS*L1]
    if l2_size > 0 {
        for &w in &l2_weights { write_i16_le(&mut buf, w); } // [L1][BUCKETS*L2]
        for &b in &l2_biases { write_i16_le(&mut buf, b); }   // [BUCKETS*L2]
    }
    for &w in &output_weights { write_i16_le(&mut buf, w); } // [BUCKETS][L2]
    for &b in &output_bias { write_i32_le(&mut buf, b); }     // [BUCKETS]

    std::fs::File::create(output_path)
        .and_then(|mut f| f.write_all(&buf))
        .map_err(|e| format!("write {}: {}", output_path, e))?;

    println!("Saved {} ({} bytes, v7 FT={} L1={} L2={})", output_path, buf.len(), h, l1_size, l2_size);
    Ok(())
}
