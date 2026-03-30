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
) -> Result<(), String> {
    let data = std::fs::read(input_path).map_err(|e| format!("read {}: {}", input_path, e))?;
    let data_len = strip_footer(&data);

    // Infer hidden size
    let out_mul = if use_pairwise { NNUE_OUTPUT_BUCKETS } else { NNUE_OUTPUT_BUCKETS * 2 };
    let bytes_per_neuron = NNUE_INPUT_SIZE * 2 + 2 + out_mul * 2;
    let bias_bytes = NNUE_OUTPUT_BUCKETS * 4;
    let h = (data_len - bias_bytes) / bytes_per_neuron;
    let output_width = if use_pairwise { h } else { 2 * h };

    let expected = NNUE_INPUT_SIZE * h * 2 + h * 2 + output_width * NNUE_OUTPUT_BUCKETS * 2 + NNUE_OUTPUT_BUCKETS * 4;
    println!("Input: {} bytes, hidden size: {}, expected: {} bytes", data.len(), h, expected);
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

    // Read output weights: [outputWidth][buckets] i16 — transpose to [bucket][outputWidth]
    let mut out_w_raw = vec![[0i16; NNUE_OUTPUT_BUCKETS]; output_width];
    for i in 0..output_width {
        for b in 0..NNUE_OUTPUT_BUCKETS {
            out_w_raw[i][b] = read_i16_le(&data, offset);
            offset += 2;
        }
    }
    let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * output_width];
    for b in 0..NNUE_OUTPUT_BUCKETS {
        for i in 0..output_width {
            output_weights[b * output_width + i] = out_w_raw[i][b];
        }
    }

    // Read output bias: [buckets] i32
    let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
    for b in 0..NNUE_OUTPUT_BUCKETS {
        output_bias[b] = read_i32_le(&data, offset);
        offset += 4;
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
) -> Result<(), String> {
    let data = std::fs::read(input_path).map_err(|e| format!("read {}: {}", input_path, e))?;
    let data_len = strip_footer(&data);

    let out_input_width = if l2_size > 0 { l2_size } else { l1_size };
    let l1w_bytes_per = if int8_l1 { 1 } else { 2 };
    let l2_bytes = if l2_size > 0 { l1_size * l2_size * 2 + l2_size * 2 } else { 0 };
    let fixed_bytes = l1_size * 2 + l2_bytes + out_input_width * NNUE_OUTPUT_BUCKETS * 2 + NNUE_OUTPUT_BUCKETS * 4;
    let bytes_per_neuron = NNUE_INPUT_SIZE * 2 + 2 + 2 * l1_size * l1w_bytes_per;
    let h = (data_len - fixed_bytes) / bytes_per_neuron;

    let l1_scale: i32 = if int8_l1 { 64 } else { 255 };

    println!("Input: {} bytes, FT={} L1={}{} L2={}", data.len(), h, l1_size,
        if int8_l1 { "(i8)" } else { "" }, l2_size);

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

    // l1w: [2*H][L1] — int8 or int16
    let mut l1_weights = vec![0i16; 2 * h * l1_size];
    if int8_l1 {
        for i in 0..2 * h * l1_size {
            l1_weights[i] = data[offset] as i8 as i16;
            offset += 1;
        }
    } else {
        for i in 0..2 * h * l1_size {
            l1_weights[i] = read_i16_le(&data, offset);
            offset += 2;
        }
    }

    // l1b: [L1] i16
    let mut l1_biases = vec![0i16; l1_size];
    for i in 0..l1_size {
        l1_biases[i] = read_i16_le(&data, offset);
        offset += 2;
    }

    // L2 (if present)
    let mut l2_weights = vec![0i16; l1_size * l2_size];
    let mut l2_biases = vec![0i16; l2_size];
    if l2_size > 0 {
        for i in 0..l1_size * l2_size {
            l2_weights[i] = read_i16_le(&data, offset);
            offset += 2;
        }
        for i in 0..l2_size {
            l2_biases[i] = read_i16_le(&data, offset);
            offset += 2;
        }
    }

    // Output weights: [outInputWidth][buckets] i16 — transpose
    let mut out_w_raw = vec![[0i16; NNUE_OUTPUT_BUCKETS]; out_input_width];
    for i in 0..out_input_width {
        for b in 0..NNUE_OUTPUT_BUCKETS {
            out_w_raw[i][b] = read_i16_le(&data, offset);
            offset += 2;
        }
    }
    let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * out_input_width];
    for b in 0..NNUE_OUTPUT_BUCKETS {
        for i in 0..out_input_width {
            output_weights[b * out_input_width + i] = out_w_raw[i][b];
        }
    }

    // Output bias: [buckets] i32
    let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
    for b in 0..NNUE_OUTPUT_BUCKETS {
        output_bias[b] = read_i32_le(&data, offset);
        offset += 4;
    }

    println!("Parsed {} bytes of {}", offset, data.len());

    // Write v7 .nnue
    let mut buf = Vec::new();
    write_u32_le(&mut buf, NNUE_MAGIC);
    write_u32_le(&mut buf, 7); // v7

    let mut flags = 0u8;
    if use_screlu { flags |= 1; }
    if use_pairwise { flags |= 2; }
    if int8_l1 { flags |= 4; }
    buf.push(flags);
    write_u16_le(&mut buf, h as u16);
    write_u16_le(&mut buf, l1_size as u16);
    write_u16_le(&mut buf, l2_size as u16);

    for &w in &input_weights { write_i16_le(&mut buf, w); }
    for &b in &input_biases { write_i16_le(&mut buf, b); }
    for &w in &l1_weights { write_i16_le(&mut buf, w); }
    for &b in &l1_biases { write_i16_le(&mut buf, b); }
    if l2_size > 0 {
        for &w in &l2_weights { write_i16_le(&mut buf, w); }
        for &b in &l2_biases { write_i16_le(&mut buf, b); }
    }
    for &w in &output_weights { write_i16_le(&mut buf, w); }
    for &b in &output_bias { write_i32_le(&mut buf, b); }

    std::fs::File::create(output_path)
        .and_then(|mut f| f.write_all(&buf))
        .map_err(|e| format!("write {}: {}", output_path, e))?;

    println!("Saved {} ({} bytes, v7 FT={} L1={} L2={})", output_path, buf.len(), h, l1_size, l2_size);
    Ok(())
}
