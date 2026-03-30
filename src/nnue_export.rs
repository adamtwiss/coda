/// Convert .nnue file to Bullet checkpoint format for transfer learning.
///
/// Reads a v5/v6 .nnue file, dequantises the FT weights (l0w, l0b) to float32,
/// and writes a Bullet-compatible checkpoint directory with:
///   - weights.bin: l0w, l0b from .nnue + zeroed L1/L2/output for v7
///   - momentum.bin: all zeros
///   - velocity.bin: all zeros
///
/// The v7 training config then loads this checkpoint, freezes l0w/l0b,
/// and trains only the hidden layers.

use std::io::Write;

const NNUE_INPUT_SIZE: usize = 12288; // 768 * 16 king buckets
const NNUE_OUTPUT_BUCKETS: usize = 8;

/// Write a single weight entry in Bullet checkpoint format.
/// Format: [id + '\n'] [size: usize LE] [f32 × size]
fn write_weight_entry(buf: &mut Vec<u8>, id: &str, values: &[f32]) {
    buf.extend_from_slice(id.as_bytes());
    buf.push(b'\n');
    buf.extend_from_slice(&values.len().to_le_bytes());
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Convert a .nnue file to a Bullet checkpoint for v7 transfer learning.
///
/// Args:
///   nnue_path: path to the v5/v6 .nnue file (source of FT weights)
///   output_dir: path to the output checkpoint directory
///   l1_size: L1 hidden layer size for the v7 architecture
///   l2_size: L2 hidden layer size (0 = no L2)
pub fn nnue_to_bullet_checkpoint(
    nnue_path: &str,
    output_dir: &str,
    ft_size: usize,
    l1_size: usize,
    l2_size: usize,
) -> Result<(), String> {
    // Load the .nnue net to get FT weights
    let net = crate::nnue::NNUENet::load(nnue_path)?;
    let h = net.hidden_size;

    if h != ft_size {
        return Err(format!("FT size mismatch: .nnue has {}, expected {}", h, ft_size));
    }

    println!("Loaded .nnue: FT size = {}, SCReLU = {}", h, net.use_screlu);

    // Dequantise FT weights: int16 at scale QA=255 → float32
    let qa = 255.0f32;
    let l0w: Vec<f32> = net.input_weights.iter().map(|&w| w as f32 / qa).collect();
    let l0b: Vec<f32> = net.input_biases.iter().map(|&b| b as f32 / qa).collect();

    println!("FT weights: {} values ({}×{})", l0w.len(), NNUE_INPUT_SIZE, h);
    println!("FT biases: {} values", l0b.len());

    // Determine output layer size
    let out_input_size = if l2_size > 0 { l2_size } else { l1_size };

    // Create zeroed weights for hidden layers
    let l1w = vec![0.0f32; 2 * h * l1_size];  // [2*FT_size × L1]
    let l1b = vec![0.0f32; l1_size];
    let l2w = if l2_size > 0 { vec![0.0f32; l1_size * l2_size] } else { Vec::new() };
    let l2b = if l2_size > 0 { vec![0.0f32; l2_size] } else { Vec::new() };
    let out_w = vec![0.0f32; NNUE_OUTPUT_BUCKETS * out_input_size];
    let out_b = vec![0.0f32; NNUE_OUTPUT_BUCKETS];

    // Create output directory
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create directory {}: {}", output_dir, e))?;

    // Write weights.bin
    let mut weights_buf = Vec::new();
    write_weight_entry(&mut weights_buf, "l0w", &l0w);
    write_weight_entry(&mut weights_buf, "l0b", &l0b);
    write_weight_entry(&mut weights_buf, "l1w", &l1w);
    write_weight_entry(&mut weights_buf, "l1b", &l1b);
    if l2_size > 0 {
        write_weight_entry(&mut weights_buf, "l2w", &l2w);
        write_weight_entry(&mut weights_buf, "l2b", &l2b);
    }
    // Output layer — name depends on architecture
    let out_name_w = if l2_size > 0 { "l3w" } else { "l2w" };
    let out_name_b = if l2_size > 0 { "l3b" } else { "l2b" };
    // Only write output if not already written as l2
    if l2_size > 0 {
        write_weight_entry(&mut weights_buf, out_name_w, &out_w);
        write_weight_entry(&mut weights_buf, out_name_b, &out_b);
    } else {
        // l2w/l2b is the output layer when no L2 hidden
        // Already written above as l2w/l2b with zero init
    }

    let weights_path = format!("{}/weights.bin", output_dir);
    std::fs::File::create(&weights_path)
        .and_then(|mut f| f.write_all(&weights_buf))
        .map_err(|e| format!("Failed to write {}: {}", weights_path, e))?;

    // Write momentum.bin and velocity.bin (all zeros, same structure)
    let mut zero_buf = Vec::new();
    write_weight_entry(&mut zero_buf, "l0w", &vec![0.0f32; l0w.len()]);
    write_weight_entry(&mut zero_buf, "l0b", &vec![0.0f32; l0b.len()]);
    write_weight_entry(&mut zero_buf, "l1w", &vec![0.0f32; l1w.len()]);
    write_weight_entry(&mut zero_buf, "l1b", &vec![0.0f32; l1b.len()]);
    if l2_size > 0 {
        write_weight_entry(&mut zero_buf, "l2w", &vec![0.0f32; l2w.len()]);
        write_weight_entry(&mut zero_buf, "l2b", &vec![0.0f32; l2b.len()]);
        write_weight_entry(&mut zero_buf, out_name_w, &vec![0.0f32; out_w.len()]);
        write_weight_entry(&mut zero_buf, out_name_b, &vec![0.0f32; out_b.len()]);
    }

    for name in &["momentum.bin", "velocity.bin"] {
        let path = format!("{}/{}", output_dir, name);
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&zero_buf))
            .map_err(|e| format!("Failed to write {}: {}", path, e))?;
    }

    let total_params = l0w.len() + l0b.len() + l1w.len() + l1b.len()
        + l2w.len() + l2b.len() + out_w.len() + out_b.len();

    println!("Wrote Bullet checkpoint to {}/", output_dir);
    println!("  weights.bin: {} params ({:.1} MB)", total_params, total_params as f64 * 4.0 / 1e6);
    println!("  momentum.bin + velocity.bin: zeroed");
    println!("  FT (l0w, l0b): dequantised from .nnue");
    println!("  L1 ({l1_size}), L2 ({l2_size}), output: zero-initialised");
    println!();
    println!("Usage in Bullet training config:");
    println!("  trainer.load_from_checkpoint(\"{}\");", output_dir);
    println!("  trainer.optimiser.freeze(\"l0w\");");
    println!("  trainer.optimiser.freeze(\"l0b\");");

    Ok(())
}
