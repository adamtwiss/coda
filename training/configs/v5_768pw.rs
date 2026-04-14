/// Coda v5 768 pairwise training config
/// Architecture: (768×16 → 768)×2 → CReLU → pairwise → 384×2 = 768 → 1×8
/// Pairwise variant — matches 1024s in self-play with fewer parameters.
///
/// Usage on GPU host:
///   cd ~/code/bullet
///   cargo run --release --example coda_v5_768pw -- \
///     --dataset /workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack \
///     --superbatches 800 --wdl 0.07 --save-rate 100
///
/// Key training findings:
/// - Low final LR (0.001 * 0.3^5 = 2.43e-6): +47 Elo vs old 0.0001
/// - Filtering (quiet positions only): +22 untuned, +48 with retune
/// - Power-2.5 loss: +17 Elo vs MSE (2026-04-12, OB #291 H1)
/// - Combined: +80 Elo over baseline at s120
///
/// Output: quantised.bin in checkpoints/
///
/// Convert:
///   coda convert-bullet -v5 -input quantised.bin -output net.nnue -pairwise
/// Name: net-v5-768pw-w7-e800sNNN.nnue

use bullet_lib::{
    game::{
        inputs::ChessBucketsMirrored,
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::SfBinpackLoader},
};

use sfbinpack::chess::{piecetype::PieceType, r#move::MoveType};

fn main() {
    let ft_size = 768; // after pairwise: 384 per perspective, 768 concat

    let args: Vec<String> = std::env::args().collect();
    let dataset_dir = get_arg(&args, "--dataset-dir", "/workspace/data");
    let superbatches: usize = get_arg(&args, "--superbatches", "800").parse().unwrap();
    let wdl_proportion: f32 = get_arg(&args, "--wdl", "0.07").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let final_lr: f32 = get_arg(&args, "--final-lr", &format!("{}", initial_lr * 0.3f32.powi(5))).parse().unwrap();
    let loss_power: f32 = get_arg(&args, "--loss-power", "2.5").parse().unwrap();
    let save_rate: usize = get_arg(&args, "--save-rate", "100").parse().unwrap();
    let bucket_layout_name = get_arg(&args, "--bucket-layout", "uniform");

    const NUM_OUTPUT_BUCKETS: usize = 8;

    // Uniform: original Coda layout (2 ranks × 1 file per bucket)
    #[rustfmt::skip]
    const LAYOUT_UNIFORM: [usize; 32] = [
         0,  4,  8, 12,
         0,  4,  8, 12,
         1,  5,  9, 13,
         1,  5,  9, 13,
         2,  6, 10, 14,
         2,  6, 10, 14,
         3,  7, 11, 15,
         3,  7, 11, 15,
    ];

    // Consensus: fine-near, coarse-far (Alexandria/Viridithas pattern)
    // Ranks 1-2: per-square (8 buckets), ranks 3-4: shared (4), ranks 5-8: 2x2 (4)
    #[rustfmt::skip]
    const LAYOUT_CONSENSUS: [usize; 32] = [
         0,  1,  2,  3,   // rank 1: per-square (4 buckets)
         4,  5,  6,  7,   // rank 2: per-square (4 buckets)
         8,  9, 10, 11,   // rank 3-4: per-square, 2 ranks share (4 buckets)
         8,  9, 10, 11,
        12, 12, 13, 13,   // rank 5-6: 2x2 groups (2 buckets)
        12, 12, 13, 13,
        14, 14, 15, 15,   // rank 7-8: 2x2 groups (2 buckets)
        14, 14, 15, 15,
    ];

    let bucket_layout: [usize; 32] = match bucket_layout_name.as_str() {
        "uniform" => LAYOUT_UNIFORM,
        "consensus" => LAYOUT_CONSENSUS,
        _ => panic!("Unknown bucket layout '{}'. Use 'uniform' or 'consensus'.", bucket_layout_name),
    };

    // Both layouts use 16 buckets (max index = 15)
    let num_input_buckets: usize = 16;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .use_threads(4)
        .inputs(ChessBucketsMirrored::new(bucket_layout))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(move |store, weights| {
                    let factoriser = store.get("l0f").values.repeat(num_input_buckets);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            // Output: 768 inputs (384 per perspective after pairwise, concatenated)
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i32>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().power_error(target, 2.5))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(num_input_buckets);

            let mut l0 = builder.new_affine("l0", 768 * num_input_buckets, ft_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            // After pairwise: 384 per perspective, 768 concatenated
            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS);

            let stm = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm = l0.forward(ntm_inputs).crelu().pairwise_mul();
            l1.forward(stm.concat(ntm)).select(output_buckets)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: "coda-v5-768pw".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate,
    };

    let settings = LocalSettings {
        threads: 4,
        batch_queue_size: 32,
        output_directory: "checkpoints",
        test_set: None,
    };

    // Standard filter: quiet non-tactical positions only (Bullet example pattern).
    // Skip openings (ply < 16), in-check, captures, and tactical moves.
    // NNUE eval is only called at quiet nodes — train on what matters.
    let filter = |entry: &sfbinpack::TrainingDataEntry| {
        let stm = entry.pos.side_to_move();
        entry.ply >= 16
            && !entry.pos.is_checked(stm)
            && entry.score.unsigned_abs() <= 10000
            && entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
    };

    let data_files: Vec<String> = std::fs::read_dir(&dataset_dir)
        .unwrap_or_else(|e| panic!("Cannot read dataset dir {}: {}", dataset_dir, e))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().map_or(false, |ext| ext == "binpack") && path.file_name().map_or(false, |n| n.to_string_lossy().starts_with("test80")) {
                Some(path.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    assert!(!data_files.is_empty(), "No test80*.binpack files found in {}", dataset_dir);
    let data_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();

    let dataloader = SfBinpackLoader::new_concat_multiple(&data_refs, 256, 4, filter);

    println!("=== Coda v5 768 Pairwise ===");
    println!("FT: {} → CReLU → pairwise → {} per perspective", ft_size, ft_size / 2);
    println!("King buckets: {} ({})", num_input_buckets, bucket_layout_name);
    println!("Schedule: {} SBs, WDL: {}, LR: {}→{}, Loss: power({})", superbatches, wdl_proportion, initial_lr, final_lr, loss_power);
    println!("Data: {} ({} files)", dataset_dir, data_files.len());
    println!();

    trainer.run(&schedule, &settings, &dataloader);
}

fn get_arg(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|s| s == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}
