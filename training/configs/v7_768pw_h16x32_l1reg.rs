/// Coda v7 pairwise training config WITH L1 ACTIVATION REGULARIZATION
/// Architecture: (768×16 → 1536)×2 → pairwise → 768×2 → 16 → 32 → 1×8
/// This matches the consensus architecture used by Reckless, Obsidian, Viridithas, Stormphrax.
///
/// L1 regularization encourages sparsity in the FT accumulator activations.
/// Sparse accumulators enable chunk-skipping SIMD in inference (~2× NPS gain).
/// Lambda controls the regularization strength — too high kills accuracy, too low has no effect.
/// Start with lambda=0.001 and check sparsity at s100.
///
/// Usage on GPU host:
///   cd ~/code/bullet
///   cargo run --release --example coda_v7_768pw_l1reg -- \
///     --dataset-dir /training/sf \
///     --superbatches 100 \
///     --wdl 0.0 \
///     --lr 0.001 \
///     --warmup 20 \
///     --lambda 0.001
///
/// Output: quantised.bin in checkpoints/
///
/// Convert (after pulling latest Coda):
///   coda convert-bullet -input quantised.bin -output net.nnue -pairwise -hidden 16 -hidden2 32 -int8l1
/// Check-net:
///   tuner check-net net.nnue
/// Name output as: net-v7-768pwh16x32s-w0-e100sNNN-l1reg.nnue

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
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
    // Architecture: (768×16 → 1536) → CReLU → pairwise → 768 per perspective
    // Then: 768×2 concat → L1(16) → L2(32) → output(8 buckets)
    let ft_size = 1536;      // FT output before pairwise (after pairwise: 768)
    let l1_size = 16;
    let l2_size = 32;

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let dataset_dir = get_arg(&args, "--dataset-dir", "/workspace/data");
    let superbatches: usize = get_arg(&args, "--superbatches", "100").parse().unwrap();
    let wdl_proportion: f32 = get_arg(&args, "--wdl", "0.0").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "20").parse().unwrap();
    let lambda: f32 = get_arg(&args, "--lambda", "0.001").parse().unwrap();
    let final_lr = initial_lr * 0.3f32.powi(5); // Bullet example: ~2.43e-6

    const NUM_OUTPUT_BUCKETS: usize = 8;

    // GoChess/Coda king bucket layout: mirroredFile * 4 + rankGroup
    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
         0,  4,  8, 12,
         0,  4,  8, 12,
         1,  5,  9, 13,
         1,  5,  9, 13,
         2,  6, 10, 14,
         2,  6, 10, 14,
         3,  7, 11, 15,
         3,  7, 11, 15,
    ];

    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .use_threads(4)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i8>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), targets| {
            // No factoriser — plain FT (factoriser kills hidden layers)
            let l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);

            // Shared hidden layers (not bucketed) — only output is bucketed
            let l1 = builder.new_affine("l1", ft_size, l1_size);
            let l2 = builder.new_affine("l2", l1_size, l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            // Forward: FT → CReLU (branch for L1 penalty) → pairwise → concat → hidden → output
            let stm_crelu = l0.forward(stm_inputs).crelu();
            let stm_for_reg = stm_crelu.copy();
            let stm_hidden = stm_crelu.pairwise_mul();

            let ntm_crelu = l0.forward(ntm_inputs).crelu();
            let ntm_for_reg = ntm_crelu.copy();
            let ntm_hidden = ntm_crelu.pairwise_mul();

            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).screlu();
            let hl3 = l2.forward(hl2).screlu();
            let output = l3.forward(hl3).select(output_buckets);

            // Main loss: power-2.5 error
            let main_loss = output.sigmoid().power_error(targets, 2.5);

            // L1 activation regularization on FT outputs (encourages sparsity)
            // abs_pow(1.0) = L1 norm, reduce across batch to get scalar penalty
            let stm_l1 = stm_for_reg.abs_pow(1.0).reduce_avg_across_batch();
            let ntm_l1 = ntm_for_reg.abs_pow(1.0).reduce_avg_across_batch();
            let reg_penalty = stm_l1.linear_comb(lambda, ntm_l1, lambda);

            // Total loss = main_loss + lambda * (stm_l1 + ntm_l1)
            let total_loss = main_loss.linear_comb(1.0, reg_penalty, 1.0);

            (output, total_loss)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);

    // LR schedule: warmup then cosine decay
    let schedule = TrainingSchedule {
        net_id: "coda-v7-768pw".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::Sequence {
            first: lr::LinearDecayLR {
                initial_lr: initial_lr * 0.1,
                final_lr: initial_lr,
                final_superbatch: warmup_sbs,
            },
            second: lr::CosineDecayLR {
                initial_lr,
                final_lr,
                final_superbatch: superbatches - warmup_sbs,
            },
            first_scheduler_final_superbatch: warmup_sbs,
        },
        save_rate: 100,
    };

    let settings = LocalSettings {
        threads: 4,
        batch_queue_size: 32,
        output_directory: "checkpoints",
        test_set: None,
    };

    // Standard filter: quiet non-tactical positions only.
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

    println!("=== Coda v7 768pw Training (L1 Reg) ===");
    println!("FT: {} → CReLU → pairwise → {}", ft_size, ft_size / 2);
    println!("Hidden: {} → {} → 1×{}", l1_size, l2_size, NUM_OUTPUT_BUCKETS);
    println!("Warmup: {} SBs, Schedule: {} SBs, WDL: {}", warmup_sbs, superbatches, wdl_proportion);
    println!("L1 reg lambda: {}", lambda);
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
