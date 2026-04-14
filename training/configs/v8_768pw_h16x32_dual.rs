/// Coda v8 pairwise training config — dual L1 activation
/// Architecture: (768×16 → 1536)×2 → pairwise → 768×2 → L1(16) → dual(CReLU+SCReLU=32) → L2(32) → 1×8
///
/// The dual activation applies CReLU and SCReLU to L1 output and concatenates them,
/// doubling L2 input from 16 to 32 without changing L1 weights. This matches
/// Alexandria and Obsidian's approach — 4/5 top engines use dual L1 activation.
///
/// Usage on GPU host:
///   cd ~/code/bullet
///   cargo run --release --example coda_v8_768pw_dual -- \
///     --dataset-dir /workspace/data \
///     --superbatches 400 \
///     --wdl 0.1 \
///     --lr 0.001 \
///     --warmup 20
///
/// Output: quantised.bin in checkpoints/coda-v8-768pw-dual/
///
/// Convert (after pulling latest Coda):
///   coda convert-bullet -input quantised.bin -output net.nnue -pairwise -hidden 16 -hidden2 32 -int8l1 -dual
///
/// Name output as: net-v8-768pwdh16x32-wNN-eNNNsNNN.nnue
///   'd' after 'pw' denotes dual L1 activation

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
    // Then: 768×2 concat → L1(16) → dual(CReLU+SCReLU=32) → L2(32) → output(8 buckets)
    let ft_size = 1536;      // FT output before pairwise (after pairwise: 768)
    let l1_size = 16;        // L1 output neurons
    let l2_input = l1_size * 2; // 32: dual activation doubles L1→L2 connection
    let l2_size = 32;

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let dataset_dir = get_arg(&args, "--dataset-dir", "/workspace/data");
    let superbatches: usize = get_arg(&args, "--superbatches", "400").parse().unwrap();
    let wdl_proportion: f32 = get_arg(&args, "--wdl", "0.1").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "20").parse().unwrap();
    let final_lr = initial_lr * 0.3f32.powi(5); // ~2.43e-6

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
        .loss_fn(|output, target| output.sigmoid().power_error(target, 2.5))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // No factoriser — plain FT (factoriser kills hidden layers)
            let l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);

            // L1: 1536 → 16 (same as v7)
            let l1 = builder.new_affine("l1", ft_size, l1_size);
            // L2: 32 → 32 (input doubled by dual activation)
            let l2 = builder.new_affine("l2", l2_input, l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            // Forward: FT → CReLU → pairwise → concat → L1 → dual(CReLU+SCReLU) → L2 → SCReLU → output
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let l1_raw = l1.forward(hl1);
            // Dual activation: CReLU(L1) concat SCReLU(L1) → 32 inputs for L2
            let l1_crelu = l1_raw.crelu();
            let l1_screlu = l1_raw.screlu();
            let hl2 = l1_crelu.concat(l1_screlu);
            let hl3 = l2.forward(hl2).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);

    // LR schedule: warmup then cosine decay
    let schedule = TrainingSchedule {
        net_id: "coda-v8-768pw-dual".to_string(),
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

    println!("=== Coda v8 768pw Dual Activation Training ===");
    println!("FT: {} → CReLU → pairwise → {}", ft_size, ft_size / 2);
    println!("L1: {} → dual(CReLU+SCReLU) → {}", l1_size, l2_input);
    println!("L2: {} → {} → 1×{}", l2_size, l2_size, NUM_OUTPUT_BUCKETS);
    println!("Warmup: {} SBs, Schedule: {} SBs, WDL: {}", warmup_sbs, superbatches, wdl_proportion);
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
