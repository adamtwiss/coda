/// Experiment: v7 1024 SCReLU — no factoriser, i16 quant (GoChess-aligned)
/// Architecture: (768×16 → 1024)×2 → SCReLU → concat 2048 → L1(16) → L2(32) → 1×8
///
/// Isolates why our v7 hidden layers collapse by aligning with GoChess on
/// the 3 structural differences while keeping our proven training improvements:
///   CHANGED (vs standard v7 config):
///     1. No factoriser (no l0f, no init_with_effective_input_size)
///     2. L1 quantised as i16 at QA=255 (was i8 at 64)
///     3. L2/L3 quantised as i16 (was float)
///   KEPT (our improvements):
///     - Position filtering (ply≥16, quiet only)
///     - Power-2.5 loss
///     - Low final LR
///     - LR warmup
///
/// Usage:
///   cd ~/code/bullet
///   cargo run --release --example coda_v7_gochess_style -- \
///     --dataset-dir /workspace/data --superbatches 100 --save-rate 20
///
/// Convert (note: NO --int8l1 since L1 is i16):
///   coda convert-bullet -input quantised.bin -output net.nnue -screlu -hidden 16 -hidden2 32

use bullet_lib::{
    game::{
        inputs::ChessBucketsMirrored,
        outputs::MaterialCount,
    },
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::SfBinpackLoader},
};

use sfbinpack::chess::{piecetype::PieceType, r#move::MoveType};

const FT_SIZE: usize = 1024;
const L1_SIZE: usize = 16;
const L2_SIZE: usize = 32;
const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dataset_dir = get_arg(&args, "--dataset-dir", "/workspace/data");
    let superbatches: usize = get_arg(&args, "--superbatches", "100").parse().unwrap();
    let save_rate: usize = get_arg(&args, "--save-rate", "20").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "20").parse().unwrap();
    let final_lr = initial_lr * 0.3f32.powi(5); // low final LR (proven +47 Elo)

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

    // No factoriser — plain FT like GoChess
    // i16 quantisation for L1/L2/L3 like GoChess
    let trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QA),
            SavedFormat::id("l1b").round().quantise::<i16>(QA),
            SavedFormat::id("l2w").round().quantise::<i16>(QA),
            SavedFormat::id("l2b").round().quantise::<i16>(QA),
            SavedFormat::id("l3w").round().quantise::<i16>(QB),
            SavedFormat::id("l3b").round().quantise::<i32>(QA as i32 * QB as i32),
        ])
        .loss_fn(|output, target| output.sigmoid().power_error(target, 2.5))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0 = builder.new_affine("l0", 768 * 16, FT_SIZE);
            let l1 = builder.new_affine("l1", 2 * FT_SIZE, L1_SIZE);
            let l2 = builder.new_affine("l2", L1_SIZE, L2_SIZE);
            let l3 = builder.new_affine("l3", L2_SIZE, NUM_OUTPUT_BUCKETS);

            let stm = l0.forward(stm_inputs).screlu();
            let ntm = l0.forward(ntm_inputs).screlu();
            let h1 = l1.forward(stm.concat(ntm)).screlu();
            let h2 = l2.forward(h1).screlu();
            l3.forward(h2).select(output_buckets)
        });

    // LR warmup + cosine decay with low final LR
    let schedule = TrainingSchedule {
        net_id: "coda-v7-gochess-style".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
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
        save_rate,
    };

    let settings = LocalSettings {
        threads: 4,
        batch_queue_size: 32,
        output_directory: "checkpoints",
        test_set: None,
    };

    // Our proven filter: quiet non-tactical positions only
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
            if path.extension().map_or(false, |ext| ext == "binpack") {
                Some(path.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    assert!(!data_files.is_empty(), "No .binpack files found in {}", dataset_dir);
    let data_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();

    let dataloader = SfBinpackLoader::new_concat_multiple(&data_refs, 256, 4, filter);

    println!("=== Coda v7 GoChess-Style Experiment ===");
    println!("FT: {} → SCReLU (no factoriser), L1: {} (i16), L2: {} (i16)", FT_SIZE, L1_SIZE, L2_SIZE);
    println!("Warmup: {} SBs, Schedule: {} SBs, LR: {}→{}", warmup_sbs, superbatches, initial_lr, final_lr);
    println!("Loss: power(2.5), Filter: quiet positions, Data: {} ({} files)", dataset_dir, data_files.len());
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
