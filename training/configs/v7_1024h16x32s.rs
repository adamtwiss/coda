/// Coda v7 1024 SCReLU + hidden layers (16→32) training config
/// Architecture: (768×16 → 1024)×2 → SCReLU → concat 2048 → L1(16) → L2(32) → 1×8
/// Requires LR warmup to prevent hidden layer collapse.
///
/// Usage on GPU host:
///   cd ~/code/bullet
///   cargo run --release --example coda_v7_1024h16x32s -- \
///     --dataset /workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack \
///     --superbatches 800 --wdl 0.0 --warmup 20 --save-rate 100
///
/// Convert:
///   coda convert-bullet -input quantised.bin -output net.nnue -screlu -hidden 16 -hidden2 32 -int8l1
/// Note: hidden layers are now shared (unbucketed). Do NOT pass --bucketed-hidden.
/// Name: net-v7-1024h16x32s-w0-e800sNNN.nnue
///
/// CRITICAL: Monitor hidden layer health at every checkpoint (check-net).
/// CReLU kills hidden neurons — always use SCReLU for v7.
/// LR warmup is essential — without it, L1 neurons die by SB40.

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
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


use sfbinpack::chess::{piecetype::PieceType, r#move::MoveType};
};

fn main() {
    let ft_size = 1024;
    let l1_size = 16;
    let l2_size = 32;

    let args: Vec<String> = std::env::args().collect();
    let dataset_path = get_arg(&args, "--dataset", "/workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack");
    let superbatches: usize = get_arg(&args, "--superbatches", "800").parse().unwrap();
    let wdl_proportion: f32 = get_arg(&args, "--wdl", "0.0").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "20").parse().unwrap();
    let save_rate: usize = get_arg(&args, "--save-rate", "100").parse().unwrap();
    let loss_power: f32 = get_arg(&args, "--loss-power", "2.5").parse().unwrap();
    let final_lr = initial_lr * 0.3f32.powi(5); // Bullet example: ~2.43e-6

    const NUM_OUTPUT_BUCKETS: usize = 8;

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
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.repeat(NUM_INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(move |output, target| output.sigmoid().power_error(target, loss_power))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            // Shared hidden layers (not bucketed) — only output is bucketed
            // Bucketing L1/L2 starves them of gradient signal (16×32 = 512 params per bucket)
            let l1 = builder.new_affine("l1", 2 * ft_size, l1_size);
            let l2 = builder.new_affine("l2", l1_size, l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            let stm = l0.forward(stm_inputs).screlu();
            let ntm = l0.forward(ntm_inputs).screlu();
            let h1 = l1.forward(stm.concat(ntm)).screlu();
            let h2 = l2.forward(h1).screlu();
            l3.forward(h2).select(output_buckets)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    // LR schedule: warmup then cosine decay (critical for hidden layers)
    let schedule = TrainingSchedule {
        net_id: "coda-v7-1024h16x32s".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: lr::Sequence {
            first: Box::new(lr::LinearDecayLR {
                initial_lr: initial_lr * 0.1,
                final_lr: initial_lr,
                final_superbatch: warmup_sbs,
            }),
            second: Box::new(lr::CosineDecayLR {
                initial_lr,
                final_lr,
                final_superbatch: superbatches - warmup_sbs,
            }),
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

    // Standard filter: quiet non-tactical positions only.
    let filter = |entry: &sfbinpack::TrainingDataEntry| {
        let stm = entry.pos.side_to_move();
        entry.ply >= 16
            && !entry.pos.is_checked(stm)
            && entry.score.unsigned_abs() <= 10000
            && entry.mv.mtype() == MoveType::Normal
            && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
    };

    let dataloader = SfBinpackLoader::new(&dataset_path, 256, 4, filter);

    println!("=== Coda v7 1024 SCReLU + Hidden Layers ===");
    println!("FT: {} → SCReLU, L1: {}, L2: {}, Output: 1×{}", ft_size, l1_size, l2_size, NUM_OUTPUT_BUCKETS);
    println!("Warmup: {} SBs, Schedule: {} SBs, WDL: {}, Loss: power({})", warmup_sbs, superbatches, wdl_proportion, loss_power);
    println!("Data: {}", dataset_path);
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
