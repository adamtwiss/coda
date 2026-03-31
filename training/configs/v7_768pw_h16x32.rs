/// Coda v7 pairwise training config
/// Architecture: (768×16 → 1536)×2 → pairwise → 768×2 → 16 → 32 → 1×8
/// This matches the consensus architecture used by Reckless, Obsidian, Viridithas, Stormphrax.
///
/// Usage on GPU host:
///   cd ~/code/bullet
///   cargo run --release --example coda_v7_768pw -- \
///     --dataset /training/sf/test80-2024-01-jan-2tb7p.min-v2.v6.binpack \
///     --superbatches 800 \
///     --wdl 0.0 \
///     --lr 0.001 \
///     --warmup 20
///
/// Output: quantised.bin in coda/nets/v7-768pw-h16x32-w0-e800/
/// Convert: coda convert-bullet -input quantised.bin -output net.nnue -pairwise -hidden 16 -hidden2 32 -int8l1

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
};

fn main() {
    // Architecture: (768×16 → 1536) → CReLU → pairwise → 768 per perspective
    // Then: 768×2 concat → L1(16) → L2(32) → output(8 buckets)
    let ft_size = 1536;      // FT output before pairwise (after pairwise: 768)
    let l1_size = 16;
    let l2_size = 32;

    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    let dataset_path = get_arg(&args, "--dataset", "/training/sf/test80-2024-01-jan-2tb7p.min-v2.v6.binpack");
    let superbatches: usize = get_arg(&args, "--superbatches", "800").parse().unwrap();
    let wdl_proportion: f32 = get_arg(&args, "--wdl", "0.0").parse().unwrap();
    let initial_lr: f32 = get_arg(&args, "--lr", "0.001").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "20").parse().unwrap();
    let final_lr = initial_lr * 0.01; // 100x decay

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
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // Factoriser for weight sharing across king buckets
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            // FT: 768×16 → 1536
            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            // Hidden layers with output buckets
            // After pairwise: 768 per perspective, concat = 1536 (= ft_size)
            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS * l1_size);
            let l2 = builder.new_affine("l2", l1_size, NUM_OUTPUT_BUCKETS * l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            // Forward: FT → CReLU → pairwise → concat → L1 → ReLU → L2 → ReLU → output
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).relu();
            let hl3 = l2.forward(hl2).select(output_buckets).relu();
            l3.forward(hl3).select(output_buckets)
        });

    // Stricter clipping for factoriser weights
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

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
        save_rate: 100,
    };

    let settings = LocalSettings {
        threads: 4,
        batch_queue_size: 32,
    };

    let dataloader = SfBinpackLoader::new(&dataset_path, 256, 4, |entry| {
        entry.score.unsigned_abs() < 10000
    });

    println!("=== Coda v7 768pw Training ===");
    println!("FT: {} → CReLU → pairwise → {}", ft_size, ft_size / 2);
    println!("Hidden: {} → {} → 1×{}", l1_size, l2_size, NUM_OUTPUT_BUCKETS);
    println!("Warmup: {} SBs, Schedule: {} SBs, WDL: {}", warmup_sbs, superbatches, wdl_proportion);
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
