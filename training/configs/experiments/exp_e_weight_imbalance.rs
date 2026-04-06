/// Experiment E: Upweight material-imbalanced positions 2×
/// Uses datapoint_weight_function to give more training signal to
/// positions with |score| > 500 (material imbalances). Addresses
/// the check-net calibration issue without needing force-capture data.

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
    let ft_size = 1536;
    let l1_size = 16;
    let l2_size = 32;

    let args: Vec<String> = std::env::args().collect();
    let dataset_path = get_arg(&args, "--dataset", "/training/sf/test80-2024-01-jan-2tb7p.min-v2.v6.binpack");
    let superbatches: usize = get_arg(&args, "--superbatches", "100").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "10").parse().unwrap();
    let save_rate: usize = get_arg(&args, "--save-rate", "50").parse().unwrap();
    let initial_lr: f32 = 0.001;
    let final_lr: f32 = 0.00001;

    const NUM_OUTPUT_BUCKETS: usize = 8;

    #[rustfmt::skip]
    const BUCKET_LAYOUT: [usize; 32] = [
         0,  4,  8, 12,   0,  4,  8, 12,
         1,  5,  9, 13,   1,  5,  9, 13,
         2,  6, 10, 14,   2,  6, 10, 14,
         3,  7, 11, 15,   3,  7, 11, 15,
    ];

    const NUM_INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .use_threads(4)
        .inputs(ChessBucketsMirrored::new(BUCKET_LAYOUT))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        // Upweight positions with material imbalances
        .datapoint_weight_function(|entry| {
            if entry.score.unsigned_abs() > 500 { 2.0 } else { 1.0 }
        })
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
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS * l1_size);
            let l2 = builder.new_affine("l2", l1_size, NUM_OUTPUT_BUCKETS * l2_size);
            let l3 = builder.new_affine("l3", l2_size, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).screlu();
            let hl3 = l2.forward(hl2).select(output_buckets).screlu();
            l3.forward(hl3).select(output_buckets)
        });

    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id: "exp-e-weight-imbalance".to_string(),
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

    let dataloader = SfBinpackLoader::new(&dataset_path, 256, 4, |entry| {
        entry.score.unsigned_abs() < 10000
    });

    println!("=== Experiment E: Upweight Imbalanced Positions ===");
    println!("Positions with |score|>500 get 2× loss weight");
    trainer.run(&schedule, &settings, &dataloader);
}

fn get_arg(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|s| s == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}
