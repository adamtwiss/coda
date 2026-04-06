/// Experiment K: v5 768pw, 8 output buckets, MSE loss (control)
/// Architecture: (768×16 → 1536)×2 → CReLU → pairwise → 768×2 → output ×8
/// Same as production but with: power-2.6 loss, all 12 T80 files
/// Tests if training tricks improve v5 without any architecture change.
///
/// Run: cargo run --release --example exp_h_v5_768pw_tricks -- --superbatches 100

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

    let args: Vec<String> = std::env::args().collect();
    let superbatches: usize = get_arg(&args, "--superbatches", "100").parse().unwrap();
    let warmup_sbs: usize = get_arg(&args, "--warmup", "10").parse().unwrap();
    let save_rate: usize = get_arg(&args, "--save-rate", "50").parse().unwrap();
    let wdl_val: f32 = get_arg(&args, "--wdl", "0.07").parse().unwrap();
    let initial_lr: f32 = 0.001;
    let final_lr: f32 = 0.00001;

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
            SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        // Standard MSE loss (control, matches production)
        .loss_fn(|output, target| {
            output.sigmoid().squared_error(target)
        })
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(ft_size, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(NUM_INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * NUM_INPUT_BUCKETS, ft_size);
            l0.init_with_effective_input_size(32);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", ft_size, NUM_OUTPUT_BUCKETS);

            // CReLU → pairwise (same as production 768pw)
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    let ft_params = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", ft_params);
    trainer.optimiser.set_params_for_weight("l0f", ft_params);

    let schedule = TrainingSchedule {
        net_id: "exp-k-v5-8bucket-mse".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_val },
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

    let data_files: Vec<&str> = vec![
        "/workspace/data/test80-2023-06-jun-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-07-jul-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-09-sep-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-10-oct-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-11-nov-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2023-12-dec-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-02-feb-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-03-mar-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-04-apr-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-05-may-2tb7p.min-v2.v6.binpack",
        "/workspace/data/test80-2024-06-jun-2tb7p.min-v2.v6.binpack",
    ];

    let dataloader = SfBinpackLoader::new_concat_multiple(&data_files, 256, 4, |entry| {
        entry.score.unsigned_abs() < 10000
    });

    println!("=== Experiment K: 8 output buckets, MSE (control)
    println!("768pw → CReLU → pairwise, 8 output buckets");
    println!("Standard MSE loss (control, matches production)
    trainer.run(&schedule, &settings, &dataloader);
}

fn get_arg(args: &[String], flag: &str, default: &str) -> String {
    args.iter()
        .position(|s| s == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}
