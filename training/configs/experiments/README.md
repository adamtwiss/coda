# V7 Training Experiments (2026-04-06)

Quick SB100 experiments to identify what improves v7 hidden layer training.
Each experiment changes ONE variable from a baseline.

## Baseline
- Architecture: 768pw → CReLU → pairwise → 16 → 32 → 1×8
- Data: single T80 file (test80-2024-01-jan)
- WDL: 0.0
- LR: 0.001 → 0.00001 (cosine), 20 SB warmup
- SB: 100
- Hidden activation: SCReLU (this is the fix from our old ReLU config)

## Experiments

| ID | File | Change from baseline | GPU host |
|----|------|---------------------|----------|
| A | exp_a_baseline_screlu.rs | Baseline with SCReLU fix (control) | GPU1 |
| B | exp_b_wdl_030.rs | WDL=0.3 (Bullet example uses 0.75) | GPU1 |
| C | exp_c_lower_final_lr.rs | Final LR ×0.3⁵ (match Bullet example) | GPU2 |
| D | exp_d_wdl_030_low_lr.rs | WDL=0.3 + lower final LR | GPU2 |
| E | exp_e_weight_imbalance.rs | Upweight |score|>500 positions 2× | GPU3 |
| F | exp_f_wdl_dynamic.rs | WDL scales with |score| (more WDL when decisive) | GPU3 |

## Running
```bash
cd ~/code/bullet
cargo run --release --example exp_a_baseline_screlu -- \
  --dataset /training/sf/test80-2024-01-jan-2tb7p.min-v2.v6.binpack \
  --superbatches 100 --warmup 10 --save-rate 50
```

## Evaluation
After training, convert and check-net:
```bash
cd ~/code/coda
./coda convert-bullet -input checkpoints/quantised-100.bin -output net.nnue -pairwise -hidden 16 -hidden2 32 -int8l1
./tuner check-net net.nnue    # (run from gochess dir)
```

Also run `coda bench` with each net to check move ordering quality:
```bash
./coda --nnue net.nnue bench 2>&1 | grep "Move ordering"
```
