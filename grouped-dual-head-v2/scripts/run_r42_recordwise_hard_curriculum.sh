#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2

output_dir=results/r42_recordwise_hard_curriculum_w48_v1_20260828
console_log=results/r42_recordwise_hard_curriculum_w48_v1_console.log

test ! -e "$output_dir"
test -f results/r42_hard_curriculum_manifest_v1_20260828.json
test -f scripts/train_r42_recordwise_hard_curriculum.py
test -f scripts/train_r40_frequency_residual_operator.py

/root/miniconda3/bin/torchrun --standalone --nproc-per-node=4 \
  scripts/train_r42_recordwise_hard_curriculum.py \
  --r40-script scripts/train_r40_frequency_residual_operator.py \
  --fit-cache \
    results/r40_frequency_cache_v2_20260828/fit_shard0.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard1.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard2.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard3.h5 \
  --holdout-cache \
    results/r40_frequency_cache_v2_20260828/holdout_shard0.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard1.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard2.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard3.h5 \
  --curriculum-manifest results/r42_hard_curriculum_manifest_v1_20260828.json \
  --output-dir "$output_dir" \
  --epochs 20 \
  --records-per-rank-epoch 256 \
  --frequency-chunk 15 \
  --eval-batch-size 15 \
  --eval-every 1 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --warmup-epochs 1 \
  --width 48 \
  --modes 16 \
  --blocks 4 \
  --correction-cap 1.5 \
  --hinge-weight 2.0 \
  --shape-weight 0.002 \
  --probe-count 32 \
  --seed 420829 \
  --amp \
  --stop-on-pass \
  2>&1 | tee "$console_log"
