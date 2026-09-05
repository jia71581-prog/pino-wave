#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2

/root/miniconda3/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_r45_physical_space_full_curriculum.py \
  --r40-script scripts/train_r40_frequency_residual_operator.py \
  --r42-script scripts/train_r42_recordwise_hard_curriculum.py \
  --r44-script scripts/run_r44_physical_space_hard_pilot.py \
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
  --curriculum-manifest \
    results/r42_hard_curriculum_manifest_v1_20260828.json \
  --output-dir \
    results/r45_physical_space_full_curriculum_w48_v1_20260828 \
  --epochs 30 \
  --records-per-rank-epoch 256 \
  --frequency-batch 8 \
  --eval-batch-size 4 \
  --eval-every 1 \
  --learning-rate 0.0001 \
  --weight-decay 0.00001 \
  --warmup-epochs 1 \
  --width 48 \
  --modes 24 \
  --blocks 4 \
  --correction-cap 3.0 \
  --tail-weight 1.0 \
  --hinge-weight 2.0 \
  --shape-weight 0.002 \
  --probe-count 32 \
  --seed 4501 \
  --amp \
  --stop-on-pass
