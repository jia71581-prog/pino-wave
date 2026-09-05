#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export CUDA_VISIBLE_DEVICES=0

/root/miniconda3/bin/python scripts/run_r44_physical_space_hard_pilot.py \
  --r40-script scripts/train_r40_frequency_residual_operator.py \
  --fit-caches \
    results/r40_frequency_cache_v2_20260828/fit_shard0.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard1.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard2.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard3.h5 \
  --holdout-caches \
    results/r40_frequency_cache_v2_20260828/holdout_shard0.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard1.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard2.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard3.h5 \
  --curriculum results/r42_hard_curriculum_manifest_v1_20260828.json \
  --output results/r44_physical_space_capacity_hard8_v2_20260828.json \
  --hard-records 8 \
  --steps 2000 \
  --frequency-batch 8 \
  --eval-batch 4 \
  --eval-every 250 \
  --width 24 \
  --modes 16 \
  --blocks 4 \
  --correction-cap 3.0 \
  --learning-rate 0.0003 \
  --weight-decay 0.00001 \
  --seed 4401 \
  --amp
