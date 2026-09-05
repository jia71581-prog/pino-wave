#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2

/root/miniconda3/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_r47_block_complex_union_curriculum.py \
  --r40-script scripts/train_r40_frequency_residual_operator.py \
  --r42-script scripts/train_r42_recordwise_hard_curriculum.py \
  --r44-script scripts/run_r44_physical_space_hard_pilot.py \
  --union-script scripts/r46_union_frequency_cache.py \
  --r47-model-script scripts/r47_block_complex_modulation_model.py \
  --base-fit-cache \
    results/r40_frequency_cache_v2_20260828/fit_shard0.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard1.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard2.h5 \
    results/r40_frequency_cache_v2_20260828/fit_shard3.h5 \
  --supplement-fit-cache \
    results/r46_frequency_cache_marmousi_supplement_v1_20260828/fit_supplement_shard0.h5 \
    results/r46_frequency_cache_marmousi_supplement_v1_20260828/fit_supplement_shard1.h5 \
    results/r46_frequency_cache_marmousi_supplement_v1_20260828/fit_supplement_shard2.h5 \
    results/r46_frequency_cache_marmousi_supplement_v1_20260828/fit_supplement_shard3.h5 \
  --holdout-cache \
    results/r40_frequency_cache_v2_20260828/holdout_shard0.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard1.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard2.h5 \
    results/r40_frequency_cache_v2_20260828/holdout_shard3.h5 \
  --curriculum-manifest \
    results/r46_union_hard_curriculum_v1_20260828.json \
  --output-dir \
    results/r47_block_complex_union_g8_w96_v1_20260828 \
  --epochs 40 \
  --records-per-rank-epoch 293 \
  --frequency-batch 8 \
  --eval-batch-size 4 \
  --eval-every 1 \
  --learning-rate 0.0003 \
  --weight-decay 0.00001 \
  --warmup-epochs 2 \
  --block-grid 8 \
  --encoder-width 96 \
  --low-resolution-blocks 4 \
  --gain-cap 0.5 \
  --gain-loss-weight 0.05 \
  --tail-weight 1.0 \
  --hinge-weight 2.0 \
  --shape-weight 0.002 \
  --probe-count 32 \
  --seed 4701 \
  --amp \
  --stop-on-pass
