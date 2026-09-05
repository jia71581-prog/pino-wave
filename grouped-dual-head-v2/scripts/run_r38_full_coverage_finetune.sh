#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
CACHE="$ROOT/results/r38_full_coverage_cache_20260828"
INIT="$ROOT/results/r28_expanded_tail_spectral_v1_20260828/best.pt"
OUT="$ROOT/results/r38_full_coverage_finetune_1e5_5x1000_v1_20260828"
TORCHRUN="/root/miniconda3/bin/torchrun"

cd "$ROOT"
if [[ -e "$OUT" ]]; then
  echo "refusing to reuse existing R38 training directory: $OUT" >&2
  exit 2
fi

for shard in 0 1 2 3; do
  test -s "$CACHE/fit_shard_${shard}.h5"
  test -s "$CACHE/fit_shard_${shard}.h5.summary.json"
  test -s "$CACHE/holdout_shard_${shard}.h5"
  test -s "$CACHE/holdout_shard_${shard}.h5.summary.json"
done
test -s "$INIT"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_r29a_late_tail_finetune.py \
  --fit-cache \
    "$CACHE/fit_shard_0.h5" \
    "$CACHE/fit_shard_1.h5" \
    "$CACHE/fit_shard_2.h5" \
    "$CACHE/fit_shard_3.h5" \
  --holdout-cache \
    "$CACHE/holdout_shard_0.h5" \
    "$CACHE/holdout_shard_1.h5" \
    "$CACHE/holdout_shard_2.h5" \
    "$CACHE/holdout_shard_3.h5" \
  --init-checkpoint "$INIT" \
  --output-dir "$OUT" \
  --epochs 5 \
  --max-steps-per-epoch 1000 \
  --batch-size 16 \
  --eval-batch-size 16 \
  --learning-rate 1.0e-5 \
  --weight-decay 1.0e-4 \
  --hinge-weight 3.0 \
  --gradient-weight 0.05 \
  --late-start-s 0.50 \
  --seed 380828 \
  --amp
