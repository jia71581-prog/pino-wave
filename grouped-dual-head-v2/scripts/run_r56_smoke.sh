#!/usr/bin/env bash
# R56 smoke: arm A 2-epoch (EMA infra + step-100 replication), then arm B 1-epoch (SOAP infra + throughput).
set -uo pipefail
W=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CACHE=$W/results/r38_full_coverage_cache_20260828
cd "$W" || exit 1

OUT_A=$W/results/r56_smokeA_adamw_ema_2ep_20260830
[[ -e "$OUT_A" ]] && { echo "refusing to reuse $OUT_A"; exit 2; }
mkdir -p "$OUT_A"
echo "[r56smokeA] launch $(date '+%F %T')"
R56_OPTIMIZER=adamw R56_EMA_DECAY=0.999 torchrun --standalone --nproc_per_node=4 \
  scripts/train_r56_optim_ema.py \
  --fit-cache "$CACHE"/fit_shard_0.h5 "$CACHE"/fit_shard_1.h5 "$CACHE"/fit_shard_2.h5 "$CACHE"/fit_shard_3.h5 \
  --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
  --output-dir "$OUT_A" \
  --epochs 2 --eval-every 1 --batch-size 16 \
  --learning-rate 2.0e-4 --weight-decay 1.0e-4 \
  --base-width 32 --correction-cap 0.25 \
  --hinge-weight 3.0 --gradient-weight 0.05 \
  --seed 540829 --amp
echo "[r56smokeA] exited $? at $(date '+%F %T')"

OUT_B=$W/results/r56_smokeB_soap_ema_1ep_20260830
[[ -e "$OUT_B" ]] && { echo "refusing to reuse $OUT_B"; exit 2; }
mkdir -p "$OUT_B"
echo "[r56smokeB] launch $(date '+%F %T')"
R56_OPTIMIZER=soap R56_EMA_DECAY=0.999 torchrun --standalone --nproc_per_node=4 \
  scripts/train_r56_optim_ema.py \
  --fit-cache "$CACHE"/fit_shard_0.h5 "$CACHE"/fit_shard_1.h5 "$CACHE"/fit_shard_2.h5 "$CACHE"/fit_shard_3.h5 \
  --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
  --output-dir "$OUT_B" \
  --epochs 1 --eval-every 1 --batch-size 16 \
  --learning-rate 2.0e-4 --weight-decay 1.0e-4 \
  --base-width 32 --correction-cap 0.25 \
  --hinge-weight 3.0 --gradient-weight 0.05 \
  --seed 540829 --amp
echo "[r56smokeB] exited $? at $(date '+%F %T')"
