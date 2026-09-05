#!/usr/bin/env bash
# R56 long runs: arm A (AdamW+EMA) then arm B (SOAP+EMA), serial single-writer.
# Each arm gets the frozen R55 selective evaluation afterwards (CPU, additive).
set -uo pipefail
W=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CACHE=$W/results/r38_full_coverage_cache_20260828
cd "$W" || exit 1

run_arm() {
  local NAME=$1 OPT=$2 OUT=$3
  [[ -e "$OUT" ]] && { echo "refusing to reuse $OUT"; return 2; }
  mkdir -p "$OUT"
  echo "[$NAME] launch $(date '+%F %T')"
  R56_OPTIMIZER=$OPT R56_EMA_DECAY=0.999 torchrun --standalone --nproc_per_node=4 \
    scripts/train_r56_optim_ema.py \
    --fit-cache "$CACHE"/fit_shard_0.h5 "$CACHE"/fit_shard_1.h5 "$CACHE"/fit_shard_2.h5 "$CACHE"/fit_shard_3.h5 \
    --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
    --output-dir "$OUT" \
    --epochs 40 --eval-every 1 --batch-size 64 \
    --learning-rate 8.0e-4 --weight-decay 1.0e-4 \
    --base-width 32 --correction-cap 0.25 \
    --hinge-weight 3.0 --gradient-weight 0.05 \
    --seed 540829 --amp
  local STATUS=$?
  echo "[$NAME] training exited $STATUS at $(date '+%F %T')"
  if [ "$STATUS" -eq 0 ] && [ -f "$OUT/best.json" ]; then
    CUDA_VISIBLE_DEVICES="" nice -n 19 python3 scripts/evaluate_r54_selective_prediction.py \
      --probe-json "$W/results/r55_tailchain_abstention_probe_v1_20260829.json" \
      --run-best-json "$OUT/best.json" \
      --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
      --output "$OUT/r55_selective_prediction.json"
    echo "[$NAME] selective eval exited $?"
  fi
  return $STATUS
}

run_arm r56armA adamw "$W/results/r56_armA_adamw_ema_bs64_40ep_20260830"
run_arm r56armB soap  "$W/results/r56_armB_soap_ema_bs64_40ep_20260830"
echo "[r56] all arms done $(date '+%F %T')"
