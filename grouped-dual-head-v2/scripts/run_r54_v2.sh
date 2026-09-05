#!/usr/bin/env bash
# R54 v2: worker-parallel from-scratch full-pool run + frozen R55 selective eval.
set -uo pipefail
W=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CACHE=$W/results/r38_full_coverage_cache_20260828
OUT=$W/results/r54_scratch_fullpool_40ep_v2_20260829
cd "$W" || exit 1
[[ -e "$OUT" ]] && { echo "refusing to reuse $OUT"; exit 2; }
mkdir -p "$OUT"
echo "[r54v2] launch $(date '+%F %T')"
torchrun --standalone --nproc_per_node=4 \
  scripts/train_r54_scratch_fullpool.py \
  --fit-cache "$CACHE"/fit_shard_0.h5 "$CACHE"/fit_shard_1.h5 "$CACHE"/fit_shard_2.h5 "$CACHE"/fit_shard_3.h5 \
  --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
  --output-dir "$OUT" \
  --epochs 40 --eval-every 1 --batch-size 16 \
  --learning-rate 2.0e-4 --weight-decay 1.0e-4 \
  --base-width 32 --correction-cap 0.25 \
  --hinge-weight 3.0 --gradient-weight 0.05 \
  --seed 540829 --amp
STATUS=$?
echo "[r54v2] training exited $STATUS at $(date '+%F %T')"
if [ "$STATUS" -eq 0 ] && [ -f "$OUT/best.json" ]; then
  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 scripts/evaluate_r54_selective_prediction.py \
    --probe-json "$W/results/r55_tailchain_abstention_probe_v1_20260829.json" \
    --run-best-json "$OUT/best.json" \
    --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
    --output "$OUT/r55_selective_prediction.json"
  echo "[r54v2] selective eval exited $?"
fi
exit $STATUS
