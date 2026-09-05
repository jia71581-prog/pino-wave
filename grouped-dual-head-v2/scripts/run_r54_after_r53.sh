#!/usr/bin/env bash
# R54: from-scratch full-pool tail-spectral training to saturation.
# Waits for the R53 writer to finish, then launches. Single GPU writer at a time.
set -uo pipefail

W=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CACHE=$W/results/r38_full_coverage_cache_20260828
R53_DIR=$W/results/r53_spatiotemporal_8ep_v1_20260829
OUT=$W/results/r54_scratch_fullpool_40ep_v1_20260829
cd "$W" || exit 1

echo "[r54-chain] waiting for R53 terminal at $(date '+%F %T')"
while [ ! -f "$R53_DIR/terminal.json" ]; do
  if ! pgrep -f "train_r53_spatiotemporal_tail.py" >/dev/null 2>&1; then
    sleep 30
    if [ ! -f "$R53_DIR/terminal.json" ]; then
      echo "[r54-chain] R53 process gone with no terminal.json; aborting rather than guessing"
      exit 2
    fi
  fi
  sleep 60
done
echo "[r54-chain] R53 terminal present at $(date '+%F %T')"

while pgrep -f "train_r5[0-9].*\.py" >/dev/null 2>&1; do
  echo "[r54-chain] a tail-chain writer is still alive; waiting"
  sleep 60
done

FREE_GB=$(df -BG --output=avail "$W" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 20 ]; then
  echo "[r54-chain] only ${FREE_GB}G free, below the 20G floor; not launching"
  exit 3
fi
echo "[r54-chain] ${FREE_GB}G free, launching R54 at $(date '+%F %T')"
mkdir -p "$OUT"

torchrun --standalone --nproc_per_node=4 \
  scripts/train_r28_expanded_tail_spectral.py \
  --fit-cache "$CACHE"/fit_shard_0.h5 "$CACHE"/fit_shard_1.h5 "$CACHE"/fit_shard_2.h5 "$CACHE"/fit_shard_3.h5 \
  --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
  --output-dir "$OUT" \
  --epochs 40 \
  --eval-every 1 \
  --batch-size 16 \
  --learning-rate 2.0e-4 \
  --weight-decay 1.0e-4 \
  --base-width 32 \
  --correction-cap 0.25 \
  --hinge-weight 3.0 \
  --gradient-weight 0.05 \
  --seed 540829 \
  --amp
STATUS=$?
echo "[r54-chain] R54 exited with $STATUS at $(date '+%F %T')"

# Post-step: apply the frozen R55 abstention rule to R54's best checkpoint.
# CPU only, no refitting; the rule is read from the R55 artifact.
PROBE=$W/results/r55_tailchain_abstention_probe_v1_20260829.json
if [ "$STATUS" -eq 0 ] && [ -f "$OUT/best.json" ] && [ -f "$PROBE" ]; then
  echo "[r54-chain] running R55 selective-prediction evaluation at $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 scripts/evaluate_r54_selective_prediction.py \
    --probe-json "$PROBE" \
    --run-best-json "$OUT/best.json" \
    --holdout-cache "$CACHE"/holdout_shard_0.h5 "$CACHE"/holdout_shard_1.h5 \
                    "$CACHE"/holdout_shard_2.h5 "$CACHE"/holdout_shard_3.h5 \
    --output "$OUT/r55_selective_prediction.json"
  echo "[r54-chain] selective evaluation exited with $?"
else
  echo "[r54-chain] skipping selective evaluation (status=$STATUS, best.json/probe presence checked)"
fi

exit $STATUS
