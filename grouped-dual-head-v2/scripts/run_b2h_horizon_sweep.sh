#!/usr/bin/env bash
# B2-H training-horizon diagnostic.
# Four single-GPU trainings (rollout_steps in {8,16,32,64}) run concurrently on
# GPUs 0-3, each judged afterwards by the FIXED 401-step strict free rollout.
# Everything except the training horizon is held identical.
set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
RES="$REPO/results/b2h_horizon_sweep"
mkdir -p "$RES"
HORIZONS=(08 16 32 64)
GPUS=(0 1 2 3)

echo "[$(date '+%F %T')] launching ${#HORIZONS[@]} trainings"
PIDS=()
for i in "${!HORIZONS[@]}"; do
  h="${HORIZONS[$i]}"; g="${GPUS[$i]}"
  cfg="$REPO/configs/b2h_horizon_sweep/h${h}.yaml"
  log="$RES/h${h}_train.log"
  CUDA_VISIBLE_DEVICES="$g" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    setsid nohup python "$REPO/scripts/train_b2h.py" --config "$cfg" >"$log" 2>&1 &
  pid=$!
  echo "$pid" > "$RES/h${h}.pid"
  PIDS+=("$pid")
  echo "  h${h} -> gpu${g} pid ${pid} log ${log}"
done

echo "[$(date '+%F %T')] waiting for trainings..."
fail=0
for i in "${!HORIZONS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "  h${HORIZONS[$i]} training OK"
  else
    echo "  h${HORIZONS[$i]} training FAILED (see log)"; fail=1
  fi
done

echo "[$(date '+%F %T')] strict 401-step rollout eval (gpu0)"
for h in "${HORIZONS[@]}"; do
  run="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/b2h/horizon_sweep/h${h}/run"
  cfg="$REPO/configs/b2h_horizon_sweep/h${h}.yaml"
  for which in latest best; do
    ckpt="$run/${which}.pt"
    out="$RES/h${h}_${which}_strict.json"
    if [[ -f "$ckpt" ]]; then
      CUDA_VISIBLE_DEVICES=0 python "$REPO/scripts/evaluate_b2h_full_rollout.py" \
        --config "$cfg" --checkpoint "$ckpt" --records-per-family 2 \
        --output "$out" >"$RES/h${h}_${which}_strict.log" 2>&1 \
        && echo "  h${h}/${which} strict OK -> $out" \
        || echo "  h${h}/${which} strict FAILED (see log)"
    else
      echo "  h${h}/${which} checkpoint missing: $ckpt"
    fi
  done
done

echo "[$(date '+%F %T')] sweep done (fail=${fail})"
exit "$fail"
