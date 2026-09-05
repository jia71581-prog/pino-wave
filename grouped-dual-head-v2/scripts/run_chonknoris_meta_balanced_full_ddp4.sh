#!/bin/bash
# Four-GPU synchronous Stage-B training for the reduced CHONKNORIS model.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_ROOT="${CHONKNORIS_ARTIFACT_ROOT:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/meta_hypernet_chonknoris}"
CONFIG_PATH="${CHONKNORIS_CONFIG_PATH:-configs/saved_time_v5/meta_hypernet_chonknoris.yaml}"
PER_RANK_BATCH_SIZE="${CHONKNORIS_PER_RANK_BATCH_SIZE:-1}"
JACOBIAN_VMAP_SIZE="${CHONKNORIS_JACOBIAN_VMAP_SIZE:-1}"
PER_FAMILY="${CHONKNORIS_PER_FAMILY:-64}"
EPOCHS="${CHONKNORIS_EPOCHS:-5}"
TIME_POINTS="${CHONKNORIS_TIME_POINTS:-48}"
ADAPTIVE_TIME_CANDIDATES="${CHONKNORIS_ADAPTIVE_TIME_CANDIDATES:-0}"
ADAPTIVE_HIGH_FRACTION="${CHONKNORIS_ADAPTIVE_HIGH_FRACTION:-0.75}"
RESIDUAL_SAMPLING="${CHONKNORIS_RESIDUAL_SAMPLING:-average_pool}"
OBSERVED_WEIGHT="${CHONKNORIS_OBSERVED_WEIGHT:-0.25}"
LEARNING_RATE="${CHONKNORIS_LEARNING_RATE:-0.0002}"
CLOSED_FORM_OUTPUT_RIDGE="${CHONKNORIS_CLOSED_FORM_OUTPUT_RIDGE:-0}"
RUN_TAG="${CHONKNORIS_RUN_TAG:-balanced_ddp4_pf64_ep5_tp48_20260807T0929CST}"
GLOBAL_EPISODES="$((3 * PER_FAMILY))"
if [ "$((GLOBAL_EPISODES % 4))" -ne 0 ]; then
  echo "Global episodes must be divisible by four: $GLOBAL_EPISODES" >&2
  exit 64
fi
LOCAL_EPISODES="$((GLOBAL_EPISODES / 4))"
SYNCHRONOUS_OPTIMIZER_STEPS="$(( ((LOCAL_EPISODES + PER_RANK_BATCH_SIZE - 1) / PER_RANK_BATCH_SIZE) * EPOCHS ))"
export CHONKNORIS_PER_RANK_BATCH_SIZE="$PER_RANK_BATCH_SIZE"
export CHONKNORIS_JACOBIAN_VMAP_SIZE="$JACOBIAN_VMAP_SIZE"
export CHONKNORIS_SYNCHRONOUS_STEPS="$SYNCHRONOUS_OPTIMIZER_STEPS"
export CHONKNORIS_CONFIG_PATH="$CONFIG_PATH"
export CHONKNORIS_PER_FAMILY="$PER_FAMILY"
export CHONKNORIS_EPOCHS="$EPOCHS"
export CHONKNORIS_TIME_POINTS="$TIME_POINTS"
export CHONKNORIS_GLOBAL_EPISODES="$GLOBAL_EPISODES"
export CHONKNORIS_LOCAL_EPISODES="$LOCAL_EPISODES"
export CHONKNORIS_ADAPTIVE_TIME_CANDIDATES="$ADAPTIVE_TIME_CANDIDATES"
export CHONKNORIS_ADAPTIVE_HIGH_FRACTION="$ADAPTIVE_HIGH_FRACTION"
export CHONKNORIS_RESIDUAL_SAMPLING="$RESIDUAL_SAMPLING"
export CHONKNORIS_OBSERVED_WEIGHT="$OBSERVED_WEIGHT"
export CHONKNORIS_LEARNING_RATE="$LEARNING_RATE"
export CHONKNORIS_CLOSED_FORM_OUTPUT_RIDGE="$CLOSED_FORM_OUTPUT_RIDGE"
RUN_DIR="$ARTIFACT_ROOT/$RUN_TAG"
OUTPUT_CKPT="$RUN_DIR/meta_hypernet_chonknoris.pt"
STARTED_AT="$(date -Is)"
STARTED_EPOCH="$(date +%s)"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

write_json() {
  local target_path="$1"
  local status_value="$2"
  local return_code="$3"
  local finished_at="$4"
  local elapsed_seconds="$5"
  local checkpoint_sha256="$6"
  /root/miniconda3/bin/python - \
    "$target_path" "$status_value" "$return_code" "$STARTED_AT" \
    "$finished_at" "$elapsed_seconds" "$OUTPUT_CKPT" "$checkpoint_sha256" <<'PY'
import json
import os
from pathlib import Path
import sys

(
    target_path,
    status_value,
    return_code,
    started_at,
    finished_at,
    elapsed_seconds,
    output_checkpoint,
    checkpoint_sha256,
) = sys.argv[1:]
payload = {
    "schema": "chonknoris_meta_training_status_v1",
    "status": status_value,
    "return_code": int(return_code),
    "started_at": started_at,
    "finished_at": finished_at or None,
    "elapsed_seconds": int(elapsed_seconds),
    "output_checkpoint": output_checkpoint,
    "checkpoint_sha256": checkpoint_sha256 or None,
    "gpus": [0, 1, 2, 3],
    "world_size": 4,
    "config": os.environ["CHONKNORIS_CONFIG_PATH"],
    "per_family": int(os.environ["CHONKNORIS_PER_FAMILY"]),
    "episodes": int(os.environ["CHONKNORIS_GLOBAL_EPISODES"]),
    "local_episodes_per_rank": int(os.environ["CHONKNORIS_LOCAL_EPISODES"]),
    "epochs": int(os.environ["CHONKNORIS_EPOCHS"]),
    "per_rank_batch_size": int(os.environ["CHONKNORIS_PER_RANK_BATCH_SIZE"]),
    "jacobian_vmap_size": int(os.environ["CHONKNORIS_JACOBIAN_VMAP_SIZE"]),
    "global_batch_size": 4 * int(os.environ["CHONKNORIS_PER_RANK_BATCH_SIZE"]),
    "synchronous_optimizer_steps": int(os.environ["CHONKNORIS_SYNCHRONOUS_STEPS"]),
    "effective_sample_updates": (
        int(os.environ["CHONKNORIS_GLOBAL_EPISODES"])
        * int(os.environ["CHONKNORIS_EPOCHS"])
    ),
    "time_points": int(os.environ["CHONKNORIS_TIME_POINTS"]),
    "adaptive_time_candidates": int(os.environ["CHONKNORIS_ADAPTIVE_TIME_CANDIDATES"]),
    "adaptive_high_fraction": float(os.environ["CHONKNORIS_ADAPTIVE_HIGH_FRACTION"]),
    "learning_rate": float(os.environ["CHONKNORIS_LEARNING_RATE"]),
    "observed_weight": float(os.environ["CHONKNORIS_OBSERVED_WEIGHT"]),
    "closed_form_output_ridge": float(os.environ["CHONKNORIS_CLOSED_FORM_OUTPUT_RIDGE"]),
    "chonknoris_weight": 0.01,
    "chonknoris_relaxations": [0.001, 0.01, 0.1],
    "chonknoris_pool_size": 4,
    "chonknoris_residual_sampling": os.environ["CHONKNORIS_RESIDUAL_SAMPLING"],
    "promotion_policy": "validate before replacing the deployment checkpoint",
}
path = Path(target_path)
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

finish_run() {
  local rc="$?"
  trap - EXIT
  local finished_at
  local finished_epoch
  local elapsed_seconds
  local checkpoint_sha256=""
  local status_value="failed"
  finished_at="$(date -Is)"
  finished_epoch="$(date +%s)"
  elapsed_seconds="$((finished_epoch - STARTED_EPOCH))"
  if [ "$rc" -eq 0 ] && [ -s "$OUTPUT_CKPT" ]; then
    status_value="complete"
    checkpoint_sha256="$(sha256sum "$OUTPUT_CKPT" | awk '{print $1}')"
  fi
  write_json \
    "$RUN_DIR/terminal.json" "$status_value" "$rc" "$finished_at" \
    "$elapsed_seconds" "$checkpoint_sha256"
  echo "[chonknoris-ddp4] status=$status_value rc=$rc finished_at=$finished_at elapsed_s=$elapsed_seconds"
  exit "$rc"
}
trap finish_run EXIT

if [ -e "$OUTPUT_CKPT" ]; then
  echo "Refusing to overwrite existing checkpoint: $OUTPUT_CKPT" >&2
  exit 64
fi

write_json "$RUN_DIR/status.json" "running" 0 "" 0 ""
echo "[chonknoris-ddp4] started_at=$STARTED_AT"
echo "[chonknoris-ddp4] run_dir=$RUN_DIR"
echo "[chonknoris-ddp4] output=$OUTPUT_CKPT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1

/root/miniconda3/bin/torchrun --standalone --nproc_per_node=4 \
  scripts/train_meta_hypernet.py \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_CKPT" \
  --device cuda \
  --per-family "$PER_FAMILY" \
  --epochs "$EPOCHS" \
  --time-points "$TIME_POINTS" \
  --learning-rate "$LEARNING_RATE" \
  --chonknoris-weight 0.01 \
  --chonknoris-relaxation 0.001 \
  --chonknoris-relaxation 0.01 \
  --chonknoris-relaxation 0.1 \
  --chonknoris-pool-size 4 \
  --chonknoris-residual-sampling "$RESIDUAL_SAMPLING" \
  --adaptive-time-candidates "$ADAPTIVE_TIME_CANDIDATES" \
  --adaptive-high-fraction "$ADAPTIVE_HIGH_FRACTION" \
  --observed-weight "$OBSERVED_WEIGHT" \
  --closed-form-output-ridge "$CLOSED_FORM_OUTPUT_RIDGE" \
  --per-rank-batch-size "$PER_RANK_BATCH_SIZE" \
  --jacobian-vmap-size "$JACOBIAN_VMAP_SIZE"
