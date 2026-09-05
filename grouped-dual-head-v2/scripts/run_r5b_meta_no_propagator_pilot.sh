#!/bin/bash
# Reproducible single-GPU train-only pilot for r5b meta initialization.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
RUN_DIR="${R5B_META_RUN_DIR:?R5B_META_RUN_DIR is required}"
GPU="${R5B_META_GPU:-0}"
PER_FAMILY="${R5B_META_PER_FAMILY:-4}"
EPOCHS="${R5B_META_EPOCHS:-5}"
TIME_POINTS="${R5B_META_TIME_POINTS:-32}"
LEARNING_RATE="${R5B_META_LEARNING_RATE:-0.00002}"
OBSERVED_WEIGHT="${R5B_META_OBSERVED_WEIGHT:-0.25}"
CLOSED_FORM_RIDGE="${R5B_META_CLOSED_FORM_RIDGE:-0}"
PER_RANK_BATCH_SIZE="${R5B_META_PER_RANK_BATCH_SIZE:-1}"
JACOBIAN_VMAP_SIZE="${R5B_META_JACOBIAN_VMAP_SIZE:-1}"
export R5B_META_PER_RANK_BATCH_SIZE="$PER_RANK_BATCH_SIZE"
export R5B_META_JACOBIAN_VMAP_SIZE="$JACOBIAN_VMAP_SIZE"
OUTPUT_CKPT="$RUN_DIR/meta_pilot.pt"
CONFIG_PATH=configs/saved_time_v5/meta_hypernet_r5b_marmousi1_4m_v2_no_propagator.yaml
STARTED_AT="$(date -Is)"
STARTED_EPOCH="$(date +%s)"

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

write_status() {
  local target_path="$1"
  local status_value="$2"
  local return_code="$3"
  local finished_at="$4"
  local elapsed_seconds="$5"
  local checkpoint_sha256="$6"
  /root/miniconda3/bin/python - \
    "$target_path" "$status_value" "$return_code" "$STARTED_AT" \
    "$finished_at" "$elapsed_seconds" "$OUTPUT_CKPT" "$checkpoint_sha256" \
    "$GPU" "$PER_FAMILY" "$EPOCHS" "$TIME_POINTS" "$LEARNING_RATE" \
    "$OBSERVED_WEIGHT" "$CLOSED_FORM_RIDGE" <<'PY'
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
    gpu,
    per_family,
    epochs,
    time_points,
    learning_rate,
    observed_weight,
    closed_form_ridge,
) = sys.argv[1:]
payload = {
    "schema": "r5b_meta_no_propagator_pilot_status_v1",
    "status": status_value,
    "return_code": int(return_code),
    "started_at": started_at,
    "finished_at": finished_at or None,
    "elapsed_seconds": int(elapsed_seconds),
    "supervisor_pid": os.getppid(),
    "output_checkpoint": output_checkpoint,
    "checkpoint_sha256": checkpoint_sha256 or None,
    "gpu": int(gpu),
    "selection_split": "train",
    "future_truth_scope": "train_only",
    "parent": "r5b_epoch_0003",
    "external_propagator": False,
    "per_family": int(per_family),
    "episodes": 3 * int(per_family),
    "epochs": int(epochs),
    "time_points": int(time_points),
    "learning_rate": float(learning_rate),
    "observed_weight": float(observed_weight),
    "closed_form_output_ridge": float(closed_form_ridge),
    "per_rank_batch_size": int(os.environ["R5B_META_PER_RANK_BATCH_SIZE"]),
    "jacobian_vmap_size": int(os.environ["R5B_META_JACOBIAN_VMAP_SIZE"]),
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
  write_status \
    "$RUN_DIR/terminal.json" "$status_value" "$rc" "$finished_at" \
    "$elapsed_seconds" "$checkpoint_sha256"
  echo "[r5b-meta-pilot] status=$status_value rc=$rc elapsed_s=$elapsed_seconds"
  exit "$rc"
}
trap finish_run EXIT

if [ -e "$OUTPUT_CKPT" ] || [ -e "$RUN_DIR/terminal.json" ]; then
  echo "Refusing to overwrite completed or partial pilot artifact: $RUN_DIR" >&2
  exit 64
fi

write_status "$RUN_DIR/status.json" "running" 0 "" 0 ""

export CUDA_VISIBLE_DEVICES="$GPU"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:src

/root/miniconda3/bin/python -u scripts/train_meta_hypernet.py \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_CKPT" \
  --device cuda \
  --per-family "$PER_FAMILY" \
  --epochs "$EPOCHS" \
  --time-points "$TIME_POINTS" \
  --learning-rate "$LEARNING_RATE" \
  --physics-weight 0.0 \
  --observed-weight "$OBSERVED_WEIGHT" \
  --closed-form-output-ridge "$CLOSED_FORM_RIDGE" \
  --per-rank-batch-size "$PER_RANK_BATCH_SIZE" \
  --jacobian-vmap-size "$JACOBIAN_VMAP_SIZE"
