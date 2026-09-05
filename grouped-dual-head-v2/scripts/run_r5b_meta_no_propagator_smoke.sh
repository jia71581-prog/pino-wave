#!/bin/bash
# Detached single-GPU smoke for train-only r5b meta initialization.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
RUN_DIR="${R5B_META_SMOKE_RUN_DIR:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/instance_adaptation/r5b_meta_no_propagator_smoke_r1_20260815}"
OUTPUT_CKPT="$RUN_DIR/meta_smoke.pt"
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
    "schema": "r5b_meta_no_propagator_smoke_status_v1",
    "status": status_value,
    "return_code": int(return_code),
    "started_at": started_at,
    "finished_at": finished_at or None,
    "elapsed_seconds": int(elapsed_seconds),
    "supervisor_pid": os.getppid(),
    "output_checkpoint": output_checkpoint,
    "checkpoint_sha256": checkpoint_sha256 or None,
    "gpu": 0,
    "selection_split": "train",
    "future_truth_scope": "train_only",
    "parent": "r5b_epoch_0003",
    "external_propagator": False,
    "time_points": 32,
    "epochs": 1,
    "smoke": True,
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
  echo "[r5b-meta-smoke] status=$status_value rc=$rc elapsed_s=$elapsed_seconds"
  exit "$rc"
}
trap finish_run EXIT

if [ -e "$OUTPUT_CKPT" ] || [ -e "$RUN_DIR/terminal.json" ]; then
  echo "Refusing to overwrite completed or partial smoke artifact: $RUN_DIR" >&2
  exit 64
fi

write_status "$RUN_DIR/status.json" "running" 0 "" 0 ""

export CUDA_VISIBLE_DEVICES=0
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:src

/root/miniconda3/bin/python -u scripts/train_meta_hypernet.py \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_CKPT" \
  --device cuda \
  --per-family 4 \
  --epochs 1 \
  --time-points 32 \
  --learning-rate 2e-4 \
  --physics-weight 0.0 \
  --observed-weight 0.25 \
  --closed-form-output-ridge 1e-6 \
  --per-rank-batch-size 1 \
  --jacobian-vmap-size 1 \
  --smoke
