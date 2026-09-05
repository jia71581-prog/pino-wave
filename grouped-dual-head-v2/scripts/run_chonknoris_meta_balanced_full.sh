#!/bin/bash
# Balanced Stage-B training candidate for the reduced CHONKNORIS model.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
ARTIFACT_ROOT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/meta_hypernet_chonknoris
RUN_DIR="$ARTIFACT_ROOT/balanced_pf64_ep5_tp48_20260807T0922CST"
OUTPUT_CKPT="$RUN_DIR/meta_hypernet_chonknoris.pt"
CONFIG_PATH=configs/saved_time_v5/meta_hypernet_chonknoris.yaml
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
    "gpu": "physical:0",
    "config": "configs/saved_time_v5/meta_hypernet_chonknoris.yaml",
    "per_family": 64,
    "episodes": 192,
    "epochs": 5,
    "optimizer_steps": 960,
    "time_points": 48,
    "learning_rate": 0.0002,
    "chonknoris_weight": 0.01,
    "chonknoris_relaxations": [0.001, 0.01, 0.1],
    "chonknoris_pool_size": 4,
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
  echo "[chonknoris-full] status=$status_value rc=$rc finished_at=$finished_at elapsed_s=$elapsed_seconds"
  exit "$rc"
}
trap finish_run EXIT

if [ -e "$OUTPUT_CKPT" ]; then
  echo "Refusing to overwrite existing checkpoint: $OUTPUT_CKPT" >&2
  exit 64
fi

write_json "$RUN_DIR/status.json" "running" 0 "" 0 ""
echo "[chonknoris-full] started_at=$STARTED_AT"
echo "[chonknoris-full] run_dir=$RUN_DIR"
echo "[chonknoris-full] output=$OUTPUT_CKPT"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

/root/miniconda3/bin/python scripts/train_meta_hypernet.py \
  --config "$CONFIG_PATH" \
  --output "$OUTPUT_CKPT" \
  --device cuda \
  --per-family 64 \
  --epochs 5 \
  --time-points 48 \
  --learning-rate 2e-4 \
  --chonknoris-weight 0.01 \
  --chonknoris-relaxation 0.001 \
  --chonknoris-relaxation 0.01 \
  --chonknoris-relaxation 0.1 \
  --chonknoris-pool-size 4
