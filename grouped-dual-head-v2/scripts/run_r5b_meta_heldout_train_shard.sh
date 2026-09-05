#!/bin/bash
# Evaluate one explicit family shard on train groups excluded from meta-training.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
RUN_DIR="${R5B_META_HELDOUT_RUN_DIR:?R5B_META_HELDOUT_RUN_DIR is required}"
GPU="${R5B_META_HELDOUT_GPU:?R5B_META_HELDOUT_GPU is required}"
FAMILY="${R5B_META_HELDOUT_FAMILY:?R5B_META_HELDOUT_FAMILY is required}"
CONFIG_PATH=configs/saved_time_v5/meta_hypernet_r5b_marmousi1_4m_v2_no_propagator.yaml
CONDITIONER=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/instance_adaptation/r5b_meta_no_propagator_fullbatch_lr1e6_r1_20260815/meta_pilot.pt
EXCLUSION=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/instance_adaptation/r5b_meta_no_propagator_fullbatch_lr1e6_r1_20260815/training_episode_manifest.json
STARTED_AT="$(date -Is)"
STARTED_EPOCH="$(date +%s)"

if [ "$#" -eq 0 ]; then
  echo "At least one held-out train sample ID is required" >&2
  exit 64
fi
SAMPLES=("$@")

mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"

write_status() {
  local target_path="$1"
  local status_value="$2"
  local return_code="$3"
  local finished_at="$4"
  local elapsed_seconds="$5"
  /root/miniconda3/bin/python - \
    "$target_path" "$status_value" "$return_code" "$STARTED_AT" \
    "$finished_at" "$elapsed_seconds" "$GPU" "$FAMILY" "${SAMPLES[@]}" <<'PY'
import json
import os
from pathlib import Path
import sys

target, status, return_code, started, finished, elapsed, gpu, family, *samples = sys.argv[1:]
payload = {
    "schema": "r5b_meta_heldout_train_shard_status_v1",
    "status": status,
    "return_code": int(return_code),
    "started_at": started,
    "finished_at": finished or None,
    "elapsed_seconds": int(elapsed),
    "supervisor_pid": os.getppid(),
    "gpu": int(gpu),
    "family": family,
    "sample_ids": samples,
    "selection_split": "train",
    "meta_training_groups_excluded": True,
    "future_truth_opened_after_adaptation_seal": True,
    "external_propagator": False,
}
path = Path(target)
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
  local status_value="failed"
  finished_at="$(date -Is)"
  finished_epoch="$(date +%s)"
  elapsed_seconds="$((finished_epoch - STARTED_EPOCH))"
  if [ "$rc" -eq 0 ] && [ -s "$RUN_DIR/summary.json" ]; then
    status_value="complete"
  fi
  write_status "$RUN_DIR/terminal.json" "$status_value" "$rc" "$finished_at" "$elapsed_seconds"
  echo "[r5b-meta-heldout] family=$FAMILY status=$status_value rc=$rc elapsed_s=$elapsed_seconds"
  exit "$rc"
}
trap finish_run EXIT

if [ -e "$RUN_DIR/terminal.json" ] || [ -e "$RUN_DIR/summary.json" ]; then
  echo "Refusing to overwrite held-out evaluation artifact: $RUN_DIR" >&2
  exit 64
fi

write_status "$RUN_DIR/status.json" "running" 0 "" 0

sample_args=()
for sample_id in "${SAMPLES[@]}"; do
  sample_args+=(--sample-id "$sample_id")
done

export CUDA_VISIBLE_DEVICES="$GPU"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=.:src

/root/miniconda3/bin/python -u scripts/run_meta_instance_adaptation.py \
  --config "$CONFIG_PATH" \
  --conditioner-checkpoint "$CONDITIONER" \
  --selection-split train \
  --exclude-manifest "$EXCLUSION" \
  "${sample_args[@]}" \
  --no-fields \
  --no-plots \
  --device cuda \
  --output-dir "$RUN_DIR"
