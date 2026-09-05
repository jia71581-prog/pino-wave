#!/bin/bash
# Sealed three-family Phase4b deployment gate for one CHONKNORIS checkpoint.
set -Eeuo pipefail

REPO_ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
CONFIG_PATH="${PHASE4B_GATE_CONFIG:-configs/saved_time_v5/meta_hypernet_chonknoris_phase4b.yaml}"
CANDIDATE_CHECKPOINT="${PHASE4B_GATE_CHECKPOINT:?Set PHASE4B_GATE_CHECKPOINT}"
OUTPUT_DIR="${PHASE4B_GATE_OUTPUT_DIR:?Set PHASE4B_GATE_OUTPUT_DIR}"
STARTED_AT="$(date -Is)"
STARTED_EPOCH="$(date +%s)"
HARD_PROJECT_ONSET="${PHASE4B_GATE_HARD_PROJECT_ONSET:-0}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"
if [ ! -s "$CANDIDATE_CHECKPOINT" ]; then
  echo "Candidate checkpoint is missing: $CANDIDATE_CHECKPOINT" >&2
  exit 66
fi
if [ -e "$OUTPUT_DIR/summary.json" ]; then
  echo "Refusing to overwrite an existing validation summary: $OUTPUT_DIR" >&2
  exit 64
fi

write_terminal() {
  local status_value="$1"
  local return_code="$2"
  local finished_at="$3"
  local elapsed_seconds="$4"
  /root/miniconda3/bin/python - \
    "$OUTPUT_DIR/terminal.json" "$status_value" "$return_code" \
    "$STARTED_AT" "$finished_at" "$elapsed_seconds" \
    "$CANDIDATE_CHECKPOINT" "$HARD_PROJECT_ONSET" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

target, status, rc, started, finished, elapsed, checkpoint, hard_project = sys.argv[1:]
digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
payload = {
    "schema": "phase4b_chonknoris_heldout3_status_v1",
    "status": status,
    "return_code": int(rc),
    "started_at": started,
    "finished_at": finished or None,
    "elapsed_seconds": int(elapsed),
    "candidate_checkpoint": str(Path(checkpoint).resolve()),
    "candidate_checkpoint_sha256": digest,
    "hard_project_onset": bool(int(hard_project)),
    "sample_ids": [
        "validation_uniform_00000",
        "validation_layered_00000",
        "validation_marmousi_00000",
    ],
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
  local status_value=failed
  finished_at="$(date -Is)"
  finished_epoch="$(date +%s)"
  if [ "$rc" -eq 0 ] && [ -s "$OUTPUT_DIR/summary.json" ]; then
    status_value=complete
  fi
  write_terminal \
    "$status_value" "$rc" "$finished_at" "$((finished_epoch - STARTED_EPOCH))"
  echo "[phase4b-heldout3] status=$status_value rc=$rc finished_at=$finished_at"
  exit "$rc"
}
trap finish_run EXIT

write_terminal running 0 "" 0
export CUDA_VISIBLE_DEVICES="${PHASE4B_GATE_GPU:-0}"
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

HARD_PROJECT_ARGS=()
if [ "$HARD_PROJECT_ONSET" = 1 ]; then
  HARD_PROJECT_ARGS+=(--hard-project-onset)
fi

/root/miniconda3/bin/python scripts/run_v5_instance_adaptation.py \
  --config "$CONFIG_PATH" \
  --conditioner-checkpoint "$CANDIDATE_CHECKPOINT" \
  --sample-id validation_uniform_00000 \
  --sample-id validation_layered_00000 \
  --sample-id validation_marmousi_00000 \
  --deployment-lora \
  "${HARD_PROJECT_ARGS[@]}" \
  --no-plots \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
