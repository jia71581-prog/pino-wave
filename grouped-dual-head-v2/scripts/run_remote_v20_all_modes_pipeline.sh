#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
V19_ART="$HOME_PROJECT/artifacts/saved_time_v19_residual_wakeup_bigbatch192_4gpu_remote_r1"
V20_ART="$HOME_PROJECT/artifacts/saved_time_v20_all_modes_bigbatch192_4gpu_remote_r1"
V21_ART="$HOME_PROJECT/artifacts/saved_time_v21_all_modes_long_bigbatch192_4gpu_remote_r1"
V22_ART="$HOME_PROJECT/artifacts/saved_time_v22_local_differential_bigbatch192_4gpu_remote_r1"
V23_ART="$HOME_PROJECT/artifacts/saved_time_v23_local_differential_long_bigbatch192_4gpu_remote_r1"
V19_CONFIG=configs/saved_time_v4/v19_residual_wakeup_bigbatch192_4gpu_remote.yaml
V20_CONFIG=configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml
V21_CONFIG=configs/saved_time_v4/generated/v21_all_modes_long_bigbatch192_4gpu_remote.yaml
V22_CONFIG=configs/saved_time_v4/generated/v22_local_differential_bigbatch192_4gpu_remote.yaml
V23_CONFIG=configs/saved_time_v4/generated/v23_local_differential_long_bigbatch192_4gpu_remote.yaml
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun

mkdir -p "$V20_ART" "$WORK/configs/saved_time_v4/generated"
exec >>"$V20_ART/pipeline.log" 2>&1
echo "$(date --iso-8601=seconds) waiting for v19 pilot"

while [[ ! -s "$V19_ART/pilot/terminal.json" ]]; do
  sleep 60
done
echo "$(date --iso-8601=seconds) v19 terminal detected"
sleep 15

cd "$WORK"
"$PYTHON" scripts/prepare_saved_time_all_modes_candidate.py \
  --candidate-config "$V19_CONFIG" \
  --output-config "$V20_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v20_all_modes_bigbatch192_4gpu_remote_r1 \
  --report "$V20_ART/parent_selection.json"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "$(date --iso-8601=seconds) starting v20 four-GPU smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V20_CONFIG" --smoke-updates 1 \
  >>"$V20_ART/smoke_screen.log" 2>&1

smoke_status="$($PYTHON - "$V20_ART/smoke/terminal.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
if [[ "$smoke_status" != complete ]]; then
  echo "$(date --iso-8601=seconds) v20 smoke status=$smoke_status"
  exit 2
fi

echo "$(date --iso-8601=seconds) starting v20 two-epoch pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V20_CONFIG" --pilot \
  >>"$V20_ART/pilot_screen.log" 2>&1
echo "$(date --iso-8601=seconds) v20 pilot finished"

set +e
"$PYTHON" scripts/gate_saved_time_all_modes_candidate.py \
  --selection-report "$V20_ART/parent_selection.json" \
  --candidate-metrics "$V20_ART/pilot/metrics.jsonl" \
  --output "$V20_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -eq 0 ]]; then
  mkdir -p "$V21_ART"
  "$PYTHON" scripts/prepare_saved_time_long_continuation.py \
    --candidate-config "$V20_CONFIG" \
    --output-config "$V21_CONFIG" \
    --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v21_all_modes_long_bigbatch192_4gpu_remote_r1 \
    --report "$V21_ART/parent_selection.json" \
    --epochs 40

  echo "$(date --iso-8601=seconds) starting gate-approved v21 long training"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$V21_CONFIG" \
    >>"$V21_ART/long_screen.log" 2>&1
  echo "$(date --iso-8601=seconds) v21 long training finished"
else
  echo "$(date --iso-8601=seconds) v20 gate rejected; starting local-differential fallback"
  mkdir -p "$V22_ART"
  "$PYTHON" scripts/prepare_saved_time_local_differential_candidate.py \
    --candidate-config "$V19_CONFIG" \
    --candidate-config "$V20_CONFIG" \
    --output-config "$V22_CONFIG" \
    --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v22_local_differential_bigbatch192_4gpu_remote_r1 \
    --report "$V22_ART/parent_selection.json"

  echo "$(date --iso-8601=seconds) starting v22 four-GPU smoke"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$V22_CONFIG" --smoke-updates 1 \
    >>"$V22_ART/smoke_screen.log" 2>&1

  fallback_smoke_status="$($PYTHON - "$V22_ART/smoke/terminal.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"
  if [[ "$fallback_smoke_status" != complete ]]; then
    echo "$(date --iso-8601=seconds) v22 smoke status=$fallback_smoke_status"
    exit 2
  fi

  echo "$(date --iso-8601=seconds) starting v22 two-epoch pilot"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$V22_CONFIG" --pilot \
    >>"$V22_ART/pilot_screen.log" 2>&1

  set +e
  "$PYTHON" scripts/gate_saved_time_local_differential_candidate.py \
    --selection-report "$V22_ART/parent_selection.json" \
    --candidate-metrics "$V22_ART/pilot/metrics.jsonl" \
    --output "$V22_ART/evidence_gate.json"
  fallback_gate_rc=$?
  set -e
  if [[ $fallback_gate_rc -ne 0 ]]; then
    echo "$(date --iso-8601=seconds) v22 evidence gate rejected long training rc=$fallback_gate_rc"
    exit "$fallback_gate_rc"
  fi

  mkdir -p "$V23_ART"
  "$PYTHON" scripts/prepare_saved_time_long_continuation.py \
    --candidate-config "$V22_CONFIG" \
    --output-config "$V23_CONFIG" \
    --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v23_local_differential_long_bigbatch192_4gpu_remote_r1 \
    --report "$V23_ART/parent_selection.json" \
    --epochs 40

  echo "$(date --iso-8601=seconds) starting gate-approved v23 long training"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$V23_CONFIG" \
    >>"$V23_ART/long_screen.log" 2>&1
  echo "$(date --iso-8601=seconds) v23 long training finished"
fi
