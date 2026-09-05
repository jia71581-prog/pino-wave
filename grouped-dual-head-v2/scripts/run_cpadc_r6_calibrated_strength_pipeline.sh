#!/usr/bin/env bash
# Calibrate a train-only CPADC causal-strength abstention rule, then run the
# sealed 480-record validation protocol with one shard per GPU.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2.yaml
source_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_bilevel_trust_a3e1_marmousi1_4m_v2_r5/run/best.pt
source_checkpoint_sha256=4c01dea0dafd876106a8798e4962fba6cf25d3635a61c75429b5e7c32d1d09bc
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_calibrated_strength_a3e1_marmousi1_4m_v2_r6
python_bin=/root/miniconda3/bin/python
shard_count=4
calibration_per_family=64
calibration_seed=10372
stage=initializing
children=()

mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/pipeline.lock"
if ! flock -n 9; then
  echo "another r6 CPADC pipeline owns the lock" >&2
  exit 73
fi
cd "$project_root"

write_terminal() {
  local status=$1
  local rc=$2
  "$python_bin" - "$artifact_dir/pipeline_terminal.json" "$status" "$stage" "$rc" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "pipeline_pid": os.getppid(),
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

on_exit() {
  local rc=$?
  for pid in "${children[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed "$rc" || true
  fi
}
trap on_exit EXIT

write_children() {
  local path=$1
  shift
  "$python_bin" - "$path" "$stage" "$@" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "controller_pid": os.getppid(),
    "stage": sys.argv[2],
    "child_pids": [int(value) for value in sys.argv[3:]],
    "started_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

require_four_free_gpus() {
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  if [[ "$gpu_count" -ne "$shard_count" ]]; then
    echo "expected $shard_count GPUs, observed $gpu_count" >&2
    exit 74
  fi
  local compute_pids
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ -n "$compute_pids" ]]; then
    echo "GPU compute processes appeared before stage $stage: $compute_pids" >&2
    exit 75
  fi
}

observed_sha256=$(sha256sum "$source_checkpoint" | awk '{print $1}')
if [[ "$observed_sha256" != "$source_checkpoint_sha256" ]]; then
  echo "r5 source checkpoint hash mismatch: $observed_sha256" >&2
  exit 76
fi
require_four_free_gpus
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

stage=disjoint_train_risk_calibration_4way
write_terminal running 0
mkdir -p "$artifact_dir/calibration_shards"
for shard_index in 0 1 2 3; do
  shard_dir="$artifact_dir/calibration_shards/shard_$shard_index"
  mkdir -p "$shard_dir"
  CUDA_VISIBLE_DEVICES="$shard_index" "$python_bin" \
    scripts/run_causal_defect_adaptation.py \
    --config "$config" \
    --basis-checkpoint "$source_checkpoint" \
    --output-dir "$shard_dir" \
    --calibration-per-family "$calibration_per_family" \
    --calibration-seed "$calibration_seed" \
    --shard-index "$shard_index" \
    --shard-count "$shard_count" \
    --no-fields --device cuda \
    >"$artifact_dir/calibration_shards/shard_${shard_index}.log" 2>&1 &
  children+=("$!")
done
write_children "$artifact_dir/calibration_children.json" "${children[@]}"
for pid in "${children[@]}"; do
  wait "$pid"
done
children=()

stage=calibrate_strength_threshold
write_terminal running 0
calibration_args=()
for shard_index in 0 1 2 3; do
  calibration_args+=(--shard-dir "$artifact_dir/calibration_shards/shard_$shard_index")
done
"$python_bin" scripts/calibrate_causal_defect_risk.py \
  --config "$config" \
  --basis-checkpoint "$source_checkpoint" \
  --output-dir "$artifact_dir/calibration" \
  --calibration-per-family "$calibration_per_family" \
  --calibration-seed "$calibration_seed" \
  --minimum-nonworse-fraction 0.90 \
  --minimum-family-nonworse-fraction 0.90 \
  --minimum-mean-improvement 0.01 \
  "${calibration_args[@]}" \
  >"$artifact_dir/calibration.log" 2>&1

calibrated_checkpoint="$artifact_dir/calibration/calibrated.pt"
"$python_bin" - "$artifact_dir/calibration/terminal.json" "$calibrated_checkpoint" <<'PY'
import json, pathlib, sys, torch
terminal = json.loads(pathlib.Path(sys.argv[1]).read_text())
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
selection = terminal["risk_calibration"]["selection"]
checks = {
    "terminal_complete": terminal.get("status") == "complete",
    "schema_version": int(checkpoint.get("schema_version", 0)) == 5,
    "train_only_truth": (checkpoint.get("risk_calibration") or {}).get("future_truth_scope") == "disjoint_train_split_only",
    "record_count": int((checkpoint.get("risk_calibration") or {}).get("record_count", 0)) == 192,
    "online_future_truth_unused": terminal.get("online_future_truth_used") is False,
    "strength_floor_positive": float(selection.get("strength_floor", 0.0)) > 0.0,
    "calibration_nonworse": float(selection.get("nonworse_fraction", 0.0)) >= 0.90,
    "calibration_improved": float(selection.get("mean_relative_improvement", 0.0)) >= 0.01,
    "solve_contract": (checkpoint.get("online_solve_contract") or {}).get("name") == "ridge_direction_calibrated_strength_abstention_v1",
}
if not all(checks.values()):
    raise ValueError(f"r6 calibration gate failed: {checks}")
print(checks)
print({"strength_floor": selection["strength_floor"], "mean_improvement": selection["mean_relative_improvement"], "nonworse_fraction": selection["nonworse_fraction"]})
PY

stage=sealed_full_validation_4way
write_terminal running 0
require_four_free_gpus
mkdir -p "$artifact_dir/validation_shards"
for shard_index in 0 1 2 3; do
  shard_dir="$artifact_dir/validation_shards/shard_$shard_index"
  mkdir -p "$shard_dir"
  CUDA_VISIBLE_DEVICES="$shard_index" "$python_bin" \
    scripts/run_causal_defect_adaptation.py \
    --config "$config" \
    --basis-checkpoint "$calibrated_checkpoint" \
    --output-dir "$shard_dir" \
    --all-validation \
    --shard-index "$shard_index" \
    --shard-count "$shard_count" \
    --no-fields --device cuda \
    >"$artifact_dir/validation_shards/shard_${shard_index}.log" 2>&1 &
  children+=("$!")
done
write_children "$artifact_dir/validation_children.json" "${children[@]}"
for pid in "${children[@]}"; do
  wait "$pid"
done
children=()

stage=merge_and_gate
write_terminal running 0
validation_args=()
for shard_index in 0 1 2 3; do
  validation_args+=(--shard-dir "$artifact_dir/validation_shards/shard_$shard_index")
done
"$python_bin" scripts/merge_causal_defect_evaluations.py \
  --config "$config" \
  --output-dir "$artifact_dir/evaluation" \
  "${validation_args[@]}" \
  >"$artifact_dir/merge.log" 2>&1

stage=complete
"$python_bin" - "$artifact_dir/evaluation/terminal.json" "$artifact_dir/pipeline_terminal.json" <<'PY'
import json, os, pathlib, sys, time
evaluation = json.loads(pathlib.Path(sys.argv[1]).read_text())
if evaluation.get("status") != "complete":
    raise ValueError("r6 sealed CPADC evaluation did not complete")
payload = {
    "status": "complete",
    "stage": "complete",
    "exit_code": 0,
    "pipeline_pid": os.getppid(),
    "updated_unix_s": time.time(),
    "calibration_record_count": 192,
    "validation_record_count": 480,
    "validation_shard_count": 4,
    "same_protocol_validation_passed": evaluation.get("same_protocol_validation_passed") is True,
    "claim": evaluation.get("claim"),
}
path = pathlib.Path(sys.argv[2])
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
trap - EXIT
