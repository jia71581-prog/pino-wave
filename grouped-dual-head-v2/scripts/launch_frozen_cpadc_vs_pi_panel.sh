#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 {validation|test_id} OUTPUT_DIR" >&2
  exit 64
fi

split_name=$1
output_dir=$2
if [[ "$split_name" != validation && "$split_name" != test_id ]]; then
  echo "split must be validation or test_id" >&2
  exit 64
fi

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python_bin=/root/miniconda3/bin/python
config="$project_root/configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2.yaml"
checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7/calibration/calibrated.pt
checkpoint_sha256=6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b
parent_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1/pilot/checkpoints/epoch_0001.pt
parent_checkpoint_sha256=395b83560db3342b1e2331de2d8dde2994640d56a53dfbec8c99ece4bbb177bf
validation_travel=/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5
test_id_travel=/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2_test_id.h5
pi_dir="$project_root/results/frozen_pi_deeponet_${split_name}_r10_20260815"
archived_cpadc_root=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7
travel_time_h5=$validation_travel
if [[ "$split_name" == test_id ]]; then
  travel_time_h5=$test_id_travel
  archived_cpadc_root=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7_test_id_confirmation
fi

mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
exec 9>"$output_dir/launch.lock"
if ! flock -n 9; then
  echo "another exact-panel comparison owns $output_dir" >&2
  exit 73
fi
cd "$project_root"

write_terminal() {
  local status=$1
  local stage=$2
  local rc=$3
  "$python_bin" - "$output_dir/terminal.json" "$status" "$stage" "$rc" "$split_name" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "frozen_cpadc_vs_pi_panel_terminal_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "split": sys.argv[5],
    "supervisor_pid": os.getppid(),
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

children=()
on_exit() {
  local rc=$?
  for child in "${children[@]:-}"; do
    if kill -0 "$child" 2>/dev/null; then
      kill -TERM "$child" 2>/dev/null || true
    fi
  done
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed interrupted "$rc" || true
  fi
}
trap on_exit EXIT

write_terminal running verifying 0
if [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" != "$checkpoint_sha256" ]]; then
  echo "frozen CPADC-R7 checkpoint hash mismatch" >&2
  exit 74
fi
if [[ "$(sha256sum "$parent_checkpoint" | awk '{print $1}')" != "$parent_checkpoint_sha256" ]]; then
  echo "frozen CPADC-R7 parent checkpoint hash mismatch" >&2
  exit 74
fi
for index in 0 1 2 3; do
  test -f "$pi_dir/worker_${index}.json"
done
test -f "$pi_dir/complete_${split_name}_comparison.json"
test -f "$travel_time_h5"

compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$compute_pids" ]]; then
  echo "comparison launch requires four free GPUs; active PIDs: $compute_pids" >&2
  exit 75
fi

write_terminal running evaluating 0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pi_worker_args=()
for index in 0 1 2 3; do
  pi_worker_args+=("$pi_dir/worker_${index}.json")
done
for index in 0 1 2 3; do
  mkdir -p "$output_dir/shard_$index"
  CUDA_VISIBLE_DEVICES="$index" "$python_bin" scripts/evaluate_cpadc_on_pi_panel.py \
    --config "$config" \
    --basis-checkpoint "$checkpoint" \
    --parent-checkpoint "$parent_checkpoint" \
    --archived-cpadc-root "$archived_cpadc_root" \
    --travel-time-h5 "$travel_time_h5" \
    --pi-workers "${pi_worker_args[@]}" \
    --split "$split_name" \
    --shard-index "$index" \
    --num-shards 4 \
    --device cuda \
    --output-dir "$output_dir/shard_$index" \
    --output "$output_dir/worker_${index}.json" \
    >"$output_dir/worker_${index}.log" 2>&1 &
  children+=("$!")
done
"$python_bin" - "$output_dir/children.json" "$split_name" "${children[@]}" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "split": sys.argv[2],
    "supervisor_pid": os.getppid(),
    "child_pids": [int(value) for value in sys.argv[3:]],
    "started_unix_s": time.time(),
}, indent=2, sort_keys=True) + "\n")
PY
for index in 0 1 2 3; do
  child=${children[$index]}
  if wait "$child"; then
    echo 0 >"$output_dir/worker_${index}.exitcode"
  else
    rc=$?
    echo "$rc" >"$output_dir/worker_${index}.exitcode"
    exit "$rc"
  fi
done
children=()

write_terminal running aggregating 0
worker_args=()
for index in 0 1 2 3; do
  worker_args+=("$output_dir/worker_${index}.json")
done
"$python_bin" scripts/aggregate_cpadc_vs_pi_panel.py \
  --workers "${worker_args[@]}" \
  --pi-complete "$pi_dir/complete_${split_name}_comparison.json" \
  --split "$split_name" \
  --output "$output_dir/complete_${split_name}_comparison.json" \
  >"$output_dir/aggregate.log" 2>&1
write_terminal complete complete 0
trap - EXIT
