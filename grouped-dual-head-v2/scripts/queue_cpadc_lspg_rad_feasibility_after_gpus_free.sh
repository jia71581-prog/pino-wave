#!/usr/bin/env bash
# Wait for all four GPUs to become naturally idle, then launch the registered
# train-only CPADC-LSPG-RAD feasibility pilot.  This supervisor never signals,
# stops, restarts, or changes the priority of an existing process.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
config_path="$project_root/configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2_target10_rank32.yaml"
config_sha256=4f05d3f62613cca6cafdee9b6934f0d86ddcd43118c618520cec195ab39c416a
train_script="$project_root/scripts/train_causal_defect_basis.py"
train_script_sha256=09ed5b68eeedb67a3280c4d7584f269e038f3080971a051066cdb86c7ec9b60d
parent_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r3_adamw99/run/checkpoints/epoch_0000.pt
parent_sha256=60a2130bea7d725faff405096d5f459196cd566f5aca3f6e83b97112422453bc
output_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_lspg_rad_r1_trainonly_feasibility_a3r3e0_r5_20260815
poll_seconds=60

mkdir -p "$output_dir"
exec 9>"$output_dir/queue.lock"
if ! flock -n 9; then
  echo "another CPADC-LSPG-RAD feasibility queue owns the lock" >&2
  exit 73
fi

write_status() {
  local status=$1
  local detail=$2
  local path=$3
  "$python_bin" - "$path" "$status" "$detail" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
payload = {
    "schema": "cpadc_lspg_rad_feasibility_queue_v2",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "supervisor_pid": os.getppid(),
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

status_path="$output_dir/queue_status.json"
terminal_path="$output_dir/queue_terminal.json"
if [[ "$(sha256sum "$config_path" | awk '{print $1}')" != "$config_sha256" ]]; then
  write_status failed config_digest_mismatch "$terminal_path"
  exit 74
fi
if [[ "$(sha256sum "$train_script" | awk '{print $1}')" != "$train_script_sha256" ]]; then
  write_status failed train_script_digest_mismatch "$terminal_path"
  exit 77
fi
if [[ "$(sha256sum "$parent_checkpoint" | awk '{print $1}')" != "$parent_sha256" ]]; then
  write_status failed parent_digest_mismatch "$terminal_path"
  exit 75
fi
if [[ -f "$output_dir/run/terminal.json" ]]; then
  write_status refused existing_run_terminal "$terminal_path"
  exit 76
fi

while true; do
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ "$gpu_count" -eq 4 && -z "$compute_pids" ]]; then
    break
  fi
  write_status waiting_for_four_idle_gpus "compute_pids=${compute_pids//$'\n'/,}" "$status_path"
  sleep "$poll_seconds"
done

write_status launching_train_only_feasibility all_four_gpus_idle "$status_path"
cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
set +e
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_causal_defect_basis.py \
  --config "$config_path" \
  --output-dir "$output_dir/run" \
  --device cuda \
  --per-family 4 \
  --epochs 1 \
  --warmup-epochs 0 \
  --patch-size 201 \
  >"$output_dir/train.log" 2>&1
return_code=$?
set -e
if [[ "$return_code" -eq 0 && -f "$output_dir/run/terminal.json" ]]; then
  write_status complete train_only_feasibility_finished "$terminal_path"
else
  write_status failed "torchrun_return_code=$return_code" "$terminal_path"
fi
exit "$return_code"
