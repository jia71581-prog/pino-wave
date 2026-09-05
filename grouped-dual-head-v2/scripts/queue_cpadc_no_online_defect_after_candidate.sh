#!/usr/bin/env bash
# Conditionally launch the strict no-online-defect ablation after the registered
# CPADC-LSPG-RAD feasibility candidate completes successfully.  This supervisor
# only reads process/artifact state and never signals or reprioritizes jobs.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
config_path="$project_root/configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2_target10_rank32_no_online_defect_ablation.yaml"
config_sha256=32ccf7eb169562790bb29146e5442db575146432ecb3bd384b341725297ac133
train_script="$project_root/scripts/train_causal_defect_basis.py"
train_script_sha256=09ed5b68eeedb67a3280c4d7584f269e038f3080971a051066cdb86c7ec9b60d
parent_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r3_adamw99/run/checkpoints/epoch_0000.pt
parent_sha256=60a2130bea7d725faff405096d5f459196cd566f5aca3f6e83b97112422453bc
candidate_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_lspg_rad_r1_trainonly_feasibility_a3r3e0_r5_20260815
output_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_no_online_defect_trainonly_feasibility_a3r3e0_r4_20260815
poll_seconds=60

mkdir -p "$output_dir"
exec 9>"$output_dir/queue.lock"
if ! flock -n 9; then
  echo "another no-online-defect feasibility queue owns the lock" >&2
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
    "schema": "cpadc_no_online_defect_conditional_queue_v1",
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
  exit 75
fi
if [[ "$(sha256sum "$parent_checkpoint" | awk '{print $1}')" != "$parent_sha256" ]]; then
  write_status failed parent_digest_mismatch "$terminal_path"
  exit 76
fi
if [[ -f "$output_dir/run/terminal.json" ]]; then
  write_status refused existing_run_terminal "$terminal_path"
  exit 77
fi

candidate_queue_terminal="$candidate_dir/queue_terminal.json"
while [[ ! -f "$candidate_queue_terminal" ]]; do
  write_status waiting_for_candidate candidate_terminal_absent "$status_path"
  sleep "$poll_seconds"
done

set +e
candidate_detail=$("$python_bin" - "$candidate_dir" 2>&1 <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
queue_terminal = json.loads((root / "queue_terminal.json").read_text())
if queue_terminal.get("status") != "complete":
    raise SystemExit("candidate_queue_not_complete")
terminal_path = root / "run" / "terminal.json"
if not terminal_path.is_file():
    raise SystemExit("candidate_training_terminal_absent")
terminal = json.loads(terminal_path.read_text())
improvement = float(terminal.get("best_train_patch_relative_improvement", math.nan))
if terminal.get("status") != "complete" or not math.isfinite(improvement) or improvement <= 0.0:
    raise SystemExit("candidate_nonpositive_or_invalid_train_improvement")
checkpoint = Path(str(terminal.get("checkpoint", "")))
if not checkpoint.is_file():
    raise SystemExit("candidate_checkpoint_absent")
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while block := stream.read(8 * 1024 * 1024):
        hasher.update(block)
digest = hasher.hexdigest()
if digest != terminal.get("checkpoint_sha256"):
    raise SystemExit("candidate_checkpoint_digest_mismatch")
print(f"candidate_positive_improvement={improvement:.17g},checkpoint_sha256={digest}")
PY
)
candidate_return_code=$?
set -e
if [[ "$candidate_return_code" -ne 0 ]]; then
  write_status blocked "candidate_gate_failed:${candidate_detail}" "$terminal_path"
  exit 78
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

write_status launching_strict_ablation "$candidate_detail" "$status_path"
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
  write_status complete strict_ablation_feasibility_finished "$terminal_path"
else
  write_status failed "torchrun_return_code=$return_code" "$terminal_path"
fi
exit "$return_code"
