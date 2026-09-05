#!/usr/bin/env bash
# Run the one-variable 3x3 weak-defect CPADC arm after the already registered
# width-1/ablation chain, then compare on the identical disjoint train protocol.
set -Eeuo pipefail

readonly project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
readonly python_bin=/root/miniconda3/bin/python
readonly torchrun_bin=/root/miniconda3/bin/torchrun
readonly current_chain_terminal=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/continuation_after_pi_resume1_v4_20260815/terminal.json
readonly strong_summary=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_lspg_rad_r1_trainonly_pair_a3r3e0_r4_20260815/candidate_eval/summary.json
readonly weak_config="$project_root/configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2_target10_rank32_weak3.yaml"
readonly weak_config_sha256=c726343cab3fe96c881bcf611253163c1c0ad964e091f1fe2dd1f6cebad3e11f
readonly train_script="$project_root/scripts/train_causal_defect_basis.py"
readonly train_script_sha256=09ed5b68eeedb67a3280c4d7584f269e038f3080971a051066cdb86c7ec9b60d
readonly evaluation_script="$project_root/scripts/run_causal_defect_adaptation.py"
readonly evaluation_script_sha256=c8eb6a82fccb624fc71d6748a22d54a6c00ef0a791a979e57c7456cef5acf47f
readonly comparison_script="$project_root/scripts/compare_cpadc_trainonly_pair.py"
readonly comparison_script_sha256=2674505c0106cce05bf84f5a12608c6fd578ebe9620c21c937f68978ef5a546b
readonly precheck="$project_root/results/cpadc_weak_closure_trainonly_precheck_20260815.json"
readonly precheck_sha256=be66ca07815a84fc2af07c94126e890976688105a5b0bcf3cb28316d6a797a20
readonly parent_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r3_adamw99/run/checkpoints/epoch_0000.pt
readonly parent_sha256=60a2130bea7d725faff405096d5f459196cd566f5aca3f6e83b97112422453bc
readonly output_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_weak3_lspg_rad_trainonly_a3r3e0_r4_20260815
readonly poll_seconds=60

mkdir -p "$output_dir"
exec 9>"$output_dir/supervisor.lock"
if ! flock -n 9; then
  echo "another CPADC weak3 supervisor owns the lock" >&2
  exit 73
fi
if [[ -f "$output_dir/terminal.json" ]]; then
  echo "refusing to replace existing weak3 terminal" >&2
  exit 74
fi
printf '%s\n' "$$" >"$output_dir/supervisor.pid"

write_record() {
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
    "schema": "cpadc_weak3_trainonly_queue_v1",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "supervisor_pid": os.getppid(),
    "updated_unix_s": time.time(),
    "selection_split": "train",
    "validation_access": False,
    "test_id_access": False,
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

readonly status_path="$output_dir/status.json"
readonly terminal_path="$output_dir/terminal.json"
for specification in \
  "$weak_config:$weak_config_sha256" \
  "$train_script:$train_script_sha256" \
  "$evaluation_script:$evaluation_script_sha256" \
  "$comparison_script:$comparison_script_sha256" \
  "$precheck:$precheck_sha256" \
  "$parent_checkpoint:$parent_sha256"; do
  path=${specification%:*}
  expected=${specification##*:}
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    write_record failed "digest_mismatch:$path" "$terminal_path"
    exit 75
  fi
done

"$python_bin" - "$precheck" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("selection_split") != "train":
    raise SystemExit("precheck_not_train_only")
if payload.get("validation_access") is not False or payload.get("test_id_access") is not False:
    raise SystemExit("precheck_touched_sealed_splits")
candidate = payload.get("candidate") or {}
if int(candidate.get("width", -1)) != 3 or candidate.get("precheck_pass") is not True:
    raise SystemExit("weak3_precheck_failed")
PY

while [[ ! -f "$current_chain_terminal" ]]; do
  write_record waiting_for_registered_pair current_chain_terminal_absent "$status_path"
  sleep "$poll_seconds"
done
if [[ ! -f "$strong_summary" ]]; then
  write_record blocked registered_width1_summary_absent "$terminal_path"
  exit 76
fi

wait_for_four_idle_gpus() {
  while true; do
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
    if [[ "$gpu_count" -eq 4 && -z "$compute_pids" ]]; then
      return
    fi
    write_record waiting_for_four_idle_gpus "compute_pids=${compute_pids//$'\n'/,}" "$status_path"
    sleep "$poll_seconds"
  done
}

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
wait_for_four_idle_gpus
write_record training_weak3 train_only_per_family4_epoch1 "$status_path"
set +e
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_causal_defect_basis.py \
  --config "$weak_config" \
  --output-dir "$output_dir/run" \
  --device cuda \
  --per-family 4 \
  --epochs 1 \
  --warmup-epochs 0 \
  --patch-size 201 \
  >"$output_dir/train.log" 2>&1
train_return_code=$?
set -e
if [[ "$train_return_code" -ne 0 || ! -f "$output_dir/run/terminal.json" ]]; then
  write_record failed "weak3_train_return_code=$train_return_code" "$terminal_path"
  exit 77
fi

weak_checkpoint=$("$python_bin" - "$output_dir/run" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

run_dir = Path(sys.argv[1])
terminal = json.loads((run_dir / "terminal.json").read_text())
if terminal.get("status") != "complete":
    raise SystemExit("training_terminal_not_complete")
checkpoint = Path(str(terminal.get("checkpoint", "")))
if not checkpoint.is_file():
    raise SystemExit("checkpoint_absent")
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while block := stream.read(8 * 1024 * 1024):
        hasher.update(block)
if hasher.hexdigest() != terminal.get("checkpoint_sha256"):
    raise SystemExit("checkpoint_digest_mismatch")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
contract = payload.get("online_solve_contract") or {}
if contract.get("name") != "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7":
    raise SystemExit("weak_solve_contract_mismatch")
if int(contract.get("online_defect_test_function_width", -1)) != 3:
    raise SystemExit("weak_test_function_width_mismatch")
print(checkpoint)
PY
)

wait_for_four_idle_gpus
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
write_record evaluating_weak3 disjoint_train_only "$status_path"
set +e
"$python_bin" scripts/run_causal_defect_adaptation.py \
  --config "$weak_config" \
  --basis-checkpoint "$weak_checkpoint" \
  --output-dir "$output_dir/weak_eval" \
  --device cuda \
  --adaptation-device cpu \
  --calibration-per-family 4 \
  --calibration-seed 10372 \
  --basis-training-per-family 4 \
  --no-fields \
  >"$output_dir/weak_eval.log" 2>&1
evaluation_return_code=$?
set -e
if [[ "$evaluation_return_code" -ne 0 || ! -f "$output_dir/weak_eval/summary.json" ]]; then
  write_record failed "weak3_evaluation_return_code=$evaluation_return_code" "$terminal_path"
  exit 78
fi

write_record comparing_against_width1 identical_disjoint_train_protocol "$status_path"
"$python_bin" scripts/compare_cpadc_trainonly_pair.py \
  --candidate-summary "$output_dir/weak_eval/summary.json" \
  --ablation-summary "$strong_summary" \
  --output "$output_dir/weak3_vs_width1_gate.json" \
  >"$output_dir/compare.log" 2>&1
passed=$("$python_bin" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["passed"])).lower())' "$output_dir/weak3_vs_width1_gate.json")
if [[ "$passed" == "true" ]]; then
  write_record complete weak3_trainonly_pair_gate_passed "$terminal_path"
else
  write_record rejected weak3_trainonly_pair_gate_failed "$terminal_path"
fi
