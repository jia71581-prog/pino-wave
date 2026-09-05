#!/usr/bin/env bash
# Run the sealed, disjoint-train paired CPADC evaluation only after both
# feasibility arms finish.  This supervisor never opens validation/test_id and
# never signals, restarts, or reprioritizes another process.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python_bin=/root/miniconda3/bin/python
candidate_config="$project_root/configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2_target10_rank32.yaml"
candidate_config_sha256=4f05d3f62613cca6cafdee9b6934f0d86ddcd43118c618520cec195ab39c416a
ablation_config="$project_root/configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2_target10_rank32_no_online_defect_ablation.yaml"
ablation_config_sha256=32ccf7eb169562790bb29146e5442db575146432ecb3bd384b341725297ac133
evaluation_script="$project_root/scripts/run_causal_defect_adaptation.py"
evaluation_script_sha256=c8eb6a82fccb624fc71d6748a22d54a6c00ef0a791a979e57c7456cef5acf47f
comparison_script="$project_root/scripts/compare_cpadc_trainonly_pair.py"
comparison_script_sha256=2674505c0106cce05bf84f5a12608c6fd578ebe9620c21c937f68978ef5a546b
parent_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r3_adamw99/run/checkpoints/epoch_0000.pt
parent_sha256=60a2130bea7d725faff405096d5f459196cd566f5aca3f6e83b97112422453bc
candidate_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_lspg_rad_r1_trainonly_feasibility_a3r3e0_r5_20260815
ablation_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_no_online_defect_trainonly_feasibility_a3r3e0_r4_20260815
output_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_lspg_rad_r1_trainonly_pair_a3r3e0_r4_20260815
calibration_per_family=4
calibration_seed=10372
basis_training_per_family=4
poll_seconds=60

mkdir -p "$output_dir"
exec 9>"$output_dir/queue.lock"
if ! flock -n 9; then
  echo "another CPADC train-only pair queue owns the lock" >&2
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
    "schema": "cpadc_trainonly_pair_conditional_queue_v1",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "supervisor_pid": os.getppid(),
    "updated_unix_s": time.time(),
    "validation_access": False,
    "test_id_access": False,
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

status_path="$output_dir/queue_status.json"
terminal_path="$output_dir/queue_terminal.json"
for specification in \
  "$candidate_config:$candidate_config_sha256" \
  "$ablation_config:$ablation_config_sha256" \
  "$evaluation_script:$evaluation_script_sha256" \
  "$comparison_script:$comparison_script_sha256" \
  "$parent_checkpoint:$parent_sha256"; do
  path=${specification%:*}
  expected=${specification##*:}
  if [[ "$(sha256sum "$path" | awk '{print $1}')" != "$expected" ]]; then
    write_status failed "digest_mismatch:$path" "$terminal_path"
    exit 74
  fi
done
if [[ -f "$output_dir/paired_gate.json" ]]; then
  write_status refused existing_paired_gate "$terminal_path"
  exit 75
fi

ablation_queue_terminal="$ablation_dir/queue_terminal.json"
while [[ ! -f "$ablation_queue_terminal" ]]; do
  write_status waiting_for_ablation ablation_terminal_absent "$status_path"
  sleep "$poll_seconds"
done
if [[ "$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' "$ablation_queue_terminal")" != "complete" ]]; then
  write_status blocked ablation_queue_not_complete "$terminal_path"
  exit 76
fi

verify_checkpoint() {
  local run_dir=$1
  "$python_bin" - "$run_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

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
digest = hasher.hexdigest()
if digest != terminal.get("checkpoint_sha256"):
    raise SystemExit("checkpoint_digest_mismatch")
print(checkpoint)
PY
}

set +e
candidate_checkpoint=$(verify_checkpoint "$candidate_dir/run" 2>&1)
candidate_return_code=$?
ablation_checkpoint=$(verify_checkpoint "$ablation_dir/run" 2>&1)
ablation_return_code=$?
set -e
if [[ "$candidate_return_code" -ne 0 || "$ablation_return_code" -ne 0 ]]; then
  write_status blocked "checkpoint_gate_failed:candidate=$candidate_checkpoint;ablation=$ablation_checkpoint" "$terminal_path"
  exit 77
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

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
candidate_eval="$output_dir/candidate_eval"
ablation_eval="$output_dir/ablation_eval"

write_status evaluating_candidate disjoint_train_only "$status_path"
set +e
"$python_bin" scripts/run_causal_defect_adaptation.py \
  --config "$candidate_config" \
  --basis-checkpoint "$candidate_checkpoint" \
  --output-dir "$candidate_eval" \
  --device cuda \
  --adaptation-device cpu \
  --calibration-per-family "$calibration_per_family" \
  --calibration-seed "$calibration_seed" \
  --basis-training-per-family "$basis_training_per_family" \
  --no-fields \
  >"$output_dir/candidate_eval.log" 2>&1
candidate_eval_return_code=$?
set -e
if [[ "$candidate_eval_return_code" -ne 0 || ! -f "$candidate_eval/summary.json" ]]; then
  write_status failed "candidate_eval_return_code=$candidate_eval_return_code" "$terminal_path"
  exit 78
fi

write_status evaluating_ablation same_disjoint_train_records "$status_path"
set +e
"$python_bin" scripts/run_causal_defect_adaptation.py \
  --config "$ablation_config" \
  --basis-checkpoint "$ablation_checkpoint" \
  --output-dir "$ablation_eval" \
  --device cuda \
  --adaptation-device cpu \
  --calibration-per-family "$calibration_per_family" \
  --calibration-seed "$calibration_seed" \
  --basis-training-per-family "$basis_training_per_family" \
  --no-fields \
  >"$output_dir/ablation_eval.log" 2>&1
ablation_eval_return_code=$?
set -e
if [[ "$ablation_eval_return_code" -ne 0 || ! -f "$ablation_eval/summary.json" ]]; then
  write_status failed "ablation_eval_return_code=$ablation_eval_return_code" "$terminal_path"
  exit 79
fi

write_status comparing_pair sealed_train_summaries "$status_path"
"$python_bin" scripts/compare_cpadc_trainonly_pair.py \
  --candidate-summary "$candidate_eval/summary.json" \
  --ablation-summary "$ablation_eval/summary.json" \
  --output "$output_dir/paired_gate.json" \
  >"$output_dir/paired_compare.log" 2>&1
paired_passed=$("$python_bin" -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1]))["passed"])).lower())' "$output_dir/paired_gate.json")
if [[ "$paired_passed" == "true" ]]; then
  write_status complete trainonly_pair_gate_passed "$terminal_path"
else
  write_status rejected trainonly_pair_gate_failed "$terminal_path"
fi
