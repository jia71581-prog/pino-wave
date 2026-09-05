#!/usr/bin/env bash
# Wait-only handoff: never controls training/evaluation processes.  After the
# preregistered position pipeline exits naturally, run same-sample comparison
# and conditionally promote one-checkpoint paper evidence.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
position_root=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/evaluation/marmousi_fixed19hz_position_240_post_r5d_r1
comparison_root=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/evaluation/latest_vs_phase4b_fixed15_post_position_r1
state_root=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/evaluation/latest_model_paper_promotion_r1
python_bin=/root/miniconda3/bin/python
terminal_path="$state_root/pipeline_terminal.json"
log_path="$state_root/pipeline.log"

mkdir -p "$state_root" "$comparison_root"
printf '%s\n' "$$" >"$state_root/supervisor.pid"
exec 9>"$state_root/supervisor.lock"
if ! flock -n 9; then
  printf '%s\n' "another latest-model comparison supervisor owns the lock" >&2
  exit 73
fi
cd "$project_root"

write_terminal() {
  local status=$1
  local stage=$2
  local rc=$3
  "$python_bin" - "$terminal_path" "$status" "$stage" "$rc" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "latest_model_comparison_and_promotion_supervisor_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "supervisor_pid": os.getppid(),
    "training_or_evaluation_process_control_used": False,
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed supervisor "$rc" || true
  fi
}
trap on_exit EXIT

if [[ -e "$comparison_root/comparison_report.json" || -e "$comparison_root/prediction_manifest.json" ]]; then
  printf '%s\n' "comparison output already exists; refusing overwrite" >&2
  exit 74
fi

write_terminal waiting position_pipeline_natural_exit 0
wait_check=0
while true; do
  ready=$(
    "$python_bin" - "$position_root" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
terminal_path = root / "pipeline_terminal.json"
if not terminal_path.is_file():
    print("no")
    raise SystemExit
terminal = json.loads(terminal_path.read_text())
status = terminal.get("status")
if status == "complete":
    required = ("checkpoint_selection.json", "prediction_manifest.json", "reference_manifest.json", "score_summary.json")
    print("yes" if all((root / name).is_file() for name in required) else "invalid")
elif status == "failed":
    print("failed")
else:
    print("no")
PY
  )
  if [[ "$ready" == yes ]]; then break; fi
  if [[ "$ready" == failed || "$ready" == invalid ]]; then
    printf '%s\n' "position pipeline did not produce a complete valid result" >&2
    exit 75
  fi
  if [[ "$wait_check" -ge 8640 ]]; then
    printf '%s\n' "bounded wait for position pipeline exhausted" >&2
    exit 76
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

write_terminal waiting exclusive_idle_gpus 0
wait_check=0
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  if [[ "$wait_check" -ge 360 ]]; then
    printf '%s\n' "GPUs did not become exclusively free after position evaluation" >&2
    exit 77
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

available_bytes=$(df -B1 --output=avail "$comparison_root" | tail -1 | tr -d '[:space:]')
minimum_available_bytes=$((2 * 1024 * 1024 * 1024))
if [[ ! "$available_bytes" =~ ^[0-9]+$ ]] || [[ "$available_bytes" -lt "$minimum_available_bytes" ]]; then
  printf '%s\n' "less than 2 GiB is available for same-sample comparison" >&2
  exit 78
fi

eval "$($python_bin - "$position_root/checkpoint_selection.json" <<'PY'
import json, pathlib, shlex, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
for name in ("config", "checkpoint", "checkpoint_identity"):
    path = pathlib.Path(p[name])
    if not path.is_file():
        raise SystemExit(f"selected artifact missing: {path}")
    print(f"selected_{name}=" + shlex.quote(str(path.resolve())))
PY
)"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
write_terminal running same_sample_fixed15_comparison 0
"$python_bin" scripts/evaluate_latest_vs_phase4b_fixed15.py \
  --config "$selected_config" \
  --checkpoint "$selected_checkpoint" \
  --checkpoint-identity "$selected_checkpoint_identity" \
  --output-dir "$comparison_root" \
  --device cuda:0 --time-block 16 \
  >>"$log_path" 2>&1

write_terminal running conditional_evidence_promotion 0
CUDA_VISIBLE_DEVICES='' "$python_bin" scripts/promote_latest_model_evidence_bundle.py \
  --comparison-dir "$comparison_root" \
  --position-dir "$position_root" \
  --decision-output "$state_root/bundle_promotion_decision.json" \
  >>"$log_path" 2>&1

write_terminal complete complete 0
trap - EXIT
