#!/usr/bin/env bash
# Wait-only handoff: never signals r5c, then launches the preregistered r5d run once.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
r5c_artifact=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5c_shared_dynamic_adapter_train_diag
r5d_artifact=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5d_metric_aligned_time_train_diag
r5d_config=configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5d_metric_aligned_time_train_diag.yaml
preregistration=results/r5d_metric_aligned_time_train_diag_preregistration_20260813.json
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
terminal_path="$r5d_artifact/full_pipeline_terminal.json"
log_path="$r5d_artifact/four_gpu_train.log"

mkdir -p "$r5d_artifact"
printf '%s\n' "$$" >"$r5d_artifact/supervisor.pid"
exec 9>"$r5d_artifact/supervisor.lock"
if ! flock -n 9; then
  printf '%s\n' "another r5d supervisor owns the lock" >&2
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
    "schema": "r5d_after_r5c_natural_exit_supervisor_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "supervisor_pid": os.getppid(),
    "r5c_signal_or_process_control_used": False,
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

if [[ -e "$r5d_artifact/run/run_identity.json" ]]; then
  printf '%s\n' "r5d artifact already has a run identity; refusing overwrite" >&2
  exit 74
fi

"$python_bin" - "$r5d_config" "$preregistration" <<'PY'
import hashlib, json, pathlib, sys, yaml
config_path, preregistration_path = map(pathlib.Path, sys.argv[1:])
config = yaml.safe_load(config_path.read_text())
preregistered = json.loads(preregistration_path.read_text())
parent = pathlib.Path(config["parent_checkpoint"])
digest = hashlib.sha256(parent.read_bytes()).hexdigest()
expected = preregistered["parent"]
if str(parent) != expected["checkpoint"] or digest != expected["checkpoint_sha256"]:
    raise SystemExit("r5d parent checkpoint no longer matches preregistration")
if config.get("parent_checkpoint_selection_split") != "train":
    raise SystemExit("r5d parent selection is not train-only")
if config.get("time_policy") != "fixed_train_gate" or int(config.get("training_frames_per_record", 0)) != 24:
    raise SystemExit("r5d metric-aligned time intervention changed")
if config.get("epoch_validation_control", {}).get("evaluation_split") != "train":
    raise SystemExit("r5d epoch controller is not train-only")
PY

write_terminal waiting r5c_natural_exit 0
maximum_wait_checks=8640
wait_check=0
while true; do
  ready=$(
    "$python_bin" - "$r5c_artifact" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
terminal_path = root / "full_pipeline_terminal.json"
if not terminal_path.is_file():
    print("no")
    raise SystemExit
terminal = json.loads(terminal_path.read_text())
if terminal.get("status") not in {"complete", "failed"}:
    print("no")
    raise SystemExit
pid_paths = [root / "torchrun.pid"]
pid_paths.extend(root / "run" / name for name in ())
pids = []
for path in pid_paths:
    if path.is_file():
        text = path.read_text().strip()
        if text.isdigit():
            pids.append(int(text))
if any((pathlib.Path("/proc") / str(pid)).exists() for pid in pids):
    print("no")
    raise SystemExit
if terminal.get("status") == "failed":
    history = root / "run" / "epoch_validation_control.jsonl"
    if not history.is_file():
        print("invalid")
        raise SystemExit
    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    rejections = [row for row in rows if row.get("event") == "epoch_rejected"]
    if not rejections or rejections[-1].get("maximum_attempts_exhausted") is not True:
        print("invalid")
        raise SystemExit
print("yes")
PY
  )
  if [[ "$ready" == yes ]]; then
    break
  fi
  if [[ "$ready" == invalid ]]; then
    printf '%s\n' "r5c failed for a reason other than natural epoch-gate exhaustion" >&2
    exit 75
  fi
  if [[ "$wait_check" -ge "$maximum_wait_checks" ]]; then
    printf '%s\n' "bounded wait for r5c natural exit exhausted" >&2
    exit 76
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

write_terminal waiting exclusive_gpus_after_r5c 0
wait_check=0
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  if [[ "$wait_check" -ge 360 ]]; then
    printf '%s\n' "GPUs did not become exclusively free after r5c" >&2
    exit 77
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
write_terminal running four_gpu_train 0
set +e
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$r5d_config" --keep-all-checkpoints --workers-override 0 \
  >>"$log_path" 2>&1 &
training_pid=$!
printf '%s\n' "$training_pid" >"$r5d_artifact/torchrun.pid"
wait "$training_pid"
training_rc=$?
set -e

if [[ "$training_rc" -eq 0 ]]; then
  write_terminal complete complete 0
  trap - EXIT
  exit 0
fi
write_terminal failed training "$training_rc"
trap - EXIT
exit "$training_rc"
