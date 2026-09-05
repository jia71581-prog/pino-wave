#!/usr/bin/env bash
# One-shot four-GPU diagnostic continuation.  No restart and no process signals.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag.yaml
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
terminal_path="$artifact_dir/full_pipeline_terminal.json"
log_path="$artifact_dir/four_gpu_train.log"

mkdir -p "$artifact_dir"
printf '%s\n' "$$" >"$artifact_dir/supervisor.pid"
exec 9>"$artifact_dir/supervisor.lock"
if ! flock -n 9; then
  printf '%s\n' "another r5b supervisor owns the lock" >&2
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
    "schema": "r5b_train_gate_record_l2_supervisor_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "supervisor_pid": os.getppid(),
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

if [[ -e "$artifact_dir/run/run_identity.json" ]]; then
  printf '%s\n' "r5b artifact already has a run identity; refusing overwrite" >&2
  exit 74
fi
maximum_wait_checks=720
wait_check=0
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  if [[ "$wait_check" -eq 0 ]]; then
    write_terminal waiting waiting_for_exclusive_gpus 0
  fi
  if [[ "$wait_check" -ge "$maximum_wait_checks" ]]; then
    printf '%s\n' "exclusive GPU wait limit exhausted" >&2
    exit 75
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
  --config "$config" --keep-all-checkpoints --workers-override 0 \
  >>"$log_path" 2>&1 &
training_pid=$!
printf '%s\n' "$training_pid" >"$artifact_dir/torchrun.pid"
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
