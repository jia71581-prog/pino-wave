#!/usr/bin/env bash
# Continue the accepted A3 optimum on four GPUs with phase-adapted AdamW.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r3_adamw99.yaml
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r3_adamw99
manifest=/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5
travel=/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
terminal_path="$artifact_dir/full_pipeline_terminal.json"
resume_log="$artifact_dir/four_gpu_adamw99_resume.log"

mkdir -p "$artifact_dir"
printf '%s\n' "$$" >"$artifact_dir/four_gpu_adamw99_supervisor.pid"
exec 9>"$artifact_dir/four_gpu_adamw99.lock"
if ! flock -n 9; then
  echo "another adamw99 continuation supervisor owns the lock" >&2
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
    "status": sys.argv[2], "stage": sys.argv[3], "exit_code": int(sys.argv[4]),
    "pipeline_pid": os.getppid(), "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed adamw99_recovery "$rc" || true
  fi
}
trap on_exit EXIT

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
external_attempt=0
maximum_external_attempts=4

while true; do
  if "$python_bin" - "$artifact_dir/run/terminal.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and json.loads(path.read_text()).get("status") == "complete" else 1)
PY
  then
    break
  fi
  if [[ "$external_attempt" -ge "$maximum_external_attempts" ]]; then
    echo "external infrastructure recovery attempts exhausted" >&2
    exit 75
  fi
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
    write_terminal running waiting_for_exclusive_gpus 0
    sleep 10
  done
  write_terminal running dataset_cache_eviction 0
  "$python_bin" scripts/evict_saved_time_dataset_cache.py \
    --manifest "$manifest" --extra-file "$travel" >>"$resume_log" 2>&1
  external_attempt=$((external_attempt + 1))
  write_terminal running "four_gpu_attempt_${external_attempt}" 0
  set +e
  "$torchrun_bin" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$config" --keep-all-checkpoints --workers-override 0 \
    >>"$resume_log" 2>&1 &
  training_pid=$!
  "$python_bin" scripts/steward_saved_time_dataset_cache.py \
    --manifest "$manifest" --extra-file "$travel" \
    --control-history "$artifact_dir/run/epoch_validation_control.jsonl" \
    --training-pid "$training_pid" \
    --output "$artifact_dir/run/dataset_cache_evictions.jsonl" \
    >>"$resume_log" 2>&1 &
  steward_pid=$!
  wait "$training_pid"
  training_rc=$?
  wait "$steward_pid"
  set -e
  "$python_bin" - "$resume_log" "$external_attempt" "$training_rc" <<'PY'
import json, pathlib, sys, time
with pathlib.Path(sys.argv[1]).open("a", encoding="utf8") as handle:
    handle.write(json.dumps({
        "event": "external_training_exit", "attempt": int(sys.argv[2]),
        "exit_code": int(sys.argv[3]), "time": time.time(),
    }, sort_keys=True) + "\n")
PY
  if [[ "$training_rc" -ne 0 ]] && "$python_bin" - "$artifact_dir/run/terminal.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text()) if path.is_file() else {}
is_plateau = (
    payload.get("status") == "failed"
    and "validation did not improve" in str(payload.get("error", ""))
)
raise SystemExit(0 if is_plateau else 1)
PY
  then
    echo "same-protocol validation plateau requires a new audited parameter branch" >&2
    exit 76
  fi
  sleep 5
done

write_terminal complete complete 0
trap - EXIT
