#!/usr/bin/env bash
# Run one sealed CPADC validation shard per GPU, then merge exact manifest coverage.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2.yaml
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_bilevel_trust_a3e1_marmousi1_4m_v2_r5
parallel_dir="$artifact_dir/evaluation_parallel"
checkpoint="$artifact_dir/run/best.pt"
checkpoint_sha256=4c01dea0dafd876106a8798e4962fba6cf25d3635a61c75429b5e7c32d1d09bc
python_bin=/root/miniconda3/bin/python
shard_count=4
stage=initializing
children=()

mkdir -p "$parallel_dir"
exec 9>"$artifact_dir/pipeline.lock"
if ! flock -n 9; then
  echo "another CPADC pipeline owns the lock" >&2
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

observed_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
if [[ "$observed_sha256" != "$checkpoint_sha256" ]]; then
  echo "CPADC basis checkpoint hash mismatch: $observed_sha256" >&2
  exit 74
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -ne "$shard_count" ]]; then
  echo "expected $shard_count GPUs, observed $gpu_count" >&2
  exit 75
fi
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$compute_pids" ]]; then
  echo "GPU compute processes appeared before parallel validation: $compute_pids" >&2
  exit 76
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
stage=sealed_full_validation_4way
write_terminal running 0

for shard_index in 0 1 2 3; do
  shard_dir="$parallel_dir/shard_$shard_index"
  mkdir -p "$shard_dir"
  CUDA_VISIBLE_DEVICES="$shard_index" "$python_bin" \
    scripts/run_causal_defect_adaptation.py \
    --config "$config" \
    --basis-checkpoint "$checkpoint" \
    --output-dir "$shard_dir" \
    --all-validation \
    --shard-index "$shard_index" \
    --shard-count "$shard_count" \
    --no-fields --device cuda \
    >"$parallel_dir/shard_${shard_index}.log" 2>&1 &
  children+=("$!")
done

"$python_bin" - "$parallel_dir/children.json" "${children[@]}" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "controller_pid": os.getppid(),
    "child_pids": [int(value) for value in sys.argv[2:]],
    "started_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

for pid in "${children[@]}"; do
  wait "$pid"
done
children=()

stage=merge_and_gate
write_terminal running 0
merge_args=()
for shard_index in 0 1 2 3; do
  merge_args+=(--shard-dir "$parallel_dir/shard_$shard_index")
done
"$python_bin" scripts/merge_causal_defect_evaluations.py \
  --config "$config" \
  --output-dir "$parallel_dir/merged" \
  "${merge_args[@]}" \
  >"$parallel_dir/merge.log" 2>&1

stage=complete
"$python_bin" - "$parallel_dir/merged/terminal.json" "$artifact_dir/pipeline_terminal.json" <<'PY'
import json, os, pathlib, sys, time
evaluation = json.loads(pathlib.Path(sys.argv[1]).read_text())
if evaluation.get("status") != "complete":
    raise ValueError("parallel sealed CPADC evaluation did not complete")
payload = {
    "status": "complete",
    "stage": "complete",
    "exit_code": 0,
    "pipeline_pid": os.getppid(),
    "updated_unix_s": time.time(),
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
