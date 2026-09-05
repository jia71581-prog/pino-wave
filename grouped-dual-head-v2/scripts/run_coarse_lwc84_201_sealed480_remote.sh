#!/usr/bin/env bash
set -euo pipefail

code_root=${CODE_ROOT:-/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2}
source_h5=${SOURCE_H5:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5}
artifact_root=${ARTIFACT_ROOT:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/coarse_lwc84_201_sealed480}
v41_root=${V41_ARTIFACT_ROOT:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v41_4gpu_full_dense_lr1e4_batch192_curriculum_r1}

if [[ -f "$v41_root/resume.pid" ]]; then
  v41_pid=$(<"$v41_root/resume.pid")
  if [[ -n "$v41_pid" ]] && kill -0 "$v41_pid" 2>/dev/null; then
    echo "refusing to launch while V41 PID $v41_pid is alive" >&2
    exit 3
  fi
fi

timestamp=$(date +%Y%m%dT%H%M%S)
attempt="$artifact_root/attempt_$timestamp"
mkdir -p "$attempt/shards"
printf '%s\n' "$attempt" >"$artifact_root/attempt.path"
printf '%s\n' "$$" >"$artifact_root/supervisor.pid"

telemetry="$attempt/gpu_telemetry.csv"
telemetry_stop="$attempt/telemetry.stop"
printf 'timestamp,index,memory_used_mib,utilization_percent,power_w\n' >"$telemetry"
(
  while [[ ! -e "$telemetry_stop" ]]; do
    sample_time=$(date '+%F %T %Z')
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits \
      | while IFS= read -r gpu_row; do
          printf '%s,%s\n' "$sample_time" "$gpu_row" >>"$telemetry"
        done
    sleep 60
  done
) &
telemetry_pid=$!

cleanup_telemetry() {
  touch "$telemetry_stop"
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
}
trap cleanup_telemetry EXIT INT TERM

cd "$code_root"
declare -a pids
for shard in 0 1 2 3; do
  shard_name=$(printf 'shard_%02d' "$shard")
  CUDA_VISIBLE_DEVICES=$shard PYTHONUNBUFFERED=1 \
    /root/miniconda3/bin/python scripts/evaluate_coarse_lwc84_201_shard.py \
      --source-h5 "$source_h5" \
      --output-dir "$attempt/shards/$shard_name" \
      --shard-index "$shard" \
      --shard-count 4 \
      --solver-batch-size 120 \
      --device cuda \
      --internal-dt-s 0.00025 \
      --npml 20 \
      --c-ref-mps 6750 \
      --metric-block-size 20 \
      >"$attempt/shard_${shard}.log" 2>&1 &
  pids[$shard]=$!
  printf '%s\n' "${pids[$shard]}" >"$attempt/shard_${shard}.pid"
done

failed=0
for shard in 0 1 2 3; do
  if ! wait "${pids[$shard]}"; then
    echo "shard $shard failed" >&2
    failed=1
  fi
done

cleanup_telemetry
trap - EXIT INT TERM
if [[ "$failed" -ne 0 ]]; then
  exit 4
fi

/root/miniconda3/bin/python scripts/merge_coarse_lwc84_201_shards.py \
  --source-h5 "$source_h5" \
  --artifact-dir "$attempt" \
  --shard-count 4 \
  >"$attempt/merge.log" 2>&1

printf '%s\n' "$attempt" >"$artifact_root/completed.path"
echo "sealed 480x401 evaluation complete: $attempt"
