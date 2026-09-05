#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <validation|test_id> <output-dir>" >&2
  exit 64
fi
split_name="$1"
output_dir="$2"
if [[ "$split_name" != "validation" && "$split_name" != "test_id" ]]; then
  echo "unsupported split: $split_name" >&2
  exit 64
fi

project_root="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
training_config="$project_root/configs/baselines/pi_deeponet_lwc84_long100e_r7_authorized_20260814.yaml"
checkpoint="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/baselines/pi_deeponet_lwc84_long100e_r7/full_long100e_r1_resume1/best.pt"
if [[ "$split_name" == "test_id" ]]; then
  travel_time_h5="/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2_test_id.h5"
else
  travel_time_h5="/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5"
fi
mkdir -p "$output_dir"
cd "$project_root"

worker_pids=()
worker_outputs=()
for gpu_index in 0 1 2 3; do
  worker_output="$output_dir/worker_${gpu_index}.json"
  worker_log="$output_dir/worker_${gpu_index}.log"
  CUDA_VISIBLE_DEVICES="$gpu_index" python scripts/evaluate_frozen_pi_deeponet_split.py \
    --training-config "$training_config" \
    --checkpoint "$checkpoint" \
    --travel-time-h5 "$travel_time_h5" \
    --split "$split_name" \
    --shard-index "$gpu_index" \
    --num-shards 4 \
    --frames-per-record 32 \
    --device cuda:0 \
    --output "$worker_output" \
    >"$worker_log" 2>&1 &
  worker_pid=$!
  worker_pids+=("$worker_pid")
  worker_outputs+=("$worker_output")
  printf '%s\n' "$worker_pid" >"$output_dir/worker_${gpu_index}.pid"
done

worker_failure=0
for gpu_index in 0 1 2 3; do
  if wait "${worker_pids[$gpu_index]}"; then
    printf '0\n' >"$output_dir/worker_${gpu_index}.exitcode"
  else
    worker_status=$?
    printf '%s\n' "$worker_status" >"$output_dir/worker_${gpu_index}.exitcode"
    worker_failure=1
  fi
done
if [[ "$worker_failure" -ne 0 ]]; then
  python -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"schema":"frozen_pi_deeponet_split_terminal_v1","status":"worker_failed","split":sys.argv[2]},indent=2)+"\n")' "$output_dir/terminal.json" "$split_name"
  exit 1
fi

python scripts/aggregate_frozen_pi_deeponet_split.py \
  --workers "${worker_outputs[@]}" \
  --split "$split_name" \
  --expected-records 480 \
  --output "$output_dir/complete_${split_name}_comparison.json" \
  >"$output_dir/aggregate.log" 2>&1
python -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"schema":"frozen_pi_deeponet_split_terminal_v1","status":"complete","split":sys.argv[2],"aggregate_output":sys.argv[3]},indent=2)+"\n")' "$output_dir/terminal.json" "$split_name" "$output_dir/complete_${split_name}_comparison.json"
