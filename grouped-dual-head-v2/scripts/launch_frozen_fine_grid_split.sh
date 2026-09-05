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
source_h5="/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
manifest="/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/manifest.jsonl"
marmousi_npy="/root/autodl-tmp/home/jiayh/Data/data/marmousi_zenodo_16114161/marmousi1_vp_zx_751x2301_4m.npy"

mkdir -p "$output_dir"
cd "$project_root"

worker_pids=()
worker_outputs=()
for gpu_index in 0 1 2 3; do
  worker_output="$output_dir/worker_${gpu_index}.json"
  worker_log="$output_dir/worker_${gpu_index}.log"
  CUDA_VISIBLE_DEVICES="$gpu_index" python scripts/evaluate_frozen_fine_grid_split.py \
    --source-h5 "$source_h5" \
    --manifest "$manifest" \
    --marmousi-npy "$marmousi_npy" \
    --split "$split_name" \
    --shard-index "$gpu_index" \
    --num-shards 4 \
    --internal-dt-s 0.000625 \
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
  python -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"schema":"frozen_fine_grid_split_terminal_v1","status":"worker_failed","split":sys.argv[2]},indent=2)+"\n")' "$output_dir/terminal.json" "$split_name"
  exit 1
fi

if python scripts/aggregate_frozen_fine_grid_split.py \
    --workers "${worker_outputs[@]}" \
    --split "$split_name" \
    --expected-records 480 \
    --speed-reference-s 20.34137312322855 \
    --maximum-relative-l2 0.05 \
    --output "$output_dir/complete_${split_name}_gate.json" \
    >"$output_dir/aggregate.log" 2>&1; then
  aggregate_status=0
else
  aggregate_status=$?
fi
python -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"schema":"frozen_fine_grid_split_terminal_v1","status":"complete" if int(sys.argv[3])==0 else "gate_failed","split":sys.argv[2],"aggregate_output":sys.argv[4],"aggregate_exitcode":int(sys.argv[3])},indent=2)+"\n")' "$output_dir/terminal.json" "$split_name" "$aggregate_status" "$output_dir/complete_${split_name}_gate.json"
exit "$aggregate_status"
