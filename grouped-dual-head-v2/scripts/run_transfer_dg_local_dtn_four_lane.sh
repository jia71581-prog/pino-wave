#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PREREGISTRATION" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1
PREREG=$1
SUPERVISOR=results/transfer_dg_local_dtn_supervisor_20260902
SEEDS=(372 733 1049 1403)
if [[ ! -f "$PREREG" || -e "$SUPERVISOR" ]]; then
  echo "missing preregistration or existing supervisor" >&2
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  output="results/transfer_dg_local_dtn_s${seed}_20260902"
  if [[ -e "$output" || -e "${output}.log" ]]; then
    echo "refusing to reuse output: $output" >&2
    exit 1
  fi
done
mkdir -p "$SUPERVISOR"
pids=()
for gpu in 0 1 2 3; do
  seed=${SEEDS[$gpu]}
  output="results/transfer_dg_local_dtn_s${seed}_20260902"
  env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_transfer_dg_local_dtn.py \
    --dataset results/transfer_dg_local_frequency_elements_24rec_20260902.h5 \
    --preregistration "$PREREG" --output-dir "$output" \
    --seed "$seed" --epochs 120 --batch-size 256 --learning-rate 0.001 \
    > "${output}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$SUPERVISOR/s${seed}.pid"
done
codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  codes+=("$?")
done
python -c 'import json,sys; values=[int(v) for v in sys.argv[1:5]]; json.dump({"status":"lanes_complete" if values==[0,0,0,0] else "failed","exit_codes":values},open(sys.argv[5],"w"),indent=2)' \
  "${codes[@]}" "$SUPERVISOR/lane_exit_codes.json"
for value in "${codes[@]}"; do
  if [[ $value -ne 0 ]]; then
    exit 1
  fi
done
python scripts/summarize_transfer_dg_local_dtn.py \
  --terminal results/transfer_dg_local_dtn_s372_20260902/terminal.json \
  --terminal results/transfer_dg_local_dtn_s733_20260902/terminal.json \
  --terminal results/transfer_dg_local_dtn_s1049_20260902/terminal.json \
  --terminal results/transfer_dg_local_dtn_s1403_20260902/terminal.json \
  --output "$SUPERVISOR/terminal.json"
