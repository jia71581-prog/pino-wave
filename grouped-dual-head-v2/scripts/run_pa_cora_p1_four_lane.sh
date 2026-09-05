#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PREREGISTRATION" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PREREG=$1
SUPERVISOR=results/pa_cora_p1_multiwindow_supervisor_20260901
MANIFESTS=(
  results/b2_v9_parent_fit_manifest_240rec_20260901.json
  results/pa_cora_p1_fit_slot1.json
  results/pa_cora_p1_fit_slot2.json
  results/pa_cora_p1_fit_slot3.json
)
CACHES=(
  results/b2_v9_parent_fit_cache_240rec_20260901.h5
  results/pa_cora_p1_fit_slot1.h5
  results/pa_cora_p1_fit_slot2.h5
  results/pa_cora_p1_fit_slot3.h5
)
SEEDS=(372 733 1049 1403)

if [[ ! -f "$PREREG" || -e "$SUPERVISOR" ]]; then
  echo "missing preregistration or existing supervisor" >&2
  exit 1
fi
for path in "${MANIFESTS[@]}" "${CACHES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing PA-CORA P1 input: $path" >&2
    exit 1
  fi
done
for seed in "${SEEDS[@]}"; do
  output="results/pa_cora_p1_multiwindow_s${seed}_20260901"
  if [[ -e "$output" || -e "${output}.log" ]]; then
    echo "refusing to reuse output: $output" >&2
    exit 1
  fi
done

mkdir -p "$SUPERVISOR"
pids=()
for gpu in 0 1 2 3; do
  seed=${SEEDS[$gpu]}
  output="results/pa_cora_p1_multiwindow_s${seed}_20260901"
  env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_pa_cora_p1_multiwindow.py \
    --fit-manifest "${MANIFESTS[0]}" --fit-cache "${CACHES[0]}" \
    --fit-manifest "${MANIFESTS[1]}" --fit-cache "${CACHES[1]}" \
    --fit-manifest "${MANIFESTS[2]}" --fit-cache "${CACHES[2]}" \
    --fit-manifest "${MANIFESTS[3]}" --fit-cache "${CACHES[3]}" \
    --holdout-manifest results/b2_v9_parent_holdout_manifest_24rec_20260901.json \
    --holdout-cache results/b2_v9_parent_holdout_cache_24rec_20260901.h5 \
    --preregistration "$PREREG" \
    --output-dir "$output" \
    --seed "$seed" --epochs 23 --micro-records 4 \
    > "${output}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$SUPERVISOR/s${seed}.pid"
done

exit_codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  exit_codes+=("$?")
done
python -c 'import json,sys; values=[int(v) for v in sys.argv[1:5]]; json.dump({"status":"complete" if values==[0,0,0,0] else "failed","exit_codes":values},open(sys.argv[5],"w"),indent=2)' \
  "${exit_codes[@]}" "$SUPERVISOR/terminal.json"
for value in "${exit_codes[@]}"; do
  if [[ $value -ne 0 ]]; then
    exit 1
  fi
done
