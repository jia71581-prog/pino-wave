#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PREREGISTRATION" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1
PREREG=$1
SUPERVISOR=results/pa_cora_p1b_curriculum_supervisor_20260902
SEEDS=(372 733 1049 1403)
CHECKPOINTS=(
  results/b2_v9_parent_spectral_s372_20260901/best.pt
  results/b2_v9_parent_spectral_s733_20260901/best.pt
  results/b2_v9_parent_spectral_s372_20260901/best.pt
  results/b2_v9_parent_spectral_s733_20260901/best.pt
)

if [[ ! -f "$PREREG" || -e "$SUPERVISOR" ]]; then
  echo "missing preregistration or existing supervisor" >&2
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  output="results/pa_cora_p1b_curriculum_s${seed}_20260902"
  if [[ -e "$output" || -e "${output}.log" ]]; then
    echo "refusing to reuse output: $output" >&2
    exit 1
  fi
done

mkdir -p "$SUPERVISOR"
pids=()
for gpu in 0 1 2 3; do
  seed=${SEEDS[$gpu]}
  checkpoint=${CHECKPOINTS[$gpu]}
  output="results/pa_cora_p1b_curriculum_s${seed}_20260902"
  env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_pa_cora_p1b_curriculum.py \
    --checkpoint "$checkpoint" \
    --fit-manifest results/b2_v9_parent_fit_manifest_240rec_20260901.json \
    --fit-cache results/b2_v9_parent_fit_cache_240rec_20260901.h5 \
    --fit-manifest results/pa_cora_p1_fit_slot1.json \
    --fit-cache results/pa_cora_p1_fit_slot1.h5 \
    --fit-manifest results/pa_cora_p1_fit_slot2.json \
    --fit-cache results/pa_cora_p1_fit_slot2.h5 \
    --holdout-manifest results/b2_v9_parent_holdout_manifest_24rec_20260901.json \
    --holdout-cache results/b2_v9_parent_holdout_cache_24rec_20260901.h5 \
    --holdout-manifest results/pa_cora_p1_holdout_slot1.json \
    --holdout-cache results/pa_cora_p1_holdout_slot1.h5 \
    --holdout-manifest results/pa_cora_p1_holdout_slot2.json \
    --holdout-cache results/pa_cora_p1_holdout_slot2.h5 \
    --preregistration "$PREREG" \
    --output-dir "$output" \
    --seed "$seed" --epochs 10 --learning-rate 0.0001 --micro-records 4 \
    > "${output}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$SUPERVISOR/s${seed}.pid"
done

codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  codes+=("$?")
done
python -c 'import json,sys; values=[int(v) for v in sys.argv[1:5]]; json.dump({"status":"complete" if values==[0,0,0,0] else "failed","exit_codes":values},open(sys.argv[5],"w"),indent=2)' \
  "${codes[@]}" "$SUPERVISOR/terminal.json"
for value in "${codes[@]}"; do
  if [[ $value -ne 0 ]]; then
    exit 1
  fi
done
