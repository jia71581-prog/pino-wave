#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PREREGISTRATION" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1
PREREG=$1
SUPERVISOR=results/transfer_dg_flux_pretrain_supervisor_20260902
ARMS=(dg_flux dg_flux control control)
SEEDS=(372 733 372 733)
LABELS=(dg_flux_s372 dg_flux_s733 control_s372 control_s733)
CHECKPOINTS=(
  results/pa_cora_p1b_curriculum_s372_20260902/best.pt
  results/pa_cora_p1b_curriculum_s733_20260902/best.pt
  results/pa_cora_p1b_curriculum_s372_20260902/best.pt
  results/pa_cora_p1b_curriculum_s733_20260902/best.pt
)

if [[ ! -f "$PREREG" || -e "$SUPERVISOR" ]]; then
  echo "missing preregistration or existing supervisor" >&2
  exit 1
fi
for label in "${LABELS[@]}"; do
  output="results/transfer_dg_flux_pretrain_${label}_20260902"
  if [[ -e "$output" || -e "${output}.log" ]]; then
    echo "refusing to reuse output: $output" >&2
    exit 1
  fi
done

mkdir -p "$SUPERVISOR"
pids=()
for gpu in 0 1 2 3; do
  arm=${ARMS[$gpu]}
  seed=${SEEDS[$gpu]}
  label=${LABELS[$gpu]}
  checkpoint=${CHECKPOINTS[$gpu]}
  output="results/transfer_dg_flux_pretrain_${label}_20260902"
  env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_transfer_dg_flux_pretrain.py \
    --arm "$arm" --checkpoint "$checkpoint" \
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
    --preregistration "$PREREG" --output-dir "$output" \
    --seed "$seed" --epochs 8 --steps-per-epoch 120 \
    --learning-rate 0.00005 --micro-records 4 \
    > "${output}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$SUPERVISOR/${label}.pid"
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
python scripts/summarize_transfer_dg_flux_pretrain.py \
  --lane-terminal results/transfer_dg_flux_pretrain_dg_flux_s372_20260902/terminal.json \
  --lane-terminal results/transfer_dg_flux_pretrain_dg_flux_s733_20260902/terminal.json \
  --lane-terminal results/transfer_dg_flux_pretrain_control_s372_20260902/terminal.json \
  --lane-terminal results/transfer_dg_flux_pretrain_control_s733_20260902/terminal.json \
  --output "$SUPERVISOR/terminal.json"
