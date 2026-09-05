#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

P1_TERMINAL=results/pa_cora_p1_pipeline_20260901/terminal.json
OUTPUT_DIR=results/pa_cora_supplement_pipeline_20260901
CACHE_PREREG=results/pa_cora_p1_holdout_cache_preregistration_20260901.json
mkdir -p "$OUTPUT_DIR"

write_terminal() {
  python -c 'import json,sys; json.dump({"status":sys.argv[1],"message":sys.argv[2]},open(sys.argv[3],"w"),indent=2)' \
    "$1" "$2" "$OUTPUT_DIR/terminal.json"
}

printf '%s\n' "waiting_for_pa_cora_p1_terminal"
while [[ ! -f "$P1_TERMINAL" ]]; do
  sleep 30
done

printf '%s\n' "building_three_multiphase_holdout_caches"
pids=()
for slot in 1 2 3; do
  cache="results/pa_cora_p1_holdout_slot${slot}.h5"
  build="results/pa_cora_p1_holdout_slot${slot}_cache_build_20260901"
  log="results/pa_cora_p1_holdout_slot${slot}_cache_build_20260901.log"
  if [[ -e "$cache" || -e "$build" || -e "$log" ]]; then
    write_terminal failed "preexisting holdout cache artifact for slot ${slot}"
    exit 1
  fi
  env CUDA_VISIBLE_DEVICES=$((slot - 1)) python scripts/build_b2_v5_cache.py \
    --manifest "results/pa_cora_p1_holdout_slot${slot}.json" \
    --cache "$cache" \
    --preregistration "$CACHE_PREREG" \
    --output-dir "$build" > "$log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$OUTPUT_DIR/holdout_slot${slot}.pid"
done

codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  codes+=("$?")
done
for value in "${codes[@]}"; do
  if [[ $value -ne 0 ]]; then
    write_terminal failed "one or more holdout cache builders failed"
    exit 1
  fi
done

printf '%s\n' "running_final_v9_multiphase_residual_diagnostic"
if ! env CUDA_VISIBLE_DEVICES=0 python scripts/analyze_v9_multiphase_residual_structure.py \
  --checkpoint results/b2_v9_parent_spectral_s372_20260901/best.pt \
  --manifest results/b2_v9_parent_holdout_manifest_24rec_20260901.json \
  --cache results/b2_v9_parent_holdout_cache_24rec_20260901.h5 \
  --manifest results/pa_cora_p1_holdout_slot1.json \
  --cache results/pa_cora_p1_holdout_slot1.h5 \
  --manifest results/pa_cora_p1_holdout_slot2.json \
  --cache results/pa_cora_p1_holdout_slot2.h5 \
  --manifest results/pa_cora_p1_holdout_slot3.json \
  --cache results/pa_cora_p1_holdout_slot3.h5 \
  --output results/v9_multiphase_residual_structure_20260901.json \
  > results/v9_multiphase_residual_structure_20260901.log 2>&1; then
  write_terminal failed "final V9 residual diagnostic failed"
  exit 1
fi

write_terminal complete "multiphase holdout caches and final V9 residual diagnostic completed"
