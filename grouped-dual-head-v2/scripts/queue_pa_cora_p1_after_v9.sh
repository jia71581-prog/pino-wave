#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PIPELINE=results/pa_cora_p1_pipeline_20260901
V9_TERMINAL=results/b2_v9_parent_full_supervisor_20260901/terminal.json
V9_SELECTION=results/b2_v9_parent_selection_20260901.json
P1_PREREG=results/pa_cora_p1_training_preregistration_20260901.json
CACHE_PREREG=results/pa_cora_p1_cache_preregistration_20260901.json
mkdir -p "$PIPELINE"

write_terminal() {
  local status=$1
  local message=$2
  python -c 'import json,sys; json.dump({"status":sys.argv[1],"message":sys.argv[2]},open(sys.argv[3],"w"),indent=2)' \
    "$status" "$message" "$PIPELINE/terminal.json"
}

printf '%s\n' "waiting_for_v9_terminal"
while [[ ! -f "$V9_TERMINAL" ]]; do
  sleep 30
done
if ! python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status")=="complete" and p.get("exit_codes")==[0,0,0,0]' "$V9_TERMINAL"; then
  write_terminal failed "V9 supervisor did not complete all lanes"
  exit 1
fi

if [[ ! -f "$V9_SELECTION" ]]; then
  if ! python scripts/select_b2_v9_parent.py --output "$V9_SELECTION"; then
    write_terminal rejected "V9 parent selection failed"
    exit 1
  fi
fi
if ! python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status")=="passed" and p["selected"]["variant"]=="spectral"' "$V9_SELECTION"; then
  write_terminal blocked "V9 spectral variant was not the frozen winner"
  exit 1
fi

printf '%s\n' "building_three_multiwindow_caches"
pids=()
for slot in 1 2 3; do
  cache="results/pa_cora_p1_fit_slot${slot}.h5"
  output="results/pa_cora_p1_fit_slot${slot}_cache_build_20260901"
  log="results/pa_cora_p1_fit_slot${slot}_cache_build_20260901.log"
  if [[ -e "$cache" || -e "$output" || -e "$log" ]]; then
    write_terminal failed "preexisting PA-CORA P1 cache artifact for slot ${slot}"
    exit 1
  fi
  env CUDA_VISIBLE_DEVICES=$((slot - 1)) python scripts/build_b2_v5_cache.py \
    --manifest "results/pa_cora_p1_fit_slot${slot}.json" \
    --cache "$cache" \
    --preregistration "$CACHE_PREREG" \
    --output-dir "$output" > "$log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$PIPELINE/cache_slot${slot}.pid"
done

cache_exit_codes=()
for pid in "${pids[@]}"; do
  wait "$pid"
  cache_exit_codes+=("$?")
done
for value in "${cache_exit_codes[@]}"; do
  if [[ $value -ne 0 ]]; then
    write_terminal failed "one or more PA-CORA cache builders failed"
    exit 1
  fi
done

for slot in 1 2 3; do
  python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status")=="complete"' \
    "results/pa_cora_p1_fit_slot${slot}_cache_build_20260901/terminal.json" || {
      write_terminal failed "cache terminal audit failed"
      exit 1
    }
done

if ! python scripts/prepare_pa_cora_p1_prereg.py \
  --v9-selection "$V9_SELECTION" --output "$P1_PREREG"; then
  write_terminal failed "P1 preregistration binding failed"
  exit 1
fi

printf '%s\n' "launching_four_pa_cora_p1_lanes"
if ! bash scripts/run_pa_cora_p1_four_lane.sh "$P1_PREREG"; then
  write_terminal failed "PA-CORA P1 four-lane runner failed"
  exit 1
fi

write_terminal complete "PA-CORA P1 cache and four-lane stages completed"
