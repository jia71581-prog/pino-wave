#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PREREG=results/b2_v5_four_arm_pilot_preregistration.json
FIT_MANIFEST=results/b2_v5_pilot_fit_manifest.json
CAL_MANIFEST=results/b2_v5_pilot_calibration_manifest.json
FIT_CACHE=results/b2_v5_pilot_fit_cache.h5
CAL_CACHE=results/b2_v5_pilot_calibration_cache.h5
SUPERVISOR=results/b2_v5_four_arm_pilot_supervisor
mkdir -p "$SUPERVISOR"

arms=(control physics_cond nonworse_hinge spectral)
pids=()
for gpu in 0 1 2 3; do
  arm=${arms[$gpu]}
  env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_b2_v5_pilot.py \
    --arm "$arm" --fit-manifest "$FIT_MANIFEST" \
    --calibration-manifest "$CAL_MANIFEST" --fit-cache "$FIT_CACHE" \
    --calibration-cache "$CAL_CACHE" --preregistration "$PREREG" \
    --output-dir "results/b2_v5_pilot_${arm}" --seed 372 --epochs 10 \
    --micro-records 3 --nonworse-weight 0.3 --spectral-weight 0.1 \
    --spectral-energy-floor-fraction 0.005 --maximum-epoch1-s 240 \
    > "results/b2_v5_pilot_${arm}.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "$!" > "$SUPERVISOR/${arm}.pid"
done

exit_codes=()
for pid in "${pids[@]}"; do
  wait "$pid"; exit_codes+=("$?")
done
python -c 'import json,sys; values=[int(x) for x in sys.argv[1:5]]; json.dump({"status":"complete" if values==[0,0,0,0] else "failed","exit_codes":values},open(sys.argv[5],"w"),indent=2)' \
  "${exit_codes[@]}" "$SUPERVISOR/terminal.json"
for value in "${exit_codes[@]}"; do
  if [[ $value -ne 0 ]]; then exit 1; fi
done
