#!/usr/bin/env bash
set -uo pipefail
ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2; cd "$ROOT" || exit 1
PREREG=results/b2_v8_parent_pilot_preregistration_20260901.json
FIT_MANIFEST=results/b2_v6_offline_fit_manifest_24rec_20260901.json
CAL_MANIFEST=results/b2_v8_parent_holdout_manifest_12rec_20260901.json
FIT_CACHE=results/b2_v8_parent_fit_cache_24rec_20260901.h5
CAL_CACHE=results/b2_v8_parent_holdout_cache_12rec_20260901.h5
SUP=results/b2_v8_parent_four_arm_supervisor_20260901; mkdir -p "$SUP"
arms=(control physics_cond nonworse_hinge spectral); pids=()
for gpu in 0 1 2 3; do
 arm=${arms[$gpu]}; env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_b2_v5_pilot.py --arm "$arm" --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" --preregistration "$PREREG" --output-dir "results/b2_v8_parent_${arm}_20260901" --seed 372 --epochs 10 --micro-records 3 --nonworse-weight 0.3 --spectral-weight 0.1 --spectral-energy-floor-fraction 0.005 --maximum-epoch1-s 240 > "results/b2_v8_parent_${arm}_20260901.log" 2>&1 &
 pids+=("$!"); printf '%s\n' "$!" > "$SUP/${arm}.pid"
done
rc=(); for pid in "${pids[@]}"; do wait "$pid"; rc+=("$?"); done
python -c 'import json,sys;r=[int(x) for x in sys.argv[1:5]];json.dump({"status":"complete" if r==[0,0,0,0] else "failed","exit_codes":r},open(sys.argv[5],"w"),indent=2)' "${rc[@]}" "$SUP/terminal.json"
for value in "${rc[@]}"; do if [[ $value -ne 0 ]]; then exit 1; fi; done
