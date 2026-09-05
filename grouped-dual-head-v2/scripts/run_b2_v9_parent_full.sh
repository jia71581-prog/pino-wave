#!/usr/bin/env bash
set -uo pipefail
ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2; cd "$ROOT" || exit 1
P=results/b2_v9_parent_full_preregistration_20260901.json; FM=results/b2_v9_parent_fit_manifest_240rec_20260901.json; CM=results/b2_v9_parent_holdout_manifest_24rec_20260901.json; FC=results/b2_v9_parent_fit_cache_240rec_20260901.h5; CC=results/b2_v9_parent_holdout_cache_24rec_20260901.h5; SUP=results/b2_v9_parent_full_supervisor_20260901; mkdir -p "$SUP"
arms=(physics_cond physics_cond spectral spectral); seeds=(372 733 372 733); pids=()
for gpu in 0 1 2 3; do
 arm=${arms[$gpu]}; seed=${seeds[$gpu]}; out="results/b2_v9_parent_${arm}_s${seed}_20260901"
 env CUDA_VISIBLE_DEVICES=$gpu python scripts/train_b2_v5_pilot.py --arm "$arm" --fit-manifest "$FM" --calibration-manifest "$CM" --fit-cache "$FC" --calibration-cache "$CC" --preregistration "$P" --output-dir "$out" --seed "$seed" --epochs 90 --micro-records 4 --nonworse-weight 0.3 --spectral-weight 0.1 --spectral-energy-floor-fraction 0.005 --maximum-epoch1-s 300 > "${out}.log" 2>&1 &
 pids+=("$!"); printf '%s\n' "$!" > "$SUP/${arm}_s${seed}.pid"
done
rc=();for pid in "${pids[@]}";do wait "$pid";rc+=("$?");done
python -c 'import json,sys;r=[int(x) for x in sys.argv[1:5]];json.dump({"status":"complete" if r==[0,0,0,0] else "failed","exit_codes":r},open(sys.argv[5],"w"),indent=2)' "${rc[@]}" "$SUP/terminal.json"
for value in "${rc[@]}";do if [[ $value -ne 0 ]];then exit 1;fi;done
