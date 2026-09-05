#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PREREG=results/b2_v4_aux_seed_preregistration_20260831.json
FIT_MANIFEST=results/b2_v4_fit_manifest_240rec_20260831.json
CAL_MANIFEST=results/b2_v4_calibration_manifest_60rec_20260831.json
FIT_CACHE=results/b2_v4_fit_cache_240rec_20260831.h5
CAL_CACHE=results/b2_v4_calibration_cache_60rec_20260831.h5
SUPERVISOR=results/b2_v4_group_disjoint_aux_supervisor_20260831
mkdir -p "$SUPERVISOR"

env CUDA_VISIBLE_DEVICES=2 python scripts/train_b2_group_disjoint.py \
  --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" \
  --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_group_disjoint_aux_s1049_20260831 \
  --seed 1049 --epochs 70 --micro-records 3 --maximum-epoch1-s 360 \
  > results/b2_v4_group_disjoint_aux_s1049_20260831.log 2>&1 &
pid_1049=$!

env CUDA_VISIBLE_DEVICES=3 python scripts/train_b2_group_disjoint.py \
  --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" \
  --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_group_disjoint_aux_s1403_20260831 \
  --seed 1403 --epochs 70 --micro-records 3 --maximum-epoch1-s 360 \
  > results/b2_v4_group_disjoint_aux_s1403_20260831.log 2>&1 &
pid_1403=$!

printf '%s\n' "$pid_1049" > "$SUPERVISOR/seed1049.pid"
printf '%s\n' "$pid_1403" > "$SUPERVISOR/seed1403.pid"
wait "$pid_1049"; rc_1049=$?
wait "$pid_1403"; rc_1403=$?
python -c 'import json,sys; r=[int(sys.argv[1]),int(sys.argv[2])]; json.dump({"status":"complete" if r==[0,0] else "failed","seed1049_exit_code":r[0],"seed1403_exit_code":r[1]},open(sys.argv[3],"w"),indent=2)' "$rc_1049" "$rc_1403" "$SUPERVISOR/terminal.json"
if [[ $rc_1049 -ne 0 || $rc_1403 -ne 0 ]]; then
  exit 1
fi
