#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PREREG=results/b2_v4_group_disjoint_preregistration_20260831.json
FIT_MANIFEST=results/b2_v4_fit_manifest_240rec_20260831.json
CAL_MANIFEST=results/b2_v4_calibration_manifest_60rec_20260831.json
FIT_CACHE=results/b2_v4_fit_cache_240rec_20260831.h5
CAL_CACHE=results/b2_v4_calibration_cache_60rec_20260831.h5
SUPERVISOR=results/b2_v4_group_disjoint_supervisor_20260831
mkdir -p "$SUPERVISOR"

env CUDA_VISIBLE_DEVICES=0 python scripts/build_b2_bound_cache.py \
  --manifest "$FIT_MANIFEST" --cache "$FIT_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_fit_cache_build_20260831 \
  > results/b2_v4_fit_cache_build_20260831.log 2>&1
fit_cache_rc=$?
if [[ $fit_cache_rc -ne 0 ]]; then
  python -c 'import json,sys; json.dump({"status":"failed","stage":"fit_cache","exit_code":int(sys.argv[1])},open(sys.argv[2],"w"),indent=2)' "$fit_cache_rc" "$SUPERVISOR/terminal.json"
  exit "$fit_cache_rc"
fi

env CUDA_VISIBLE_DEVICES=0 python scripts/build_b2_bound_cache.py \
  --manifest "$CAL_MANIFEST" --cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_calibration_cache_build_20260831 \
  > results/b2_v4_calibration_cache_build_20260831.log 2>&1
cal_cache_rc=$?
if [[ $cal_cache_rc -ne 0 ]]; then
  python -c 'import json,sys; json.dump({"status":"failed","stage":"calibration_cache","exit_code":int(sys.argv[1])},open(sys.argv[2],"w"),indent=2)' "$cal_cache_rc" "$SUPERVISOR/terminal.json"
  exit "$cal_cache_rc"
fi

env CUDA_VISIBLE_DEVICES=0 python scripts/train_b2_group_disjoint.py \
  --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" \
  --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_group_disjoint_s372_20260831 \
  --seed 372 --epochs 70 --micro-records 3 --maximum-epoch1-s 360 \
  > results/b2_v4_group_disjoint_s372_20260831.log 2>&1 &
pid_372=$!

env CUDA_VISIBLE_DEVICES=1 python scripts/train_b2_group_disjoint.py \
  --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" \
  --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_group_disjoint_s733_20260831 \
  --seed 733 --epochs 70 --micro-records 3 --maximum-epoch1-s 360 \
  > results/b2_v4_group_disjoint_s733_20260831.log 2>&1 &
pid_733=$!

printf '%s\n' "$pid_372" > "$SUPERVISOR/seed372.pid"
printf '%s\n' "$pid_733" > "$SUPERVISOR/seed733.pid"
wait "$pid_372"; rc_372=$?
wait "$pid_733"; rc_733=$?
python -c 'import json,sys; r=[int(sys.argv[1]),int(sys.argv[2])]; json.dump({"status":"complete" if r==[0,0] else "failed","seed372_exit_code":r[0],"seed733_exit_code":r[1]},open(sys.argv[3],"w"),indent=2)' "$rc_372" "$rc_733" "$SUPERVISOR/terminal.json"
if [[ $rc_372 -ne 0 || $rc_733 -ne 0 ]]; then
  exit 1
fi
