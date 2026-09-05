#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1

PREREG=results/b2_v4_ddp_restart_preregistration_20260831.json
FIT_MANIFEST=results/b2_v4_fit_manifest_240rec_20260831.json
CAL_MANIFEST=results/b2_v4_calibration_manifest_60rec_20260831.json
FIT_CACHE=results/b2_v4_fit_cache_240rec_20260831.h5
CAL_CACHE=results/b2_v4_calibration_cache_60rec_20260831.h5
SUPERVISOR=results/b2_v4_group_disjoint_ddp_supervisor_20260831
mkdir -p "$SUPERVISOR"

env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --master-port=29672 \
  --nproc-per-node=2 scripts/train_b2_group_disjoint_ddp.py \
  --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" \
  --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_group_disjoint_ddp_s372_20260831 \
  --seed 372 --epochs 70 --global-micro-records 3 \
  --calibration-micro-records 3 --maximum-epoch1-s 240 \
  > results/b2_v4_group_disjoint_ddp_s372_20260831.log 2>&1 &
pid_372=$!

env CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --master-port=29733 \
  --nproc-per-node=2 scripts/train_b2_group_disjoint_ddp.py \
  --fit-manifest "$FIT_MANIFEST" --calibration-manifest "$CAL_MANIFEST" \
  --fit-cache "$FIT_CACHE" --calibration-cache "$CAL_CACHE" \
  --preregistration "$PREREG" \
  --output-dir results/b2_v4_group_disjoint_ddp_s733_20260831 \
  --seed 733 --epochs 70 --global-micro-records 3 \
  --calibration-micro-records 3 --maximum-epoch1-s 240 \
  > results/b2_v4_group_disjoint_ddp_s733_20260831.log 2>&1 &
pid_733=$!

printf '%s\n' "$pid_372" > "$SUPERVISOR/seed372_torchrun.pid"
printf '%s\n' "$pid_733" > "$SUPERVISOR/seed733_torchrun.pid"
wait "$pid_372"; rc_372=$?
wait "$pid_733"; rc_733=$?
python -c 'import json,sys; r=[int(sys.argv[1]),int(sys.argv[2])]; json.dump({"status":"complete" if r==[0,0] else "failed","seed372_exit_code":r[0],"seed733_exit_code":r[1]},open(sys.argv[3],"w"),indent=2)' "$rc_372" "$rc_733" "$SUPERVISOR/terminal.json"
if [[ $rc_372 -ne 0 || $rc_733 -ne 0 ]]; then
  exit 1
fi
