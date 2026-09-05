#!/usr/bin/env bash
set -euo pipefail

root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
script=scripts/eval_coarse_lwc84_dispersion_validation_streaming_v23b.py
out=results/r16_dscp_v23b_coarse_lwc84_dispersion_validation
cd "$root"

if pgrep -af "python.*eval_coarse_lwc84_dispersion_validation_streaming_v23b.py" >/dev/null; then
  echo "v23b worker already running" >&2
  exit 2
fi
if find "$out" -maxdepth 1 -name 'worker_*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "v23b worker result already exists; one-shot relaunch refused" >&2
  exit 3
fi

for worker in 0 1 2 3; do
  log="results/r16_dscp_v23b_worker_${worker}.log"
  pid_file="results/r16_dscp_v23b_worker_${worker}.pid"
  env CUDA_VISIBLE_DEVICES="$worker" WORKER="$worker" \
    nohup /root/miniconda3/bin/python "$script" \
    >"$log" 2>&1 </dev/null &
  echo "$!" >"$pid_file"
done

echo "launched"
for worker in 0 1 2 3; do
  printf 'worker_%s pid=' "$worker"
  sed -n '1p' "results/r16_dscp_v23b_worker_${worker}.pid"
done
