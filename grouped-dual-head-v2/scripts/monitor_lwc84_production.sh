#!/usr/bin/env bash
set -u

worker_pid="${1:?usage: monitor_lwc84_production.sh WORKER_PID OUTPUT_ROOT}"
output_root="${2:?usage: monitor_lwc84_production.sh WORKER_PID OUTPUT_ROOT}"

while kill -0 "${worker_pid}" 2>/dev/null; do
  printf '[%s] worker_pid=%s alive=yes\n' "$(date --iso-8601=seconds)" "${worker_pid}"
  ps -o pid,ppid,sid,etime,%cpu,%mem,stat --no-headers -p "${worker_pid}" || true
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader || true
  find "${output_root}/shards" -type f \( -name '*.h5' -o -name '*.h5.tmp' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS size=%s path=%p\n' 2>/dev/null | sort | tail -n 10
  printf '\n'
  sleep 60
done

printf '[%s] worker_pid=%s alive=no monitor_stopped=yes\n' \
  "$(date --iso-8601=seconds)" "${worker_pid}"
