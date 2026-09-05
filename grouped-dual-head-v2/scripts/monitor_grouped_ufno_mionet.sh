#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID="${1:?usage: monitor_grouped_ufno_mionet.sh PID [LOG]}"
LOG="${2:-$ROOT/artifacts/grouped_ufno_mionet/production/logs/monitor.log}"
TRAIN_LOG="${3:-$ROOT/artifacts/grouped_ufno_mionet/production/logs/train.nohup.log}"
mkdir -p "$(dirname "$LOG")"
while kill -0 "$PID" 2>/dev/null; do
  {
    date --iso-8601=seconds
    ps -p "$PID" -o pid,stat,etime,%cpu,%mem --no-headers || true
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true
    tail -n 1 "$TRAIN_LOG" 2>/dev/null || true
    echo
  } >> "$LOG"
  sleep 30
done
echo "$(date --iso-8601=seconds) process=$PID stopped" >> "$LOG"
