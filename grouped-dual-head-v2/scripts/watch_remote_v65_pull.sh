#!/usr/bin/env bash
set -u

SSH=(
  ssh
  -i /home/jiayh/.ssh/id_ed25519_seetacloud_codex
  -p 20586
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o StrictHostKeyChecking=no
)
RSYNC_SSH='ssh -i /home/jiayh/.ssh/id_ed25519_seetacloud_codex -p 20586 -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no'
HOST=root@connect.cqa1.seetacloud.com
PYTHON=/home/jiayh/miniconda3/bin/python
WORK=/home/jiayh/.config/superpowers/worktrees/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
REMOTE_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
REMOTE_RUN=$REMOTE_PROJECT/1/pretraining/saved_time_v65_lwc84_multifidelity_r2
REMOTE_TEACHER=$REMOTE_PROJECT/1/pretraining/lwc84_multifidelity_teacher64_401solver_v2
LOCAL_PROJECT=/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
LOCAL_PRETRAINING=$LOCAL_PROJECT/1/pretraining
LOCAL_TEACHER=$LOCAL_PRETRAINING/lwc84_multifidelity_teacher64_401solver_v2
LOG=$LOCAL_PROJECT/1/logs/watch_remote_v65_pull.log
PID_FILE=$LOCAL_PROJECT/1/logs/watch_remote_v65_pull.pid
PARENT_REPORT=$LOCAL_PRETRAINING/saved_time_v65_lwc84_multifidelity_r2/parent_same_panel_seed401.json
EPOCH_SUMMARY=$LOCAL_PRETRAINING/saved_time_v65_lwc84_multifidelity_r2/monitor/epoch_summary.json

mkdir -p "$LOCAL_PRETRAINING" "$LOCAL_TEACHER" "$(dirname "$LOG")"
printf '%s\n' "$$" >"$PID_FILE"
printf '%s watcher start pid=%s\n' "$(date --iso-8601=seconds)" "$$" >>"$LOG"

cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

sync_outputs() {
  rsync -az --partial --timeout=120 -e "$RSYNC_SSH" \
    "$HOST:$REMOTE_RUN" "$LOCAL_PRETRAINING/" >>"$LOG" 2>&1 || true
  rsync -az --partial --timeout=120 -e "$RSYNC_SSH" \
    --include='/teacher_vds.h5' \
    --include='/teacher_vds.summary.json' \
    --include='/logs/' \
    --include='/logs/***' \
    --exclude='*' \
    "$HOST:$REMOTE_TEACHER/" "$LOCAL_TEACHER/" >>"$LOG" 2>&1 || true
}

summarize_epochs() {
  PYTHONPATH="$WORK${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m saved_time_phase_operator_v4.epoch_monitor \
    --candidate-metrics "$LOCAL_PRETRAINING/saved_time_v65_lwc84_multifidelity_r2/pilot/metrics.jsonl" \
    --parent-report "$PARENT_REPORT" \
    --output "$EPOCH_SUMMARY" >/dev/null 2>>"$LOG" || true
}

while true; do
  sync_outputs
  summarize_epochs
  if "${SSH[@]}" "$HOST" "test -s '$REMOTE_RUN/supervisor/pipeline_terminal.json'"; then
    sync_outputs
    summarize_epochs
    printf '%s watcher observed terminal and stopped\n' "$(date --iso-8601=seconds)" >>"$LOG"
    exit 0
  fi
  sleep 60
done
