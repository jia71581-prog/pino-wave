#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
CONFIG=${CONFIG:-$WORK/configs/saved_time_v4/generated/v66_lwc84_multifidelity_long_4gpu.yaml}
RUN_ROOT=${RUN_ROOT:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/saved_time_v65_lwc84_multifidelity_r2}
STATE_ROOT=${STATE_ROOT:-$RUN_ROOT/postprocess_v1}
GATE=${GATE:-$RUN_ROOT/v65_gate.json}
RUN_TERMINAL=${RUN_TERMINAL:-$RUN_ROOT/run/terminal.json}
SEALED_ROOT=${SEALED_ROOT:-$RUN_ROOT/run/sealed_evaluation}
SEALED_REPORT=${SEALED_REPORT:-$SEALED_ROOT/evaluation_report.json}
FIGURE_ROOT=${FIGURE_ROOT:-$RUN_ROOT/run/three_family_figures}
FIGURE_REPORT=${FIGURE_REPORT:-$FIGURE_ROOT/three_family_figure_evaluation_report.json}
CHECKPOINT=${CHECKPOINT:-$RUN_ROOT/run/best.pt}
RUN_IDENTITY=${RUN_IDENTITY:-$RUN_ROOT/run/run_identity.json}

mkdir -p "$STATE_ROOT"
exec 9>"$STATE_ROOT/postprocess.lock"
if ! flock -n 9; then
  echo "V66 postprocess supervisor is already active" >&2
  exit 73
fi
exec >>"$STATE_ROOT/postprocess.log" 2>&1

terminal_written=0
write_terminal() {
  local status=$1
  local reason=$2
  "$PYTHON" - "$STATE_ROOT/terminal.json" "$status" "$reason" <<'PY'
import json, os, sys
path, status, reason = sys.argv[1:]
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "x", encoding="utf8") as handle:
    json.dump({"status": status, "reason": reason}, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(partial, path)
PY
  terminal_written=1
}

finalize() {
  local rc=$?
  if [[ $rc -ne 0 && $terminal_written -eq 0 ]]; then
    write_terminal failed "postprocess command exited with code $rc"
  fi
}
trap finalize EXIT

cd "$WORK"
while true; do
  decision=$(
    "$PYTHON" -m saved_time_phase_operator_v4.postprocess_supervisor \
      --gate "$GATE" \
      --run-terminal "$RUN_TERMINAL" \
      --sealed-report "$SEALED_REPORT" \
      --figures-report "$FIGURE_REPORT"
  )
  action=$(printf '%s' "$decision" | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["action"])')
  echo "$(date --iso-8601=seconds) $decision"
  case "$action" in
    "wait_gate"|"wait_run")
      sleep 60
      ;;
    "evaluate_sealed")
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/evaluate_saved_time_v4_full_support.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --output "$SEALED_ROOT" \
        --device cuda:0 --time-block 16 >"$STATE_ROOT/sealed_evaluation_screen.log" 2>&1
      ;;
    "render_figures")
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/evaluate_saved_time_v62_three_family_figures.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --run-identity "$RUN_IDENTITY" --output-dir "$FIGURE_ROOT" \
        --device cuda:0 --time-block 8 --snapshot-count 6 \
        >"$STATE_ROOT/three_family_figures_screen.log" 2>&1
      ;;
    "pilot_rejected"|"run_failed")
      write_terminal "$action" "training evidence does not permit postprocessing"
      exit 0
      ;;
    "continue_experiments")
      write_terminal target_not_met "sealed neural-operator thresholds were not met"
      exit 0
      ;;
    "complete")
      write_terminal complete "sealed thresholds and three-family figures are verified"
      exit 0
      ;;
    *)
      echo "unknown postprocess action: $action" >&2
      exit 2
      ;;
  esac
done
