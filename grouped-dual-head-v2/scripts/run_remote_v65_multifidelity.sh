#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2}
HOME_PROJECT=${HOME_PROJECT:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
TORCHRUN=${TORCHRUN:-/root/miniconda3/bin/torchrun}
SOURCE_H5=${SOURCE_H5:-$DATA_ROOT/dataset_v1.h5}
FROZEN_CONFIG=${FROZEN_CONFIG:-$DATA_ROOT/frozen_config.yaml}
FROZEN_MANIFEST=${FROZEN_MANIFEST:-$DATA_ROOT/manifest.jsonl}
PILOT_CONFIG=${PILOT_CONFIG:-$WORK/configs/saved_time_v4/generated/v65_lwc84_multifidelity_pilot_4gpu.yaml}
LONG_CONFIG=${LONG_CONFIG:-$WORK/configs/saved_time_v4/generated/v66_lwc84_multifidelity_long_4gpu.yaml}
CACHE_ROOT=${CACHE_ROOT:-$HOME_PROJECT/1/pretraining/lwc84_multifidelity_teacher64_401solver_v2}
RUN_ROOT=${RUN_ROOT:-$HOME_PROJECT/1/pretraining/saved_time_v65_lwc84_multifidelity_r2}
SUPERVISOR=${SUPERVISOR:-$RUN_ROOT/supervisor}
SAME_PANEL_PARENT=${SAME_PANEL_PARENT:-$RUN_ROOT/parent_same_panel_seed401.json}
SOLVER_BATCH_SIZE=${SOLVER_BATCH_SIZE:-120}
REPLAY_AUDIT=${REPLAY_AUDIT:-0}

mkdir -p "$SUPERVISOR" "$CACHE_ROOT/shards" "$CACHE_ROOT/logs" "$RUN_ROOT"
exec 9>"$SUPERVISOR/pipeline.lock"
if ! flock -n 9; then
  echo "V65 exact-grid numerical-teacher pipeline is already active" >&2
  exit 73
fi
exec >>"$SUPERVISOR/pipeline.log" 2>&1

finalize() {
  rc=$?
  "$PYTHON" - "$SUPERVISOR/pipeline_terminal.json" "$rc" <<'PY'
import json, os, sys
path, code = sys.argv[1], int(sys.argv[2])
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "complete" if code == 0 else "failed", "return_code": code}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
}
trap finalize EXIT

cd "$WORK"
echo "$(date --iso-8601=seconds) V65 preflight"
for required in "$SOURCE_H5" "$FROZEN_CONFIG" "$FROZEN_MANIFEST"; do
  [[ -s "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
"$PYTHON" -m pytest -q \
  tests/test_lwc84_velocity_models.py \
  tests/saved_time_phase_operator_v4/test_multifidelity.py \
  tests/saved_time_phase_operator_v4/test_multifidelity_cache_cli.py \
  tests/saved_time_phase_operator_v4/test_v65_multifidelity.py
"$PYTHON" scripts/train_saved_time_v4_full_support.py \
  --config "$PILOT_CONFIG" --audit-only

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [[ "$REPLAY_AUDIT" == 1 ]]; then
  echo "$(date --iso-8601=seconds) optional exact-grid solver replay audit"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/build_lwc84_multifidelity_cache.py \
    --source-h5 "$SOURCE_H5" --dataset-config "$FROZEN_CONFIG" \
    --manifest "$FROZEN_MANIFEST" \
    --output-h5 "$CACHE_ROOT/shards/replay_audit_shard_00.h5" \
    --shard-index 0 --shard-count 4 --solver-batch-size "$SOLVER_BATCH_SIZE" \
    --device cuda:0 --time-count 64 --max-batches 1 \
    >"$CACHE_ROOT/logs/replay_audit.log" 2>&1
fi

echo "$(date --iso-8601=seconds) zero-copy exact 401-to-201 numerical teacher"
"$PYTHON" scripts/merge_lwc84_multifidelity_cache.py \
  --source-h5 "$SOURCE_H5" --dataset-config "$FROZEN_CONFIG" \
  --manifest "$FROZEN_MANIFEST" --source-direct --time-count 64 \
  --output-h5 "$CACHE_ROOT/teacher_vds.h5" \
  >"$CACHE_ROOT/logs/merge.log" 2>&1

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "$(date --iso-8601=seconds) exact V49 parent on seed-401 fixed panel"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/evaluate_saved_time_family_expert_parent.py \
  --config "$PILOT_CONFIG" --output "$SAME_PANEL_PARENT" --device cuda:0 \
  >"$RUN_ROOT/parent_same_panel_screen.log" 2>&1

if "$PYTHON" - "$RUN_ROOT/smoke/terminal.json" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "complete" and int(payload.get("global_step", 0)) >= 2 else 1)
PY
then
  echo "$(date --iso-8601=seconds) reusing completed four-GPU smoke"
else
  echo "$(date --iso-8601=seconds) four-GPU smoke"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$PILOT_CONFIG" --smoke-updates 2 \
    >"$RUN_ROOT/smoke_screen.log" 2>&1
fi

echo "$(date --iso-8601=seconds) four-epoch pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$PILOT_CONFIG" --pilot \
  >"$RUN_ROOT/pilot_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_v65_multifidelity.py \
  --parent-report "$SAME_PANEL_PARENT" \
  --smoke-metrics "$RUN_ROOT/smoke/metrics.jsonl" \
  --smoke-terminal "$RUN_ROOT/smoke/terminal.json" \
  --pilot-metrics "$RUN_ROOT/pilot/metrics.jsonl" \
  --cache-summary "$CACHE_ROOT/teacher_vds.summary.json" \
  --output "$RUN_ROOT/v65_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -eq 2 ]]; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]; partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "pilot_rejected", "long_run_launched": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
  exit 0
fi
[[ $gate_rc -eq 0 ]] || exit "$gate_rc"

echo "$(date --iso-8601=seconds) V65 passed; starting solver-free V66"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$LONG_CONFIG" >"$RUN_ROOT/v66_long_screen.log" 2>&1
"$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]; partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "long_complete", "long_run_launched": True}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
