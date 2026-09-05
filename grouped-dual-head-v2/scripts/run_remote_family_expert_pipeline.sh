#!/usr/bin/env bash
set -euo pipefail

WORK=${WORK:-/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2}
HOME_PROJECT=${HOME_PROJECT:-/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation}
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
TORCHRUN=${TORCHRUN:-/root/miniconda3/bin/torchrun}
PREDECESSOR_TERMINAL=${PREDECESSOR_TERMINAL:?set the V37 predecessor terminal path}
PREDECESSOR_FRESHNESS=${PREDECESSOR_FRESHNESS:?set a marker created before the predecessor launch}
PREDECESSOR_GATE=${PREDECESSOR_GATE:?set the V37 direct-parent gate path}
PREDECESSOR_GLOBAL_GATE=${PREDECESSOR_GLOBAL_GATE:?set the V37 global-parent gate path}
GLOBAL_PARENT_METRICS=${GLOBAL_PARENT_METRICS:?set the globally accepted parent metrics JSONL}
CANDIDATE_CONFIG=${CANDIDATE_CONFIG:?set the rejected direct-parent candidate config}
RUN_NAME=${RUN_NAME:-saved_time_family_experts_rank16_batch96_4gpu_r1}
TRAINING_TIME_BLOCK=${TRAINING_TIME_BLOCK:-2}
ADAMW_IMPLEMENTATION=${ADAMW_IMPLEMENTATION:-fused}
ROOT="$HOME_PROJECT/artifacts/$RUN_NAME"
SUPERVISOR="$ROOT/supervisor"
BASE_CONFIG="$WORK/configs/saved_time_v4/generated/${RUN_NAME}_base.yaml"
PILOT_CONFIG="$WORK/configs/saved_time_v4/generated/${RUN_NAME}_pilot.yaml"
LONG_CONFIG="$WORK/configs/saved_time_v4/generated/${RUN_NAME}_long.yaml"

mkdir -p "$SUPERVISOR" "$ROOT" "$WORK/configs/saved_time_v4/generated"
exec 9>"$SUPERVISOR/pipeline.lock"
if ! flock -n 9; then
  echo "family expert pipeline is already active" >&2
  exit 73
fi
if [[ -e "$SUPERVISOR/pipeline_terminal.json" ]]; then
  mv "$SUPERVISOR/pipeline_terminal.json" \
    "$SUPERVISOR/pipeline_terminal.superseded.$(date +%Y%m%dT%H%M%S).json"
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
json_passes() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
try:
    passed = json.load(open(sys.argv[1])).get("passes") is True
except (FileNotFoundError, json.JSONDecodeError):
    passed = False
raise SystemExit(0 if passed else 1)
PY
}

cd "$WORK"
while [[ ! -s "$PREDECESSOR_TERMINAL" || ! "$PREDECESSOR_TERMINAL" -nt "$PREDECESSOR_FRESHNESS" ]]; do
  echo "$(date --iso-8601=seconds) waiting for fresh V37 terminal"
  sleep 60
done
if json_passes "$PREDECESSOR_GATE" && json_passes "$PREDECESSOR_GLOBAL_GATE"; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]; partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "skipped_predecessor_promoted", "long_run_launched": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
  exit 0
fi

"$PYTHON" scripts/prepare_saved_time_family_expert_candidate.py \
  --candidate-config "$CANDIDATE_CONFIG" \
  --output-config "$BASE_CONFIG" \
  --artifact-dir "$ROOT/pilot" \
  --report "$ROOT/parent_selection.json" \
  --family-expert-rank 16 --physical-microbatch-records 3

"$PYTHON" - "$BASE_CONFIG" "$TRAINING_TIME_BLOCK" "$ADAMW_IMPLEMENTATION" <<'PY'
import os, pathlib, sys, yaml
path, time_block, implementation = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if time_block <= 0:
    raise ValueError("TRAINING_TIME_BLOCK must be positive")
if implementation not in {"single_tensor", "foreach", "fused"}:
    raise ValueError("ADAMW_IMPLEMENTATION is invalid")
config = yaml.safe_load(path.read_text())
optimizer = dict(config["optimizer"])
optimizer["training_time_block"] = time_block
optimizer["adamw_implementation"] = implementation
config["optimizer"] = optimizer
partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
partial.write_text(yaml.safe_dump(config, sort_keys=False))
os.replace(partial, path)
PY

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
probe_reports=()
for microbatch in 12 6 4 3; do
  probe_dir="$ROOT/memory_probe/micro${microbatch}"
  probe_config="$WORK/configs/saved_time_v4/generated/${RUN_NAME}_probe_micro${microbatch}.yaml"
  if [[ -d "$probe_dir" ]]; then
    mv "$probe_dir" "${probe_dir}.superseded.$(date +%Y%m%dT%H%M%S)"
  fi
  mkdir -p "$probe_dir"
  "$PYTHON" - "$BASE_CONFIG" "$probe_config" "$probe_dir" "$microbatch" <<'PY'
import pathlib, sys, yaml
source, output, artifact, microbatch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
config = yaml.safe_load(open(source))
config["artifact_dir"] = str(pathlib.Path(artifact).resolve())
config["microbatch_records"] = microbatch
family_experts = dict(config["family_experts"])
dense_epoch = int(family_experts.get("dense_unfreeze_epoch", 2))
family_experts["stage_epoch_offset"] = max(1, dense_epoch - 1)
config["family_experts"] = family_experts
with open(output, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
  set +e
  "$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
    --config "$probe_config" --smoke-updates 2 >"$probe_dir/screen.log" 2>&1
  probe_rc=$?
  set -e
  oom=false
  if grep -Eqi 'CUDA out of memory|OutOfMemoryError' "$probe_dir/screen.log"; then oom=true; fi
  "$PYTHON" - "$probe_dir/report.json" "$probe_dir/smoke/metrics.jsonl" \
    "$microbatch" "$probe_rc" "$oom" <<'PY'
import json, os, pathlib, sys
output, metrics, microbatch, rc, oom = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "true"
peak = None
if rc == 0 and metrics.exists():
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    epochs = [row for row in rows if row.get("event") == "epoch"]
    if epochs:
        peak = max(int(row["peak_cuda_bytes"]) for row in epochs)
report = {"physical_microbatch_records": microbatch, "return_code": rc, "oom": oom, "peak_cuda_bytes": peak}
partial = f"{output}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump(report, handle, indent=2, sort_keys=True); handle.write("\n")
os.replace(partial, output)
PY
  probe_reports+=(--report "$probe_dir/report.json")
done

"$PYTHON" scripts/select_saved_time_physical_microbatch.py \
  "${probe_reports[@]}" --maximum-gib 23 --output "$ROOT/memory_selection.json"
selected_microbatch=$("$PYTHON" - "$ROOT/memory_selection.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["selected_physical_microbatch_records"]))
PY
)
"$PYTHON" - "$BASE_CONFIG" "$PILOT_CONFIG" "$ROOT/pilot" "$selected_microbatch" <<'PY'
import pathlib, sys, yaml
source, output, artifact, microbatch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
config = yaml.safe_load(open(source))
config["artifact_dir"] = str(pathlib.Path(artifact).resolve())
config["microbatch_records"] = microbatch
with open(output, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

"$PYTHON" scripts/diagnose_saved_time_temporal_three_record_overfit.py \
  --candidate-config "$PILOT_CONFIG" --artifact-dir "$ROOT/overfit30" \
  --overfit-updates 30 --evaluate-every 10 \
  --microbatch-records 1 >"$ROOT/overfit30_screen.log" 2>&1

"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$PILOT_CONFIG" --pilot >"$ROOT/pilot_screen.log" 2>&1

"$PYTHON" - "$GLOBAL_PARENT_METRICS" "$ROOT/global_parent_metrics.json" <<'PY'
import json, os, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
epochs = [row for row in rows if row.get("event") == "epoch" and row.get("validation_scope") == "pilot_fixed_panel"]
if not epochs: raise ValueError("global parent has no fixed-panel epoch")
best = min(epochs, key=lambda row: (float(row["metrics"]["aggregate_relative_l2"]), int(row["epoch"])))
output = sys.argv[2]; partial = f"{output}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"metrics": best["metrics"], "epoch": int(best["epoch"]), "checkpoint": best["checkpoint"]}, handle, indent=2, sort_keys=True); handle.write("\n")
os.replace(partial, output)
PY

"$PYTHON" scripts/evaluate_saved_time_family_expert_parent.py \
  --config "$PILOT_CONFIG" \
  --output "$ROOT/same_panel_parent.json" \
  >"$ROOT/same_panel_parent_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_family_expert_candidate.py \
  --selection-report "$ROOT/parent_selection.json" \
  --global-parent-report "$ROOT/global_parent_metrics.json" \
  --same-panel-parent-report "$ROOT/same_panel_parent.json" \
  --overfit-report "$ROOT/overfit30/terminal.json" \
  --candidate-metrics "$ROOT/pilot/metrics.jsonl" \
  --output "$ROOT/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -eq 2 ]]; then
  echo '{"status":"candidate_rejected","long_run_launched":false}' >"$SUPERVISOR/outcome.json"
  exit 0
fi
if [[ $gate_rc -ne 0 ]]; then
  exit "$gate_rc"
fi

"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$PILOT_CONFIG" --output-config "$LONG_CONFIG" \
  --artifact-dir "$ROOT/long40" --report "$ROOT/long_parent_selection.json" \
  --epochs 40 --physical-microbatch-records "$selected_microbatch"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$LONG_CONFIG" >"$ROOT/long40_screen.log" 2>&1
