#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
V31_ART="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis96_batch96_micro4_4gpu_remote_r1"
V31_SUP="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis_supervisor_r1"
V33_ART="$HOME_PROJECT/artifacts/saved_time_v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote_r1"
V33_SUP="$HOME_PROJECT/artifacts/saved_time_v33_temporal_delta_floor_supervisor_r1"
V32_CONFIG=configs/saved_time_v4/generated/v32_temporal_basis96_long_4gpu_remote.yaml
V32_ART="$HOME_PROJECT/artifacts/saved_time_v32_temporal_basis96_long_4gpu_remote_r1"
V34_CONFIG=configs/saved_time_v4/generated/v34_temporal_basis96_deltafloor01_long_4gpu_remote.yaml
V34_ART="$HOME_PROJECT/artifacts/saved_time_v34_temporal_basis96_deltafloor01_long_4gpu_remote_r1"
V35_ART="$HOME_PROJECT/artifacts/saved_time_v35_temporal_basis96_gate_absorb_batch96_micro4_4gpu_remote_r1"
V35_SUP="$HOME_PROJECT/artifacts/saved_time_v35_temporal_gate_absorption_supervisor_r1"
V36_CONFIG=configs/saved_time_v4/generated/v36_temporal_basis96_gate_absorb_long_4gpu_remote.yaml
V36_ART="$HOME_PROJECT/artifacts/saved_time_v36_temporal_basis96_gate_absorb_long_4gpu_remote_r1"
SUPERVISOR="$HOME_PROJECT/artifacts/saved_time_temporal_final_accuracy_gate_r1"

mkdir -p "$SUPERVISOR"
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
echo "$(date --iso-8601=seconds) waiting for a promoted temporal long run"
while [[ ! -s "$V31_SUP/pipeline_terminal.json" ]]; do sleep 60; done
if json_passes "$V31_ART/evidence_gate.json"; then
  TARGET_CONFIG=$V32_CONFIG
  TARGET_ART=$V32_ART
  TARGET_NAME=v32
else
  while [[ ! -s "$V33_SUP/pipeline_terminal.json" ]]; do sleep 60; done
  if json_passes "$V33_ART/evidence_gate.json"; then
    TARGET_CONFIG=$V34_CONFIG
    TARGET_ART=$V34_ART
    TARGET_NAME=v34
  else
    while [[ ! -s "$V35_SUP/pipeline_terminal.json" ]]; do sleep 60; done
    if json_passes "$V35_ART/evidence_gate.json"; then
      TARGET_CONFIG=$V36_CONFIG
      TARGET_ART=$V36_ART
      TARGET_NAME=v36
    else
      "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "no_promoted_temporal_candidate", "accuracy_gate_passed": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
      echo "$(date --iso-8601=seconds) V31, V33 and V35 all missed their pilot gates"
      exit 0
    fi
  fi
fi

echo "$(date --iso-8601=seconds) selected $TARGET_NAME; waiting for run terminal"
while [[ ! -s "$TARGET_ART/run/terminal.json" ]]; do sleep 60; done
if ! "$PYTHON" - "$TARGET_ART/run/terminal.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)
PY
then
  echo "$(date --iso-8601=seconds) promoted long run did not complete"
  exit 3
fi

"$PYTHON" - "$TARGET_ART/run/metrics.jsonl" "$SUPERVISOR/development_selection.json" <<'PY'
import json, math, os, sys
source, output = sys.argv[1:]
candidates=[]
for line in open(source):
    row=json.loads(line)
    if row.get("event") != "epoch" or row.get("validation_scope") != "fixed_full_time_panel":
        continue
    for scale, metrics in ((0.0, row["metrics"].get("coarse_metrics")), (1.0, row.get("metrics"))):
        if not isinstance(metrics, dict) or metrics.get("frame_count") != 48*401:
            continue
        family=metrics.get("family_relative_l2", {})
        score=float(metrics.get("aggregate_relative_l2", math.inf))
        if set(family) != {"uniform", "layered", "marmousi"} or not math.isfinite(score):
            continue
        passed=score < 0.10 and all(float(family[name]) < 0.12 for name in family)
        candidates.append({"epoch":row["epoch"], "checkpoint":row["checkpoint"], "scale":scale, "metrics":metrics, "passes":passed})
if not candidates:
    raise SystemExit("no complete 48x401 development evidence")
passing=[row for row in candidates if row["passes"]]
selected=min(passing or candidates, key=lambda row:(row["metrics"]["aggregate_relative_l2"], row["epoch"], row["scale"]))
partial=f"{output}.partial.{os.getpid()}"
with open(partial,"w") as handle:
    json.dump(selected,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(partial,output)
print(json.dumps(selected,sort_keys=True))
PY

if ! "$PYTHON" - "$SUPERVISOR/development_selection.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("passes") is True else 1)
PY
then
  "$PYTHON" - "$SUPERVISOR/outcome.json" "$TARGET_NAME" <<'PY'
import json, os, sys
path, name = sys.argv[1:]
partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as handle:
    json.dump({"status":"development_48x401_target_failed","candidate":name,"accuracy_gate_passed":False},handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) development target failed; full census withheld"
  exit 0
fi

CHECKPOINT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$SUPERVISOR/development_selection.json")
SCALE=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scale"])' "$SUPERVISOR/development_selection.json")
PANEL="$SUPERVISOR/heldout_panel2_48x401.json"
FULL="$SUPERVISOR/heldout_all480_401.json"

"$PYTHON" -u scripts/evaluate_saved_time_v6_scale_sweep.py \
  --config "$TARGET_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-identity "$TARGET_ART/run/run_identity.json" \
  --output "$PANEL" \
  --validation-records 48 \
  --validation-frames 401 \
  --panel-epoch 2 \
  --scales "$SCALE"

if ! "$PYTHON" - "$PANEL" "$SCALE" <<'PY'
import json, sys
report=json.load(open(sys.argv[1])); metrics=report["scales"][sys.argv[2]]
family=metrics["family_relative_l2"]
passed=metrics["aggregate_relative_l2"] < .10 and all(float(family[name]) < .12 for name in ("uniform","layered","marmousi"))
raise SystemExit(0 if passed else 1)
PY
then
  "$PYTHON" - "$SUPERVISOR/outcome.json" "$TARGET_NAME" <<'PY'
import json, os, sys
path,name=sys.argv[1:]; partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as handle:
    json.dump({"status":"disjoint_panel_48x401_target_failed","candidate":name,"accuracy_gate_passed":False},handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) disjoint panel target failed; full census withheld"
  exit 0
fi

"$PYTHON" -u scripts/evaluate_saved_time_v6_scale_sweep.py \
  --config "$TARGET_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-identity "$TARGET_ART/run/run_identity.json" \
  --output "$FULL" \
  --validation-records 480 \
  --validation-frames 401 \
  --panel-epoch 2 \
  --scales "$SCALE"

"$PYTHON" - "$FULL" "$SCALE" "$SUPERVISOR/outcome.json" "$TARGET_NAME" "$CHECKPOINT" <<'PY'
import json, os, sys
report=json.load(open(sys.argv[1])); scale=sys.argv[2]; path=sys.argv[3]; name=sys.argv[4]; checkpoint=sys.argv[5]
metrics=report["scales"][scale]; family=metrics["family_relative_l2"]
passed=metrics["aggregate_relative_l2"] < .10 and all(float(family[key]) < .12 for key in ("uniform","layered","marmousi"))
value={"status":"success" if passed else "full_480x401_target_failed","candidate":name,"checkpoint":checkpoint,"correction_scale":float(scale),"metrics":metrics,"accuracy_gate_passed":passed,"record_count":480,"saved_time_count":401,"one_source_per_record":True,"receiver_input":False,"wavefield_shape":[201,201]}
partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as handle: json.dump(value,handle,indent=2,sort_keys=True); handle.write("\n")
os.replace(partial,path)
raise SystemExit(0 if passed else 2)
PY
