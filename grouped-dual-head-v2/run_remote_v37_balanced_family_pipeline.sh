#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun
V31_CONFIG=configs/saved_time_v4/generated/v31_temporal_basis96_batch96_micro4_4gpu_remote.yaml
V31_ART="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis96_batch96_micro4_4gpu_remote_r1"
V33_ART="$HOME_PROJECT/artifacts/saved_time_v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote_r1"
V35_CONFIG=configs/saved_time_v4/generated/v35_temporal_basis96_gate_absorb_batch96_micro4_4gpu_remote.yaml
V35_ART="$HOME_PROJECT/artifacts/saved_time_v35_temporal_basis96_gate_absorb_batch96_micro4_4gpu_remote_r1"
V35_SUP="$HOME_PROJECT/artifacts/saved_time_v35_temporal_gate_absorption_supervisor_r1"
FAMILY_GRADIENT_REPORT="$HOME_PROJECT/artifacts/saved_time_family_gradient_conflict_v31_gateabsorb_fixed32_r1/report.json"
V37_CONFIG=configs/saved_time_v4/generated/v37_temporal_basis96_balanced_family_batch96_micro3_4gpu_remote.yaml
V37_ART="$HOME_PROJECT/artifacts/saved_time_v37_temporal_basis96_balanced_family_batch96_micro3_4gpu_remote_r1"
V38_CONFIG=configs/saved_time_v4/generated/v38_temporal_basis96_balanced_family_long_4gpu_remote.yaml
V38_ART="$HOME_PROJECT/artifacts/saved_time_v38_temporal_basis96_balanced_family_long_4gpu_remote_r1"
SUPERVISOR="$HOME_PROJECT/artifacts/saved_time_v37_balanced_family_supervisor_r1"

mkdir -p "$SUPERVISOR" "$WORK/configs/saved_time_v4/generated"
if [[ -e "$SUPERVISOR/pipeline_terminal.json" ]]; then
  mv "$SUPERVISOR/pipeline_terminal.json" \
    "$SUPERVISOR/pipeline_terminal.superseded.$(date +%Y%m%dT%H%M%S).json"
fi
WAIT_MARKER="$SUPERVISOR/v35_wait_started"
touch "$WAIT_MARKER"
exec >>"$SUPERVISOR/pipeline.log" 2>&1
finalize() {
  rc=$?
  "$PYTHON" - "$SUPERVISOR/pipeline_terminal.json" "$rc" <<'PY'
import json,os,sys
path,code=sys.argv[1],int(sys.argv[2]); partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as h: json.dump({"status":"complete" if code==0 else "failed","return_code":code},h,indent=2,sort_keys=True); h.write("\n")
os.replace(partial,path)
PY
}
trap finalize EXIT
json_passes() {
  "$PYTHON" - "$1" <<'PY'
import json,sys
try: passed=json.load(open(sys.argv[1])).get("passes") is True
except (FileNotFoundError,json.JSONDecodeError): passed=False
raise SystemExit(0 if passed else 1)
PY
}

cd "$WORK"
echo "$(date --iso-8601=seconds) waiting for V35 terminal"
while [[ ! -s "$V35_SUP/pipeline_terminal.json" || ! "$V35_SUP/pipeline_terminal.json" -nt "$WAIT_MARKER" ]]; do sleep 60; done
if json_passes "$V33_ART/evidence_gate.json" || json_passes "$V35_ART/evidence_gate.json"; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json,os,sys
path=sys.argv[1]; partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as h: json.dump({"status":"skipped_prior_candidate_promoted","launched":False},h,indent=2,sort_keys=True); h.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) prior candidate passed; V37 skipped"
  exit 0
fi

if [[ -s "$V35_ART/pilot/metrics.jsonl" ]]; then
  candidate_args=(--candidate-config "$V35_CONFIG")
else
  candidate_args=(--candidate-config "$V31_CONFIG")
fi
mkdir -p "$V37_ART"
"$PYTHON" scripts/prepare_saved_time_update_density_candidate.py \
  "${candidate_args[@]}" \
  --output-config "$V37_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v37_temporal_basis96_balanced_family_batch96_micro3_4gpu_remote_r1 \
  --report "$V37_ART/parent_selection.json" \
  --effective-batch 96 --pilot-epochs 3 --physical-microbatch-records 2 \
  --balanced-families \
  --family-gradient-report "$FAMILY_GRADIENT_REPORT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
echo "$(date --iso-8601=seconds) starting V37 balanced-family smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$V37_CONFIG" --smoke-updates 2 >>"$V37_ART/smoke_screen.log" 2>&1

"$PYTHON" - "$V37_ART/smoke/terminal.json" "$V37_ART/smoke/metrics.jsonl" "$V37_CONFIG" <<'PY'
import json,math,sys,yaml
terminal=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
row=[x for x in rows if x.get("event")=="epoch"][-1]; config=yaml.safe_load(open(sys.argv[3])); gradients=row.get("gradient_norms",{})
stages=config.get("family_curriculum",{}).get("stages",[])
weights=config.get("family_gradient_weights",{})
checks={
 "complete":terminal.get("status")=="complete",
 "balanced_pattern":stages==[{"epochs":3,"macro_pattern":["uniform","layered","marmousi"]}],
 "inverse_norm_weights":all(math.isclose(float(weights.get(k,0.0)),v,rel_tol=1e-6) for k,v in {"layered":0.19387488473748973,"marmousi":2.2541190770111976,"uniform":0.5520060382513123}.items()),
 "physical_microbatch_2":int(row["physical_microbatch_records"])==2,
 "global_macros_8":int(row["ddp"]["global_macros_per_update"])==8,
 "temporal_features_active":float(gradients.get("temporal_basis_features",0.0))>0.0,
 "cuda_below_23_gib":int(row["peak_cuda_bytes"])<23*1024**3,
}
print(json.dumps({"checks":checks,"peak_cuda_bytes":row["peak_cuda_bytes"]},sort_keys=True))
if not all(checks.values()): raise SystemExit(5)
PY

echo "$(date --iso-8601=seconds) starting V37 three-epoch balanced-family pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$V37_CONFIG" --pilot >>"$V37_ART/pilot_screen.log" 2>&1
set +e
"$PYTHON" scripts/gate_saved_time_update_density_candidate.py \
  --selection-report "$V37_ART/parent_selection.json" \
  --candidate-metrics "$V37_ART/pilot/metrics.jsonl" \
  --output "$V37_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json,os,sys
path=sys.argv[1]; partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as h: json.dump({"status":"candidate_rejected","long_run_launched":False},h,indent=2,sort_keys=True); h.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) V37 rejected; no V38 long run"
  exit 0
fi

# A rejected V35 can still be the warm-start parent for V37.  In that case the
# ordinary continuation gate only proves improvement over V35, so additionally
# require the promoted epoch to beat the last globally accepted V31 checkpoint.
set +e
"$PYTHON" - "$V31_ART/pilot/metrics.jsonl" "$V37_ART/pilot/metrics.jsonl" \
  "$V37_ART/global_parent_gate.json" <<'PY'
import json
import os
import sys

parent_path, candidate_path, output_path = map(str, sys.argv[1:])

def best_fixed_panel(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [
        row for row in rows
        if row.get("event") == "epoch"
        and row.get("validation_scope") == "pilot_fixed_panel"
    ]
    if not rows:
        raise ValueError(f"no pilot_fixed_panel epoch in {path}")
    return min(rows, key=lambda row: (float(row["metrics"]["aggregate_relative_l2"]), int(row["epoch"])))

parent = best_fixed_panel(parent_path)
candidate = best_fixed_panel(candidate_path)
parent_metrics = parent["metrics"]
candidate_metrics = candidate["metrics"]
parent_aggregate = float(parent_metrics["aggregate_relative_l2"])
candidate_aggregate = float(candidate_metrics["aggregate_relative_l2"])
family_tolerance = 0.01
family_checks = {
    family: float(value) <= float(parent_metrics["family_relative_l2"][family]) + family_tolerance
    for family, value in candidate_metrics["family_relative_l2"].items()
}
checks = {
    "aggregate_better_than_v31": candidate_aggregate < parent_aggregate,
    "all_families_within_v31_tolerance": all(family_checks.values()),
}
report = {
    "schema": "saved_time_global_parent_gate_v1",
    "passes": all(checks.values()),
    "checks": checks,
    "family_checks": family_checks,
    "family_tolerance": family_tolerance,
    "parent": {
        "epoch": int(parent["epoch"]),
        "aggregate_relative_l2": parent_aggregate,
        "family_relative_l2": parent_metrics["family_relative_l2"],
    },
    "candidate": {
        "epoch": int(candidate["epoch"]),
        "aggregate_relative_l2": candidate_aggregate,
        "family_relative_l2": candidate_metrics["family_relative_l2"],
    },
}
partial = f"{output_path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, output_path)
print(json.dumps(report, sort_keys=True))
raise SystemExit(0 if report["passes"] else 2)
PY
global_gate_rc=$?
set -e
if [[ $global_gate_rc -ne 0 ]]; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json,os,sys
path=sys.argv[1]; partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as h: json.dump({"status":"candidate_rejected_by_global_parent","long_run_launched":False},h,indent=2,sort_keys=True); h.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) V37 did not beat V31 global parent; no V38 long run"
  exit 0
fi

mkdir -p "$V38_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$V37_CONFIG" --output-config "$V38_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v38_temporal_basis96_balanced_family_long_4gpu_remote_r1 \
  --report "$V38_ART/parent_selection.json" --epochs 40
echo "$(date --iso-8601=seconds) starting evidence-approved V38 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$V38_CONFIG" >>"$V38_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) V38 long training finished"
