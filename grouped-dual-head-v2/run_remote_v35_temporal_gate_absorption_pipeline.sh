#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun
V31_ART="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis96_batch96_micro4_4gpu_remote_r1"
V33_CONFIG=configs/saved_time_v4/generated/v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote.yaml
V33_ART="$HOME_PROJECT/artifacts/saved_time_v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote_r1"
V33_SUP="$HOME_PROJECT/artifacts/saved_time_v33_temporal_delta_floor_supervisor_r1"
V35_CONFIG=configs/saved_time_v4/generated/v35_temporal_basis96_gate_absorb_batch96_micro4_4gpu_remote.yaml
V35_ART="$HOME_PROJECT/artifacts/saved_time_v35_temporal_basis96_gate_absorb_batch96_micro4_4gpu_remote_r1"
V36_CONFIG=configs/saved_time_v4/generated/v36_temporal_basis96_gate_absorb_long_4gpu_remote.yaml
V36_ART="$HOME_PROJECT/artifacts/saved_time_v36_temporal_basis96_gate_absorb_long_4gpu_remote_r1"
SUPERVISOR="$HOME_PROJECT/artifacts/saved_time_v35_temporal_gate_absorption_supervisor_r1"

mkdir -p "$SUPERVISOR" "$WORK/configs/saved_time_v4/generated"
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
echo "$(date --iso-8601=seconds) waiting for V33 terminal"
while [[ ! -s "$V33_SUP/pipeline_terminal.json" ]]; do sleep 60; done
if json_passes "$V31_ART/evidence_gate.json" || json_passes "$V33_ART/evidence_gate.json"; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json,os,sys
path=sys.argv[1]; partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as h: json.dump({"status":"skipped_prior_candidate_promoted","launched":False},h,indent=2,sort_keys=True); h.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) prior temporal candidate passed; V35 skipped"
  exit 0
fi

mkdir -p "$V35_ART"
"$PYTHON" scripts/prepare_saved_time_temporal_gate_absorption_candidate.py \
  --candidate-config "$V33_CONFIG" \
  --output-config "$V35_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v35_temporal_basis96_gate_absorb_batch96_micro4_4gpu_remote_r1 \
  --report "$V35_ART/parent_selection.json" \
  --pilot-epochs 3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
echo "$(date --iso-8601=seconds) starting V35 exact-reparameterization smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$V35_CONFIG" --smoke-updates 2 >>"$V35_ART/smoke_screen.log" 2>&1

"$PYTHON" - "$V35_ART/smoke/terminal.json" "$V35_ART/smoke/metrics.jsonl" "$V35_CONFIG" <<'PY'
import json,math,sys,yaml
terminal=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
row=[x for x in rows if x.get("event")=="epoch"][-1]; config=yaml.safe_load(open(sys.argv[3])); gradients=row.get("gradient_norms",{})
absorption=row.get("residual_activation",{}).get("temporal_basis_gate_absorption",{})
checks={
 "complete":terminal.get("status")=="complete",
 "rank_96":int(config["variant_overrides"]["temporal_basis_rank"])==96,
 "absorption_requested":config["residual_recovery"].get("absorb_temporal_basis_gate") is True,
 "post_gate_one":math.isclose(float(absorption.get("post_gate",0.0)),1.0),
 "pre_gate_finite":math.isfinite(float(absorption.get("pre_gate",float("nan")))),
 "physical_microbatch_4":int(row["physical_microbatch_records"])==4,
 "global_macros_8":int(row["ddp"]["global_macros_per_update"])==8,
 "temporal_features_active":float(gradients.get("temporal_basis_features",0.0))>0.0,
 "cuda_below_23_gib":int(row["peak_cuda_bytes"])<23*1024**3,
}
print(json.dumps({"checks":checks,"absorption":absorption,"peak_cuda_bytes":row["peak_cuda_bytes"]},sort_keys=True))
if not all(checks.values()): raise SystemExit(5)
PY

echo "$(date --iso-8601=seconds) starting V35 three-epoch pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$V35_CONFIG" --pilot >>"$V35_ART/pilot_screen.log" 2>&1
set +e
"$PYTHON" scripts/gate_saved_time_temporal_delta_floor_candidate.py \
  --selection-report "$V35_ART/parent_selection.json" \
  --candidate-metrics "$V35_ART/pilot/metrics.jsonl" \
  --output "$V35_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json,os,sys
path=sys.argv[1]; partial=f"{path}.partial.{os.getpid()}"
with open(partial,"w") as h: json.dump({"status":"candidate_rejected","long_run_launched":False},h,indent=2,sort_keys=True); h.write("\n")
os.replace(partial,path)
PY
  echo "$(date --iso-8601=seconds) V35 rejected; no V36 long run"
  exit 0
fi

mkdir -p "$V36_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$V35_CONFIG" --output-config "$V36_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v36_temporal_basis96_gate_absorb_long_4gpu_remote_r1 \
  --report "$V36_ART/parent_selection.json" --epochs 40
echo "$(date --iso-8601=seconds) starting evidence-approved V36 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 scripts/train_saved_time_v4_full_support.py \
  --config "$V36_CONFIG" >>"$V36_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) V36 long training finished"
