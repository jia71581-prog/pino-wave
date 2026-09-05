#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun

V19_CONFIG=configs/saved_time_v4/generated/v19_residual_wakeup_bigbatch192_materialized.yaml
V28_CONFIG=configs/saved_time_v4/generated/v28_update_density_batch96_micro3_4gpu_remote.yaml
V29_CONFIG=configs/saved_time_v4/generated/v29_family_curriculum_batch96_micro3_4gpu_remote.yaml
V31_CONFIG=configs/saved_time_v4/generated/v31_temporal_basis96_batch96_micro6_4gpu_remote.yaml
V32_CONFIG=configs/saved_time_v4/generated/v32_temporal_basis96_long_4gpu_remote.yaml

V19_ART="$HOME_PROJECT/artifacts/saved_time_v19_residual_wakeup_bigbatch192_4gpu_remote_r1"
V28_SUP="$HOME_PROJECT/artifacts/saved_time_v28_v29_update_density_supervisor_r1"
V28_ART="$HOME_PROJECT/artifacts/saved_time_v28_update_density_batch96_micro3_4gpu_remote_r1"
V29_ART="$HOME_PROJECT/artifacts/saved_time_v29_family_curriculum_batch96_micro3_4gpu_remote_r1"
V31_ART="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis96_batch96_micro6_4gpu_remote_r1"
V32_ART="$HOME_PROJECT/artifacts/saved_time_v32_temporal_basis96_long_4gpu_remote_r1"
SUPERVISOR="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis_supervisor_r1"

mkdir -p "$SUPERVISOR" "$WORK/configs/saved_time_v4/generated"
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
echo "$(date --iso-8601=seconds) installed; waiting for V28/V29 terminal"
while [[ ! -s "$V28_SUP/pipeline_terminal.json" ]]; do
  sleep 60
done
echo "$(date --iso-8601=seconds) V28/V29 terminal observed"

"$PYTHON" - "$V19_ART/pilot/run_identity.json" "$V19_CONFIG" <<'PY'
import json, os, sys, yaml
source, target = sys.argv[1:]
partial = f"{target}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    yaml.safe_dump(json.load(open(source))["config"], handle, sort_keys=False)
os.replace(partial, target)
PY

candidates=(--candidate-config "$V19_CONFIG")
for pair in "$V28_CONFIG:$V28_ART/evidence_gate.json" "$V29_CONFIG:$V29_ART/evidence_gate.json"; do
  config=${pair%%:*}
  gate=${pair#*:}
  if "$PYTHON" - "$gate" <<'PY'
import json, sys
try:
    passed = json.load(open(sys.argv[1])).get("passes") is True
except (FileNotFoundError, json.JSONDecodeError):
    passed = False
raise SystemExit(0 if passed else 1)
PY
  then
    candidates+=(--candidate-config "$config")
    echo "$(date --iso-8601=seconds) adding gate-approved parent $config"
  fi
done

mkdir -p "$V31_ART"
"$PYTHON" scripts/prepare_saved_time_temporal_basis_candidate.py \
  "${candidates[@]}" \
  --output-config "$V31_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v31_temporal_basis96_batch96_micro6_4gpu_remote_r1 \
  --report "$V31_ART/parent_selection.json" \
  --temporal-basis-rank 96 \
  --effective-batch 96 \
  --pilot-epochs 3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "$(date --iso-8601=seconds) starting V31 two-update occupancy/wake-up smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V31_CONFIG" --smoke-updates 2 \
  >>"$V31_ART/smoke_screen.log" 2>&1

"$PYTHON" - "$V31_ART/smoke/terminal.json" "$V31_ART/smoke/metrics.jsonl" "$V31_CONFIG" <<'PY'
import json, math, sys, yaml
terminal = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
row = [item for item in rows if item.get("event") == "epoch"][-1]
config = yaml.safe_load(open(sys.argv[3]))
gradients = row.get("gradient_norms", {})
checks = {
    "complete": terminal.get("status") == "complete",
    "rank_96": int(config["variant_overrides"]["temporal_basis_rank"]) == 96,
    "physical_microbatch_6": int(row["physical_microbatch_records"]) == 6,
    "global_macros_per_update_8": int(row["ddp"]["global_macros_per_update"]) == 8,
    "temporal_gate_active": math.isfinite(float(gradients.get("temporal_basis_gate", 0.0))) and float(gradients.get("temporal_basis_gate", 0.0)) > 0.0,
    "temporal_features_active": math.isfinite(float(gradients.get("temporal_basis_features", 0.0))) and float(gradients.get("temporal_basis_features", 0.0)) > 0.0,
    "cuda_below_23_gib": int(row["peak_cuda_bytes"]) < 23 * 1024**3,
}
print(json.dumps({"checks": checks, "peak_cuda_bytes": row["peak_cuda_bytes"]}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(5)
PY

echo "$(date --iso-8601=seconds) starting V31 three-epoch pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V31_CONFIG" --pilot \
  >>"$V31_ART/pilot_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_temporal_basis_candidate.py \
  --selection-report "$V31_ART/parent_selection.json" \
  --candidate-metrics "$V31_ART/pilot/metrics.jsonl" \
  --output "$V31_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  echo "$(date --iso-8601=seconds) V31 did not pass evidence gate; no long run"
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "candidate_rejected", "long_run_launched": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
  exit 0
fi

mkdir -p "$V32_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$V31_CONFIG" \
  --output-config "$V32_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v32_temporal_basis96_long_4gpu_remote_r1 \
  --report "$V32_ART/parent_selection.json" \
  --epochs 40

echo "$(date --iso-8601=seconds) starting evidence-approved V32 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V32_CONFIG" \
  >>"$V32_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) V32 long training finished"
