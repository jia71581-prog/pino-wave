#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun

V31_CONFIG=configs/saved_time_v4/generated/v31_temporal_basis96_batch96_micro4_4gpu_remote.yaml
V31_ART="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis96_batch96_micro4_4gpu_remote_r1"
V31_SUP="$HOME_PROJECT/artifacts/saved_time_v31_temporal_basis_supervisor_r1"
V33_CONFIG=configs/saved_time_v4/generated/v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote.yaml
V33_ART="$HOME_PROJECT/artifacts/saved_time_v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote_r1"
V34_CONFIG=configs/saved_time_v4/generated/v34_temporal_basis96_deltafloor01_long_4gpu_remote.yaml
V34_ART="$HOME_PROJECT/artifacts/saved_time_v34_temporal_basis96_deltafloor01_long_4gpu_remote_r1"
SUPERVISOR="$HOME_PROJECT/artifacts/saved_time_v33_temporal_delta_floor_supervisor_r1"

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
echo "$(date --iso-8601=seconds) waiting for V31/V32 pipeline terminal"
while [[ ! -s "$V31_SUP/pipeline_terminal.json" ]]; do
  sleep 60
done
echo "$(date --iso-8601=seconds) V31/V32 terminal observed"

if "$PYTHON" - "$V31_ART/evidence_gate.json" <<'PY'
import json, sys
try:
    passed = json.load(open(sys.argv[1])).get("passes") is True
except (FileNotFoundError, json.JSONDecodeError):
    passed = False
raise SystemExit(0 if passed else 1)
PY
then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "skipped_v31_promoted", "launched": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
  echo "$(date --iso-8601=seconds) V31 passed; V32 remains the promoted path"
  exit 0
fi

mkdir -p "$V33_ART"
"$PYTHON" scripts/prepare_saved_time_temporal_delta_floor_candidate.py \
  --candidate-config "$V31_CONFIG" \
  --output-config "$V33_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v33_temporal_basis96_deltafloor01_batch96_micro4_4gpu_remote_r1 \
  --report "$V33_ART/parent_selection.json" \
  --delta-energy-floor-fraction 0.1 \
  --dense-learning-rate-multiplier 2.0 \
  --pilot-epochs 3

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "$(date --iso-8601=seconds) starting V33 two-update smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V33_CONFIG" --smoke-updates 2 \
  >>"$V33_ART/smoke_screen.log" 2>&1

"$PYTHON" - "$V33_ART/smoke/terminal.json" "$V33_ART/smoke/metrics.jsonl" "$V33_CONFIG" <<'PY'
import json, math, sys, yaml
terminal = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
row = [item for item in rows if item.get("event") == "epoch"][-1]
config = yaml.safe_load(open(sys.argv[3]))
gradients = row.get("gradient_norms", {})
checks = {
    "complete": terminal.get("status") == "complete",
    "rank_96": int(config["variant_overrides"]["temporal_basis_rank"]) == 96,
    "delta_floor_01": math.isclose(float(config["loss"]["delta_energy_floor_fraction"]), 0.1),
    "physical_microbatch_4": int(row["physical_microbatch_records"]) == 4,
    "global_macros_per_update_8": int(row["ddp"]["global_macros_per_update"]) == 8,
    "temporal_gate_active": float(gradients.get("temporal_basis_gate", 0.0)) > 0.0,
    "temporal_features_active": float(gradients.get("temporal_basis_features", 0.0)) > 0.0,
    "cuda_below_23_gib": int(row["peak_cuda_bytes"]) < 23 * 1024**3,
}
print(json.dumps({"checks": checks, "peak_cuda_bytes": row["peak_cuda_bytes"]}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(5)
PY

echo "$(date --iso-8601=seconds) starting V33 three-epoch pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V33_CONFIG" --pilot \
  >>"$V33_ART/pilot_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_temporal_delta_floor_candidate.py \
  --selection-report "$V33_ART/parent_selection.json" \
  --candidate-metrics "$V33_ART/pilot/metrics.jsonl" \
  --output "$V33_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "candidate_rejected", "long_run_launched": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
  echo "$(date --iso-8601=seconds) V33 rejected; no V34 long run"
  exit 0
fi

mkdir -p "$V34_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$V33_CONFIG" \
  --output-config "$V34_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v34_temporal_basis96_deltafloor01_long_4gpu_remote_r1 \
  --report "$V34_ART/parent_selection.json" \
  --epochs 40

echo "$(date --iso-8601=seconds) starting evidence-approved V34 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V34_CONFIG" \
  >>"$V34_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) V34 long training finished"
