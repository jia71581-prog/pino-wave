#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun

V19_CONFIG=configs/saved_time_v4/generated/v19_residual_wakeup_bigbatch192_materialized.yaml
V20_CONFIG=configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml
V26_CONFIG=configs/saved_time_v4/generated/v26_coupled_2d_bigbatch192_micro8_4gpu_remote.yaml
V28_CONFIG=configs/saved_time_v4/generated/v28_update_density_batch96_micro3_4gpu_remote.yaml
V29_CONFIG=configs/saved_time_v4/generated/v29_family_curriculum_batch96_micro3_4gpu_remote.yaml
V30_CONFIG=configs/saved_time_v4/generated/v30_update_density_winner_long_4gpu_remote.yaml

V19_ART="$HOME_PROJECT/artifacts/saved_time_v19_residual_wakeup_bigbatch192_4gpu_remote_r1"
V26_ART="$HOME_PROJECT/artifacts/saved_time_v26_coupled_2d_bigbatch192_micro8_4gpu_remote_r1"
V28_ART="$HOME_PROJECT/artifacts/saved_time_v28_update_density_batch96_micro3_4gpu_remote_r1"
V29_ART="$HOME_PROJECT/artifacts/saved_time_v29_family_curriculum_batch96_micro3_4gpu_remote_r1"
V30_ART="$HOME_PROJECT/artifacts/saved_time_v30_update_density_winner_long_4gpu_remote_r1"
SUPERVISOR="$HOME_PROJECT/artifacts/saved_time_v28_v29_update_density_supervisor_r1"

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
echo "$(date --iso-8601=seconds) installed; waiting for v26/v27 pipeline to release the four GPUs"
while [[ ! -s "$V26_ART/pipeline_terminal.json" ]]; do
  sleep 60
done
echo "$(date --iso-8601=seconds) v26/v27 pipeline terminal observed"

# The strongest historical corrected candidate (v19) was generated from a run
# identity rather than retained as a YAML file. Materialize that exact config.
"$PYTHON" - "$V19_ART/pilot/run_identity.json" "$V19_CONFIG" <<'PY'
import os, sys, yaml, json
source, target = sys.argv[1:]
config = json.load(open(source))["config"]
partial = f"{target}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
os.replace(partial, target)
PY

candidates=(--candidate-config "$V19_CONFIG" --candidate-config "$V20_CONFIG")
if "$PYTHON" - "$V26_ART/evidence_gate.json" <<'PY'
import json, sys
try:
    value = json.load(open(sys.argv[1]))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if value.get("passes") is True else 1)
PY
then
  echo "$(date --iso-8601=seconds) adding gate-approved v26 to corrected parent pool"
  candidates+=(--candidate-config "$V26_CONFIG")
else
  echo "$(date --iso-8601=seconds) v26 did not pass its evidence gate; excluding it from parent selection"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

verify_smoke() {
  local artifact=$1
  local config=$2
  "$PYTHON" - "$artifact/smoke/terminal.json" "$artifact/smoke/metrics.jsonl" "$config" <<'PY'
import json, math, sys, yaml
terminal = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
row = [item for item in rows if item.get("event") == "epoch"][-1]
config = yaml.safe_load(open(sys.argv[3]))
gradients = row.get("gradient_norms", {})
values = [float(value) for value in gradients.values()]
checks = {
    "complete": terminal.get("status") == "complete",
    "macro_records_12": int(config["macro_records"]) == 12,
    "physical_microbatch_3": int(row["physical_microbatch_records"]) == 3,
    "global_macros_per_update_8": int(row["ddp"]["global_macros_per_update"]) == 8,
    "finite_nonzero_gradient": bool(values) and all(math.isfinite(value) for value in values) and any(value > 0 for value in values),
    "cuda_below_23_gib": int(row["peak_cuda_bytes"]) < 23 * 1024**3,
}
print(json.dumps({"checks": checks, "peak_cuda_bytes": row["peak_cuda_bytes"]}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(5)
PY
}

run_candidate() {
  local config=$1
  local artifact=$2
  local intervention=$3
  shift 3

  mkdir -p "$artifact"
  "$PYTHON" scripts/prepare_saved_time_update_density_candidate.py \
    "${candidates[@]}" \
    --output-config "$config" \
    --artifact-dir "${artifact#/root/autodl-tmp}" \
    --report "$artifact/parent_selection.json" \
    --effective-batch 96 \
    --pilot-epochs 3 \
    "$@"

  echo "$(date --iso-8601=seconds) starting $intervention two-update occupancy smoke"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$config" --smoke-updates 2 \
    >>"$artifact/smoke_screen.log" 2>&1
  verify_smoke "$artifact" "$config"

  echo "$(date --iso-8601=seconds) starting $intervention three-epoch pilot"
  "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$config" --pilot \
    >>"$artifact/pilot_screen.log" 2>&1

  set +e
  "$PYTHON" scripts/gate_saved_time_update_density_candidate.py \
    --selection-report "$artifact/parent_selection.json" \
    --candidate-metrics "$artifact/pilot/metrics.jsonl" \
    --output "$artifact/evidence_gate.json"
  local gate_rc=$?
  set -e
  echo "$(date --iso-8601=seconds) $intervention evidence gate rc=$gate_rc"
}

run_candidate "$V28_CONFIG" "$V28_ART" update_density
run_candidate "$V29_CONFIG" "$V29_ART" family_curriculum --family-curriculum

passing=()
for pair in "$V28_CONFIG:$V28_ART/evidence_gate.json" "$V29_CONFIG:$V29_ART/evidence_gate.json"; do
  config=${pair%%:*}
  gate=${pair#*:}
  if "$PYTHON" - "$gate" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("passes") is True else 1)
PY
  then
    passing+=(--candidate-config "$config")
  fi
done

if [[ ${#passing[@]} -eq 0 ]]; then
  echo "$(date --iso-8601=seconds) neither independent intervention passed; no long run launched"
  "$PYTHON" - "$SUPERVISOR/outcome.json" <<'PY'
import json, os, sys
path = sys.argv[1]
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump({"status": "no_candidate_passed", "long_run_launched": False}, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(partial, path)
PY
  exit 0
fi

mkdir -p "$V30_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  "${passing[@]}" \
  --output-config "$V30_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v30_update_density_winner_long_4gpu_remote_r1 \
  --report "$V30_ART/parent_selection.json" \
  --epochs 40

echo "$(date --iso-8601=seconds) starting evidence-approved v30 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V30_CONFIG" \
  >>"$V30_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) v30 long training finished"
