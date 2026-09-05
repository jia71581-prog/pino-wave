#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
V20_CONFIG=configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml
V26_CONFIG=configs/saved_time_v4/generated/v26_coupled_2d_bigbatch192_micro8_4gpu_remote.yaml
V27_CONFIG=configs/saved_time_v4/generated/v27_coupled_2d_long_bigbatch192_micro8_4gpu_remote.yaml
V26_ART="$HOME_PROJECT/artifacts/saved_time_v26_coupled_2d_bigbatch192_micro8_4gpu_remote_r1"
V27_ART="$HOME_PROJECT/artifacts/saved_time_v27_coupled_2d_long_bigbatch192_micro8_4gpu_remote_r1"
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun

mkdir -p "$V26_ART" "$WORK/configs/saved_time_v4/generated"
exec >>"$V26_ART/pipeline.log" 2>&1

finalize() {
  rc=$?
  "$PYTHON" - "$V26_ART/pipeline_terminal.json" "$rc" <<'PY'
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
echo "$(date --iso-8601=seconds) stopping superseded v24 microbatch-6 pipeline"
screen -S fno_v24_coupled2d_pipeline -X quit 2>/dev/null || true
pkill -TERM -f 'train_saved_time_v4_full_support.py --config configs/saved_time_v4/generated/v24_coupled_2d_bigbatch192_4gpu_remote.yaml' 2>/dev/null || true
for _ in $(seq 1 30); do
  if ! pgrep -f 'train_saved_time_v4_full_support.py --config configs/saved_time_v4/generated/v24_coupled_2d_bigbatch192_4gpu_remote.yaml' >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f 'train_saved_time_v4_full_support.py --config configs/saved_time_v4/generated/v24_coupled_2d_bigbatch192_4gpu_remote.yaml' >/dev/null; then
  echo "$(date --iso-8601=seconds) v24 workers did not stop cleanly"
  exit 3
fi

"$PYTHON" scripts/prepare_saved_time_coupled_2d_candidate.py \
  --candidate-config "$V20_CONFIG" \
  --output-config "$V26_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v26_coupled_2d_bigbatch192_micro8_4gpu_remote_r1 \
  --report "$V26_ART/parent_selection.json" \
  --coupling-rank 16

actual_microbatch="$($PYTHON - "$V26_CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["microbatch_records"])
PY
)"
if [[ "$actual_microbatch" != 8 ]]; then
  echo "generated physical microbatch is $actual_microbatch, expected 8"
  exit 4
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "$(date --iso-8601=seconds) starting v26 microbatch-8 two-update smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V26_CONFIG" --smoke-updates 2 \
  >>"$V26_ART/smoke_screen.log" 2>&1

"$PYTHON" - "$V26_ART/smoke/terminal.json" "$V26_ART/smoke/metrics.jsonl" <<'PY'
import json, sys
terminal = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
row = [item for item in rows if item.get("event") == "epoch"][-1]
gradients = row["gradient_norms"]
checks = {
    "complete": terminal.get("status") == "complete",
    "physical_microbatch_8": int(row["physical_microbatch_records"]) == 8,
    "gate_gradient": float(gradients.get("coupled_2d_gate", 0.0)) > 0.0,
    "feature_gradient": float(gradients.get("coupled_2d_features", 0.0)) > 0.0,
    "cuda_below_23_gib": int(row["peak_cuda_bytes"]) < 23 * 1024**3,
}
print(json.dumps({"checks": checks, "peak_cuda_bytes": row["peak_cuda_bytes"]}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(5)
PY

echo "$(date --iso-8601=seconds) starting v26 microbatch-8 two-epoch pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V26_CONFIG" --pilot \
  >>"$V26_ART/pilot_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_coupled_2d_candidate.py \
  --selection-report "$V26_ART/parent_selection.json" \
  --candidate-metrics "$V26_ART/pilot/metrics.jsonl" \
  --output "$V26_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  echo "$(date --iso-8601=seconds) v26 evidence gate rejected long training rc=$gate_rc"
  exit "$gate_rc"
fi

mkdir -p "$V27_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$V26_CONFIG" \
  --output-config "$V27_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v27_coupled_2d_long_bigbatch192_micro8_4gpu_remote_r1 \
  --report "$V27_ART/parent_selection.json" \
  --epochs 40

echo "$(date --iso-8601=seconds) starting gate-approved v27 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V27_CONFIG" \
  >>"$V27_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) v27 long training finished"
