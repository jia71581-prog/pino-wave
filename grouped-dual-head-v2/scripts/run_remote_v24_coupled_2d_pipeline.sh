#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
HOME_PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
V20_ART="$HOME_PROJECT/artifacts/saved_time_v20_all_modes_bigbatch192_4gpu_remote_r1"
V24_ART="$HOME_PROJECT/artifacts/saved_time_v24_coupled_2d_bigbatch192_4gpu_remote_r1"
V25_ART="$HOME_PROJECT/artifacts/saved_time_v25_coupled_2d_long_bigbatch192_4gpu_remote_r1"
V20_CONFIG=configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml
V24_CONFIG=configs/saved_time_v4/generated/v24_coupled_2d_bigbatch192_4gpu_remote.yaml
V25_CONFIG=configs/saved_time_v4/generated/v25_coupled_2d_long_bigbatch192_4gpu_remote.yaml
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun

mkdir -p "$V24_ART" "$WORK/configs/saved_time_v4/generated"
exec >>"$V24_ART/pipeline.log" 2>&1

finalize() {
  rc=$?
  "$PYTHON" - "$V24_ART/pipeline_terminal.json" "$rc" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w") as handle:
    json.dump(
        {
            "status": "complete" if code == 0 else "failed",
            "return_code": code,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
os.replace(partial, path)
PY
}
trap finalize EXIT

cd "$WORK"
echo "$(date --iso-8601=seconds) waiting only for the first complete v20 fixed-panel epoch"
until "$PYTHON" - "$V20_ART/pilot/metrics.jsonl" <<'PY'
import json
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
for line in path.read_text().splitlines():
    row = json.loads(line)
    checkpoint = pathlib.Path(str(row.get("checkpoint", "")))
    if (
        row.get("event") == "epoch"
        and row.get("validation_scope") == "pilot_fixed_panel"
        and int(row.get("epoch", 0)) >= 1
        and checkpoint.is_file()
    ):
        raise SystemExit(0)
raise SystemExit(1)
PY
do
  sleep 5
done

echo "$(date --iso-8601=seconds) v20 epoch checkpoint available; stopping superseded pipeline"
screen -S fno_v20_allmodes_pipeline -X quit 2>/dev/null || true
pkill -TERM -f 'train_saved_time_v4_full_support.py --config configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml' 2>/dev/null || true
for _ in $(seq 1 30); do
  if ! pgrep -f 'train_saved_time_v4_full_support.py --config configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml' >/dev/null; then
    break
  fi
  sleep 1
done
if pgrep -f 'train_saved_time_v4_full_support.py --config configs/saved_time_v4/generated/v20_all_modes_bigbatch192_4gpu_remote.yaml' >/dev/null; then
  echo "$(date --iso-8601=seconds) v20 workers did not stop cleanly"
  exit 3
fi

"$PYTHON" scripts/prepare_saved_time_coupled_2d_candidate.py \
  --candidate-config "$V20_CONFIG" \
  --output-config "$V24_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v24_coupled_2d_bigbatch192_4gpu_remote_r1 \
  --report "$V24_ART/parent_selection.json" \
  --coupling-rank 16

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "$(date --iso-8601=seconds) starting v24 four-GPU two-update smoke"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V24_CONFIG" --smoke-updates 2 \
  >>"$V24_ART/smoke_screen.log" 2>&1

"$PYTHON" - "$V24_ART/smoke/terminal.json" "$V24_ART/smoke/metrics.jsonl" <<'PY'
import json
import sys
terminal = json.load(open(sys.argv[1]))
rows = [json.loads(line) for line in open(sys.argv[2]) if line.strip()]
epochs = [row for row in rows if row.get("event") == "epoch"]
if terminal.get("status") != "complete" or not epochs:
    raise SystemExit("v24 smoke did not complete")
row = epochs[-1]
gradients = row.get("gradient_norms", {})
if float(gradients.get("coupled_2d_gate", 0.0)) <= 0.0:
    raise SystemExit("v24 coupled gate gradient is missing")
if float(gradients.get("coupled_2d_features", 0.0)) <= 0.0:
    raise SystemExit("v24 coupled feature gradient is missing after update two")
if int(row.get("peak_cuda_bytes", 1 << 70)) >= 23 * 1024**3:
    raise SystemExit("v24 smoke exceeded the CUDA memory gate")
print(json.dumps({
    "peak_cuda_bytes": row["peak_cuda_bytes"],
    "gradient_norms": gradients,
    "physical_microbatch_records": row["physical_microbatch_records"],
}, sort_keys=True))
PY

echo "$(date --iso-8601=seconds) starting v24 two-epoch fixed-panel pilot"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V24_CONFIG" --pilot \
  >>"$V24_ART/pilot_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_coupled_2d_candidate.py \
  --selection-report "$V24_ART/parent_selection.json" \
  --candidate-metrics "$V24_ART/pilot/metrics.jsonl" \
  --output "$V24_ART/evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  echo "$(date --iso-8601=seconds) v24 evidence gate rejected long training rc=$gate_rc"
  exit "$gate_rc"
fi

mkdir -p "$V25_ART"
"$PYTHON" scripts/prepare_saved_time_long_continuation.py \
  --candidate-config "$V24_CONFIG" \
  --output-config "$V25_CONFIG" \
  --artifact-dir /home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/saved_time_v25_coupled_2d_long_bigbatch192_4gpu_remote_r1 \
  --report "$V25_ART/parent_selection.json" \
  --epochs 40

echo "$(date --iso-8601=seconds) starting gate-approved v25 long training"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$V25_CONFIG" \
  >>"$V25_ART/long_screen.log" 2>&1
echo "$(date --iso-8601=seconds) v25 long training finished"
