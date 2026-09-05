#!/usr/bin/env bash
set -uo pipefail

ROOT=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
cd "$ROOT" || exit 1
UPSTREAM=results/transfer_dg_flux_pretrain_fast_supervisor_20260902/terminal.json
PREREG=results/transfer_dg_local_dtn_preregistration_20260902.json
LOG=results/transfer_dg_local_dtn_queue_20260902.log
while [[ ! -f "$UPSTREAM" ]]; do
  sleep 30
done
python - "$UPSTREAM" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
if row.get("status") != "complete":
    raise SystemExit("upstream Transfer DG flux supervisor did not complete")
PY
exec bash scripts/run_transfer_dg_local_dtn_four_lane.sh "$PREREG" >> "$LOG" 2>&1
