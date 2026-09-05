#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
ART="$PROJECT/1/pretraining/saved_time_v69_dynamic_propagator_r1"
PARENT="$PROJECT/artifacts/saved_time_v49_family_experts_structural_prior_pilot_r1/pilot/metrics.jsonl"
PYTHON=/root/miniconda3/bin/python
STATUS="$ART/supervisor/postpilot_terminal.json"

mkdir -p "$ART/supervisor"
printf '%s\n' "$$" >"$ART/supervisor/postpilot.pid"
while [[ ! -s "$ART/pilot/terminal.json" ]]; do
  sleep 60
done

pilot_status=$(
  "$PYTHON" - "$ART/pilot/terminal.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf8")).get("status", "unknown"))
PY
)
if [[ "$pilot_status" != complete ]]; then
  printf '{"status":"pilot_failed"}\n' >"$STATUS"
  exit 0
fi

cd "$WORK"
set +e
"$PYTHON" scripts/gate_saved_time_v63_full_dataset.py \
  --parent-metrics "$PARENT" \
  --smoke-metrics "$ART/smoke/metrics.jsonl" \
  --smoke-terminal "$ART/smoke/terminal.json" \
  --pilot-metrics "$ART/pilot/metrics.jsonl" \
  --minimum-relative-improvement 0.02 \
  --output "$ART/pilot_evidence_gate.json" \
  >"$ART/supervisor/postpilot_gate.log" 2>&1
gate_rc=$?
set -e
if [[ $gate_rc -eq 0 ]]; then
  printf '{"status":"promotion_eligible","minimum_relative_improvement":0.02}\n' >"$STATUS"
else
  printf '{"status":"candidate_rejected","minimum_relative_improvement":0.02}\n' >"$STATUS"
fi
