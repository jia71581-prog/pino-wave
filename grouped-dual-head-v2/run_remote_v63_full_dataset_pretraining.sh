#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
PROJECT=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation
PYTHON=/root/miniconda3/bin/python
TORCHRUN=/root/miniconda3/bin/torchrun
PILOT_CONFIG=configs/saved_time_v4/generated/v63_full_dataset_allband_pilot_4gpu.yaml
LONG_CONFIG=configs/saved_time_v4/generated/v64_full_dataset_allband_long_4gpu.yaml
ART="$PROJECT/1/pretraining/saved_time_v63_full_dataset_allband_r1"
SUP="$ART/supervisor"
PARENT_METRICS="$PROJECT/artifacts/saved_time_v49_family_experts_structural_prior_pilot_r1/pilot/metrics.jsonl"

mkdir -p "$SUP"
if [[ -s "$SUP/pipeline_terminal.json" ]]; then
  mv "$SUP/pipeline_terminal.json" \
    "$SUP/pipeline_terminal.superseded.$(date +%Y%m%dT%H%M%S).json"
fi
exec >>"$SUP/pipeline.log" 2>&1

gpu_monitor_pid=""
atomic_status() {
  local path=$1
  local status=$2
  local code=$3
  "$PYTHON" - "$path" "$status" "$code" <<'PY'
import json, os, sys, time
path, status, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = {"status": status, "return_code": code, "unix_time": time.time()}
partial = f"{path}.partial.{os.getpid()}"
with open(partial, "w", encoding="utf8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(partial, path)
PY
}

cleanup() {
  local rc=$?
  if [[ -n "$gpu_monitor_pid" ]]; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
  fi
  if [[ ! -s "$SUP/pipeline_terminal.json" ]]; then
    if [[ $rc -eq 0 ]]; then
      atomic_status "$SUP/pipeline_terminal.json" complete "$rc"
    else
      atomic_status "$SUP/pipeline_terminal.json" failed "$rc"
    fi
  fi
}
trap cleanup EXIT

monitor_gpu() {
  local output=$1
  printf 'timestamp,index,name,memory_used_mib,memory_total_mib,utilization_percent,power_w,power_limit_w\n' >"$output"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit \
      --format=csv,noheader,nounits >>"$output" 2>&1 || true
    sleep 1
  done
}

cd "$WORK"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
ulimit -n 65536

printf '%s V63 pipeline start\n' "$(date --iso-8601=seconds)"
printf '%s\n' "$$" >"$SUP/pipeline.pid"
printf '%s\n' "$(ps -o pgid= -p $$ | tr -d ' ')" >"$SUP/pipeline.pgid"
nvidia-smi --query-gpu=index,name,memory.total,power.limit --format=csv >"$SUP/gpu_inventory.csv"

"$PYTHON" -m pytest -q \
  tests/saved_time_phase_operator_v4/test_v63_full_dataset_pretraining.py \
  tests/saved_time_phase_operator_v4/test_full_support_runner.py \
  >"$SUP/preflight_pytest.log" 2>&1
"$PYTHON" -m compileall -q \
  scripts/train_saved_time_v4_full_support.py \
  scripts/gate_saved_time_v63_full_dataset.py
"$PYTHON" scripts/train_saved_time_v4_full_support.py \
  --config "$PILOT_CONFIG" --audit-only >"$SUP/pilot_schedule_audit.json"

monitor_gpu "$SUP/gpu_samples.csv" &
gpu_monitor_pid=$!

printf '%s starting four-GPU two-update smoke\n' "$(date --iso-8601=seconds)"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$PILOT_CONFIG" --smoke-updates 2 \
  >>"$ART/smoke_screen.log" 2>&1

"$PYTHON" - "$ART/smoke/terminal.json" "$ART/smoke/metrics.jsonl" <<'PY'
import json, math, sys
terminal = json.load(open(sys.argv[1], encoding="utf8"))
rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf8") if line.strip()]
row = [value for value in rows if value.get("event") == "epoch"][-1]
gradients = row.get("gradient_norms", {})
checks = {
    "complete": terminal.get("status") == "complete",
    "two_updates": int(terminal.get("global_step", -1)) >= 2 and int(row.get("global_step", -1)) >= 2,
    "physical_microbatch_24": int(row.get("physical_microbatch_records", -1)) == 24,
    "four_gpu_world": int(row.get("ddp", {}).get("world_size", -1)) == 4,
    "global_macros_4": int(row.get("ddp", {}).get("global_macros_per_update", -1)) == 4,
    "finite_loss": math.isfinite(float(row.get("train_loss", float("nan")))),
    "adapter_gradient_active": float(gradients.get("dense_decoder.band_limited_adapter", 0.0)) > 0.0,
    "cuda_below_23_gib": 0 < int(row.get("peak_cuda_bytes", 0)) < 23 * 1024**3,
}
print(json.dumps({"checks": checks, "peak_cuda_bytes": row.get("peak_cuda_bytes")}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(5)
PY

printf '%s starting six-epoch full-dataset pilot\n' "$(date --iso-8601=seconds)"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$PILOT_CONFIG" --pilot \
  >>"$ART/pilot_screen.log" 2>&1

set +e
"$PYTHON" scripts/gate_saved_time_v63_full_dataset.py \
  --parent-metrics "$PARENT_METRICS" \
  --smoke-metrics "$ART/smoke/metrics.jsonl" \
  --smoke-terminal "$ART/smoke/terminal.json" \
  --pilot-metrics "$ART/pilot/metrics.jsonl" \
  --output "$ART/pilot_evidence_gate.json"
gate_rc=$?
set -e
if [[ $gate_rc -ne 0 ]]; then
  atomic_status "$SUP/pipeline_terminal.json" candidate_rejected "$gate_rc"
  printf '%s pilot rejected; retained best checkpoint without unsupported promotion\n' "$(date --iso-8601=seconds)"
  exit 0
fi

"$PYTHON" scripts/train_saved_time_v4_full_support.py \
  --config "$LONG_CONFIG" --audit-only >"$SUP/long_schedule_audit.json"
printf '%s starting forty-epoch promoted continuation\n' "$(date --iso-8601=seconds)"
"$TORCHRUN" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py \
  --config "$LONG_CONFIG" \
  >>"$ART/long_screen.log" 2>&1

atomic_status "$SUP/pipeline_terminal.json" complete 0
printf '%s V63/V64 training pipeline complete\n' "$(date --iso-8601=seconds)"
