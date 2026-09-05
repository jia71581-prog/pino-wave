#!/usr/bin/env bash
# Wait for the direct remote push, verify/prepare the repaired dataset, run a
# short DDP smoke and same-protocol pilot gate, then continue the 40-epoch run.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1.yaml
source_dir=/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1
expected_h5_count=127
expected_h5_bytes=39260071133
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun

mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/pipeline.lock"
if ! flock -n 9; then
  echo "another Marmousi1 4m v2 pipeline owns the lock" >&2
  exit 73
fi

cd "$project_root"
stage=initializing

write_pipeline_terminal() {
  local status=$1
  local rc=$2
  "$python_bin" - "$artifact_dir/pipeline_terminal.json" "$status" "$stage" "$rc" <<'PY'
import json, os, pathlib, sys, tempfile, time
path = pathlib.Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "pipeline_pid": os.getppid(),
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

on_exit() {
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_pipeline_terminal failed "$rc" || true
  fi
}
trap on_exit EXIT

gpu_snapshot() {
  echo "$(date --iso-8601=seconds) GPU snapshot stage=$stage"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader || true
}

require_four_free_gpus() {
  local gpu_count
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  if [[ "$gpu_count" -ne 4 ]]; then
    echo "expected four GPUs, observed $gpu_count" >&2
    exit 74
  fi
  local compute_pids
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ -n "$compute_pids" ]]; then
    echo "GPU compute processes appeared before stage $stage: $compute_pids" >&2
    exit 75
  fi
}

stage=waiting_for_transfer
write_pipeline_terminal running 0
while true; do
  read -r h5_count h5_bytes < <(
    find "$source_dir/shards" -mindepth 2 -maxdepth 2 -type f -name '*.h5' \
      -printf '%s\n' 2>/dev/null | awk '{s+=$1;n++} END {printf "%d %.0f\n",n,s}'
  )
  sidecar_count=$(find "$source_dir/shards" -mindepth 2 -maxdepth 2 -type f \
    -name '*.h5.sha256' 2>/dev/null | wc -l)
  echo "$(date --iso-8601=seconds) transfer h5=$h5_count/$expected_h5_count bytes=$h5_bytes/$expected_h5_bytes sidecars=$sidecar_count/$expected_h5_count"
  if [[ "$h5_count" -eq "$expected_h5_count" \
        && "$h5_bytes" -eq "$expected_h5_bytes" \
        && "$sidecar_count" -eq "$expected_h5_count" ]]; then
    break
  fi
  sleep 30
done

stage=preparing_data
write_pipeline_terminal running 0
"$python_bin" scripts/prepare_marmousi1_4m_v2_continuation.py \
  >"$artifact_dir/data_preparation.log" 2>&1

stage=audit_only
write_pipeline_terminal running 0
"$python_bin" scripts/train_saved_time_v4_full_support.py \
  --config "$config" --audit-only >"$artifact_dir/audit_only.log" 2>&1

available_kib=$(df --output=avail -k /root/autodl-tmp | tail -n 1 | tr -d ' ')
if [[ "$available_kib" -lt 20971520 ]]; then
  echo "less than 20 GiB free after data preparation" >&2
  exit 76
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

stage=smoke
write_pipeline_terminal running 0
require_four_free_gpus
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py --config "$config" \
  --smoke-updates 2 --smoke-epochs 3 >"$artifact_dir/smoke_screen.log" 2>&1

"$python_bin" - "$artifact_dir/smoke/terminal.json" \
  "$artifact_dir/smoke/metrics.jsonl" "$config" <<'PY'
import json, math, pathlib, sys, yaml
terminal = json.loads(pathlib.Path(sys.argv[1]).read_text())
rows = [json.loads(line) for line in pathlib.Path(sys.argv[2]).read_text().splitlines() if line.strip()]
epochs = [row for row in rows if row.get("event") == "epoch"]
config = yaml.safe_load(pathlib.Path(sys.argv[3]).read_text())
limit = float(config["gate"]["maximum_peak_cuda_gib"]) * 1024**3
checks = {
    "terminal_complete": terminal.get("status") == "complete",
    "three_epochs": len(epochs) == 3,
    "finite_losses": bool(epochs) and all(math.isfinite(float(row["train_loss"])) for row in epochs),
    "local_field_gradient_active": bool(epochs) and all(float(row["gradient_norms"].get("local_field", 0.0)) > 0.0 for row in epochs),
    "cuda_within_limit": bool(epochs) and max(int(row["peak_cuda_bytes"]) for row in epochs) <= limit,
    "four_rank_ddp": bool(epochs) and all(int(row["ddp"]["world_size"]) == 4 for row in epochs),
}
print(json.dumps({"checks": checks}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(77)
PY

stage=pilot
write_pipeline_terminal running 0
require_four_free_gpus
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
set +e
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py --config "$config" --pilot \
  >"$artifact_dir/pilot_screen.log" 2>&1
pilot_rc=$?
set -e
if [[ "$pilot_rc" -ne 0 ]]; then
  echo "same-protocol pilot failed with exit code $pilot_rc; full run is not authorized" >&2
  exit "$pilot_rc"
fi

"$python_bin" - "$artifact_dir/pilot/terminal.json" <<'PY'
import json, pathlib, sys
terminal = json.loads(pathlib.Path(sys.argv[1]).read_text())
gate = terminal.get("pilot_gate") or {}
checks = {
    "terminal_complete": terminal.get("status") == "complete",
    "pilot_gate_passed": gate.get("passed") is True,
    "family_safe": gate.get("family_safe") is True,
    "best_nonincreasing": gate.get("best_nonincreasing") is True,
}
print(json.dumps({"checks": checks, "pilot_gate": gate}, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(78)
PY

stage=full_run
write_pipeline_terminal running 0
require_four_free_gpus
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py --config "$config" \
  >"$artifact_dir/run_screen.log" 2>&1

stage=checkpoint_verification
write_pipeline_terminal running 0
"$python_bin" - "$artifact_dir/run" "$artifact_dir/checkpoint_verification.json" <<'PY'
import hashlib, json, os, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
terminal = json.loads((root / "terminal.json").read_text())
identity = json.loads((root / "run_identity.json").read_text())
latest = root / "latest.pt"
best = root / "best.pt"
if terminal.get("status") != "complete":
    raise ValueError(f"full-run terminal is not complete: {terminal.get('status')}")
if terminal.get("run_digest") != identity.get("run_digest"):
    raise ValueError("terminal/run identity digest mismatch")
if not latest.is_file() or not best.is_file():
    raise FileNotFoundError("latest.pt or best.pt is missing")
checkpoint = torch.load(latest, map_location="cpu")
checks = {
    "format": checkpoint.get("format") == "phase_aligned_complex_fno_mionet_v3",
    "epoch_complete": int(checkpoint.get("epoch", -1)) == 40,
    "config_digest": checkpoint.get("config_digest") == identity.get("run_digest"),
    "manifest_digest": checkpoint.get("manifest_digest") == identity.get("manifest_digest"),
    "model_state_nonempty": bool(checkpoint.get("model_state")),
    "optimizer_state_nonempty": bool(checkpoint.get("optimizer_state")),
}
if not all(checks.values()):
    raise ValueError(f"checkpoint reproducibility checks failed: {checks}")
digest = hashlib.sha256()
with latest.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
payload = {
    "status": "complete",
    "checks": checks,
    "checkpoint": str(latest.resolve()),
    "checkpoint_sha256": digest.hexdigest(),
    "checkpoint_bytes": latest.stat().st_size,
    "epoch": int(checkpoint["epoch"]),
    "global_step": int(checkpoint["global_step"]),
    "run_digest": identity["run_digest"],
    "manifest_digest": identity["manifest_digest"],
}
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY

stage=complete
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
write_pipeline_terminal complete 0
trap - EXIT
