#!/usr/bin/env bash
# Launch or resume only the authorized 40-epoch A3-WARM full run.
# The completed smoke/pilot artifacts are read for authorization and preserved.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1.yaml
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
terminal_path="$artifact_dir/full_pipeline_terminal.json"

mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/pipeline.lock"
if ! flock -n 9; then
  echo "another Marmousi1 4m v2 pipeline owns the lock" >&2
  exit 73
fi

cd "$project_root"
stage=initializing

write_terminal() {
  local status=$1
  local rc=$2
  "$python_bin" - "$terminal_path" "$status" "$stage" "$rc" <<'PY'
import json, os, pathlib, sys, time
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
    write_terminal failed "$rc" || true
  fi
}
trap on_exit EXIT

stage=authorization
write_terminal running 0
"$python_bin" - "$artifact_dir/pilot/terminal.json" "$config" <<'PY'
import json, pathlib, sys, yaml
terminal = json.loads(pathlib.Path(sys.argv[1]).read_text())
config = yaml.safe_load(pathlib.Path(sys.argv[2]).read_text())
gate = terminal.get("pilot_gate") or {}
checks = {
    "pilot_complete": terminal.get("status") == "complete",
    "pilot_gate_passed": gate.get("passed") is True,
    "pilot_family_safe": gate.get("family_safe") is True,
    "registered_epochs_40": int(config.get("epochs", 0)) == 40,
    "training_frames_24": int(config.get("training_frames_per_record", 0)) == 24,
    "peak_limit_23p5_gib": float(config.get("gate", {}).get("maximum_peak_cuda_gib", 0.0)) == 23.5,
    "epoch_validation_control": bool(
        config.get("epoch_validation_control", {}).get("enabled", False)
    ),
    "strict_aggregate_relative_l2": (
        config.get("epoch_validation_control", {}).get("metric")
        == "aggregate_relative_l2"
        and float(
            config.get("epoch_validation_control", {}).get(
                "minimum_absolute_improvement", -1.0
            )
        ) == 0.0
    ),
}
print(json.dumps({"checks": checks}, sort_keys=True), flush=True)
if not all(checks.values()):
    raise SystemExit(78)
PY

available_kib=$(df --output=avail -k /root/autodl-tmp | tail -n 1 | tr -d ' ')
if [[ "$available_kib" -lt 10485760 ]]; then
  echo "less than 10 GiB free before the full run" >&2
  exit 76
fi

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$gpu_count" -ne 4 ]]; then
  echo "expected four GPUs, observed $gpu_count" >&2
  exit 74
fi

stage=waiting_for_gpus
write_terminal running 0
while true; do
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ -z "$compute_pids" ]]; then
    break
  fi
  echo "$(date --iso-8601=seconds) waiting for GPU processes: $(tr '\n' ' ' <<<"$compute_pids")"
  sleep 30
done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

stage=full_run
write_terminal running 0
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_saved_time_v4_full_support.py --config "$config" --keep-all-checkpoints \
  --workers-override 0 \
  >>"$artifact_dir/run_screen.log" 2>&1

stage=checkpoint_verification
write_terminal running 0
"$python_bin" - "$artifact_dir/run" "$artifact_dir/checkpoint_verification_full.json" <<'PY'
import hashlib, json, os, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
terminal = json.loads((root / "terminal.json").read_text())
identity = json.loads((root / "run_identity.json").read_text())
control = json.loads((root / "epoch_validation_control.json").read_text())
metric_rows = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
control_rows = [
    json.loads(line)
    for line in (root / "epoch_validation_control.jsonl").read_text().splitlines()
    if line.strip()
]
latest = root / "latest.pt"
best = root / "best.pt"
if terminal.get("status") != "complete":
    raise ValueError(f"full-run terminal is not complete: {terminal.get('status')}")
if terminal.get("run_digest") != identity.get("run_digest"):
    raise ValueError("terminal/run identity digest mismatch")
if not latest.is_file() or not best.is_file():
    raise FileNotFoundError("latest.pt or best.pt is missing")
checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
baseline_rows = [row for row in control_rows if row.get("event") == "validation_baseline"]
accepted_rows = [row for row in control_rows if row.get("event") == "epoch_accepted"]
scores = (
    [float(baseline_rows[0]["score"])]
    + [float(row["metrics"]["aggregate_relative_l2"]) for row in metric_rows]
    if len(baseline_rows) == 1
    else []
)
checks = {
    "format": checkpoint.get("format") == "phase_aligned_complex_fno_mionet_v3",
    "epoch_complete": int(checkpoint.get("epoch", -1)) == 40,
    "config_digest": checkpoint.get("config_digest") == identity.get("run_digest"),
    "manifest_digest": checkpoint.get("manifest_digest") == identity.get("manifest_digest"),
    "model_state_nonempty": bool(checkpoint.get("model_state")),
    "optimizer_state_nonempty": bool(checkpoint.get("optimizer_state")),
    "epoch_gate_state_complete": (
        int(control.get("accepted_epoch", -1)) == 40
        and int(control.get("next_attempt", -1)) == 1
    ),
    "one_accepted_metric_per_epoch": (
        [int(row.get("epoch", -1)) for row in metric_rows] == list(range(1, 41))
        and [int(row.get("epoch", -1)) for row in accepted_rows] == list(range(1, 41))
    ),
    "same_protocol_every_epoch": all(
        row.get("validation_scope") == "fixed_epoch_gate"
        and row.get("epoch_validation_gate_passed") is True
        for row in metric_rows
    ),
    "aggregate_relative_l2_strictly_decreased": (
        len(scores) == 41
        and all(current < previous for previous, current in zip(scores, scores[1:]))
    ),
    "full_time_audits_present": all(
        metric_rows[epoch - 1].get("full_time_audit_metrics") is not None
        for epoch in range(4, 41, 4)
    ),
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
print(json.dumps(payload, sort_keys=True), flush=True)
PY

stage=complete
write_terminal complete 0
trap - EXIT
