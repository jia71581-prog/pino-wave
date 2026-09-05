#!/usr/bin/env bash
# Wait-only evaluation handoff. It never controls r5c/r5d processes.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
r5d_artifact=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5d_metric_aligned_time_train_diag
r5b_run=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag/run
output_root=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/evaluation/marmousi_fixed19hz_position_240_post_r5d_r1
preregistration=results/marmousi_fixed19hz_position_240_post_r5d_preregistration_20260813.json
protocol=paper/tgrs_helmholtz_operator/marmousi_fixed_frequency_source_ood_protocol_20260813.json
evaluation_script=scripts/evaluate_marmousi_source_position_control.py
python_bin=/root/miniconda3/bin/python
terminal_path="$output_root/pipeline_terminal.json"

mkdir -p "$output_root"
printf '%s\n' "$$" >"$output_root/supervisor.pid"
exec 9>"$output_root/supervisor.lock"
if ! flock -n 9; then
  printf '%s\n' "another position-evaluation supervisor owns the lock" >&2
  exit 73
fi
cd "$project_root"

write_terminal() {
  local status=$1
  local stage=$2
  local rc=$3
  "$python_bin" - "$terminal_path" "$status" "$stage" "$rc" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "post_r5d_position_evaluation_supervisor_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "supervisor_pid": os.getppid(),
    "training_process_control_used": False,
    "fixed_source_frequency_hz": 19.0,
    "frequency_generalization_claim_permitted": False,
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed supervisor "$rc" || true
  fi
}
trap on_exit EXIT

if [[ -e "$output_root/prediction_manifest.json" || -e "$output_root/reference_manifest.json" || -e "$output_root/score_summary.json" ]]; then
  printf '%s\n' "position-evaluation output already exists; refusing overwrite" >&2
  exit 74
fi

"$python_bin" - "$preregistration" "$protocol" "$evaluation_script" <<'PY'
import hashlib, json, pathlib, sys
prereg_path, protocol_path, script_path = map(pathlib.Path, sys.argv[1:])
p = json.loads(prereg_path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
checks = {
    "protocol": (protocol_path, p["protocol_sha256"]),
    "evaluation_script": (script_path, p["evaluation_script_sha256"]),
    "r5b_config": (pathlib.Path(p["r5b_fallback"]["config"]), p["r5b_fallback"]["config_sha256"]),
    "r5b_checkpoint": (pathlib.Path(p["r5b_fallback"]["checkpoint"]), p["r5b_fallback"]["checkpoint_sha256"]),
    "r5b_identity": (pathlib.Path(p["r5b_fallback"]["checkpoint_identity"]), p["r5b_fallback"]["checkpoint_identity_sha256"]),
    "r5d_config": (pathlib.Path(p["r5d_candidate"]["config"]), p["r5d_candidate"]["config_sha256"]),
    "r5d_preregistration": (pathlib.Path(p["r5d_candidate"]["training_preregistration"]), p["r5d_candidate"]["training_preregistration_sha256"]),
}
for name, (path, expected) in checks.items():
    if not path.is_file() or digest(path) != expected:
        raise SystemExit(f"preregistered artifact changed: {name}")
PY

write_terminal waiting r5d_natural_exit 0
wait_check=0
while true; do
  ready=$(
    "$python_bin" - "$r5d_artifact" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
terminal_path = root / "full_pipeline_terminal.json"
if not terminal_path.is_file():
    print("no")
    raise SystemExit
terminal = json.loads(terminal_path.read_text())
if terminal.get("status") not in {"complete", "failed"}:
    print("no")
    raise SystemExit
pid_path = root / "torchrun.pid"
if pid_path.is_file():
    value = pid_path.read_text().strip()
    if value.isdigit() and (pathlib.Path("/proc") / value).exists():
        print("no")
        raise SystemExit
if terminal.get("status") == "failed":
    history_path = root / "run" / "epoch_validation_control.jsonl"
    if not history_path.is_file():
        print("invalid")
        raise SystemExit
    rows = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    rejected = [row for row in rows if row.get("event") == "epoch_rejected"]
    if not rejected or rejected[-1].get("maximum_attempts_exhausted") is not True:
        print("invalid")
        raise SystemExit
print("yes")
PY
  )
  if [[ "$ready" == yes ]]; then break; fi
  if [[ "$ready" == invalid ]]; then
    printf '%s\n' "r5d failed for a reason other than natural gate exhaustion" >&2
    exit 75
  fi
  if [[ "$wait_check" -ge 8640 ]]; then
    printf '%s\n' "bounded wait for r5d natural exit exhausted" >&2
    exit 76
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

write_terminal waiting exclusive_gpus_after_r5d 0
wait_check=0
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  if [[ "$wait_check" -ge 360 ]]; then
    printf '%s\n' "GPUs did not become exclusively free after r5d" >&2
    exit 77
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

available_bytes=$(df -B1 --output=avail "$output_root" | tail -1 | tr -d '[:space:]')
minimum_available_bytes=$((22 * 1024 * 1024 * 1024))
if [[ ! "$available_bytes" =~ ^[0-9]+$ ]] || [[ "$available_bytes" -lt "$minimum_available_bytes" ]]; then
  printf '%s\n' "less than 22 GiB is available; refusing the sealed 240-record evaluation" >&2
  exit 78
fi

selection_json="$output_root/checkpoint_selection.json"
eval "$($python_bin - "$r5d_artifact" "$r5b_run" "$preregistration" "$selection_json" <<'PY'
import hashlib, json, pathlib, shlex, sys
r5d, r5b, prereg_path, output = map(pathlib.Path, sys.argv[1:])
prereg = json.loads(prereg_path.read_text())
history = r5d / "run" / "epoch_validation_control.jsonl"
rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
accepted = [row for row in rows if row.get("event") == "epoch_accepted"]
if accepted:
    config = pathlib.Path(prereg["r5d_candidate"]["config"])
    checkpoint = r5d / "run" / "best.pt"
    identity = r5d / "run" / "run_identity.json"
    source = "r5d_train_gate_best"
    accepted_score = min(float(row["score"]) for row in accepted)
else:
    config = pathlib.Path(prereg["r5b_fallback"]["config"])
    checkpoint = pathlib.Path(prereg["r5b_fallback"]["checkpoint"])
    identity = pathlib.Path(prereg["r5b_fallback"]["checkpoint_identity"])
    source = "r5b_preregistered_fallback"
    accepted_score = 0.2772543673381241
for path in (config, checkpoint, identity):
    if not path.is_file(): raise SystemExit(f"selected artifact missing: {path}")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if source == "r5b_preregistered_fallback" and digest(checkpoint) != prereg["r5b_fallback"]["checkpoint_sha256"]:
    raise SystemExit("r5b fallback checkpoint changed")
payload = {
    "schema": "post_r5d_train_gate_checkpoint_selection_v1",
    "selection_split": "train",
    "selection_source": source,
    "accepted_r5d_epoch_count": len(accepted),
    "accepted_train_gate_score": accepted_score,
    "config": str(config.resolve()),
    "config_sha256": digest(config),
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": digest(checkpoint),
    "checkpoint_identity": str(identity.resolve()),
    "checkpoint_identity_sha256": digest(identity),
}
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print("selected_config=" + shlex.quote(str(config.resolve())))
print("selected_checkpoint=" + shlex.quote(str(checkpoint.resolve())))
print("selected_identity=" + shlex.quote(str(identity.resolve())))
PY
)"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
write_terminal running sealed_prediction 0
"$python_bin" "$evaluation_script" predict \
  --config "$selected_config" \
  --checkpoint "$selected_checkpoint" \
  --checkpoint-identity "$selected_identity" \
  --protocol "$protocol" \
  --output-dir "$output_root" \
  --device cuda:0 --time-block 16 \
  >"$output_root/predict.log" 2>&1

write_terminal running sealed_reference_generation 0
"$python_bin" "$evaluation_script" generate-reference \
  --prediction-manifest "$output_root/prediction_manifest.json" \
  --protocol "$protocol" \
  --output-dir "$output_root" \
  --device cuda:0 --batch-size 2 --reproduction-tolerance 1e-5 \
  >"$output_root/generate_reference.log" 2>&1

write_terminal running complete_transient_scoring 0
CUDA_VISIBLE_DEVICES='' "$python_bin" "$evaluation_script" score \
  --prediction-manifest "$output_root/prediction_manifest.json" \
  --reference-manifest "$output_root/reference_manifest.json" \
  --protocol "$protocol" \
  --output-dir "$output_root" \
  >"$output_root/score.log" 2>&1

write_terminal complete complete 0
trap - EXIT
