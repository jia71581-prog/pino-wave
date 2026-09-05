#!/usr/bin/env bash
# User-authorized pivot from the interrupted A3 pilot to CPADC.  Preserve the
# epoch-1 parent, gate a small four-GPU smoke, train the offline error basis,
# then run the sealed all-validation deployment protocol.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2.yaml
parent=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1/pilot/best.pt
parent_sha256=395b83560db3342b1e2331de2d8dde2994640d56a53dfbec8c99ece4bbb177bf
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_bilevel_trust_a3e1_marmousi1_4m_v2_r5
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun

mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/pipeline.lock"
if ! flock -n 9; then
  echo "another CPADC pipeline owns the lock" >&2
  exit 73
fi

cd "$project_root"
stage=initializing

write_terminal() {
  local status=$1
  local rc=$2
  "$python_bin" - "$artifact_dir/pipeline_terminal.json" "$status" "$stage" "$rc" <<'PY'
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
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed "$rc" || true
  fi
}
trap on_exit EXIT

gpu_snapshot() {
  echo "$(date --iso-8601=seconds) stage=$stage"
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

stage=parent_verification
write_terminal running 0
"$python_bin" - "$parent" "$parent_sha256" <<'PY'
import hashlib, pathlib, sys, torch
path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != sys.argv[2]:
    raise ValueError(f"interrupted A3 parent hash mismatch: {digest}")
payload = torch.load(path, map_location="cpu", weights_only=False)
checks = {
    "format": payload.get("format") == "phase_aligned_complex_fno_mionet_v3",
    "epoch": int(payload.get("epoch", -1)) == 1,
    "global_step": int(payload.get("global_step", -1)) == 91,
    "model_state": len(payload.get("model_state") or {}) == 377,
    "manifest": payload.get("manifest_digest") == "55fbffa9a66b0cb547657d2d5cd8cc140c4f7d970e37f3828144e7778d182e09",
}
if not all(checks.values()):
    raise ValueError(f"interrupted A3 parent verification failed: {checks}")
print(checks)
PY

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

stage=smoke
write_terminal running 0
require_four_free_gpus
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_causal_defect_basis.py \
  --config "$config" \
  --output-dir "$artifact_dir/smoke" \
  --smoke --epochs 1 --rank 16 --phase-rank 4 --width 16 \
  --patch-size 64 --physics-point-count 128 --rank-chunk-size 2 \
  >"$artifact_dir/smoke_screen.log" 2>&1

"$python_bin" - "$artifact_dir/smoke" <<'PY'
import json, math, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
terminal = json.loads((root / "terminal.json").read_text())
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
checks = {
    "terminal_complete": terminal.get("status") == "complete",
    "one_epoch": len(rows) == 1 and int(rows[0].get("epoch", -1)) == 1,
    "finite_loss": len(rows) == 1 and math.isfinite(float(rows[0].get("mean_outer_loss", math.nan))),
    "inner_solve_accepted": len(rows) == 1 and float(rows[0].get("inner_acceptance_fraction", 0.0)) == 1.0,
    "checkpoint_schema": checkpoint.get("schema") == "causal_physics_aligned_defect_correction_v1",
    "schema_version": int(checkpoint.get("schema_version", 0)) == 4,
    "source_consistent_defect": (checkpoint.get("defect_contract") or {}).get("name") == "source_consistent_effective_saved_grid_lwc84_v1",
    "bounded_online_solve": (checkpoint.get("online_solve_contract") or {}).get("name") == "ridge_direction_learned_energy_ball_projection_v1",
    "future_truth_scope": checkpoint.get("future_truth_used_only_for_outer_train_loss") is True,
}
if not all(checks.values()):
    raise ValueError(f"CPADC smoke gate failed: {checks}")
print(checks)
PY

stage=offline_meta_training
write_terminal running 0
require_four_free_gpus
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
"$torchrun_bin" --standalone --nproc_per_node=4 \
  scripts/train_causal_defect_basis.py \
  --config "$config" \
  --output-dir "$artifact_dir/run" \
  >"$artifact_dir/run_screen.log" 2>&1

"$python_bin" - "$artifact_dir/run" <<'PY'
import json, pathlib, sys, torch
root = pathlib.Path(sys.argv[1])
terminal = json.loads((root / "terminal.json").read_text())
checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
best_epoch = int(terminal.get("best_epoch", -1))
best_rows = [row for row in rows if int(row.get("epoch", -1)) == best_epoch]
best_row = best_rows[0] if len(best_rows) == 1 else {}
checks = {
    "terminal_complete": terminal.get("status") == "complete",
    "hundred_epochs": len(rows) == 100 and 1 <= int(checkpoint.get("epoch", -1)) <= 100,
    "checkpoint_schema": checkpoint.get("schema") == "causal_physics_aligned_defect_correction_v1",
    "schema_version": int(checkpoint.get("schema_version", 0)) == 4,
    "differentiable_inner_solve": checkpoint.get("differentiable_inner_solve") is True,
    "parent_bound": bool(checkpoint.get("parent_checkpoint_sha256")),
    "manifest_bound": bool(checkpoint.get("manifest_digest")),
    "online_future_truth_unused": checkpoint.get("online_future_truth_used") is False,
    "train_patch_improved": bool(best_row) and float(best_row.get("relative_improvement", -1.0)) > 0.0,
}
if not all(checks.values()):
    raise ValueError(f"CPADC offline training verification failed: {checks}")
print(checks)
PY

stage=sealed_full_validation
write_terminal running 0
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
"$python_bin" scripts/run_causal_defect_adaptation.py \
  --config "$config" \
  --basis-checkpoint "$artifact_dir/run/best.pt" \
  --output-dir "$artifact_dir/evaluation" \
  --all-validation --no-fields --device cuda \
  >"$artifact_dir/evaluation_screen.log" 2>&1

stage=complete
gpu_snapshot >>"$artifact_dir/gpu_snapshots.log" 2>&1
"$python_bin" - "$artifact_dir/evaluation/terminal.json" "$artifact_dir/pipeline_terminal.json" <<'PY'
import json, os, pathlib, sys, time
evaluation = json.loads(pathlib.Path(sys.argv[1]).read_text())
if evaluation.get("status") != "complete":
    raise ValueError("sealed CPADC evaluation did not complete")
payload = {
    "status": "complete",
    "stage": "complete",
    "exit_code": 0,
    "pipeline_pid": os.getppid(),
    "updated_unix_s": time.time(),
    "same_protocol_validation_passed": evaluation.get("same_protocol_validation_passed") is True,
    "claim": evaluation.get("claim"),
}
path = pathlib.Path(sys.argv[2])
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
trap - EXIT
