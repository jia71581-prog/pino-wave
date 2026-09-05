#!/usr/bin/env bash
# Sequential, wait-only train-gate ablations. This script never signals,
# restarts, or changes the priority of any existing process.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
python_bin=/root/miniconda3/bin/python
torchrun_bin=/root/miniconda3/bin/torchrun
parent_checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag/run/checkpoints/epoch_0003.pt
summary_path="$project_root/results/r5efg_aligned_time_ablation_terminal_20260813.json"
supervisor_lock="$project_root/results/r5efg_aligned_time_ablation_supervisor.lock"
supervisor_pid="$project_root/results/r5efg_aligned_time_ablation_supervisor.pid"

names=(
  r5e_true_metric_aligned_time_train_diag
  r5f_uniform_random_aligned_time_train_diag
  r5g_dropout005_aligned_time_train_diag
)
configs=(
  configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5e_true_metric_aligned_time_train_diag.yaml
  configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5f_uniform_random_aligned_time_train_diag.yaml
  configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5g_dropout005_aligned_time_train_diag.yaml
)
preregs=(
  results/r5e_true_metric_aligned_time_train_preregistration_20260813.json
  results/r5f_uniform_random_aligned_time_train_preregistration_20260813.json
  results/r5g_dropout005_aligned_time_train_preregistration_20260813.json
)
artifacts=(
  /root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5e_true_metric_aligned_time_train_diag
  /root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5f_uniform_random_aligned_time_train_diag
  /root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5g_dropout005_aligned_time_train_diag
)

cd "$project_root"
printf '%s\n' "$$" >"$supervisor_pid"
exec 9>"$supervisor_lock"
if ! flock -n 9; then
  printf '%s\n' "another aligned-time ablation supervisor owns the lock" >&2
  exit 73
fi

write_summary() {
  local status=$1
  local stage=$2
  local current_run=$3
  local rc=$4
  "$python_bin" - "$summary_path" "$status" "$stage" "$current_run" "$rc" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "r5efg_aligned_time_ablation_supervisor_v1",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "current_run": sys.argv[4] or None,
    "exit_code": int(sys.argv[5]),
    "supervisor_pid": os.getppid(),
    "process_control_used": False,
    "updated_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

write_run_terminal() {
  local artifact=$1
  local name=$2
  local status=$3
  local rc=$4
  "$python_bin" - "$artifact/full_pipeline_terminal.json" "$name" "$status" "$rc" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "aligned_time_train_ablation_terminal_v1",
    "experiment": sys.argv[2],
    "status": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "process_control_used": False,
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
    write_summary failed supervisor "" "$rc" || true
  fi
}
trap on_exit EXIT

"$python_bin" - "${configs[@]}" -- "${preregs[@]}" <<'PY'
import hashlib, json, pathlib, sys, yaml

separator = sys.argv.index("--")
config_paths = [pathlib.Path(value) for value in sys.argv[1:separator]]
prereg_paths = [pathlib.Path(value) for value in sys.argv[separator + 1:]]
if len(config_paths) != 3 or len(prereg_paths) != 3:
    raise SystemExit("expected exactly three configs and preregistrations")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

trainer = pathlib.Path("scripts/train_saved_time_v4_full_support.py")
parent = pathlib.Path("/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag/run/checkpoints/epoch_0003.pt")
configs = []
for config_path, prereg_path in zip(config_paths, prereg_paths, strict=True):
    config = yaml.safe_load(config_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    bindings = prereg["bindings"]
    checks = {
        "config path": str(config_path) == bindings["config"],
        "config digest": sha256(config_path) == bindings["config_sha256"],
        "trainer path": str(trainer) == bindings["trainer"],
        "trainer digest": sha256(trainer) == bindings["trainer_sha256"],
        "parent path": str(parent) == bindings["parent_checkpoint"] == config["parent_checkpoint"],
        "parent digest": sha256(parent) == bindings["parent_checkpoint_sha256"],
        "train parent selection": config.get("parent_checkpoint_selection_split") == "train",
        "train-only gate": config.get("epoch_validation_control", {}).get("evaluation_split") == "train",
        "aligned time policy": config.get("time_policy") == "fixed_train_gate",
        "24 training frames": config.get("training_frames_per_record") == 24,
        "32 gate frames": config.get("validation", {}).get("frames_per_record") == 32,
        "zero selector offset": config.get("epoch_validation_control", {}).get("time_selector_seed_offset") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"{config_path}: preflight failed: {', '.join(failed)}")
    configs.append(config)

def flatten(value, prefix=""):
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            output.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return output
    return {prefix: value}

control = dict(configs[0]); control.pop("artifact_dir")
expected = [set(), {"adaptive_sampling.record_axis_strategy"}, {"variant_overrides.band_adapter_dropout"}]
for index, candidate in enumerate(configs):
    candidate = dict(candidate); candidate.pop("artifact_dir")
    left, right = flatten(control), flatten(candidate)
    changed = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    if changed != expected[index]:
        raise SystemExit(f"config {index} is not single-variable: {sorted(changed)}")
if configs[1]["adaptive_sampling"]["record_axis_strategy"] != "uniform":
    raise SystemExit("r5f is not uniform replay")
if configs[2]["variant_overrides"]["band_adapter_dropout"] != 0.05:
    raise SystemExit("r5g dropout is not 0.05")
PY

for artifact in "${artifacts[@]}"; do
  if [[ -e "$artifact/run/run_identity.json" ]]; then
    printf '%s\n' "$artifact already has a run identity; refusing overwrite" >&2
    exit 74
  fi
done

available_kib=$(df --output=avail /root/autodl-tmp | tail -1 | tr -d ' ')
if [[ -z "$available_kib" || "$available_kib" -lt 6291456 ]]; then
  printf '%s\n' "less than 6 GiB remains on /root/autodl-tmp; refusing long launches" >&2
  exit 75
fi

write_summary waiting exclusive_gpus "" 0
wait_check=0
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  if [[ "$wait_check" -ge 360 ]]; then
    printf '%s\n' "exclusive GPU wait limit exhausted" >&2
    exit 76
  fi
  wait_check=$((wait_check + 1))
  sleep 10
done

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

for index in 0 1 2; do
  name=${names[$index]}
  config=${configs[$index]}
  artifact=${artifacts[$index]}
  log_path="$artifact/four_gpu_train.log"
  mkdir -p "$artifact"
  write_summary running train "$name" 0
  set +e
  "$torchrun_bin" --standalone --nproc_per_node=4 \
    scripts/train_saved_time_v4_full_support.py \
    --config "$config" --keep-all-checkpoints --workers-override 0 \
    >>"$log_path" 2>&1 &
  training_pid=$!
  printf '%s\n' "$training_pid" >"$artifact/torchrun.pid"
  wait "$training_pid"
  training_rc=$?
  set -e

  classification=$(
    "$python_bin" - "$artifact" "$training_rc" <<'PY'
import json, pathlib, sys
artifact = pathlib.Path(sys.argv[1])
return_code = int(sys.argv[2])
terminal_path = artifact / "run" / "terminal.json"
control_path = artifact / "run" / "epoch_validation_control.json"
if return_code == 0:
    print("complete")
elif terminal_path.is_file() and control_path.is_file():
    terminal = json.loads(terminal_path.read_text())
    control = json.loads(control_path.read_text())
    rejection = control.get("last_rejection", {})
    if (
        terminal.get("error") == "epoch validation did not improve after the configured parameter retries"
        and rejection.get("maximum_attempts_exhausted") is True
    ):
        print("rejected")
    else:
        print("failed")
else:
    print("failed")
PY
  )
  write_run_terminal "$artifact" "$name" "$classification" "$training_rc"
  if [[ "$classification" == failed ]]; then
    printf '%s\n' "$name failed abnormally; later ablations were not launched" >&2
    exit "$training_rc"
  fi
done

write_summary complete complete "" 0
trap - EXIT
