#!/usr/bin/env bash
# One-shot frozen R7 confirmation on the previously unused complete test_id
# split. The checkpoint, sample census, thresholds, code, config, travel cache,
# and promotion gate are all bound by the preregistration JSON.
set -Eeuo pipefail

project_root=/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
config=configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2.yaml
checkpoint=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7/calibration/calibrated.pt
checkpoint_sha256=6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b
preregistration=results/cpadc_r7_test_id_preregistration_20260810.json
preregistration_sha256=8d55119f034dcfd536696c5c3d5eb3fc880af256ed8527ed96824dcc08935267
travel_cache=/root/autodl-tmp/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2_test_id.h5
travel_cache_sha256=0149ef9d29b55d45caf29c477518070413781d88980db553b69dbbf276dbb388
travel_content_sha256=fa1a4d8bf5b017bbdef588197e9108a5975423401261416deaae6e9dc58b0064
artifact_dir=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7_test_id_confirmation
python_bin=/root/miniconda3/bin/python
shard_count=4
stage=initializing
children=()

mkdir -p "$artifact_dir"
exec 9>"$artifact_dir/pipeline.lock"
if ! flock -n 9; then
  echo "another frozen test_id confirmation owns the lock" >&2
  exit 73
fi
cd "$project_root"

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
  for pid in "${children[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$rc" -ne 0 ]]; then
    write_terminal failed "$rc" || true
  fi
}
trap on_exit EXIT

require_four_free_gpus() {
  local gpu_count
  local compute_pids
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  if [[ "$gpu_count" -ne "$shard_count" || -n "$compute_pids" ]]; then
    echo "frozen confirmation requires four free GPUs; count=$gpu_count pids=$compute_pids" >&2
    exit 74
  fi
}

write_children() {
  "$python_bin" - "$artifact_dir/test_id_children.json" "$stage" "$@" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "controller_pid": os.getppid(),
    "stage": sys.argv[2],
    "child_pids": [int(value) for value in sys.argv[3:]],
    "started_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

stage=frozen_protocol_verification
write_terminal running 0
if find "$artifact_dir" -mindepth 3 -name evaluation.json -type f -print -quit | grep -q .; then
  echo "test_id future truth was already opened in this artifact directory" >&2
  exit 75
fi
if [[ -f "$artifact_dir/evaluation/terminal.json" || -f "$artifact_dir/test_id_opening_authorized.json" ]]; then
  echo "frozen test_id confirmation is one-shot and cannot be resumed or repeated" >&2
  exit 76
fi
if [[ "$(sha256sum "$checkpoint" | awk '{print $1}')" != "$checkpoint_sha256" ]]; then
  echo "frozen R7 checkpoint hash mismatch" >&2
  exit 77
fi
if [[ "$(sha256sum "$preregistration" | awk '{print $1}')" != "$preregistration_sha256" ]]; then
  echo "test_id preregistration hash mismatch" >&2
  exit 78
fi
if [[ "$(sha256sum "$travel_cache" | awk '{print $1}')" != "$travel_cache_sha256" ]]; then
  echo "test_id travel cache hash mismatch" >&2
  exit 79
fi

"$python_bin" - "$preregistration" "$checkpoint" "$travel_cache" "$config" <<'PY'
import hashlib, h5py, json, pathlib, sys, torch
from grouped_ufno_mionet_v3.data.index import build_manifest
from scripts.run_causal_defect_adaptation import build_complete_evaluation_manifest

prereg_path, checkpoint_path, travel_path, config_path = map(pathlib.Path, sys.argv[1:])
prereg = json.loads(prereg_path.read_text())

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = build_manifest(prereg["source_h5"])
rows = build_complete_evaluation_manifest(manifest, split="test_id")
sample_digest = hashlib.sha256(
    json.dumps(
        [row.sample_id for row in rows], sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
contract = checkpoint.get("online_solve_contract") or {}
with h5py.File(travel_path, "r", swmr=True) as travel:
    travel_checks = {
        "record_count": int(travel.attrs.get("record_count", 0)) == 480,
        "test_id_only": str(travel.attrs.get("splits")) == "test_id",
        "content_digest": str(travel.attrs.get("content_sha256"))
        == prereg["travel_cache"]["content_sha256"],
    }
checks = {
    "registered_before_inference": prereg.get("status")
    == "registered_before_first_test_id_inference",
    "no_previous_test_id_artifacts": int(
        prereg.get("previous_cpadc_test_id_artifact_hits", -1)
    )
    == 0,
    "manifest_digest": manifest.digest == prereg["manifest_digest"],
    "sample_count": len(rows) == int(prereg["sample_count"]) == 480,
    "sample_digest": sample_digest == prereg["sample_ids_sha256"],
    "checkpoint_digest": sha256(checkpoint_path) == prereg["checkpoint_sha256"],
    "config_digest": sha256(config_path) == prereg["config_sha256"],
    "evaluator_digest": sha256(pathlib.Path("scripts/run_causal_defect_adaptation.py"))
    == prereg["evaluator_sha256"],
    "merger_digest": sha256(pathlib.Path("scripts/merge_causal_defect_evaluations.py"))
    == prereg["merger_sha256"],
    "family_contract": contract.get("name")
    == "ridge_direction_family_calibrated_strength_abstention_v1",
    "family_floors": contract.get(
        "minimum_unconstrained_correction_ratio_by_family"
    )
    == prereg["strength_floor_by_family"],
    "travel_cache": all(travel_checks.values()),
}
if not all(checks.values()):
    raise ValueError(f"frozen test_id protocol verification failed: {checks}")
print({"checks": checks, "travel_checks": travel_checks})
PY

require_four_free_gpus
stage=sealed_complete_test_id_4way
write_terminal running 0
"$python_bin" - "$artifact_dir/test_id_opening_authorized.json" "$preregistration_sha256" <<'PY'
import json, os, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "cpadc_test_id_opening_authorization_v1",
    "preregistration_sha256": sys.argv[2],
    "controller_pid": os.getppid(),
    "authorized_unix_s": time.time(),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
mkdir -p "$artifact_dir/test_id_shards"
for shard_index in 0 1 2 3; do
  shard_dir="$artifact_dir/test_id_shards/shard_$shard_index"
  mkdir -p "$shard_dir"
  CUDA_VISIBLE_DEVICES="$shard_index" "$python_bin" \
    scripts/run_causal_defect_adaptation.py \
    --config "$config" \
    --basis-checkpoint "$checkpoint" \
    --travel-time-h5 "$travel_cache" \
    --output-dir "$shard_dir" \
    --all-test-id \
    --shard-index "$shard_index" \
    --shard-count "$shard_count" \
    --no-fields --device cuda \
    >"$artifact_dir/test_id_shards/shard_${shard_index}.log" 2>&1 &
  children+=("$!")
done
write_children "${children[@]}"
for pid in "${children[@]}"; do
  wait "$pid"
done
children=()

stage=merge_and_gate_test_id
write_terminal running 0
merge_args=()
for shard_index in 0 1 2 3; do
  merge_args+=(--shard-dir "$artifact_dir/test_id_shards/shard_$shard_index")
done
"$python_bin" scripts/merge_causal_defect_evaluations.py \
  --config "$config" \
  --evaluation-split test_id \
  --output-dir "$artifact_dir/evaluation" \
  "${merge_args[@]}" \
  >"$artifact_dir/merge.log" 2>&1

stage=complete
"$python_bin" - "$artifact_dir/evaluation/terminal.json" "$artifact_dir/pipeline_terminal.json" "$preregistration_sha256" <<'PY'
import json, os, pathlib, sys, time
evaluation = json.loads(pathlib.Path(sys.argv[1]).read_text())
if evaluation.get("status") != "complete" or evaluation.get("evaluation_split") != "test_id":
    raise ValueError("frozen test_id evaluation is incomplete")
payload = {
    "status": "complete",
    "stage": "complete",
    "exit_code": 0,
    "pipeline_pid": os.getppid(),
    "updated_unix_s": time.time(),
    "evaluation_split": "test_id",
    "record_count": 480,
    "shard_count": 4,
    "checkpoint_sha256": "6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b",
    "preregistration_sha256": sys.argv[3],
    "same_protocol_evaluation_passed": evaluation.get("same_protocol_evaluation_passed") is True,
    "claim": evaluation.get("claim"),
}
path = pathlib.Path(sys.argv[2])
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
trap - EXIT
