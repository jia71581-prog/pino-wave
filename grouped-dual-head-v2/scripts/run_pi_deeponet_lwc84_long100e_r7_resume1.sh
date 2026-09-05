#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKDIR="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
readonly CONFIG_PATH="${WORKDIR}/configs/baselines/pi_deeponet_lwc84_long100e_r7_authorized_20260814.yaml"
readonly CONFIG_SHA256="012f47073dc5b20a182b6caf6394487110a5714dc114cc2f1750a2b6bd7f8efc"
readonly TRAIN_SCRIPT="${WORKDIR}/scripts/train_patch_deeponet_baseline.py"
readonly TRAIN_SCRIPT_SHA256="44ae9d72c920af3f76f1e132c12b07e0865593cd44de5e5b61964c2220c9c45f"
readonly ARTIFACT_BASE="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/baselines/pi_deeponet_lwc84_long100e_r7"
readonly PARENT_ROOT="${ARTIFACT_BASE}/full_long100e_r1"
readonly PARENT_CHECKPOINT="${PARENT_ROOT}/latest.pt"
readonly PARENT_CHECKPOINT_SHA256="85aea56025cae9c28e345053cb18cbc2d247b8f2c73eafef4256a81c2c1646ae"
readonly PARENT_BEST_METADATA="${PARENT_ROOT}/best.json"
readonly PARENT_BEST_METADATA_SHA256="5848137fd380901b46720611ec75e7cd38d2a96ab49ffae92134db6bd210bfa3"
readonly ENTRY_REPORT="${ARTIFACT_BASE}/entry_gate_r1/entry_gate.json"
readonly ENTRY_REPORT_SHA256="30c97dd8fe14fadc7d4635f64db13f26ec1a2a4e14d459be1cf3fdf072e18c79"
readonly RUN_LABEL="long100e_r1_resume1"
readonly RUN_ROOT="${ARTIFACT_BASE}/full_${RUN_LABEL}"
readonly PID_FILE="${ARTIFACT_BASE}/${RUN_LABEL}.supervisor.pid"
readonly LOCK_FILE="${ARTIFACT_BASE}/${RUN_LABEL}.supervisor.lock"
readonly TERMINAL_FILE="${ARTIFACT_BASE}/${RUN_LABEL}.terminal.json"
readonly TORCHRUN="/root/miniconda3/bin/torchrun"
readonly MINIMUM_AVAILABLE_KIB=1572864

mkdir -p "${ARTIFACT_BASE}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another resume supervisor already holds ${LOCK_FILE}" >&2
  exit 73
fi
if [[ -d "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to reuse non-empty run directory: ${RUN_ROOT}" >&2
  exit 73
fi
for specification in \
  "${CONFIG_PATH}:${CONFIG_SHA256}" \
  "${TRAIN_SCRIPT}:${TRAIN_SCRIPT_SHA256}" \
  "${PARENT_CHECKPOINT}:${PARENT_CHECKPOINT_SHA256}" \
  "${PARENT_BEST_METADATA}:${PARENT_BEST_METADATA_SHA256}" \
  "${ENTRY_REPORT}:${ENTRY_REPORT_SHA256}"; do
  path=${specification%:*}
  expected=${specification##*:}
  if [[ ! -f "${path}" ]] || [[ "$(sha256sum "${path}" | awk '{print $1}')" != "${expected}" ]]; then
    echo "bound resume input is missing or changed: ${path}" >&2
    exit 74
  fi
done
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ne 4 ]]; then
  echo "resume requires exactly four visible GPUs" >&2
  exit 75
fi
compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
if [[ -n "${compute_pids}" ]]; then
  echo "refusing to race active GPU processes: ${compute_pids//$'\n'/,}" >&2
  exit 76
fi
available_kib=$(df --output=avail "${ARTIFACT_BASE}" | tail -1 | tr -d ' ')
if [[ "${available_kib}" -lt "${MINIMUM_AVAILABLE_KIB}" ]]; then
  echo "insufficient free disk for atomic rolling and best checkpoints" >&2
  exit 77
fi

printf '%s\n' "$$" >"${PID_FILE}"
export PI_RESUME_STARTED_UTC
PI_RESUME_STARTED_UTC="$(date -u +%FT%TZ)"

write_terminal_record() {
  local exit_code="$1"
  trap - EXIT
  export PI_RESUME_EXIT_CODE="${exit_code}"
  export PI_RESUME_ENDED_UTC
  PI_RESUME_ENDED_UTC="$(date -u +%FT%TZ)"
  export PI_RESUME_TERMINAL_FILE="${TERMINAL_FILE}"
  export PI_RESUME_RUN_ROOT="${RUN_ROOT}"
  export PI_RESUME_PARENT_ROOT="${PARENT_ROOT}"
  /root/miniconda3/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

terminal = Path(os.environ["PI_RESUME_TERMINAL_FILE"])
run_root = Path(os.environ["PI_RESUME_RUN_ROOT"])
parent_root = Path(os.environ["PI_RESUME_PARENT_ROOT"])
exit_code = int(os.environ["PI_RESUME_EXIT_CODE"])

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

latest_path = run_root / "latest.json"
latest = json.loads(latest_path.read_text()) if latest_path.is_file() else None
new_best = run_root / "best.pt"
selected_best = new_best if new_best.is_file() else parent_root / "best.pt"
payload = {
    "schema": "detached_training_resume_terminal_v1",
    "status": "completed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "started_utc": os.environ["PI_RESUME_STARTED_UTC"],
    "ended_utc": os.environ["PI_RESUME_ENDED_UTC"],
    "run_root": str(run_root),
    "parent_root": str(parent_root),
    "latest": latest,
    "latest_checkpoint_sha256": sha256(run_root / "latest.pt"),
    "new_best_checkpoint_sha256": sha256(new_best),
    "selected_best_checkpoint": str(selected_best),
    "selected_best_checkpoint_sha256": sha256(selected_best),
}
temporary = terminal.with_suffix(terminal.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(terminal)
PY
}
trap 'write_terminal_record "$?"' EXIT

cd "${WORKDIR}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
"${TORCHRUN}" --standalone --nproc_per_node=4 \
  scripts/train_patch_deeponet_baseline.py \
  --config "${CONFIG_PATH}" \
  --mode full \
  --entry-report "${ENTRY_REPORT}" \
  --allow-failed-entry-for-comparison \
  --microbatch-records 4 \
  --checkpoint-every-updates 20 \
  --run-label "${RUN_LABEL}" \
  --resume-checkpoint "${PARENT_CHECKPOINT}"

/root/miniconda3/bin/python - "${RUN_ROOT}/latest.json" <<'PY'
import json
from pathlib import Path
import sys

latest = json.loads(Path(sys.argv[1]).read_text())
expected = {"epoch": 100, "update_in_epoch": 140, "global_step": 14000}
for key, value in expected.items():
    if int(latest.get(key, -1)) != value:
        raise SystemExit(f"final rolling checkpoint failed {key}={value} gate")
PY
