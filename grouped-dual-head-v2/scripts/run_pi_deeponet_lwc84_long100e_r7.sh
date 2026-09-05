#!/usr/bin/env bash
set -uo pipefail

readonly WORKDIR="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
readonly CONFIG_PATH="${WORKDIR}/configs/baselines/pi_deeponet_lwc84_long100e_r7_authorized_20260814.yaml"
readonly ARTIFACT_BASE="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/baselines/pi_deeponet_lwc84_long100e_r7"
readonly RUN_LABEL="long100e_r1"
readonly RUN_ROOT="${ARTIFACT_BASE}/full_${RUN_LABEL}"
readonly ENTRY_REPORT="${ARTIFACT_BASE}/entry_gate_r1/entry_gate.json"
readonly PID_FILE="${ARTIFACT_BASE}/long100e_r1.supervisor.pid"
readonly LOCK_FILE="${ARTIFACT_BASE}/long100e_r1.supervisor.lock"
readonly TERMINAL_FILE="${ARTIFACT_BASE}/long100e_r1.terminal.json"
readonly TORCHRUN="/root/miniconda3/bin/torchrun"

mkdir -p "${ARTIFACT_BASE}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another long100e_r1 supervisor already holds ${LOCK_FILE}" >&2
  exit 73
fi
if [[ ! -f "${ENTRY_REPORT}" ]]; then
  echo "missing bound entry report: ${ENTRY_REPORT}" >&2
  exit 66
fi
if [[ -d "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to reuse non-empty run directory: ${RUN_ROOT}" >&2
  exit 73
fi

printf '%s\n' "$$" >"${PID_FILE}"
export PI_LONG_STARTED_UTC
PI_LONG_STARTED_UTC="$(date -u +%FT%TZ)"

write_terminal_record() {
  local exit_code="$1"
  trap - EXIT
  export PI_LONG_EXIT_CODE="${exit_code}"
  export PI_LONG_ENDED_UTC
  PI_LONG_ENDED_UTC="$(date -u +%FT%TZ)"
  export PI_LONG_TERMINAL_FILE="${TERMINAL_FILE}"
  export PI_LONG_RUN_ROOT="${RUN_ROOT}"
  /root/miniconda3/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

terminal = Path(os.environ["PI_LONG_TERMINAL_FILE"])
run_root = Path(os.environ["PI_LONG_RUN_ROOT"])
exit_code = int(os.environ["PI_LONG_EXIT_CODE"])

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

latest_json = run_root / "latest.json"
latest = json.loads(latest_json.read_text()) if latest_json.is_file() else None
payload = {
    "schema": "detached_training_terminal_v1",
    "status": "completed" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "started_utc": os.environ["PI_LONG_STARTED_UTC"],
    "ended_utc": os.environ["PI_LONG_ENDED_UTC"],
    "run_root": str(run_root),
    "latest": latest,
    "latest_checkpoint_sha256": sha256(run_root / "latest.pt"),
    "best_checkpoint_sha256": sha256(run_root / "best.pt"),
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
  --run-label "${RUN_LABEL}"
