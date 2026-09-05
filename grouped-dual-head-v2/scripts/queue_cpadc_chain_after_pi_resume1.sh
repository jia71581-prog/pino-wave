#!/usr/bin/env bash
# Continue the registered CPADC train-only chain only after the interrupted
# PI-DeepONet comparison run reaches its exact terminal schedule cursor.
set -Eeuo pipefail

readonly PROJECT_ROOT="/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2"
readonly PYTHON_BIN="/root/miniconda3/bin/python"
readonly PI_TERMINAL="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/baselines/pi_deeponet_lwc84_long100e_r7/long100e_r1_resume1.terminal.json"
readonly CANDIDATE_SCRIPT="${PROJECT_ROOT}/scripts/queue_cpadc_lspg_rad_feasibility_after_gpus_free.sh"
readonly CANDIDATE_SCRIPT_SHA256="6382770de9768179cf551fcdb268b4017c4427231eb7848c6a3db88876e0b735"
readonly ABLATION_SCRIPT="${PROJECT_ROOT}/scripts/queue_cpadc_no_online_defect_after_candidate.sh"
readonly ABLATION_SCRIPT_SHA256="0463b283d3d13a659cbaa068b4b9a92666f060dc722b37135ebe2e9b326b50bc"
readonly PAIR_SCRIPT="${PROJECT_ROOT}/scripts/queue_cpadc_trainonly_pair_after_ablation.sh"
readonly PAIR_SCRIPT_SHA256="3b7f38c47280429fc5b30eee6f30b613160b48238acbe414ff1ca22c001e6c83"
readonly CHAIN_ROOT="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/continuation_after_pi_resume1_v4_20260815"
readonly STATUS_PATH="${CHAIN_ROOT}/status.json"
readonly TERMINAL_PATH="${CHAIN_ROOT}/terminal.json"
readonly LOCK_PATH="${CHAIN_ROOT}/supervisor.lock"
readonly PID_PATH="${CHAIN_ROOT}/supervisor.pid"
readonly POLL_SECONDS=60

mkdir -p "${CHAIN_ROOT}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "another PI-to-CPADC continuation supervisor owns the lock" >&2
  exit 73
fi
if [[ -f "${TERMINAL_PATH}" ]]; then
  echo "refusing to replace existing continuation terminal: ${TERMINAL_PATH}" >&2
  exit 74
fi
for specification in \
  "${CANDIDATE_SCRIPT}:${CANDIDATE_SCRIPT_SHA256}" \
  "${ABLATION_SCRIPT}:${ABLATION_SCRIPT_SHA256}" \
  "${PAIR_SCRIPT}:${PAIR_SCRIPT_SHA256}"; do
  path=${specification%:*}
  expected=${specification##*:}
  if [[ "$(sha256sum "${path}" | awk '{print $1}')" != "${expected}" ]]; then
    echo "registered CPADC queue script changed: ${path}" >&2
    exit 75
  fi
done
printf '%s\n' "$$" >"${PID_PATH}"

write_record() {
  local status=$1
  local detail=$2
  local path=$3
  "${PYTHON_BIN}" - "${path}" "${status}" "${detail}" <<'PY'
import json
import os
from pathlib import Path
import sys
import time

path = Path(sys.argv[1])
payload = {
    "schema": "pi_resume_to_cpadc_trainonly_chain_v1",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "supervisor_pid": os.getppid(),
    "updated_unix_s": time.time(),
    "validation_access": False,
    "test_id_access": False,
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

while [[ ! -f "${PI_TERMINAL}" ]]; do
  write_record waiting_for_pi_resume pi_terminal_absent "${STATUS_PATH}"
  sleep "${POLL_SECONDS}"
done
set +e
pi_detail=$("${PYTHON_BIN}" - "${PI_TERMINAL}" 2>&1 <<'PY'
import json
from pathlib import Path
import sys

terminal = json.loads(Path(sys.argv[1]).read_text())
if terminal.get("status") != "completed" or int(terminal.get("exit_code", -1)) != 0:
    raise SystemExit("pi_resume_terminal_not_complete")
latest = terminal.get("latest") or {}
expected = {"epoch": 100, "update_in_epoch": 140, "global_step": 14000}
for key, value in expected.items():
    if int(latest.get(key, -1)) != value:
        raise SystemExit(f"pi_resume_cursor_mismatch:{key}")
print("pi_resume_complete_at_global_step_14000")
PY
)
pi_return_code=$?
set -e
if [[ "${pi_return_code}" -ne 0 ]]; then
  write_record blocked "${pi_detail}" "${TERMINAL_PATH}"
  exit 76
fi

cd "${PROJECT_ROOT}"
write_record running_candidate "${pi_detail}" "${STATUS_PATH}"
set +e
bash "${CANDIDATE_SCRIPT}"
candidate_return_code=$?
set -e
if [[ "${candidate_return_code}" -ne 0 ]]; then
  write_record failed "candidate_return_code=${candidate_return_code}" "${TERMINAL_PATH}"
  exit "${candidate_return_code}"
fi

write_record running_strict_ablation candidate_complete "${STATUS_PATH}"
set +e
bash "${ABLATION_SCRIPT}"
ablation_return_code=$?
set -e
if [[ "${ablation_return_code}" -ne 0 ]]; then
  write_record rejected "ablation_gate_return_code=${ablation_return_code}" "${TERMINAL_PATH}"
  exit 0
fi

write_record running_disjoint_train_pair ablation_complete "${STATUS_PATH}"
set +e
bash "${PAIR_SCRIPT}"
pair_return_code=$?
set -e
if [[ "${pair_return_code}" -ne 0 ]]; then
  write_record failed "paired_gate_return_code=${pair_return_code}" "${TERMINAL_PATH}"
  exit "${pair_return_code}"
fi

pair_terminal=/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/cpadc/cpadc_lspg_rad_r1_trainonly_pair_a3r3e0_r4_20260815/queue_terminal.json
pair_status=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${pair_terminal}")
write_record "${pair_status}" "paired_trainonly_gate_${pair_status}" "${TERMINAL_PATH}"
