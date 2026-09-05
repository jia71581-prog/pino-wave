#!/usr/bin/env python3
"""One-shot detached launcher for the preregistered K4 pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_transfer_dg_parent_anchored_k4 import (  # noqa: E402
    CANDIDATE,
    _resolved,
    atomic_json,
    sha256,
    validate_stage_prerequisite,
    validate_training_bindings,
)


def validate_prerequisites(prereg: dict) -> None:
    for name in ("mechanism", "smoke"):
        validate_stage_prerequisite(prereg, name)


def validate_launcher_contract(
    prereg: dict, preregistration: Path
) -> tuple[list[str], Path, Path, Path, str]:
    if prereg.get("schema") != "transfer_dg_parent_anchored_k4_preregistration_v1":
        raise RuntimeError("launcher preregistration schema drift")
    if prereg.get("candidate") != CANDIDATE:
        raise RuntimeError("launcher candidate drift")
    if prereg.get("status") != "pilot_pending_independent_audit":
        raise RuntimeError("launcher status contract rejected")
    if preregistration.resolve() != _resolved(prereg["paths"]["preregistration"]):
        raise RuntimeError("launcher preregistration path override rejected")
    validate_prerequisites(prereg)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None:
        raise RuntimeError("launcher requires CUBLAS_WORKSPACE_CONFIG to be unset")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("launcher requires CUDA_VISIBLE_DEVICES=0")
    command = shlex.split(prereg["exact_argv"]["pilot_child"])
    expected_command = [
        "env",
        "-u",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES=0",
        "python",
        "scripts/train_transfer_dg_parent_anchored_k4.py",
        "--pilot",
        "--preregistration",
        prereg["paths"]["preregistration"],
        "--cache",
        prereg["paths"]["cache"],
        "--source-h5",
        prereg["paths"]["source_h5"],
        "--output-dir",
        prereg["paths"]["pilot_output_dir"],
    ]
    if command != expected_command:
        raise RuntimeError("launcher child argv override")
    output_dir = _resolved(prereg["paths"]["pilot_output_dir"])
    log_path = _resolved(prereg["paths"]["pilot_log"])
    supervisor_path = _resolved(prereg["paths"]["supervisor_identity"])
    if output_dir.exists() or log_path.exists():
        raise FileExistsError("pilot output/log already exists")
    if supervisor_path.exists():
        raise FileExistsError("supervisor identity already exists")
    binding_args = argparse.Namespace(
        cache=_resolved(prereg["paths"]["cache"]),
        source_h5=_resolved(prereg["paths"]["source_h5"]),
    )
    validate_training_bindings(binding_args, prereg)
    preregistration_sha256 = sha256(preregistration)
    return command, output_dir, log_path, supervisor_path, preregistration_sha256


def query_compute_processes() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append({"pid": fields[0], "gpu_uuid": fields[1], "used_memory_mib": fields[2]})
    return rows


def verify_child_launch(
    *,
    child: Any,
    run_identity: Path,
    log_path: Path,
    candidate: str,
    preregistration_sha256: str,
    output_dir: Path,
    query_processes: Callable[[], list[dict[str, str]]] = query_compute_processes,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    deadline = monotonic() + float(timeout_seconds)
    last_processes: list[dict[str, str]] = []
    while monotonic() <= deadline:
        if child.poll() is not None:
            raise RuntimeError(f"K4 pilot child exited during launch verification: {child.returncode}")
        last_processes = query_processes()
        pid_visible = any(row.get("pid") == str(child.pid) for row in last_processes)
        identity = None
        if run_identity.is_file():
            try:
                identity = json.loads(run_identity.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                identity = None
        log_ready = (
            log_path.is_file()
            and log_path.stat().st_size > 0
            and '"event": "k4_identity"' in log_path.read_text(encoding="utf-8")
        )
        identity_ready = bool(
            identity
            and identity.get("candidate") == candidate
            and identity.get("mode") == "pilot"
            and identity.get("preregistration_sha256_observed_before") == preregistration_sha256
            and Path(identity.get("output_dir", "")).resolve() == output_dir.resolve()
        )
        if pid_visible and log_ready and identity_ready:
            return {
                "child_alive": True,
                "run_identity_verified": True,
                "log_identity_event_verified": True,
                "compute_pid_verified": True,
                "compute_processes": last_processes,
            }
        sleep(0.25)
    raise TimeoutError(
        f"K4 pilot launch verification exceeded {timeout_seconds}s; last_processes={last_processes}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    supervisor_path = _resolved(prereg["paths"]["supervisor_identity"])
    identity: dict[str, Any] = {
        "schema": "transfer_dg_parent_anchored_k4_supervisor_v1",
        "candidate": prereg.get("candidate"),
        "status": "starting",
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": sha256(args.preregistration),
        "argv": list(sys.argv),
        "restart_authorized": False,
        "kill_authorized": False,
        "validation_opened": False,
        "test_id_opened": False,
    }
    try:
        command, output_dir, log_path, supervisor_path, preregistration_sha256 = validate_launcher_contract(
            prereg, args.preregistration
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_environment = dict(os.environ)
        child_environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        child_environment["CUDA_VISIBLE_DEVICES"] = "0"
        with log_path.open("x", encoding="utf-8") as log_handle:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        verification = verify_child_launch(
            child=child,
            run_identity=output_dir / "run_identity.json",
            log_path=log_path,
            candidate=CANDIDATE,
            preregistration_sha256=preregistration_sha256,
            output_dir=output_dir,
        )
        identity.update(
            {
                "status": "launched",
                "pid": child.pid,
                "child_argv": command,
                "child_argv_sha256": __import__("hashlib").sha256(
                    json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode()
                ).hexdigest(),
                "trainer_sha256": prereg["bindings"]["trainer_sha256"],
                "output_dir": str(output_dir),
                "log": str(log_path),
                "verification": verification,
            }
        )
        atomic_json(identity, supervisor_path)
        print(json.dumps({"status": "launched", "pid": child.pid, "log": str(log_path)}, sort_keys=True))
        return 0
    except Exception as error:
        identity.update({"status": "failed", "error": repr(error)})
        if not supervisor_path.exists():
            atomic_json(identity, supervisor_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
