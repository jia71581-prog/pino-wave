#!/usr/bin/env python3
"""Launch adaptive DDP from the preserved epoch-20 mid-epoch checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREREGISTRATION = RESULTS / "transfer_dg_coupled_pyramid_moe_adaptive_epoch20_preregistration_v2_20260904.json"
HANDOFF_DIR = RESULTS / "transfer_dg_coupled_pyramid_moe_adaptive_epoch20_handoff_v2_20260904"
SMOKE_OUT = HANDOFF_DIR / "smoke"
OUTPUT = RESULTS / "transfer_dg_coupled_pyramid_moe_adaptive_epoch20_ddp4_v2_20260904"
RESUME = RESULTS / "transfer_dg_pyramid_u13650_marmousi_00260_instance_adapt_cpu_20260904/parent_snapshot.pt"
TRAINER = ROOT / "scripts/train_transfer_dg_coupled_pyramid_moe_adaptive_epoch20_ddp4.py"
CONTROLLER = ROOT / "saved_time_phase_operator_v4/adaptive_pretraining.py"
MODEL = ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py"
MANIFEST = RESULTS / "transfer_dg_wfp_full2800_manifest_20260902.json"
INIT = RESULTS / "transfer_dg_coupled_pyramid_moe_pilot_20260903/mhc_muon/best.pt"
RESIDUAL = RESULTS / "transfer_dg_phase_scatter64_full_20260903/cache"
BASE = RESULTS / "transfer_dg_wfp_full2800_cache_20260902"
TRAVEL = RESULTS / "transfer_dg_wfp_full2800_travel_20260902"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checkpoint_cursor(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "schema": checkpoint.get("schema"),
        "epoch": int(checkpoint["epoch"]),
        "next_step": int(checkpoint["next_step"]),
        "update": int(checkpoint["update"]),
        "validation_opened": bool(checkpoint.get("validation_opened", False)),
        "test_id_opened": bool(checkpoint.get("test_id_opened", False)),
    }


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(value.strip()) for value in result.stdout.splitlines() if value.strip()]


def wait_for_no_gpu_processes(timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if not gpu_compute_pids():
            return
        time.sleep(2.0)
    raise TimeoutError(f"GPU processes did not exit: {gpu_compute_pids()}")


def training_command(output: Path, *, max_updates: int = 0) -> list[str]:
    residual_args = sum(
        (["--residual-cache", str(RESIDUAL / f"shard_{index}.h5")] for index in range(4)),
        [],
    )
    base_args = sum(
        (["--base-cache", str(BASE / f"shard_{index}.h5")] for index in range(4)),
        [],
    )
    travel_args = sum(
        (["--travel", str(TRAVEL / f"shard_{index}.h5")] for index in range(4)),
        [],
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(TRAINER),
        *residual_args,
        *base_args,
        *travel_args,
        "--manifest",
        str(MANIFEST),
        "--preregistration",
        str(PREREGISTRATION),
        "--init-checkpoint",
        str(INIT),
        "--resume",
        str(RESUME),
        "--output-dir",
        str(output),
    ]
    if max_updates:
        command.extend(("--max-updates", str(max_updates)))
    return command


def main() -> int:
    if HANDOFF_DIR.exists() or OUTPUT.exists():
        raise FileExistsError("epoch-20 handoff or output already exists")
    HANDOFF_DIR.mkdir(parents=True)
    terminal_path = HANDOFF_DIR / "terminal.json"
    started = time.time()
    try:
        prereg = json.loads(PREREGISTRATION.read_text())
        bindings = prereg["bindings"]
        for key, path in {
            "launcher_sha256": Path(__file__),
            "trainer_sha256": TRAINER,
            "controller_sha256": CONTROLLER,
            "model_sha256": MODEL,
            "manifest_sha256": MANIFEST,
            "init_checkpoint_sha256": INIT,
            "resume_checkpoint_sha256": RESUME,
        }.items():
            observed = sha256(path)
            if observed != bindings[key]:
                raise RuntimeError(f"epoch-20 launch binding drift: {key} {observed}")
        cursor = checkpoint_cursor(RESUME)
        expected = prereg["handoff"]
        if (
            cursor["epoch"] != int(expected["expected_start_epoch"])
            or cursor["next_step"] != int(expected["expected_next_step"])
            or cursor["update"] != int(expected["expected_update"])
        ):
            raise RuntimeError(f"epoch-20 resume cursor drift: {cursor}")
        if cursor["validation_opened"] or cursor["test_id_opened"]:
            raise RuntimeError("epoch-20 checkpoint opened a sealed split")
        wait_for_no_gpu_processes(120.0)

        environment = os.environ.copy()
        environment.update(
            {
                "HDF5_USE_FILE_LOCKING": "FALSE",
                "OMP_NUM_THREADS": "8",
                "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
        )
        identity = {
            "schema": "adaptive_epoch20_handoff_identity_v1",
            "pid": os.getpid(),
            "resume": str(RESUME.resolve()),
            "resume_sha256": bindings["resume_checkpoint_sha256"],
            "resume_cursor": cursor,
            "output": str(OUTPUT.resolve()),
            "controller_first_update_epoch": 21,
            "started_unix_s": started,
            "validation_opened": False,
            "test_id_opened": False,
        }
        atomic_json(identity, HANDOFF_DIR / "identity.json")

        smoke_command = training_command(
            SMOKE_OUT, max_updates=int(prereg["smoke"]["max_updates"])
        )
        atomic_json(
            {"status": "smoke_running", "command": smoke_command, **identity},
            HANDOFF_DIR / "status.json",
        )
        smoke = subprocess.run(smoke_command, cwd=ROOT, env=environment)
        smoke_terminal_path = SMOKE_OUT / "terminal.json"
        if smoke.returncode != 0 or not smoke_terminal_path.exists():
            raise RuntimeError(f"epoch-20 smoke failed with return code {smoke.returncode}")
        smoke_terminal = json.loads(smoke_terminal_path.read_text())
        smoke_checkpoint = checkpoint_cursor(SMOKE_OUT / "latest.pt")
        expected_smoke = prereg["smoke"]["expected_checkpoint"]
        if (
            smoke_terminal.get("status") != "smoke_complete"
            or smoke_checkpoint["epoch"] != int(expected_smoke["epoch"])
            or smoke_checkpoint["next_step"] != int(expected_smoke["next_step"])
            or smoke_checkpoint["update"] != int(expected_smoke["update"])
            or smoke_terminal.get("validation_opened")
            or smoke_terminal.get("test_id_opened")
        ):
            raise RuntimeError("epoch-20 smoke terminal or resume cursor is invalid")
        wait_for_no_gpu_processes(120.0)

        full_command = training_command(OUTPUT)
        child = subprocess.Popen(full_command, cwd=ROOT, env=environment)
        atomic_json(
            {
                "schema": "adaptive_epoch20_handoff_live_v1",
                "status": "adaptive_epoch20_training_running",
                "supervisor_pid": os.getpid(),
                "child_pid": child.pid,
                "command": full_command,
                "resume": str(RESUME.resolve()),
                "resume_sha256": bindings["resume_checkpoint_sha256"],
                "smoke_terminal": str(smoke_terminal_path.resolve()),
                "smoke_checkpoint": smoke_checkpoint,
                "output": str(OUTPUT.resolve()),
                "started_unix_s": time.time(),
                "validation_opened": False,
                "test_id_opened": False,
            },
            HANDOFF_DIR / "status.json",
        )
        return_code = child.wait()
        worker_terminal = OUTPUT / "terminal.json"
        status = "complete" if return_code == 0 and worker_terminal.exists() else "failed"
        atomic_json(
            {
                "schema": "adaptive_epoch20_handoff_terminal_v1",
                "status": status,
                "return_code": return_code,
                "worker_terminal": str(worker_terminal.resolve()),
                "worker_terminal_exists": worker_terminal.exists(),
                "elapsed_s": time.time() - started,
                "validation_opened": False,
                "test_id_opened": False,
            },
            terminal_path,
        )
        return 0 if status == "complete" else 2
    except BaseException as error:
        atomic_json(
            {
                "schema": "adaptive_epoch20_handoff_terminal_v1",
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
                "validation_opened": False,
                "test_id_opened": False,
            },
            terminal_path,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
