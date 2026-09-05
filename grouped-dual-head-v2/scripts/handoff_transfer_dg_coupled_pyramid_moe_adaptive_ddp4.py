#!/usr/bin/env python3
"""Hand off the live fixed-weight DDP run to adaptive training at epoch 22."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import torch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE_OUT = RESULTS / "transfer_dg_coupled_pyramid_moe_full2800_ddp4_20260903"
SOURCE_CHECKPOINT = SOURCE_OUT / "latest.pt"
PREREGISTRATION = RESULTS / "transfer_dg_coupled_pyramid_moe_adaptive_ddp4_preregistration_20260904.json"
HANDOFF_DIR = RESULTS / "transfer_dg_coupled_pyramid_moe_adaptive_handoff_20260904"
SNAPSHOT = HANDOFF_DIR / "fixed_epoch21_boundary.pt"
SMOKE_OUT = HANDOFF_DIR / "smoke"
ADAPTIVE_OUT = RESULTS / "transfer_dg_coupled_pyramid_moe_adaptive_ddp4_20260904"
TRAINER = ROOT / "scripts/train_transfer_dg_coupled_pyramid_moe_adaptive_ddp4.py"
CONTROLLER = ROOT / "saved_time_phase_operator_v4/adaptive_pretraining.py"
SOURCE_TRAINER = ROOT / "scripts/train_transfer_dg_coupled_pyramid_moe_full_ddp4.py"
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


def checkpoint_cursor(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "schema": checkpoint.get("schema"),
        "epoch": int(checkpoint["epoch"]),
        "next_step": int(checkpoint["next_step"]),
        "update": int(checkpoint["update"]),
        "validation_opened": bool(checkpoint.get("validation_opened", False)),
        "test_id_opened": bool(checkpoint.get("test_id_opened", False)),
    }


def process_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.exists():
        return ""
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def gpu_compute_pids() -> list[int]:
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [int(value.strip()) for value in output.splitlines() if value.strip()]


def wait_for_no_gpu_processes(timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not gpu_compute_pids():
            return
        time.sleep(2.0)
    raise TimeoutError(f"GPU processes did not exit: {gpu_compute_pids()}")


def training_command(resume: Path, output: Path, *, max_updates: int = 0) -> list[str]:
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
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(TRAINER),
        *residual_args,
        *base_args,
        *travel_args,
        "--manifest", str(MANIFEST),
        "--preregistration", str(PREREGISTRATION),
        "--init-checkpoint", str(INIT),
        "--resume", str(resume),
        "--output-dir", str(output),
    ]
    if max_updates:
        command.extend(("--max-updates", str(max_updates)))
    return command


def terminate_source_group(pid: int, process_group: int) -> None:
    if not Path(f"/proc/{pid}").exists():
        raise ProcessLookupError(f"source torchrun PID {pid} disappeared before handoff")
    if os.getpgid(pid) != process_group:
        raise RuntimeError("source process-group identity drift")
    os.killpg(process_group, signal.SIGSTOP)
    time.sleep(0.5)
    frozen = checkpoint_cursor(SOURCE_CHECKPOINT)
    if frozen["epoch"] != 22 or frozen["next_step"] != 0 or frozen["update"] != 14700:
        os.killpg(process_group, signal.SIGCONT)
        raise RuntimeError(f"source was not frozen on the epoch-21 boundary: {frozen}")
    if SNAPSHOT.exists():
        raise FileExistsError(SNAPSHOT)
    os.link(SOURCE_CHECKPOINT, SNAPSHOT)
    snapshot_hash = sha256(SNAPSHOT)
    atomic_json(
        {
            "schema": "fixed_to_adaptive_epoch_boundary_handoff_v1",
            "status": "handed_off",
            "source_torchrun_pid": pid,
            "source_process_group": process_group,
            "cursor": frozen,
            "snapshot": str(SNAPSHOT.resolve()),
            "snapshot_sha256": snapshot_hash,
            "validation_opened": False,
            "test_id_opened": False,
        },
        HANDOFF_DIR / "source_terminal.json",
    )
    os.killpg(process_group, signal.SIGTERM)
    os.killpg(process_group, signal.SIGCONT)
    deadline = time.time() + 60.0
    while time.time() < deadline and Path(f"/proc/{pid}").exists():
        time.sleep(1.0)
    if Path(f"/proc/{pid}").exists():
        os.killpg(process_group, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-torchrun-pid", type=int, required=True)
    parser.add_argument("--source-process-group", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if HANDOFF_DIR.exists() or ADAPTIVE_OUT.exists():
        raise FileExistsError("adaptive handoff or output directory already exists")
    HANDOFF_DIR.mkdir(parents=True)
    terminal_path = HANDOFF_DIR / "terminal.json"
    started = time.time()
    try:
        prereg = json.loads(PREREGISTRATION.read_text())
        bindings = prereg["bindings"]
        for key, path in {
            "handoff_sha256": Path(__file__),
            "trainer_sha256": TRAINER,
            "controller_sha256": CONTROLLER,
            "source_trainer_sha256": SOURCE_TRAINER,
            "model_sha256": MODEL,
            "manifest_sha256": MANIFEST,
            "init_checkpoint_sha256": INIT,
        }.items():
            observed = sha256(path)
            if observed != bindings[key]:
                raise RuntimeError(f"handoff binding drift: {key} {observed}")
        command = process_command(args.source_torchrun_pid)
        if str(SOURCE_TRAINER) not in command:
            raise RuntimeError("source PID is not the preregistered fixed-weight trainer")
        if os.getpgid(args.source_torchrun_pid) != args.source_process_group:
            raise RuntimeError("source process-group identity mismatch")

        atomic_json(
            {
                "schema": "adaptive_pretraining_handoff_identity_v1",
                "status": "waiting_for_epoch_21_boundary",
                "pid": os.getpid(),
                "source_torchrun_pid": args.source_torchrun_pid,
                "source_process_group": args.source_process_group,
                "expected_checkpoint": {"epoch": 22, "next_step": 0, "update": 14700},
                "adaptive_output": str(ADAPTIVE_OUT.resolve()),
                "started_unix_s": started,
                "validation_opened": False,
                "test_id_opened": False,
            },
            HANDOFF_DIR / "identity.json",
        )

        deadline = time.time() + float(prereg["handoff"]["wait_timeout_s"])
        observed_mtime = None
        while time.time() < deadline:
            if not Path(f"/proc/{args.source_torchrun_pid}").exists():
                raise ProcessLookupError("source training stopped before the epoch boundary")
            stat = SOURCE_CHECKPOINT.stat()
            if stat.st_mtime_ns != observed_mtime:
                observed_mtime = stat.st_mtime_ns
                cursor = checkpoint_cursor(SOURCE_CHECKPOINT)
                if cursor["validation_opened"] or cursor["test_id_opened"]:
                    raise RuntimeError("source checkpoint opened a sealed split")
                atomic_json(
                    {
                        "status": "waiting_for_epoch_21_boundary",
                        "observed": cursor,
                        "checkpoint_mtime_ns": observed_mtime,
                        "observed_unix_s": time.time(),
                    },
                    HANDOFF_DIR / "monitor.json",
                )
                if cursor["epoch"] == 22 and cursor["next_step"] == 0 and cursor["update"] == 14700:
                    break
                if cursor["epoch"] > 22 or cursor["update"] > 14700:
                    raise RuntimeError(f"missed the preregistered handoff checkpoint: {cursor}")
            time.sleep(max(float(args.poll_seconds), 0.5))
        else:
            raise TimeoutError("timed out waiting for the epoch-21 boundary")

        terminate_source_group(args.source_torchrun_pid, args.source_process_group)
        wait_for_no_gpu_processes(120.0)
        snapshot_hash = sha256(SNAPSHOT)
        environment = os.environ.copy()
        environment.update(
            {
                "HDF5_USE_FILE_LOCKING": "FALSE",
                "OMP_NUM_THREADS": "8",
                "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
        )

        smoke_command = training_command(
            SNAPSHOT, SMOKE_OUT, max_updates=int(prereg["smoke"]["max_updates"])
        )
        atomic_json(
            {
                "status": "smoke_running",
                "snapshot_sha256": snapshot_hash,
                "command": smoke_command,
                "started_unix_s": time.time(),
            },
            HANDOFF_DIR / "status.json",
        )
        smoke = subprocess.run(smoke_command, cwd=ROOT, env=environment)
        smoke_terminal_path = SMOKE_OUT / "terminal.json"
        if smoke.returncode != 0 or not smoke_terminal_path.exists():
            raise RuntimeError(f"adaptive DDP smoke failed with return code {smoke.returncode}")
        smoke_terminal = json.loads(smoke_terminal_path.read_text())
        if (
            smoke_terminal.get("status") != "smoke_complete"
            or smoke_terminal.get("validation_opened")
            or smoke_terminal.get("test_id_opened")
        ):
            raise RuntimeError(f"adaptive DDP smoke terminal is invalid: {smoke_terminal}")
        wait_for_no_gpu_processes(120.0)

        full_command = training_command(SNAPSHOT, ADAPTIVE_OUT)
        child = subprocess.Popen(full_command, cwd=ROOT, env=environment)
        atomic_json(
            {
                "schema": "adaptive_pretraining_handoff_live_v1",
                "status": "adaptive_training_running",
                "supervisor_pid": os.getpid(),
                "child_pid": child.pid,
                "command": full_command,
                "snapshot": str(SNAPSHOT.resolve()),
                "snapshot_sha256": snapshot_hash,
                "smoke_terminal": str(smoke_terminal_path.resolve()),
                "smoke_status": smoke_terminal["status"],
                "adaptive_output": str(ADAPTIVE_OUT.resolve()),
                "started_unix_s": time.time(),
                "validation_opened": False,
                "test_id_opened": False,
            },
            HANDOFF_DIR / "status.json",
        )
        return_code = child.wait()
        worker_terminal = ADAPTIVE_OUT / "terminal.json"
        status = "complete" if return_code == 0 and worker_terminal.exists() else "failed"
        atomic_json(
            {
                "schema": "adaptive_pretraining_handoff_terminal_v1",
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
                "schema": "adaptive_pretraining_handoff_terminal_v1",
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
