#!/usr/bin/env python3
"""Supervise the single-model four-GPU full-data mHC+Muon pretraining run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREREG = RESULTS / "transfer_dg_coupled_pyramid_moe_full2800_ddp4_preregistration_20260903.json"
MANIFEST = RESULTS / "transfer_dg_wfp_full2800_manifest_20260902.json"
INIT = RESULTS / "transfer_dg_coupled_pyramid_moe_pilot_20260903/mhc_muon/best.pt"
PARENT = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903/latest.pt"
RESIDUAL = RESULTS / "transfer_dg_phase_scatter64_full_20260903/cache"
BASE = RESULTS / "transfer_dg_wfp_full2800_cache_20260902"
TRAVEL = RESULTS / "transfer_dg_wfp_full2800_travel_20260902"
TRAINER = ROOT / "scripts/train_transfer_dg_coupled_pyramid_moe_full_ddp4.py"
PILOT_TRAINER = ROOT / "scripts/train_transfer_dg_coupled_mhc_muon_pilot.py"
MODEL = ROOT / "saved_time_phase_operator_v4/coupled_pyramid_moe_wave.py"
BASE_MODEL = ROOT / "saved_time_phase_operator_v4/coupled_mhc_wave.py"
OUT = RESULTS / "transfer_dg_coupled_pyramid_moe_full2800_ddp4_20260903"
IDENTITY = RESULTS / "transfer_dg_coupled_pyramid_moe_full2800_ddp4_supervisor_identity_20260903.json"
TERMINAL = RESULTS / "transfer_dg_coupled_pyramid_moe_full2800_ddp4_supervisor_terminal_20260903.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    prereg = json.loads(PREREG.read_text())
    bindings = prereg["bindings"]
    paths = {
        "trainer_sha256": TRAINER,
        "pilot_trainer_sha256": PILOT_TRAINER,
        "model_sha256": MODEL,
        "base_model_sha256": BASE_MODEL,
        "launcher_sha256": Path(__file__),
        "manifest_sha256": MANIFEST,
        "init_checkpoint_sha256": INIT,
        "parent_checkpoint_sha256": PARENT,
    }
    for key, path in paths.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"full pretraining launcher binding drift: {key} {observed}")
    active = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if active:
        raise RuntimeError(f"other GPU compute processes are still active: {active}")

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
        str(PREREG),
        "--init-checkpoint",
        str(INIT),
        "--output-dir",
        str(OUT),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "OMP_NUM_THREADS": "8",
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        }
    )
    started = time.time()
    atomic(
        {
            "schema": "transfer_dg_coupled_pyramid_moe_full_ddp4_supervisor_identity_v1",
            "pid": os.getpid(),
            "command": command,
            "gpu_ids": [0, 1, 2, 3],
            "output_dir": str(OUT.resolve()),
            "started_unix_s": started,
            "validation_opened": False,
            "test_id_opened": False,
        },
        IDENTITY,
    )
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    worker_terminal = OUT / "terminal.json"
    payload = {
        "schema": "transfer_dg_coupled_pyramid_moe_full_ddp4_supervisor_terminal_v1",
        "status": "complete" if completed.returncode == 0 and worker_terminal.exists() else "failed",
        "return_code": completed.returncode,
        "worker_terminal": str(worker_terminal.resolve()),
        "worker_terminal_exists": worker_terminal.exists(),
        "elapsed_s": time.time() - started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic(payload, TERMINAL)
    return 0 if payload["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
