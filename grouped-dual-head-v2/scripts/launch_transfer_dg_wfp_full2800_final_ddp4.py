#!/usr/bin/env python3
"""Gate and supervise the four-GPU final full-2,800 pretraining run."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOURCE = RESULTS / "transfer_dg_wfp_full2800_pretraining_20260902"
SUMMARY = SOURCE / "summary.json"
SOURCE_TERMINAL = SOURCE / "terminal.json"
PREREG = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_preregistration_20260903.json"
TRAINER = ROOT / "scripts/train_transfer_dg_wfp_full2800_final_ddp.py"
OUT = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903"
LOG = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903.log"
IDENTITY = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_supervisor_identity_20260903.json"
TERMINAL = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_supervisor_terminal_20260903.json"
MANIFEST = RESULTS / "transfer_dg_wfp_full2800_manifest_20260902.json"
CACHE = RESULTS / "transfer_dg_wfp_full2800_cache_20260902"
TRAVEL = RESULTS / "transfer_dg_wfp_full2800_travel_20260902"


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if TERMINAL.exists() or IDENTITY.exists() or OUT.exists():
        raise FileExistsError("refusing to reuse final-run artifacts")
    summary = json.loads(SUMMARY.read_text())
    source_terminal = json.loads(SOURCE_TERMINAL.read_text())
    prereg = json.loads(PREREG.read_text())
    bindings = prereg["bindings"]
    bound_paths = {
        "trainer_sha256": TRAINER,
        "launcher_sha256": Path(__file__),
        "wfp_module_sha256": ROOT / "saved_time_phase_operator_v4/wfp.py",
        "manifest_sha256": MANIFEST,
        "capacity_summary_sha256": SUMMARY,
        "capacity_terminal_sha256": SOURCE_TERMINAL,
        "cache_terminal_sha256": CACHE / "terminal.json",
        "travel_terminal_sha256": RESULTS / "transfer_dg_wfp_full2800_travel_terminal_20260902.json",
    }
    for key, path in bound_paths.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    if source_terminal.get("status") != "complete":
        raise RuntimeError("capacity comparison is not complete")
    if summary.get("status") != "accepted":
        raise RuntimeError("high-capacity comparison gate did not pass")
    if summary.get("decision") != "auto_approve_final_all2800_retrain":
        raise RuntimeError("capacity summary does not authorize final retraining")
    if summary.get("validation_opened") or summary.get("test_id_opened"):
        raise RuntimeError("sealed split flag opened before final retraining")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(TRAINER),
    ]
    for index in range(4):
        command.extend(["--cache", str(CACHE / f"shard_{index}.h5")])
    for index in range(4):
        command.extend(["--travel", str(TRAVEL / f"shard_{index}.h5")])
    command.extend(
        [
            "--manifest",
            str(MANIFEST),
            "--preregistration",
            str(PREREG),
            "--output-dir",
            str(OUT),
            "--epochs",
            str(prereg["training"]["epochs"]),
            "--seed",
            str(prereg["training"]["seed"]),
        ]
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
    environment["OMP_NUM_THREADS"] = "4"
    environment["PYTHONPATH"] = f'{ROOT}:{ROOT / "src"}'
    started = time.time()
    with LOG.open("w") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        atomic_json(
            {
                "schema": "transfer_dg_wfp_full2800_final_ddp4_supervisor_identity_v1",
                "pid": os.getpid(),
                "torchrun_pid": process.pid,
                "command": command,
                "gpu_ids": [0, 1, 2, 3],
                "preregistration": str(PREREG),
                "output_dir": str(OUT),
                "log": str(LOG),
                "self_approved": True,
                "validation_opened": False,
                "test_id_opened": False,
                "started_unix_s": started,
            },
            IDENTITY,
        )
        return_code = process.wait()
    trainer_terminal = None
    if (OUT / "terminal.json").exists():
        trainer_terminal = json.loads((OUT / "terminal.json").read_text())
    complete = (
        return_code == 0
        and trainer_terminal is not None
        and trainer_terminal.get("status") == "complete"
    )
    atomic_json(
        {
            "schema": "transfer_dg_wfp_full2800_final_ddp4_supervisor_terminal_v1",
            "status": "complete" if complete else "failed",
            "return_code": return_code,
            "trainer_terminal": trainer_terminal,
            "elapsed_s": time.time() - started,
            "validation_opened": False,
            "test_id_opened": False,
        },
        TERMINAL,
    )
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
