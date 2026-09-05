#!/usr/bin/env python3
"""Supervise full-train residual caching and four-rank phase/scatter fitting."""
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
MANIFEST = RESULTS / "transfer_dg_wfp_full2800_manifest_20260902.json"
PREREG = RESULTS / "transfer_dg_phase_scatter64_full_preregistration_20260903.json"
PILOT_SUMMARY = RESULTS / "transfer_dg_phase_scatter64_pilot_20260903/summary.json"
PARENT = RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903/latest.pt"
BASE_CACHE = RESULTS / "transfer_dg_wfp_full2800_cache_20260902"
TRAVEL = RESULTS / "transfer_dg_wfp_full2800_travel_20260902"
CACHE_BUILDER = ROOT / "scripts/build_transfer_dg_phase_scatter64_full_cache.py"
TRAINER = ROOT / "scripts/train_transfer_dg_phase_scatter64_full_ddp.py"
OUT = RESULTS / "transfer_dg_phase_scatter64_full_20260903"


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


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    prereg = json.loads(PREREG.read_text())
    bindings = prereg["bindings"]
    bound = {
        "train_manifest_sha256": MANIFEST,
        "pilot_summary_sha256": PILOT_SUMMARY,
        "parent_checkpoint_sha256": PARENT,
        "cache_builder_sha256": CACHE_BUILDER,
        "trainer_sha256": TRAINER,
        "phase_scatter_module_sha256": ROOT / "saved_time_phase_operator_v4/phase_scatter.py",
        "launcher_sha256": Path(__file__),
    }
    for key, path in bound.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    pilot = json.loads(PILOT_SUMMARY.read_text())
    if pilot.get("status") != "accepted" or pilot.get("decision") != "scale_to_full_train_pool":
        raise RuntimeError("train-only phase/scatter pilot did not authorize scaling")
    if pilot.get("validation_opened") or pilot.get("test_id_opened"):
        raise RuntimeError("pilot lineage opened a frozen split")
    OUT.mkdir(parents=True)
    cache_dir = OUT / "cache"
    cache_dir.mkdir()
    started = time.time()
    base_args = sum((["--base-cache", str(BASE_CACHE / f"shard_{i}.h5")] for i in range(4)), [])
    travel_args = sum((["--travel", str(TRAVEL / f"shard_{i}.h5")] for i in range(4)), [])
    processes, handles, commands = [], [], []
    for worker in range(4):
        command = [
            sys.executable, str(CACHE_BUILDER),
            "--manifest", str(MANIFEST), "--preregistration", str(PREREG),
            "--parent-checkpoint", str(PARENT), *base_args, *travel_args,
            "--output", str(cache_dir / f"shard_{worker}.h5"),
            "--worker-index", str(worker), "--worker-count", "4",
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(worker), "HDF5_USE_FILE_LOCKING": "FALSE",
            "OMP_NUM_THREADS": "4", "PYTHONPATH": f'{ROOT}:{ROOT / "src"}',
        })
        handle = (OUT / f"cache_worker_{worker}.log").open("w")
        handles.append(handle)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT))
        commands.append(command)
    atomic_json({
        "schema": "transfer_dg_phase_scatter64_full_supervisor_identity_v1",
        "pid": os.getpid(), "state": "building_cache",
        "cache_pids": [p.pid for p in processes], "cache_commands": commands,
        "gpu_ids": [0, 1, 2, 3], "parent_checkpoint_sha256": bindings["parent_checkpoint_sha256"],
        "validation_opened": False, "test_id_opened": False,
        "started_unix_s": started,
    }, OUT / "run_identity.json")
    cache_codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    if any(code != 0 for code in cache_codes):
        atomic_json({
            "schema": "transfer_dg_phase_scatter64_full_supervisor_terminal_v1",
            "status": "cache_failed", "cache_return_codes": cache_codes,
            "validation_opened": False, "test_id_opened": False,
            "elapsed_s": time.time() - started,
        }, OUT / "terminal.json")
        return 2
    cache_args = sum((["--cache", str(cache_dir / f"shard_{i}.h5")] for i in range(4)), [])
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(TRAINER), *cache_args, "--manifest", str(MANIFEST),
        "--preregistration", str(PREREG), "--output-dir", str(OUT / "training"),
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "HDF5_USE_FILE_LOCKING": "FALSE",
        "OMP_NUM_THREADS": "4", "PYTHONPATH": f'{ROOT}:{ROOT / "src"}',
    })
    with (OUT / "training.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        identity = json.loads((OUT / "run_identity.json").read_text())
        identity.update({"state": "training", "cache_return_codes": cache_codes,
                         "torchrun_pid": process.pid, "train_command": command})
        atomic_json(identity, OUT / "run_identity.json")
        train_code = process.wait()
    trainer_terminal_path = OUT / "training/terminal.json"
    trainer_terminal = json.loads(trainer_terminal_path.read_text()) if trainer_terminal_path.exists() else None
    complete = train_code == 0 and trainer_terminal and trainer_terminal.get("status") == "complete"
    atomic_json({
        "schema": "transfer_dg_phase_scatter64_full_supervisor_terminal_v1",
        "status": "complete" if complete else "training_failed",
        "cache_return_codes": cache_codes, "train_return_code": train_code,
        "trainer_terminal": trainer_terminal, "elapsed_s": time.time() - started,
        "validation_opened": False, "test_id_opened": False,
    }, OUT / "terminal.json")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
