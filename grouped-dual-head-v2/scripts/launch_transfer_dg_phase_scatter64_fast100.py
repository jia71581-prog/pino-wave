#!/usr/bin/env python3
"""Launch the high-memory 100-epoch phase/scatter-64 DDP run."""
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
PREREG = RESULTS / "transfer_dg_phase_scatter64_fast100_preregistration_20260903.json"
MANIFEST = RESULTS / "transfer_dg_wfp_full2800_manifest_20260902.json"
TRAINER = ROOT / "scripts/train_transfer_dg_phase_scatter64_full_ddp.py"
CACHE = RESULTS / "transfer_dg_phase_scatter64_full_20260903/cache"
OUT = RESULTS / "transfer_dg_phase_scatter64_fast100_20260903"


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
    paths = {
        "trainer_sha256": TRAINER,
        "launcher_sha256": Path(__file__),
        "train_manifest_sha256": MANIFEST,
        "pilot_summary_sha256": RESULTS / "transfer_dg_phase_scatter64_pilot_20260903/summary.json",
        "parent_checkpoint_sha256": RESULTS / "transfer_dg_wfp_full2800_final_ddp4_20260903/latest.pt",
        "superseded_terminal_sha256": RESULTS / "transfer_dg_phase_scatter64_full_20260903/terminal.json",
    }
    for index in range(4):
        paths[f"cache_summary_{index}_sha256"] = CACHE / f"shard_{index}.h5.summary.json"
    for key, path in paths.items():
        observed = sha256(path)
        if observed != bindings[key]:
            raise RuntimeError(f"binding drift for {key}: {observed}")
    OUT.mkdir(parents=True)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(TRAINER),
    ]
    for index in range(4):
        command.extend(["--cache", str(CACHE / f"shard_{index}.h5")])
    command.extend([
        "--manifest", str(MANIFEST), "--preregistration", str(PREREG),
        "--output-dir", str(OUT / "training"),
    ])
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "HDF5_USE_FILE_LOCKING": "FALSE",
        "OMP_NUM_THREADS": "8", "PYTHONPATH": f'{ROOT}:{ROOT / "src"}',
    })
    started = time.time()
    with (OUT / "training.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        atomic_json({
            "schema": "transfer_dg_phase_scatter64_fast100_supervisor_identity_v1",
            "pid": os.getpid(), "torchrun_pid": process.pid, "command": command,
            "gpu_ids": [0, 1, 2, 3], "rank_batch_size": prereg["training"]["rank_batch_size"],
            "global_batch_size": prereg["training"]["global_batch_size"],
            "epochs": prereg["training"]["epochs"], "validation_opened": False,
            "test_id_opened": False, "started_unix_s": started,
        }, OUT / "run_identity.json")
        code = process.wait()
    trainer_path = OUT / "training/terminal.json"
    trainer_terminal = json.loads(trainer_path.read_text()) if trainer_path.exists() else None
    complete = code == 0 and trainer_terminal and trainer_terminal.get("status") == "complete"
    atomic_json({
        "schema": "transfer_dg_phase_scatter64_fast100_supervisor_terminal_v1",
        "status": "complete" if complete else "failed", "return_code": code,
        "trainer_terminal": trainer_terminal, "elapsed_s": time.time() - started,
        "validation_opened": False, "test_id_opened": False,
    }, OUT / "terminal.json")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
