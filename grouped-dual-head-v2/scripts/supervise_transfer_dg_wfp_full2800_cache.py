#!/usr/bin/env python3
"""Supervise four GPU shards for the complete 2800-record WFP cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/transfer_dg_wfp_full2800_manifest_20260902.json"
PREREG = ROOT / "results/transfer_dg_wfp_full2800_cache_preregistration_20260902.json"
OUTPUT = ROOT / "results/transfer_dg_wfp_full2800_cache_20260902"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload, path):
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.time()
    processes, logs, commands = [], [], []
    for gpu in range(4):
        command = [
            sys.executable, str(ROOT / "scripts/build_transfer_dg_wfp_e1_cache.py"),
            "--manifest", str(MANIFEST),
            "--output", str(OUTPUT / f"shard_{gpu}.h5"),
            "--shard-index", str(gpu), "--shard-count", "4", "--device", "cuda:0",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
        environment["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
        log = (OUTPUT / f"shard_{gpu}.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process); logs.append(log); commands.append(command)
    atomic_json({
        "schema": "transfer_dg_wfp_full2800_cache_identity_v1",
        "supervisor_pid": os.getpid(), "child_pids": [p.pid for p in processes],
        "commands": commands, "manifest_sha256": sha256(MANIFEST),
        "preregistration_sha256": sha256(PREREG), "started_unix_s": started,
        "training_full_truth_allowed": True, "test_future_truth_access": False,
        "validation_opened": False, "test_id_opened": False,
    }, OUTPUT / "run_identity.json")
    codes = [process.wait() for process in processes]
    for log in logs: log.close()
    summaries = []
    for gpu in range(4):
        path = OUTPUT / f"shard_{gpu}.h5.summary.json"
        if path.is_file(): summaries.append(json.loads(path.read_text()))
    complete = all(code == 0 for code in codes) and sum(int(row["record_count"]) for row in summaries) == 2800
    terminal = {
        "schema": "transfer_dg_wfp_full2800_cache_terminal_v1",
        "status": "complete" if complete else "failed",
        "return_codes": codes,
        "record_count": sum(int(row["record_count"]) for row in summaries),
        "output_bytes": sum(int(row["output_bytes"]) for row in summaries),
        "elapsed_s": time.time() - started,
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(terminal, OUTPUT / "terminal.json")
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
