#!/usr/bin/env python3
"""Wait for the full background cache and then build four eikonal/energy shards."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/transfer_dg_wfp_full2800_cache_20260902"
OUTPUT = ROOT / "results/transfer_dg_wfp_full2800_travel_20260902"


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    queue = ROOT / "results/transfer_dg_wfp_full2800_travel_queue_20260902.json"
    atomic_json({
        "schema": "transfer_dg_wfp_full2800_travel_queue_v1",
        "pid": os.getpid(), "state": "waiting_for_cache",
        "cache_terminal": str(CACHE / "terminal.json"),
        "training_full_truth_allowed": True, "test_future_truth_access": False,
        "validation_opened": False, "test_id_opened": False,
    }, queue)
    while not (CACHE / "terminal.json").is_file():
        time.sleep(60)
    terminal = json.loads((CACHE / "terminal.json").read_text())
    if terminal.get("status") != "complete" or int(terminal.get("record_count", -1)) != 2800:
        atomic_json({"schema":"transfer_dg_wfp_full2800_travel_terminal_v1","status":"blocked_by_cache","cache_terminal":terminal}, ROOT / "results/transfer_dg_wfp_full2800_travel_terminal_20260902.json")
        return 2
    OUTPUT.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    for index in range(4):
        command = [
            sys.executable, str(ROOT / "scripts/build_transfer_dg_wfp_e1d_travel.py"),
            "--cache", str(CACHE / f"shard_{index}.h5"),
            "--output", str(OUTPUT / f"shard_{index}.h5"),
        ]
        environment = os.environ.copy()
        environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
        environment["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
        log = (OUTPUT / f"shard_{index}.log").open("w")
        processes.append(subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT))
        logs.append(log)
    codes = [process.wait() for process in processes]
    for log in logs: log.close()
    summaries = []
    for index in range(4):
        path = OUTPUT / f"shard_{index}.h5.summary.json"
        if path.is_file(): summaries.append(json.loads(path.read_text()))
    complete = all(code == 0 for code in codes) and sum(int(row["record_count"]) for row in summaries) == 2800
    result = {
        "schema": "transfer_dg_wfp_full2800_travel_terminal_v1",
        "status": "complete" if complete else "failed", "return_codes": codes,
        "record_count": sum(int(row["record_count"]) for row in summaries),
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(result, ROOT / "results/transfer_dg_wfp_full2800_travel_terminal_20260902.json")
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
