#!/usr/bin/env python3
"""Launch and supervise four isolated GPU cache builders for WFP E1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/transfer_dg_wfp_e1_manifest_256_20260902.json"
PREREG = ROOT / "results/transfer_dg_wfp_e1_preregistration_20260902.json"
OUTPUT = ROOT / "results/transfer_dg_wfp_e1_cache_20260902"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    commands = []
    processes = []
    logs = []
    started = time.time()
    for gpu in range(4):
        shard = OUTPUT / f"shard_{gpu}.h5"
        command = [
            sys.executable,
            str(ROOT / "scripts/build_transfer_dg_wfp_e1_cache.py"),
            "--manifest", str(MANIFEST),
            "--output", str(shard),
            "--shard-index", str(gpu),
            "--shard-count", "4",
            "--device", "cuda:0",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
        environment["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
        log_path = OUTPUT / f"shard_{gpu}.log"
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        commands.append(command)
        processes.append(process)
        logs.append(log)
    identity = {
        "schema": "transfer_dg_wfp_e1_cache_supervisor_identity_v1",
        "supervisor_pid": os.getpid(),
        "child_pids": [process.pid for process in processes],
        "commands": commands,
        "manifest_sha256": sha256(MANIFEST),
        "preregistration_sha256": sha256(PREREG),
        "started_unix_s": started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(identity, OUTPUT / "run_identity.json")
    return_codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    summaries = []
    if all(code == 0 for code in return_codes):
        for gpu in range(4):
            summaries.append(
                json.loads((OUTPUT / f"shard_{gpu}.h5.summary.json").read_text())
            )
    terminal = {
        "schema": "transfer_dg_wfp_e1_cache_supervisor_terminal_v1",
        "status": "complete" if all(code == 0 for code in return_codes) else "failed",
        "return_codes": return_codes,
        "record_count": sum(int(row["record_count"]) for row in summaries),
        "output_bytes": sum(int(row["output_bytes"]) for row in summaries),
        "elapsed_s": time.time() - started,
        "summaries": {
            str(OUTPUT / f"shard_{gpu}.h5.summary.json"): sha256(
                OUTPUT / f"shard_{gpu}.h5.summary.json"
            )
            for gpu in range(4)
            if (OUTPUT / f"shard_{gpu}.h5.summary.json").is_file()
        },
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(terminal, OUTPUT / "terminal.json")
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    return 0 if terminal["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
