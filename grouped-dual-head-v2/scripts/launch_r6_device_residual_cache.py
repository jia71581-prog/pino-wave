#!/usr/bin/env python3
"""Launch four detached, non-restarting R6 device-cache workers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from build_r6_device_residual_cache import CANDIDATE, cache_output, predicted_cache_bytes, verify_stage_inputs  # noqa: E402
from reattest_frozen_fine_grid_r6_train import atomic_write_bytes, report_bytes_fixed_point, sha256  # noqa: E402


def disk_gate(predicted_bytes: int, free_bytes: int, *, after: bool = False) -> bool:
    minimum = 40 * 2**30 if after else int(predicted_bytes) + 40 * 2**30
    return int(free_bytes) >= minimum and int(predicted_bytes) <= 28 * 2**30


def child_argv(preregistration: Path, shard: int) -> list[str]:
    return [sys.executable, str(ROOT / "scripts/build_r6_device_residual_cache.py"), "--worker",
            "--preregistration", str(preregistration), "--shard-index", str(shard),
            "--device", "cuda:0"]


def validate_launch(prereg: Mapping[str, Any], preregistration: Path) -> None:
    if (prereg.get("schema") != "r6_device_residual_cache_preregistration_v1"
            or prereg.get("candidate") != CANDIDATE or prereg.get("status") != "cache_build_authorized"):
        raise RuntimeError("cache launch status/candidate rejected")
    if preregistration.resolve() != (ROOT / prereg["paths"]["preregistration"]).resolve():
        raise RuntimeError("cache preregistration path override")
    runtime = prereg["prerequisites"]["runtime"]
    path = (ROOT / runtime["path"]).resolve()
    if runtime.get("status") != "passed" or sha256(path) != runtime.get("sha256"):
        raise RuntimeError("passed device runtime prerequisite missing")
    smoke = prereg["prerequisites"]["smoke"]
    smoke_path = (ROOT / smoke["path"]).resolve()
    if smoke.get("status") != "passed" or sha256(smoke_path) != smoke.get("sha256"):
        raise RuntimeError("passed cache smoke prerequisite missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--preregistration", type=Path, required=True)
    args = parser.parse_args(); prereg = json.loads(args.preregistration.read_text()); validate_launch(prereg, args.preregistration)
    stage = ROOT / prereg["paths"]["stage_dir"]; supervisor = stage / "launch_supervisor.json"
    if supervisor.exists() or (stage / "build_terminal.json").exists(): raise FileExistsError("cache launch output exists")
    before = verify_stage_inputs(prereg)
    for shard in range(4):
        for role in ("fit", "development"):
            if cache_output(stage, role, shard, smoke=False).exists(): raise FileExistsError("cache shard output exists")
    predicted = predicted_cache_bytes(); free = __import__("shutil").disk_usage(stage.parent).free
    if not disk_gate(predicted, free): raise RuntimeError("cache prelaunch disk gate")
    compute = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True).stdout.strip()
    if compute: raise RuntimeError("GPU compute process already active")
    (stage / "logs").mkdir(parents=True, exist_ok=True); (stage / "identities").mkdir(parents=True, exist_ok=True)
    children = []
    for shard in range(4):
        argv = child_argv(args.preregistration.resolve(), shard); log_path = stage / "logs" / f"worker_{shard}.log"
        environment = os.environ.copy(); environment.pop("CUBLAS_WORKSPACE_CONFIG", None); environment["CUDA_VISIBLE_DEVICES"] = str(shard)
        log = log_path.open("xb")
        process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        identity = {"schema": "r6_device_residual_cache_worker_identity_v1", "candidate": CANDIDATE,
                    "shard_index": shard, "physical_gpu": shard, "pid": process.pid,
                    "argv": argv, "log": str(log_path), "preregistration_sha256": sha256(args.preregistration),
                    "restart_authorized": False, "kill_authorized": False}
        identity_path = stage / "identities" / f"worker_{shard}.json"
        atomic_write_bytes(report_bytes_fixed_point(identity), identity_path)
        children.append({"shard_index": shard, "pid": process.pid, "argv": argv, "log": str(log_path),
                         "identity": str(identity_path), "alive_after_spawn": process.poll() is None})
        log.close()
    deadline = time.monotonic() + 30.0; verification = None
    while time.monotonic() < deadline:
        live = all(child["pid"] and __import__("os").path.exists(f"/proc/{child['pid']}") for child in children)
        logs = all(Path(child["log"]).is_file() and "R6_DEVICE_CACHE_WORKER_START" in Path(child["log"]).read_text(errors="replace") for child in children)
        progress = all(cache_output(stage, "fit", shard, smoke=False).with_suffix(".progress.json").exists() for shard in range(4))
        raw = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True).stdout
        gpu_pids = {int(value.strip()) for value in raw.splitlines() if value.strip().isdigit()}
        gpu = all(child["pid"] in gpu_pids for child in children)
        verification = {"live": live, "logs_have_start_event": logs, "fit_progress_exists": progress,
                        "GPU_processes_include_all_children": gpu, "nvidia_compute_pids": sorted(gpu_pids)}
        if all((live, logs, progress, gpu)): break
        time.sleep(.5)
    if verification is None or not all((verification["live"], verification["logs_have_start_event"],
                                        verification["fit_progress_exists"], verification["GPU_processes_include_all_children"])):
        failure = {"schema": "r6_device_residual_cache_launch_supervisor_v1", "status": "failed",
                   "candidate": CANDIDATE, "children": children, "verification": verification,
                   "no_restart": True, "no_kill": True, "resources": {"output_bytes": 0}}
        atomic_write_bytes(report_bytes_fixed_point(failure), supervisor); raise RuntimeError("bounded worker verification failure")
    after = verify_stage_inputs(prereg)
    if before != after: raise RuntimeError("cache launcher input hash drift")
    payload = {"schema": "r6_device_residual_cache_launch_supervisor_v1", "status": "launched",
               "candidate": CANDIDATE, "children": children, "predicted_cache_bytes": predicted,
               "free_bytes_before": free, "no_restart": True, "no_kill": True,
               "input_hashes_before": before, "input_hashes_after": after,
               "input_hashes_unchanged": True, "verification": verification,
               "promotion_authorized": False, "validation_opened": False, "test_id_opened": False,
               "resources": {"output_bytes": 0}}
    atomic_write_bytes(report_bytes_fixed_point(payload), supervisor); return 0


def entrypoint() -> int:
    try:
        return main()
    except Exception as error:
        try:
            index = sys.argv.index("--preregistration") + 1
            prereg = json.loads(Path(sys.argv[index]).read_text())
            supervisor = ROOT / prereg["paths"]["launch_supervisor"]
            if not supervisor.exists():
                failure = {"schema": "r6_device_residual_cache_launch_supervisor_v1", "status": "failed",
                           "candidate": CANDIDATE, "error": repr(error), "no_restart": True, "no_kill": True,
                           "promotion_authorized": False, "resources": {"output_bytes": 0}}
                atomic_write_bytes(report_bytes_fixed_point(failure), supervisor)
        except Exception:
            pass
        raise


if __name__ == "__main__": raise SystemExit(entrypoint())
