from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch


def assert_no_cpu_fallback(*, device: str, no_cpu_fallback: bool) -> None:
    if no_cpu_fallback and str(device).lower() != "cuda":
        raise RuntimeError("CPU fallback is forbidden for v3 GPU production")


def _query_nvidia_smi() -> list[dict[str, Any]]:
    query = "index,name,uuid,memory.total,memory.free,utilization.gpu"
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_gpu_percent": int(parts[5]),
            }
        )
    return rows


def _query_gpu_processes() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    processes: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        processes.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": int(parts[3].replace(" MiB", "")),
            }
        )
    return processes


def check_cuda_preflight(
    *,
    require_cuda: bool,
    min_free_vram_gib: float,
    no_cpu_fallback: bool,
) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report: dict[str, Any] = {
        "require_cuda": bool(require_cuda),
        "cpu_fallback": not bool(no_cpu_fallback),
        "min_free_vram_gib": float(min_free_vram_gib),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "gpus": _query_nvidia_smi(),
        "processes": _query_gpu_processes(),
    }
    if require_cuda and not report["cuda_available"]:
        report.update({"status": "CUDA_UNAVAILABLE", "exit_code": 3})
        return report
    free_enough = [
        gpu
        for gpu in report["gpus"]
        if float(gpu["memory_free_mib"]) / 1024.0 >= float(min_free_vram_gib)
    ]
    if require_cuda and not free_enough:
        report.update({"status": "INSUFFICIENT_FREE_VRAM", "exit_code": 3})
        return report
    if no_cpu_fallback and not require_cuda:
        report.update({"status": "CONFIG_ERROR", "exit_code": 2})
        return report
    report.update({"status": "OK", "exit_code": 0, "selected_gpus": free_enough})
    return report


def has_usable_cuda(*, min_free_vram_gib: float = 1.0) -> bool:
    report = check_cuda_preflight(require_cuda=True, min_free_vram_gib=min_free_vram_gib, no_cpu_fallback=True)
    return int(report["exit_code"]) == 0


def write_preflight_report(path: str | Path, report: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
