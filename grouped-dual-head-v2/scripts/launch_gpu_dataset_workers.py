#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fno_acoustic.data_generation.config import artifact_root, load_config
from fno_acoustic.data_generation.gpu_backend import check_cuda_preflight
from fno_acoustic.data_generation.hdf5_lwc84 import (
    build_lwc84_dataset_vds,
    validate_lwc84_shard,
)
from fno_acoustic.data_generation.holdout_audit import (
    audit_candidate_split_disjointness,
    audit_generated_sample_sha256_disjointness,
    audit_historical_manifest_disjointness,
)
from fno_acoustic.data_generation.lwc84_manifest import validate_lwc84_manifest
from fno_acoustic.data_generation.pipeline_lwc84 import (
    plan_dataset,
    sha256_file,
    storage_budget,
)


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _filter_manifest_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    selected_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    selected = [row for row in rows if str(row["split"]) in selected_splits]
    if args.medium_types:
        media = {value.strip() for value in args.medium_types.split(",") if value.strip()}
        selected = [row for row in selected if str(row["medium_type"]) in media]
    if args.sample_ids:
        sample_ids = {value.strip() for value in args.sample_ids.split(",") if value.strip()}
        selected = [row for row in selected if str(row["sample_id"]) in sample_ids]
    if not selected:
        raise ValueError("the requested filtered production scope is empty")
    return selected


def _dataset_vds_path(config: dict, output: Path, rows: list[dict]) -> Path:
    filename = Path(str(config["storage"]["vds_filename"]))
    if not bool(config["storage"].get("split_scoped_vds", False)):
        return output / filename
    splits = sorted(
        {str(row["split"]) for row in rows},
        key=("train", "validation", "test_id", "ood_canonical").index,
    )
    if not splits:
        raise ValueError("split-scoped VDS requires at least one selected split")
    scope = "_".join(splits)
    return output / f"{filename.stem}_{scope}{filename.suffix}"


def _collect_bound_shards(
    output: Path,
    rows: list[dict],
    *,
    config_sha256: str,
    manifest_sha256: str,
    marmousi_sha256: str,
) -> list[Path]:
    layouts = _expected_shard_layout(output, rows)
    selected_paths: list[Path] = []
    missing: list[Path] = []
    for path, split, shard_rows in layouts:
        if not path.is_file():
            missing.append(path)
            continue
        _validate_expected_shard(
            path,
            split=split,
            expected_ids=[str(row["sample_id"]) for row in shard_rows],
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            marmousi_sha256=marmousi_sha256,
            strict=True,
        )
        selected_paths.append(path)
    if missing:
        raise ValueError(
            f"selected production shards are incomplete: {len(missing)} missing, "
            f"first={[str(path) for path in missing[:3]]}"
        )
    return selected_paths


def _expected_shard_layout(
    output: Path, rows: list[dict], *, shard_size: int = 8
) -> list[tuple[Path, str, list[dict]]]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("selected production rows contain duplicate sample IDs")
    layouts: list[tuple[Path, str, list[dict]]] = []
    for split in ("train", "validation", "test_id", "ood_canonical"):
        split_rows = [row for row in rows if str(row["split"]) == split]
        for ordinal, start in enumerate(range(0, len(split_rows), int(shard_size))):
            layouts.append(
                (
                    output / "shards" / split / f"{split}-{ordinal:05d}.h5",
                    split,
                    split_rows[start : start + int(shard_size)],
                )
            )
    return layouts


def _validate_expected_shard(
    path: Path,
    *,
    split: str,
    expected_ids: list[str],
    config_sha256: str,
    manifest_sha256: str,
    marmousi_sha256: str,
    strict: bool,
) -> None:
    if strict:
        validate_lwc84_shard(path, strict=True, expected_n=len(expected_ids))
    with h5py.File(path, "r") as h5:
        actual_ids = [_decode(value) for value in h5["sample_id"][:]]
        completed = np.asarray(h5["completed_mask"][:], dtype=bool)
        actual_attrs = {
            name: _decode(h5.attrs.get(name, "<missing>"))
            for name in ("split", "config_sha256", "manifest_sha256", "marmousi_sha256")
        }
    expected_attrs = {
        "split": split,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "marmousi_sha256": marmousi_sha256,
    }
    if actual_ids != expected_ids:
        raise ValueError(
            f"shard ordinal/sample ordering mismatch for {path}: "
            f"actual={actual_ids[:3]}, expected={expected_ids[:3]}"
        )
    mismatched = {
        name: (actual_attrs[name], expected)
        for name, expected in expected_attrs.items()
        if actual_attrs[name] != expected
    }
    if mismatched:
        raise ValueError(f"shard provenance binding mismatch for {path}: {mismatched}")
    if completed.shape != (len(expected_ids),) or not bool(completed.all()):
        raise ValueError(f"finalized shard is incomplete: {path}")


def _completed_bound_sample_count(
    output: Path,
    rows: list[dict],
    *,
    config_sha256: str,
    manifest_sha256: str,
    marmousi_sha256: str,
) -> int:
    completed = 0
    for path, split, shard_rows in _expected_shard_layout(output, rows):
        if not path.is_file():
            continue
        expected_ids = [str(row["sample_id"]) for row in shard_rows]
        _validate_expected_shard(
            path,
            split=split,
            expected_ids=expected_ids,
            config_sha256=config_sha256,
            manifest_sha256=manifest_sha256,
            marmousi_sha256=marmousi_sha256,
            strict=False,
        )
        completed += len(expected_ids)
    return completed


def _normalized_worker_exit_code(return_codes: list[int]) -> int:
    return 0 if return_codes and all(code == 0 for code in return_codes) else 1


def _write_worker_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run_workers(specs: list[dict], *, status_path: Path) -> int:
    processes: list[dict] = []
    try:
        for spec in specs:
            log_path = Path(spec["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                spec["command"],
                env=spec["environment"],
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append({**spec, "process": process, "log_stream": log_stream})

        failure_rank: int | None = None
        while any(item["process"].poll() is None for item in processes):
            for item in processes:
                code = item["process"].poll()
                if code is not None and code != 0 and failure_rank is None:
                    failure_rank = int(item["rank"])
                    for other in processes:
                        if other["process"].poll() is None:
                            other["process"].terminate()
                    deadline = time.monotonic() + 10.0
                    for other in processes:
                        if other["process"].poll() is None:
                            timeout = max(0.0, deadline - time.monotonic())
                            try:
                                other["process"].wait(timeout=timeout)
                            except subprocess.TimeoutExpired:
                                other["process"].kill()
                    break
            if failure_rank is not None:
                break
            time.sleep(0.2)

        return_codes = [int(item["process"].wait()) for item in processes]
        if failure_rank is None:
            failure_rank = next(
                (
                    int(item["rank"])
                    for item, code in zip(processes, return_codes, strict=True)
                    if code != 0
                ),
                None,
            )
        payload = {
            "status": "COMPLETE" if all(code == 0 for code in return_codes) else "FAILED",
            "failure_rank": failure_rank,
            "workers": [
                {
                    "rank": int(item["rank"]),
                    "gpu_id": str(item["gpu_id"]),
                    "pid": int(item["process"].pid),
                    "return_code": code,
                    "signal": -code if code < 0 else None,
                    "log": str(Path(item["log_path"]).resolve()),
                }
                for item, code in zip(processes, return_codes, strict=True)
            ],
        }
        _write_worker_status(status_path, payload)
        if payload["status"] != "COMPLETE":
            print(json.dumps(payload, indent=2, sort_keys=True))
        return _normalized_worker_exit_code(return_codes)
    finally:
        for item in processes:
            item["log_stream"].close()


def _frozen_plan(config: dict, output: Path) -> tuple[dict, int]:
    manifest = output / "manifest.jsonl"
    frozen_config = output / "frozen_config.yaml"
    plan_summary_path = output / "plan_summary.json"
    if not manifest.is_file() or not frozen_config.is_file() or not plan_summary_path.is_file():
        raise FileNotFoundError("--frozen-manifest requires manifest, frozen_config, and plan_summary")
    frozen = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    if str(frozen.get("config_sha256", "")) != str(config["config_sha256"]):
        raise ValueError("frozen config SHA-256 differs from the requested config")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = validate_lwc84_manifest(rows, config)
    internal_split_audit = audit_candidate_split_disjointness(rows)
    if not bool(internal_split_audit["passed"]):
        raise ValueError("frozen manifest has internal split identity overlap")
    historical_manifests = config["dataset"].get("historical_manifests")
    holdout_audit = None
    if historical_manifests is not None:
        holdout_audit = audit_historical_manifest_disjointness(
            rows, historical_manifests
        )
        if not bool(holdout_audit["passed"]):
            raise ValueError("frozen manifest overlaps a historical manifest")
    manifest_hash = sha256_file(manifest)
    prior_plan = json.loads(plan_summary_path.read_text(encoding="utf-8"))
    if str(prior_plan.get("manifest_sha256", "")) != manifest_hash:
        raise ValueError("frozen manifest SHA-256 differs from plan_summary.json")
    return {
        "status": "FROZEN_MANIFEST_REUSED",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "manifest_summary": summary,
        "historical_overlap_audit": holdout_audit,
        "internal_split_overlap_audit": internal_split_audit,
    }, 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch one v3 GPU dataset worker per selected GPU.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--splits", default="train")
    parser.add_argument("--gpu-ids", default="auto")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--worker-ranks", default=None)
    parser.add_argument("--partition-worker-count", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--confirm-production", default=None)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--batch-config", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--autotune-batch", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--async-write", action="store_true")
    parser.add_argument("--run-convergence", action="store_true")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frozen-manifest", action="store_true")
    parser.add_argument("--medium-types", default=None)
    parser.add_argument("--sample-ids", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if str(config.get("schema_version", "")).startswith("acoustic-lwc84"):
        expected = str(config["production"]["confirm_token"])
        if args.confirm_production != expected:
            print(json.dumps({"status": "BLOCKED_CONFIRMATION", "required": expected}, indent=2))
            return 2
        output = Path(args.output or config["paths"]["output_root"])
        plan, plan_exit = (
            _frozen_plan(config, output)
            if args.frozen_manifest
            else plan_dataset(config, output=output)
        )
        all_rows = [
            json.loads(line)
            for line in Path(plan["manifest"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = _filter_manifest_rows(all_rows, args)
        if plan_exit != 0 and plan.get("status") != "BLOCKED_DISK":
            print(json.dumps(plan, indent=2, sort_keys=True))
            return plan_exit
        completed_samples = _completed_bound_sample_count(
            output,
            rows,
            config_sha256=str(config["config_sha256"]),
            manifest_sha256=str(plan["manifest_sha256"]),
            marmousi_sha256=str(config["marmousi"]["sha256"]),
        )
        pending_samples = len(rows) - completed_samples
        free_bytes = int(shutil.disk_usage(output).free)
        filtered_budget = (
            storage_budget(config, sample_count=pending_samples)
            if pending_samples > 0
            else None
        )
        disk_safety_passed = (
            filtered_budget is None
            or free_bytes >= int(filtered_budget["required_free_bytes"])
        )
        plan["filtered_generation"] = {
            "sample_count": len(rows),
            "completed_sample_count": completed_samples,
            "pending_sample_count": pending_samples,
            "storage_budget": filtered_budget,
            "disk_free_bytes": free_bytes,
            "disk_safety_passed": disk_safety_passed,
        }
        if not disk_safety_passed:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 3
        selected = args.devices or args.gpu_ids
        if selected == "auto":
            selected = "0"
        devices = [value.strip() for value in selected.split(",") if value.strip()]
        if not devices:
            raise ValueError("--devices must select at least one GPU")
        worker_ranks = (
            [int(value.strip()) for value in args.worker_ranks.split(",") if value.strip()]
            if args.worker_ranks
            else list(range(len(devices)))
        )
        partition_worker_count = int(args.partition_worker_count or len(devices))
        if len(worker_ranks) != len(devices):
            raise ValueError("--worker-ranks must contain one logical rank per device")
        if len(set(worker_ranks)) != len(worker_ranks):
            raise ValueError("--worker-ranks must be unique")
        if partition_worker_count <= 0 or any(
            rank < 0 or rank >= partition_worker_count for rank in worker_ranks
        ):
            raise ValueError("logical worker ranks must be within --partition-worker-count")
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{os.getpid()}"
        worker_root = output / "reports" / "worker_runs" / run_id
        specs: list[dict] = []
        for launch_index, (rank, gpu_id) in enumerate(zip(worker_ranks, devices, strict=True)):
            command = [
                sys.executable,
                str(_ROOT / "scripts" / "generate_acoustic_dataset.py"),
                "--config", args.config,
                "--preset", "production",
                "--device", "cuda",
                "--output", str(output),
                "--manifest", str(plan["manifest"]),
                "--worker-rank", str(rank),
                "--num-workers", str(partition_worker_count),
                "--confirm-production", expected,
                "--batch-size", str(args.batch_size),
                "--splits", args.splits,
            ]
            if args.resume:
                command.append("--resume")
            if args.medium_types:
                command.extend(["--medium-types", args.medium_types])
            if args.sample_ids:
                command.extend(["--sample-ids", args.sample_ids])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            specs.append(
                {
                    "rank": rank,
                    "launch_index": launch_index,
                    "gpu_id": gpu_id,
                    "command": command,
                    "environment": environment,
                    "log_path": worker_root / f"worker-{rank:03d}.log",
                }
            )
        if pending_samples > 0:
            worker_exit = _run_workers(specs, status_path=worker_root / "status.json")
            if worker_exit != 0:
                return worker_exit
        shard_paths = _collect_bound_shards(
            output,
            rows,
            config_sha256=str(config["config_sha256"]),
            manifest_sha256=str(plan["manifest_sha256"]),
            marmousi_sha256=str(config["marmousi"]["sha256"]),
        )
        dataset_path = _dataset_vds_path(config, output, rows)
        # _collect_bound_shards has already recomputed every sidecar SHA-256 and
        # checked exact ordinal/sample ordering, so avoid a second full-data read.
        build_lwc84_dataset_vds(dataset_path, shard_paths, strict_validation=False)
        sample_hash_audit = None
        historical_datasets = list(
            config["dataset"].get("historical_datasets") or []
        )
        additional_by_split = config["dataset"].get(
            "postgeneration_additional_historical_by_split", {}
        ) or {}
        selected_splits = {str(row["split"]) for row in rows}
        for split in selected_splits:
            historical_datasets.extend(additional_by_split.get(split, ()) or ())
        if historical_datasets:
            sample_hash_audit = audit_generated_sample_sha256_disjointness(
                dataset_path, historical_datasets
            )
            audit_scope = (
                "_".join(sorted(selected_splits))
                if bool(config["storage"].get("split_scoped_vds", False))
                else "all"
            )
            audit_path = output / f"postgeneration_sample_sha256_audit_{audit_scope}.json"
            _write_worker_status(audit_path, sample_hash_audit)
            if not bool(sample_hash_audit["passed"]):
                print(json.dumps(sample_hash_audit, indent=2, sort_keys=True))
                return 3
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "dataset_vds": str(dataset_path.resolve()),
                    "sample_count": len(rows),
                    "worker_status": str((worker_root / "status.json").resolve())
                    if pending_samples > 0
                    else None,
                    "sample_sha256_audit": sample_hash_audit,
                },
                indent=2,
            )
        )
        return 0
    if not args.profile:
        raise ValueError("legacy launcher requires --profile")
    report = check_cuda_preflight(require_cuda=True, min_free_vram_gib=float(config["compute"]["min_free_vram_gib"]), no_cpu_fallback=args.no_cpu_fallback)
    if int(report["exit_code"]) != 0:
        path = artifact_root(config) / "worker_launch_blocked.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "BLOCKED_GPU_RESOURCE", "preflight": report}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "BLOCKED_GPU_RESOURCE", "preflight": report}, indent=2, sort_keys=True))
        return 3
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "generate_acoustic_dataset.py"),
        "--config",
        args.config,
        "--profile",
        args.profile,
        "--splits",
        args.splits,
        "--device",
        "cuda",
        "--backend",
        args.backend,
    ]
    if args.cuda_graphs:
        cmd.append("--cuda-graphs")
    if args.no_cpu_fallback:
        cmd.append("--no-cpu-fallback")
    if args.resume:
        cmd.append("--resume")
    proc = subprocess.run(cmd, text=True, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
