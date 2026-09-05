#!/usr/bin/env python3
"""Prune historical training artifacts while retaining research best weights.

The script is deliberately scoped to the two legacy artifact roots below.  It
never traverses the current ``results`` tree (which contains the Helmholtz
checkpoints) or any dataset root.

Policy:

* preserve every A3, B2-H/B2H, and Helmholtz weight;
* retain one ``best.pt`` per non-smoke research run;
* prefer an existing ``best.pt``, then ``terminal.json:best_checkpoint``, then
  the lowest evaluated aggregate relative L2, then ``latest.pt``;
* remove redundant latest/periodic checkpoints from those runs;
* remove engineering smoke/memory/superseded/invalid checkpoint weights;
* remove large prediction tensors and the legacy multifidelity teacher cache;
* retain configs, metrics, logs, terminal reports, figures, and small traces.

Run without ``--apply`` for a read-only plan.  ``--apply`` first creates any
missing best links, writes the audit report, and only then unlinks the exact
planned files.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOTS = (
    Path("/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts"),
    Path("/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1"),
)
DEFAULT_REPORT = Path(
    "/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/"
    "grouped-dual-head-v2/results/storage_cleanup_20260808"
)
PROTECTED_MARKERS = (
    "/b2h/",
    "/b2h_",
    "/b2hm_",
    "temporal_latent_a3",
    "/a3_",
    "helmholtz",
)
ENGINEERING_MARKERS = (
    "/smoke/",
    "_smoke/",
    "smoke_",
    "/memory_probe/",
    "/memscan_",
    "/capacity_attempts/",
    "superseded",
    "invalid_",
    "/v39_gpu22g_tuning/",
    "/v40_dense_unfreeze_memory/",
    "cuda_smoke",
)
PAYLOAD_NAMES = {"sealed_predictions.pt", "fields.pt"}


@dataclass(frozen=True)
class Selection:
    run_dir: Path
    source: Path
    best_path: Path
    reason: str
    metric: float | None


def is_within_roots(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in ROOTS)


def normalized(path: Path) -> str:
    return "/" + path.as_posix().lower().lstrip("/") + "/"


def protected(path: Path) -> bool:
    value = normalized(path)
    return any(marker in value for marker in PROTECTED_MARKERS)


def engineering_only(path: Path) -> bool:
    value = normalized(path)
    return any(marker in value for marker in ENGINEERING_MARKERS)


def is_periodic(path: Path) -> bool:
    name = path.name.lower()
    return path.parent.name == "checkpoints" or name.startswith(
        ("epoch_", "update_", "step_", "lbfgs_step_")
    )


def is_conventional_weight(path: Path) -> bool:
    name = path.name.lower()
    return name in {"best.pt", "latest.pt", "last.pt"} or is_periodic(path)


def run_dir_for(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def checkpoint_magic_ok(path: Path) -> bool:
    if path.stat().st_size <= 0:
        return False
    with path.open("rb") as handle:
        prefix = handle.read(4)
    return prefix.startswith(b"PK") or prefix.startswith(b"\x80")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def metric_candidates(run_dir: Path, weights: set[Path]) -> list[tuple[float, Path]]:
    candidates: list[tuple[float, Path]] = []
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return candidates
    try:
        lines = metrics_path.read_text().splitlines()
    except OSError:
        return candidates
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        checkpoint = row.get("checkpoint")
        metrics = row.get("metrics")
        if not isinstance(checkpoint, str) or not isinstance(metrics, dict):
            continue
        value = metrics.get("aggregate_relative_l2")
        if not isinstance(value, (int, float)):
            continue
        candidate = Path(checkpoint)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        candidate = candidate.resolve(strict=False)
        if candidate in weights and candidate.is_file():
            candidates.append((float(value), candidate))
    return candidates


def select_best(run_dir: Path, weights: set[Path]) -> Selection:
    existing_best = run_dir / "best.pt"
    if existing_best in weights:
        return Selection(run_dir, existing_best, existing_best, "existing_best", None)

    terminal = read_json(run_dir / "terminal.json")
    if terminal:
        raw = terminal.get("best_checkpoint")
        if isinstance(raw, str) and raw:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            candidate = candidate.resolve(strict=False)
            if candidate in weights and candidate.is_file():
                metric = terminal.get("best_fixed_aggregate_relative_l2")
                return Selection(
                    run_dir,
                    candidate,
                    existing_best,
                    "terminal_best_checkpoint",
                    float(metric) if isinstance(metric, (int, float)) else None,
                )

    measured = metric_candidates(run_dir, weights)
    if measured:
        metric, candidate = min(measured, key=lambda item: item[0])
        return Selection(run_dir, candidate, existing_best, "minimum_evaluated_metric", metric)

    latest = [p for p in weights if p.parent == run_dir and p.name in {"latest.pt", "last.pt"}]
    if latest:
        candidate = sorted(latest)[0]
        return Selection(run_dir, candidate, existing_best, "latest_fallback", None)

    numbered = sorted(weights, key=lambda p: p.name)
    candidate = numbered[-1]
    return Selection(run_dir, candidate, existing_best, "numbered_fallback", None)


def inode_key(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def apparent_bytes(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def physical_reclaim_bytes(all_files: list[Path], delete_paths: set[Path], selections: list[Selection]) -> int:
    inode_paths: dict[tuple[int, int], list[Path]] = defaultdict(list)
    inode_stat: dict[tuple[int, int], os.stat_result] = {}
    for path in all_files:
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        inode_paths[key].append(path)
        inode_stat[key] = stat

    planned_link_inodes = {
        inode_key(selection.source)
        for selection in selections
        if selection.best_path != selection.source and not selection.best_path.exists()
    }
    reclaimed = 0
    for key, paths in inode_paths.items():
        stat = inode_stat[key]
        known_deleted_links = sum(path in delete_paths for path in paths)
        outside_links = max(0, stat.st_nlink - len(paths))
        remaining_known = len(paths) - known_deleted_links
        if remaining_known == 0 and outside_links == 0 and key not in planned_link_inodes:
            reclaimed += stat.st_blocks * 512
    return reclaimed


def build_plan() -> dict[str, Any]:
    for root in ROOTS:
        if not root.is_dir() or root.resolve() != root:
            raise RuntimeError(f"unsafe or missing cleanup root: {root}")

    all_files = sorted(path for root in ROOTS for path in root.rglob("*") if path.is_file())
    pt_files = {path.resolve() for path in all_files if path.suffix.lower() == ".pt"}
    conventional = {path for path in pt_files if is_conventional_weight(path)}

    grouped: dict[Path, set[Path]] = defaultdict(set)
    for path in conventional:
        grouped[run_dir_for(path)].add(path)

    delete_paths: set[Path] = set()
    selections: list[Selection] = []
    protected_weights: set[Path] = set()
    engineering_weights: set[Path] = set()

    for run_dir, weights in sorted(grouped.items()):
        if protected(run_dir):
            protected_weights.update(weights)
            continue
        if engineering_only(run_dir):
            delete_paths.update(weights)
            engineering_weights.update(weights)
            continue
        selection = select_best(run_dir, weights)
        if not checkpoint_magic_ok(selection.source):
            raise RuntimeError(f"selected checkpoint has invalid magic: {selection.source}")
        selections.append(selection)
        delete_paths.update(weights - {selection.best_path})

    payloads = {
        path
        for path in pt_files
        if path.name in PAYLOAD_NAMES or "predictions" in path.parts
    }
    delete_paths.update(path for path in payloads if not protected(path))

    unconventional_engineering = {
        path
        for path in pt_files - conventional - payloads
        if engineering_only(path) and not protected(path)
    }
    delete_paths.update(unconventional_engineering)

    teacher_cache = {
        path.resolve()
        for path in all_files
        if path.suffix.lower() == ".h5"
        and "lwc84_multifidelity_teacher64_401solver_v2" in path.parts
    }
    delete_paths.update(teacher_cache)

    backup_weights = {
        path.resolve()
        for path in all_files
        if path.name.endswith(".pt..bak") and not protected(path)
    }
    delete_paths.update(backup_weights)

    if any(not is_within_roots(path) for path in delete_paths):
        raise RuntimeError("plan contains a deletion target outside the fixed legacy roots")
    if any(protected(path) for path in delete_paths):
        raise RuntimeError("plan contains a protected A3/B2-H/Helmholtz target")

    all_resolved = [path.resolve() for path in all_files]
    expected_reclaim = physical_reclaim_bytes(all_resolved, delete_paths, selections)
    return {
        "roots": [str(root) for root in ROOTS],
        "all_file_count": len(all_files),
        "all_physical_bytes": sum(path.stat().st_blocks * 512 for path in all_files),
        "checkpoint_run_count": len(grouped),
        "selected_best_count": len(selections),
        "protected_weight_count": len(protected_weights),
        "engineering_weight_delete_count": len(engineering_weights),
        "payload_delete_count": len(payloads - protected_weights),
        "teacher_cache_delete_count": len(teacher_cache),
        "backup_weight_delete_count": len(backup_weights),
        "delete_file_count": len(delete_paths),
        "delete_apparent_bytes": apparent_bytes(delete_paths),
        "expected_physical_reclaim_bytes": expected_reclaim,
        "selections": [
            {
                "run_dir": str(item.run_dir),
                "source": str(item.source),
                "best_path": str(item.best_path),
                "reason": item.reason,
                "metric": item.metric,
                "size_bytes": item.source.stat().st_size,
            }
            for item in sorted(selections, key=lambda item: str(item.run_dir))
        ],
        "protected_weights": [str(path) for path in sorted(protected_weights)],
        "delete_paths": [str(path) for path in sorted(delete_paths)],
    }


def apply_plan(plan: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    before_path = report_dir / "cleanup_plan_before_apply.json"
    before_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    created_links: list[str] = []
    for raw in plan["selections"]:
        source = Path(raw["source"])
        best_path = Path(raw["best_path"])
        if source == best_path or best_path.exists():
            continue
        if not source.is_file() or not checkpoint_magic_ok(source):
            raise RuntimeError(f"best source changed before apply: {source}")
        best_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, best_path)
        created_links.append(str(best_path))

    deleted: list[str] = []
    for raw in plan["delete_paths"]:
        path = Path(raw)
        if not path.exists():
            continue
        if not is_within_roots(path) or protected(path) or not path.is_file():
            raise RuntimeError(f"refusing changed/unsafe deletion target: {path}")
        path.unlink()
        deleted.append(str(path))

    for root in ROOTS:
        directories = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            if directory.name in {"checkpoints", "predictions", "shards"}:
                try:
                    directory.rmdir()
                except OSError:
                    pass

    retained_best = []
    invalid_best = []
    for raw in plan["selections"]:
        best_path = Path(raw["best_path"])
        if best_path.is_file() and checkpoint_magic_ok(best_path):
            retained_best.append(str(best_path))
        else:
            invalid_best.append(str(best_path))

    remaining_payloads = [
        str(path)
        for root in ROOTS
        for path in root.rglob("*.pt")
        if (path.name in PAYLOAD_NAMES or "predictions" in path.parts) and not protected(path)
    ]
    remaining_teacher_cache = [
        str(path)
        for root in ROOTS
        for path in root.rglob("*.h5")
        if "lwc84_multifidelity_teacher64_401solver_v2" in path.parts
    ]
    result = {
        "status": "PASS" if not invalid_best and not remaining_payloads and not remaining_teacher_cache else "FAIL",
        "created_best_links": created_links,
        "deleted_file_count": len(deleted),
        "deleted_paths": deleted,
        "retained_selected_best_count": len(retained_best),
        "invalid_or_missing_selected_best": invalid_best,
        "remaining_nonprotected_payloads": remaining_payloads,
        "remaining_teacher_cache": remaining_teacher_cache,
    }
    (report_dir / "cleanup_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    plan = build_plan()
    summary = {key: value for key, value in plan.items() if key not in {"selections", "protected_weights", "delete_paths"}}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.apply:
        return 0

    result = apply_plan(plan, args.report_dir)
    print(json.dumps({key: value for key, value in result.items() if key not in {"deleted_paths", "created_best_links"}}, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
