#!/usr/bin/env python3
"""Keep one best/latest artifact per run and prune checkpoint histories safely."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def build_plan(results_root: Path, protected_prefixes: tuple[str, ...]) -> dict:
    root = results_root.resolve()
    if root.name != "results" or not root.is_dir():
        raise RuntimeError("cleanup root must be an existing directory named results")
    groups: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            continue
        relative = path.relative_to(root)
        if "checkpoints" not in relative.parts:
            continue
        position = relative.parts.index("checkpoints")
        run = root.joinpath(*relative.parts[:position])
        groups.setdefault(run, []).append(path.resolve())

    kept, deleted = [], []
    for run, files in sorted(groups.items(), key=lambda item: str(item[0])):
        relative_run = str(run.relative_to(root))
        if any(
            relative_run == prefix or relative_run.startswith(f"{prefix}/")
            for prefix in protected_prefixes
        ):
            kept.extend(files)
            continue
        preferred = (run / "latest.pt", run / "best.pt", run / "checkpoints/best.pt")
        retained = next(
            (path.resolve() for path in preferred if path.exists()),
            max(files, key=lambda path: path.stat().st_mtime),
        )
        if retained in files:
            kept.append(retained)
        for path in files:
            if path != retained:
                if root not in path.parents or "checkpoints" not in path.relative_to(root).parts:
                    raise RuntimeError(f"unsafe deletion candidate: {path}")
                deleted.append(path)
    return {
        "root": str(root),
        "group_count": len(groups),
        "protected_prefixes": list(protected_prefixes),
        "kept": [str(path) for path in kept],
        "delete": [str(path) for path in deleted],
        "delete_bytes": sum(path.stat().st_size for path in deleted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protect-prefix", action="append", default=[])
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.audit_output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.audit_output}")
    plan = build_plan(args.results_root, tuple(args.protect_prefix))
    records = []
    if args.execute:
        for raw in plan["delete"]:
            path = Path(raw)
            records.append(
                {
                    "path": raw,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            path.unlink()
    payload = {
        "schema": "intermediate_checkpoint_prune_audit_v1",
        "status": "executed" if args.execute else "dry_run",
        "policy": "retain root latest.pt, else root best.pt, else checkpoints/best.pt, else newest checkpoint; prune only other files below a checkpoints directory",
        "root": plan["root"],
        "group_count": plan["group_count"],
        "protected_prefixes": plan["protected_prefixes"],
        "deleted_count": len(plan["delete"]) if args.execute else 0,
        "planned_delete_count": len(plan["delete"]),
        "deleted_bytes": sum(row["bytes"] for row in records),
        "planned_delete_bytes": plan["delete_bytes"],
        "kept_count": len(plan["kept"]),
        "deleted": records,
    }
    _atomic_json(payload, args.audit_output)
    print(json.dumps({key: value for key, value in payload.items() if key != "deleted"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
