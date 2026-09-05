#!/usr/bin/env python3
"""Build one preregistration-bound train-only B2 cache with terminal records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_b2_snapshot_ic import build_cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    if args.cache.exists():
        raise FileExistsError(f"refusing to reuse cache: {args.cache}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    try:
        manifest = json.loads(args.manifest.read_text())
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if manifest.get("split") != "train":
            raise RuntimeError("cache manifest must be train-only")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        if _sha256(Path(__file__)) != bindings["cache_builder_sha256"]:
            raise RuntimeError("cache builder binding drift")
        if _sha256(args.manifest) != bindings["manifest_sha256"][str(args.manifest)]:
            raise RuntimeError("manifest binding drift")
        build_cache(manifest, torch.device("cuda"), cache_path=args.cache, data_split="train")
        identity = {
            "schema": "b2_bound_cache_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "manifest_selection_sha256": manifest["selection_sha256"],
            "cache": str(args.cache),
            "cache_sha256": _sha256(args.cache),
            "cache_builder_sha256": _sha256(Path(__file__)),
            "record_count": len(manifest["records"]),
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        _atomic_json({
            "status": "complete",
            "cache": str(args.cache),
            "cache_sha256": identity["cache_sha256"],
            "run_identity": str(args.output_dir / "run_identity.json"),
        }, terminal)
        print(json.dumps(json.loads(terminal.read_text()), indent=2))
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
