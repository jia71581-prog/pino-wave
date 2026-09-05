#!/usr/bin/env python3
"""Read-only tensor/storage audit for the frozen CREST parent checkpoint."""
from __future__ import annotations

from collections.abc import Mapping
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_summary(value: Any) -> dict[str, int]:
    tensor_count = 0
    element_count = 0
    tensor_bytes = 0

    def visit(item: Any) -> None:
        nonlocal tensor_count, element_count, tensor_bytes
        if torch.is_tensor(item):
            tensor_count += 1
            element_count += int(item.numel())
            tensor_bytes += int(item.numel() * item.element_size())
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "tensor_bytes": tensor_bytes,
    }


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch.load(
            str(checkpoint), map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, ValueError, RuntimeError):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    top_level = {
        str(key): {
            "python_type": type(value).__name__,
            **tensor_summary(value),
        }
        for key, value in payload.items()
    }
    model_key = next(
        (
            key
            for key in ("model_state", "model", "state_dict")
            if key in payload and isinstance(payload[key], Mapping)
        ),
        None,
    )
    if model_key is None:
        raise RuntimeError("no recognized model state mapping in checkpoint")
    model_summary = tensor_summary(payload[model_key])
    if model_summary["tensor_count"] < 1:
        raise RuntimeError("model state contains no tensors")
    report = {
        "schema": "crest_parent_checkpoint_storage_audit_v1",
        "status": "complete_read_only",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "file_bytes": checkpoint.stat().st_size,
        },
        "top_level_keys": sorted(str(key) for key in payload),
        "top_level_tensor_summaries": top_level,
        "model_state_key": model_key,
        "model_state": model_summary,
        "model_fp32_equivalent_bytes": model_summary["element_count"] * 4,
        "checkpoint_to_model_tensor_byte_ratio": checkpoint.stat().st_size
        / max(model_summary["tensor_bytes"], 1),
        "interpretation": (
            "File bytes describe the archived training checkpoint. Model tensor bytes "
            "describe the deployable learned state before serialization overhead; neither "
            "quantity is a finite-difference working-memory comparison."
        ),
    }
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite storage audit: {args.output}")
    atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
