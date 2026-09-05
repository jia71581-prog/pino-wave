#!/usr/bin/env python3
"""Create an explicitly uncalibrated train-only CPADC parent-transfer candidate."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebind_trainonly_transfer(
    source: str | Path,
    output: str | Path,
    *,
    parent_checkpoint: str | Path,
) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    parent_path = Path(parent_checkpoint).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a CPADC transfer candidate")
    if not source_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("source CPADC and target parent checkpoints are required")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "causal_physics_aligned_defect_correction_v1":
        raise ValueError("source is not a CPADC basis checkpoint")
    if int(payload.get("schema_version", 0)) not in (4, 5, 6, 7, 8, 9):
        raise ValueError("unsupported CPADC transfer schema")
    if not isinstance(payload.get("basis_state"), dict):
        raise ValueError("source CPADC checkpoint lacks basis_state")

    source_digest = sha256_file(source_path)
    parent_digest = sha256_file(parent_path)
    old_parent = str(payload.get("parent_checkpoint", ""))
    old_parent_digest = str(payload.get("parent_checkpoint_sha256", ""))
    solve = dict(payload.get("online_solve_contract") or {})
    original_solve = dict(solve)
    solve["name"] = "ridge_direction_learned_energy_ball_projection_v1"
    solve.pop("minimum_unconstrained_correction_ratio", None)
    solve.pop("minimum_unconstrained_correction_ratio_by_family", None)
    payload["online_solve_contract"] = solve
    payload["risk_calibration"] = {}
    payload["parent_checkpoint"] = str(parent_path)
    payload["parent_checkpoint_sha256"] = parent_digest
    payload["trainonly_parent_transfer"] = {
        "status": "uncalibrated_candidate_not_for_validation_or_test",
        "basis_weights_modified": False,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_digest,
        "source_parent_checkpoint": old_parent,
        "source_parent_checkpoint_sha256": old_parent_digest,
        "target_parent_checkpoint": str(parent_path),
        "target_parent_checkpoint_sha256": parent_digest,
        "original_online_solve_contract": original_solve,
        "transferred_online_solve_contract": solve,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    try:
        torch.save(payload, partial)
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "source_sha256": source_digest,
        "parent_sha256": parent_digest,
        "basis_rank": int(payload["basis_rank"]),
        "phase_rank": int(payload["phase_rank"]),
        "status": "uncalibrated_trainonly_transfer",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    args = parser.parse_args(argv)
    print(
        rebind_trainonly_transfer(
            args.source,
            args.output,
            parent_checkpoint=args.parent_checkpoint,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

