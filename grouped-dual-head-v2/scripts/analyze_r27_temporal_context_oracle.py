#!/usr/bin/env python3
"""Read-only temporal-context oracle on train-only R25 holdout caches.

For each record, solve a per-record least-squares finite impulse response filter
that maps neighboring coarse LWC84 frames to the high-fidelity frame.  Because
the coefficients use the record's truth, this is not a deployable method or an
accuracy claim.  It is only a representation diagnostic: if even this oracle
cannot pass the target, a learned temporal-context model is not worth training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def shifted(value: torch.Tensor, offset: int) -> torch.Tensor:
    """Return output[t] = value[t + offset], zero outside the time axis."""

    result = torch.zeros_like(value)
    if offset > 0:
        result[:-offset] = value[offset:]
    elif offset < 0:
        result[-offset:] = value[:offset]
    else:
        result.copy_(value)
    return result


def summarize(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows]
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "mean_lte_0p05": bool(float(np.mean(values)) <= 0.05),
        "max_lte_0p05": bool(float(np.max(values)) <= 0.05),
        "passed": bool(float(np.mean(values)) <= 0.05 and float(np.max(values)) <= 0.05),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 3, 7])
    args = parser.parse_args()

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("the temporal oracle requires CUDA")
    radii = sorted(set(int(value) for value in args.radii))
    if not radii or radii[0] < 0:
        raise ValueError("radii must be nonnegative")
    max_radius = max(radii)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False

    paths = [path.expanduser().resolve() for path in args.cache]
    rows: list[dict] = []
    cache_hashes: dict[str, str] = {}
    selection_sha256: str | None = None
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        cache_hashes[str(path)] = sha256_file(path)
        with h5py.File(path, "r", swmr=True) as handle:
            if str(handle.attrs.get("subset", "")) != "holdout":
                raise RuntimeError(f"not a holdout cache: {path}")
            if str(handle.attrs.get("truth_policy", "")) != "train_only_supervision_not_deployment_input":
                raise RuntimeError(f"truth policy mismatch: {path}")
            selection = str(handle.attrs.get("selection_sha256", ""))
            if selection_sha256 is None:
                selection_sha256 = selection
            elif selection_sha256 != selection:
                raise RuntimeError("selection digests disagree")
            if int(handle["coarse_norm"].shape[1]) != 401:
                raise RuntimeError("oracle requires all 401 holdout frames")

            for local_index in range(int(handle["coarse_norm"].shape[0])):
                sample_id = str(handle["sample_id"].asstr()[local_index])
                family = str(handle["family"].asstr()[local_index])
                coarse = torch.from_numpy(
                    np.asarray(handle["coarse_norm"][local_index], dtype=np.float32)
                ).to(device)
                truth = torch.from_numpy(
                    np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
                ).to(device)
                offsets = list(range(-max_radius, max_radius + 1))
                basis = torch.stack([shifted(coarse, offset) for offset in offsets])
                flat_basis = basis.reshape(len(offsets), -1)
                flat_truth = truth.reshape(-1)
                gram = (flat_basis @ flat_basis.T).double().cpu().numpy()
                rhs = (flat_basis @ flat_truth).double().cpu().numpy()
                target_square = float(torch.sum(truth.double().square()).cpu())
                parent_square = float(torch.sum((coarse - truth).double().square()).cpu())
                row = {
                    "sample_id": sample_id,
                    "family": family,
                    "parent_rel_l2": math.sqrt(parent_square / max(target_square, 1.0e-30)),
                    "oracles": {},
                }
                for radius in radii:
                    left = max_radius - radius
                    right = max_radius + radius + 1
                    local_gram = gram[left:right, left:right]
                    local_rhs = rhs[left:right]
                    ridge = max(float(np.trace(local_gram)) / max(local_gram.shape[0], 1), 1.0) * 1.0e-8
                    coefficients = np.linalg.solve(
                        local_gram + ridge * np.eye(local_gram.shape[0]), local_rhs
                    )
                    coefficient_tensor = torch.as_tensor(
                        coefficients, dtype=torch.float32, device=device
                    )
                    candidate = torch.sum(
                        coefficient_tensor[:, None, None, None] * basis[left:right], dim=0
                    )
                    error_square = float(
                        torch.sum((candidate - truth).double().square()).cpu()
                    )
                    row["oracles"][str(radius)] = {
                        "offsets": list(range(-radius, radius + 1)),
                        "coefficients": [float(value) for value in coefficients],
                        "rel_l2": math.sqrt(error_square / max(target_square, 1.0e-30)),
                    }
                    del candidate, coefficient_tensor
                rows.append(row)
                del basis, flat_basis, flat_truth, coarse, truth
                torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "family": family,
                            "parent_rel_l2": row["parent_rel_l2"],
                            "oracle_rel_l2": {
                                key: value["rel_l2"] for key, value in row["oracles"].items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    aggregate = {
        str(radius): summarize(
            [dict(row, value=row["oracles"][str(radius)]["rel_l2"]) for row in rows],
            "value",
        )
        for radius in radii
    }
    per_family = {
        family: {
            str(radius): summarize(
                [
                    dict(row, value=row["oracles"][str(radius)]["rel_l2"])
                    for row in rows
                    if row["family"] == family
                ],
                "value",
            )
            for radius in radii
        }
        for family in sorted({row["family"] for row in rows})
    }
    payload = {
        "schema": "r27_temporal_context_oracle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "train_holdout_representation_diagnostic_not_deployable_accuracy",
        "selection_sha256": selection_sha256,
        "cache_sha256": cache_hashes,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "validation_opened": False,
        "test_id_opened": False,
        "truth_used_for_per_record_oracle_coefficients": True,
        "radii": radii,
        "aggregate": aggregate,
        "per_family": per_family,
        "records": rows,
    }
    atomic_json(payload, args.output.expanduser().resolve())
    print(json.dumps({"event": "R27_TEMPORAL_ORACLE_COMPLETE", "aggregate": aggregate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
