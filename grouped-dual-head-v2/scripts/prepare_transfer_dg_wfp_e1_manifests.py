#!/usr/bin/env python3
"""Register all train records and select a diverse 256-record WFP E1 panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
DEFAULT_POOL = ROOT / "results/transfer_dg_wfp_train_pool_2800_20260902.json"
DEFAULT_E1 = ROOT / "results/transfer_dg_wfp_e1_manifest_256_20260902.json"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _farthest(features: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count > len(features):
        raise ValueError("invalid farthest-point count")
    values = np.asarray(features, dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    values = (values - values.mean(axis=0)) / scale
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(values)))
    selected = [first]
    distance = np.sum((values - values[first]) ** 2, axis=1)
    distance[first] = -1.0
    while len(selected) < count:
        maximum = float(distance.max())
        candidates = np.flatnonzero(np.isclose(distance, maximum))
        index = int(candidates[int(rng.integers(0, len(candidates)))])
        selected.append(index)
        distance = np.minimum(distance, np.sum((values - values[index]) ** 2, axis=1))
        distance[np.asarray(selected, dtype=np.int64)] = -1.0
    return np.asarray(selected, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pool-output", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--e1-output", type=Path, default=DEFAULT_E1)
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument("--per-family", type=int, default=64)
    parser.add_argument("--holdout-per-family", type=int, default=16)
    args = parser.parse_args()
    if not args.source_h5.is_file():
        raise FileNotFoundError(args.source_h5)
    if args.per_family <= args.holdout_per_family or args.holdout_per_family <= 0:
        raise ValueError("E1 requires positive holdout smaller than each family panel")

    with h5py.File(args.source_h5, "r", swmr=True) as source:
        split = source["split"].asstr()[:]
        medium = np.asarray(
            [value.split("_")[0] for value in source["medium_type"].asstr()[:]],
            dtype=str,
        )
        sample_id = source["sample_id"].asstr()[:]
        group_id = source["group_id"].asstr()[:]
        sample_sha = source["sample_sha256"].asstr()[:]
        f0 = np.asarray(source["source_f0_hz"], dtype=np.float64)
        sx = np.asarray(source["source_x_m"], dtype=np.float64)
        sz = np.asarray(source["source_z_m"], dtype=np.float64)
        vmin = np.asarray(source["vmin_mps"], dtype=np.float64)
        vmax = np.asarray(source["vmax_mps"], dtype=np.float64)
        crop_x = np.asarray(source["crop_x0_m"], dtype=np.float64)
        crop_z = np.asarray(source["crop_z0_m"], dtype=np.float64)
        manifest_sha = str(source.attrs["manifest_sha256"])

    train_indices = np.flatnonzero(split == "train")
    if len(train_indices) != 2800:
        raise RuntimeError(f"registered train census changed: {len(train_indices)}")

    def row(index: int, *, role: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_index": int(index),
            "sample_id": str(sample_id[index]),
            "sample_sha256": str(sample_sha[index]),
            "group_id": str(group_id[index]),
            "family": str(medium[index]),
            "source_f0_hz": float(f0[index]),
            "source_x_m": float(sx[index]),
            "source_z_m": float(sz[index]),
            "vmin_mps": float(vmin[index]),
            "vmax_mps": float(vmax[index]),
            "crop_x0_m": None if not np.isfinite(crop_x[index]) else float(crop_x[index]),
            "crop_z0_m": None if not np.isfinite(crop_z[index]) else float(crop_z[index]),
        }
        if role is not None:
            payload["role"] = role
        return payload

    pool_rows = [row(int(index)) for index in train_indices]
    pool_payload: dict[str, object] = {
        "schema": "transfer_dg_wfp_train_pool_v1",
        "source_h5": str(args.source_h5.resolve()),
        "source_h5_sha256": _sha256(args.source_h5),
        "source_manifest_sha256": manifest_sha,
        "split": "train",
        "record_count": len(pool_rows),
        "family_counts": {
            family: int(sum(item["family"] == family for item in pool_rows))
            for family in FAMILIES
        },
        "unique_group_count": len({item["group_id"] for item in pool_rows}),
        "records": pool_rows,
        "validation_opened": False,
        "test_id_opened": False,
    }
    pool_payload["selection_sha256"] = _canonical_sha(pool_rows)
    _atomic_json(pool_payload, args.pool_output)

    selected_rows: list[dict[str, object]] = []
    family_summary: dict[str, object] = {}
    for family_index, family in enumerate(FAMILIES):
        candidates = train_indices[medium[train_indices] == family]
        # One deterministic representative per group prevents source siblings
        # from crossing the E1 fit/holdout boundary.
        by_group: dict[str, list[int]] = {}
        for index in candidates.tolist():
            by_group.setdefault(str(group_id[index]), []).append(int(index))
        representatives = []
        rng = np.random.default_rng(args.seed + 1009 * family_index)
        for group in sorted(by_group):
            members = by_group[group]
            representatives.append(members[int(rng.integers(0, len(members)))])
        representatives = np.asarray(representatives, dtype=np.int64)
        if len(representatives) < args.per_family:
            raise RuntimeError(f"{family} has too few distinct train groups")
        features = np.stack(
            (
                f0[representatives],
                sx[representatives],
                sz[representatives],
                vmin[representatives],
                vmax[representatives],
                np.nan_to_num(crop_x[representatives]),
                np.nan_to_num(crop_z[representatives]),
            ),
            axis=1,
        )
        chosen = representatives[
            _farthest(features, args.per_family, args.seed + 7919 * family_index)
        ]
        holdout_positions = set(
            np.linspace(
                0,
                args.per_family - 1,
                args.holdout_per_family,
                dtype=np.int64,
            ).tolist()
        )
        family_rows = []
        for position, source_index in enumerate(chosen.tolist()):
            role = "holdout" if position in holdout_positions else "fit"
            item = row(source_index, role=role)
            family_rows.append(item)
            selected_rows.append(item)
        family_summary[family] = {
            "fit": sum(item["role"] == "fit" for item in family_rows),
            "holdout": sum(item["role"] == "holdout" for item in family_rows),
            "unique_groups": len({item["group_id"] for item in family_rows}),
            "f0_min_hz": min(item["source_f0_hz"] for item in family_rows),
            "f0_max_hz": max(item["source_f0_hz"] for item in family_rows),
            "source_x_min_m": min(item["source_x_m"] for item in family_rows),
            "source_x_max_m": max(item["source_x_m"] for item in family_rows),
            "vmin_mps": min(item["vmin_mps"] for item in family_rows),
            "vmax_mps": max(item["vmax_mps"] for item in family_rows),
        }

    fit_groups = {item["group_id"] for item in selected_rows if item["role"] == "fit"}
    holdout_groups = {
        item["group_id"] for item in selected_rows if item["role"] == "holdout"
    }
    if fit_groups & holdout_groups:
        raise RuntimeError("E1 fit and holdout groups overlap")
    e1_payload: dict[str, object] = {
        "schema": "transfer_dg_wfp_e1_manifest_v1",
        "source_h5": str(args.source_h5.resolve()),
        "source_h5_sha256": pool_payload["source_h5_sha256"],
        "source_manifest_sha256": manifest_sha,
        "parent_pool": str(args.pool_output.resolve()),
        "parent_pool_selection_sha256": pool_payload["selection_sha256"],
        "split": "train",
        "seed": args.seed,
        "selection": "group-distinct standardized farthest-point coverage over f0, source position, velocity range and crop location",
        "record_count": len(selected_rows),
        "fit_count": sum(item["role"] == "fit" for item in selected_rows),
        "holdout_count": sum(item["role"] == "holdout" for item in selected_rows),
        "family_summary": family_summary,
        "fit_holdout_group_overlap_count": 0,
        "records": selected_rows,
        "validation_opened": False,
        "test_id_opened": False,
    }
    e1_payload["selection_sha256"] = _canonical_sha(selected_rows)
    _atomic_json(e1_payload, args.e1_output)
    print(json.dumps({
        "pool": str(args.pool_output),
        "pool_count": len(pool_rows),
        "e1": str(args.e1_output),
        "e1_count": len(selected_rows),
        "fit_count": e1_payload["fit_count"],
        "holdout_count": e1_payload["holdout_count"],
        "family_summary": family_summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
