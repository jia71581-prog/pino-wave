#!/usr/bin/env python3
"""Cap-limited 8x8 complex-correction capacity on opened development.

This nondeployable diagnostic selects a safe correction-gain range before R47
training.  It never accesses R29B, final validation, test data, or manuscripts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_complex(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array[:, 0] + 1j * array[:, 1]


def nearest_edges(size: int, blocks: int) -> np.ndarray:
    indices = np.arange(blocks + 1, dtype=np.int64)
    return (indices * int(size) + int(blocks) - 1) // int(blocks)


def cap_limited_errors(
    base: np.ndarray,
    truth: np.ndarray,
    frequency_weights: np.ndarray,
    *,
    blocks: int,
    caps: tuple[float, ...],
) -> dict[float, float]:
    height, width = base.shape[-2:]
    y_edges = nearest_edges(height, blocks)
    x_edges = nearest_edges(width, blocks)
    errors = {
        cap: np.zeros(base.shape[0], dtype=np.float64) for cap in caps
    }
    for iy in range(blocks):
        for ix in range(blocks):
            y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
            x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
            base_block = base[:, y0:y1, x0:x1]
            truth_block = truth[:, y0:y1, x0:x1]
            denominator = np.sum(
                np.abs(base_block) ** 2, axis=(1, 2), dtype=np.float64
            )
            residual_block = truth_block - base_block
            delta = np.sum(
                np.conj(base_block) * residual_block,
                axis=(1, 2),
                dtype=np.complex128,
            ) / np.maximum(denominator, 1.0e-30)
            for cap in caps:
                clipped = np.clip(delta.real, -cap, cap) + 1j * np.clip(
                    delta.imag, -cap, cap
                )
                prediction = base_block + clipped[:, None, None] * base_block
                errors[cap] += np.sum(
                    np.abs(prediction - truth_block) ** 2,
                    axis=(1, 2),
                    dtype=np.float64,
                )
    return {
        cap: float(np.sum(frequency_weights * value, dtype=np.float64))
        for cap, value in errors.items()
    }


def aggregate(records: list[dict]) -> dict:
    candidates = np.asarray(
        [row["candidate_rel_l2"] for row in records], dtype=np.float64
    )
    parents = np.asarray(
        [row["parent_rel_l2"] for row in records], dtype=np.float64
    )
    return {
        "count": len(records),
        "candidate_mean": float(candidates.mean()),
        "candidate_max": float(candidates.max()),
        "parent_mean": float(parents.mean()),
        "parent_max": float(parents.max()),
        "mean_relative_improvement": float(1.0 - candidates.mean() / parents.mean()),
        "max_relative_improvement": float(1.0 - candidates.max() / parents.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r40_path = args.r40_script.expanduser().resolve()
    r40 = load_module(r40_path, "r47_gain_cap_oracle_r40")
    cache = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    caps = (0.25, 0.5, 1.0, 2.0)
    records_by_cap = {cap: [] for cap in caps}
    try:
        weights = r40.rfft_weights(r40.TIME_COUNT)[
            cache.frequency_indices
        ].astype(np.float64)
        for position, (file_index, local_index) in enumerate(cache.records):
            handle = cache.handles[file_index]
            base = as_complex(handle["base_spectrum_selected"][local_index])
            truth = as_complex(handle["truth_spectrum_selected"][local_index])
            identity_selected = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(base - truth) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            unselected = float(
                handle["base_error_square_unselected"][local_index]
            )
            target_total = float(handle["target_square_total"][local_index])
            parent = math.sqrt(
                (unselected + identity_selected) / max(target_total, 1.0e-30)
            )
            selected_by_cap = cap_limited_errors(
                base, truth, weights, blocks=8, caps=caps
            )
            for cap in caps:
                candidate = math.sqrt(
                    (unselected + selected_by_cap[cap])
                    / max(target_total, 1.0e-30)
                )
                records_by_cap[cap].append(
                    {
                        "sample_id": str(cache.sample_ids[position]),
                        "group_id": str(cache.group_ids[position]),
                        "family": str(cache.families[position]),
                        "parent_rel_l2": parent,
                        "candidate_rel_l2": candidate,
                        "relative_improvement": float(1.0 - candidate / parent),
                    }
                )
        results = {}
        for cap, records in records_by_cap.items():
            stats = aggregate(records)
            results[str(cap)] = {
                "gain_cap_per_real_imag_component": cap,
                "aggregate": stats,
                "absolute_goal": {
                    "mean_lte_0p05": stats["candidate_mean"] <= 0.05,
                    "max_lte_0p05": stats["candidate_max"] <= 0.05,
                    "passed": stats["candidate_mean"] <= 0.05
                    and stats["candidate_max"] <= 0.05,
                },
                "worst_record": max(
                    records, key=lambda row: row["candidate_rel_l2"]
                ),
                "records": records,
            }
        payload = {
            "schema": "r47_gain_cap_oracle_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "deployable": False,
            "purpose": "freeze a sufficient R47 gain cap before training",
            "block_grid": 8,
            "block_contract": "ceil_spaced_nonoverlapping_nearest_upsampling_alignment",
            "selection_sha256": cache.selection_sha256,
            "results": results,
            "data_boundary": {
                "fit_truth_used": False,
                "opened_development_truth_used": True,
                "r29b_opened": False,
                "final_validation_opened": False,
                "test_id_opened": False,
                "paper_modified": False,
            },
            "evidence": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "r40_script": str(r40_path),
                "r40_script_sha256": sha256_file(r40_path),
            },
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "r47_gain_cap_oracle_complete",
                    "results": {
                        key: value["aggregate"] for key, value in results.items()
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
