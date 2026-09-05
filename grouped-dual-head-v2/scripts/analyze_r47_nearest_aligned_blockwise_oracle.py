#!/usr/bin/env python3
"""Implementation-aligned blockwise complex-gain capacity diagnostic.

This is a nondeployable diagnostic on the already-opened development split.
It uses the exact ceil-spaced blocks produced when an 8x8 gain grid is enlarged
to 201x201 with ``torch.nn.functional.interpolate(mode='nearest')``.
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


def blockwise_error(
    base: np.ndarray,
    truth: np.ndarray,
    frequency_weights: np.ndarray,
    blocks: int,
) -> float:
    height, width = base.shape[-2:]
    y_edges = nearest_edges(height, blocks)
    x_edges = nearest_edges(width, blocks)
    error_by_frequency = np.zeros(base.shape[0], dtype=np.float64)
    for iy in range(blocks):
        for ix in range(blocks):
            y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
            x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
            base_block = base[:, y0:y1, x0:x1]
            truth_block = truth[:, y0:y1, x0:x1]
            base_energy = np.sum(
                np.abs(base_block) ** 2, axis=(1, 2), dtype=np.float64
            )
            truth_energy = np.sum(
                np.abs(truth_block) ** 2, axis=(1, 2), dtype=np.float64
            )
            cross = np.sum(
                np.conj(base_block) * truth_block,
                axis=(1, 2),
                dtype=np.complex128,
            )
            optimal_error = truth_energy - np.abs(cross) ** 2 / np.maximum(
                base_energy, 1.0e-30
            )
            error_by_frequency += np.maximum(optimal_error, 0.0)
    return float(np.sum(frequency_weights * error_by_frequency, dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r40_path = args.r40_script.expanduser().resolve()
    r40 = load_module(r40_path, "r47_nearest_aligned_oracle_r40")
    cache = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    try:
        weights = r40.rfft_weights(r40.TIME_COUNT)[
            cache.frequency_indices
        ].astype(np.float64)
        block_counts = (1, 2, 4, 8, 16)
        rows = {blocks: [] for blocks in block_counts}
        for position, (file_index, local_index) in enumerate(cache.records):
            handle = cache.handles[file_index]
            base_complex = as_complex(
                handle["base_spectrum_selected"][local_index]
            )
            truth_complex = as_complex(
                handle["truth_spectrum_selected"][local_index]
            )
            identity_selected = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(base_complex - truth_complex) ** 2,
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
            parent_error = math.sqrt(
                (unselected + identity_selected) / max(target_total, 1.0e-30)
            )
            for blocks in block_counts:
                selected_error = blockwise_error(
                    base_complex, truth_complex, weights, blocks
                )
                candidate = math.sqrt(
                    (unselected + selected_error) / max(target_total, 1.0e-30)
                )
                rows[blocks].append(
                    {
                        "sample_id": cache.sample_ids[position],
                        "group_id": cache.group_ids[position],
                        "family": cache.families[position],
                        "parent_rel_l2": parent_error,
                        "candidate_rel_l2": candidate,
                        "relative_improvement": float(1.0 - candidate / parent_error),
                    }
                )
        results = {}
        for blocks, records in rows.items():
            stats = aggregate(records)
            results[f"{blocks}x{blocks}"] = {
                "spatial_blocks": blocks * blocks,
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
            "schema": "r47_nearest_aligned_blockwise_complex_gain_oracle_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "deployable": False,
            "purpose": "architecture capacity only; development truth is not a deployable input",
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
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "r47_nearest_aligned_oracle_complete",
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
