#!/usr/bin/env python3
"""Development-only capacity of spatial-DCT block complex gains.

The diagnostic asks whether a low-dimensional correction indexed by temporal
frequency and spatial-wavenumber block can express the missing accuracy.  It is
nondeployable because each development record uses its own truth-derived gain.
R29B, final validation, test data, and manuscripts remain frozen.
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


def complex_channels(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[1] != 2:
        raise ValueError(f"expected [frequency,2,y,x], got {array.shape}")
    return array[:, 0] + 1j * array[:, 1]


def spectral_block_error(
    base: np.ndarray,
    truth: np.ndarray,
    frequency_weights: np.ndarray,
    blocks: int,
) -> float:
    height, width = base.shape[-2:]
    if height % blocks != 0 or width % blocks != 0:
        raise ValueError("spectral block count must divide retained DCT geometry")
    y_edges = np.arange(blocks + 1, dtype=np.int64) * (height // blocks)
    x_edges = np.arange(blocks + 1, dtype=np.int64) * (width // blocks)
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
            error_by_frequency += np.maximum(
                truth_energy
                - np.abs(cross) ** 2 / np.maximum(base_energy, 1.0e-30),
                0.0,
            )
    return float(np.sum(frequency_weights * error_by_frequency, dtype=np.float64))


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
    r40 = load_module(r40_path, "r48_spectral_oracle_r40")
    holdout = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    block_counts = (1, 2, 4, 8, 16)
    records_by_blocks = {blocks: [] for blocks in block_counts}
    parent_contract_differences = []
    try:
        weights = r40.rfft_weights(r40.TIME_COUNT)[
            holdout.frequency_indices
        ].astype(np.float64)
        for position, (file_index, local_index) in enumerate(holdout.records):
            handle = holdout.handles[file_index]
            base_normalized = complex_channels(
                handle["base_dct_norm"][local_index]
            )
            residual_normalized = complex_channels(
                handle["residual_dct_norm"][local_index]
            )
            scale = np.asarray(
                handle["frequency_scale"][local_index], dtype=np.float64
            )[:, None, None]
            base = base_normalized * scale
            residual = residual_normalized * scale
            truth = base + residual
            base_spatial = complex_channels(
                handle["base_spectrum_selected"][local_index]
            )
            truth_spatial = complex_channels(
                handle["truth_spectrum_selected"][local_index]
            )
            unselected = float(
                handle["base_error_square_unselected"][local_index]
            )
            target_total = float(handle["target_square_total"][local_index])
            parent_selected = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(residual) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            parent_selected_full = float(
                np.sum(
                    weights
                    * np.sum(
                        np.abs(base_spatial - truth_spatial) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            omitted_spatial_error = max(
                parent_selected_full - parent_selected, 0.0
            )
            parent = math.sqrt(
                (unselected + parent_selected_full) / max(target_total, 1.0e-30)
            )
            reconstructed_parent = math.sqrt(
                (unselected + omitted_spatial_error + parent_selected)
                / max(target_total, 1.0e-30)
            )
            parent_contract_differences.append(abs(parent - reconstructed_parent))
            common = {
                "sample_id": str(holdout.sample_ids[position]),
                "group_id": str(holdout.group_ids[position]),
                "family": str(holdout.families[position]),
                "parent_rel_l2": parent,
            }
            for blocks in block_counts:
                selected_error = spectral_block_error(
                    base, truth, weights, blocks
                )
                candidate = math.sqrt(
                    (unselected + omitted_spatial_error + selected_error)
                    / max(target_total, 1.0e-30)
                )
                records_by_blocks[blocks].append(
                    {
                        **common,
                        "candidate_rel_l2": candidate,
                        "relative_improvement": float(1.0 - candidate / parent),
                    }
                )

        results = {}
        for blocks, records in records_by_blocks.items():
            stats = aggregate(records)
            results[f"{blocks}x{blocks}"] = {
                "spatial_dct_blocks": blocks * blocks,
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
            "schema": "r48_development_spectral_block_complex_gain_oracle_v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "deployable": False,
            "purpose": "capacity diagnostic for a fit-only low-variance spatial-wavenumber correction",
            "selection_sha256": holdout.selection_sha256,
            "retained_dct": int(holdout.retained),
            "maximum_parent_energy_contract_difference": float(
                max(parent_contract_differences, default=0.0)
            ),
            "unmodeled_error_policy": "preserve temporal-unselected and spatial-DCT-truncated parent error exactly",
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
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "event": "r48_development_spectral_block_oracle_complete",
                    "results": {
                        key: value["aggregate"] for key, value in results.items()
                    },
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        holdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
