#!/usr/bin/env python3
"""Nondeployable opened-development oracle for complex scalar gain capacity."""

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


def aggregate(records):
    values = np.asarray([row["candidate_rel_l2"] for row in records])
    parents = np.asarray([row["parent_rel_l2"] for row in records])
    return {
        "count": len(records),
        "candidate_mean": float(values.mean()),
        "candidate_max": float(values.max()),
        "parent_mean": float(parents.mean()),
        "parent_max": float(parents.max()),
        "mean_relative_improvement": float(1.0 - values.mean() / parents.mean()),
        "max_relative_improvement": float(1.0 - values.max() / parents.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r40_path = args.r40_script.expanduser().resolve()
    r40 = load_module(r40_path, "r47_gain_oracle_r40")
    holdout = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    try:
        weights = r40.rfft_weights(r40.TIME_COUNT)[
            holdout.frequency_indices
        ].astype(np.float64)
        variants = {
            "identity": [],
            "per_frequency_complex_oracle": [],
            "per_frequency_phase_oracle": [],
            "per_frequency_amplitude_oracle": [],
            "single_complex_gain_oracle": [],
        }
        for position, (file_index, local_index) in enumerate(holdout.records):
            handle = holdout.handles[file_index]
            base = as_complex(handle["base_spectrum_selected"][local_index])
            truth = as_complex(handle["truth_spectrum_selected"][local_index])
            base_energy = np.sum(np.abs(base) ** 2, axis=(1, 2), dtype=np.float64)
            truth_energy = np.sum(np.abs(truth) ** 2, axis=(1, 2), dtype=np.float64)
            cross = np.sum(
                np.conj(base) * truth, axis=(1, 2), dtype=np.complex128
            )
            unselected = float(handle["base_error_square_unselected"][local_index])
            target_total = float(handle["target_square_total"][local_index])
            gains = {
                "identity": np.ones_like(cross),
                "per_frequency_complex_oracle": cross
                / np.maximum(base_energy, 1.0e-30),
                "per_frequency_phase_oracle": np.exp(1j * np.angle(cross)),
                "per_frequency_amplitude_oracle": (
                    np.maximum(np.real(cross), 0.0)
                    / np.maximum(base_energy, 1.0e-30)
                ).astype(np.complex128),
            }
            scalar = np.sum(weights * cross, dtype=np.complex128) / max(
                float(np.sum(weights * base_energy, dtype=np.float64)), 1.0e-30
            )
            gains["single_complex_gain_oracle"] = np.full_like(cross, scalar)
            errors = {}
            for name, gain in gains.items():
                frequency_error = np.maximum(
                    np.abs(gain) ** 2 * base_energy
                    + truth_energy
                    - 2.0 * np.real(np.conj(gain) * cross),
                    0.0,
                )
                selected_error = float(
                    np.sum(weights * frequency_error, dtype=np.float64)
                )
                errors[name] = math.sqrt(
                    (unselected + selected_error) / max(target_total, 1.0e-30)
                )
            parent = errors["identity"]
            common = {
                "sample_id": str(holdout.sample_ids[position]),
                "group_id": str(holdout.group_ids[position]),
                "family": str(holdout.families[position]),
                "parent_rel_l2": parent,
            }
            for name in variants:
                variants[name].append(
                    {
                        **common,
                        "candidate_rel_l2": errors[name],
                        "relative_improvement": float(1.0 - errors[name] / parent),
                    }
                )

        results = {}
        for name, records in variants.items():
            metrics = aggregate(records)
            results[name] = {
                "aggregate": metrics,
                "absolute_goal": {
                    "mean_lte_0p05": metrics["candidate_mean"] <= 0.05,
                    "max_lte_0p05": metrics["candidate_max"] <= 0.05,
                    "passed": metrics["candidate_mean"] <= 0.05
                    and metrics["candidate_max"] <= 0.05,
                },
                "worst_record": max(
                    records, key=lambda row: row["candidate_rel_l2"]
                ),
                "records": records,
            }
        payload = {
            "schema": "r47_development_complex_gain_oracle_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "deployable": False,
            "role": "opened-development architecture-capacity diagnostic only",
            "selection_sha256": holdout.selection_sha256,
            "results": results,
            "data_boundary": {
                "opened_development_truth_used": True,
                "fit_truth_used": False,
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
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output)
        summary = {
            name: {
                "mean": result["aggregate"]["candidate_mean"],
                "max": result["aggregate"]["candidate_max"],
                "passed": result["absolute_goal"]["passed"],
                "worst_sample_id": result["worst_record"]["sample_id"],
            }
            for name, result in results.items()
        }
        print(
            json.dumps(
                {
                    "event": "r47_complex_gain_oracle_complete",
                    "results": summary,
                    "output": str(output),
                    "r29b_opened": False,
                    "paper_modified": False,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        holdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
