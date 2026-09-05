#!/usr/bin/env python3
"""Quantify the train-only R40 representation floor for hard fit records."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("r40_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--fit-caches", type=Path, nargs="+", required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r40 = load_module(args.r40_script.resolve())
    fit = r40.FrequencyCacheCollection(
        [path.resolve() for path in args.fit_caches], expected_subset="fit"
    )
    curriculum = json.loads(args.curriculum.read_text(encoding="utf-8"))
    records = list(curriculum["records"])
    base_errors = np.asarray(
        [float(record["base_record_rel_l2"]) for record in records],
        dtype=np.float64,
    )
    positions = np.argsort(-base_errors, kind="stable")[: int(args.top)]
    weights = r40.rfft_weights(r40.TIME_COUNT)[fit.frequency_indices]
    rows = []
    for position_value in positions:
        position = int(position_value)
        file_index, local_index = fit.records[position]
        handle = fit.handles[file_index]
        residual = np.asarray(
            handle["residual_dct_norm"][local_index], dtype=np.float64
        )
        frequency_scale = np.asarray(
            handle["frequency_scale"][local_index], dtype=np.float64
        )
        target_square = float(handle["target_square_total"][local_index])
        retained = float(
            np.sum(
                weights
                * frequency_scale**2
                * np.sum(residual**2, axis=(1, 2, 3), dtype=np.float64)
            )
        )
        parent_total = float(base_errors[position]) ** 2 * target_square
        irreducible = max(parent_total - retained, 0.0)
        temporal_unselected = float(
            handle["base_error_square_unselected"][local_index]
        )
        rows.append(
            {
                "rank": len(rows) + 1,
                "record_position": position,
                "sample_id": fit.sample_ids[position],
                "group_id": fit.group_ids[position],
                "family": fit.families[position],
                "parent_rel_l2": float(base_errors[position]),
                "representation_oracle_rel_l2": math.sqrt(
                    irreducible / max(target_square, 1.0e-30)
                ),
                "temporal_unselected_floor_rel_l2": math.sqrt(
                    temporal_unselected / max(target_square, 1.0e-30)
                ),
                "retained_error_fraction": retained
                / max(parent_total, 1.0e-30),
            }
        )
    payload = {
        "schema": "r44_fit_representation_oracle_v1",
        "data_boundary": {
            "fit_truth_used": True,
            "opened_development_used": False,
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
            "paper_modified": False,
        },
        "count": len(rows),
        "parent_max": max(row["parent_rel_l2"] for row in rows),
        "oracle_max": max(row["representation_oracle_rel_l2"] for row in rows),
        "oracle_mean": float(
            np.mean([row["representation_oracle_rel_l2"] for row in rows])
        ),
        "all_oracle_lte_0p05": all(
            row["representation_oracle_rel_l2"] <= 0.05 for row in rows
        ),
        "records": rows,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}))
    fit.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
