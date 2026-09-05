#!/usr/bin/env python3
"""Nearest-source-group block-gain transfer probe for a possible R49.

R48 established that per-record oracle block gains pass the 0.05 gate while
record-independent global/family fit gains transfer nothing.  The untested
middle ground is spatial transfer: per-fit-group gains moved to a development
record from the source-position-nearest fit groups of the same family.  Donor
selection uses only deployment-available metadata (source position); gains are
estimated solely from audited fit-union truth.  The opened group-disjoint
development set is used only to diagnose transfer.  R29B, final validation,
test data, and manuscript files remain frozen.

Outputs:
  * dev-side gate table for knn donor policies (blocks x smoothing x scale)
  * dev-side single-donor rank ladder (transfer quality vs donor distance)
  * fit-side leave-one-group-out decorrelation curve in retained-DCT space
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


def block_sufficient_statistics(base, truth, blocks):
    frequency_count, height, width = base.shape
    if height % blocks != 0 or width % blocks != 0:
        raise ValueError("block count must divide retained DCT geometry")
    block_height, block_width = height // blocks, width // blocks
    base_blocks = base.reshape(
        frequency_count, blocks, block_height, blocks, block_width
    ).transpose(0, 1, 3, 2, 4)
    truth_blocks = truth.reshape(
        frequency_count, blocks, block_height, blocks, block_width
    ).transpose(0, 1, 3, 2, 4)
    base_energy = np.sum(np.abs(base_blocks) ** 2, axis=(-2, -1), dtype=np.float64)
    truth_energy = np.sum(np.abs(truth_blocks) ** 2, axis=(-2, -1), dtype=np.float64)
    cross = np.sum(
        np.conj(base_blocks) * truth_blocks, axis=(-2, -1), dtype=np.complex128
    )
    return base_energy, truth_energy, cross


def smooth_delta(gain: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(gain, dtype=np.complex128).copy()
    if width % 2 != 1:
        raise ValueError("smoothing width must be odd")
    delta = np.asarray(gain, dtype=np.complex128) - 1.0
    radius = width // 2
    padded = np.pad(delta, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    smoothed = np.zeros_like(delta)
    for offset in range(width):
        smoothed += padded[offset : offset + len(delta)]
    return 1.0 + smoothed / float(width)


def bounded_gain(gain: np.ndarray, *, smooth: int, cap: float) -> np.ndarray:
    value = smooth_delta(gain, smooth)
    delta = value - 1.0
    clipped = np.clip(delta.real, -cap, cap) + 1j * np.clip(delta.imag, -cap, cap)
    return 1.0 + clipped


def parse_position(group_id: str):
    parts = group_id.split(":")
    if len(parts) != 4:
        return None
    x_token, z_token = parts[2], parts[3]
    if not x_token.startswith("x") or not z_token.startswith("z"):
        return None
    return float(x_token[1:]), float(z_token[1:])


def aggregate(records):
    candidates = np.asarray(
        [row["candidate_rel_l2"] for row in records], dtype=np.float64
    )
    parents = np.asarray([row["parent_rel_l2"] for row in records], dtype=np.float64)
    return {
        "count": len(records),
        "candidate_mean": float(candidates.mean()),
        "candidate_median": float(np.median(candidates)),
        "candidate_max": float(candidates.max()),
        "parent_mean": float(parents.mean()),
        "parent_max": float(parents.max()),
        "mean_relative_improvement": float(1.0 - candidates.mean() / parents.mean()),
        "max_relative_improvement": float(1.0 - candidates.max() / parents.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--union-script", type=Path, required=True)
    parser.add_argument("--base-fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--supplement-fit-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    r40_path = args.r40_script.expanduser().resolve()
    union_path = args.union_script.expanduser().resolve()
    r40 = load_module(r40_path, "r49_transfer_r40")
    union_module = load_module(union_path, "r49_transfer_union")
    fit = union_module.UnionFrequencyCacheCollection(
        r40,
        [
            [path.expanduser().resolve() for path in args.base_fit_cache],
            [path.expanduser().resolve() for path in args.supplement_fit_cache],
        ],
        expected_subset="fit",
    )
    holdout = r40.FrequencyCacheCollection(
        [path.expanduser().resolve() for path in args.holdout_cache],
        expected_subset="holdout",
    )
    block_counts = (8, 16)
    smoothing_widths = (1, 3)
    correction_scales = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    donor_policies = ("knn1", "knn3_idw", "knn8_idw")
    rank_ladder_depth = 8
    gain_cap = 0.5
    distance_epsilon_m = 1.0
    try:
        if str(fit.component_selection_sha256[0]) != str(holdout.selection_sha256):
            raise RuntimeError("base fit/holdout selection mismatch")
        if set(fit.group_ids) & set(holdout.group_ids):
            raise RuntimeError("fit/holdout group leakage")
        if int(fit.retained) != int(holdout.retained):
            raise RuntimeError("fit/holdout retained DCT mismatch")
        if not np.array_equal(fit.frequency_indices, holdout.frequency_indices):
            raise RuntimeError("fit/holdout frequency mismatch")

        weights = r40.rfft_weights(r40.TIME_COUNT)[holdout.frequency_indices].astype(
            np.float64
        )

        group_num = {}
        group_den = {}
        group_position = {}
        fit_marmousi_records = []
        for position, (file_index, local_index) in enumerate(fit.records):
            group_id = str(fit.group_ids[position])
            coordinates = parse_position(group_id)
            if coordinates is None:
                continue
            handle = fit.handles[file_index]
            base = complex_channels(handle["base_dct_norm"][local_index])
            residual = complex_channels(handle["residual_dct_norm"][local_index])
            scale = np.asarray(
                handle["frequency_scale"][local_index], dtype=np.float64
            )[:, None, None]
            base = base * scale
            truth = base + residual * scale
            group_position[group_id] = coordinates
            if group_id not in group_num:
                group_num[group_id] = {
                    blocks: np.zeros(
                        (fit.frequency_count, blocks, blocks), dtype=np.complex128
                    )
                    for blocks in block_counts
                }
                group_den[group_id] = {
                    blocks: np.zeros(
                        (fit.frequency_count, blocks, blocks), dtype=np.float64
                    )
                    for blocks in block_counts
                }
            record_statistics = {}
            for blocks in block_counts:
                base_energy, truth_energy, cross = block_sufficient_statistics(
                    base, truth, blocks
                )
                group_num[group_id][blocks] += cross
                group_den[group_id][blocks] += base_energy
                record_statistics[blocks] = (base_energy, truth_energy, cross)
            fit_marmousi_records.append(
                {
                    "group_id": group_id,
                    "sample_id": str(fit.sample_ids[position]),
                    "statistics": record_statistics,
                    "truth_weighted_energy": float(
                        np.sum(
                            weights * np.sum(np.abs(truth) ** 2, axis=(1, 2)),
                            dtype=np.float64,
                        )
                    ),
                }
            )

        positioned_groups = sorted(group_position)
        if not positioned_groups:
            raise RuntimeError("no positioned fit groups found")

        def donor_gain(
            target_position,
            *,
            blocks,
            neighbor_count,
            inverse_distance,
            exclude_group=None,
        ):
            candidates = []
            for group_id in positioned_groups:
                if group_id == exclude_group:
                    continue
                gx, gz = group_position[group_id]
                distance = math.hypot(gx - target_position[0], gz - target_position[1])
                candidates.append((distance, group_id))
            candidates.sort(key=lambda item: (item[0], item[1]))
            selected = candidates[:neighbor_count]
            numerator = np.zeros(
                (fit.frequency_count, blocks, blocks), dtype=np.complex128
            )
            denominator = np.zeros(
                (fit.frequency_count, blocks, blocks), dtype=np.float64
            )
            donors = []
            for distance, group_id in selected:
                weight = (
                    1.0 / (distance + distance_epsilon_m) if inverse_distance else 1.0
                )
                numerator += weight * group_num[group_id][blocks]
                denominator += weight * group_den[group_id][blocks]
                donors.append({"group_id": group_id, "distance_m": distance})
            return numerator / np.maximum(denominator, 1.0e-30), donors

        policy_settings = {
            "knn1": {"neighbor_count": 1, "inverse_distance": False},
            "knn3_idw": {"neighbor_count": 3, "inverse_distance": True},
            "knn8_idw": {"neighbor_count": 8, "inverse_distance": True},
        }

        specifications = []
        for policy in donor_policies:
            for blocks in block_counts:
                for smooth in smoothing_widths:
                    for correction_scale in correction_scales:
                        specifications.append(
                            {
                                "donor_policy": policy,
                                "blocks": blocks,
                                "smoothing_width": smooth,
                                "gain_cap": gain_cap,
                                "correction_scale": correction_scale,
                                "records": [],
                            }
                        )
        rank_records = []
        parent_contract_differences = []
        development_donor_map = {}
        for position, (file_index, local_index) in enumerate(holdout.records):
            handle = holdout.handles[file_index]
            base = complex_channels(handle["base_dct_norm"][local_index])
            residual = complex_channels(handle["residual_dct_norm"][local_index])
            frequency_scale = np.asarray(
                handle["frequency_scale"][local_index], dtype=np.float64
            )[:, None, None]
            base = base * frequency_scale
            residual = residual * frequency_scale
            truth = base + residual
            base_spatial = complex_channels(
                handle["base_spectrum_selected"][local_index]
            )
            truth_spatial = complex_channels(
                handle["truth_spectrum_selected"][local_index]
            )
            unselected = float(handle["base_error_square_unselected"][local_index])
            target_total = float(handle["target_square_total"][local_index])
            retained_parent = float(
                np.sum(
                    weights
                    * np.sum(np.abs(residual) ** 2, axis=(1, 2), dtype=np.float64),
                    dtype=np.float64,
                )
            )
            full_parent = float(
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
            omitted_spatial = max(full_parent - retained_parent, 0.0)
            parent = math.sqrt((unselected + full_parent) / max(target_total, 1.0e-30))
            reconstructed_parent = math.sqrt(
                (unselected + omitted_spatial + retained_parent)
                / max(target_total, 1.0e-30)
            )
            parent_contract_differences.append(abs(parent - reconstructed_parent))
            family = str(holdout.families[position])
            group_id = str(holdout.group_ids[position])
            sample_id = str(holdout.sample_ids[position])
            coordinates = parse_position(group_id)
            statistics = {
                blocks: block_sufficient_statistics(base, truth, blocks)
                for blocks in block_counts
            }

            def candidate_from_gain(gain, blocks, correction_scale):
                base_energy, truth_energy, cross = statistics[blocks]
                applied = 1.0 + float(correction_scale) * (gain - 1.0)
                block_error = np.maximum(
                    np.abs(applied) ** 2 * base_energy
                    + truth_energy
                    - 2.0 * np.real(np.conj(applied) * cross),
                    0.0,
                )
                retained_error = float(
                    np.sum(weights * block_error.sum(axis=(1, 2)), dtype=np.float64)
                )
                return math.sqrt(
                    (unselected + omitted_spatial + retained_error)
                    / max(target_total, 1.0e-30)
                )

            policy_gain_cache = {}
            if coordinates is not None:
                for policy in donor_policies:
                    settings = policy_settings[policy]
                    for blocks in block_counts:
                        raw_gain, donors = donor_gain(
                            coordinates,
                            blocks=blocks,
                            neighbor_count=settings["neighbor_count"],
                            inverse_distance=settings["inverse_distance"],
                        )
                        if (
                            policy == "knn8_idw"
                            and blocks == block_counts[0]
                            and group_id not in development_donor_map
                        ):
                            development_donor_map[group_id] = donors
                        for smooth in smoothing_widths:
                            policy_gain_cache[(policy, blocks, smooth)] = bounded_gain(
                                raw_gain, smooth=smooth, cap=gain_cap
                            )

            for specification in specifications:
                blocks = int(specification["blocks"])
                if coordinates is None:
                    candidate = parent
                    donor_used = False
                else:
                    gain = policy_gain_cache[
                        (
                            specification["donor_policy"],
                            blocks,
                            int(specification["smoothing_width"]),
                        )
                    ]
                    candidate = candidate_from_gain(
                        gain, blocks, float(specification["correction_scale"])
                    )
                    donor_used = True
                specification["records"].append(
                    {
                        "sample_id": sample_id,
                        "group_id": group_id,
                        "family": family,
                        "donor_used": donor_used,
                        "parent_rel_l2": parent,
                        "candidate_rel_l2": candidate,
                        "relative_improvement": float(1.0 - candidate / parent),
                    }
                )

            if coordinates is not None:
                ranked = []
                for donor_id in positioned_groups:
                    gx, gz = group_position[donor_id]
                    ranked.append(
                        (
                            math.hypot(gx - coordinates[0], gz - coordinates[1]),
                            donor_id,
                        )
                    )
                ranked.sort(key=lambda item: (item[0], item[1]))
                for rank, (distance, donor_id) in enumerate(
                    ranked[:rank_ladder_depth], start=1
                ):
                    gain = bounded_gain(
                        group_num[donor_id][8]
                        / np.maximum(group_den[donor_id][8], 1.0e-30),
                        smooth=1,
                        cap=gain_cap,
                    )
                    rank_records.append(
                        {
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "donor_group_id": donor_id,
                            "donor_rank": rank,
                            "donor_distance_m": distance,
                            "blocks": 8,
                            "smoothing_width": 1,
                            "correction_scale": 1.0,
                            "parent_rel_l2": parent,
                            "candidate_rel_l2": candidate_from_gain(gain, 8, 1.0),
                        }
                    )

        configurations = []
        for specification in specifications:
            records = specification.pop("records")
            marmousi_records = [row for row in records if row["donor_used"]]
            stats = aggregate(records)
            configurations.append(
                {
                    **specification,
                    "aggregate": stats,
                    "aggregate_positioned_only": aggregate(marmousi_records)
                    if marmousi_records
                    else None,
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
            )
        score = lambda item: item["aggregate"]["candidate_max"] + 0.1 * item[
            "aggregate"
        ]["candidate_mean"]
        best = min(configurations, key=score)

        logo_records = []
        for record in fit_marmousi_records:
            group_id = record["group_id"]
            coordinates = group_position[group_id]
            raw_gain, donors = donor_gain(
                coordinates,
                blocks=8,
                neighbor_count=1,
                inverse_distance=False,
                exclude_group=group_id,
            )
            gain = bounded_gain(raw_gain, smooth=1, cap=gain_cap)
            base_energy, truth_energy, cross = record["statistics"][8]
            parent_error = np.maximum(
                base_energy + truth_energy - 2.0 * np.real(cross), 0.0
            )
            candidate_error = np.maximum(
                np.abs(gain) ** 2 * base_energy
                + truth_energy
                - 2.0 * np.real(np.conj(gain) * cross),
                0.0,
            )
            truth_energy_weighted = max(record["truth_weighted_energy"], 1.0e-30)
            logo_records.append(
                {
                    "group_id": group_id,
                    "sample_id": record["sample_id"],
                    "donor_group_id": donors[0]["group_id"],
                    "donor_distance_m": donors[0]["distance_m"],
                    "parent_retained_rel_l2": math.sqrt(
                        float(
                            np.sum(
                                weights * parent_error.sum(axis=(1, 2)),
                                dtype=np.float64,
                            )
                        )
                        / truth_energy_weighted
                    ),
                    "candidate_retained_rel_l2": math.sqrt(
                        float(
                            np.sum(
                                weights * candidate_error.sum(axis=(1, 2)),
                                dtype=np.float64,
                            )
                        )
                        / truth_energy_weighted
                    ),
                }
            )

        payload = {
            "schema": "r49_nearest_group_gain_transfer_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "decide whether fit-distribution densification can close the"
                " 0.05 development maximum gate: if per-group gains transfer"
                " across ~70 m source offsets the data route is viable;"
                " if not, the gain field decorrelates below the existing fit"
                " grid spacing and selective abstention is the remaining path"
            ),
            "donor_selection_deployability": (
                "donor choice uses only source position, which is a"
                " deployment-available input; development truth is used only"
                " for diagnosis"
            ),
            "selection_sha256": fit.selection_sha256,
            "component_selection_sha256": list(fit.component_selection_sha256),
            "fit_record_count": len(fit.records),
            "positioned_fit_group_count": len(positioned_groups),
            "development_record_count": len(holdout.records),
            "fit_development_group_overlap": 0,
            "retained_dct": int(fit.retained),
            "gain_cap": gain_cap,
            "distance_epsilon_m": distance_epsilon_m,
            "maximum_parent_energy_contract_difference": float(
                max(parent_contract_differences, default=0.0)
            ),
            "best": best,
            "configurations": configurations,
            "development_donor_map": development_donor_map,
            "single_donor_rank_ladder": rank_records,
            "fit_logo_curve": logo_records,
            "data_boundary": {
                "fit_truth_used_for_gain_estimation": True,
                "opened_group_disjoint_development_used_for_diagnosis": True,
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
                "union_script": str(union_path),
                "union_script_sha256": sha256_file(union_path),
            },
        }
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
        print(
            json.dumps(
                {
                    "event": "r49_nearest_group_gain_transfer_complete",
                    "best": {
                        key: best[key]
                        for key in (
                            "donor_policy",
                            "blocks",
                            "smoothing_width",
                            "gain_cap",
                            "correction_scale",
                            "aggregate",
                            "aggregate_positioned_only",
                            "absolute_goal",
                            "worst_record",
                        )
                    },
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        fit.close()
        holdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
