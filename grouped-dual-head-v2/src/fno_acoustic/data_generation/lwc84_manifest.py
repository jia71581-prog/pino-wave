from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np


MEDIUM_IDS = {"uniform": 0, "layered": 1, "anomaly": 2, "marmousi": 3}
SPLIT_IDS = {"train": 0, "validation": 1, "test_id": 2, "ood_canonical": 3}
NON_OOD_SPLITS = ("train", "validation", "test_id")


def configured_lwc84_counts(config: dict[str, Any]) -> dict[str, dict[str, int] | int]:
    """Return config-derived counts while rejecting an inconsistent composition.

    The original 4003-sample contract remains the default configuration, but
    frozen confirmation datasets can now contain zero-sized training splits and
    independently chosen validation/test sizes without changing generator code.
    """
    dataset = config["dataset"]
    split_config = dataset["splits"]
    if set(split_config) != set(NON_OOD_SPLITS):
        raise ValueError(
            "dataset.splits must define exactly train, validation, and test_id"
        )
    split_counts = {split: int(split_config[split]) for split in NON_OOD_SPLITS}
    if any(count < 0 for count in split_counts.values()):
        raise ValueError("dataset split counts must be non-negative")

    composition = dataset["composition"]
    if set(composition) != set(MEDIUM_IDS):
        raise ValueError(
            f"dataset.composition must define exactly {sorted(MEDIUM_IDS)}"
        )
    medium_counts: dict[str, int] = {}
    for medium in MEDIUM_IDS:
        medium_splits = composition[medium]
        if set(medium_splits) != set(NON_OOD_SPLITS):
            raise ValueError(
                f"dataset.composition.{medium} must define exactly "
                "train, validation, and test_id"
            )
        values = {split: int(medium_splits[split]) for split in NON_OOD_SPLITS}
        if any(count < 0 for count in values.values()):
            raise ValueError(f"dataset.composition.{medium} counts must be non-negative")
        medium_counts[medium] = sum(values.values())

    for split in NON_OOD_SPLITS:
        composed = sum(int(composition[medium][split]) for medium in MEDIUM_IDS)
        if composed != split_counts[split]:
            raise ValueError(
                f"dataset split {split} declares {split_counts[split]} samples "
                f"but composition sums to {composed}"
            )
    ood_count = len(dataset.get("ood_canonical", []))
    return {
        "split_counts": {**split_counts, "ood_canonical": int(ood_count)},
        "medium_counts": medium_counts,
        "sample_count": int(sum(split_counts.values()) + ood_count),
    }


def _namespaced_id(config: dict[str, Any], value: str) -> str:
    namespace = config["dataset"].get("sample_namespace")
    if namespace is None:
        return value
    namespace = str(namespace).strip()
    if not namespace:
        raise ValueError("dataset.sample_namespace must be non-empty when provided")
    return f"{namespace}__{value}"


def _frequency_values(count: int, *, seed: int, config: dict[str, Any]) -> np.ndarray:
    source = config["source"]
    low = float(source["f0_hz_min"])
    high = float(source["f0_hz_max"])
    windows = [tuple(float(value) for value in window) for window in source["excluded_frequency_windows_hz"]]
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for index in range(int(count)):
        stratum_low = low + (high - low) * index / max(count, 1)
        stratum_high = low + (high - low) * (index + 1) / max(count, 1)
        for _ in range(1000):
            value = float(rng.uniform(stratum_low, stratum_high))
            if not any(left <= value <= right for left, right in windows):
                values.append(value)
                break
        else:
            midpoint = 0.5 * (stratum_low + stratum_high)
            allowed = [
                candidate
                for candidate in (stratum_low, midpoint, stratum_high)
                if not any(left <= candidate <= right for left, right in windows)
            ]
            if not allowed:
                # A narrow stratum can be fully covered by an OOD window. Move to
                # the closest legal edge while staying inside the global range.
                edges = [edge for window in windows for edge in (window[0] - 1.0e-4, window[1] + 1.0e-4)]
                allowed = [edge for edge in edges if low <= edge <= high]
            values.append(float(min(allowed, key=lambda candidate: abs(candidate - midpoint))))
    permutation = rng.permutation(count)
    return np.asarray(values, dtype=np.float64)[permutation]


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _marmousi_crop_plan(
    geometry: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]:
    shape = [int(value) for value in geometry["shape"]]
    dx = float(geometry["dx_m"])
    dz = float(geometry["dz_m"])
    width = (shape[1] - 1) * dx
    depth = (shape[0] - 1) * dz
    if width < 2000.0 or depth < 2000.0:
        raise ValueError(f"Marmousi geometry {shape} at {dx}x{dz} m is smaller than 2 km x 2 km")
    marmousi = config["marmousi"]
    x0_config = marmousi.get("ood_crop_x0_m")
    z0_config = marmousi.get("ood_crop_z0_m")
    x0 = (
        round((width - 2000.0) / 2.0 / 5.0) * 5.0
        if x0_config is None
        else float(x0_config)
    )
    z0 = 0.0 if z0_config is None else float(z0_config)
    crop_x0_min = float(marmousi.get("crop_x0_min_m", 0.0))
    crop_x0_max = float(marmousi.get("crop_x0_max_m", width - 2000.0))
    crop_z0_min = float(marmousi.get("crop_z0_min_m", 0.0))
    crop_z0_max = float(marmousi.get("crop_z0_max_m", depth - 2000.0))
    if not (
        0.0 <= crop_x0_min <= crop_x0_max <= width - 2000.0
        and 0.0 <= crop_z0_min <= crop_z0_max <= depth - 2000.0
    ):
        raise ValueError("configured Marmousi crop-origin bounds are outside the source model")
    if (
        x0 < crop_x0_min
        or z0 < crop_z0_min
        or x0 > crop_x0_max
        or z0 > crop_z0_max
    ):
        raise ValueError("configured OOD Marmousi crop violates crop-origin bounds")
    if x0 + 2000.0 > width or z0 + 2000.0 > depth:
        raise ValueError("configured OOD Marmousi crop is outside the source model")
    guard = float(marmousi["ood_guard_band_m"])
    ood_bbox = (x0, x0 + 2000.0, z0, z0 + 2000.0)
    guard_bbox = (max(0.0, x0 - guard), min(width, x0 + 2000.0 + guard), max(0.0, z0 - guard), min(depth, z0 + 2000.0 + guard))
    stride = float(marmousi.get("crop_stride_m", 50.0))
    if stride <= 0.0:
        raise ValueError("Marmousi crop stride must be positive")
    def candidate_starts(x_min: float, x_max: float) -> list[tuple[float, float]]:
        starts: list[tuple[float, float]] = []
        for crop_z in np.arange(crop_z0_min, crop_z0_max + 1.0e-9, stride):
            for crop_x in np.arange(x_min, x_max + 1.0e-9, stride):
                bbox = (float(crop_x), float(crop_x + 2000.0), float(crop_z), float(crop_z + 2000.0))
                if not _bbox_intersects(bbox, guard_bbox):
                    starts.append((float(crop_x), float(crop_z)))
        return starts

    split_ranges = marmousi.get("crop_split_x0_ranges_m")
    if split_ranges is not None:
        pools = {}
        for split_index, split in enumerate(("train", "validation", "test_id")):
            split_min, split_max = (float(value) for value in split_ranges[split])
            if not crop_x0_min <= split_min <= split_max <= crop_x0_max:
                raise ValueError(f"Marmousi {split} crop range is outside global crop bounds")
            starts = candidate_starts(split_min, split_max)
            rng = np.random.default_rng(int(config["seed"]) + 3_041_989 + split_index)
            pools[split] = [starts[int(index)] for index in rng.permutation(len(starts))]
    else:
        starts = candidate_starts(crop_x0_min, crop_x0_max)
        if not starts:
            raise ValueError("no Marmousi training crop remains after applying the OOD guard band")
        pool_order = str(marmousi.get("crop_pool_order", "coordinate_order"))
        if pool_order == "seeded_permutation":
            rng = np.random.default_rng(int(config["seed"]) + 3_041_989)
            starts = [starts[int(index)] for index in rng.permutation(len(starts))]
        elif pool_order != "coordinate_order":
            raise ValueError(f"unsupported Marmousi crop_pool_order: {pool_order}")
        pools = {
            split: [value for index, value in enumerate(starts) if index % 3 == split_index]
            for split_index, split in enumerate(("train", "validation", "test_id"))
        }
    if any(not values for values in pools.values()):
        raise ValueError("Marmousi geometry cannot provide split-isolated crop pools")
    if bool(marmousi.get("require_cross_split_nonoverlap", False)):
        active_splits = tuple(
            split
            for split in NON_OOD_SPLITS
            if int(config["dataset"]["composition"]["marmousi"][split]) > 0
        )
        for left_index, left_split in enumerate(active_splits):
            for right_split in active_splits[left_index + 1 :]:
                if any(
                    _bbox_intersects(
                        (x1, x1 + 2000.0, z1, z1 + 2000.0),
                        (x2, x2 + 2000.0, z2, z2 + 2000.0),
                    )
                    for x1, z1 in pools[left_split]
                    for x2, z2 in pools[right_split]
                ):
                    raise ValueError(f"Marmousi spatial leakage between {left_split} and {right_split}")
    return pools, {
        "crop_x0_m": x0,
        "crop_z0_m": z0,
        "crop_bbox_m": list(ood_bbox),
        "guard_bbox_m": list(guard_bbox),
        "source_sha256": str(geometry["sha256"]),
    }


def _uniform_velocity(rng: np.random.Generator, config: dict[str, Any]) -> float:
    low, high = (float(value) for value in config["models"]["uniform"]["velocity_mps"])
    excluded_low, excluded_high = (
        float(value) for value in config["models"]["uniform"]["excluded_velocity_window_mps"]
    )
    while True:
        value = float(rng.uniform(low, high))
        if not excluded_low <= value <= excluded_high:
            return value


def build_lwc84_manifest(
    config: dict[str, Any],
    *,
    marmousi_geometry: dict[str, Any],
) -> list[dict[str, Any]]:
    configured_lwc84_counts(config)
    seed = int(config["seed"])
    crop_pools, ood_crop = _marmousi_crop_plan(marmousi_geometry, config)
    rows: list[dict[str, Any]] = []
    sample_counter = 0
    group_sizes = {"uniform": 1, "layered": 4, "anomaly": 2, "marmousi": 5}
    for medium in ("uniform", "layered", "anomaly", "marmousi"):
        for split in NON_OOD_SPLITS:
            count = int(config["dataset"]["composition"][medium][split])
            frequencies = _frequency_values(
                count,
                seed=seed + 100_000 * MEDIUM_IDS[medium] + 10_000 * SPLIT_IDS[split],
                config=config,
            )
            for local_index in range(count):
                group_ordinal = local_index // group_sizes[medium]
                group_seed = seed + 1_000_000 * MEDIUM_IDS[medium] + 100_000 * SPLIT_IDS[split] + group_ordinal
                sample_seed = seed + sample_counter * 104729
                rng = np.random.default_rng(sample_seed)
                source_x = float(rng.uniform(config["source"]["x_m_min"], config["source"]["x_m_max"]))
                source_z = float(rng.uniform(config["source"]["z_m_min"], config["source"]["z_m_max"]))
                group_id = f"{split}:{medium}:{group_ordinal:05d}"
                crop_x = crop_z = None
                medium_parameters: dict[str, Any] = {"seed": int(group_seed)}
                if medium == "uniform":
                    medium_parameters["velocity_mps"] = _uniform_velocity(
                        np.random.default_rng(group_seed), config
                    )
                elif medium == "marmousi":
                    pool = crop_pools[split]
                    if group_ordinal >= len(pool):
                        raise ValueError(f"Marmousi {split} requires more base crops than the source model provides")
                    crop_x, crop_z = pool[group_ordinal]
                    group_id = f"{split}:marmousi:x{crop_x:.1f}:z{crop_z:.1f}"
                    medium_parameters.update(
                        {
                            "source_sha256": str(marmousi_geometry["sha256"]),
                            "geology_sha256": str(
                                config["marmousi"].get(
                                    "geology_sha256", marmousi_geometry["sha256"]
                                )
                            ),
                            "crop_x0_m": crop_x,
                            "crop_z0_m": crop_z,
                        }
                    )
                frequency = float(frequencies[local_index])
                rows.append(
                    {
                        "sample_id": _namespaced_id(
                            config, f"{split}_{medium}_{local_index:05d}"
                        ),
                        "split": split,
                        "split_id": SPLIT_IDS[split],
                        "medium_type": medium,
                        "medium_type_id": MEDIUM_IDS[medium],
                        "group_id": _namespaced_id(config, group_id),
                        "seed": int(sample_seed),
                        "medium_parameters": medium_parameters,
                        "source_x_m": source_x,
                        "source_z_m": source_z,
                        "source_f0_hz": frequency,
                        "source_t0_s": 1.5 / frequency,
                        "source_amplitude": float(config["source"]["amplitude"]),
                        "crop_x0_m": crop_x,
                        "crop_z0_m": crop_z,
                    }
                )
                sample_counter += 1

    for case in config["dataset"].get("ood_canonical", []):
        row = {
            "sample_id": _namespaced_id(config, str(case["case_id"])),
            "case_id": case["case_id"],
            "split": "ood_canonical",
            "split_id": SPLIT_IDS["ood_canonical"],
            "medium_type": case["medium_type"],
            "medium_type_id": MEDIUM_IDS[case["medium_type"]],
            "group_id": _namespaced_id(config, f"ood:{case['case_id']}"),
            "seed": seed + sample_counter * 104729,
            "source_x_m": float(case["source_x_m"]),
            "source_z_m": float(case["source_z_m"]),
            "source_f0_hz": float(case["source_f0_hz"]),
            "source_t0_s": 1.5 / float(case["source_f0_hz"]),
            "source_amplitude": float(config["source"]["amplitude"]),
            "crop_x0_m": None,
            "crop_z0_m": None,
            "medium_parameters": {
                key: value
                for key, value in case.items()
                if key not in {"case_id", "medium_type", "source_x_m", "source_z_m", "source_f0_hz"}
            },
        }
        if case["medium_type"] == "marmousi":
            row["crop_x0_m"] = ood_crop["crop_x0_m"]
            row["crop_z0_m"] = ood_crop["crop_z0_m"]
            row["medium_parameters"].update(ood_crop)
        rows.append(row)
        sample_counter += 1
    return rows


def validate_lwc84_manifest(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    configured = configured_lwc84_counts(config)
    errors: list[str] = []
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [value for value, count in Counter(sample_ids).items() if count > 1]
        errors.append(f"duplicate sample IDs: {duplicates[:3]}")
    split_counts = Counter(row["split"] for row in rows)
    expected_splits = configured["split_counts"]
    if {key: split_counts.get(key, 0) for key in expected_splits} != expected_splits:
        errors.append(f"split counts differ: {dict(split_counts)}")
    non_ood = [row for row in rows if row["split"] != "ood_canonical"]
    medium_counts = Counter(row["medium_type"] for row in non_ood)
    expected_medium = configured["medium_counts"]
    if {key: medium_counts.get(key, 0) for key in expected_medium} != expected_medium:
        errors.append(f"medium counts differ: {dict(medium_counts)}")
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_group[str(row["group_id"])].add(str(row["split"]))
    leaking_groups = [group for group, splits in splits_by_group.items() if len(splits) != 1]
    if leaking_groups:
        errors.append(f"group split leakage: {leaking_groups[:3]}")
    if bool(config["marmousi"].get("require_cross_split_nonoverlap", False)):
        active_splits = tuple(
            split
            for split in NON_OOD_SPLITS
            if int(config["dataset"]["composition"]["marmousi"][split]) > 0
        )
        origins_by_split = {
            split: {
                (float(row["crop_x0_m"]), float(row["crop_z0_m"]))
                for row in non_ood
                if row["medium_type"] == "marmousi" and row["split"] == split
            }
            for split in active_splits
        }
        for left_index, left_split in enumerate(active_splits):
            for right_split in active_splits[left_index + 1 :]:
                overlap = next(
                    (
                        (left_origin, right_origin)
                        for left_origin in origins_by_split[left_split]
                        for right_origin in origins_by_split[right_split]
                        if _bbox_intersects(
                            (
                                left_origin[0],
                                left_origin[0] + 2000.0,
                                left_origin[1],
                                left_origin[1] + 2000.0,
                            ),
                            (
                                right_origin[0],
                                right_origin[0] + 2000.0,
                                right_origin[1],
                                right_origin[1] + 2000.0,
                            ),
                        )
                    ),
                    None,
                )
                if overlap is not None:
                    errors.append(
                        "Marmousi spatial leakage between "
                        f"{left_split} and {right_split}: {overlap}"
                    )
    windows = [tuple(float(value) for value in window) for window in config["source"]["excluded_frequency_windows_hz"]]
    if any(any(left <= float(row["source_f0_hz"]) <= right for left, right in windows) for row in non_ood):
        errors.append("OOD frequency window leaked into a non-OOD split")
    source = config["source"]
    if any(
        not float(source["x_m_min"]) <= float(row["source_x_m"]) <= float(source["x_m_max"])
        or not float(source["z_m_min"]) <= float(row["source_z_m"]) <= float(source["z_m_max"])
        for row in non_ood
    ):
        errors.append("non-OOD source coordinate is outside the configured range")
    ood_rows = [row for row in rows if row["split"] == "ood_canonical"]
    ood_case_ids = [str(row["case_id"]) for row in ood_rows]
    expected_ood_ids = [
        str(case["case_id"]) for case in config["dataset"].get("ood_canonical", [])
    ]
    if ood_case_ids != expected_ood_ids:
        errors.append(f"OOD case IDs differ: {ood_case_ids}")
    marmousi_ood = next((row for row in ood_rows if row["medium_type"] == "marmousi"), None)
    expects_marmousi_ood = any(
        str(case["medium_type"]) == "marmousi"
        for case in config["dataset"].get("ood_canonical", [])
    )
    if expects_marmousi_ood and marmousi_ood is None:
        errors.append("Marmousi OOD row is missing")
    elif marmousi_ood is not None:
        guard_bbox = tuple(float(value) for value in marmousi_ood["medium_parameters"]["guard_bbox_m"])
        for row in non_ood:
            if row["medium_type"] != "marmousi":
                continue
            bbox = (
                float(row["crop_x0_m"]),
                float(row["crop_x0_m"]) + 2000.0,
                float(row["crop_z0_m"]),
                float(row["crop_z0_m"]) + 2000.0,
            )
            if _bbox_intersects(bbox, guard_bbox):
                errors.append(f"Marmousi crop intersects OOD guard band: {row['sample_id']}")
                break
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "sample_count": len(rows),
        "split_counts": {key: int(split_counts[key]) for key in expected_splits},
        "medium_counts": {key: int(medium_counts[key]) for key in expected_medium},
        "ood_case_ids": ood_case_ids,
        "group_split_leakage_count": 0,
        "frequency_window_leakage_count": 0,
        "marmousi_cross_split_overlap_count": 0,
        "marmousi_ood_overlap_count": 0,
    }
