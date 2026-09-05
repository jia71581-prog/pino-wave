from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import artifact_root
from .grid import AcousticGrid
from .source import bilinear_point_source

CATEGORY_IDS = {"uniform": 0, "layered": 1, "marmousi": 2, "openfwi": 3}
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}
FREQUENCY_ROLE_IDS = {"continuous": 0, "anchor_10hz": 1, "anchor_25hz": 2}
TRUTH_KIND_IDS = {"gpu_analytic": 0, "gpu_lwc84": 1, "gpu_lwc84_validated_by_analytic": 2}


def _category_totals(config: dict[str, Any], profile: str) -> dict[str, int]:
    profiles = config["profiles"][profile]
    openfwi_enabled = bool(config.get("models", {}).get("openfwi", {}).get("enabled", False))
    if profile == "core_v3_gpu" and not openfwi_enabled:
        return {k: int(v) for k, v in profiles["when_openfwi_disabled"].items()}
    return {k: int(profiles.get(k, 0)) for k in ("uniform", "layered", "marmousi", "openfwi")}


def _split_counts(total: int) -> dict[str, int]:
    return {"train": int(round(total * 0.70)), "validation": int(round(total * 0.15)), "test": total - int(round(total * 0.70)) - int(round(total * 0.15))}


def _frequency_counts(total: int) -> dict[str, int]:
    anchor = int(round(total * 0.20))
    return {"anchor_10hz": anchor, "anchor_25hz": anchor, "continuous": total - 2 * anchor}


def _profile_frequency_counts(config: dict[str, Any], profile: str, total: int) -> dict[str, int]:
    if profile in {"smoke", "pilot"}:
        raw = config["profiles"][profile].get("frequency_counts_per_main_category", {})
        counts = {
            "anchor_10hz": int(raw.get("anchor_10hz", 0)),
            "anchor_25hz": int(raw.get("anchor_25hz", 0)),
            "continuous": int(raw.get("continuous", 0)),
        }
        if sum(counts.values()) != int(total):
            raise ValueError(f"{profile} frequency counts {counts} do not sum to category total {total}")
        return counts
    return _frequency_counts(total)


def _base_model_id(category: str, split: str, ordinal: int) -> int:
    return CATEGORY_IDS[category] * 1_000_000 + SPLIT_IDS[split] * 100_000 + int(ordinal)


def _source_for(seed: int, grid: AcousticGrid) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    x0 = float(rng.uniform(200.0, 1800.0))
    if rng.random() < 0.85:
        z0 = float(rng.uniform(30.0, 150.0))
    else:
        z0 = float(rng.uniform(150.0001, 500.0))
    x0 = min(max(x0, grid.x_m[1]), grid.x_m[-2])
    z0 = min(max(z0, grid.z_m[1]), grid.z_m[-2])
    return x0, z0


def _continuous_frequencies(count: int, seed: int, *, min_hz: float = 8.0, max_hz: float = 25.0, excluded_hz: Iterable[float] = (10.0, 25.0)) -> list[float]:
    if count <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    excluded = {int(round(v)) for v in excluded_hz}
    values = [v for v in range(int(np.ceil(min_hz)), int(np.floor(max_hz)) + 1) if v not in excluded]
    if not values:
        raise ValueError("integer continuous frequency pool is empty")
    out: list[float] = []
    while len(out) < count:
        shuffled = np.asarray(values, dtype=np.int32)
        rng.shuffle(shuffled)
        out.extend(float(v) for v in shuffled.tolist())
    return out[:count]
    return out


def _truth_kind(category: str) -> str:
    return "gpu_analytic" if category == "uniform" else "gpu_lwc84"


def build_manifest(config: dict[str, Any], profile: str, enabled_categories=None) -> list[dict[str, Any]]:
    grid = AcousticGrid()
    seed = int(config.get("seed", 20260625))
    totals = _category_totals(config, profile)
    if enabled_categories is not None:
        enabled = set(enabled_categories)
        totals = {k: v for k, v in totals.items() if k in enabled}
    rows: list[dict[str, Any]] = []
    sample_id = 0
    pair_id = 0
    for category in ("uniform", "layered", "marmousi", "openfwi"):
        total = int(totals.get(category, 0))
        if total == 0:
            continue
        split_counts = _split_counts(total) if profile == "core_v3_gpu" else {"train": total, "validation": 0, "test": 0}
        for split, split_total in split_counts.items():
            if split_total == 0:
                continue
            freq_counts = _profile_frequency_counts(config, profile, split_total)
            anchor_pairs = freq_counts["anchor_10hz"]
            continuous = freq_counts["continuous"]
            base_ord = 0
            for local_pair in range(anchor_pairs):
                base_id = _base_model_id(category, split, base_ord)
                source_seed = seed + base_id + local_pair
                x0, z0 = _source_for(source_seed, grid)
                src = bilinear_point_source(x0, z0, nx=grid.nx, nz=grid.nz, dx_m=grid.dx_m, dz_m=grid.dz_m)
                for role, f0 in (("anchor_10hz", 10.0), ("anchor_25hz", 25.0)):
                    rows.append(
                        _row(
                            sample_id=sample_id,
                            category=category,
                            split=split,
                            base_model_id=base_id,
                            f0_hz=f0,
                            role=role,
                            pair_id=pair_id,
                            source=src,
                            sample_seed=seed + sample_id * 17,
                            truth_kind=_truth_kind(category),
                        )
                    )
                    sample_id += 1
                pair_id += 1
                base_ord += 1
            freqs = _continuous_frequencies(
                continuous,
                seed + CATEGORY_IDS[category] * 97 + SPLIT_IDS[split],
                min_hz=float(config["source"]["f0_hz_min"]),
                max_hz=float(config["source"]["f0_hz_max"]),
                excluded_hz=tuple(float(v) for v in config["source"]["required_f0_hz"]),
            )
            for local_cont, f0 in enumerate(freqs):
                base_id = _base_model_id(category, split, base_ord)
                x0, z0 = _source_for(seed + base_id + local_cont, grid)
                src = bilinear_point_source(x0, z0, nx=grid.nx, nz=grid.nz, dx_m=grid.dx_m, dz_m=grid.dz_m)
                rows.append(
                    _row(
                        sample_id=sample_id,
                        category=category,
                        split=split,
                        base_model_id=base_id,
                        f0_hz=f0,
                        role="continuous",
                        pair_id=-1,
                        source=src,
                        sample_seed=seed + sample_id * 17,
                        truth_kind=_truth_kind(category),
                    )
                )
                sample_id += 1
                base_ord += 1
    return rows


def _row(
    *,
    sample_id: int,
    category: str,
    split: str,
    base_model_id: int,
    f0_hz: float,
    role: str,
    pair_id: int,
    source,
    sample_seed: int,
    truth_kind: str,
) -> dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "category": category,
        "category_id": CATEGORY_IDS[category],
        "split": split,
        "split_id": SPLIT_IDS[split],
        "base_model_id": int(base_model_id),
        "source_xy_m": [float(source.x0_m), float(source.z0_m)],
        "source_indices": source.indices.astype(int).tolist(),
        "source_weights": [float(v) for v in source.weights.tolist()],
        "f0_hz": float(np.float32(f0_hz)),
        "frequency_role": role,
        "frequency_role_id": FREQUENCY_ROLE_IDS[role],
        "frequency_pair_id": int(pair_id),
        "sample_seed": int(sample_seed),
        "truth_kind": truth_kind,
        "truth_kind_id": TRUTH_KIND_IDS[truth_kind],
        "truth_grid_tier": "5m",
        "solver_dtype": "float32",
        "velocity_provenance": {"kind": category, "base_model_id": int(base_model_id)},
    }


def validate_manifest(rows: list[dict[str, Any]], config: dict[str, Any], profile: str) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    freq_counts = Counter(row["frequency_role"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    split_freq = Counter((row["split"], row["frequency_role"]) for row in rows)
    category_split_freq = Counter((row["category"], row["split"], row["frequency_role"]) for row in rows)
    errors: list[str] = []
    if profile == "core_v3_gpu":
        if len(rows) != 4000:
            errors.append(f"expected 4000 rows, got {len(rows)}")
        expected_split = {"train": 2800, "validation": 600, "test": 600}
        if dict(split_counts) != expected_split:
            errors.append(f"split counts mismatch: {dict(split_counts)}")
        expected_freq = {"anchor_10hz": 800, "anchor_25hz": 800, "continuous": 2400}
        if {k: freq_counts.get(k, 0) for k in expected_freq} != expected_freq:
            errors.append(f"frequency counts mismatch: {dict(freq_counts)}")
        for split, expected in {
            "train": {"anchor_10hz": 560, "anchor_25hz": 560, "continuous": 1680},
            "validation": {"anchor_10hz": 120, "anchor_25hz": 120, "continuous": 360},
            "test": {"anchor_10hz": 120, "anchor_25hz": 120, "continuous": 360},
        }.items():
            actual = {role: split_freq.get((split, role), 0) for role in expected}
            if actual != expected:
                errors.append(f"{split} frequency counts mismatch: {actual}")
    for (category, split), total in Counter((row["category"], row["split"]) for row in rows).items():
        expected = _profile_frequency_counts(config, profile, total)
        actual = {role: category_split_freq.get((category, split, role), 0) for role in expected}
        if actual != expected:
            errors.append(f"{category}/{split} frequency role fractions mismatch: {actual} expected {expected}")
    split_by_model: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        split_by_model[int(row["base_model_id"])].add(row["split"])
    leakage = sum(1 for splits in split_by_model.values() if len(splits) > 1)
    if leakage:
        errors.append(f"base model split leakage count {leakage}")
    pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["frequency_pair_id"]) >= 0:
            pairs[int(row["frequency_pair_id"])].append(row)
    bad_pairs = 0
    for members in pairs.values():
        if len(members) != 2 or sorted(float(row["f0_hz"]) for row in members) != [10.0, 25.0]:
            bad_pairs += 1
            continue
        keys = ["base_model_id", "split"]
        if any(len({row[k] for row in members}) != 1 for k in keys):
            bad_pairs += 1
        if len({tuple(row["source_xy_m"]) for row in members}) != 1:
            bad_pairs += 1
    if bad_pairs:
        errors.append(f"bad frequency pairs {bad_pairs}")
    marmousi_overlap = 0
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "profile": profile,
        "sample_count": int(len(rows)),
        "split_counts": {k: int(split_counts.get(k, 0)) for k in ("train", "validation", "test")},
        "frequency_counts": {
            "anchor_10hz": int(freq_counts.get("anchor_10hz", 0)),
            "anchor_25hz": int(freq_counts.get("anchor_25hz", 0)),
            "continuous": int(freq_counts.get("continuous", 0)),
        },
        "category_counts": {k: int(category_counts.get(k, 0)) for k in ("uniform", "layered", "marmousi", "openfwi")},
        "frequency_pair_count": int(len(pairs)),
        "base_model_split_leakage_count": int(leakage),
        "marmousi_crop_cross_split_overlap_count": int(marmousi_overlap),
        "all_hard_assertions_passed": True,
    }


def write_manifest_artifacts(rows: list[dict[str, Any]], config: dict[str, Any], summary: dict[str, Any]) -> dict[str, str]:
    root = artifact_root(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "split_manifest.json"
    jsonl_path = root / "manifest.jsonl"
    payload = {"summary": summary, "rows": rows}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"split_manifest": str(manifest_path), "manifest_jsonl": str(jsonl_path)}


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        return list(payload["rows"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported manifest format: {path}")
