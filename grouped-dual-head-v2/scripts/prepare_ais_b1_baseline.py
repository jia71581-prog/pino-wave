#!/usr/bin/env python3
"""Publish the pinned B1 evaluator outputs in Task12's canonical schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.run_ais_v2_screen import (  # noqa: E402
    B1_CHECKPOINT_SHA256,
    B1_CONFIG_PATH,
    B1_CONFIG_SHA256,
    REPO_ROOT,
    atomic_write_json,
    canonical_sha256,
    compute_source_binding,
    sha256_file,
)
from fno_acoustic.query_census import (  # noqa: E402
    AIS_CATEGORIES,
    REQUIRED_NUMERIC_COLUMNS,
    SAMPLE_COLUMNS,
    aggregate_category_metrics,
)
from fno_acoustic.ais_dataset_binding import (  # noqa: E402
    DatasetContentBinding,
    load_dataset_content_binding,
)
from fno_acoustic.query_data import DenseCPUQueryStore  # noqa: E402


def _metrics_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _metrics_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def _validated_evaluator_artifacts(
    evaluation_path: Path,
    samples_path: Path,
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    split_sha256: str,
    normalization_sha256: str,
    expected_ids: set[int],
    expected_categories: dict[int, str] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, float | int]]]:
    evaluation = json.loads(evaluation_path.read_bytes())
    if not isinstance(evaluation, dict):
        raise ValueError("B1 evaluation summary must be a mapping")
    expected_provenance = {
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": split_sha256,
        "normalization_sha256": normalization_sha256,
    }
    provenance = evaluation.get("provenance")
    if (
        evaluation.get("split") != "val"
        or evaluation.get("sample_count") != len(expected_ids)
        or evaluation.get("evaluation_provenance") is not None
        or not isinstance(provenance, dict)
        or any(provenance.get(key) != value for key, value in expected_provenance.items())
    ):
        raise ValueError("B1 evaluation split, count, or provenance mismatch")
    with samples_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SAMPLE_COLUMNS:
            raise ValueError("B1 samples CSV column contract mismatch")
        raw_rows = list(reader)
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        row: dict[str, object] = dict(raw_row)
        try:
            row["sample_id"] = int(raw_row["sample_id"])
            for name in REQUIRED_NUMERIC_COLUMNS:
                row[name] = float(raw_row[name])
        except (TypeError, ValueError) as error:
            raise ValueError("B1 census numeric column is invalid") from error
        rows.append(row)
    seen: set[int] = set()
    for row in rows:
        try:
            sample_id = int(row["sample_id"])
        except (TypeError, ValueError) as error:
            raise ValueError("B1 sample_id must be an integer") from error
        if sample_id in seen:
            raise ValueError("B1 samples contain duplicate sample_id")
        seen.add(sample_id)
        category = row["category"]
        if category not in AIS_CATEGORIES:
            raise ValueError("B1 sample category is not registered")
        if expected_categories is not None and expected_categories.get(sample_id) != category:
            raise ValueError("B1 sample category differs from registered split")
        if row["split"] != "val" or any(
            row[key] != value for key, value in expected_provenance.items()
        ):
            raise ValueError("B1 sample row provenance mismatch")
    if seen != expected_ids:
        raise ValueError("B1 samples IDs differ from the complete validation split")
    aggregated = aggregate_category_metrics(rows)
    expected_mean = {
        metric: aggregated["global"][metric]
        for metric in REQUIRED_NUMERIC_COLUMNS
    }
    expected_summary_categories = {
        category: {
            "sample_count": values["sample_count"],
            "mean": {
                metric: values[metric] for metric in REQUIRED_NUMERIC_COLUMNS
            },
        }
        for category, values in aggregated.items()
        if category != "global"
    }
    if not _metrics_equal(evaluation.get("mean"), expected_mean) or not _metrics_equal(
        evaluation.get("categories"), expected_summary_categories
    ):
        raise ValueError("B1 summary metrics differ from strict CSV aggregation")
    if not _metrics_equal(evaluation.get("category_metrics"), aggregated):
        raise ValueError("B1 summary category metrics differ from strict CSV aggregation")
    return evaluation, aggregated


def _validated_dataset_manifest(
    path: Path, data_path: Path
) -> DatasetContentBinding:
    return load_dataset_content_binding(path, data_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-content-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha256_file(args.checkpoint) != B1_CHECKPOINT_SHA256:
        raise ValueError("B1 checkpoint differs from pinned identity")
    config = yaml.safe_load(B1_CONFIG_PATH.read_bytes())
    if canonical_sha256(config) != B1_CONFIG_SHA256:
        raise ValueError("B1 config differs from pinned identity")
    split_path = REPO_ROOT / config["data"]["split_manifest"]
    stats_path = REPO_ROOT / config["normalization"]["stats_path"]
    split = json.loads(split_path.read_bytes())
    data_path = Path(config["data"]["path"])
    dataset_manifest = _validated_dataset_manifest(
        args.dataset_content_manifest, data_path
    )
    expected_ids = set(split["val"])
    if expected_ids and min(expected_ids) < 0:
        raise ValueError("B1 validation ID must be nonnegative")
    if expected_ids and max(expected_ids) >= dataset_manifest.sample_count:
        raise ValueError("B1 validation ID exceeds model_type dataset")
    store = DenseCPUQueryStore(
        data_path, dataset_content_manifest=args.dataset_content_manifest
    )
    expected_categories = {
        sample_id: str(store.read_scene(sample_id).metadata["model_type"])
        for sample_id in sorted(expected_ids)
    }
    split_sha256 = sha256_file(split_path)
    normalization_sha256 = sha256_file(stats_path)
    _, aggregated = _validated_evaluator_artifacts(
        args.evaluation_summary,
        args.samples,
        checkpoint_sha256=B1_CHECKPOINT_SHA256,
        config_sha256=B1_CONFIG_SHA256,
        split_sha256=split_sha256,
        normalization_sha256=normalization_sha256,
        expected_ids=expected_ids,
        expected_categories=expected_categories,
    )
    destination = args.output_dir / "baseline_b1"
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.checkpoint, destination / "checkpoints/best.pt")
    shutil.copyfile(args.samples, destination / "samples.csv")
    source = compute_source_binding()
    atomic_write_json(
        destination / "summary.json",
        {
            "schema_version": 1, "purpose": "ais_b1_validation_baseline",
            "baseline_id": "B1", "checkpoint_sha256": B1_CHECKPOINT_SHA256,
            "config_sha256": B1_CONFIG_SHA256, "split": "val",
            "sample_count": len(split["val"]), "representative_manifest": None,
            "split_sha256": split_sha256,
            "normalization_stats_sha256": normalization_sha256,
            **source.to_dict(), "samples_path": str(destination / "samples.csv"),
            "samples_sha256": sha256_file(destination / "samples.csv"),
            "dataset_content_manifest_sha256": sha256_file(args.dataset_content_manifest),
            **store.binding_summary(),
            "category_metrics": aggregated,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
