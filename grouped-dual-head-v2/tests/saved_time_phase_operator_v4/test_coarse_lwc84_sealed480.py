from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.coarse_lwc84 import (
    complete_field_relative_l2,
    merge_record_rows,
    select_all_validation_records,
    shard_records,
)


EXPECTED_COUNTS = {"uniform": 3, "layered": 4, "marmousi": 2}


def _write_index_fixture(path: Path) -> None:
    families = np.asarray(
        [
            "uniform",
            "anomaly",
            "layered",
            "marmousi",
            "uniform",
            "layered",
            "anomaly",
            "marmousi",
            "layered",
            "uniform",
            "layered",
            "uniform",
        ],
        dtype="S16",
    )
    splits = np.asarray(["validation"] * 11 + ["train"], dtype="S16")
    sample_ids = np.asarray(
        [f"{split.decode()}_{family.decode()}_{index:03d}" for index, (split, family) in enumerate(zip(splits, families, strict=True))],
        dtype="S48",
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("medium_type", data=families)
        handle.create_dataset("split", data=splits)
        handle.create_dataset("sample_id", data=sample_ids)
        handle.create_dataset(
            "group_id", data=np.asarray([f"g{index}" for index in range(12)], dtype="S8")
        )
        handle.create_dataset(
            "sample_sha256",
            data=np.asarray([f"h{index}" for index in range(12)], dtype="S8"),
        )


def _record_rows_for(records):
    return [
        {
            "source_index": record.source_index,
            "sample_id": record.sample_id,
            "family": record.medium_type,
            "relative_l2": 0.02,
            "stored_time_count": 401,
            "truth_opened_after_all_shard_predictions_sealed": True,
        }
        for record in records
    ]


def test_complete_census_excludes_anomaly_and_preserves_source_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "index.h5"
    _write_index_fixture(source)
    rows = select_all_validation_records(
        source, expected_family_counts=EXPECTED_COUNTS
    )
    assert len(rows) == 9
    assert [row.source_index for row in rows] == sorted(
        row.source_index for row in rows
    )
    assert {row.medium_type for row in rows} == {
        "uniform",
        "layered",
        "marmousi",
    }
    assert not any("anomaly" in row.sample_id for row in rows)


def test_four_stride_shards_are_disjoint_and_complete(tmp_path: Path) -> None:
    source = tmp_path / "index.h5"
    _write_index_fixture(source)
    rows = select_all_validation_records(
        source, expected_family_counts=EXPECTED_COUNTS
    )
    shards = [
        shard_records(rows, shard_index=index, shard_count=4) for index in range(4)
    ]
    flattened = [row.source_index for shard in shards for row in shard]
    assert len(flattened) == len(set(flattened)) == 9
    assert sorted(flattened) == [row.source_index for row in rows]
    assert [len(shard) for shard in shards] == [3, 2, 2, 2]


def test_complete_field_metric_matches_joint_float64_norm() -> None:
    target = torch.arange(11 * 5 * 5, dtype=torch.float32).reshape(11, 5, 5)
    prediction = target * 0.875
    expected = float(
        (prediction.double() - target.double()).norm() / target.double().norm()
    )
    assert complete_field_relative_l2(
        prediction, target, block_size=3
    ) == pytest.approx(expected)
    assert complete_field_relative_l2(
        prediction, target, block_size=11
    ) == pytest.approx(expected)


def test_merge_record_rows_reconstructs_family_means_and_gate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "index.h5"
    _write_index_fixture(source)
    expected = select_all_validation_records(
        source, expected_family_counts=EXPECTED_COUNTS
    )
    rows = [
        {
            "source_index": row.source_index,
            "sample_id": row.sample_id,
            "family": row.medium_type,
            "relative_l2": 0.01 + 0.001 * position,
            "stored_time_count": 401,
            "truth_opened_after_all_shard_predictions_sealed": True,
        }
        for position, row in enumerate(expected)
    ]
    merged = merge_record_rows(rows, expected_records=expected)
    assert merged["record_count"] == 9
    assert merged["stored_time_count"] == 401
    assert merged["gate"]["action"] == "direct_baseline"
    assert merged["worst_record"]["relative_l2"] == pytest.approx(0.018)


def test_merge_rejects_duplicate_missing_or_unsealed_rows(tmp_path: Path) -> None:
    source = tmp_path / "index.h5"
    _write_index_fixture(source)
    expected = select_all_validation_records(
        source, expected_family_counts=EXPECTED_COUNTS
    )
    good = _record_rows_for(expected)
    with pytest.raises(ValueError, match="duplicate"):
        merge_record_rows(good + [dict(good[0])], expected_records=expected)
    with pytest.raises(ValueError, match="does not match expected census"):
        merge_record_rows(good[:-1], expected_records=expected)
    bad = [dict(row) for row in good]
    bad[0]["truth_opened_after_all_shard_predictions_sealed"] = False
    with pytest.raises(ValueError, match="seal-before-truth"):
        merge_record_rows(bad, expected_records=expected)
