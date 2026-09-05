from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from saved_time_phase_operator_v4.coarse_lwc84 import (
    select_all_validation_records,
    shard_records,
)
from scripts.merge_coarse_lwc84_201_shards import merge_shards


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
        ],
        dtype="S16",
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("medium_type", data=families)
        handle.create_dataset(
            "split", data=np.asarray(["validation"] * 11, dtype="S16")
        )
        handle.create_dataset(
            "sample_id", data=np.asarray([f"s{index}" for index in range(11)], dtype="S8")
        )
        handle.create_dataset(
            "group_id", data=np.asarray([f"g{index}" for index in range(11)], dtype="S8")
        )
        handle.create_dataset(
            "sample_sha256", data=np.asarray([f"h{index}" for index in range(11)], dtype="S8")
        )
        handle.attrs["manifest_sha256"] = "fixture-manifest"
        handle.attrs["config_sha256"] = "fixture-config"


def _source_identity(source: Path) -> dict[str, object]:
    return {
        "path": str(source.resolve()),
        "byte_count": source.stat().st_size,
        "manifest_sha256": "fixture-manifest",
        "config_sha256": "fixture-config",
    }


def _write_four_shard_rows(artifact: Path, source: Path) -> None:
    records = select_all_validation_records(
        source, expected_family_counts=EXPECTED_COUNTS
    )
    identity = _source_identity(source)
    numerics = {"stored_time_count": 401, "grid_shape": [201, 201]}
    for shard_index in range(4):
        shard = shard_records(
            records, shard_index=shard_index, shard_count=4
        )
        root = artifact / "shards" / f"shard_{shard_index:02d}"
        prediction_dir = root / "predictions"
        prediction_dir.mkdir(parents=True)
        rows = []
        for record in shard:
            prediction = prediction_dir / f"{record.sample_id}.pt"
            prediction.write_bytes(b"sealed")
            rows.append(
                {
                    "source_index": record.source_index,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "sample_sha256": record.sample_sha256,
                    "family": record.medium_type,
                    "relative_l2": 0.02,
                    "stored_time_count": 401,
                    "prediction_path": str(prediction),
                    "prediction_sha256": "fixture-hash",
                    "prediction_byte_count": prediction.stat().st_size,
                    "truth_opened_after_all_shard_predictions_sealed": True,
                }
            )
        (root / "records.json").write_text(json.dumps(rows))
        summary = {
            "status": "complete",
            "shard_index": shard_index,
            "shard_count": 4,
            "record_count": len(shard),
            "sample_ids": [record.sample_id for record in shard],
            "source_identity": identity,
            "numerical_contract": numerics,
            "prediction_file_count": len(shard),
            "prediction_byte_count": sum(row["prediction_byte_count"] for row in rows),
            "truth_opened_after_all_shard_predictions_sealed": True,
            "runtime": {"wall_seconds": 1.0 + shard_index},
            "peak_cuda_bytes": 1024 * (shard_index + 1),
        }
        (root / "shard_summary.json").write_text(json.dumps(summary))


def test_merger_requires_four_complete_disjoint_shards(tmp_path: Path) -> None:
    source = tmp_path / "tiny.h5"
    artifact = tmp_path / "artifact"
    _write_index_fixture(source)
    _write_four_shard_rows(artifact, source)
    summary = merge_shards(
        source_h5=source,
        artifact_dir=artifact,
        expected_family_counts=EXPECTED_COUNTS,
        shard_count=4,
    )
    assert summary["metrics"]["record_count"] == 9
    assert summary["metrics"]["gate"]["action"] == "direct_baseline"
    assert summary["prediction_file_count"] == 9
    assert summary["truth_opened_after_all_predictions_sealed"] is True
    assert (artifact / "success.json").is_file()


def test_merger_rejects_a_missing_shard(tmp_path: Path) -> None:
    source = tmp_path / "tiny.h5"
    artifact = tmp_path / "artifact"
    _write_index_fixture(source)
    _write_four_shard_rows(artifact, source)
    (artifact / "shards" / "shard_03" / "shard_summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="shard_03"):
        merge_shards(
            source_h5=source,
            artifact_dir=artifact,
            expected_family_counts=EXPECTED_COUNTS,
            shard_count=4,
        )


def test_supervisor_binds_four_gpus_and_merges_only_after_wait() -> None:
    script = Path("scripts/run_coarse_lwc84_201_sealed480_remote.sh").read_text()
    assert "CUDA_VISIBLE_DEVICES=$shard" in script
    assert "--shard-count 4" in script
    assert "--solver-batch-size 120" in script
    assert 'wait "${pids[$shard]}"' in script
    assert "merge_coarse_lwc84_201_shards.py" in script
    assert "nvidia-smi" in script
