from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import (
    ALLOWED_MEDIUM_TYPES,
    assert_allowed_families,
    build_manifest,
    write_manifest_atomic,
)


REAL_DATASET = Path(
    "/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5"
)


@pytest.fixture()
def tiny_source_h5(tmp_path: Path) -> Path:
    path = tmp_path / "source.h5"
    families = np.asarray(
        ["uniform", "layered", "marmousi", "anomaly", "uniform"],
        dtype=h5py.string_dtype(),
    )
    splits = np.asarray(
        ["train", "train", "validation", "train", "validation"],
        dtype=h5py.string_dtype(),
    )
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            schema_version="test-v1",
            manifest_sha256="source-manifest",
            config_sha256="source-config",
        )
        h5["medium_type"] = families
        h5["split"] = splits
        h5["sample_id"] = np.asarray([f"sample-{i}" for i in range(5)], dtype=h5py.string_dtype())
        h5["group_id"] = np.asarray([f"group-{i}" for i in range(5)], dtype=h5py.string_dtype())
        h5["sample_sha256"] = np.asarray([f"sha-{i}" for i in range(5)], dtype=h5py.string_dtype())
        h5["split_id"] = np.asarray([0, 0, 1, 0, 1], dtype=np.uint8)
        h5["time_s"] = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        h5["x_m"] = np.asarray([0.0, 10.0], dtype=np.float64)
        h5["z_m"] = np.asarray([0.0, 10.0], dtype=np.float64)
    return path


def test_v3_config_has_immutable_three_family_contract(tiny_source_h5: Path):
    config = V3Config.from_mapping({"data": {"source_h5": str(tiny_source_h5)}})
    assert config.data.allowed_medium_types == ALLOWED_MEDIUM_TYPES
    assert isinstance(config.data.allowed_medium_types, tuple)
    with pytest.raises(ValueError, match="allowed_medium_types"):
        V3Config.from_mapping(
            {
                "data": {
                    "source_h5": str(tiny_source_h5),
                    "allowed_medium_types": ["uniform", "anomaly"],
                }
            }
        )


def test_manifest_filters_anomaly_and_has_stable_digest(tiny_source_h5: Path):
    first = build_manifest(tiny_source_h5)
    second = build_manifest(tiny_source_h5)
    assert first.digest == second.digest
    assert first.allowed_medium_types == ALLOWED_MEDIUM_TYPES
    assert first.counts_before["train"] == {
        "anomaly": 1,
        "layered": 1,
        "uniform": 1,
    }
    assert first.counts_after == {"train": 2, "validation": 2}
    assert first.indices_by_split == {"train": (0, 1), "validation": (2, 4)}
    assert all(record.medium_type != "anomaly" for record in first.records)


def test_assert_allowed_families_is_a_hard_error():
    assert_allowed_families(["uniform", "layered", "marmousi"])
    with pytest.raises(ValueError, match="anomaly"):
        assert_allowed_families(["uniform", "anomaly"])
    with pytest.raises(ValueError, match="unknown"):
        assert_allowed_families(["uniform", "salt"])


def test_atomic_manifest_writer_round_trips(tiny_source_h5: Path, tmp_path: Path):
    output = tmp_path / "manifest.json"
    manifest = build_manifest(tiny_source_h5)
    write_manifest_atomic(manifest, output)
    decoded = json.loads(output.read_text(encoding="utf8"))
    assert decoded["digest"] == manifest.digest
    assert decoded["counts_after"] == {"train": 2, "validation": 2}
    assert not list(tmp_path.glob("*.partial.*"))


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="production VDS is unavailable")
def test_production_vds_matches_approved_three_family_census():
    manifest = build_manifest(REAL_DATASET)
    assert manifest.counts_after["train"] == 2240
    assert manifest.counts_after["validation"] == 480
    assert manifest.counts_before["train"]["anomaly"] == 560
    assert manifest.counts_before["validation"]["anomaly"] == 120
    assert not any(record.medium_type == "anomaly" for record in manifest.records)
