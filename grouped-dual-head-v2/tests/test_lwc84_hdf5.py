from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from fno_acoustic.data_generation.hdf5_lwc84 import (
    LWC84ShardWriter,
    build_lwc84_dataset_vds,
    build_lwc84_vds,
    compute_train_only_dataset_stats,
    compute_train_only_stats,
    validate_lwc84_dataset_vds,
    validate_lwc84_shard,
)
from scripts.launch_gpu_dataset_workers import _validate_expected_shard


def _writer(
    path: Path,
    *,
    split: str = "train",
    resume: bool = False,
    expected_sample_ids: list[str] | None = None,
) -> LWC84ShardWriter:
    return LWC84ShardWriter(
        path,
        sample_count=2,
        split=split,
        time_s=np.arange(3, dtype=np.float64) * 0.005,
        x_m=np.arange(5, dtype=np.float64) * 10.0,
        z_m=np.arange(5, dtype=np.float64) * 10.0,
        attrs={
            "dt_requested_s": 0.0002,
            "dt_used_s": 0.0002,
            "snapshot_stride": 25,
            "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "marmousi_sha256": "c" * 64,
            "git_commit": "test",
            "software_environment": json.dumps({"torch": "test"}),
        },
        resume=resume,
        expected_sample_ids=expected_sample_ids,
    )


def _sample(value: float, index: int) -> dict[str, object]:
    return {
        "velocity_mps": np.full((5, 5), 2000.0 + value, dtype=np.float32),
        "wavefield": np.full((3, 5, 5), value, dtype=np.float32),
        "source_map": np.eye(5, dtype=np.float32) / 5.0,
        "source_wavelet": np.asarray([0.0, value, 0.0], dtype=np.float32),
        "source_x_m": 20.0,
        "source_z_m": 10.0,
        "source_f0_hz": 10.0 + index,
        "source_t0_s": 0.15,
        "source_amplitude": 1.0,
        "medium_type": "uniform",
        "sample_id": f"sample-{index}",
        "group_id": f"group-{index}",
        "seed": 100 + index,
        "vmin_mps": 2000.0 + value,
        "vmax_mps": 2000.0 + value,
        "cfl_2d": 0.12,
        "lwc_qmax": 0.4,
        "dt_used_s": 0.0002,
        "crop_x0_m": np.nan,
        "crop_z0_m": np.nan,
        "qc_max_abs": abs(value),
        "qc_final_energy_ratio": 0.1,
        "qc_status": "passed",
    }


def _complete(
    path: Path,
    offset: float = 0.0,
    *,
    split: str = "train",
    sample_id_offset: int = 0,
) -> Path:
    writer = _writer(path, split=split)
    for index in range(2):
        writer.write_sample(
            index,
            **_sample(offset + index + 1.0, sample_id_offset + index),
        )
    return writer.commit()


def test_atomic_resume_schema_and_chunks(tmp_path: Path) -> None:
    final = tmp_path / "train-000.h5"
    writer = _writer(final)
    writer.write_sample(0, **_sample(1.0, 0))
    writer.close_partial()
    assert not final.exists()
    assert final.with_suffix(".h5.tmp").exists()

    resumed = _writer(final, resume=True)
    assert resumed.is_complete(0)
    resumed.write_sample(1, **_sample(2.0, 1))
    assert resumed.commit() == final
    assert not final.with_suffix(".h5.tmp").exists()
    assert final.with_suffix(".h5.sha256").exists()

    summary = validate_lwc84_shard(final, strict=True)
    assert summary["sample_count"] == 2
    with h5py.File(final, "r") as h5:
        assert h5.attrs["axis_order"] == "NTZX"
        assert h5.attrs["schema_version"] == "acoustic_lwc84_401_to_201_v1"
        assert h5["wavefield"].shape == (2, 3, 5, 5)
        assert h5["wavefield"].chunks == (1, 1, 5, 5)
        assert h5["wavefield"].compression == "lzf"
        assert h5["velocity_mps"].dtype == np.float32
        assert np.allclose(h5["source_map"][:].sum(axis=(1, 2)), 1.0)


def test_resume_rejects_completed_sample_from_another_filtered_scope(tmp_path: Path) -> None:
    final = tmp_path / "train-000.h5"
    writer = _writer(
        final,
        expected_sample_ids=["sample-0", "sample-1"],
    )
    writer.write_sample(0, **_sample(1.0, 0))
    writer.close_partial()

    with pytest.raises(ValueError, match="sample-ID binding|completed sample binding"):
        _writer(
            final,
            resume=True,
            expected_sample_ids=["other-0", "other-1"],
        )


def test_validator_rejects_root_and_per_sample_timestep_drift(tmp_path: Path) -> None:
    path = _complete(tmp_path / "train-000.h5")
    with h5py.File(path, "r+") as h5:
        h5.attrs["dt_used_s"] = 1.0e-4

    with pytest.raises(ValueError, match="do not align|differs from the root protocol"):
        validate_lwc84_shard(path, strict=False)


def test_empty_legacy_partial_can_be_bound_before_safe_resume(tmp_path: Path) -> None:
    final = tmp_path / "train-000.h5"
    writer = _writer(final)
    writer.close_partial()

    resumed = _writer(
        final,
        resume=True,
        expected_sample_ids=["sample-0", "sample-1"],
    )
    resumed.write_sample(0, **_sample(1.0, 0))
    resumed.write_sample(1, **_sample(2.0, 1))
    resumed.commit()

    with h5py.File(final, "r") as h5:
        assert len(h5.attrs["expected_sample_ids_sha256"]) == 64


def test_vds_ignores_tmp_and_rejects_corrupt_final(tmp_path: Path) -> None:
    shard0 = _complete(tmp_path / "train-000.h5", 0.0)
    partial = _writer(tmp_path / "train-001.h5")
    partial.write_sample(0, **_sample(9.0, 0))
    partial.close_partial()
    vds = build_lwc84_vds(tmp_path / "train-vds.h5", [shard0, partial.partial_path])
    with h5py.File(vds, "r") as h5:
        assert h5["wavefield"].is_virtual
        assert h5["wavefield"].shape[0] == 2

    shard0.with_suffix(".h5.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="SHA-256"):
        build_lwc84_vds(tmp_path / "bad-vds.h5", [shard0])


def test_final_shard_collector_rejects_valid_ids_in_wrong_ordinal_order(
    tmp_path: Path,
) -> None:
    shard = _complete(tmp_path / "train-000.h5")

    with pytest.raises(ValueError, match="ordinal/sample ordering mismatch"):
        _validate_expected_shard(
            shard,
            split="train",
            expected_ids=["sample-1", "sample-0"],
            config_sha256="a" * 64,
            manifest_sha256="b" * 64,
            marmousi_sha256="c" * 64,
            strict=True,
        )


def test_dataset_vds_combines_all_splits_into_one_file(tmp_path: Path) -> None:
    train = _complete(tmp_path / "train" / "train-000.h5", 0.0, split="train")
    validation = _complete(
        tmp_path / "validation" / "validation-000.h5",
        10.0,
        split="validation",
        sample_id_offset=2,
    )
    ood = _complete(
        tmp_path / "ood_canonical" / "ood_canonical-000.h5",
        20.0,
        split="ood_canonical",
        sample_id_offset=4,
    )

    output = build_lwc84_dataset_vds(tmp_path / "dataset_v1.h5", [ood, validation, train])
    summary = validate_lwc84_dataset_vds(output, strict=True, expected_n=6)
    assert summary["sample_count"] == 6
    assert summary["source_shard_count"] == 3
    stats = compute_train_only_dataset_stats(output, time_chunk_size=2)
    assert stats["train_sample_count"] == 2
    assert stats["velocity"]["count"] == 50
    assert stats["wavefield"]["count"] == 150

    with h5py.File(output, "r") as h5:
        assert h5.attrs["split"] == "all"
        assert h5.attrs["vds_sample_count"] == 6
        assert h5["wavefield"].is_virtual
        assert h5["wavefield"].shape == (6, 3, 5, 5)
        assert h5["split"].asstr()[:].tolist() == [
            "train", "train", "validation", "validation", "ood_canonical", "ood_canonical"
        ]
        assert h5["split_id"][:].tolist() == [0, 0, 1, 1, 3, 3]


def test_dataset_vds_can_reuse_already_strictly_validated_shards(tmp_path: Path) -> None:
    train = _complete(tmp_path / "train" / "train-000.h5", 0.0, split="train")
    train.with_suffix(".h5.sha256").write_text("0" * 64 + "\n", encoding="ascii")

    output = build_lwc84_dataset_vds(
        tmp_path / "dataset_v1.h5", [train], strict_validation=False
    )

    with h5py.File(output, "r") as h5:
        assert h5["wavefield"].is_virtual
        assert h5["split"].asstr()[:].tolist() == ["train", "train"]


def test_train_only_streaming_stats_rejects_non_train(tmp_path: Path) -> None:
    train = _complete(tmp_path / "train.h5", 0.0)
    stats = compute_train_only_stats([train])
    assert stats["computed_from_split"] == "train"
    assert stats["train_sample_count"] == 2
    expected_wavefield = np.concatenate(
        [np.full(75, 1.0), np.full(75, 2.0)]
    )
    assert stats["wavefield"]["mean"] == pytest.approx(float(expected_wavefield.mean()))
    assert stats["wavefield"]["std"] == pytest.approx(float(expected_wavefield.std(ddof=1)))

    validation = _complete(tmp_path / "validation.h5", 0.0, split="validation")
    with pytest.raises(ValueError, match="train"):
        compute_train_only_stats([validation])
