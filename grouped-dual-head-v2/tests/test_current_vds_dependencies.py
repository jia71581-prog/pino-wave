from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from scripts.audit_current_vds_dependencies import audit_vds


def _tiny_vds(tmp_path: Path, *, remove_source: bool = False) -> Path:
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("velocity_mps", shape=(1, 2, 3), dtype="f4")
        handle.create_dataset("wavefield", shape=(1, 4, 2, 3), dtype="f4")
    layout_velocity = h5py.VirtualLayout(shape=(1, 2, 3), dtype="f4")
    layout_velocity[:] = h5py.VirtualSource(str(source), "velocity_mps", shape=(1, 2, 3))
    layout_wavefield = h5py.VirtualLayout(shape=(1, 4, 2, 3), dtype="f4")
    layout_wavefield[:] = h5py.VirtualSource(str(source), "wavefield", shape=(1, 4, 2, 3))
    vds = tmp_path / "dataset_v1.h5"
    with h5py.File(vds, "w", libver="latest") as handle:
        for name in (
            "completed_mask",
            "group_id",
            "medium_type",
            "sample_id",
            "source_amplitude",
            "source_f0_hz",
            "source_map",
            "source_t0_s",
            "source_x_m",
            "source_z_m",
            "split",
        ):
            handle.create_dataset(name, shape=(1,), dtype="f4")
        handle.create_dataset("time_s", shape=(4,), dtype="f4")
        handle.create_dataset("x_m", shape=(3,), dtype="f4")
        handle.create_dataset("z_m", shape=(2,), dtype="f4")
        handle.create_virtual_dataset("velocity_mps", layout_velocity)
        handle.create_virtual_dataset("wavefield", layout_wavefield)
    if remove_source:
        source.unlink()
    return vds


def test_metadata_only_vds_audit_passes_with_complete_sources(tmp_path: Path) -> None:
    report = audit_vds(_tiny_vds(tmp_path))
    assert report["status"] == "pass"
    assert report["virtual_datasets"]["wavefield"]["missing_source_count"] == 0


def test_metadata_only_vds_audit_fails_closed_on_missing_source(tmp_path: Path) -> None:
    report = audit_vds(_tiny_vds(tmp_path, remove_source=True))
    assert report["status"] == "fail"
    assert report["virtual_datasets"]["velocity_mps"]["missing_source_count"] == 1


def test_vds_audit_rejects_absent_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit_vds(tmp_path / "absent.h5")
