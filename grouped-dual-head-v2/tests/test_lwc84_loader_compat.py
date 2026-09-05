from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from fno_acoustic.data import PinoHDF5Dataset, resolve_lwc84_hdf5_aliases


def _file(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["schema_version"] = "acoustic_lwc84_401_to_201_v1"
        h5.attrs["axis_order"] = "NTZX"
        h5.create_dataset("velocity_mps", (2, 5, 5), dtype="f4")
        h5.create_dataset("wavefield", (2, 3, 5, 5), dtype="f4")
        h5.create_dataset("source_map", data=np.ones((2, 5, 5), dtype=np.float32) / 25)
        h5.create_dataset("source_wavelet", (2, 3), dtype="f4")
        h5.create_dataset("time_s", data=np.arange(3, dtype=np.float64))
        h5.create_dataset("source_f0_hz", data=np.asarray([10, 15], dtype=np.float32))
        h5.create_dataset("source_amplitude", data=np.ones(2, dtype=np.float32))
        h5.create_dataset("source_t0_s", data=np.asarray([0.5, 0.25], dtype=np.float32))
        h5.create_dataset("travel_time_s", data=np.full((2, 5, 5), 0.25, dtype=np.float32))
        h5.create_dataset("medium_type", data=np.asarray([b"uniform", b"layered"]))


def test_legacy_names_resolve_to_new_physical_names(tmp_path: Path) -> None:
    path = tmp_path / "new.h5"
    _file(path)
    data = {
        "path": str(path),
        "velocity_key": "nu",
        "wavefield_key": "tensor",
        "source_map_key": "source_mask",
        "wavelet_key": "wavelet",
        "time_key": "t-coordinate",
        "frequency_key": "fm",
        "amplitude_key": "amplitude",
        "model_type_key": "model_type",
        "velocity_axes": ["sample", "z", "x"],
        "wavefield_axes": ["sample", "time", "z", "x"],
        "source_map_axes": ["sample", "z", "x"],
    }
    with h5py.File(path, "r") as h5:
        resolved = resolve_lwc84_hdf5_aliases(data, h5)
    assert resolved["velocity_key"] == "velocity_mps"
    assert resolved["wavefield_key"] == "wavefield"
    assert resolved["source_map_key"] == "source_map"
    assert resolved["time_key"] == "time_s"


def test_alias_layer_does_not_hide_axis_order_error(tmp_path: Path) -> None:
    path = tmp_path / "new.h5"
    _file(path)
    bad = {
        "velocity_key": "nu",
        "wavefield_key": "tensor",
        "velocity_axes": ["sample", "x", "z"],
        "wavefield_axes": ["sample", "time", "x", "z"],
    }
    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="NTZX"):
        resolve_lwc84_hdf5_aliases(bad, h5)


def test_missing_optional_alias_key_resolves_without_none_membership(tmp_path: Path) -> None:
    path = tmp_path / "new.h5"
    _file(path)
    data = {
        "velocity_key": "velocity_mps",
        "wavefield_key": "wavefield",
        "source_map_key": "source_map",
        "velocity_axes": ["sample", "z", "x"],
        "wavefield_axes": ["sample", "time", "z", "x"],
        "source_map_axes": ["sample", "z", "x"],
    }
    with h5py.File(path, "r") as h5:
        resolved = resolve_lwc84_hdf5_aliases(data, h5)
    assert resolved["model_type_key"] == "medium_type"


def test_retarded_time_feature_uses_onset_and_travel_time(tmp_path: Path) -> None:
    path = tmp_path / "new.h5"
    _file(path)
    config = {
        "data": {
            "path": str(path),
            "velocity_key": "velocity_mps",
            "wavefield_key": "wavefield",
            "source_map_key": "source_map",
            "time_key": "time_s",
            "travel_time_key": "travel_time_s",
            "source_t0_key": "source_t0_s",
            "frequency_key": "source_f0_hz",
            "source_frequency_scale_hz": 30.0,
            "velocity_axes": ["sample", "z", "x"],
            "wavefield_axes": ["sample", "time", "z", "x"],
            "source_map_axes": ["sample", "z", "x"],
            "input_features": [
                "retarded_time",
                "source_map",
                "velocity",
                "source_frequency",
                "ricker_retarded",
                "green2d_retarded",
            ],
        },
        "sampling": {"target_height": 5, "target_width": 5, "max_time_steps": 3},
        "source_map": {"normalize_max": True},
        "normalization": {"eps": 1.0e-16},
    }
    dataset = PinoHDF5Dataset(config, [0], return_normalized=False)
    try:
        item = dataset[0]
    finally:
        dataset.close()
    expected = torch.tensor([-0.375, 0.125, 0.625])
    assert torch.allclose(item["input"][0, 0, :, 0], expected)
    assert torch.allclose(
        item["input"][..., 3],
        torch.full((5, 5, 3), 1.0 / 3.0),
    )
    tau = torch.tensor([-0.75, 0.25, 1.25])
    phase2 = (torch.pi * 10.0 * tau).square()
    expected_ricker = (1.0 - 2.0 * phase2) * torch.exp(-phase2)
    assert torch.allclose(item["input"][0, 0, :, 4], expected_ricker)
    green = item["input"][..., 5]
    assert torch.isfinite(green).all()
    assert float(green.abs().max()) <= 1.0 + 1.0e-6
