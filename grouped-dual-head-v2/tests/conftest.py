from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture()
def tiny_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_pino.hdf5"
    n, h, w, t = 4, 16, 12, 10
    x = np.linspace(0, 1, h, dtype=np.float32)
    z = np.linspace(0, 1, w, dtype=np.float32)
    tt = np.linspace(0, 0.009, t, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    nu = np.stack([(2.0 + i * 0.1 + xx + 0.2 * zz).astype(np.float32) for i in range(n)])
    tensor = np.empty((n, t, h, w), dtype=np.float32)
    source_mask = np.empty((n, h, w), dtype=np.float32)
    sx = np.array([4, 5, 6, 7], dtype=np.int32)
    sz = np.array([3, 4, 5, 6], dtype=np.int32)
    for i in range(n):
        dist2 = (np.arange(h)[:, None] - sx[i]) ** 2 + (np.arange(w)[None, :] - sz[i]) ** 2
        source_mask[i] = np.exp(-dist2 / (2 * 2.0**2)).astype(np.float32)
        source_mask[i] /= source_mask[i].max()
        for ti, tv in enumerate(tt):
            tensor[i, ti] = np.sin(2 * np.pi * (ti + 1) / t) * source_mask[i] + 0.01 * nu[i]
    with h5py.File(path, "w") as h5:
        h5.attrs["dt"] = 0.001
        h5.attrs["fm"] = 25.0
        h5.create_dataset("nu", data=nu, chunks=(1, h, w))
        h5.create_dataset("tensor", data=tensor, chunks=(1, t, h, w))
        h5.create_dataset("source_mask", data=source_mask, chunks=(1, h, w))
        h5.create_dataset("source_x_idx", data=sx)
        h5.create_dataset("source_z_idx", data=sz)
        h5.create_dataset("t-coordinate", data=tt)
        h5.create_dataset("x-coordinate", data=x)
        h5.create_dataset("y-coordinate", data=z)
        h5.create_dataset("dx", data=np.full((n,), 5.0, dtype=np.float32))
        h5.create_dataset("source_frequency_hz", data=np.full((n,), 25.0, dtype=np.float32))
        h5.create_dataset("source_amplitude", data=np.ones((n,), dtype=np.float32))
        h5.create_dataset("wavelet", data=np.zeros((n, 20), dtype=np.float32))
    return path


@pytest.fixture()
def tiny_config(tiny_hdf5: Path) -> dict:
    return {
        "seed": 2026,
        "data": {
            "path": str(tiny_hdf5),
            "velocity_key": "nu",
            "wavefield_key": "tensor",
            "source_map_key": "source_mask",
            "source_position_x_key": "source_x_idx",
            "source_position_z_key": "source_z_idx",
            "frequency_key": "source_frequency_hz",
            "amplitude_key": "source_amplitude",
            "time_key": "t-coordinate",
            "x_key": "x-coordinate",
            "z_key": "y-coordinate",
            "dx_key": "dx",
            "velocity_axes": ["sample", "x", "z"],
            "wavefield_axes": ["sample", "time", "x", "z"],
            "source_map_axes": ["sample", "x", "z"],
            "input_features": ["time", "source_map", "velocity"],
            "split": [0.5, 0.25, 0.25],
            "split_seed": 2026,
            "max_samples": None,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "sampling": {
            "target_height": 8,
            "target_width": 8,
            "time_start": 0,
            "time_stop": None,
            "time_stride": None,
            "max_time_steps": 5,
        },
        "source_map": {"sigma_grid": 1.5, "normalize_max": True},
        "normalization": {"eps": 1e-6, "stats_path": str(tiny_hdf5.parent / "stats.json"), "max_stats_samples": None},
        "model": {
            "in_features": 3,
            "out_channels": 1,
            "modes_x": 2,
            "modes_z": 2,
            "modes_t": 2,
            "width": 4,
            "n_layers": 1,
            "padding_ratio": 0.0,
            "activation": "gelu",
            "normalization": "batch",
        },
        "train": {
            "batch_size": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1,
            "max_train_batches": 1,
            "max_val_batches": 1,
            "device": "cpu",
            "amp": False,
            "grad_clip": 1.0,
            "checkpoint_dir": str(tiny_hdf5.parent / "ckpts"),
            "log_dir": str(tiny_hdf5.parent / "logs"),
        },
        "loss": {"relative_l2_weight": 1.0, "mse_weight": 0.0, "eps": 1e-8},
        "evaluation": {"output_dir": str(tiny_hdf5.parent / "eval"), "max_samples": 1},
    }
