from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .normalization import encode_standard
from .schema import canonicalize_velocity, canonicalize_wavefield


_LWC84_SCHEMA = "acoustic_lwc84_401_to_201_v1"
_LWC84_KEY_ALIASES = {
    "velocity_key": "velocity_mps",
    "wavefield_key": "wavefield",
    "source_map_key": "source_map",
    "wavelet_key": "source_wavelet",
    "time_key": "time_s",
    "frequency_key": "source_f0_hz",
    "amplitude_key": "source_amplitude",
    "model_type_key": "medium_type",
}
_LWC84_AXES = {
    "velocity_axes": ["sample", "z", "x"],
    "wavefield_axes": ["sample", "time", "z", "x"],
    "source_map_axes": ["sample", "z", "x"],
}


def _decode_hdf5_attr(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def resolve_lwc84_hdf5_aliases(
    data_config: dict[str, Any], h5: h5py.File
) -> dict[str, Any]:
    """Resolve legacy notebook keys against a physical LWC84 dataset.

    The alias layer is deliberately schema-scoped.  It validates the physical
    ``NTZX`` order before replacing missing legacy names, so compatibility can
    never silently reinterpret the spatial axes.
    """

    resolved = dict(data_config)
    schema = _decode_hdf5_attr(h5.attrs.get("schema_version", ""))
    if schema != _LWC84_SCHEMA:
        return resolved
    axis_order = _decode_hdf5_attr(h5.attrs.get("axis_order", ""))
    if axis_order != "NTZX":
        raise ValueError(f"LWC84 axis_order must be NTZX, got {axis_order!r}")
    for config_name, expected_axes in _LWC84_AXES.items():
        declared = resolved.get(config_name)
        if declared is not None and list(declared) != expected_axes:
            raise ValueError(
                f"LWC84 NTZX requires {config_name}={expected_axes}, got {list(declared)}"
            )
        resolved[config_name] = list(expected_axes)
    for config_name, physical_name in _LWC84_KEY_ALIASES.items():
        configured_name = resolved.get(config_name)
        if isinstance(configured_name, str) and configured_name in h5:
            continue
        if physical_name in h5:
            resolved[config_name] = physical_name
    return resolved


def make_time_indices(total: int, sampling: dict[str, Any]) -> np.ndarray:
    start = int(sampling.get("time_start") or 0)
    stop = sampling.get("time_stop")
    stop = total if stop is None else min(int(stop), total)
    stride = sampling.get("time_stride")
    max_steps = sampling.get("max_time_steps")
    if stride is None:
        if max_steps is None or stop - start <= int(max_steps):
            stride = 1
        else:
            stride = max(1, math.ceil((stop - start) / int(max_steps)))
    indices = np.arange(start, stop, int(stride), dtype=np.int64)
    if max_steps is not None:
        indices = indices[: int(max_steps)]
    if indices.size == 0:
        raise ValueError("time sampling selected zero time steps")
    return indices


def make_gaussian_source_map(
    height: int,
    width: int,
    x_idx: float,
    z_idx: float,
    sigma_grid: float = 1.5,
    normalize_max: bool = True,
) -> torch.Tensor:
    gx, gz = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    dist2 = (gx.float() - float(x_idx)) ** 2 + (gz.float() - float(z_idx)) ** 2
    source = torch.exp(-dist2 / (2.0 * float(sigma_grid) ** 2))
    if normalize_max:
        source = source / source.max().clamp_min(1e-12)
    return source


def create_splits(
    sample_count: int,
    ratios: list[float] | tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 2026,
    max_samples: int | None = None,
) -> dict[str, Any]:
    n = sample_count if max_samples is None else min(sample_count, int(max_samples))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    if n < 3:
        train = indices[:1]
        val = indices[1:2] if n > 1 else indices[:1]
        test = val
    else:
        n_train = max(1, int(round(n * float(ratios[0]))))
        n_val = max(1, int(round(n * float(ratios[1]))))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train = indices[:n_train]
        val = indices[n_train : n_train + n_val]
        test = indices[n_train + n_val :]
        if test.size == 0:
            test = val.copy()
    return {
        "seed": int(seed),
        "strategy": "fixed_seed_random_complete_samples",
        "grouping_field": None,
        "train": sorted(int(i) for i in train.tolist()),
        "val": sorted(int(i) for i in val.tolist()),
        "test": sorted(int(i) for i in test.tolist()),
        "sample_counts": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test)), "total_used": int(n)},
    }


class PinoHDF5Dataset(Dataset):
    def __init__(
        self,
        config: dict[str, Any],
        indices: list[int],
        normalization_stats: dict[str, Any] | None = None,
        return_normalized: bool = True,
    ) -> None:
        self.config = config
        self.data_config = dict(config["data"])
        with h5py.File(self.data_config["path"], "r") as handle:
            self.data_config = resolve_lwc84_hdf5_aliases(self.data_config, handle)
        self.sampling = config.get("sampling", {})
        self.source_config = config.get("source_map", {})
        self.indices = [int(i) for i in indices]
        self.normalization_stats = normalization_stats
        self.return_normalized = return_normalized
        self._file: h5py.File | None = None
        self._time_indices: np.ndarray | None = None
        self._time_values: torch.Tensor | None = None
        self._init_shape_metadata()

    def _init_shape_metadata(self) -> None:
        with h5py.File(self.data_config["path"], "r") as h5:
            wavefield = h5[self.data_config["wavefield_key"]]
            axes = list(self.data_config["wavefield_axes"])
            sample_axis = axes.index("sample")
            time_axis = axes.index("time")
            self.sample_count = int(wavefield.shape[sample_axis])
            self.total_time_steps = int(wavefield.shape[time_axis])
            self.time_indices = make_time_indices(self.total_time_steps, self.sampling)
            if self.data_config.get("time_key") and self.data_config["time_key"] in h5:
                time_values = np.asarray(h5[self.data_config["time_key"]][self.time_indices], dtype=np.float32)
            else:
                time_values = self.time_indices.astype(np.float32)
            self.raw_time_values = time_values
        self.target_height = int(self.sampling.get("target_height") or 0)
        self.target_width = int(self.sampling.get("target_width") or 0)

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.data_config["path"], "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _resize_2d(self, array: np.ndarray, mode: str = "bilinear") -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
        out = F.interpolate(
            tensor,
            size=(self.target_height, self.target_width),
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
            antialias=mode in {"bilinear", "bicubic"},
        )
        return out[0, 0]

    def _resize_wavefield(self, wavefield_hwt: np.ndarray) -> torch.Tensor:
        # [H, W, T] -> [T, 1, H, W] -> resize -> [H, W, T]
        tensor = torch.as_tensor(np.transpose(wavefield_hwt, (2, 0, 1)), dtype=torch.float32)[:, None]
        out = F.interpolate(
            tensor,
            size=(self.target_height, self.target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return out[:, 0].permute(1, 2, 0).contiguous()

    def _read_wavefield(self, h5: h5py.File, sample_index: int) -> np.ndarray:
        dset = h5[self.data_config["wavefield_key"]]
        axes = list(self.data_config["wavefield_axes"])
        if axes == ["sample", "time", "x", "z"]:
            raw = dset[sample_index, self.time_indices, :, :]
            return canonicalize_wavefield(np.asarray(raw), ["time", "x", "z"])
        if axes == ["sample", "time", "z", "x"]:
            raw = dset[sample_index, self.time_indices, :, :]
            return canonicalize_wavefield(np.asarray(raw), ["time", "z", "x"])
        if axes == ["sample", "x", "z", "time"]:
            raw = dset[sample_index, :, :, self.time_indices]
            return canonicalize_wavefield(np.asarray(raw), ["x", "z", "time"])
        if axes == ["sample", "z", "x", "time"]:
            raw = dset[sample_index, :, :, self.time_indices]
            return canonicalize_wavefield(np.asarray(raw), ["z", "x", "time"])
        raise NotImplementedError(f"wavefield_axes {axes} need explicit adapter")

    def _read_velocity(self, h5: h5py.File, sample_index: int) -> torch.Tensor:
        dset = h5[self.data_config["velocity_key"]]
        axes = list(self.data_config["velocity_axes"])
        if "sample" in axes:
            raw = dset[sample_index]
            raw_axes = [axis for axis in axes if axis != "sample"]
        else:
            raw = dset[()]
            raw_axes = axes
        velocity = canonicalize_velocity(np.asarray(raw), raw_axes)
        return self._resize_2d(velocity)

    def _read_source_map(self, h5: h5py.File, sample_index: int) -> torch.Tensor:
        key = self.data_config.get("source_map_key")
        if key and key in h5:
            axes = list(self.data_config.get("source_map_axes", ["sample", "x", "z"]))
            raw = h5[key][sample_index] if "sample" in axes else h5[key][()]
            raw_axes = [axis for axis in axes if axis != "sample"]
            source = canonicalize_velocity(np.asarray(raw), raw_axes)
            out = self._resize_2d(source)
        else:
            x_key = self.data_config.get("source_position_x_key")
            z_key = self.data_config.get("source_position_z_key")
            if not (x_key and z_key and x_key in h5 and z_key in h5):
                raise ValueError("source representation is missing; source_map or source indices required")
            raw_h = h5[self.data_config["velocity_key"]].shape[-2]
            raw_w = h5[self.data_config["velocity_key"]].shape[-1]
            x = float(h5[x_key][sample_index]) * self.target_height / raw_h
            z = float(h5[z_key][sample_index]) * self.target_width / raw_w
            out = make_gaussian_source_map(
                self.target_height,
                self.target_width,
                x,
                z,
                sigma_grid=float(self.source_config.get("sigma_grid", 1.5)),
                normalize_max=bool(self.source_config.get("normalize_max", True)),
            )
        if self.source_config.get("normalize_max", True):
            out = out / out.max().clamp_min(1e-12)
        return out

    def _read_travel_time(self, h5: h5py.File, sample_index: int) -> torch.Tensor:
        key = self.data_config.get("travel_time_key")
        if not key or key not in h5:
            raise ValueError("retarded-time input requires a travel_time_key dataset")
        return self._resize_2d(np.asarray(h5[key][sample_index], dtype=np.float32))

    def _metadata(self, h5: h5py.File, sample_index: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for cfg_key in ["source_position_x_key", "source_position_z_key", "frequency_key", "amplitude_key", "dx_key"]:
            key = self.data_config.get(cfg_key)
            if key and key in h5:
                value = h5[key][sample_index] if h5[key].shape else h5[key][()]
                metadata[cfg_key.replace("_key", "")] = float(value) if np.isscalar(value) else np.asarray(value).tolist()
        return metadata

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample_index = self.indices[item]
        h5 = self._get_file()
        wavefield = self._resize_wavefield(self._read_wavefield(h5, sample_index))
        velocity = self._read_velocity(h5, sample_index)
        source_map = self._read_source_map(h5, sample_index)
        requested_features = tuple(
            self.data_config.get("input_features", ["time", "source_map", "velocity"])
        )
        travel_time = (
            self._read_travel_time(h5, sample_index)
            if any(feature in {"travel_time", "retarded_time"} for feature in requested_features)
            else None
        )

        time = torch.as_tensor(self.raw_time_values, dtype=torch.float32)
        denom = (time.max() - time.min()).clamp_min(1e-12)
        time_norm = (time - time.min()) / denom

        if self.return_normalized and self.normalization_stats:
            velocity_in = encode_standard(velocity, self.normalization_stats["velocity"], self.config["normalization"]["eps"])
            target = encode_standard(wavefield, self.normalization_stats["wavefield"], self.config["normalization"]["eps"])
        else:
            velocity_in = velocity
            target = wavefield

        h, w, t = target.shape
        features = []
        for feature in requested_features:
            if feature == "time":
                features.append(time_norm.view(1, 1, t).expand(h, w, t))
            elif feature == "source_map":
                features.append(source_map.view(h, w, 1).expand(h, w, t))
            elif feature == "velocity":
                features.append(velocity_in.view(h, w, 1).expand(h, w, t))
            elif feature == "travel_time":
                if travel_time is None:
                    raise RuntimeError("travel-time feature was not loaded")
                features.append((travel_time / denom).view(h, w, 1).expand(h, w, t))
            elif feature == "retarded_time":
                if travel_time is None:
                    raise RuntimeError("retarded-time feature was not loaded")
                onset_key = self.data_config.get("source_t0_key", "source_t0_s")
                onset = (
                    float(h5[onset_key][sample_index])
                    if onset_key in h5
                    else 0.0
                )
                retarded = (
                    time.view(1, 1, t)
                    - float(onset)
                    - travel_time.view(h, w, 1)
                ) / denom
                features.append(retarded.clamp(-1.0, 1.0).expand(h, w, t))
            elif feature == "source_frequency":
                frequency_key = self.data_config.get(
                    "frequency_key", "source_f0_hz"
                )
                if frequency_key not in h5:
                    raise ValueError(
                        "source-frequency input requires a frequency_key dataset"
                    )
                scale_hz = float(
                    self.data_config.get("source_frequency_scale_hz", 30.0)
                )
                if not math.isfinite(scale_hz) or scale_hz <= 0.0:
                    raise ValueError("source_frequency_scale_hz must be positive")
                frequency = float(h5[frequency_key][sample_index]) / scale_hz
                features.append(
                    target.new_full((h, w, t), frequency)
                )
            elif feature == "ricker_retarded":
                if travel_time is None:
                    raise RuntimeError("Ricker-retarded feature requires travel time")
                frequency_key = self.data_config.get(
                    "frequency_key", "source_f0_hz"
                )
                onset_key = self.data_config.get("source_t0_key", "source_t0_s")
                if frequency_key not in h5 or onset_key not in h5:
                    raise ValueError(
                        "Ricker-retarded input requires frequency and onset datasets"
                    )
                frequency_hz = float(h5[frequency_key][sample_index])
                onset = float(h5[onset_key][sample_index])
                tau = (
                    time.view(1, 1, t)
                    - onset
                    - travel_time.view(h, w, 1)
                )
                phase2 = (math.pi * frequency_hz * tau).square()
                features.append((1.0 - 2.0 * phase2) * torch.exp(-phase2))
            elif feature == "green2d_retarded":
                if travel_time is None:
                    raise RuntimeError("2D Green feature requires travel time")
                frequency_key = self.data_config.get(
                    "frequency_key", "source_f0_hz"
                )
                onset_key = self.data_config.get("source_t0_key", "source_t0_s")
                time_key = self.data_config.get("time_key", "time_s")
                if (
                    frequency_key not in h5
                    or onset_key not in h5
                    or time_key not in h5
                ):
                    raise ValueError(
                        "2D Green input requires frequency, onset, and time datasets"
                    )
                frequency_hz = float(h5[frequency_key][sample_index])
                onset = float(h5[onset_key][sample_index])
                full_time = torch.as_tensor(
                    np.asarray(h5[time_key][:], dtype=np.float32)
                )
                full_dt = torch.median(torch.diff(full_time))
                source_tau = full_time - onset
                source_phase2 = (math.pi * frequency_hz * source_tau).square()
                source_trace = (1.0 - 2.0 * source_phase2) * torch.exp(
                    -source_phase2
                )
                lag = (
                    torch.arange(len(full_time), dtype=torch.float32) + 0.5
                ) * full_dt
                kernel = lag.rsqrt()
                fft_size = 1
                while fft_size < 2 * len(full_time) - 1:
                    fft_size *= 2
                convolved = torch.fft.irfft(
                    torch.fft.rfft(source_trace, n=fft_size)
                    * torch.fft.rfft(kernel, n=fft_size),
                    n=fft_size,
                )[: len(full_time)] * full_dt
                convolved = convolved / convolved.abs().amax().clamp_min(1.0e-12)
                shifted = (
                    time.view(1, 1, t) - travel_time.view(h, w, 1)
                ) / full_dt
                left = torch.floor(shifted).long()
                fraction = shifted - left.to(shifted.dtype)
                valid = (left >= 0) & (left < len(full_time) - 1)
                safe_left = left.clamp(0, len(full_time) - 2)
                green = (
                    convolved[safe_left] * (1.0 - fraction)
                    + convolved[safe_left + 1] * fraction
                )
                features.append(torch.where(valid, green, torch.zeros_like(green)))
            else:
                raise ValueError(f"unsupported input feature: {feature}")
        model_input = torch.stack(features, dim=-1).contiguous()
        return {
            "input": model_input,
            "target": target.contiguous(),
            "velocity": velocity.unsqueeze(0).contiguous(),
            "source_map": source_map.unsqueeze(0).contiguous(),
            "time": time.contiguous(),
            "sample_index": int(sample_index),
            "metadata": self._metadata(h5, sample_index),
        }


def collate_pino(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["input", "target", "velocity", "source_map", "time"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["sample_index"] = torch.tensor([item["sample_index"] for item in batch], dtype=torch.long)
    out["metadata"] = [item["metadata"] for item in batch]
    return out


def sample_count_from_config(config: dict[str, Any]) -> int:
    with h5py.File(config["data"]["path"], "r") as h5:
        data_config = resolve_lwc84_hdf5_aliases(config["data"], h5)
        dset = h5[data_config["wavefield_key"]]
        axes = list(data_config["wavefield_axes"])
        return int(dset.shape[axes.index("sample")])
