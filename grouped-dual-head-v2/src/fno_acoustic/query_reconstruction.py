"""Deterministic, CPU-backed reconstruction of native full160 wavefields."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch

from .temporal_operator import _ValidatedTimeGrid, _make_validated_time_grid


_TIME_STEPS = 160


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_scene_device(
    requested: torch.device, actual_devices: tuple[torch.device, ...]
) -> torch.device:
    """Resolve an unindexed device without weakening cross-device validation."""

    if not actual_devices:
        raise ValueError("at least one tensor device is required")
    actual = actual_devices[0]
    if any(device != actual for device in actual_devices[1:]):
        raise ValueError("scene tensors must all be on the same device")
    requested_matches = requested.type == actual.type and (
        requested.index is None or requested.index == actual.index
    )
    if not requested_matches:
        raise ValueError("scene.device must match the actual tensor device")
    return actual


@dataclass(frozen=True)
class QueryInferenceScene:
    """Validated inputs for reconstructing one batch on a native spatial grid."""

    global_inputs: torch.Tensor
    native_static: torch.Tensor
    time_s: torch.Tensor
    height: int
    width: int
    batch: int
    device: torch.device
    dx_m: float | None = None
    dz_m: float | None = None
    validated_time_grid: _ValidatedTimeGrid = field(init=False, repr=False)

    def __post_init__(self) -> None:
        height = _positive_integer("height", self.height)
        width = _positive_integer("width", self.width)
        batch = _positive_integer("batch", self.batch)
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device")
        if (self.dx_m is None) != (self.dz_m is None):
            raise ValueError("dx_m and dz_m must be provided together")
        for name, value in (("dx_m", self.dx_m), ("dz_m", self.dz_m)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if not isinstance(self.global_inputs, torch.Tensor):
            raise ValueError("global_inputs must be a torch.Tensor")
        if not isinstance(self.native_static, torch.Tensor):
            raise ValueError("native_static must be a torch.Tensor")
        if not isinstance(self.time_s, torch.Tensor):
            raise ValueError("time_s must be a torch.Tensor")
        if self.global_inputs.ndim != 5:
            raise ValueError("global_inputs must have shape [B,Hg,Wg,160,Cg]")
        if (
            self.global_inputs.shape[0] != batch
            or self.global_inputs.shape[3] != _TIME_STEPS
            or min(
                self.global_inputs.shape[1],
                self.global_inputs.shape[2],
                self.global_inputs.shape[4],
            )
            < 1
        ):
            raise ValueError("global_inputs must have shape [B,Hg,Wg,160,Cg]")
        if self.native_static.ndim != 4:
            raise ValueError("native_static must have shape [B,C,H,W]")
        if self.native_static.shape[0] != batch or self.native_static.shape[2:] != (
            height,
            width,
        ):
            raise ValueError("native_static must have shape [B,C,H,W]")
        if self.native_static.shape[1] < 1:
            raise ValueError("native_static must contain at least one channel")
        if self.time_s.ndim == 1:
            valid_time_shape = self.time_s.shape == (_TIME_STEPS,)
        elif self.time_s.ndim == 2:
            valid_time_shape = self.time_s.shape == (batch, _TIME_STEPS)
        else:
            valid_time_shape = False
        if not valid_time_shape:
            raise ValueError("time_s must have shape [160] or [B,160]")
        if self.time_s.ndim == 2 and not torch.equal(
            self.time_s, self.time_s[0].expand_as(self.time_s)
        ):
            raise ValueError("batched time_s must contain one shared time vector")
        shared_time = self.time_s if self.time_s.ndim == 1 else self.time_s[0]
        time_grid = _make_validated_time_grid(shared_time)
        resolved_device = _resolve_scene_device(
            self.device,
            (
                self.global_inputs.device,
                self.native_static.device,
                self.time_s.device,
            ),
        )
        object.__setattr__(self, "device", resolved_device)
        object.__setattr__(self, "time_s", time_grid.time_s)
        object.__setattr__(self, "validated_time_grid", time_grid)


@dataclass(frozen=True)
class SpatialQueryChunk:
    """A row-major subset of native sites and normalized physical ``(x,z)``."""

    site_indices: torch.Tensor
    query_xz: torch.Tensor


@dataclass(frozen=True)
class NativeReconstruction:
    """A CPU float32 ``[B,H,W,160]`` field and flat CPU coverage counts."""

    field: torch.Tensor | np.memmap
    coverage: torch.Tensor


def iter_spatial_query_chunks(
    height: int, width: int, chunk_sites: int
) -> Iterator[SpatialQueryChunk]:
    """Yield every native site once in deterministic row-major chunks."""

    height = _positive_integer("height", height)
    width = _positive_integer("width", width)
    chunk_sites = _positive_integer("chunk_sites", chunk_sites)
    total = height * width
    for start in range(0, total, chunk_sites):
        indices = torch.arange(start, min(total, start + chunk_sites), dtype=torch.int64)
        x_index = torch.div(indices, width, rounding_mode="floor")
        z_index = indices.remainder(width)
        query_xz = torch.stack(
            (
                x_index.to(torch.float32) / max(height - 1, 1),
                z_index.to(torch.float32) / max(width - 1, 1),
            ),
            dim=-1,
        )
        yield SpatialQueryChunk(site_indices=indices, query_xz=query_xz)


def _validate_output(
    output: torch.Tensor | np.memmap | None, shape: tuple[int, int, int, int]
) -> torch.Tensor | np.memmap:
    if output is None:
        return torch.empty(shape, dtype=torch.float32, device="cpu")
    if isinstance(output, torch.Tensor):
        if output.shape != shape:
            raise ValueError(f"output must have shape {shape}")
        if output.dtype != torch.float32:
            raise ValueError("output must have dtype torch.float32")
        if output.device.type != "cpu":
            raise ValueError("output tensor must be on CPU")
        if not output.is_contiguous():
            raise ValueError("output tensor must be contiguous")
        return output
    if isinstance(output, np.memmap):
        if output.shape != shape:
            raise ValueError(f"output must have shape {shape}")
        if output.dtype != np.dtype(np.float32):
            raise ValueError("output memmap must have dtype numpy.float32")
        if not output.flags.c_contiguous:
            raise ValueError("output memmap must be C-contiguous")
        if not output.flags.writeable:
            raise ValueError("output memmap must be writable")
        return output
    raise TypeError("output must be a CPU torch.Tensor, numpy.memmap, or None")


def _invalidate_destination(destination: torch.Tensor | np.memmap) -> None:
    """Mark a full output unusable without allocating another native field."""

    if isinstance(destination, torch.Tensor):
        destination.fill_(torch.nan)
    else:
        destination.fill(np.nan)
        destination.flush()


@torch.no_grad()
def reconstruct_native(
    model: object,
    scene: QueryInferenceScene,
    chunk_sites: int,
    output: torch.Tensor | np.memmap | None = None,
) -> NativeReconstruction:
    """Decode native sites in bounded device chunks into a CPU float32 field."""

    if not isinstance(scene, QueryInferenceScene):
        raise TypeError("scene must be a QueryInferenceScene")
    chunk_sites = _positive_integer("chunk_sites", chunk_sites)
    shape = (scene.batch, scene.height, scene.width, _TIME_STEPS)
    destination = _validate_output(output, shape)
    _invalidate_destination(destination)
    if isinstance(destination, np.memmap):
        flat_memmap = destination.reshape(
            scene.batch, scene.height * scene.width, _TIME_STEPS
        )
        flat_tensor = None
    else:
        flat_tensor = destination.reshape(
            scene.batch, scene.height * scene.width, _TIME_STEPS
        )
        flat_memmap = None
    coverage = torch.zeros(scene.height * scene.width, dtype=torch.int32, device="cpu")

    try:
        scene_validation = getattr(model, "supports_validated_time_grid", False)
        if scene_validation:
            context = model._encode_global_validated(
                scene.global_inputs, scene.validated_time_grid
            )
        else:
            context = model.encode_global(scene.global_inputs, scene.time_s)
        for chunk in iter_spatial_query_chunks(scene.height, scene.width, chunk_sites):
            coords = (
                chunk.query_xz.to(scene.device)
                .unsqueeze(0)
                .expand(scene.batch, -1, -1)
            )
            kwargs = (
                {"dx_m": scene.dx_m, "dz_m": scene.dz_m}
                if getattr(model, "supports_physical_spacing", False)
                else {}
            )
            if scene_validation:
                values = model._decode_queries_validated(
                    context,
                    scene.native_static,
                    coords,
                    scene.validated_time_grid,
                    **kwargs,
                    spacing_host_validated=(
                        scene.dx_m is not None and scene.dz_m is not None
                    ),
                )
            else:
                values = model.decode_queries(
                    context, scene.native_static, coords, scene.time_s, **kwargs
                )
            expected = (scene.batch, chunk.site_indices.numel(), _TIME_STEPS)
            if not isinstance(values, torch.Tensor) or values.shape != expected:
                raise ValueError(
                    f"decode_queries must return a tensor with shape {expected}"
                )
            if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
                raise ValueError("decode_queries output must contain finite floats")
            values_cpu = values.detach().to(device="cpu", dtype=torch.float32)
            if flat_memmap is None:
                assert flat_tensor is not None
                flat_tensor[:, chunk.site_indices, :] = values_cpu
            else:
                flat_memmap[:, chunk.site_indices.numpy(), :] = values_cpu.numpy()
            coverage[chunk.site_indices] += 1
            del values, values_cpu, coords

        if not bool(torch.all(coverage == 1)):
            raise RuntimeError("native reconstruction coverage must equal one at every site")
        if isinstance(destination, np.memmap):
            destination.flush()
        return NativeReconstruction(field=destination, coverage=coverage)
    except BaseException:
        try:
            _invalidate_destination(destination)
        except BaseException:
            pass
        raise


__all__ = [
    "NativeReconstruction",
    "QueryInferenceScene",
    "SpatialQueryChunk",
    "iter_spatial_query_chunks",
    "reconstruct_native",
]
