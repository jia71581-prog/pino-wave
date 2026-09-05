"""Grid shortest-path approximation to source-dependent Eikonal travel time."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


def _primitive_offsets(radius: int = 4) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for dz in range(0, int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            if dz == 0 and dx <= 0:
                continue
            if dz == 0 and dx == 0:
                continue
            if math.hypot(dz, dx) > float(radius) + 0.5:
                continue
            if math.gcd(abs(dz), abs(dx)) != 1:
                continue
            offsets.append((dz, dx))
    if not offsets:
        raise ValueError("Eikonal graph radius selected no offsets")
    return tuple(offsets)


def _travel_graph(
    velocity_mps: np.ndarray,
    *,
    dx_m: float,
    dz_m: float,
) -> sparse.csr_matrix:
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    height, width = velocity.shape
    node = np.arange(height * width, dtype=np.int64).reshape(height, width)
    slowness = np.reciprocal(velocity)
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for offset_z, offset_x in _primitive_offsets():
        if offset_z >= height or abs(offset_x) >= width:
            continue
        z_from = slice(0, height - offset_z)
        z_to = slice(offset_z, height)
        if offset_x >= 0:
            x_from = slice(0, width - offset_x)
            x_to = slice(offset_x, width)
        else:
            x_from = slice(-offset_x, width)
            x_to = slice(0, width + offset_x)
        source = node[z_from, x_from].reshape(-1)
        target = node[z_to, x_to].reshape(-1)
        distance = math.hypot(offset_z * float(dz_m), offset_x * float(dx_m))
        weight = (
            0.5 * (slowness[z_from, x_from] + slowness[z_to, x_to]) * distance
        ).reshape(-1)
        row_parts.extend((source, target))
        column_parts.extend((target, source))
        weight_parts.extend((weight, weight))
    return sparse.csr_matrix(
        (
            np.concatenate(weight_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(height * width, height * width),
    )


def grid_eikonal_travel_time(
    velocity_mps: np.ndarray,
    *,
    source_indices: Sequence[tuple[int, int]],
    dx_m: float,
    dz_m: float,
) -> np.ndarray:
    """Return `[source,z,x]` first-arrival times on one velocity grid.

    A radius-four primitive-direction graph reduces directional bias relative to
    an 8-neighbour grid while allowing all sources sharing a medium to reuse one
    sparse graph.
    """

    velocity = np.asarray(velocity_mps)
    if velocity.ndim != 2 or velocity.size == 0:
        raise ValueError("velocity must be a nonempty 2-D grid")
    if not np.issubdtype(velocity.dtype, np.floating):
        velocity = velocity.astype(np.float32)
    if not np.isfinite(velocity).all() or np.any(velocity <= 0.0):
        raise ValueError("velocity must be finite and positive")
    if not math.isfinite(float(dx_m)) or not math.isfinite(float(dz_m)):
        raise ValueError("grid spacing must be finite")
    if float(dx_m) <= 0.0 or float(dz_m) <= 0.0:
        raise ValueError("grid spacing must be positive")
    sources = tuple((int(z), int(x)) for z, x in source_indices)
    if not sources:
        raise ValueError("at least one source index is required")
    height, width = velocity.shape
    if any(z < 0 or z >= height or x < 0 or x >= width for z, x in sources):
        raise ValueError("source index lies outside the velocity grid")

    graph = _travel_graph(velocity, dx_m=float(dx_m), dz_m=float(dz_m))
    flat_sources = np.asarray([z * width + x for z, x in sources], dtype=np.int64)
    travel = dijkstra(
        graph,
        directed=False,
        indices=flat_sources,
        return_predecessors=False,
    )
    return np.asarray(travel, dtype=np.float32).reshape(len(sources), height, width)


def debiased_grid_eikonal_travel_time(
    velocity_mps: np.ndarray,
    *,
    source_coordinates_m: Sequence[tuple[float, float]],
    dx_m: float,
    dz_m: float,
    x0_m: float = 0.0,
    z0_m: float = 0.0,
) -> np.ndarray:
    """Remove the radius-four graph's constant-medium directional bias.

    The variable-medium graph travel is corrected by subtracting the travel on
    the identical graph at the source-node speed and adding the exact Euclidean
    constant-speed travel from the fractional source coordinate. This is exact
    for uniform media while retaining the graph estimate of heterogeneous
    slowness perturbations.
    """

    velocity = np.asarray(velocity_mps, dtype=np.float32)
    if velocity.ndim != 2 or not np.isfinite(velocity).all() or np.any(velocity <= 0.0):
        raise ValueError("velocity must be a finite positive 2-D field")
    coordinates = tuple((float(z), float(x)) for z, x in source_coordinates_m)
    if not coordinates:
        raise ValueError("at least one source coordinate is required")
    height, width = velocity.shape
    indices = []
    for source_z, source_x in coordinates:
        z_index = int(round((source_z - float(z0_m)) / float(dz_m)))
        x_index = int(round((source_x - float(x0_m)) / float(dx_m)))
        if not 0 <= z_index < height or not 0 <= x_index < width:
            raise ValueError("source coordinate lies outside the grid")
        indices.append((z_index, x_index))
    raw = grid_eikonal_travel_time(
        velocity, source_indices=indices, dx_m=dx_m, dz_m=dz_m
    )
    z_axis = float(z0_m) + np.arange(height, dtype=np.float32) * float(dz_m)
    x_axis = float(x0_m) + np.arange(width, dtype=np.float32) * float(dx_m)
    zz, xx = np.meshgrid(z_axis, x_axis, indexing="ij")
    corrected = []
    for source_position, ((source_z, source_x), index) in enumerate(
        zip(coordinates, indices, strict=True)
    ):
        reference_speed = float(velocity[index])
        reference = grid_eikonal_travel_time(
            np.full_like(velocity, reference_speed),
            source_indices=[index],
            dx_m=dx_m,
            dz_m=dz_m,
        )[0]
        exact = np.sqrt((xx - source_x) ** 2 + (zz - source_z) ** 2) / reference_speed
        corrected.append(np.maximum(raw[source_position] - reference + exact, 0.0))
    return np.asarray(corrected, dtype=np.float32)


class EikonalTravelCache:
    """Lazy sample-ID view over the derived source travel-time HDF5."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_h5: str | Path,
        allow_source_path_mismatch: bool = False,
    ) -> None:
        self.path = str(Path(path).expanduser().resolve())
        self.source_h5 = str(Path(source_h5).expanduser().resolve())
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r", swmr=True) as handle:
            cache_source_h5 = str(Path(str(handle.attrs["source_h5"])).resolve())
            if cache_source_h5 != self.source_h5 and not bool(
                allow_source_path_mismatch
            ):
                raise ValueError("Eikonal cache source HDF5 does not match the wavefield HDF5")
            self.cache_source_h5 = cache_source_h5
            self.source_path_mismatch_allowed = bool(allow_source_path_mismatch)
            sample_ids = tuple(value.decode() for value in handle["sample_id"][:])
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError("Eikonal cache sample IDs must be unique")
            if handle["travel_time_s"].shape[0] != len(sample_ids):
                raise ValueError("Eikonal cache metadata and travel rows disagree")
            self.grid_shape = tuple(int(value) for value in handle["travel_time_s"].shape[1:])
        self.row_by_sample = {sample_id: row for row, sample_id in enumerate(sample_ids)}

    def _file(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.path, "r", swmr=True)
        return self._h5

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_h5"] = None
        return state

    def read(self, sample_ids: Sequence[str]) -> torch.Tensor:
        samples = tuple(str(value) for value in sample_ids)
        missing = tuple(sample for sample in samples if sample not in self.row_by_sample)
        if missing:
            raise KeyError(f"Eikonal cache is missing sample IDs: {missing[:3]}")
        # h5py requires increasing fancy indices. Read individually to preserve
        # the grouped batch order and keep each access aligned to one chunk.
        handle = self._file()
        rows = [
            np.asarray(handle["travel_time_s"][self.row_by_sample[sample]], dtype=np.float32)
            for sample in samples
        ]
        return torch.from_numpy(np.stack(rows, axis=0))


__all__ = [
    "EikonalTravelCache",
    "debiased_grid_eikonal_travel_time",
    "grid_eikonal_travel_time",
]
