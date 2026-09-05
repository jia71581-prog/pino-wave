from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from fno_acoustic.ais_dataset_binding import (
    DatasetContentBinding,
    dataset_stat,
    global_digest,
    load_dataset_content_binding,
    sample_digest,
    verified_ids_sha256,
)


@dataclass(frozen=True)
class Full160GridContract:
    height: int
    width: int
    time_s: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor

    def validate(self) -> None:
        if self.time_s.ndim != 1:
            raise ValueError("time coordinates must be one-dimensional")
        if self.x_m.ndim != 1 or self.z_m.ndim != 1:
            raise ValueError("physical coordinates must be one-dimensional")
        if self.time_s.numel() != 160:
            raise ValueError("AIS-MQFNO requires exactly 160 saved times")
        if not bool(torch.all(torch.isfinite(self.time_s))):
            raise ValueError("time coordinates must be finite")
        if not bool(torch.all(torch.diff(self.time_s) > 0)):
            raise ValueError("time coordinates must be strictly increasing")
        if self.x_m.numel() != self.height or self.z_m.numel() != self.width:
            raise ValueError("physical coordinate lengths do not match the grid")
        if not bool(torch.all(torch.isfinite(self.x_m))) or not bool(
            torch.all(torch.isfinite(self.z_m))
        ):
            raise ValueError("physical coordinates must be finite")
        if not bool(torch.all(torch.diff(self.x_m) > 0)) or not bool(
            torch.all(torch.diff(self.z_m) > 0)
        ):
            raise ValueError("physical coordinates must be strictly increasing")


@dataclass(frozen=True)
class QueryScene:
    sample_id: int
    target_cpu: torch.Tensor
    velocity_cpu: torch.Tensor
    source_cpu: torch.Tensor
    time_s: torch.Tensor
    x_m: torch.Tensor
    z_m: torch.Tensor
    metadata: dict[str, object]


@dataclass(frozen=True)
class QuerySites:
    sample_id: int
    site_indices: torch.Tensor
    target_cpu: torch.Tensor
    physical_xz: torch.Tensor
    time_s: torch.Tensor
    velocity_cpu: torch.Tensor
    source_cpu: torch.Tensor
    metadata: dict[str, object]


@dataclass(frozen=True)
class QueryBatch:
    sample_ids: torch.Tensor
    site_indices: torch.Tensor
    target_cpu: torch.Tensor
    physical_xz: torch.Tensor
    time_s: torch.Tensor
    velocity_cpu: torch.Tensor
    source_cpu: torch.Tensor
    metadata: tuple[dict[str, object], ...]


def collate_query_sites(items: Sequence[QuerySites]) -> QueryBatch:
    if not items or len({tuple(item.target_cpu.shape) for item in items}) != 1:
        raise ValueError("query batches require a nonempty equal-Q collection")
    return QueryBatch(
        sample_ids=torch.tensor([item.sample_id for item in items]),
        site_indices=torch.stack([item.site_indices for item in items]),
        target_cpu=torch.stack([item.target_cpu for item in items]),
        physical_xz=torch.stack([item.physical_xz for item in items]),
        time_s=torch.stack([item.time_s for item in items]),
        velocity_cpu=torch.stack([item.velocity_cpu for item in items]),
        source_cpu=torch.stack([item.source_cpu for item in items]),
        metadata=tuple(item.metadata for item in items),
    )


def resize_query_scene(scene: QueryScene, height: int, width: int) -> QueryScene:
    if (height, width) == tuple(scene.target_cpu.shape[:2]):
        return scene
    target = F.interpolate(
        scene.target_cpu.permute(2, 0, 1)[None],
        size=(height, width),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )[0].permute(1, 2, 0)
    velocity = F.interpolate(
        scene.velocity_cpu[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )[0, 0]
    source = F.interpolate(
        scene.source_cpu[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=True,
        antialias=True,
    )[0, 0]
    source = source / source.sum().clamp_min(1e-12) * scene.source_cpu.sum()
    x_m = torch.linspace(scene.x_m[0], scene.x_m[-1], height, dtype=torch.float64)
    z_m = torch.linspace(scene.z_m[0], scene.z_m[-1], width, dtype=torch.float64)
    return QueryScene(
        scene.sample_id,
        target.contiguous(),
        velocity,
        source,
        scene.time_s,
        x_m,
        z_m,
        dict(scene.metadata),
    )


class DenseCPUQueryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        dataset_content_manifest: str | Path | None = None,
    ):
        self.path = Path(path)
        self.dataset_binding: DatasetContentBinding | None = (
            None
            if dataset_content_manifest is None
            else load_dataset_content_binding(dataset_content_manifest, self.path)
        )
        self._verified_sample_ids: set[int] = set()

    def binding_summary(self) -> dict[str, object]:
        if self.dataset_binding is None:
            raise ValueError("unbound query store has no formal binding summary")
        ids = sorted(self._verified_sample_ids)
        return {
            "dataset_content_root": self.dataset_binding.dataset_content_root,
            "verified_sample_count": len(ids),
            "verified_sample_ids_sha256": verified_ids_sha256(ids),
            "verified_sample_scope": "process_unique",
        }

    @property
    def verified_sample_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._verified_sample_ids))

    def _assert_bound_stat(self, stat_result: os.stat_result) -> None:
        if self.dataset_binding is not None and dataset_stat(
            stat_result
        ) != self.dataset_binding.dataset_stat:
            raise ValueError("query dataset stat differs from formal binding")

    def read_scene(self, sample_id: int) -> QueryScene:
        flags = os.O_RDONLY
        if self.dataset_binding is not None:
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags)
        try:
            self._assert_bound_stat(os.fstat(descriptor))
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                with h5py.File(stream, "r") as h5:
                    target_raw = np.asarray(h5["tensor"][sample_id], dtype=np.float32)
                    velocity_raw = np.asarray(h5["nu"][sample_id], dtype=np.float32)
                    source_raw = np.asarray(
                        h5["source_mask"][sample_id], dtype=np.float32
                    )
                    time_raw = np.asarray(h5["t-coordinate"], dtype=np.float64)
                    x_raw = np.asarray(h5["x-coordinate"], dtype=np.float64)
                    z_raw = np.asarray(h5["y-coordinate"], dtype=np.float64)
                    model_type = h5["model_type"][sample_id]
                    if self.dataset_binding is not None:
                        actual_global, actual_schema = global_digest(h5)
                        if actual_schema != list(self.dataset_binding.required_datasets):
                            raise ValueError(
                                "query dataset schema differs from formal binding"
                            )
                        if actual_global != self.dataset_binding.global_sha256:
                            raise ValueError(
                                "query dataset global digest differs from formal binding"
                            )
                        actual = sample_digest(
                            sample_id,
                            target_raw,
                            velocity_raw,
                            source_raw,
                            model_type,
                        )
                        if actual != self.dataset_binding.sample_digests[sample_id]:
                            raise ValueError(
                                "query dataset sample leaf differs from formal binding"
                            )
            self._assert_bound_stat(os.fstat(descriptor))
            if self.dataset_binding is not None:
                self._verified_sample_ids.add(sample_id)
        except KeyError as error:
            raise ValueError(f"query HDF5 is missing required data: {error}") from error
        finally:
            os.close(descriptor)

        time_s = torch.from_numpy(time_raw)
        x_m = 1000.0 * torch.from_numpy(x_raw)
        z_m = 1000.0 * torch.from_numpy(z_raw)

        if target_raw.ndim != 3 or target_raw.shape[0] != 160:
            raise ValueError("target must have shape [H, W, 160] after axis conversion")
        target = torch.from_numpy(target_raw).permute(1, 2, 0).contiguous()
        expected_spatial_shape = tuple(target.shape[:2])
        if velocity_raw.shape != expected_spatial_shape or source_raw.shape != expected_spatial_shape:
            raise ValueError("velocity and source must have shape [H, W]")
        velocity = torch.from_numpy(velocity_raw)
        source = torch.from_numpy(source_raw)
        contract = Full160GridContract(target.shape[0], target.shape[1], time_s, x_m, z_m)
        contract.validate()
        return QueryScene(
            sample_id,
            target,
            velocity,
            source,
            time_s,
            x_m,
            z_m,
            {
                "model_type": (
                    model_type.decode() if isinstance(model_type, bytes) else str(model_type)
                )
            },
        )

    def read_sites(self, sample_id: int, site_indices: torch.Tensor) -> QuerySites:
        return self.gather_loaded_scene(self.read_scene(sample_id), site_indices)

    @staticmethod
    def gather_loaded_scene(scene: QueryScene, site_indices: torch.Tensor) -> QuerySites:
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        if (
            not isinstance(site_indices, torch.Tensor)
            or site_indices.ndim != 1
            or site_indices.dtype not in integer_dtypes
        ):
            raise ValueError("site indices must be a one-dimensional integer tensor")
        indices = site_indices.to(dtype=torch.long, device="cpu").clone()
        total = scene.target_cpu.shape[0] * scene.target_cpu.shape[1]
        if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= total:
            raise ValueError("site indices must be nonempty and inside the native grid")
        x_idx = torch.div(indices, scene.target_cpu.shape[1], rounding_mode="floor")
        z_idx = indices.remainder(scene.target_cpu.shape[1])
        coords = torch.stack((scene.x_m[x_idx], scene.z_m[z_idx]), dim=-1)
        traces = scene.target_cpu.reshape(total, 160).index_select(0, indices)
        return QuerySites(
            scene.sample_id,
            indices,
            traces,
            coords,
            scene.time_s,
            scene.velocity_cpu,
            scene.source_cpu,
            scene.metadata,
        )
