#!/usr/bin/env python3
"""Build a metadata-bound VDS joining train P_bg shards to source inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


HEAVY_SOURCE_DATASETS = ("velocity_mps", "source_map", "source_wavelet")
COORDINATE_DATASETS = ("time_s", "x_m", "z_m")
METADATA_DATASETS = (
    "cfl_2d",
    "completed_mask",
    "crop_x0_m",
    "crop_z0_m",
    "dt_used_s",
    "group_id",
    "lwc_qmax",
    "medium_type",
    "qc_final_energy_ratio",
    "qc_max_abs",
    "qc_status",
    "sample_id",
    "sample_sha256",
    "seed",
    "source_amplitude",
    "source_f0_hz",
    "source_t0_s",
    "source_x_m",
    "source_z_m",
    "split",
    "split_id",
    "vmax_mps",
    "vmin_mps",
)


def _text(values) -> tuple[str, ...]:
    return tuple(
        value.decode("utf8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def build_vds(
    *,
    source_h5: str | Path,
    background_shards: tuple[str | Path, ...],
    output: str | Path,
    travel_time_h5: str | Path | None = None,
) -> dict[str, object]:
    source_path = Path(source_h5).expanduser().resolve()
    shard_paths = tuple(Path(path).expanduser().resolve() for path in background_shards)
    output_path = Path(output).expanduser().resolve()
    travel_path = (
        None
        if travel_time_h5 is None
        else Path(travel_time_h5).expanduser().resolve()
    )
    if not shard_paths:
        raise ValueError("at least one background shard is required")
    if output_path.exists():
        raise FileExistsError(output_path)

    entries: list[tuple[str, int, Path, int]] = []
    shard_shapes: dict[Path, tuple[int, ...]] = {}
    source_manifest_hashes: set[str] = set()
    for shard_path in shard_paths:
        with h5py.File(shard_path, "r", swmr=True) as shard:
            if str(shard.attrs.get("status", "")) != "complete":
                raise ValueError(f"incomplete background shard: {shard_path}")
            if str(shard.attrs.get("background_kind", "")) != "gaussian_smoothed_velocity_lwc84_solve":
                raise ValueError(f"unexpected background kind: {shard_path}")
            sample_ids = _text(shard["sample_id"][:])
            source_indices = np.asarray(shard["source_index"][:], dtype=np.int64)
            if len(sample_ids) != len(source_indices):
                raise ValueError("background sample/source index lengths disagree")
            shard_shapes[shard_path] = tuple(int(value) for value in shard["wavefield"].shape)
            source_manifest_hashes.add(str(shard.attrs.get("source_manifest_sha256", "")))
            entries.extend(
                (sample_id, int(source_index), shard_path, row)
                for row, (sample_id, source_index) in enumerate(
                    zip(sample_ids, source_indices, strict=True)
                )
            )
    entries.sort(key=lambda item: item[0])
    sample_ids = tuple(item[0] for item in entries)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("background shard sample IDs overlap")
    if len(source_manifest_hashes) != 1:
        raise ValueError("background shard source manifests disagree")

    with h5py.File(source_path, "r", swmr=True) as source:
        if str(source.attrs.get("schema_version", "")) != "acoustic_lwc84_401_to_201_v1":
            raise ValueError("source HDF5 schema is not LWC84 401-to-201")
        source_ids = _text(source["sample_id"][:])
        for sample_id, source_index, _, _ in entries:
            if source_index < 0 or source_index >= len(source_ids):
                raise IndexError(f"source index outside source HDF5: {source_index}")
            if source_ids[source_index] != sample_id:
                raise ValueError(
                    f"background/source sample mismatch: {sample_id} != {source_ids[source_index]}"
                )
            if str(source["split"][source_index].decode("utf8")) != "train":
                raise ValueError(f"background sample is not train: {sample_id}")
            if str(source["medium_type"][source_index].decode("utf8")) != "marmousi":
                raise ValueError(f"background sample is not Marmousi: {sample_id}")

        travel_rows: dict[str, int] = {}
        travel_shape: tuple[int, ...] | None = None
        travel_content_sha256 = None
        if travel_path is not None:
            with h5py.File(travel_path, "r", swmr=True) as travel:
                travel_ids = _text(travel["sample_id"][:])
                travel_rows = {sample_id: row for row, sample_id in enumerate(travel_ids)}
                missing = sorted(set(sample_ids) - set(travel_rows))
                if missing:
                    raise ValueError(f"travel cache misses P_bg samples: {missing[:3]}")
                travel_shape = tuple(int(value) for value in travel["travel_time_s"].shape)
                travel_content_sha256 = str(travel.attrs.get("content_sha256", ""))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
        try:
            with h5py.File(partial, "w", libver="latest") as target:
                wave_shape = (len(entries),) + shard_shapes[shard_paths[0]][1:]
                wave_layout = h5py.VirtualLayout(shape=wave_shape, dtype=np.float32)
                shard_sources = {
                    path: h5py.VirtualSource(str(path), "wavefield", shape=shard_shapes[path])
                    for path in shard_paths
                }
                for output_index, (_, _, shard_path, shard_row) in enumerate(entries):
                    wave_layout[output_index] = shard_sources[shard_path][shard_row]
                target.create_virtual_dataset("wavefield", wave_layout, fillvalue=np.nan)

                source_indices = [item[1] for item in entries]
                for name in HEAVY_SOURCE_DATASETS:
                    source_shape = tuple(int(value) for value in source[name].shape)
                    layout = h5py.VirtualLayout(
                        shape=(len(entries),) + source_shape[1:], dtype=source[name].dtype
                    )
                    virtual = h5py.VirtualSource(str(source_path), name, shape=source_shape)
                    for output_index, source_index in enumerate(source_indices):
                        layout[output_index] = virtual[source_index]
                    target.create_virtual_dataset(name, layout)

                if travel_path is not None and travel_shape is not None:
                    layout = h5py.VirtualLayout(
                        shape=(len(entries),) + travel_shape[1:], dtype=np.float32
                    )
                    virtual = h5py.VirtualSource(
                        str(travel_path), "travel_time_s", shape=travel_shape
                    )
                    for output_index, sample_id in enumerate(sample_ids):
                        layout[output_index] = virtual[travel_rows[sample_id]]
                    target.create_virtual_dataset("travel_time_s", layout)

                for name in COORDINATE_DATASETS:
                    target.create_dataset(name, data=np.asarray(source[name][:]))
                for name in METADATA_DATASETS:
                    target.create_dataset(name, data=np.asarray(source[name][source_indices]))
                for key, value in source.attrs.items():
                    if key != "vds_source_shards":
                        target.attrs[key] = value
                target.attrs["split"] = "train"
                target.attrs["included_splits"] = json.dumps(["train"])
                target.attrs["vds_sample_count"] = len(entries)
                target.attrs["target_kind"] = "gaussian_smoothed_velocity_lwc84_solve"
                target.attrs["background_sigma_saved_cells"] = 2.0
                target.attrs["background_source_manifest_sha256"] = next(
                    iter(source_manifest_hashes)
                )
                target.attrs["background_shards"] = json.dumps(
                    [str(path) for path in shard_paths]
                )
                if travel_path is not None:
                    target.attrs["travel_time_h5"] = str(travel_path)
                    target.attrs["travel_time_content_sha256"] = str(
                        travel_content_sha256
                    )
                target.flush()
            os.replace(partial, output_path)
        finally:
            partial.unlink(missing_ok=True)

    return {
        "schema": "pbg_factorized_fno_vds_v1",
        "output": str(output_path),
        "source_h5": str(source_path),
        "background_shards": [str(path) for path in shard_paths],
        "sample_count": len(entries),
        "first_sample_id": sample_ids[0],
        "last_sample_id": sample_ids[-1],
        "target_kind": "gaussian_smoothed_velocity_lwc84_solve",
        "travel_time_h5": None if travel_path is None else str(travel_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--background-shard", action="append", required=True)
    parser.add_argument("--travel-time-h5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_vds(
        source_h5=args.source_h5,
        background_shards=tuple(args.background_shard),
        output=args.output,
        travel_time_h5=args.travel_time_h5,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
