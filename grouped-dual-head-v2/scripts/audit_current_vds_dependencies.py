#!/usr/bin/env python
"""Metadata-only integrity audit for the current acoustic HDF5 VDS.

The audit never indexes velocity or wavefield arrays. It resolves every virtual
source named by the VDS, checks that the backing file exists and is nonempty,
and verifies the small top-level schema needed by training and evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import h5py


REQUIRED_DATASETS = {
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
    "time_s",
    "velocity_mps",
    "wavefield",
    "x_m",
    "z_m",
}
VIRTUAL_DATASETS = ("velocity_mps", "wavefield")


def _resolve_source(vds_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate.resolve() if candidate.is_absolute() else (vds_path.parent / candidate).resolve()


def audit_vds(path: Path) -> dict[str, object]:
    requested = Path(path)
    if not requested.is_file():
        raise FileNotFoundError(requested)
    resolved_vds = requested.resolve()
    datasets: dict[str, object] = {}
    with h5py.File(resolved_vds, "r", swmr=True) as handle:
        missing_schema = sorted(REQUIRED_DATASETS - set(handle))
        if missing_schema:
            raise ValueError(f"current VDS is missing required datasets: {missing_schema}")
        record_count = int(handle["sample_id"].shape[0])
        if handle["velocity_mps"].shape[0] != record_count or handle["wavefield"].shape[0] != record_count:
            raise ValueError("VDS record axes disagree")
        for name in VIRTUAL_DATASETS:
            dataset = handle[name]
            if not dataset.is_virtual:
                raise ValueError(f"{name} is expected to be a virtual dataset")
            sources: dict[str, dict[str, object]] = {}
            for mapping in dataset.virtual_sources():
                raw = mapping.file_name.decode() if isinstance(mapping.file_name, bytes) else str(mapping.file_name)
                source = _resolve_source(resolved_vds, raw)
                source_name = mapping.dset_name.decode() if isinstance(mapping.dset_name, bytes) else str(mapping.dset_name)
                entry = sources.setdefault(
                    str(source),
                    {
                        "path": str(source),
                        "exists": source.is_file(),
                        "size_bytes": source.stat().st_size if source.is_file() else None,
                        "source_datasets": [],
                    },
                )
                if source_name not in entry["source_datasets"]:
                    entry["source_datasets"].append(source_name)
            missing = [entry for entry in sources.values() if not entry["exists"]]
            empty = [entry for entry in sources.values() if entry["exists"] and int(entry["size_bytes"]) <= 0]
            datasets[name] = {
                "shape": list(dataset.shape),
                "mapping_count": len(dataset.virtual_sources()),
                "unique_source_count": len(sources),
                "missing_source_count": len(missing),
                "empty_source_count": len(empty),
                "missing_sources": missing,
                "empty_sources": empty,
            }
        failures = {
            name: int(report["missing_source_count"]) + int(report["empty_source_count"])
            for name, report in datasets.items()
        }
        payload = {
            "schema": "current_acoustic_vds_dependency_audit_v1",
            "status": "pass" if not any(failures.values()) else "fail",
            "audit_mode": "metadata_only_no_velocity_or_wavefield_array_reads",
            "requested_vds": str(requested),
            "resolved_vds": str(resolved_vds),
            "vds_size_bytes": resolved_vds.stat().st_size,
            "record_count": record_count,
            "time_count": int(handle["time_s"].shape[0]),
            "required_dataset_count": len(REQUIRED_DATASETS),
            "missing_required_datasets": missing_schema,
            "binding_attributes": {
                name: (
                    handle.attrs[name].decode()
                    if isinstance(handle.attrs[name], bytes)
                    else str(handle.attrs[name])
                )
                for name in ("config_sha256", "git_commit", "manifest_sha256", "marmousi_sha256")
                if name in handle.attrs
            },
            "virtual_datasets": datasets,
        }
    return payload


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf8") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit_vds(args.vds)
    if args.output is not None:
        _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUIRED_DATASETS", "audit_vds"]
