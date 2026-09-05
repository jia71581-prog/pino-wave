#!/usr/bin/env python
"""Validate the transferred VDS and build continuation-only derived metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from grouped_ufno_mionet_v3.data.index import (
    build_manifest,
    validate_expected_counts,
    write_manifest_atomic,
)
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer
from saved_time_phase_operator_v4.eikonal import EikonalTravelCache


EXPECTED_VDS_SHA256 = "c168473406a5bd49c02ead00be7b445cabdbc0368ce3c052ce2fcbf503d5f98c"
EXPECTED_NEW_SHARDS = 127
EXPECTED_VDS_SOURCES = 501
EXPECTED_RECORDS = {"train": 2240, "validation": 480}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_or_match_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf8")) != payload:
            raise ValueError(f"existing JSON does not match the registered payload: {path}")
        return
    atomic_json(path, payload)


def text(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str)


def verify_new_shards(dataset_dir: Path) -> dict[str, Any]:
    shard_paths = sorted(dataset_dir.glob("shards/*/*.h5"))
    if len(shard_paths) != EXPECTED_NEW_SHARDS:
        raise ValueError(
            f"new-shard count is {len(shard_paths)}, expected {EXPECTED_NEW_SHARDS}"
        )
    total_bytes = 0
    split_counts: dict[str, int] = {}
    for shard in shard_paths:
        sidecar = shard.with_name(f"{shard.name}.sha256")
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        tokens = sidecar.read_text(encoding="utf8").strip().split()
        if not tokens or len(tokens[0]) != 64:
            raise ValueError(f"malformed SHA256 sidecar: {sidecar}")
        actual = sha256_file(shard)
        if actual != tokens[0].lower():
            raise ValueError(f"SHA256 mismatch: {shard}")
        total_bytes += shard.stat().st_size
        split_counts[shard.parent.name] = split_counts.get(shard.parent.name, 0) + 1
    expected_splits = {"train": 88, "validation": 19, "test_id": 19, "ood_canonical": 1}
    if split_counts != expected_splits:
        raise ValueError(f"new-shard split census changed: {split_counts}")
    return {
        "count": len(shard_paths),
        "bytes": total_bytes,
        "split_counts": split_counts,
    }


def verify_vds_sources(source_h5: Path) -> dict[str, Any]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        wavefield = handle["wavefield"]
        if not wavefield.is_virtual:
            raise ValueError("registered wavefield is not a virtual dataset")
        mappings = wavefield.virtual_sources()
        source_names = tuple(str(mapping.file_name) for mapping in mappings)
        unique_sources = tuple(sorted(set(source_names)))
        missing = tuple(name for name in unique_sources if not Path(name).is_file())
        if len(mappings) != EXPECTED_VDS_SOURCES or len(unique_sources) != EXPECTED_VDS_SOURCES:
            raise ValueError(
                f"VDS source census changed: mappings={len(mappings)}, unique={len(unique_sources)}"
            )
        if missing:
            raise FileNotFoundError(f"VDS source files missing: {missing[:3]}")
        if tuple(wavefield.shape) != (4003, 401, 201, 201):
            raise ValueError(f"unexpected wavefield shape: {wavefield.shape}")

        families = text(handle["medium_type"])
        splits = text(handle["split"])
        sample_ids = text(handle["sample_id"])
        sample_checks: dict[str, dict[str, Any]] = {}
        for family in ALLOWED_MEDIUM_TYPES:
            candidates = np.flatnonzero((families == family) & (splits == "train"))
            if not len(candidates):
                raise ValueError(f"no train sample for family {family}")
            index = int(candidates[len(candidates) // 2])
            velocity = np.asarray(handle["velocity_mps"][index], dtype=np.float32)
            frames = np.stack(
                [np.asarray(handle["wavefield"][index, frame], dtype=np.float32) for frame in (100, 200, 300)]
            )
            if not np.isfinite(velocity).all() or not np.isfinite(frames).all():
                raise ValueError(f"non-finite VDS sample for {family}")
            maximum = float(np.max(np.abs(frames)))
            if maximum <= 0.0:
                raise ValueError(f"zero VDS wavefield sample for {family}")
            sample_checks[family] = {
                "source_index": index,
                "sample_id": str(sample_ids[index]),
                "velocity_min_mps": float(velocity.min()),
                "velocity_max_mps": float(velocity.max()),
                "wavefield_max_abs": maximum,
            }
    return {
        "mapping_count": len(mappings),
        "unique_source_count": len(unique_sources),
        "sample_checks": sample_checks,
    }


def build_normalizer(
    old_path: Path, new_path: Path, *, manifest_digest: str
) -> dict[str, Any]:
    old = json.loads(old_path.read_text(encoding="utf8"))
    old_digest = str(old["train_manifest_sha256"])
    payload = dict(old)
    payload["train_manifest_sha256"] = str(manifest_digest)
    payload["rebound_from_train_manifest_sha256"] = old_digest
    payload["rebound_from_normalization_json"] = str(old_path.resolve())
    payload["checkpoint_compatibility"] = (
        "numeric transforms retained exactly from the parent checkpoint; only the "
        "active train-manifest binding changed"
    )
    PhysicalNormalizer.from_dict(payload, expected_manifest=manifest_digest)
    write_or_match_json(new_path, payload)
    return {
        "path": str(new_path.resolve()),
        "sha256": sha256_file(new_path),
        "old_manifest_digest": old_digest,
        "pressure_scale_pa": float(payload["pressure_scale_pa"]),
        "velocity_center_mps": float(payload["velocity_center_mps"]),
        "velocity_scale_mps": float(payload["velocity_scale_mps"]),
    }


def validate_cache_alignment(cache_path: Path, source_h5: Path) -> dict[str, Any]:
    source_resolved = str(source_h5.resolve())
    with h5py.File(source_h5, "r", swmr=True) as source, h5py.File(
        cache_path, "r", swmr=True
    ) as cache:
        cache_source = str(Path(str(cache.attrs["source_h5"])).resolve())
        if cache_source != source_resolved:
            raise ValueError(f"travel cache source mismatch: {cache_source}")
        indices = np.asarray(cache["source_index"][:], dtype=np.int64)
        if len(indices) != 2720 or len(set(indices.tolist())) != len(indices):
            raise ValueError("travel cache source indices are not a unique 2720-row census")
        source_family = text(source["medium_type"])
        source_split = text(source["split"])
        expected = np.flatnonzero(
            np.isin(source_family, ALLOWED_MEDIUM_TYPES)
            & np.isin(source_split, ("train", "validation"))
        )
        if set(indices.tolist()) != set(expected.tolist()):
            raise ValueError("travel cache does not cover the active train+validation census")
        for name in ("sample_id", "medium_type", "group_id", "split"):
            if not np.array_equal(text(cache[name]), text(source[name])[indices]):
                raise ValueError(f"travel cache {name} is not aligned to the active VDS")
        travel = cache["travel_time_s"]
        if travel.shape != (2720, 201, 201):
            raise ValueError(f"unexpected travel cache shape: {travel.shape}")
        family = text(cache["medium_type"])
        sample_ids = text(cache["sample_id"])
        probes = []
        for name in ALLOWED_MEDIUM_TYPES:
            row = int(np.flatnonzero(family == name)[0])
            values = np.asarray(travel[row], dtype=np.float32)
            if not np.isfinite(values).all() or float(values.max()) <= 0.0:
                raise ValueError(f"invalid travel-time probe for {name}")
            probes.append(str(sample_ids[row]))
        content_sha256 = str(cache.attrs["content_sha256"])
        rule = str(cache.attrs.get("travel_rule", ""))
    lazy = EikonalTravelCache(cache_path, source_h5=source_h5)
    loaded = lazy.read(probes).numpy()
    if loaded.shape != (3, 201, 201) or not np.isfinite(loaded).all():
        raise ValueError("lazy travel cache read failed")
    return {
        "path": str(cache_path.resolve()),
        "record_count": len(indices),
        "content_sha256": content_sha256,
        "travel_rule": rule,
        "probe_sample_ids": probes,
    }


def build_travel_cache(
    old_cache: Path,
    rebound_cache: Path,
    output_cache: Path,
    source_h5: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], int]:
    source_resolved = str(source_h5.resolve())
    group_mismatch_count = 0
    if not rebound_cache.exists():
        temporary = rebound_cache.with_name(f".{rebound_cache.name}.tmp-{os.getpid()}")
        try:
            shutil.copyfile(old_cache, temporary)
            with h5py.File(source_h5, "r", swmr=True) as source, h5py.File(
                temporary, "r+"
            ) as cache:
                indices = np.asarray(cache["source_index"][:], dtype=np.int64)
                source_ids = text(source["sample_id"])[indices]
                source_family = text(source["medium_type"])[indices]
                source_split = text(source["split"])[indices]
                cached_ids = text(cache["sample_id"])
                cached_family = text(cache["medium_type"])
                cached_split = text(cache["split"])
                if not np.array_equal(cached_ids, source_ids):
                    raise ValueError("old travel cache sample IDs no longer align")
                if not np.array_equal(cached_family, source_family):
                    raise ValueError("old travel cache families no longer align")
                if not np.array_equal(cached_split, source_split):
                    raise ValueError("old travel cache splits no longer align")
                old_group = text(cache["group_id"])
                new_group = text(source["group_id"])[indices]
                mismatch = old_group != new_group
                if np.any(mismatch & (source_family != "marmousi")):
                    raise ValueError("non-Marmousi travel group IDs changed")
                group_mismatch_count = int(np.sum(mismatch))
                cache["group_id"][:] = np.asarray(new_group, dtype=cache["group_id"].dtype)
                old_source = str(cache.attrs["source_h5"])
                cache.attrs["source_h5"] = source_resolved
                cache.attrs["rebound_from_source_h5"] = old_source
                cache.attrs["rebound_group_id_mismatch_count"] = group_mismatch_count
                cache.flush()
            os.replace(temporary, rebound_cache)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        with h5py.File(rebound_cache, "r", swmr=True) as cache:
            group_mismatch_count = int(cache.attrs["rebound_group_id_mismatch_count"])

    # The rebound cache preserves valid layered Eikonal rows and corrects the
    # source/group metadata.  The builder replaces every uniform/Marmousi row
    # with a fresh straight-ray computation from the repaired VDS.
    if not output_cache.exists():
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/build_hybrid_travel_cache.py"),
                "--eikonal-cache",
                str(rebound_cache),
                "--output",
                str(output_cache),
                "--workers",
                str(workers),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return validate_cache_alignment(output_cache, source_h5), group_mismatch_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-h5",
        default="/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5",
    )
    parser.add_argument(
        "--manifest-json",
        default="/home/jiayh/Data/data/processed/grouped_v3_manifest_marmousi1_4m_v2.json",
    )
    parser.add_argument(
        "--old-normalization-json",
        default="/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json",
    )
    parser.add_argument(
        "--normalization-json",
        default="/home/jiayh/Data/data/processed/grouped_v3_normalization_marmousi1_4m_v2_checkpoint_compatible.json",
    )
    parser.add_argument(
        "--old-travel-cache",
        default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5",
    )
    parser.add_argument(
        "--rebound-travel-cache",
        default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2_rebound.h5",
    )
    parser.add_argument(
        "--travel-cache",
        default="/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5",
    )
    parser.add_argument(
        "--report",
        default="/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/local_field_w128_hicap/temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r1/data_preparation.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    source_h5 = Path(args.source_h5)
    if not source_h5.is_file():
        raise FileNotFoundError(source_h5)
    vds_sha256 = sha256_file(source_h5)
    if vds_sha256 != EXPECTED_VDS_SHA256:
        raise ValueError(f"VDS SHA256 mismatch: {vds_sha256}")

    shard_report = verify_new_shards(source_h5.parent)
    vds_report = verify_vds_sources(source_h5)
    manifest = build_manifest(source_h5)
    validate_expected_counts(manifest, EXPECTED_RECORDS)
    manifest_path = Path(args.manifest_json)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf8")) != manifest.to_dict():
            raise ValueError(f"existing manifest does not match active VDS: {manifest_path}")
    else:
        write_manifest_atomic(manifest, manifest_path)

    normalization_report = build_normalizer(
        Path(args.old_normalization_json),
        Path(args.normalization_json),
        manifest_digest=manifest.digest,
    )
    travel_report, group_mismatches = build_travel_cache(
        Path(args.old_travel_cache),
        Path(args.rebound_travel_cache),
        Path(args.travel_cache),
        source_h5,
        workers=args.workers,
    )
    if group_mismatches <= 0:
        raise ValueError("expected repaired Marmousi group IDs were not observed")

    report = {
        "status": "complete",
        "schema": "marmousi1_4m_v2_continuation_preparation_v1",
        "source_h5": str(source_h5.resolve()),
        "vds_sha256": vds_sha256,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "digest": manifest.digest,
            "counts_after": manifest.counts_after,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_config_sha256": manifest.source_config_sha256,
        },
        "new_shards": shard_report,
        "vds": vds_report,
        "normalization": normalization_report,
        "travel_cache": travel_report,
        "rebound_group_id_mismatch_count": group_mismatches,
    }
    report_path = Path(args.report)
    write_or_match_json(report_path, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
