#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fno_acoustic.data_generation.hdf5_lwc84 import VDS_DATASETS


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a zero-copy dataset that replaces only train/Marmousi records "
            "while reusing every other record from the original dataset."
        )
    )
    parser.add_argument("--old-dataset", required=True)
    parser.add_argument("--old-manifest", required=True)
    parser.add_argument("--new-marmousi-dataset", required=True)
    parser.add_argument("--new-manifest", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-replacements", type=int, default=700)
    parser.add_argument(
        "--allow-mixed-internal-dt",
        action="store_true",
        help=(
            "Allow old and replacement records to use different internal solver time steps "
            "when their saved time/grid coordinates are exactly identical."
        ),
    )
    return parser.parse_args()


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_binding(value: dict[str, Any]) -> tuple[str, str]:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dataset_ids(h5: h5py.File) -> list[str]:
    ids = [_decode(value) for value in h5["sample_id"][:]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample IDs in dataset {h5.filename}")
    return ids


def _same_float(actual: float, expected: Any) -> bool:
    if expected is None:
        return bool(np.isnan(actual))
    return bool(np.isclose(actual, float(expected), rtol=0.0, atol=1.0e-10))


def _validate_row_binding(
    h5: h5py.File, index: int, row: dict[str, Any], *, expected_split: str
) -> None:
    if _decode(h5["sample_id"][index]) != str(row["sample_id"]):
        raise ValueError("sample_id differs from the selected manifest row")
    if _decode(h5["medium_type"][index]) != str(row["medium_type"]):
        raise ValueError(f"medium_type mismatch for {row['sample_id']}")
    if _decode(h5["group_id"][index]) != str(row["group_id"]):
        raise ValueError(f"group_id mismatch for {row['sample_id']}")
    if int(h5["seed"][index]) != int(row["seed"]):
        raise ValueError(f"seed mismatch for {row['sample_id']}")
    for name in (
        "source_x_m",
        "source_z_m",
        "source_f0_hz",
        "source_t0_s",
        "source_amplitude",
        "crop_x0_m",
        "crop_z0_m",
    ):
        if not _same_float(float(h5[name][index]), row[name]):
            raise ValueError(f"{name} mismatch for {row['sample_id']}")
    if "split" in h5:
        actual_split = _decode(h5["split"][index])
    else:
        actual_split = _decode(h5.attrs.get("split", ""))
    if actual_split != expected_split or actual_split != str(row["split"]):
        raise ValueError(f"split mismatch for {row['sample_id']}")


def _mapping_runs(mapping: list[tuple[str, int]]) -> list[tuple[int, int, str, int]]:
    runs: list[tuple[int, int, str, int]] = []
    start = 0
    while start < len(mapping):
        source_name, source_index = mapping[start]
        stop = start + 1
        while (
            stop < len(mapping)
            and mapping[stop][0] == source_name
            and mapping[stop][1] == source_index + (stop - start)
        ):
            stop += 1
        runs.append((start, stop, source_name, source_index))
        start = stop
    return runs


def main() -> int:
    args = _arguments()
    old_dataset = Path(args.old_dataset).resolve()
    new_dataset = Path(args.new_marmousi_dataset).resolve()
    old_manifest_path = Path(args.old_manifest).resolve()
    new_manifest_path = Path(args.new_manifest).resolve()
    output_dataset = Path(args.output_dataset).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    report_path = Path(args.report).resolve()
    for output in (output_dataset, output_manifest, report_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)

    old_rows = _read_manifest(old_manifest_path)
    new_rows = _read_manifest(new_manifest_path)
    old_manifest_sha256 = _sha256(old_manifest_path)
    new_manifest_sha256 = _sha256(new_manifest_path)
    old_by_id = {str(row["sample_id"]): row for row in old_rows}
    new_by_id = {str(row["sample_id"]): row for row in new_rows}

    with h5py.File(old_dataset, "r") as old_h5, h5py.File(new_dataset, "r") as new_h5:
        if _decode(old_h5.attrs.get("manifest_sha256", "")) != old_manifest_sha256:
            raise ValueError("old dataset manifest_sha256 does not bind the old manifest file")
        if _decode(new_h5.attrs.get("manifest_sha256", "")) != new_manifest_sha256:
            raise ValueError("new dataset manifest_sha256 does not bind the new manifest file")
        new_ids = _dataset_ids(new_h5)
        if len(new_ids) != int(args.expected_replacements):
            raise ValueError(
                f"expected {args.expected_replacements} replacement records, got {len(new_ids)}"
            )
        for coordinate in ("time_s", "x_m", "z_m"):
            if not np.array_equal(np.asarray(old_h5[coordinate]), np.asarray(new_h5[coordinate])):
                raise ValueError(f"coordinate mismatch between old and new datasets: {coordinate}")
        old_time_protocol = {
            "dt_requested_s": float(old_h5.attrs["dt_requested_s"]),
            "dt_used_s": float(old_h5.attrs["dt_used_s"]),
            "snapshot_stride": int(old_h5.attrs["snapshot_stride"]),
            "dt_output_s": float(old_h5.attrs["dt_output_s"]),
            "t_end_s": float(old_h5.attrs["t_end_s"]),
        }
        new_time_protocol = {
            "dt_requested_s": float(new_h5.attrs["dt_requested_s"]),
            "dt_used_s": float(new_h5.attrs["dt_used_s"]),
            "snapshot_stride": int(new_h5.attrs["snapshot_stride"]),
            "dt_output_s": float(new_h5.attrs["dt_output_s"]),
            "t_end_s": float(new_h5.attrs["t_end_s"]),
        }
        mixed_internal_dt = not np.isclose(
            old_time_protocol["dt_used_s"],
            new_time_protocol["dt_used_s"],
            rtol=0.0,
            atol=1.0e-15,
        )
        if mixed_internal_dt and not args.allow_mixed_internal_dt:
            raise ValueError(
                "old and replacement datasets use different internal dt; "
                "pass --allow-mixed-internal-dt only when this mixed protocol is intentional"
            )
        for key in ("dt_output_s", "t_end_s"):
            if not np.isclose(
                old_time_protocol[key], new_time_protocol[key], rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"saved time protocol mismatch between old and new datasets: {key}")
        for key in (
            "axis_order",
            "lwc_version",
            "solver_grid_shape",
            "saved_grid_shape",
            "solver_dx_m",
            "solver_dz_m",
            "saved_dx_m",
            "saved_dz_m",
            "free_surface",
            "cpml",
            "source_formula",
            "restriction",
        ):
            if _decode(old_h5.attrs.get(key, "")) != _decode(new_h5.attrs.get(key, "")):
                raise ValueError(f"non-time forward protocol mismatch: {key}")
        for sample_id in new_ids:
            if sample_id not in old_by_id or sample_id not in new_by_id:
                raise ValueError(f"replacement sample is absent from a manifest: {sample_id}")
            row = new_by_id[sample_id]
            if row["split"] != "train" or row["medium_type"] != "marmousi":
                raise ValueError(f"replacement is not train/Marmousi: {sample_id}")

        replacement_ids = set(new_ids)
        old_ids = [
            str(row["sample_id"])
            for split in ("train", "validation", "test_id", "ood_canonical")
            for row in old_rows
            if str(row["split"]) == split
        ]
        actual_old_ids = [_decode(value) for value in old_h5["sample_id"][:]]
        if len(actual_old_ids) != len(old_ids):
            raise ValueError("old dataset length differs from the split-ordered old manifest")
        invalid_old_bindings = [
            (index, actual, expected)
            for index, (actual, expected) in enumerate(zip(actual_old_ids, old_ids, strict=True))
            if actual != expected and not (not actual and expected in replacement_ids)
        ]
        if invalid_old_bindings:
            raise ValueError(
                "old dataset binding differs outside an authorized replacement slot: "
                f"{invalid_old_bindings[:3]}"
            )
        missing_old_slots = {
            expected
            for actual, expected in zip(actual_old_ids, old_ids, strict=True)
            if not actual
        }
        if not missing_old_slots <= replacement_ids:
            raise ValueError("old dataset contains a missing slot that is not supplied by the new dataset")

        new_index = {sample_id: index for index, sample_id in enumerate(new_ids)}
        hybrid_rows = [
            new_by_id[sample_id] if sample_id in replacement_ids else old_by_id[sample_id]
            for sample_id in old_ids
        ]
        mapping = [
            ("new", new_index[sample_id]) if sample_id in replacement_ids else ("old", index)
            for index, sample_id in enumerate(old_ids)
        ]
        runs = _mapping_runs(mapping)

        for sample_id, source_index in new_index.items():
            _validate_row_binding(
                new_h5,
                source_index,
                new_by_id[sample_id],
                expected_split="train",
            )

        manifest_bytes = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in hybrid_rows
        ).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        config_components = {
            "old_unreplaced_records": _decode(old_h5.attrs["config_sha256"]),
            "new_train_marmousi_records": _decode(new_h5.attrs["config_sha256"]),
        }
        config_components_json, hybrid_config_hash = _json_binding(config_components)
        marmousi_components = {
            "old_unreplaced_records": _decode(old_h5.attrs["marmousi_sha256"]),
            "new_train_marmousi_records": _decode(new_h5.attrs["marmousi_sha256"]),
        }
        marmousi_components_json, hybrid_marmousi_hash = _json_binding(marmousi_components)
        manifest_tmp = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
        manifest_tmp.write_bytes(manifest_bytes)

        dataset_tmp = output_dataset.with_suffix(output_dataset.suffix + ".tmp")
        with h5py.File(dataset_tmp, "w", libver="latest") as output_h5:
            for key, value in old_h5.attrs.items():
                output_h5.attrs[key] = value
            output_h5.attrs["config_sha256"] = hybrid_config_hash
            output_h5.attrs["config_sha256_semantics"] = "sha256_of_hybrid_component_config_mapping"
            output_h5.attrs["component_config_sha256"] = config_components_json
            output_h5.attrs["manifest_sha256"] = manifest_hash
            output_h5.attrs["marmousi_sha256"] = hybrid_marmousi_hash
            output_h5.attrs["marmousi_sha256_semantics"] = "sha256_of_hybrid_component_marmousi_mapping"
            output_h5.attrs["component_marmousi_sha256"] = marmousi_components_json
            output_h5.attrs["marmousi_replacement_scope"] = "train_only"
            output_h5.attrs["internal_dt_mixed"] = bool(mixed_internal_dt)
            output_h5.attrs["internal_dt_semantics"] = (
                "per_sample_dataset_dt_used_s; root_scalar_not_applicable_when_mixed"
            )
            output_h5.attrs["component_time_protocol"] = json.dumps(
                {
                    "old_unreplaced_records": old_time_protocol,
                    "new_train_marmousi_records": new_time_protocol,
                },
                sort_keys=True,
            )
            output_h5.attrs["component_forward_protocol_id"] = json.dumps(
                {
                    "old_unreplaced_records": _decode(
                        old_h5.attrs.get("forward_protocol_id", "legacy_unlabeled")
                    ),
                    "new_train_marmousi_records": _decode(
                        new_h5.attrs.get("forward_protocol_id", "legacy_unlabeled")
                    ),
                },
                sort_keys=True,
            )
            if mixed_internal_dt:
                output_h5.attrs["dt_requested_s"] = np.nan
                output_h5.attrs["dt_used_s"] = np.nan
                output_h5.attrs["snapshot_stride"] = -1
            output_h5.attrs["vds_source_datasets"] = json.dumps(
                {"old": str(old_dataset), "new_train_marmousi": str(new_dataset)},
                sort_keys=True,
            )
            output_h5.attrs["vds_sample_count"] = len(hybrid_rows)
            output_h5.attrs["replacement_sample_count"] = len(replacement_ids)
            for coordinate in ("time_s", "x_m", "z_m"):
                output_h5.create_dataset(coordinate, data=np.asarray(old_h5[coordinate]))
            output_h5.create_dataset(
                "split",
                data=np.asarray(
                    [str(row["split"]) for row in hybrid_rows],
                    dtype=h5py.string_dtype("utf-8", 32),
                ),
            )
            output_h5.create_dataset(
                "split_id",
                data=np.asarray([int(row["split_id"]) for row in hybrid_rows], dtype=np.uint8),
            )
            sources = {"old": old_h5, "new": new_h5}
            source_paths = {"old": old_dataset, "new": new_dataset}
            for name in VDS_DATASETS:
                reference = old_h5[name]
                for source_name, source in sources.items():
                    if source[name].shape[1:] != reference.shape[1:] or source[name].dtype != reference.dtype:
                        raise ValueError(f"incompatible {name} in {source_name} dataset")
                layout = h5py.VirtualLayout(
                    shape=(len(hybrid_rows), *reference.shape[1:]), dtype=reference.dtype
                )
                for start, stop, source_name, source_start in runs:
                    source = sources[source_name]
                    virtual = h5py.VirtualSource(
                        str(source_paths[source_name]), name, shape=source[name].shape
                    )
                    layout[start:stop] = virtual[source_start : source_start + (stop - start)]
                output_h5.create_virtual_dataset(name, layout)

        with h5py.File(dataset_tmp, "r") as output_h5:
            output_ids = [_decode(value) for value in output_h5["sample_id"][:]]
            if output_ids != old_ids:
                raise ValueError("hybrid VDS sample ordering failed publication readback")
            if len(set(output_ids)) != len(output_ids):
                raise ValueError("hybrid VDS publication readback contains duplicate sample IDs")
            completed = np.asarray(output_h5["completed_mask"][:], dtype=bool)
            if not bool(completed.all()):
                raise ValueError(
                    "hybrid VDS publication readback is incomplete: "
                    f"{int(completed.sum())}/{len(completed)}"
                )
            output_splits = [_decode(value) for value in output_h5["split"][:]]
            expected_splits = [str(row["split"]) for row in hybrid_rows]
            if output_splits != expected_splits:
                raise ValueError("hybrid VDS split ordering failed publication readback")
            if any(not _decode(value) for value in output_h5["sample_sha256"][:]):
                raise ValueError("hybrid VDS publication readback contains an empty sample hash")
            if any(_decode(value) != "passed" for value in output_h5["qc_status"][:]):
                raise ValueError("hybrid VDS publication readback contains a failed QC record")

        os.replace(manifest_tmp, output_manifest)
        os.replace(dataset_tmp, output_dataset)

    report = {
        "status": "COMPLETE",
        "old_dataset": str(old_dataset),
        "old_dataset_sha256": _sha256(old_dataset),
        "old_manifest_sha256": old_manifest_sha256,
        "new_marmousi_dataset": str(new_dataset),
        "new_marmousi_dataset_sha256": _sha256(new_dataset),
        "new_manifest_sha256": new_manifest_sha256,
        "output_dataset": str(output_dataset),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": _sha256(output_manifest),
        "hybrid_config_sha256": hybrid_config_hash,
        "component_config_sha256": config_components,
        "hybrid_marmousi_sha256": hybrid_marmousi_hash,
        "component_marmousi_sha256": marmousi_components,
        "sample_count": len(hybrid_rows),
        "replacement_sample_count": len(replacement_ids),
        "virtual_mapping_run_count": len(runs),
        "saved_coordinate_protocol_identical": True,
        "internal_dt_mixed": bool(mixed_internal_dt),
        "component_time_protocol": {
            "old_unreplaced_records": old_time_protocol,
            "new_train_marmousi_records": new_time_protocol,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
