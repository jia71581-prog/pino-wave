#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gate a train/Marmousi-only LWC84 replacement with either an identical "
            "forward protocol or an explicitly labelled mixed internal-dt protocol."
        )
    )
    parser.add_argument("--old-dataset", required=True)
    parser.add_argument("--new-marmousi-dataset", required=True)
    parser.add_argument("--hybrid-dataset", required=True)
    parser.add_argument("--hybrid-manifest", required=True)
    parser.add_argument("--replacement-report", required=True)
    parser.add_argument("--new-strict-report", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-samples", type=int, default=4003)
    parser.add_argument("--expected-replacements", type=int, default=700)
    parser.add_argument("--expected-train-samples", type=int, default=2800)
    parser.add_argument(
        "--allow-mixed-internal-dt",
        action="store_true",
        help="Accept different internal solver dt values only when saved coordinates are identical.",
    )
    args = parser.parse_args()

    old_path = Path(args.old_dataset).resolve()
    new_path = Path(args.new_marmousi_dataset).resolve()
    hybrid_path = Path(args.hybrid_dataset).resolve()
    manifest_path = Path(args.hybrid_manifest).resolve()
    replacement_report_path = Path(args.replacement_report).resolve()
    strict_report_path = Path(args.new_strict_report).resolve()
    stats_path = Path(args.normalization_stats).resolve()
    output = Path(args.output).resolve()
    for path in (
        old_path,
        new_path,
        hybrid_path,
        manifest_path,
        replacement_report_path,
        strict_report_path,
        stats_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)

    rows = _manifest(manifest_path)
    row_ids = [str(row["sample_id"]) for row in rows]
    if len(row_ids) != int(args.expected_samples) or len(set(row_ids)) != len(row_ids):
        raise ValueError("hybrid manifest count or sample IDs are invalid")
    replacement_ids = {
        str(row["sample_id"])
        for row in rows
        if str(row["split"]) == "train" and str(row["medium_type"]) == "marmousi"
    }
    if len(replacement_ids) != int(args.expected_replacements):
        raise ValueError("hybrid manifest replacement census is invalid")

    strict_report = _json(strict_report_path)
    if (
        not bool(strict_report.get("strict_passed"))
        or int(strict_report.get("sample_count", -1)) != int(args.expected_replacements)
        or Path(str(strict_report.get("dataset", ""))).resolve() != new_path
    ):
        raise ValueError("new Marmousi strict validation report is not bound or passing")
    replacement_report = _json(replacement_report_path)
    if (
        replacement_report.get("status") != "COMPLETE"
        or int(replacement_report.get("sample_count", -1)) != int(args.expected_samples)
        or int(replacement_report.get("replacement_sample_count", -1))
        != int(args.expected_replacements)
        or Path(str(replacement_report.get("output_dataset", ""))).resolve() != hybrid_path
        or Path(str(replacement_report.get("output_manifest", ""))).resolve() != manifest_path
    ):
        raise ValueError("replacement report is not bound to the requested hybrid dataset")
    manifest_sha256 = _sha256(manifest_path)
    if replacement_report.get("output_manifest_sha256") != manifest_sha256:
        raise ValueError("replacement report manifest hash differs from the hybrid manifest")

    with h5py.File(old_path, "r") as old_h5, h5py.File(new_path, "r") as new_h5, h5py.File(
        hybrid_path, "r"
    ) as hybrid_h5:
        for coordinate in ("time_s", "x_m", "z_m"):
            old_coordinate = np.asarray(old_h5[coordinate])
            if not np.array_equal(old_coordinate, np.asarray(new_h5[coordinate])):
                raise ValueError(f"new Marmousi coordinate differs: {coordinate}")
            if not np.array_equal(old_coordinate, np.asarray(hybrid_h5[coordinate])):
                raise ValueError(f"hybrid coordinate differs: {coordinate}")
        old_ids = [_decode(value) for value in old_h5["sample_id"][:]]
        old_hashes = [_decode(value) for value in old_h5["sample_sha256"][:]]
        old_completed = np.asarray(old_h5["completed_mask"][:], dtype=bool)
        new_ids = [_decode(value) for value in new_h5["sample_id"][:]]
        new_hashes = [_decode(value) for value in new_h5["sample_sha256"][:]]
        new_completed = np.asarray(new_h5["completed_mask"][:], dtype=bool)
        hybrid_ids = [_decode(value) for value in hybrid_h5["sample_id"][:]]
        hybrid_hashes = [_decode(value) for value in hybrid_h5["sample_sha256"][:]]
        hybrid_completed = np.asarray(hybrid_h5["completed_mask"][:], dtype=bool)
        hybrid_qc = [_decode(value) for value in hybrid_h5["qc_status"][:]]
        hybrid_splits = [_decode(value) for value in hybrid_h5["split"][:]]
        hybrid_media = [_decode(value) for value in hybrid_h5["medium_type"][:]]
        hybrid_attrs = {
            name: _decode(hybrid_h5.attrs.get(name, ""))
            for name in ("config_sha256", "manifest_sha256", "marmousi_sha256")
        }
        old_root_dt = float(old_h5.attrs["dt_used_s"])
        new_root_dt = float(new_h5.attrs["dt_used_s"])
        hybrid_root_dt = float(hybrid_h5.attrs["dt_used_s"])
        hybrid_root_stride = int(hybrid_h5.attrs["snapshot_stride"])
        hybrid_internal_dt_mixed = bool(hybrid_h5.attrs.get("internal_dt_mixed", False))
        hybrid_component_time_protocol = _decode(
            hybrid_h5.attrs.get("component_time_protocol", "")
        )
        old_dt_values = np.asarray(old_h5["dt_used_s"][:], dtype=np.float64)
        new_dt_values = np.asarray(new_h5["dt_used_s"][:], dtype=np.float64)
        hybrid_dt_values = np.asarray(hybrid_h5["dt_used_s"][:], dtype=np.float64)

    mixed_internal_dt = not np.isclose(old_root_dt, new_root_dt, rtol=0.0, atol=1.0e-15)
    if mixed_internal_dt and not args.allow_mixed_internal_dt:
        raise ValueError(
            "old and replacement internal dt values differ; the mixed protocol was not authorized"
        )
    if mixed_internal_dt:
        if not hybrid_internal_dt_mixed or not np.isnan(hybrid_root_dt) or hybrid_root_stride != -1:
            raise ValueError("hybrid root metadata does not explicitly encode mixed internal dt")
        try:
            component_time_protocol = json.loads(hybrid_component_time_protocol)
        except json.JSONDecodeError as exc:
            raise ValueError("hybrid component_time_protocol is invalid") from exc
        recorded_old_dt = float(component_time_protocol["old_unreplaced_records"]["dt_used_s"])
        recorded_new_dt = float(
            component_time_protocol["new_train_marmousi_records"]["dt_used_s"]
        )
        if not np.isclose(recorded_old_dt, old_root_dt, rtol=0.0, atol=1.0e-15):
            raise ValueError("hybrid old component dt binding differs")
        if not np.isclose(recorded_new_dt, new_root_dt, rtol=0.0, atol=1.0e-15):
            raise ValueError("hybrid replacement component dt binding differs")
    elif hybrid_internal_dt_mixed:
        raise ValueError("hybrid is labelled mixed although component internal dt values match")

    if hybrid_ids != row_ids or len(hybrid_ids) != int(args.expected_samples):
        raise ValueError("hybrid dataset sample ordering differs from its manifest")
    if len(set(hybrid_ids)) != len(hybrid_ids) or any(not value for value in hybrid_ids):
        raise ValueError("hybrid dataset sample IDs are empty or duplicated")
    if not bool(hybrid_completed.all()) or any(value != "passed" for value in hybrid_qc):
        raise ValueError("hybrid dataset is incomplete or contains failed QC")
    if set(new_ids) != replacement_ids or len(new_ids) != len(set(new_ids)):
        raise ValueError("new Marmousi dataset IDs differ from the replacement census")
    if not bool(new_completed.all()):
        raise ValueError("new Marmousi dataset is incomplete")
    for index, row in enumerate(rows):
        if hybrid_splits[index] != str(row["split"]):
            raise ValueError(f"hybrid split differs for {row_ids[index]}")
        if hybrid_media[index] != str(row["medium_type"]):
            raise ValueError(f"hybrid medium_type differs for {row_ids[index]}")

    new_index = {sample_id: index for index, sample_id in enumerate(new_ids)}
    replaced_changed = 0
    unchanged = 0
    unchanged_nontrain = 0
    for index, sample_id in enumerate(hybrid_ids):
        if sample_id in replacement_ids:
            if hybrid_hashes[index] != new_hashes[new_index[sample_id]]:
                raise ValueError(f"replacement sample hash differs for {sample_id}")
            if not np.isclose(
                hybrid_dt_values[index],
                new_dt_values[new_index[sample_id]],
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError(f"replacement sample internal dt differs for {sample_id}")
            replaced_changed += int(old_hashes[index] != hybrid_hashes[index])
        else:
            if old_ids[index] != sample_id or old_hashes[index] != hybrid_hashes[index]:
                raise ValueError(f"unreplaced sample differs at index {index}")
            if not np.isclose(
                hybrid_dt_values[index], old_dt_values[index], rtol=0.0, atol=1.0e-15
            ):
                raise ValueError(f"unreplaced sample internal dt differs at index {index}")
            if not old_completed[index]:
                raise ValueError(f"unreplaced old source is incomplete at index {index}")
            unchanged += 1
            unchanged_nontrain += int(hybrid_splits[index] != "train")
    if replaced_changed != int(args.expected_replacements):
        raise ValueError("not every replacement changed its sample content hash")
    if unchanged != int(args.expected_samples) - int(args.expected_replacements):
        raise ValueError("unreplaced sample count is invalid")

    stats = _json(stats_path)
    expected_velocity_count = int(args.expected_train_samples) * 201 * 201
    expected_wavefield_count = expected_velocity_count * 401
    if (
        stats.get("computed_from_split") != "train"
        or int(stats.get("train_sample_count", -1)) != int(args.expected_train_samples)
        or int(stats.get("dataset_sample_count", -1)) != int(args.expected_samples)
        or Path(str(stats.get("dataset", ""))).resolve() != hybrid_path
        or int(stats.get("velocity", {}).get("count", -1)) != expected_velocity_count
        or int(stats.get("wavefield", {}).get("count", -1)) != expected_wavefield_count
    ):
        raise ValueError("normalization statistics count or dataset binding is invalid")
    for name in ("config_sha256", "manifest_sha256", "marmousi_sha256"):
        if stats.get(name) != hybrid_attrs[name]:
            raise ValueError(f"normalization statistics {name} binding differs")
    if hybrid_attrs["manifest_sha256"] != manifest_sha256:
        raise ValueError("hybrid dataset manifest hash differs from the manifest file")
    for name in ("velocity", "wavefield"):
        values = stats[name]
        if not all(np.isfinite(float(values[key])) for key in ("mean", "std")):
            raise ValueError(f"normalization statistics {name} are not finite")
        if float(values["std"]) <= 0.0:
            raise ValueError(f"normalization statistics {name} std must be positive")

    report = {
        "status": "PASS",
        "protocol": {
            "grid_shape_zx": [201, 201],
            "saved_time_steps": 401,
            "coordinate_arrays_identical": True,
            "replacement_scope": "train/Marmousi only",
            "internal_dt_mixed": bool(mixed_internal_dt),
            "old_unreplaced_dt_s": old_root_dt,
            "new_train_marmousi_dt_s": new_root_dt,
        },
        "sample_count": len(hybrid_ids),
        "replacement_sample_count": len(replacement_ids),
        "replacement_hash_changed_count": replaced_changed,
        "unreplaced_sample_count": unchanged,
        "unchanged_nontrain_sample_count": unchanged_nontrain,
        "new_source_strict_passed": True,
        "normalization_train_sample_count": int(stats["train_sample_count"]),
        "bindings": {
            "old_dataset_sha256": _sha256(old_path),
            "new_marmousi_dataset_sha256": _sha256(new_path),
            "hybrid_dataset_sha256": _sha256(hybrid_path),
            "hybrid_manifest_sha256": manifest_sha256,
            "normalization_stats_sha256": _sha256(stats_path),
            **hybrid_attrs,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
