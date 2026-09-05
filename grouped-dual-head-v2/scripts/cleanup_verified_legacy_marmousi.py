#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CONFIRM = "DELETE_VERIFIED_LEGACY_MARMOUSI"


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


def _attempt_records(path: Path) -> dict[str, tuple[str, int, float]]:
    result: dict[str, tuple[str, int, float]] = {}
    for record_path in sorted(path.glob("shards/shard_*/records.json")):
        for row in json.loads(record_path.read_text(encoding="utf-8")):
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise ValueError(f"duplicate prediction sample in {path}: {sample_id}")
            prediction_path = Path(str(row["prediction_path"])).resolve()
            if not prediction_path.is_relative_to(path.resolve()):
                raise ValueError(f"prediction escapes its attempt root: {prediction_path}")
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            actual_bytes = int(prediction_path.stat().st_size)
            actual_sha256 = _sha256(prediction_path)
            if actual_bytes != int(row["prediction_byte_count"]):
                raise ValueError(f"prediction byte count mismatch: {prediction_path}")
            if actual_sha256 != str(row["prediction_sha256"]):
                raise ValueError(f"prediction SHA-256 mismatch: {prediction_path}")
            result[sample_id] = (
                actual_sha256,
                actual_bytes,
                float(row["relative_l2"]),
            )
    return result


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _legacy_train_shards(root: Path, legacy_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    for path in sorted((root / "train").glob("train-*.h5")):
        with h5py.File(path, "r") as h5:
            split = _decode(h5.attrs.get("split", ""))
            marmousi_sha256 = _decode(h5.attrs.get("marmousi_sha256", ""))
            media = {_decode(value) for value in h5["medium_type"][:]}
            sample_ids = [_decode(value) for value in h5["sample_id"][:]]
            complete = bool(np.asarray(h5["completed_mask"], dtype=bool).all())
        row = {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sample_count": len(sample_ids),
            "first_sample_id": sample_ids[0] if sample_ids else None,
            "last_sample_id": sample_ids[-1] if sample_ids else None,
            "medium_types": sorted(media),
        }
        is_pure_legacy_marmousi = (
            split == "train"
            and marmousi_sha256 == legacy_sha256
            and media == {"marmousi"}
            and sample_ids
            and all(sample_id.startswith("train_marmousi_") for sample_id in sample_ids)
            and complete
        )
        if is_pure_legacy_marmousi:
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if not sidecar.is_file():
                raise ValueError(f"selected legacy shard lacks checksum sidecar: {sidecar}")
            row["sidecar"] = str(sidecar.resolve())
            row["sidecar_bytes"] = int(sidecar.stat().st_size)
            selected.append(row)
        else:
            retained.append(row)
    return selected, retained


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete only hash-bound legacy train/Marmousi shards and one verified duplicate attempt.")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--legacy-marmousi-sha256", required=True)
    parser.add_argument("--superseded-attempt", required=True)
    parser.add_argument("--authoritative-attempt", required=True)
    parser.add_argument("--completed-path", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default=None)
    args = parser.parse_args()

    shard_root = Path(args.shard_root).resolve()
    superseded = Path(args.superseded_attempt).resolve()
    authoritative = Path(args.authoritative_attempt).resolve()
    completed_path = Path(args.completed_path).resolve()
    report_path = Path(args.report).resolve()
    if not shard_root.is_dir() or not superseded.is_dir() or not authoritative.is_dir():
        raise FileNotFoundError("one or more cleanup roots do not exist")
    if superseded == authoritative or superseded.parent != authoritative.parent:
        raise ValueError("duplicate attempts must be distinct siblings")
    completed_target = Path(completed_path.read_text(encoding="utf-8").strip()).resolve()
    if completed_target != authoritative:
        raise ValueError(f"completed.path does not select the authoritative attempt: {completed_target}")
    old_records = _attempt_records(superseded)
    authoritative_records = _attempt_records(authoritative)
    if len(old_records) != 480 or old_records != authoritative_records:
        raise ValueError("superseded prediction attempt is not an exact 480-record hash/size/metric duplicate")

    selected, retained = _legacy_train_shards(shard_root, str(args.legacy_marmousi_sha256))
    if len(selected) != 87 or sum(row["sample_count"] for row in selected) != 696:
        raise ValueError(
            f"expected exactly 87 pure shards / 696 legacy train Marmousi samples, got "
            f"{len(selected)} / {sum(row['sample_count'] for row in selected)}"
        )
    selected_bytes = sum(row["bytes"] + row["sidecar_bytes"] for row in selected)
    duplicate_bytes = _tree_bytes(superseded)
    if args.execute and args.confirm != CONFIRM:
        raise ValueError(f"execution requires --confirm {CONFIRM}")
    report = {
        "status": "VERIFIED_READY" if not args.execute else "DELETION_STARTED",
        "recoverability": "not_recoverable_locally; authoritative duplicate attempt is retained",
        "legacy_train_marmousi": {
            "selected_shard_count": len(selected),
            "selected_sample_count": sum(row["sample_count"] for row in selected),
            "selected_bytes": selected_bytes,
            "selected": selected,
            "retained_train_shard_count": len(retained),
        },
        "superseded_duplicate_attempt": {
            "path": str(superseded),
            "bytes": duplicate_bytes,
            "record_count": len(old_records),
            "authoritative_path": str(authoritative),
            "record_bindings_identical": True,
        },
        "total_reclaim_bytes": selected_bytes + duplicate_bytes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.execute:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    for row in selected:
        Path(row["path"]).unlink()
        Path(row["sidecar"]).unlink()
    shutil.rmtree(superseded)
    report["status"] = "COMPLETE"
    report["postconditions"] = {
        "deleted_legacy_shards_absent": all(not Path(row["path"]).exists() for row in selected),
        "superseded_attempt_absent": not superseded.exists(),
        "authoritative_attempt_present": authoritative.is_dir(),
    }
    if not all(report["postconditions"].values()):
        raise RuntimeError(f"cleanup postcondition failed: {report['postconditions']}")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
