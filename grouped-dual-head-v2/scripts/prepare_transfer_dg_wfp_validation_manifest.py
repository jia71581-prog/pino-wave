#!/usr/bin/env python3
"""Freeze the public-input-only manifest for the one-shot WFP validation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = ROOT / "results/transfer_dg_wfp_full2800_manifest_20260902.json"
OUTPUT = ROOT / "results/transfer_dg_wfp_validation600_manifest_20260903.json"
FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
EXPECTED = {"uniform": 90, "layered": 240, "anomaly": 120, "marmousi": 150}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    train = json.loads(TRAIN_MANIFEST.read_text())
    source_path = Path(train["source_h5"])
    if sha256(source_path) != train["source_h5_sha256"]:
        raise RuntimeError("source VDS binding drift")
    train_ids = {str(row["sample_id"]) for row in train["records"]}
    rows = []
    with h5py.File(source_path, "r", swmr=True) as source:
        if len(source["time_s"]) != 401:
            raise RuntimeError("validation time axis is not 401 stored frames")
        for index, raw_split in enumerate(source["split"]):
            if text(raw_split) != "validation":
                continue
            sample_id = text(source["sample_id"][index])
            family = text(source["medium_type"][index])
            if sample_id in train_ids:
                raise RuntimeError(f"train/validation sample overlap: {sample_id}")
            if family not in FAMILIES:
                raise RuntimeError(f"unexpected validation family: {family}")
            rows.append(
                {
                    "source_index": index,
                    "sample_id": sample_id,
                    "sample_sha256": text(source["sample_sha256"][index]),
                    "group_id": text(source["group_id"][index]),
                    "family": family,
                    "source_x_m": float(source["source_x_m"][index]),
                    "source_z_m": float(source["source_z_m"][index]),
                    "source_f0_hz": float(source["source_f0_hz"][index]),
                    "source_t0_s": float(source["source_t0_s"][index]),
                    "source_amplitude": float(source["source_amplitude"][index]),
                }
            )
    counts = Counter(row["family"] for row in rows)
    if len(rows) != 600 or dict(counts) != EXPECTED:
        raise RuntimeError(f"validation census mismatch: {len(rows)}, {dict(counts)}")
    payload = {
        "schema": "transfer_dg_wfp_validation600_public_manifest_v1",
        "split": "validation",
        "record_count": len(rows),
        "frame_count": 401,
        "family_counts": EXPECTED,
        "records": rows,
        "selection_sha256": canonical(rows),
        "source_h5": str(source_path.resolve()),
        "source_h5_sha256": train["source_h5_sha256"],
        "source_manifest_sha256": train["source_manifest_sha256"],
        "allowed_model_inputs": [
            "velocity_mps",
            "source_map",
            "source_wavelet",
            "source_parameters",
            "time_s",
            "fixed_cpml_profiles",
            "velocity_source_derived_eikonal"
        ],
        "model_input_wavefield_frames": 0,
        "validation_public_inputs_opened": True,
        "validation_future_truth_opened": False,
        "test_id_opened": False,
    }
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({key: payload[key] for key in (
        "schema", "record_count", "frame_count", "family_counts",
        "selection_sha256", "validation_future_truth_opened", "test_id_opened"
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
