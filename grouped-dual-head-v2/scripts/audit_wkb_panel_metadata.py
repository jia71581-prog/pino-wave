#!/usr/bin/env python3
"""Read-only source-metadata audit for frozen WKB train-only panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest


FIELDS = (
    "source_x_m",
    "source_z_m",
    "source_f0_hz",
    "source_t0_s",
    "source_amplitude",
)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
        "unique_count": int(len(np.unique(array))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("identity")
    args = parser.parse_args()
    identity_path = Path(args.identity).resolve()
    identity = json.loads(identity_path.read_text())
    source_h5 = Path(identity["bindings"]["source_h5"]).resolve()
    manifest = build_manifest(source_h5)
    by_sample = {record.sample_id: record for record in manifest.records}
    panels = {
        "fit": tuple(identity["fit_sample_ids"]),
        "calibration": tuple(identity["calibration_sample_ids"]),
        "confirm": tuple(identity["confirm_sample_ids"]),
    }
    selected = [by_sample[sample_id] for ids in panels.values() for sample_id in ids]
    if len(selected) != 108 or any(record.split != "train" for record in selected):
        raise ValueError("expected exactly 108 train-only panel records")
    if len({record.sample_id for record in selected}) != len(selected):
        raise ValueError("panel sample IDs overlap")

    result: dict[str, object] = {
        "schema": "wkb_panel_source_metadata_audit_v1",
        "identity": str(identity_path),
        "manifest_digest": manifest.digest,
        "wavefield_read": False,
        "record_fields": tuple(by_sample[next(iter(panels["fit"]))].__dataclass_fields__),
        "panels": {},
        "range_overlap": {},
    }
    raw: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(source_h5, "r", swmr=True) as h5:
        missing = tuple(field for field in FIELDS if field not in h5)
        if missing:
            raise ValueError(f"missing source metadata: {missing}")
        for panel, sample_ids in panels.items():
            records = tuple(by_sample[sample_id] for sample_id in sample_ids)
            panel_result: dict[str, object] = {}
            raw[panel] = {}
            for family in ("uniform", "layered", "marmousi"):
                family_records = tuple(r for r in records if r.medium_type == family)
                indices = np.asarray([r.source_index for r in family_records], dtype=np.int64)
                values = {
                    field: np.asarray(h5[field][indices], dtype=np.float64)
                    for field in FIELDS
                }
                panel_result[family] = {
                    "record_count": len(family_records),
                    "group_id_unique_count": len({r.group_id for r in family_records}),
                    "group_ids": sorted({r.group_id for r in family_records}),
                    "fields": {field: _summary(array) for field, array in values.items()},
                }
                for field, array in values.items():
                    raw[panel].setdefault(field, np.empty(0, dtype=np.float64))
                    raw[panel][field] = np.concatenate((raw[panel][field], array))
            result["panels"][panel] = panel_result

    for field in FIELDS:
        result["range_overlap"][field] = {}
        for left, right in (("fit", "calibration"), ("fit", "confirm"), ("calibration", "confirm")):
            a, b = raw[left][field], raw[right][field]
            result["range_overlap"][field][f"{left}_vs_{right}"] = {
                "intervals_overlap": bool(max(a.min(), b.min()) <= min(a.max(), b.max())),
                "shared_value_count": int(len(np.intersect1d(np.unique(a), np.unique(b)))),
            }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
