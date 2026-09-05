#!/usr/bin/env python3
"""Freeze the first full R40 training attempt after its cache audit passes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if bundle.get("schema") != "r40_frequency_cache_bundle_v2":
        raise RuntimeError("unexpected R40 bundle schema")
    if audit.get("schema") != "r40_frequency_cache_bundle_audit_v1":
        raise RuntimeError("unexpected R40 audit schema")
    if audit.get("status") != "pass":
        raise RuntimeError("R40 cache audit did not pass")
    if not audit.get("representation_oracle_max_lte_0p05"):
        raise RuntimeError("R40 representation cannot meet the frozen max gate")
    if bundle.get("selection_sha256") != audit.get("selection_sha256"):
        raise RuntimeError("R40 bundle/audit selection mismatch")

    payload = {
        "schema": "r40_frequency_operator_preregistration_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": bundle["selection_sha256"],
        "cache_bundle_sha256": bundle["bundle_sha256"],
        "cache_audit_sha256": sha256_file(args.audit.expanduser().resolve()),
        "train_script": str(args.train_script.expanduser().resolve()),
        "train_script_sha256": sha256_file(args.train_script.expanduser().resolve()),
        "launcher": str(args.launcher.expanduser().resolve()),
        "launcher_sha256": sha256_file(args.launcher.expanduser().resolve()),
        "fit_record_count": 1032,
        "fit_group_count": int(audit["fit_group_count"]),
        "holdout_record_count": 56,
        "holdout_group_count": int(audit["holdout_group_count"]),
        "method": {
            "input_channels": 32,
            "temporal_frequency_range_hz": [5.0, 80.0],
            "temporal_frequency_count": 75,
            "spatial_transform": "DCT-II_ortho",
            "spatial_retained": [96, 96],
            "model": "frequency_conditioned_spatial_dct_fno",
            "width": 48,
            "fourier_blocks": 4,
            "modes_per_axis": 16,
            "correction_cap": 1.5,
            "zero_initialized_output": True,
            "raw_medium_dct_and_scales": True,
            "mean_velocity_conditioning": True,
            "frequency_amplitude_conditioning": True,
        },
        "optimization": {
            "epochs": 50,
            "full_fit_pass_each_epoch": True,
            "local_batch_size": 16,
            "global_batch_size": 64,
            "optimizer": "Adam",
            "learning_rate": 3.0e-4,
            "warmup_epochs": 1,
            "cosine_minimum_factor": 0.05,
            "weight_decay": 1.0e-5,
            "tail_weight": 1.0,
            "hinge_weight": 2.0,
            "shape_weight": 0.002,
            "gradient_clip_norm": 1.0,
            "amp": "bfloat16",
            "seed": 400828,
            "evaluate_epochs": [0, 1] + list(range(2, 51, 2)),
            "stop_on_pass": True,
        },
        "frozen_accuracy_gate": {
            "record_relative_l2_mean_lte": 0.05,
            "record_relative_l2_max_lte": 0.05,
            "both_required": True,
        },
        "post_gate_sequence": [
            "open_r29b_fresh_group_disjoint_holdout_only_after_development_pass",
            "open_final_480_validation_only_after_r29b_pass",
            "modify_word_and_tex_only_after_accuracy_evidence_passes",
        ],
        "data_access": {
            "fit": "train_only",
            "holdout": "train_group_disjoint_opened_development",
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
        },
    }
    payload["preregistration_sha256"] = canonical_sha(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
