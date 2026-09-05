#!/usr/bin/env python3
"""Isolate the CPADC parent correction-scale override on one train record."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import (
    GuardedOnsetDataset,
)
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    source_causality_onset_s,
)
from scripts.run_v5_instance_adaptation import (
    _load_parent,
    _predict_parent,
    _read_future_truth,
    resolve_saved_time_parent_config,
)


R5B_CHECKPOINT_SHA256 = (
    "9295d21f2e0858dd5d2bb2ab2c10a0a39e062148a796e7ab71eec016a2ae9a3b"
)
ALLOWED_SAMPLE_ID = "train_uniform_00303"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(prediction: torch.Tensor, target: torch.Tensor) -> float:
    difference = prediction.double() - target.double()
    numerator = float(difference.square().sum())
    denominator = float(target.double().square().sum())
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def _find_exact32_row(audit: dict[str, object]) -> dict[str, object]:
    matches = [
        row
        for row in audit.get("measurements", [])
        if row.get("sample_id") == ALLOWED_SAMPLE_ID
    ]
    if len(matches) != 1:
        raise ValueError("registered sample must occur exactly once in exact-32 audit")
    return dict(matches[0])


def _find_cpadc_row(paths: tuple[Path, ...]) -> tuple[dict[str, object], Path]:
    matches: list[tuple[dict[str, object], Path]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            if row.get("sample_id") == ALLOWED_SAMPLE_ID:
                matches.append((dict(row), path))
    if len(matches) != 1:
        raise ValueError("registered sample must occur once in CPADC summaries")
    return matches[0]


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_id != ALLOWED_SAMPLE_ID:
        raise ValueError(f"diagnostic is restricted to {ALLOWED_SAMPLE_ID}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = Path(str(config["parent_checkpoint"]))
    if _sha256(checkpoint) != R5B_CHECKPOINT_SHA256:
        raise ValueError("r5b checkpoint binding changed")
    configured_scale = float(config["parent_correction_scale"])
    if configured_scale != 1.0:
        raise ValueError("registered CPADC config no longer forces scale 1.0")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_scale = float(
        checkpoint_payload["model_state"]["dense_decoder.correction_scale"]
    )
    if not 0.0 < checkpoint_scale < 0.1:
        raise ValueError("unexpected r5b checkpoint correction scale")

    audit = json.loads(args.exact32_audit.read_text(encoding="utf-8"))
    audit_row = _find_exact32_row(audit)
    cpadc_row, cpadc_summary = _find_cpadc_row(tuple(args.cpadc_summaries))
    exact_indices = tuple(int(value) for value in audit_row["time_indices"])
    if len(exact_indices) != 32 or len(set(exact_indices)) != 32:
        raise ValueError("registered panel is not exact-32")

    manifest = build_manifest(config["source_h5"])
    device = torch.device(args.device)
    parent, normalizer = _load_parent(config, manifest, device)
    parent.eval()
    loaded_forced_scale = float(parent.dense_decoder.correction_scale)
    if loaded_forced_scale != 1.0:
        raise ValueError("CPADC parent loader did not reproduce the forced scale")
    saved_time_config = resolve_saved_time_parent_config(config)
    if saved_time_config is None:
        raise ValueError("diagnostic requires saved-time parent config")
    lead_cycles = float(
        dict(saved_time_config["loss"])["hard_causality_lead_cycles"]
    )

    with GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split="train",
        sample_ids=(args.sample_id,),
        travel_time_h5=config.get("travel_time_h5"),
    ) as dataset:
        record = dataset[0]
        forced_raw = _predict_parent(
            parent,
            normalizer,
            record,
            device,
            normalized=True,
            time_block=int(config["deployment_time_block"]),
        )
        parent.dense_decoder.correction_scale.fill_(checkpoint_scale)
        restored_raw = _predict_parent(
            parent,
            normalizer,
            record,
            device,
            normalized=True,
            time_block=int(config["deployment_time_block"]),
        )
        source = record.source_parameters.to(device).unsqueeze(0)
        times = record.time_s.to(device).unsqueeze(0)
        onset = source_causality_onset_s(source, lead_cycles=lead_cycles)
        forced_causal = apply_hard_causality(forced_raw, times, onset)
        restored_causal = apply_hard_causality(restored_raw, times, onset)
        physical_truth = _read_future_truth(config["source_h5"], record.source_index)
        target = normalizer.encode_pressure(
            physical_truth.to(device).unsqueeze(0), source[:, 4]
        )

    index_tensor = torch.tensor(exact_indices, dtype=torch.long, device=device)
    observed_mask = np.ones(int(record.time_s.numel()), dtype=bool)
    observed_mask[list(record.observed_indices)] = False
    future_indices = torch.from_numpy(np.flatnonzero(observed_mask)).to(device)
    forced_exact = _relative(
        forced_causal.index_select(1, index_tensor),
        target.index_select(1, index_tensor),
    )
    restored_exact = _relative(
        restored_causal.index_select(1, index_tensor),
        target.index_select(1, index_tensor),
    )
    forced_future = _relative(
        forced_raw.index_select(1, future_indices),
        target.index_select(1, future_indices),
    )
    restored_future = _relative(
        restored_raw.index_select(1, future_indices),
        target.index_select(1, future_indices),
    )
    archived_exact = float(audit_row["r5b_relative_l2"])
    archived_cpadc_future = float(
        cpadc_row["parent_future_fullfield_relative_l2"]
    )
    restored_delta = abs(restored_exact - archived_exact)
    forced_future_delta = abs(forced_future - archived_cpadc_future)
    output: dict[str, object] = {
        "schema": "r5b_correction_scale_override_train_diagnostic_v1",
        "status": "complete",
        "role": "train_only_single_variable_parent_loading_diagnostic",
        "sample_id": record.sample_id,
        "split": "train",
        "validation_truth_opened": False,
        "test_id_opened": False,
        "correction_scale": {
            "checkpoint": checkpoint_scale,
            "cpadc_configured": configured_scale,
            "cpadc_loaded": loaded_forced_scale,
            "forced_over_checkpoint_ratio": configured_scale / checkpoint_scale,
        },
        "metrics": {
            "archived_training_audit_exact32_relative_l2": archived_exact,
            "forced_scale1_exact32_relative_l2": forced_exact,
            "restored_checkpoint_scale_exact32_relative_l2": restored_exact,
            "archived_cpadc_forced_scale_future_relative_l2": archived_cpadc_future,
            "live_forced_scale1_future_relative_l2": forced_future,
            "restored_checkpoint_scale_future_relative_l2": restored_future,
            "restored_over_forced_exact32_ratio": restored_exact
            / max(forced_exact, 1.0e-30),
            "restored_over_forced_future_ratio": restored_future
            / max(forced_future, 1.0e-30),
        },
        "decision": {
            "forced_cpadc_future_reproduced": forced_future_delta <= 1.0e-5,
            "checkpoint_scale_restores_training_audit": restored_delta <= 1.0e-5,
            "correction_scale_override_is_root_cause": forced_future_delta <= 1.0e-5
            and restored_delta <= 1.0e-5
            and restored_exact <= 0.1 * forced_exact,
            "thresholds": {
                "maximum_archive_reproduction_absolute_delta": 1.0e-5,
                "maximum_restored_over_forced_exact32_ratio": 0.1,
            },
        },
        "reproduction_deltas": {
            "forced_future_absolute": forced_future_delta,
            "restored_exact32_absolute": restored_delta,
        },
        "exact32_time_indices": list(exact_indices),
        "observed_indices": list(record.observed_indices),
        "bindings": {
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "r5b_checkpoint": str(checkpoint.resolve()),
            "r5b_checkpoint_sha256": R5B_CHECKPOINT_SHA256,
            "exact32_audit": str(args.exact32_audit.resolve()),
            "exact32_audit_sha256": _sha256(args.exact32_audit),
            "cpadc_summary": str(cpadc_summary.resolve()),
            "cpadc_summary_sha256": _sha256(cpadc_summary),
            "script_sha256": _sha256(Path(__file__)),
        },
    }
    _atomic_json(output, args.output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--exact32-audit", type=Path, required=True)
    parser.add_argument(
        "--cpadc-summaries", type=Path, nargs="+", required=True
    )
    parser.add_argument("--sample-id", default=ALLOWED_SAMPLE_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
