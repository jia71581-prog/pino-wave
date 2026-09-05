#!/usr/bin/env python3
"""Reproduce the archived r5b validation result with and without causality.

This diagnostic is deliberately restricted to a validation record that was
already opened by the frozen r14 comparison.  It does not select a candidate
or expose any new validation or test_id record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

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
CPADC_CHECKPOINT_SHA256 = (
    "e92cc37722388640d060cd78875f1ebdc8e9ca1dad395641a5123625ecafb55a"
)
ALLOWED_SAMPLE_ID = "validation_uniform_00000"


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


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    difference = prediction.double() - target.double()
    numerator = float(difference.square().sum())
    denominator = float(target.double().square().sum())
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def _find_archived_record(archive: dict[str, object]) -> dict[str, object]:
    records = archive.get("records")
    if not isinstance(records, list):
        raise ValueError("r14 archive lacks records")
    matches = [row for row in records if row.get("sample_id") == ALLOWED_SAMPLE_ID]
    if len(matches) != 1:
        raise ValueError("r14 archive must contain the registered sample exactly once")
    return dict(matches[0])


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_id != ALLOWED_SAMPLE_ID:
        raise ValueError(f"this diagnostic is restricted to {ALLOWED_SAMPLE_ID}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("CPADC config must contain a mapping")
    checkpoint = Path(str(config["parent_checkpoint"]))
    if _sha256(checkpoint) != R5B_CHECKPOINT_SHA256:
        raise ValueError("r5b checkpoint binding changed")
    basis_checkpoint = args.cpadc_checkpoint
    if _sha256(basis_checkpoint) != CPADC_CHECKPOINT_SHA256:
        raise ValueError("CPADC checkpoint binding changed")

    archived = json.loads(args.r14_summary.read_text(encoding="utf-8"))
    archived_record = _find_archived_record(archived)
    archived_basis = dict(archived.get("basis", {}))
    if archived_basis.get("checkpoint_sha256") != CPADC_CHECKPOINT_SHA256:
        raise ValueError("r14 archive is bound to a different CPADC checkpoint")
    panel = dict(archived_record["pi_matched_panel"])
    time_indices = tuple(int(value) for value in panel["time_indices"])
    if len(time_indices) != 32 or len(set(time_indices)) != 32:
        raise ValueError("archived PI panel is not exact-32")

    manifest = build_manifest(config["source_h5"])
    device = torch.device(args.device)
    parent, normalizer = _load_parent(config, manifest, device)
    parent.eval()
    saved_time_config = resolve_saved_time_parent_config(config)
    if saved_time_config is None:
        raise ValueError("diagnostic requires a saved-time parent config")
    loss_config = dict(saved_time_config.get("loss", {}))
    if loss_config.get("hard_causality") is not True:
        raise ValueError("registered r5b config no longer enables hard causality")
    lead_cycles = float(loss_config["hard_causality_lead_cycles"])

    with GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split="validation",
        sample_ids=(args.sample_id,),
        travel_time_h5=config.get("travel_time_h5"),
    ) as dataset:
        if len(dataset) != 1:
            raise RuntimeError("registered validation diagnostic must load one record")
        record = dataset[0]
        raw_parent = _predict_parent(
            parent,
            normalizer,
            record,
            device,
            normalized=True,
            time_block=int(config.get("deployment_time_block", 32)),
        )
        onset_s = source_causality_onset_s(
            record.source_parameters.to(device).unsqueeze(0),
            lead_cycles=lead_cycles,
        )
        times = record.time_s.to(device).unsqueeze(0)
        masked_parent = apply_hard_causality(raw_parent, times, onset_s)

        # Future truth is opened only after both parent variants are materialized.
        physical_truth = _read_future_truth(config["source_h5"], record.source_index)
        source = record.source_parameters.to(device).unsqueeze(0)
        target = normalizer.encode_pressure(
            physical_truth.to(device).unsqueeze(0), source[:, 4]
        )

    panel_indices = torch.tensor(time_indices, dtype=torch.long, device=device)
    raw_panel = raw_parent.index_select(1, panel_indices)
    masked_panel = masked_parent.index_select(1, panel_indices)
    target_panel = target.index_select(1, panel_indices)
    raw_exact32 = _relative_l2(raw_panel, target_panel)
    masked_exact32 = _relative_l2(masked_panel, target_panel)
    raw_all401 = _relative_l2(raw_parent, target)
    masked_all401 = _relative_l2(masked_parent, target)
    archived_raw_exact32 = float(panel["parent_relative_l2"])
    reproduction_delta = abs(raw_exact32 - archived_raw_exact32)
    masked_frame_count = int((times < onset_s[:, None]).sum())
    eliminated = raw_parent - masked_parent

    output: dict[str, object] = {
        "schema": "r5b_causality_evaluator_gap_diagnostic_v1",
        "status": "complete",
        "role": "reopened_previously_evaluated_validation_record_protocol_diagnostic",
        "sample_id": record.sample_id,
        "split": "validation",
        "validation_record_was_already_opened_by_r14": True,
        "new_validation_records_opened": 0,
        "test_id_opened": False,
        "causality_contract": {
            "enabled": True,
            "lead_cycles": lead_cycles,
            "source_peak_time_s": float(record.source_parameters[3]),
            "source_frequency_hz": float(record.source_parameters[2]),
            "causal_onset_s": float(onset_s.item()),
            "masked_saved_frame_count": masked_frame_count,
            "saved_frame_count": int(record.time_s.numel()),
        },
        "metrics": {
            "archived_r14_unmasked_exact32_relative_l2": archived_raw_exact32,
            "live_unmasked_exact32_relative_l2": raw_exact32,
            "live_hard_causal_exact32_relative_l2": masked_exact32,
            "live_unmasked_all401_relative_l2": raw_all401,
            "live_hard_causal_all401_relative_l2": masked_all401,
            "exact32_unmasked_reproduction_absolute_delta": reproduction_delta,
            "exact32_masked_over_unmasked_ratio": masked_exact32
            / max(raw_exact32, 1.0e-30),
            "eliminated_precausal_parent_l2": float(
                torch.linalg.vector_norm(eliminated.double())
            ),
        },
        "decision": {
            "unmasked_archive_reproduced": reproduction_delta <= 1.0e-5,
            "hard_causality_materially_changes_metric": masked_exact32
            <= 0.5 * raw_exact32,
            "evaluator_protocol_gap_confirmed": reproduction_delta <= 1.0e-5
            and masked_exact32 <= 0.5 * raw_exact32,
            "thresholds": {
                "maximum_archive_reproduction_absolute_delta": 1.0e-5,
                "maximum_masked_over_unmasked_ratio": 0.5,
            },
        },
        "bindings": {
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "r5b_checkpoint": str(checkpoint.resolve()),
            "r5b_checkpoint_sha256": R5B_CHECKPOINT_SHA256,
            "cpadc_checkpoint": str(basis_checkpoint.resolve()),
            "cpadc_checkpoint_sha256": CPADC_CHECKPOINT_SHA256,
            "r14_summary": str(args.r14_summary.resolve()),
            "r14_summary_sha256": _sha256(args.r14_summary),
            "script_sha256": _sha256(Path(__file__)),
        },
        "time_indices": list(time_indices),
    }
    _atomic_json(output, args.output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpadc-checkpoint", type=Path, required=True)
    parser.add_argument("--r14-summary", type=Path, required=True)
    parser.add_argument("--sample-id", default=ALLOWED_SAMPLE_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
