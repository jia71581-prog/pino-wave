#!/usr/bin/env python3
"""Train-only diagnosis of r5b exact-32 versus complete-time behavior."""
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


def _relative(numerator: float, denominator: float) -> float:
    return float(math.sqrt(float(numerator) / max(float(denominator), 1.0e-30)))


def _count_for_fraction(values: np.ndarray, fraction: float) -> int:
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = float(ordered.sum())
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(ordered), fraction * total, side="left") + 1)


def _find_audit_measurement(audit: dict[str, object]) -> dict[str, object]:
    rows = audit.get("measurements")
    if not isinstance(rows, list):
        raise ValueError("exact-32 audit lacks measurements")
    matches = [row for row in rows if row.get("sample_id") == ALLOWED_SAMPLE_ID]
    if len(matches) != 1:
        raise ValueError("registered train sample must occur exactly once in audit")
    return dict(matches[0])


def _find_cpadc_record(paths: tuple[Path, ...]) -> tuple[dict[str, object], Path]:
    matches: list[tuple[dict[str, object], Path]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("records", []):
            if row.get("sample_id") == ALLOWED_SAMPLE_ID:
                matches.append((dict(row), path))
    if len(matches) != 1:
        raise ValueError("registered train sample must occur once in CPADC summaries")
    return matches[0]


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.sample_id != ALLOWED_SAMPLE_ID:
        raise ValueError(f"diagnostic is restricted to {ALLOWED_SAMPLE_ID}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = Path(str(config["parent_checkpoint"]))
    if _sha256(checkpoint) != R5B_CHECKPOINT_SHA256:
        raise ValueError("r5b checkpoint binding changed")
    audit = json.loads(args.exact32_audit.read_text(encoding="utf-8"))
    audit_row = _find_audit_measurement(audit)
    cpadc_row, cpadc_summary_path = _find_cpadc_record(
        tuple(args.cpadc_summaries)
    )
    exact_indices = tuple(int(value) for value in audit_row["time_indices"])
    if len(exact_indices) != 32 or len(set(exact_indices)) != 32:
        raise ValueError("registered audit panel is not exact-32")

    manifest = build_manifest(config["source_h5"])
    device = torch.device(args.device)
    parent, normalizer = _load_parent(config, manifest, device)
    parent.eval()
    saved_time_config = resolve_saved_time_parent_config(config)
    if saved_time_config is None:
        raise ValueError("diagnostic requires saved-time parent config")
    loss_config = dict(saved_time_config["loss"])
    lead_cycles = float(loss_config["hard_causality_lead_cycles"])

    with GuardedOnsetDataset(
        config["source_h5"],
        manifest,
        split="train",
        sample_ids=(args.sample_id,),
        travel_time_h5=config.get("travel_time_h5"),
    ) as dataset:
        record = dataset[0]
        raw = _predict_parent(
            parent,
            normalizer,
            record,
            device,
            normalized=True,
            time_block=int(config.get("deployment_time_block", 32)),
        )
        source = record.source_parameters.to(device).unsqueeze(0)
        times = record.time_s.to(device).unsqueeze(0)
        onset = source_causality_onset_s(source, lead_cycles=lead_cycles)
        prediction = apply_hard_causality(raw, times, onset)
        physical_truth = _read_future_truth(config["source_h5"], record.source_index)
        target = normalizer.encode_pressure(
            physical_truth.to(device).unsqueeze(0), source[:, 4]
        )

    prediction64 = prediction.double()
    target64 = target.double()
    frame_error = (
        (prediction64 - target64).square().sum(dim=(-2, -1)).squeeze(0).cpu().numpy()
    )
    frame_truth = target64.square().sum(dim=(-2, -1)).squeeze(0).cpu().numpy()
    frame_prediction = (
        prediction64.square().sum(dim=(-2, -1)).squeeze(0).cpu().numpy()
    )
    selected_mask = np.zeros(len(frame_error), dtype=bool)
    selected_mask[list(exact_indices)] = True
    observed_mask = np.zeros(len(frame_error), dtype=bool)
    observed_mask[list(record.observed_indices)] = True
    future_mask = ~observed_mask

    exact_num = float(frame_error[selected_mask].sum())
    exact_den = float(frame_truth[selected_mask].sum())
    full_num = float(frame_error.sum())
    full_den = float(frame_truth.sum())
    future_num = float(frame_error[future_mask].sum())
    future_den = float(frame_truth[future_mask].sum())
    nonselected_num = float(frame_error[~selected_mask].sum())
    nonselected_den = float(frame_truth[~selected_mask].sum())
    ranked = np.argsort(frame_error)[::-1]
    top = []
    for index in ranked[:20]:
        top.append(
            {
                "time_index": int(index),
                "time_s": float(record.time_s[index]),
                "selected_by_exact32": bool(selected_mask[index]),
                "squared_error": float(frame_error[index]),
                "truth_squared_norm": float(frame_truth[index]),
                "prediction_squared_norm": float(frame_prediction[index]),
                "fraction_of_full_error": float(frame_error[index] / full_num),
            }
        )

    live_exact = _relative(exact_num, exact_den)
    live_future = _relative(future_num, future_den)
    archived_exact = float(audit_row["r5b_relative_l2"])
    archived_future = float(cpadc_row["parent_future_fullfield_relative_l2"])
    output: dict[str, object] = {
        "schema": "r5b_full_time_failure_train_diagnostic_v1",
        "status": "complete",
        "role": "train_only_temporal_generalization_diagnostic",
        "sample_id": record.sample_id,
        "split": "train",
        "validation_truth_opened": False,
        "test_id_opened": False,
        "metrics": {
            "archived_exact32_relative_l2": archived_exact,
            "live_exact32_relative_l2": live_exact,
            "archived_cpadc_future_relative_l2": archived_future,
            "live_future_relative_l2": live_future,
            "live_all401_relative_l2": _relative(full_num, full_den),
            "live_nonselected369_relative_l2": _relative(
                nonselected_num, nonselected_den
            ),
            "exact32_fraction_of_full_error_numerator": exact_num / full_num,
            "exact32_fraction_of_full_truth_energy": exact_den / full_den,
            "all401_prediction_to_truth_energy_ratio": float(
                frame_prediction.sum() / full_den
            ),
            "top_frames_needed_for_50pct_error": _count_for_fraction(
                frame_error, 0.50
            ),
            "top_frames_needed_for_90pct_error": _count_for_fraction(
                frame_error, 0.90
            ),
            "top_frames_needed_for_99pct_error": _count_for_fraction(
                frame_error, 0.99
            ),
            "maximum_frame_error_index": int(ranked[0]),
        },
        "reproduction": {
            "exact32_absolute_delta": abs(live_exact - archived_exact),
            "future_absolute_delta": abs(live_future - archived_future),
            "exact32_reproduced": abs(live_exact - archived_exact) <= 1.0e-5,
            "future_reproduced": abs(live_future - archived_future) <= 1.0e-5,
        },
        "top_error_frames": top,
        "exact32_time_indices": list(exact_indices),
        "observed_indices": list(record.observed_indices),
        "bindings": {
            "config": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "r5b_checkpoint": str(checkpoint.resolve()),
            "r5b_checkpoint_sha256": R5B_CHECKPOINT_SHA256,
            "exact32_audit": str(args.exact32_audit.resolve()),
            "exact32_audit_sha256": _sha256(args.exact32_audit),
            "cpadc_summary": str(cpadc_summary_path.resolve()),
            "cpadc_summary_sha256": _sha256(cpadc_summary_path),
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
