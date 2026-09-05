#!/usr/bin/env python3
"""Train-only diagnostic for local weak CPADC defect moments.

This script deliberately reads future wavefields only from the training split.
It measures how much a replicated local box test function suppresses the
closure error between the fine LWC-84/CPML generator plus restriction and the
coarse second-order interior operator used by online CPADC.  No result from
validation or test_id is read or used for method selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.contracts import onset_indices
from saved_time_phase_operator_v4.instance_adaptation.forced_defect import (
    lwc84_discrete_defect,
)


def _replicated_box_rms_ratio(defect: torch.Tensor, width: int) -> float:
    test_width = int(width)
    if test_width < 1 or test_width % 2 == 0:
        raise ValueError("test-function widths must be positive odd integers")
    value = torch.as_tensor(defect).float()
    reference = value.square().mean().sqrt().clamp_min(1.0e-12)
    if test_width == 1:
        return 1.0
    frames = value.reshape(-1, 1, value.shape[-2], value.shape[-1])
    radius = test_width // 2
    averaged = F.avg_pool2d(
        F.pad(frames, (radius, radius, radius, radius), mode="replicate"),
        kernel_size=test_width,
        stride=1,
    )
    return float(averaged.square().mean().sqrt() / reference)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def diagnose(
    source_h5: str | Path,
    *,
    output: str | Path,
    per_family: int,
    seed: int,
    widths: tuple[int, ...],
    thread_count: int,
) -> Path:
    source = Path(source_h5).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if int(per_family) < 2:
        raise ValueError("per_family must be at least two")
    normalized_widths = tuple(sorted({int(value) for value in widths}))
    if 1 not in normalized_widths or any(
        value < 1 or value % 2 == 0 for value in normalized_widths
    ):
        raise ValueError("widths must include one and contain only positive odd values")
    torch.set_num_threads(max(1, int(thread_count)))

    manifest = build_manifest(source)
    generator = random.Random(int(seed))
    selected = {}
    for family in ALLOWED_MEDIUM_TYPES:
        candidates = [
            record
            for record in manifest.records
            if record.split == "train" and record.medium_type == family
        ]
        if len(candidates) < int(per_family):
            raise ValueError(f"insufficient train records for {family}")
        selected[family] = generator.sample(candidates, int(per_family))

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    with h5py.File(source, "r", swmr=True) as handle:
        time_s = torch.from_numpy(np.asarray(handle["time_s"], dtype=np.float32))
        for family in ALLOWED_MEDIUM_TYPES:
            for metadata in selected[family]:
                index = int(metadata.source_index)
                field = torch.from_numpy(
                    np.asarray(handle["wavefield"][index], dtype=np.float32)
                ).unsqueeze(0)
                velocity = torch.from_numpy(
                    np.asarray(handle["velocity_mps"][index], dtype=np.float32)
                )[None, None]
                source_parameters = torch.tensor(
                    [[
                        handle["source_x_m"][index],
                        handle["source_z_m"][index],
                        handle["source_f0_hz"][index],
                        handle["source_t0_s"][index],
                        handle["source_amplitude"][index],
                    ]],
                    dtype=torch.float32,
                )
                source_map = torch.from_numpy(
                    np.asarray(handle["source_map"][index], dtype=np.float32)
                )[None, None]
                observed = onset_indices(
                    time_s,
                    t0_s=float(source_parameters[0, 3]),
                    f0_hz=float(source_parameters[0, 2]),
                )
                record_started = time.perf_counter()
                defect, scale = lwc84_discrete_defect(
                    field,
                    velocity,
                    dt=0.0025,
                    dx=10.0,
                    dz=10.0,
                    observed_indices=observed,
                    source_parameters=source_parameters,
                    source_map=source_map,
                    time_s=time_s,
                    time_order=2,
                    normalize=True,
                    return_scale=True,
                )
                rms = float(defect.float().square().mean().sqrt())
                bins = [
                    float(block.float().square().mean().sqrt())
                    for block in torch.tensor_split(defect, 4, dim=1)
                ]
                test_ratios = {
                    str(width): _replicated_box_rms_ratio(defect, width)
                    for width in normalized_widths
                }
                records.append(
                    {
                        "sample_id": metadata.sample_id,
                        "group_id": metadata.group_id,
                        "family": family,
                        "source_index": index,
                        "observed_indices": list(observed),
                        "normalization_scale": float(scale[0]),
                        "normalized_strong_rms": rms,
                        "absolute_mean_over_rms": abs(float(defect.mean()))
                        / max(rms, 1.0e-12),
                        "test_function_rms_ratio": test_ratios,
                        "l2_normalized_test_function_rms_ratio": {
                            str(width): width * test_ratios[str(width)]
                            for width in normalized_widths
                        },
                        "time_quartile_rms": bins,
                        "elapsed_s": time.perf_counter() - record_started,
                    }
                )

    families: dict[str, object] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        rows = [row for row in records if row["family"] == family]
        families[family] = {
            "record_count": len(rows),
            "mean_normalized_strong_rms": _mean(rows, "normalized_strong_rms"),
            "mean_absolute_mean_over_rms": _mean(rows, "absolute_mean_over_rms"),
            "mean_test_function_rms_ratio": {
                str(width): sum(
                    float(row["test_function_rms_ratio"][str(width)])
                    for row in rows
                )
                / len(rows)
                for width in normalized_widths
            },
            "mean_l2_normalized_test_function_rms_ratio": {
                str(width): sum(
                    float(
                        row["l2_normalized_test_function_rms_ratio"][str(width)]
                    )
                    for row in rows
                )
                / len(rows)
                for width in normalized_widths
            },
            "mean_time_quartile_rms": [
                sum(float(row["time_quartile_rms"][index]) for row in rows)
                / len(rows)
                for index in range(4)
            ],
        }

    candidate_width = 3
    threshold = 0.9
    candidate_pass = all(
        float(
            families[family]["mean_l2_normalized_test_function_rms_ratio"][
                str(candidate_width)
            ]
        )
        <= threshold
        for family in ALLOWED_MEDIUM_TYPES
    )
    payload = {
        "schema": "cpadc_trainonly_weak_closure_diagnostic_v1",
        "source_h5": str(source),
        "manifest_digest": manifest.digest,
        "selection_split": "train",
        "future_truth_scope": "offline_train_diagnostic_only",
        "deployment_future_truth_used": False,
        "seed": int(seed),
        "per_family": int(per_family),
        "test_function_widths": list(normalized_widths),
        "candidate": {
            "width": candidate_width,
            "test_function_normalization": "discrete_l2_unit",
            "maximum_mean_l2_normalized_closure_rms_ratio_each_family": threshold,
            "precheck_pass": candidate_pass,
            "next_gate": (
                "paired_disjoint_train_accuracy_and_runtime"
                if candidate_pass
                else "reject_without_validation"
            ),
        },
        "families": families,
        "records": records,
        "elapsed_s": time.perf_counter() - started,
        "validation_access": False,
        "test_id_access": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--widths", type=int, nargs="+", default=(1, 3, 5, 9))
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()
    result = diagnose(
        arguments.source_h5,
        output=arguments.output,
        per_family=arguments.per_family,
        seed=arguments.seed,
        widths=tuple(arguments.widths),
        thread_count=arguments.threads,
    )
    print(result)


if __name__ == "__main__":
    main()
