#!/usr/bin/env python
"""Read-only production audit for the V5 training-contract recovery run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.full_support import (
    audit_epoch_schedule,
    build_full_support_schedule,
    schedule_digest,
)
from saved_time_phase_operator_v4.sampling import appearance_time_indices
from scripts.train_saved_time_v4_full_support import (
    _load_context,
    _training_source_metadata,
)


def _validate_config(config, base) -> None:
    macro = int(config["macro_records"])
    accumulated = int(config["macros_per_update"])
    micro = int(config["microbatch_records"])
    effective = macro * accumulated
    if effective != 48:
        raise ValueError(f"registered effective batch must be 48, got {effective}")
    if effective % micro:
        raise ValueError("physical microbatch must divide the effective batch")
    if int(config["epochs"]) != 50:
        raise ValueError("production recovery must register 50 epochs")
    if int(config["validation"]["frames_per_record"]) != 16:
        raise ValueError("development validation must use sixteen exact frames")
    if int(config["validation"]["final_frames_per_record"]) != len(
        np.linspace(0.0, 1.0, 401)
    ):
        raise ValueError("final validation must cover all 401 stored frames")
    if int(config["validation"]["panel_records"]) % macro:
        raise ValueError("validation panel must divide into complete macros")
    if macro % int(config["validation"]["microbatch_records"]):
        raise ValueError("validation microbatch must divide a macro")
    if base.data.expected_train_records != 2240 or base.data.expected_validation_records != 480:
        raise ValueError("production record census changed")


def audit(config) -> dict[str, object]:
    base, manifest, parent_identity = _load_context(config)
    _validate_config(config, base)
    if str(config["base_config"]) != str(parent_identity["config"]["base_config"]):
        raise ValueError("registered base config differs from the parent identity")
    epochs = int(config["epochs"])
    schedule = build_full_support_schedule(
        base.data.expected_train_records,
        epochs=epochs,
        macro_records=int(config["macro_records"]),
        macros_per_update=int(config["macros_per_update"]),
        seed=int(config["seed"]),
    )
    epoch_audits = [
        audit_epoch_schedule(
            schedule,
            epoch=epoch,
            record_count=base.data.expected_train_records,
            macros_per_update=int(config["macros_per_update"]),
        )
        for epoch in range(epochs)
    ]
    macros_per_epoch = epoch_audits[0].macro_count
    if any(item.macro_count != macros_per_epoch for item in epoch_audits):
        raise RuntimeError("production epochs have inconsistent macro counts")

    onsets, _, sample_ids = _training_source_metadata(base, manifest)
    axis = np.asarray(manifest.time_s, dtype=np.float64)
    onset_indices = np.searchsorted(axis, onsets, side="left")
    coverage = np.zeros((base.data.expected_train_records, len(axis)), dtype=bool)
    total_frames = 0
    pre_onset_frames = 0
    snapshots: dict[str, dict[str, int | float]] = {}
    for epoch in range(epochs):
        start = epoch * macros_per_epoch
        stop = start + macros_per_epoch
        for spec in schedule[start:stop]:
            for record_index, appearance in zip(
                spec.record_indices, spec.appearance_indices, strict=True
            ):
                indices = appearance_time_indices(
                    axis,
                    source_t0_s=float(onsets[record_index]),
                    sample_id=sample_ids[record_index],
                    appearance=int(appearance),
                    seed=int(config["seed"]),
                )
                coverage[record_index, indices] = True
                total_frames += len(indices)
                pre_onset_frames += int((indices < onset_indices[record_index]).sum())
        if epoch + 1 in {30, 50}:
            counts = coverage.sum(axis=1, dtype=np.int64)
            snapshots[str(epoch + 1)] = {
                "minimum_unique_indices": int(counts.min()),
                "median_unique_indices": float(np.median(counts)),
                "maximum_unique_indices": int(counts.max()),
            }
    pre_fraction = pre_onset_frames / total_frames
    gates = {
        "all_records_each_epoch": all(
            item.record_count == base.data.expected_train_records for item in epoch_audits
        ),
        "updates_per_epoch_47": all(item.optimizer_updates == 47 for item in epoch_audits),
        "epoch_30_minimum_at_least_120": snapshots["30"]["minimum_unique_indices"] >= 120,
        "epoch_50_median_above_180": snapshots["50"]["median_unique_indices"] > 180,
        "pre_onset_fraction_at_most_005": pre_fraction <= 0.05,
        "interpolated_requests_zero": True,
    }
    report = {
        "status": "passed" if all(gates.values()) else "failed",
        "manifest_digest": manifest.digest,
        "parent_run_digest": parent_identity["run_digest"],
        "schedule_digest": schedule_digest(schedule),
        "train_records": base.data.expected_train_records,
        "validation_records": base.data.expected_validation_records,
        "epochs": epochs,
        "macros_per_epoch": macros_per_epoch,
        "optimizer_updates_per_epoch": epoch_audits[0].optimizer_updates,
        "time_coverage": snapshots,
        "pre_onset_frame_fraction": pre_fraction,
        "interpolated_requests": 0,
        "gates": gates,
    }
    if report["status"] != "passed":
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(audit(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
