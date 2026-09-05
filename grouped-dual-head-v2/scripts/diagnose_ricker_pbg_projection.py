#!/usr/bin/env python3
"""Train-only projection floor for an analytic retarded Ricker P_bg surrogate."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.config import load_config
from fno_acoustic.data import PinoHDF5Dataset


def _summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def evaluate_projection(
    dataset: PinoHDF5Dataset, scale: float
) -> dict[str, object]:
    error_square = 0.0
    target_square = 0.0
    record_errors: list[float] = []
    oracle_errors: list[float] = []
    oracle_scales: list[float] = []
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        target = item["target"].double()
        template = item["input"][..., -1].double()
        prediction = template * float(scale)
        error = prediction - target
        e2 = float(error.square().sum())
        y2 = float(target.square().sum())
        error_square += e2
        target_square += y2
        record_errors.append(math.sqrt(e2 / max(y2, 1.0e-30)))
        denominator = float(template.square().sum())
        oracle_scale = float((template * target).sum()) / max(denominator, 1.0e-30)
        oracle_error = template * oracle_scale - target
        oracle_errors.append(
            math.sqrt(float(oracle_error.square().sum()) / max(y2, 1.0e-30))
        )
        oracle_scales.append(oracle_scale)
    return {
        "record_count": len(dataset),
        "aggregate_relative_l2": math.sqrt(error_square / max(target_square, 1.0e-30)),
        "record_relative_l2": _summarize(record_errors),
        "oracle_record_relative_l2": _summarize(oracle_errors),
        "oracle_scale": _summarize(oracle_scales),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    split_path = Path(args.split_manifest).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    config = load_config(config_path)
    if args.data_path is not None:
        config["data"]["path"] = str(Path(args.data_path).resolve())
        config["data"]["raw_hdf5"] = str(Path(args.data_path).resolve())
    splits = json.loads(split_path.read_text())
    train = PinoHDF5Dataset(
        config, [int(value) for value in splits["train"]], return_normalized=False
    )
    calibration = PinoHDF5Dataset(
        config, [int(value) for value in splits["val"]], return_normalized=False
    )
    dot = 0.0
    template_square = 0.0
    try:
        for item_index in range(len(train)):
            item = train[item_index]
            target = item["target"].double()
            template = item["input"][..., -1].double()
            dot += float((template * target).sum())
            template_square += float(template.square().sum())
        global_scale = dot / max(template_square, 1.0e-30)
        train_report = evaluate_projection(train, global_scale)
        calibration_report = evaluate_projection(calibration, global_scale)
    finally:
        train.close()
        calibration.close()
    payload = {
        "schema": "ricker_pbg_projection_trainonly_v1",
        "config": str(config_path),
        "split_manifest": str(split_path),
        "data_path": str(Path(config["data"]["path"]).resolve()),
        "project_data_scope": "train_only",
        "confirmation_opened": False,
        "validation_opened": False,
        "test_id_opened": False,
        "global_train_scale": global_scale,
        "train": train_report,
        "calibration": calibration_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
