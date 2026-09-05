#!/usr/bin/env python
"""Evaluate an optimizer-probe V6 checkpoint on the standard fixed validation panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint  # noqa: E402
from scripts.train_grouped_v3_pilot import load_normalizer  # noqa: E402
from scripts.train_saved_time_v4_full_support import (  # noqa: E402
    _atomic_json,
    _evaluate,
    _load_context,
    _load_parent_model,
    validation_panel_indices,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-records", type=int, default=48)
    parser.add_argument("--validation-frames", type=int, default=32)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    identity = json.loads(Path(args.checkpoint_identity).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("probe checkpoint evaluation requires CUDA")

    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    metadata = load_checkpoint(
        Path(args.checkpoint),
        model=model,
        optimizer=None,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(identity["run_digest"]),
        map_location=device,
    )
    normalizer = load_normalizer(base, manifest.digest)
    indices = validation_panel_indices(
        validation_records=base.data.expected_validation_records,
        panel_records=int(args.validation_records),
        epoch=1,
        seed=int(config["seed"]),
    )
    metrics = _evaluate(
        model,
        base,
        manifest,
        normalizer,
        device,
        config,
        indices,
        epoch_offset=0,
        time_policy="validation_fixed",
        frames_per_record=int(args.validation_frames),
    )
    report = {
        "status": "complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(metadata.epoch),
        "checkpoint_global_step": int(metadata.global_step),
        "validation_records": int(args.validation_records),
        "validation_frames": int(args.validation_frames),
        "metrics": metrics,
    }
    _atomic_json(report, Path(args.output))
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
