#!/usr/bin/env python3
"""Evaluate a family-expert candidate's exact parent on its fixed panel."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _atomic_json,
    _evaluate,
    _load_context,
    _load_parent_model,
    validation_panel_indices,
)


def _family_expert_parent_is_exact(model, config: Mapping[str, object]) -> bool:
    """Accept strict loads of existing experts and proven zero-output expansions."""

    if config.get("family_experts") is None:
        return True
    path = getattr(getattr(model, "dense_decoder", None), "family_experts", None)
    if path is None:
        return False
    transfer = getattr(model, "family_expert_transfer_report", None)
    if transfer is None:
        # No expansion prefixes were authorized, so load_checkpoint completed
        # a strict load of the already-existing family-expert parameters.
        return True
    return isinstance(transfer, Mapping) and transfer.get("exact_parent_identity") is True


def _band_adapter_parent_is_exact(model, config: Mapping[str, object]) -> bool:
    """Accept strict adapter loads and proven zero-output expansions."""

    overrides = config.get("variant_overrides", {})
    if not isinstance(overrides, Mapping):
        return False
    if int(overrides.get("band_adapter_rank", 0)) == 0:
        return True
    path = getattr(getattr(model, "dense_decoder", None), "band_limited_adapter", None)
    if path is None:
        return False
    transfer = getattr(model, "band_adapter_transfer_report", None)
    if transfer is None:
        return True
    return isinstance(transfer, Mapping) and transfer.get("exact_parent_identity") is True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-identity", type=Path)
    parser.add_argument("--validation-seed", type=int)
    args = parser.parse_args(argv)
    if (args.checkpoint is None) != (args.checkpoint_identity is None):
        parser.error("--checkpoint and --checkpoint-identity must be provided together")

    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    config = yaml.safe_load(config_path.read_text())
    if args.validation_seed is not None:
        config["seed"] = int(args.validation_seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("same-panel parent evaluation requires CUDA")

    base, manifest, parent_identity = _load_context(config)
    model = _load_parent_model(config, base, manifest, parent_identity, device)
    transfer = getattr(model, "family_expert_transfer_report", None)
    if not _family_expert_parent_is_exact(model, config):
        raise ValueError("family expert expansion does not preserve the exact parent")
    band_transfer = getattr(model, "band_adapter_transfer_report", None)
    if not _band_adapter_parent_is_exact(model, config):
        raise ValueError("band adapter expansion does not preserve the exact parent")
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else Path(str(config["parent_checkpoint"])).resolve()
    )
    checkpoint_identity = (
        args.checkpoint_identity.resolve()
        if args.checkpoint_identity is not None
        else Path(str(config["parent_checkpoint_identity"])).resolve()
    )
    direct_parent_identity = json.loads(checkpoint_identity.read_text())
    if args.checkpoint is not None:
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=None,
            expected_manifest_digest=manifest.digest,
            expected_config_digest=str(direct_parent_identity["run_digest"]),
            map_location=device,
        )

    validation = config["validation"]
    indices = validation_panel_indices(
        validation_records=base.data.expected_validation_records,
        panel_records=int(validation["panel_records"]),
        epoch=1,
        seed=int(config["seed"]),
    )
    normalizer = load_normalizer(base, manifest.digest)
    torch.cuda.reset_peak_memory_stats(device)
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
        frames_per_record=int(validation["frames_per_record"]),
    )
    report = {
        "schema": "saved_time_family_expert_same_panel_parent_v1",
        "status": "complete",
        "explicit_checkpoint_load": args.checkpoint is not None,
        "binding": {
            "config": str(config_path),
            "parent_checkpoint": str(checkpoint),
            "parent_checkpoint_sha256": sha256_file(checkpoint),
            "parent_checkpoint_identity": str(checkpoint_identity),
            "parent_checkpoint_identity_sha256": sha256_file(checkpoint_identity),
            "parent_manifest_digest": str(direct_parent_identity["manifest_digest"]),
            "active_manifest_digest": str(manifest.digest),
            "time_axis_sha256": time_axis_sha256(manifest.time_s),
            "validation_seed": int(config["seed"]),
            "validation_indices": list(indices),
            "validation_records": len(indices),
            "validation_frames_per_record": int(validation["frames_per_record"]),
        },
        "family_expert_transfer": transfer,
        "band_adapter_transfer": band_transfer,
        "metrics": metrics,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    _atomic_json(report, output)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
