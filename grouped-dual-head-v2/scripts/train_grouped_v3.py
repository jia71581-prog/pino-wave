#!/usr/bin/env python
"""Guarded entry point for Phase-Aligned Complex-FNO MIONet V3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.model.operator import PhaseAlignedComplexFNOMIONet
from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT


def build_model(config: V3Config) -> PhaseAlignedComplexFNOMIONet:
    model = config.model
    token_grid = round(model.token_count**0.5)
    if token_grid * token_grid != model.token_count:
        raise ValueError("model.token_count must be a perfect square")
    return PhaseAlignedComplexFNOMIONet(
        width=model.width,
        rank=model.mionet_rank,
        spectral_rank=model.spectral_rank,
        modes=model.modes,
        dense_modes=model.dense_modes,
        dense_time_block=model.dense_time_block,
        heads=model.heads,
        token_grid=token_grid,
        position_bands=model.position_bands,
        fourier_bands=model.fourier_bands,
        gabor_scales_s=model.gabor_scales_s,
        ray_samples=model.ray_samples,
        domain_x_m=model.domain_x_m,
        domain_z_m=model.domain_z_m,
        domain_t_s=model.domain_t_s,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": config.data.expected_train_records,
            "validation": config.data.expected_validation_records,
        },
    )
    model = build_model(config)
    summary = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "config_digest": config.digest(),
        "manifest_digest": manifest.digest,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "counts_after": manifest.counts_after,
        "allowed_medium_types": manifest.allowed_medium_types,
        "device": config.train.device,
    }
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        return 0
    raise RuntimeError(
        "full-dataset V3 training is gate-protected; run smoke and overfit_grouped_v3.py first"
    )


if __name__ == "__main__":
    raise SystemExit(main())
