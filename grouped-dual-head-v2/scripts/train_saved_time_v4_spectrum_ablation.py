#!/usr/bin/env python
"""Compare plain and band-refined deep-phase V4 with batch 12."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from saved_time_phase_operator_v4.probe import probe_variants
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_probe import _atomic_json, run_variant


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    base = V3Config.from_yaml(config["base_config"])
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(manifest, {"train": base.data.expected_train_records, "validation": base.data.expected_validation_records})
    normalizer = load_normalizer(base, manifest.digest)
    variant = probe_variants()["deep_phase"]
    results = {}
    weights = {
        "deep_phase_plain": float(config["ablation"]["baseline_spectrum_weight"]),
        "deep_phase_spectrum": float(config["ablation"]["refined_spectrum_weight"]),
    }
    for name, weight in weights.items():
        active = {**config, "loss": {**config["loss"], "spectrum": weight}}
        results[name] = run_variant(
            name, variant, active, base, manifest, normalizer,
            device=torch.device("cuda"), smoke_updates=args.smoke_updates,
        )
    plain = float(results["deep_phase_plain"]["metrics"]["aggregate_floored_relative_l2"])
    refined = float(results["deep_phase_spectrum"]["metrics"]["aggregate_floored_relative_l2"])
    improvement = (plain - refined) / plain
    selected = "deep_phase_spectrum" if improvement >= float(config["ablation"]["minimum_relative_improvement"]) else None
    summary = {"status": "complete", "selected": selected, "relative_improvement": improvement, "results": results}
    suffix = "smoke_summary.json" if args.smoke_updates else "ablation_summary.json"
    _atomic_json(summary, Path(config["artifact_dir"]) / suffix)
    print(json.dumps(summary, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
