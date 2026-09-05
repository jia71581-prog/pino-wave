#!/usr/bin/env python3
"""Deployment-time, second-scale instance fine-tune of the frozen operator.

This is the online half of the two-stage "meta hypernetwork + deployment LoRA"
scheme (offline half: ``scripts/train_meta_hypernet.py``).  A meta-trained
onset hypernetwork is loaded and frozen; for every new validation instance only
the per-instance ``latent_delta`` (the deployment LoRA) is optimized for a
handful of steps while reading exactly the two early onset snapshots.  Later
truth is never accessed during adaptation; the sealed artifact is scored against
future ground truth afterwards by the same evaluator used elsewhere.

Delegates to :func:`scripts.run_v5_instance_adaptation.run` with
``deployment_lora=True`` so the causal audit, rollback gates, sealed evaluation,
and plotting are shared verbatim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from scripts.run_v5_instance_adaptation import (
    build_all_validation_manifest,
    build_instance_manifest,
    run,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--conditioner-checkpoint")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument(
        "--selection-split",
        choices=("train", "validation"),
        default="validation",
    )
    parser.add_argument("--per-family", type=int)
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--all-validation", action="store_true")
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    if args.parent_checkpoint:
        config["parent_checkpoint"] = args.parent_checkpoint
    if args.conditioner_checkpoint:
        config["conditioner_checkpoint"] = args.conditioner_checkpoint
    if not config.get("conditioner_checkpoint"):
        raise ValueError(
            "deployment adaptation requires a meta-trained conditioner_checkpoint; "
            "run scripts/train_meta_hypernet.py first"
        )
    manifest = build_manifest(config["source_h5"])

    excluded_group_ids: set[str] = set()
    if args.exclude_manifest:
        excluded_rows = json.loads(Path(args.exclude_manifest).read_text())
        excluded_group_ids = {str(row["group_id"]) for row in excluded_rows}
    if args.all_validation:
        if args.selection_split != "validation":
            raise ValueError("--all-validation requires --selection-split validation")
        if args.sample_id or args.exclude_manifest:
            raise ValueError("complete validation cannot combine sample or exclusion filters")
        rows = build_all_validation_manifest(manifest)
    elif args.sample_id:
        by_sample = {
            row.sample_id: row
            for row in manifest.records
            if row.split == args.selection_split
            and row.group_id not in excluded_group_ids
        }
        missing = tuple(value for value in args.sample_id if value not in by_sample)
        if missing:
            raise ValueError(
                f"requested {args.selection_split} sample IDs are missing: {missing[:3]}"
            )
        rows = tuple(by_sample[value] for value in args.sample_id)
    else:
        rows = build_instance_manifest(
            manifest,
            seed=(int(config.get("seed", 17)) if args.selection_seed is None else int(args.selection_seed)),
            per_family=((1 if args.pilot else 3) if args.per_family is None else int(args.per_family)),
            excluded_group_ids=excluded_group_ids,
            split=args.selection_split,
        )

    if args.dry_run:
        print(json.dumps({
            "mode": "deployment_lora",
            "family_counts": {family: sum(row.medium_type == family for row in rows) for family in ALLOWED_MEDIUM_TYPES},
            "sample_ids": [row.sample_id for row in rows],
            "selection_split": args.selection_split,
            "future_truth_opened": False,
            "conditioner_checkpoint": str(config["conditioner_checkpoint"]),
            "adaptation_optimizer": str(
                config.get("adaptation_optimizer", "adam_lbfgs")
            ),
        }, sort_keys=True))
        return 0

    run(
        args.config,
        device_name=args.device,
        output_dir=args.output_dir,
        pilot=args.pilot,
        parent_checkpoint=args.parent_checkpoint,
        conditioner_checkpoint=args.conditioner_checkpoint,
        sample_ids=tuple(row.sample_id for row in rows),
        selection_split=args.selection_split,
        selection_seed=args.selection_seed,
        per_family=args.per_family,
        save_fields=not args.no_fields,
        write_plots=not args.no_plots,
        deployment_lora=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
