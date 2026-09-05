#!/usr/bin/env python3
"""Train-only source-versus-medium generalization diagnosis for r20."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.diagnose_capacity_ladder_overfit import (
    build_base_config,
    build_probe_config,
    load_exact_model_initialization,
)
from scripts.diagnose_saved_time_temporal_three_record_overfit import _evaluate_triplet
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_probe import _atomic_json, _model


FAMILIES = ("layered", "marmousi")


def build_source_medium_panels(records, anchor_sample_ids) -> dict[str, tuple[int, ...]]:
    """Build split-relative anchor, same-medium-source, and next-medium panels."""

    train_records = tuple(record for record in records if str(record.split) == "train")
    by_sample = {str(record.sample_id): index for index, record in enumerate(train_records)}
    anchor_by_family: dict[str, int] = {}
    for sample_id in anchor_sample_ids:
        if str(sample_id) not in by_sample:
            continue
        index = by_sample[str(sample_id)]
        family = str(train_records[index].medium_type)
        if family in FAMILIES:
            anchor_by_family[family] = index
    if set(anchor_by_family) != set(FAMILIES):
        raise ValueError("anchors must contain one layered and one marmousi train record")

    anchors: list[int] = []
    same_medium: list[int] = []
    new_medium: list[int] = []
    panel_groups: dict[str, set[str]] = {
        "anchor": set(),
        "same_medium_new_source": set(),
        "new_medium_new_source": set(),
    }
    for family in FAMILIES:
        anchor_index = anchor_by_family[family]
        anchor = train_records[anchor_index]
        anchors.append(anchor_index)
        panel_groups["anchor"].add(str(anchor.group_id))
        family_indices = [
            index
            for index, record in enumerate(train_records)
            if str(record.medium_type) == family
        ]
        same = [
            index
            for index in family_indices
            if str(train_records[index].group_id) == str(anchor.group_id)
            and index != anchor_index
        ]
        if not same:
            raise ValueError(f"{family} anchor medium has no held-out sources")
        same_medium.extend(same)
        panel_groups["same_medium_new_source"].add(str(anchor.group_id))

        next_group = next(
            str(train_records[index].group_id)
            for index in family_indices
            if str(train_records[index].group_id) != str(anchor.group_id)
        )
        new = [
            index
            for index in family_indices
            if str(train_records[index].group_id) == next_group
        ]
        if not new:
            raise RuntimeError(f"{family} next-medium panel is empty")
        new_medium.extend(new)
        panel_groups["new_medium_new_source"].add(next_group)

    panels = {
        "anchor": tuple(anchors),
        "same_medium_new_source": tuple(same_medium),
        "new_medium_new_source": tuple(new_medium),
    }
    if set(panels["anchor"]) & set(panels["same_medium_new_source"]):
        raise RuntimeError("anchor and held-out-source panels overlap")
    if set(panels["anchor"] + panels["same_medium_new_source"]) & set(
        panels["new_medium_new_source"]
    ):
        raise RuntimeError("anchor-medium and new-medium panels overlap")
    return panels


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--travel-time-h5",
        default=(
            "/home/jiayh/Data/data/processed/"
            "hybrid_travel_layered_eikonal_ray12_marmousi1_4m_v2.h5"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).resolve()
    identity_path = Path(args.identity).resolve()
    preregistration_path = Path(args.preregistration).resolve()
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "result.json"
    terminal_path = output_dir / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip())
        return 0
    preregistration = json.loads(preregistration_path.read_text())
    if preregistration.get("status") != "frozen_before_launch":
        raise ValueError("preregistration must be frozen before evaluation")
    if preregistration.get("candidate") != output_dir.name:
        raise ValueError("preregistration candidate does not match output directory")

    identity = json.loads(identity_path.read_text())
    if identity.get("schema") != "saved_time_capacity_ladder_overfit_v1":
        raise ValueError("unsupported checkpoint identity")
    if identity.get("split") != "train" or not identity.get("helmholtz_frequency_softmax"):
        raise ValueError("diagnosis requires the train-only r20 frequency-gate model")

    device = torch.device(args.device)
    seed = int(identity["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    base = build_base_config(
        int(identity["width"]),
        base_config=str(identity["base_config"]),
    )
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": base.data.expected_train_records,
            "validation": base.data.expected_validation_records,
        },
    )
    if manifest.digest != identity["manifest_digest"]:
        raise ValueError("manifest digest changed")
    normalizer = load_normalizer(base, manifest.digest)

    variant = ProbeVariant(
        depth=int(identity["dense_depth"]),
        use_local_phase=True,
        spectral_rank=int(identity["dense_spectral_rank"]),
        modes=int(identity["dense_modes"]),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_channel_multipliers=tuple(identity["local_field_channel_multipliers"]),
        local_field_causal_width_s=float(identity["local_field_causal_width_s"]),
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=96,
        local_field_helmholtz_synthesis_wkb_phase=True,
        local_field_helmholtz_synthesis_rank=0,
        local_field_helmholtz_synthesis_frequency_softmax=True,
    )
    model = _model(base, manifest, variant).to(device)
    model.local_field.helmholtz_apply_causal_gate = bool(
        identity["helmholtz_apply_causal_gate"]
    )
    model.dense_apply_free_surface_factor = bool(
        identity["dense_apply_free_surface_factor"]
    )
    transfer = load_exact_model_initialization(
        model,
        checkpoint_path,
        identity_path,
        manifest_digest=manifest.digest,
        device=device,
    )
    config = build_probe_config(
        dense_lr=float(identity["dense_learning_rate"]),
        backbone_lr=float(identity["backbone_learning_rate"]),
        temporal_lr=1.0e-5,
        seed=seed,
        travel_time_h5=str(Path(args.travel_time_h5).resolve()),
    )
    config["loss"]["hard_causality"] = bool(identity["hard_causality_postprocessing"])
    panels = build_source_medium_panels(manifest.records, identity["sample_ids"])
    train_records = tuple(record for record in manifest.records if record.split == "train")

    metrics: dict[str, object] = {}
    panel_records: dict[str, object] = {}
    for name, indices in panels.items():
        metrics[name] = _evaluate_triplet(
            model,
            base,
            manifest,
            normalizer,
            device,
            config,
            indices,
            split="train",
            time_policy="all_saved",
            frames_per_record=len(manifest.time_s),
            evaluation_macro_records=1,
            evaluation_time_block=32,
        )
        panel_records[name] = {
            "indices": indices,
            "sample_ids": [train_records[index].sample_id for index in indices],
            "group_ids": sorted({train_records[index].group_id for index in indices}),
            "family_counts": {
                family: sum(
                    str(train_records[index].medium_type) == family for index in indices
                )
                for family in FAMILIES
            },
        }

    threshold = float(preregistration["acceptance"]["relative_l2_max"])
    same_family = metrics["same_medium_new_source"]["family_relative_l2"]
    new_family = metrics["new_medium_new_source"]["family_relative_l2"]
    same_source_passed = all(float(same_family[family]) <= threshold for family in FAMILIES)
    new_medium_passed = all(float(new_family[family]) <= threshold for family in FAMILIES)
    if not same_source_passed:
        diagnosis = "source_conditioning_or_source_coverage_limited"
    elif not new_medium_passed:
        diagnosis = "medium_generalization_limited"
    else:
        diagnosis = "both_source_and_next_medium_panels_within_diagnostic_gate"

    payload = {
        "schema": "r20_source_medium_generalization_diagnosis_v1",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "identity": str(identity_path),
        "identity_sha256": _sha256(identity_path),
        "preregistration": str(preregistration_path),
        "preregistration_sha256": _sha256(preregistration_path),
        "transfer": transfer,
        "panels": panel_records,
        "metrics": metrics,
        "acceptance": {
            "relative_l2_max": threshold,
            "same_medium_new_source_passed": same_source_passed,
            "new_medium_new_source_passed": new_medium_passed,
        },
        "diagnosis": diagnosis,
        "access_audit": {
            "split": "train",
            "validation_opened": False,
            "test_id_opened": False,
        },
    }
    _atomic_json(payload, result_path)
    terminal = {
        "schema": "r20_source_medium_generalization_terminal_v1",
        "status": "complete",
        "diagnosis": diagnosis,
        "acceptance": payload["acceptance"],
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(terminal, terminal_path)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
