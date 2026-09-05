#!/usr/bin/env python3
"""Read-only same-checkpoint ablation of the WKB frequency gate."""
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


def set_frequency_gate_identity(model) -> dict[str, object]:
    """Reset only the record-conditioned frequency gate to its exact identity."""

    synthesis = getattr(getattr(model, "local_field", None), "helmholtz_synthesis", None)
    gate = getattr(synthesis, "frequency_gate", None)
    if synthesis is None or not bool(getattr(synthesis, "frequency_softmax", False)):
        raise ValueError("frequency-gate ablation requires enabled Helmholtz softmax")
    if not isinstance(gate, torch.nn.Linear):
        raise ValueError("frequency-gate ablation requires the conditioned linear gate")
    before = {
        "weight_nonzero": int(torch.count_nonzero(gate.weight)),
        "bias_nonzero": int(torch.count_nonzero(gate.bias)),
        "weight_absmax": float(gate.weight.detach().abs().max()),
        "bias_absmax": float(gate.bias.detach().abs().max()),
    }
    with torch.no_grad():
        gate.weight.zero_()
        gate.bias.zero_()
    return {
        "schema": "frequency_gate_identity_ablation_v1",
        "before": before,
        "after": {
            "weight_nonzero": int(torch.count_nonzero(gate.weight)),
            "bias_nonzero": int(torch.count_nonzero(gate.bias)),
        },
        "other_parameters_modified": False,
    }


def relative_improvement(baseline: float, candidate: float) -> float:
    value = float(baseline)
    if value <= 0.0:
        raise ValueError("relative-improvement baseline must be positive")
    return (value - float(candidate)) / value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--identity", required=True)
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
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "result.json"
    terminal_path = output_dir / "terminal.json"
    if terminal_path.exists():
        print(terminal_path.read_text().strip())
        return 0
    if not checkpoint_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("checkpoint or identity is missing")

    identity = json.loads(identity_path.read_text())
    if not isinstance(identity, dict) or identity.get("schema") != "saved_time_capacity_ladder_overfit_v1":
        raise ValueError("unsupported parent run identity")
    if identity.get("split") != "train" or not bool(identity.get("helmholtz_frequency_softmax")):
        raise ValueError("ablation is restricted to a train-only frequency-gate run")

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
        raise ValueError("active manifest digest differs from checkpoint identity")
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
    config["loss"]["hard_causality"] = bool(
        identity["hard_causality_postprocessing"]
    )
    indices = tuple(int(value) for value in identity["record_indices"])
    frame_count = 401

    learned = _evaluate_triplet(
        model,
        base,
        manifest,
        normalizer,
        device,
        config,
        indices,
        split="train",
        time_policy="all_saved",
        frames_per_record=frame_count,
    )
    ablation = set_frequency_gate_identity(model)
    identity_gate = _evaluate_triplet(
        model,
        base,
        manifest,
        normalizer,
        device,
        config,
        indices,
        split="train",
        time_policy="all_saved",
        frames_per_record=frame_count,
    )

    learned_family = learned["family_relative_l2"]
    identity_family = identity_gate["family_relative_l2"]
    effects = {
        "aggregate_relative_improvement_from_learned_gate": relative_improvement(
            identity_gate["aggregate_relative_l2"], learned["aggregate_relative_l2"]
        ),
        "family_relative_improvement_from_learned_gate": {
            family: relative_improvement(identity_family[family], learned_family[family])
            for family in ("uniform", "layered", "marmousi")
        },
    }
    payload = {
        "schema": "frequency_gate_same_checkpoint_ablation_v1",
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "identity": str(identity_path),
        "identity_sha256": _sha256(identity_path),
        "transfer": transfer,
        "ablation": ablation,
        "learned_gate_metrics": learned,
        "identity_gate_metrics": identity_gate,
        "effects": effects,
        "access_audit": {
            "split": "train",
            "sample_ids": identity["sample_ids"],
            "stored_frames_per_record": frame_count,
            "validation_opened": False,
            "test_id_opened": False,
        },
    }
    _atomic_json(payload, result_path)
    terminal = {
        "schema": "frequency_gate_same_checkpoint_ablation_terminal_v1",
        "status": "complete",
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "effects": effects,
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(terminal, terminal_path)
    print(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
