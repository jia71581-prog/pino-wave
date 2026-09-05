from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from saved_time_phase_operator_v4.instance_adaptation.pretrained_subspace import (
    TemporalLatentInputCapture,
    pretrained_temporal_latent_basis,
    pretrained_temporal_latent_probe_basis,
    temporal_latent_state_sha256,
    train_only_scalar_multiplier_oracle,
)
from saved_time_phase_operator_v4.local_field import _ContinuousTemporalLatentBasis
from scripts.run_pretrained_temporal_subspace_adaptation import (
    PTLSA_RESCUE_MODEL_SCHEMA,
    PTLSA_RESCUE_RISK_SCHEMA,
    PTLSA_SCHEMA,
    _load_risk_calibration,
    _load_resumable_report,
    _risk_decision,
)


def _inputs():
    torch.manual_seed(71)
    module = _ContinuousTemporalLatentBasis(8, rank=4, harmonics=2, gate_init=0.03)
    conditioning = torch.randn(1, 8, 9, 9)
    times = torch.linspace(0.0, 0.07, 8).unsqueeze(0)
    source = torch.tensor([[20.0, 20.0, 12.0, 0.01, 1.0]])
    return module, conditioning, times, source


def test_capture_reassembles_time_blocks_and_binds_pretrained_state():
    module, conditioning, times, source = _inputs()
    with TemporalLatentInputCapture(module) as capture:
        module(conditioning, times[:, :3], source, domain_t_s=0.07)
        module(conditioning, times[:, 3:], source, domain_t_s=0.07)
    captured = capture.finalize(expected_time_s=times)
    assert captured.time_s.shape == (1, 8)
    assert len(temporal_latent_state_sha256(module)) == 64


def test_pretrained_basis_is_normalized_and_strictly_post_observation():
    module, conditioning, times, source = _inputs()
    with TemporalLatentInputCapture(module) as capture:
        module(conditioning, times, source, domain_t_s=0.07)
    captured = capture.finalize(expected_time_s=times)
    parent = torch.randn(1, 8, 9, 9)
    observed = parent[:, 1:3].clone()
    basis = pretrained_temporal_latent_basis(
        module,
        captured,
        parent,
        observed,
        (1, 2),
        ramp_steps=3,
    )
    materialized = basis.materialize()
    assert basis.rank == 4
    assert torch.count_nonzero(materialized[:, :, :3]) == 0
    probe = pretrained_temporal_latent_probe_basis(basis)
    assert torch.count_nonzero(probe.materialize(time_indices=(1, 2))) > 0
    assert torch.count_nonzero(basis.materialize(time_indices=(1, 2))) == 0
    assert torch.isfinite(materialized).all()
    spatial_rms = basis.spatial_modes.square().flatten(2).mean(dim=2).sqrt()
    torch.testing.assert_close(spatial_rms, torch.ones_like(spatial_rms), atol=1e-5, rtol=1e-5)


def test_train_only_scalar_oracle_recovers_sealed_correction_scale():
    truth = torch.zeros(1, 7, 3, 2)
    truth[:, 3:] = 2.0
    parent = torch.zeros_like(truth)
    adapted = parent.clone()
    adapted[:, 3:] = 4.0
    oracle = train_only_scalar_multiplier_oracle(
        parent,
        adapted,
        truth,
        (1, 2),
        minimum_multiplier=-1.0,
        maximum_multiplier=1.0,
        steps=81,
    )
    assert oracle.relative_l2.shape == (1, 81)
    torch.testing.assert_close(
        oracle.best_multiplier, torch.tensor([0.5], dtype=torch.float64)
    )
    torch.testing.assert_close(
        oracle.parent_relative_l2, torch.ones(1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        oracle.best_relative_l2, torch.zeros(1, dtype=torch.float64)
    )


def test_resume_loads_only_complete_identity_matched_record(tmp_path):
    record = SimpleNamespace(
        sample_id="test_id_layered_00000",
        medium_type="layered",
        input_digest="input-sha",
        observed_indices=(7, 8),
    )
    parent = {"checkpoint_sha256": "parent-sha"}
    adaptation = {
        "schema": PTLSA_SCHEMA,
        "sample_id": record.sample_id,
        "input_digest": record.input_digest,
        "parent": parent,
        "risk_calibration": None,
        "observed_probe_weight": 100.0,
        "apply_families": ("layered",),
        "causal_ramp_steps": 4,
        "maximum_correction_ratio": 0.1,
        "accessed_true_indices": record.observed_indices,
        "future_truth_used": False,
    }
    artifact = tmp_path / record.sample_id
    artifact.mkdir()
    torch.save({"adaptation": adaptation}, artifact / "adaptation.pt")
    report = {
        "sample_id": record.sample_id,
        "medium_type": record.medium_type,
        "adaptation": adaptation,
    }
    (artifact / "evaluation.json").write_text(json.dumps(report))
    loaded = _load_resumable_report(
        artifact,
        record,
        parent_identity=parent,
        risk_identity=None,
        observed_probe_weight=100.0,
        enabled_families={"layered"},
        causal_ramp_steps=4,
        maximum_correction_ratio=0.1,
    )
    assert loaded["sample_id"] == record.sample_id

    (artifact / "evaluation.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete resumable record"):
        _load_resumable_report(
            artifact,
            record,
            parent_identity=parent,
            risk_identity=None,
            observed_probe_weight=100.0,
            enabled_families={"layered"},
            causal_ramp_steps=4,
            maximum_correction_ratio=0.1,
        )


def test_linear_rescue_risk_is_parent_bound_and_uses_only_online_features(tmp_path):
    parent = {"checkpoint_sha256": "parent-sha"}
    checkpoint = tmp_path / "rescue.json"
    checkpoint.write_text(
        json.dumps(
            {
                "schema": PTLSA_RESCUE_RISK_SCHEMA,
                "future_truth_scope": "train_split_after_adaptation_seal_only",
                "family": "layered",
                "record_count": 96,
                "parent": parent,
                "base_gate": {"maximum_coefficient_l2_norm": 0.1},
                "rescue_model": {
                    "schema": PTLSA_RESCUE_MODEL_SCHEMA,
                    "features": [
                        "solver_coefficient_l2_norm",
                        "objective_relative_improvement",
                    ],
                    "feature_means": [0.1, 0.2],
                    "feature_scales": [0.01, 0.1],
                    "weights": [0.0, -1.0, 1.0],
                    "minimum_score": 0.5,
                },
            }
        )
    )
    identity = _load_risk_calibration(
        checkpoint, parent_identity=parent, enabled_families={"layered"}
    )

    safe, decision = _risk_decision(
        identity,
        family="layered",
        solver_coefficient_l2_norm=0.09,
        objective_before=10.0,
        objective_after=9.0,
    )
    assert safe and decision["route"] == "base"

    safe, decision = _risk_decision(
        identity,
        family="layered",
        solver_coefficient_l2_norm=0.11,
        objective_before=10.0,
        objective_after=6.0,
    )
    assert safe and decision["route"] == "rescue"

    safe, decision = _risk_decision(
        identity,
        family="layered",
        solver_coefficient_l2_norm=0.11,
        objective_before=10.0,
        objective_after=8.0,
    )
    assert not safe and decision["route"] == "abstain"

    with pytest.raises(ValueError, match="parent identity mismatch"):
        _load_risk_calibration(
            checkpoint,
            parent_identity={"checkpoint_sha256": "other-parent"},
            enabled_families={"layered"},
        )
