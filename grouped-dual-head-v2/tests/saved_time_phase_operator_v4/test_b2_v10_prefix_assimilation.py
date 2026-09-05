from __future__ import annotations

import pytest
import torch
from pathlib import Path

from saved_time_phase_operator_v4.instance_adaptation.b2_v10_prefix_assimilation import (
    CausalPrefixAccessAudit,
    PrefixAssimilationConfig,
    assimilate_prefix_pod,
    source_cycle_prefix_count,
)
from scripts.pretrain_b2_v10_residual_pod import _fit_pod_memory_bounded
from scripts.evaluate_b2_v10_prefix_assimilation import _observed_count
from scripts.select_b2_v9_parent import select_parent
from scripts.select_b2_v10_arm import select_arm


def test_source_cycle_prefix_is_physical_and_leaves_future():
    time_s = torch.arange(64, dtype=torch.float64) * 0.0025
    count = source_cycle_prefix_count(
        time_s,
        source_t0_s=0.025,
        source_f0_hz=20.0,
        after_peak_cycles=0.5,
        minimum_frames=8,
    )
    assert count == 21
    assert 8 <= count < len(time_s)
    capped = source_cycle_prefix_count(
        time_s,
        source_t0_s=1.0,
        source_f0_hz=8.0,
        after_peak_cycles=1.0,
        minimum_frames=8,
        maximum_frames=49,
    )
    assert capped == 49


def test_prefix_access_audit_rejects_future_truth():
    audit = CausalPrefixAccessAudit(total_frames=12, observed_count=5)
    assert audit.read((0, 4)) == (0, 4)
    with pytest.raises(PermissionError):
        audit.read((5,))
    assert audit.payload()["future_truth_used"] is False


def test_informative_prefix_recovers_future_low_rank_residual():
    generator = torch.Generator().manual_seed(20260901)
    parent = torch.randn(1, 10, 1, 5, 4, generator=generator)
    modes = torch.randn(2, 10, 1, 5, 4, generator=generator)
    true_coefficients = torch.tensor([0.25, -0.15])
    correction = torch.einsum("r,rtczx->tczx", true_coefficients, modes)[None]
    truth = parent + correction
    observed_count = 6
    candidate, report = assimilate_prefix_pod(
        parent,
        modes,
        truth[:, :observed_count],
        config=PrefixAssimilationConfig(
            anchor_frames=2,
            ridge_fraction=1.0e-10,
            trust_ratio=10.0,
        ),
    )
    assert report["accepted"] is True
    assert report["future_truth_used"] is False
    assert report["observed_gain"] > 0.999999
    assert torch.equal(candidate[:, :observed_count], parent[:, :observed_count])
    assert torch.allclose(candidate[:, observed_count:], truth[:, observed_count:], atol=1e-5)


def test_anchor_only_prefix_is_exact_parent_rollback():
    parent = torch.randn(1, 8, 1, 3, 3)
    modes = torch.randn(2, 8, 1, 3, 3)
    candidate, report = assimilate_prefix_pod(
        parent,
        modes,
        parent[:, :2],
        config=PrefixAssimilationConfig(anchor_frames=2),
    )
    assert torch.equal(candidate, parent)
    assert report["accepted"] is False
    assert report["reason"] == "no_informative_frames_after_parent_anchor"


def test_trust_region_limits_only_future_correction():
    parent = torch.ones(1, 9, 1, 3, 3)
    modes = torch.ones(1, 9, 1, 3, 3)
    truth = parent + 10.0 * modes[None, 0]
    observed_count = 5
    config = PrefixAssimilationConfig(
        anchor_frames=2,
        ridge_fraction=1.0e-10,
        trust_ratio=0.02,
    )
    candidate, report = assimilate_prefix_pod(
        parent,
        modes,
        truth[:, :observed_count],
        config=config,
    )
    assert report["accepted"] is True
    assert report["correction_ratio"] <= config.trust_ratio * 1.000001
    assert torch.equal(candidate[:, :observed_count], parent[:, :observed_count])
    ratio = (
        (candidate[:, observed_count:] - parent[:, observed_count:]).norm()
        / parent[:, observed_count:].norm()
    )
    assert float(ratio) <= config.trust_ratio * 1.000001


def test_zero_new_prefix_residual_abstains_exactly():
    parent = torch.randn(1, 9, 1, 3, 3)
    modes = torch.randn(2, 9, 1, 3, 3)
    candidate, report = assimilate_prefix_pod(
        parent,
        modes,
        parent[:, :5],
        config=PrefixAssimilationConfig(anchor_frames=2),
    )
    assert torch.equal(candidate, parent)
    assert report["accepted"] is False
    assert report["reason"] == "zero_observed_parent_residual"


def test_minimum_observed_frame_guard_abstains_before_solve():
    parent = torch.randn(1, 12, 1, 3, 3)
    modes = torch.randn(2, 12, 1, 3, 3)
    observed = parent[:, :6] + 0.1
    candidate, report = assimilate_prefix_pod(
        parent,
        modes,
        observed,
        config=PrefixAssimilationConfig(
            anchor_frames=2,
            minimum_observed_frames=7,
        ),
    )
    assert torch.equal(candidate, parent)
    assert report["accepted"] is False
    assert report["reason"] == "insufficient_observed_frames"
    assert report["minimum_observed_frames"] == 7


def test_memory_bounded_pod_recovers_rank_two_family():
    generator = torch.Generator().manual_seed(1801)
    modes_true = torch.randn(2, 6, 1, 4, 3, generator=generator)
    coefficients = torch.randn(7, 2, generator=generator)
    residuals = torch.einsum("nr,rtczx->ntczx", coefficients, modes_true)
    modes, eigenvalues = _fit_pod_memory_bounded(
        residuals, rank=2, feature_chunk=11
    )
    fitted = residuals.reshape(len(residuals), -1) @ modes.reshape(2, -1).T
    reconstructed = torch.einsum("nr,rtczx->ntczx", fitted, modes)
    assert (residuals - reconstructed).norm() / residuals.norm() < 1.0e-5
    assert bool((eigenvalues > 0.0).all())


def test_four_online_arms_have_frozen_observation_contracts():
    policy = {
        "anchor_frames": 8,
        "maximum_observed_frames": 49,
        "arms": {
            "peak": {"after_peak_cycles": 0.0},
            "cycle025": {"after_peak_cycles": 0.25},
            "cycle050": {"after_peak_cycles": 0.5},
            "fixed24": {"fixed_frames": 24},
        },
    }
    row = {"source_t0_s": 0.025, "source_f0_hz": 20.0}
    time_s = torch.arange(64, dtype=torch.float64) * 0.0025
    counts = {
        arm: _observed_count(arm, time_s, row, policy) for arm in policy["arms"]
    }
    assert counts == {"peak": 11, "cycle025": 16, "cycle050": 21, "fixed24": 24}
    assert counts["peak"] < counts["cycle025"] < counts["cycle050"]


def test_evaluator_seals_candidate_before_future_suffix_read():
    source = Path("scripts/evaluate_b2_v10_prefix_assimilation.py").read_text()
    seal = source.index("candidate_sha256 = _sha256(candidate_path)")
    future_read = source.index('cache["target"][index : index + 1, observed_count:]')
    assert seal < future_read


def test_v9_parent_selection_uses_two_seed_minimax_gate():
    lanes = {
        "physics_cond_s372": {
            "seed": 372,
            "baseline_aggregate": 0.13,
            "best_aggregate": 0.120,
            "checkpoint": "physics372.pt",
            "checkpoint_sha256": "p372",
        },
        "physics_cond_s733": {
            "seed": 733,
            "baseline_aggregate": 0.13,
            "best_aggregate": 0.129,
            "checkpoint": "physics733.pt",
            "checkpoint_sha256": "p733",
        },
        "spectral_s372": {
            "seed": 372,
            "baseline_aggregate": 0.13,
            "best_aggregate": 0.125,
            "checkpoint": "spectral372.pt",
            "checkpoint_sha256": "s372",
        },
        "spectral_s733": {
            "seed": 733,
            "baseline_aggregate": 0.13,
            "best_aggregate": 0.126,
            "checkpoint": "spectral733.pt",
            "checkpoint_sha256": "s733",
        },
    }
    decision = select_parent(lanes)
    assert decision["status"] == "passed"
    assert decision["selected"]["variant"] == "spectral"
    assert decision["selected"]["seed"] == 372


def test_online_arm_selection_uses_common_tail_then_information():
    arms = {
        "peak": {
            "passed": True,
            "adapted_common_tail_aggregate": 0.110,
            "observed_count": {"mean": 12.0},
            "adapted_aggregate": 0.12,
            "parent_aggregate": 0.13,
            "parent_common_tail_aggregate": 0.13,
        },
        "cycle025": {
            "passed": True,
            "adapted_common_tail_aggregate": 0.100,
            "observed_count": {"mean": 18.0},
            "adapted_aggregate": 0.11,
            "parent_aggregate": 0.13,
            "parent_common_tail_aggregate": 0.13,
        },
        "cycle050": {
            "passed": True,
            "adapted_common_tail_aggregate": 0.100,
            "observed_count": {"mean": 25.0},
            "adapted_aggregate": 0.105,
            "parent_aggregate": 0.13,
            "parent_common_tail_aggregate": 0.13,
        },
        "fixed24": {
            "passed": False,
            "adapted_common_tail_aggregate": 0.09,
            "observed_count": {"mean": 24.0},
            "adapted_aggregate": 0.14,
            "parent_aggregate": 0.13,
            "parent_common_tail_aggregate": 0.13,
        },
    }
    decision = select_arm(arms)
    assert decision["status"] == "passed"
    assert decision["selected"]["arm"] == "cycle025"
