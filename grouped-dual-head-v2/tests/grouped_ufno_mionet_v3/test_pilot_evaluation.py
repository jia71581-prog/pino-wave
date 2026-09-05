from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grouped_ufno_mionet_v3.training.checkpoint import CHECKPOINT_FORMAT
from scripts.evaluate_grouped_v3_pilot import (
    active_evaluation_times,
    bilinear_sample_frames,
    representative_validation_indices,
    validate_evaluation_identity,
)


def _record(index: int, family: str, split: str = "validation"):
    return SimpleNamespace(source_index=index, medium_type=family, split=split)


def test_representative_validation_indices_selects_one_record_per_allowed_family():
    manifest = SimpleNamespace(
        records=(
            _record(3, "uniform", "train"),
            _record(8, "marmousi"),
            _record(5, "uniform"),
            _record(6, "layered"),
            _record(9, "marmousi"),
        )
    )
    assert representative_validation_indices(manifest) == {
        "uniform": 1,
        "layered": 2,
        "marmousi": 0,
    }


def test_representative_validation_indices_rejects_incomplete_census():
    manifest = SimpleNamespace(records=(_record(5, "uniform"), _record(6, "layered")))
    with pytest.raises(ValueError, match="marmousi"):
        representative_validation_indices(manifest)


def test_active_evaluation_times_include_saved_frames_and_strict_midpoints():
    axis = torch.linspace(0.0, 1.0, 101)
    exact, midpoint = active_evaluation_times(axis, source_t0_s=0.1)
    assert exact.shape == midpoint.shape == (3,)
    assert torch.all(exact >= 0.1)
    assert torch.all(exact <= 0.7)
    assert torch.all(midpoint > exact)
    for value in exact:
        assert bool(torch.isclose(axis, value, rtol=0.0, atol=1.0e-7).any())
    for value in midpoint:
        assert not bool(torch.isclose(axis, value, rtol=0.0, atol=1.0e-7).any())


def test_evaluation_identity_accepts_refinement_and_checks_checkpoint_run():
    identity = {
        "schema": "grouped_v3_stable_refinement_v1",
        "manifest_digest": "manifest",
        "run_digest": "refinement-run",
    }
    checkpoint = {
        "format": "phase_aligned_complex_fno_mionet_v3",
        "manifest_digest": "manifest",
        "config_digest": "refinement-run",
    }
    origin = validate_evaluation_identity(
        identity,
        checkpoint,
        manifest_digest="manifest",
        model_config_digest="base-config",
    )
    assert origin == "refinement-run"


def test_evaluation_accepts_verified_direct_parent_checkpoint():
    identity = {
        "schema": "grouped_v3_stable_refinement_v1",
        "manifest_digest": "manifest",
        "run_digest": "child",
        "parent": {"run_digest": "parent"},
    }
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "manifest_digest": "manifest",
        "config_digest": "parent",
    }

    origin = validate_evaluation_identity(
        identity,
        checkpoint,
        manifest_digest="manifest",
        model_config_digest="base-config",
    )

    assert origin == "parent"


def test_evaluation_rejects_unrelated_checkpoint_despite_parent_identity():
    identity = {
        "schema": "grouped_v3_stable_refinement_v1",
        "manifest_digest": "manifest",
        "run_digest": "child",
        "parent": {"run_digest": "parent"},
    }
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "manifest_digest": "manifest",
        "config_digest": "unrelated",
    }

    with pytest.raises(ValueError, match="run mismatch"):
        validate_evaluation_identity(
            identity,
            checkpoint,
            manifest_digest="manifest",
            model_config_digest="base-config",
        )


def test_pilot_evaluation_identity_still_requires_model_config_digest():
    identity = {
        "manifest_digest": "manifest",
        "pilot_config_digest": "expected-base-config",
        "run_digest": "pilot-run",
    }
    checkpoint = {
        "format": "phase_aligned_complex_fno_mionet_v3",
        "manifest_digest": "manifest",
        "config_digest": "pilot-run",
    }
    with pytest.raises(ValueError, match="model config mismatch"):
        validate_evaluation_identity(
            identity,
            checkpoint,
            manifest_digest="manifest",
            model_config_digest="wrong-base-config",
        )
def test_bilinear_sample_frames_supports_off_grid_arbitrary_points():
    x = torch.tensor([0.0, 1.0, 2.0])
    z = torch.tensor([0.0, 2.0, 4.0])
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    frames = torch.stack((xx + 2.0 * zz, 10.0 + xx + 2.0 * zz))
    points = torch.tensor([[0.5, 1.0], [1.25, 3.0]])
    sampled = bilinear_sample_frames(frames, x_m=x, z_m=z, points_xy_m=points)
    np.testing.assert_allclose(sampled.numpy(), [[2.5, 7.25], [12.5, 17.25]], atol=1e-6)
