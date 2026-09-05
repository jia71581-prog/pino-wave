import pytest

from saved_time_phase_operator_v4.family_gradients import (
    homogeneous_family_scale,
    inverse_norm_family_weights,
)


def test_inverse_norm_weights_have_unit_mean_and_equalize_gradient_scale():
    norms = {"uniform": 30.0, "layered": 90.0, "marmousi": 10.0}

    weights = inverse_norm_family_weights(norms)

    assert sum(weights.values()) / 3.0 == pytest.approx(1.0)
    scaled = {name: norms[name] * weights[name] for name in norms}
    assert len({round(value, 7) for value in scaled.values()}) == 1


def test_inverse_norm_weights_require_exact_positive_three_family_evidence():
    with pytest.raises(ValueError, match="three families"):
        inverse_norm_family_weights({"uniform": 1.0})
    with pytest.raises(ValueError, match="positive"):
        inverse_norm_family_weights(
            {"uniform": 1.0, "layered": 0.0, "marmousi": 1.0}
        )


def test_homogeneous_family_scale_rejects_mixed_microbatches():
    weights = {"uniform": 0.5, "layered": 0.25, "marmousi": 2.25}

    assert homogeneous_family_scale(("uniform", "uniform"), weights) == 0.5
    with pytest.raises(ValueError, match="homogeneous"):
        homogeneous_family_scale(("uniform", "layered"), weights)


def test_equal_family_weights_allow_a_mixed_diagnostic_microbatch():
    weights = {"uniform": 1.0, "layered": 1.0, "marmousi": 1.0}

    assert homogeneous_family_scale(
        ("uniform", "layered", "marmousi"), weights
    ) == pytest.approx(1.0)
