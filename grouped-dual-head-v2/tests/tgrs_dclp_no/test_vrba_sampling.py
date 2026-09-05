from __future__ import annotations

import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.vrba_sampling import (
    VRBA_POTENTIALS,
    lambda_upper_bound,
    vrba_frame_weights,
    vrba_record_pdf,
)

_MONOTONE_POTENTIALS = ("sublinear", "quadratic", "lp", "exponential", "logarithmic")


# --------------------------------------------------------------------------- #
# (1) Boundedness: lambda never exceeds eta / (1 - gamma) at steady state.
# --------------------------------------------------------------------------- #
def test_frame_lambda_respects_upper_bound_under_repeated_updates():
    gamma, eta = 0.999, 0.01
    bound = lambda_upper_bound(gamma=gamma, eta=eta)
    T = 16
    lam = torch.zeros(T, dtype=torch.float64)
    # Worst case: maximal residual every step drives lambda_it toward its max (1).
    residual = torch.linspace(0.1, 5.0, T, dtype=torch.float64)
    for it in range(20000):
        lam = vrba_frame_weights(
            residual, lam, gamma=gamma, eta=eta, phi=0.9, potential="quadratic", iteration=it
        )
    assert torch.all(lam <= bound + 1.0e-9)
    # The hottest frame should have converged essentially to the bound.
    assert float(lam.max()) == pytest.approx(bound, rel=1.0e-3)


def test_frame_lambda_bound_holds_across_all_potentials():
    gamma, eta = 0.99, 0.05
    bound = lambda_upper_bound(gamma=gamma, eta=eta)
    residual = torch.tensor([0.01, 0.3, 1.0, 4.0], dtype=torch.float64)
    for potential in VRBA_POTENTIALS:
        lam = torch.zeros(4, dtype=torch.float64)
        for it in range(3000):
            lam = vrba_frame_weights(
                residual, lam, gamma=gamma, eta=eta, phi=1.0, potential=potential, iteration=it
            )
        assert torch.all(lam <= bound + 1.0e-9), potential


# --------------------------------------------------------------------------- #
# (2) Potential monotonicity: larger residual -> larger (or equal) weight.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("potential", _MONOTONE_POTENTIALS)
def test_potential_is_nondecreasing_in_residual(potential):
    residual = torch.tensor([0.05, 0.2, 0.8, 1.5, 3.0], dtype=torch.float64)
    lam = vrba_frame_weights(
        residual,
        torch.zeros_like(residual),
        gamma=0.9,
        eta=0.1,
        phi=1.0,
        potential=potential,
        iteration=5,
    )
    diffs = lam[1:] - lam[:-1]
    assert torch.all(diffs >= -1.0e-9), (potential, lam)
    # Strictly hotter frame gets strictly more weight for these potentials.
    assert float(lam[-1]) > float(lam[0])


def test_linear_potential_is_flat_regardless_of_residual():
    residual = torch.tensor([0.05, 0.9, 4.0], dtype=torch.float64)
    lam = vrba_frame_weights(
        residual, torch.zeros_like(residual), gamma=0.9, eta=0.1, phi=1.0, potential="linear"
    )
    assert torch.allclose(lam, lam[0].expand_as(lam), atol=1.0e-12)


# --------------------------------------------------------------------------- #
# (3) Degeneracy: eta=0 freezes; uniform_fraction=1 -> uniform PDF.
# --------------------------------------------------------------------------- #
def test_eta_zero_freezes_frame_weights():
    prev = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    residual = torch.tensor([5.0, 0.01, 3.0, 0.5], dtype=torch.float64)
    lam = vrba_frame_weights(residual, prev, gamma=0.5, eta=0.0, potential="quadratic")
    # gamma * prev + 0 * lambda_it -> pure decay, residual has zero influence.
    assert torch.allclose(lam, 0.5 * prev, atol=1.0e-12)


def test_uniform_fraction_one_gives_uniform_pdf():
    scores = torch.tensor([0.01, 2.0, 9.0, 0.3, 4.0], dtype=torch.float64)
    pdf = vrba_record_pdf(scores, potential="quadratic", uniform_fraction=1.0)
    assert np.allclose(pdf, np.full(5, 1.0 / 5), atol=1.0e-12)


def test_zero_scores_fall_back_to_uniform_pdf():
    pdf = vrba_record_pdf(torch.zeros(7, dtype=torch.float64), potential="quadratic")
    assert np.allclose(pdf, np.full(7, 1.0 / 7), atol=1.0e-12)


# --------------------------------------------------------------------------- #
# (4) PDF is a valid probability vector, and monotone in the scores.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("potential", VRBA_POTENTIALS)
def test_record_pdf_is_normalized_nonnegative(potential):
    scores = torch.tensor([0.02, 0.5, 1.3, 4.7, 0.9, 2.1], dtype=torch.float64)
    pdf = vrba_record_pdf(scores, potential=potential, uniform_fraction=0.1)
    assert pdf.shape == (6,)
    assert np.all(pdf >= 0.0)
    assert np.all(np.isfinite(pdf))
    assert float(pdf.sum()) == pytest.approx(1.0, abs=1.0e-9)


def test_record_pdf_prefers_high_error_records():
    scores = torch.tensor([0.1, 0.1, 0.1, 5.0], dtype=torch.float64)
    pdf = vrba_record_pdf(scores, potential="quadratic", uniform_fraction=0.2)
    # The high-error record (index 3) must get the largest mass.
    assert int(np.argmax(pdf)) == 3
    assert pdf[3] > pdf[0]


def test_uniform_fraction_interpolates_toward_uniform():
    scores = torch.tensor([0.1, 0.1, 0.1, 5.0], dtype=torch.float64)
    hot = vrba_record_pdf(scores, potential="quadratic", uniform_fraction=0.0)
    warm = vrba_record_pdf(scores, potential="quadratic", uniform_fraction=0.5)
    # Mixing toward uniform pulls the peak record's mass down.
    assert warm[3] < hot[3]
    assert warm[0] > hot[0]


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
def test_rejects_shape_mismatch_and_nonfinite():
    with pytest.raises(ValueError):
        vrba_frame_weights(torch.zeros(4), torch.zeros(3))
    bad = torch.tensor([0.0, float("nan"), 1.0])
    with pytest.raises(ValueError):
        vrba_frame_weights(bad, torch.zeros(3))
    with pytest.raises(ValueError):
        vrba_record_pdf(torch.tensor([[1.0, 2.0]]))  # not 1-D
    with pytest.raises(ValueError):
        vrba_frame_weights(torch.zeros(4), torch.zeros(4), gamma=1.0)  # gamma out of range


def test_unknown_potential_raises():
    with pytest.raises(ValueError):
        vrba_record_pdf(torch.ones(3), potential="mystery")
