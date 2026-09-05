import torch

from saved_time_phase_operator_v4.instance_adaptation.b2_v6_pod import (
    apply_pod_correction,
    fit_residual_pod,
    fit_ridge_prior,
    pod_coefficients,
    predict_ridge_prior,
)


def test_pod_reconstructs_rank_two_residual_family():
    a = torch.randn(2, 5, 1, 4, 4)
    residuals = torch.stack((a[0], a[1], a[0] + a[1], 2 * a[0] - a[1]))
    modes, _ = fit_residual_pod(residuals, 2)
    coeff = pod_coefficients(residuals, modes)
    reconstructed = torch.einsum("nr,rtczx->ntczx", coeff, modes)
    assert (residuals - reconstructed).norm() / residuals.norm() < 1.0e-5


def test_ridge_prior_fits_affine_coefficients():
    x = torch.randn(20, 3)
    y = x @ torch.tensor([[1.0, 0.0], [0.0, 2.0], [0.5, -1.0]]) + 0.2
    bundle = fit_ridge_prior(x, y, ridge=1.0e-8)
    assert torch.allclose(predict_ridge_prior(bundle, x), y, atol=1.0e-4)


def test_zero_pod_coefficients_are_exact_parent():
    parent = torch.randn(1, 6, 1, 5, 5)
    modes = torch.randn(3, 6, 1, 5, 5)
    assert torch.equal(apply_pod_correction(parent, modes, torch.zeros(3)), parent)
