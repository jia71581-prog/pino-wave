import pytest
import torch

from grouped_ufno_mionet_v2.losses import (
    dual_head_losses,
    inverse_probability_weights,
    spatial_fft_loss,
    trace_fft_loss,
    query_data_loss,
)


def test_zero_baseline_and_perfect_prediction_are_distinct():
    target = torch.randn(2, 8, 9, 9)
    trace = target[:, :, :1, 0].transpose(1, 2)
    perfect = dual_head_losses(target, target, trace, trace)
    zero = dual_head_losses(torch.zeros_like(target), target, torch.zeros_like(trace), trace)
    assert perfect.total.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(zero.total) and zero.total > 0
    assert zero.dense_relative_l2 > 0 and zero.trace_relative_l2 > 0


def test_spectral_losses_reject_flat_random_queries():
    with pytest.raises(ValueError, match="structured"):
        spatial_fft_loss(torch.randn(2, 1024), torch.randn(2, 1024))
    with pytest.raises(ValueError, match="receiver"):
        trace_fft_loss(torch.randn(2, 1024), torch.randn(2, 1024))


def test_inverse_probability_weights_are_normalized_per_record():
    probability = torch.tensor([[0.1, 0.2, 0.4], [0.25, 0.25, 0.5]])
    weight = inverse_probability_weights(probability)
    torch.testing.assert_close(weight.mean(dim=1), torch.ones(2))
    assert weight[0, 0] > weight[0, 2]


def test_losses_have_finite_gradients_and_coherent_axes():
    dense_target = torch.randn(2, 4, 7, 9)
    trace_target = torch.randn(2, 3, 21)
    dense_pred = torch.zeros_like(dense_target, requires_grad=True)
    trace_pred = torch.zeros_like(trace_target, requires_grad=True)
    loss = dual_head_losses(dense_pred, dense_target, trace_pred, trace_target)
    loss.total.backward()
    assert torch.isfinite(dense_pred.grad).all()
    assert torch.isfinite(trace_pred.grad).all()


def test_sparse_wave_zero_prediction_is_not_diluted_by_background():
    dense_target = torch.zeros(1, 2, 33, 33); dense_target[0, 1, 16, 16] = 1
    trace_target = torch.zeros(1, 2, 21); trace_target[0, 0, 10] = 1
    zero = dual_head_losses(torch.zeros_like(dense_target), dense_target,
                            torch.zeros_like(trace_target), trace_target)
    assert zero.total > 1.0
    query_target = torch.zeros(1, 1024); query_target[0, 17] = 1
    query, relative = query_data_loss(torch.zeros_like(query_target), query_target,
                                      torch.full_like(query_target, 1 / 1024))
    assert relative == pytest.approx(1.0)
    assert query > 0.5
