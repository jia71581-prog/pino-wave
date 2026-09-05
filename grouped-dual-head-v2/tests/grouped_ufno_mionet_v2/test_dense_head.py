import pytest
import torch
import math

from grouped_ufno_mionet_v2.model.operator import DualHeadWaveOperator
from tests.grouped_ufno_mionet_v2.test_query_head import inputs


def dense_inputs():
    values = inputs(); values.pop("coords")
    return values


def test_dense_decoder_outputs_requested_time_block():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4, dense_time_block=8)
    times = torch.linspace(.1, .3, 8).repeat(2, 1)
    dense = model.dense_normalized(**dense_inputs(), time_s=times)
    assert dense.shape == (2, 8, 33, 33)
    assert torch.isfinite(dense).all()
    assert not torch.equal(dense[:, 0], dense[:, -1])


def test_dense_parameters_receive_gradient():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4, dense_time_block=4)
    prediction = model.dense_normalized(**dense_inputs(), time_s=torch.tensor([[.1,.2],[.1,.2]]))
    prediction.square().mean().backward()
    missing = [name for name, parameter in model.dense_decoder.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert missing == []


def test_predict_wavefield_streams_blocks_and_scales_amplitude_once():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4, dense_time_block=3).eval()
    values = dense_inputs(); times = torch.linspace(.1,.5,7).repeat(2,1)
    base = model.predict_wavefield(**values, time_s=times)
    assert base.shape == (2,7,33,33)
    values["source"] = values["source"].clone(); values["source"][:,4] = 2
    doubled = model.predict_wavefield(**values, time_s=times)
    torch.testing.assert_close(doubled, 2*base)


def test_dense_rejects_oversized_training_block():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4, dense_time_block=3)
    with pytest.raises(ValueError, match="time block"):
        model.dense_normalized(**dense_inputs(), time_s=torch.linspace(.1,.5,4).repeat(2,1))


def test_dense_head_has_global_source_relative_geometry():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4, dense_time_block=2)
    prediction = model.dense_normalized(**dense_inputs(), time_s=torch.tensor([[.2137], [.3471]]))
    prediction.square().mean().backward()
    geometry_gradient = model.dense_decoder.geometry_proj.weight.grad
    assert geometry_gradient is not None
    assert torch.isfinite(geometry_gradient).all() and geometry_gradient.abs().sum() > 0


def test_arbitrary_continuous_times_are_not_snapped_to_training_grid():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4, dense_time_block=4).eval()
    times = torch.tensor([[.21371, .21379], [.34711, .34719]])
    prediction = model.dense_normalized(**dense_inputs(), time_s=times)
    assert prediction.shape == (2, 2, 33, 33)
    assert not torch.equal(prediction[:, 0], prediction[:, 1])


def test_geometry_basis_resolves_measured_wavefield_bandwidth():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4,4,4,4), heads=4)
    dense_cycles = float(model.dense_decoder.geometry_frequencies.max() / (2 * math.pi))
    query_cycles = float(model.query_head.frequencies.max() / (2 * math.pi))
    assert dense_cycles >= 64
    assert query_cycles >= 64
