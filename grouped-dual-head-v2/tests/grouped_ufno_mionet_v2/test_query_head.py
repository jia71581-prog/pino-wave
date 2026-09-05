import torch
from torch import nn

from grouped_ufno_mionet_v2.model.operator import DualHeadWaveOperator
from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer, ScaleMetadata


def normalizer():
    return PhysicalNormalizer(ScaleMetadata(2500, 1000, 1e-8, (2000, 2000, 50, 1.2, 1), "train"))


def inputs():
    velocity = torch.full((1, 1, 33, 33), 2200.0)
    source = torch.tensor([[500., 500., 10., .05, 1.], [1000., 600., 12., .08, 1.]])
    source_map = torch.zeros(2, 1, 33, 33); source_map[0, 0, 8, 8] = 1; source_map[1, 0, 10, 16] = 1
    coords = torch.rand(2, 64, 3); coords[..., :2] *= 2000; coords[..., 2] = .2 + coords[..., 2] * .5
    return dict(velocity_mps=velocity, source=source, source_map=source_map, coords=coords,
                record_to_medium=torch.zeros(2, dtype=torch.long), normalizer=normalizer())


def test_query_head_is_finite_and_source_map_has_gradient():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4, 4, 4, 4), heads=4)
    prediction = model.query_normalized(**inputs())
    assert prediction.shape == (2, 64)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradient = model.source_encoder.map_encoder[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_query_chunking_is_equivalent():
    torch.manual_seed(4)
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4, 4, 4, 4), heads=4).eval()
    values = inputs()
    full = model.query_normalized(**values)
    chunked = model.query_normalized(**values, chunk_size=17)
    torch.testing.assert_close(chunked, full, rtol=2e-5, atol=2e-5)


def test_physical_query_scales_once_with_source_amplitude():
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4, 4, 4, 4), heads=4).eval()
    values = inputs(); first = {key: (value[:1] if isinstance(value, torch.Tensor) and value.shape[:1] == (2,) else value) for key, value in values.items()}
    base = model.query_pressure(**first)
    first["source"] = first["source"].clone(); first["source"][:, 4] = 2
    doubled = model.query_pressure(**first)
    torch.testing.assert_close(doubled, 2 * base)


def test_source_t0_is_not_hard_clamped_to_zero():
    """The dataset's t0 is the wavelet center, not a zero-field causal onset."""
    torch.manual_seed(7)
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4, 4, 4, 4), heads=4).eval()
    values = inputs()
    values["coords"][..., 2] = values["source"][:, 3, None]
    prediction = model.query_normalized(**values)
    assert torch.count_nonzero(prediction).item() > 0


def test_uniform_medium_query_is_translation_equivariant_away_from_boundaries():
    torch.manual_seed(3)
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4, 4, 4, 4), heads=4).eval()
    velocity = torch.full((1, 1, 33, 33), 2200.0)
    source = torch.tensor([[500., 500., 10., .05, 1.], [1000., 500., 10., .05, 1.]])
    source_map = torch.zeros(2, 1, 33, 33)
    source_map[0, 0, 8, 8] = 1; source_map[1, 0, 8, 16] = 1
    coords = torch.tensor([[[700., 625., .3]], [[1200., 625., .3]]])
    prediction = model.query_normalized(velocity, source, source_map, coords, normalizer(),
                                        record_to_medium=torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(prediction[0], prediction[1], rtol=5e-3, atol=5e-4)


def test_query_local_features_have_direct_residual_path():
    class ZeroAttention(nn.Module):
        def forward(self, query, key, value, need_weights=False):
            return torch.zeros_like(query), None

    torch.manual_seed(5)
    model = DualHeadWaveOperator(width=16, rank=8, modes=(4, 4, 4, 4), heads=4).eval()
    model.query_head.attention = ZeroAttention()
    nn.init.zeros_(model.medium_encoder.rank_proj.weight); nn.init.zeros_(model.medium_encoder.rank_proj.bias)
    nn.init.zeros_(model.source_encoder.rank_proj.weight); nn.init.zeros_(model.source_encoder.rank_proj.bias)
    values = inputs()
    one = {key: (value[:1] if isinstance(value, torch.Tensor) and value.shape[:1] == (2,) else value)
           for key, value in values.items()}
    one["coords"] = torch.tensor([[[600., 700., .3], [1300., 700., .3]]])
    prediction = model.query_normalized(**one)
    assert not torch.allclose(prediction[:, 0], prediction[:, 1])
