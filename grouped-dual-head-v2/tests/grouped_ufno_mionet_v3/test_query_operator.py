from __future__ import annotations

import inspect

import torch

from grouped_ufno_mionet_v3.model.operator import PhaseAlignedComplexFNOMIONet
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer, ScaleMetadata


def _normalizer() -> PhysicalNormalizer:
    return PhysicalNormalizer(
        ScaleMetadata(
            velocity_center_mps=2000.0,
            velocity_scale_mps=500.0,
            pressure_scale_pa=2.0e-8,
            source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
            train_manifest_sha256="test-train",
            allowed_medium_types=("uniform", "layered", "marmousi"),
            record_count=3,
            algorithm="test",
        )
    )


def _model() -> PhaseAlignedComplexFNOMIONet:
    return PhaseAlignedComplexFNOMIONet(
        width=8,
        rank=6,
        spectral_rank=4,
        modes=(3, 2),
        heads=2,
        token_grid=2,
        position_bands=2,
        fourier_bands=2,
        gabor_scales_s=(0.03, 0.08),
        ray_samples=4,
        domain_x_m=2000.0,
        domain_z_m=2000.0,
        domain_t_s=1.0,
    )


def _inputs():
    velocity = torch.full((1, 1, 17, 19), 2000.0)
    velocity[:, :, 9:, :] += 250.0
    sources = torch.tensor(
        [
            [400.0, 300.0, 10.0, 0.10, 1.0],
            [1200.0, 350.0, 16.0, 0.12, 2.0],
        ]
    )
    source_map = torch.zeros(2, 1, 17, 19)
    source_map[0, 0, 2, 4] = 1.0
    source_map[1, 0, 3, 11] = 1.0
    coords = torch.tensor(
        [
            [[333.3, 0.0, 0.2], [777.7, 432.1, 0.35], [1500.5, 900.2, 0.7]],
            [[444.4, 0.0, 0.2], [888.8, 512.3, 0.35], [1600.1, 950.6, 0.7]],
        ]
    )
    return velocity, sources, source_map, coords


def test_query_supports_multiple_sources_per_encoded_medium_and_off_grid_points():
    model = _model()
    velocity, sources, source_map, coords = _inputs()
    medium = model.encode_medium(velocity, _normalizer())
    prepared = model.prepare_sources(
        medium,
        sources,
        source_map,
        _normalizer(),
        record_to_medium=torch.tensor([0, 0]),
    )
    output = model.query_normalized(prepared, coords)
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    assert output[:, 0].abs().max() == 0.0
    assert not torch.allclose(output[0, 1:], output[1, 1:])


def test_query_chunking_is_numerically_equivalent_and_reuses_medium_state():
    torch.manual_seed(8)
    model = _model().eval()
    velocity, sources, source_map, _ = _inputs()
    medium = model.encode_medium(velocity, _normalizer())
    prepared = model.prepare_sources(
        medium, sources, source_map, _normalizer(), record_to_medium=torch.tensor([0, 0])
    )
    generator = torch.Generator().manual_seed(10)
    coords = torch.rand(2, 37, 3, generator=generator)
    coords[..., 0] *= 2000.0
    coords[..., 1] *= 2000.0
    coords[..., 2] *= 1.0
    full = model.query_normalized(prepared, coords)
    chunked = model.query_normalized(prepared, coords, chunk_size=7)
    torch.testing.assert_close(full, chunked, rtol=2.0e-5, atol=2.0e-6)
    assert prepared.medium is medium


def test_physical_query_decoding_applies_each_source_amplitude_once():
    model = _model()
    velocity, sources, source_map, coords = _inputs()
    normalizer = _normalizer()
    prepared = model.prepare_sources(
        model.encode_medium(velocity, normalizer),
        sources,
        source_map,
        normalizer,
        record_to_medium=torch.tensor([0, 0]),
    )
    normalized = model.query_normalized(prepared, coords)
    physical = model.query_pressure(prepared, coords)
    expected = normalized * 2.0e-8 * sources[:, 4:5]
    torch.testing.assert_close(physical, expected)


def test_all_four_mionet_inputs_and_v2_local_residual_receive_gradients():
    torch.manual_seed(9)
    model = _model()
    velocity, sources, source_map, coords = _inputs()
    velocity.requires_grad_()
    sources.requires_grad_()
    prepared = model.prepare_sources(
        model.encode_medium(velocity, _normalizer()),
        sources,
        source_map,
        _normalizer(),
        record_to_medium=torch.tensor([0, 0]),
    )
    output = model.query_normalized(prepared, coords)
    output[:, 1:].square().mean().backward()
    groups = model.required_gradient_groups()
    query_groups = {
        "medium_spectral",
        "medium_local",
        "medium_tokens",
        "medium_rank",
        "source_parameters",
        "source_map",
        "source_local_medium",
        "source_rank",
        "travel_branch",
        "periodic_trunk",
        "query_local_residual",
        "mionet_product",
    }
    assert query_groups <= set(groups)
    for name in query_groups:
        parameters = groups[name]
        assert parameters, name
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in parameters
        ), name
    assert velocity.grad is not None and velocity.grad.abs().sum() > 0
    assert sources.grad is not None and sources.grad.abs().sum() > 0


def test_operator_api_has_no_receiver_inputs():
    model = _model()
    for method_name in ("encode_medium", "prepare_sources", "query_normalized", "query_pressure"):
        parameters = inspect.signature(getattr(model, method_name)).parameters
        assert not any("receiver" in name for name in parameters)
