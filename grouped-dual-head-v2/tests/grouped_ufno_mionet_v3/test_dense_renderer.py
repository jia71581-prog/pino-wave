from __future__ import annotations

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
            record_count=2,
            algorithm="test",
        )
    )


def _model() -> PhaseAlignedComplexFNOMIONet:
    return PhaseAlignedComplexFNOMIONet(
        width=8,
        rank=6,
        spectral_rank=4,
        modes=(3, 2),
        dense_modes=(3, 2),
        dense_time_block=2,
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


def _prepared(model: PhaseAlignedComplexFNOMIONet):
    velocity = torch.full((1, 1, 9, 11), 2000.0)
    velocity[:, :, 5:] += 200.0
    source = torch.tensor(
        [[400.0, 300.0, 10.0, 0.1, 1.0], [1200.0, 350.0, 14.0, 0.12, 2.0]]
    )
    source_map = torch.zeros(2, 1, 9, 11)
    source_map[0, 0, 1, 2] = 1.0
    source_map[1, 0, 1, 6] = 1.0
    normalizer = _normalizer()
    return model.prepare_sources(
        model.encode_medium(velocity, normalizer),
        source,
        source_map,
        normalizer,
        record_to_medium=torch.tensor([0, 0]),
    )


def _axes():
    return torch.linspace(0.0, 2000.0, 11), torch.linspace(0.0, 2000.0, 9)


def test_dense_renderer_returns_complete_nonuniform_time_snapshots():
    torch.manual_seed(12)
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([0.13, 0.41, 0.77])
    output = model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m)
    assert output.shape == (2, 3, 9, 11)
    assert torch.isfinite(output).all()
    assert output[:, :, 0].abs().max() == 0.0
    assert not torch.allclose(output[0], output[1])


def test_dense_shared_coarse_path_matches_arbitrary_query_path():
    torch.manual_seed(13)
    model = _model().eval()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([0.17, 0.63])
    dense = model.dense_normalized(
        prepared,
        times,
        x_m=x_m,
        z_m=z_m,
        apply_correction=False,
    )
    zz, xx = torch.meshgrid(z_m, x_m, indexing="ij")
    frames = []
    for time in times:
        coords = torch.stack(
            (xx.reshape(-1), zz.reshape(-1), torch.full_like(xx.reshape(-1), time)), dim=-1
        )
        frames.append(coords)
    query_coords = torch.cat(frames, dim=0)[None].expand(2, -1, -1)
    queried = model.query_normalized(prepared, query_coords).reshape(2, 2, 9, 11)
    torch.testing.assert_close(dense, queried, rtol=3.0e-5, atol=3.0e-6)


def test_dense_time_blocking_and_cached_travel_are_equivalent():
    torch.manual_seed(14)
    model = _model().eval()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    cache = model.prepare_dense_grid(prepared, x_m=x_m, z_m=z_m)
    times = torch.tensor([0.15, 0.26, 0.48, 0.73, 0.92])
    one = model.dense_normalized(
        prepared, times, dense_grid=cache, time_block=5
    )
    blocked = model.dense_normalized(
        prepared, times, dense_grid=cache, time_block=2
    )
    torch.testing.assert_close(one, blocked, rtol=3.0e-5, atol=3.0e-6)


def test_prepare_dense_grid_accepts_source_eikonal_override():
    model = _model().eval()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    eikonal = torch.full((2, 9, 11), 0.25)
    eikonal[0, 1, 2] = 0.0
    eikonal[1, 1, 6] = 0.0

    cache = model.prepare_dense_grid(
        prepared,
        x_m=x_m,
        z_m=z_m,
        travel_time_s=eikonal,
    )

    torch.testing.assert_close(cache.travel.seconds, eikonal.flatten(1))
    assert torch.isfinite(cache.travel.path_velocity_mps).all()
    assert torch.isfinite(cache.travel.mean_slowness_s_per_m).all()


def test_streaming_preserves_all_401_requested_times_in_order():
    model = _model().eval()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.linspace(0.0, 1.0, 401)
    pieces = list(
        model.iter_dense_normalized(
            prepared,
            times,
            x_m=x_m,
            z_m=z_m,
            time_block=64,
            apply_correction=False,
        )
    )
    assert [start for start, _ in pieces] == [0, 64, 128, 192, 256, 320, 384]
    output = torch.cat([value for _, value in pieces], dim=1)
    assert output.shape == (2, 401, 9, 11)
    assert torch.isfinite(output).all()


def test_dense_physical_decode_applies_source_amplitude_once():
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    times = torch.tensor([0.2, 0.6])
    normalized = model.dense_normalized(prepared, times, x_m=x_m, z_m=z_m)
    physical = model.predict_wavefield(prepared, times, x_m=x_m, z_m=z_m)
    expected = normalized * 2.0e-8 * prepared.source_parameters[:, 4, None, None, None]
    torch.testing.assert_close(physical, expected)


def test_dense_complex_correction_has_nonzero_gradients():
    torch.manual_seed(15)
    model = _model()
    prepared = _prepared(model)
    x_m, z_m = _axes()
    output = model.dense_normalized(prepared, torch.tensor([0.3, 0.7]), x_m=x_m, z_m=z_m)
    output[:, :, 1:].square().mean().backward()
    groups = model.required_gradient_groups()
    for name in ("dense_spectral", "dense_film", "dense_local"):
        assert name in groups and groups[name]
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in groups[name]
        ), name
