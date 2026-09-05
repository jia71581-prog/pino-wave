from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.bridge import make_onset_bridge
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    LossWeights,
    build_fixed_physics_points,
    build_rad_physics_points,
    build_r3_rams_physics_points,
    instance_loss_terms,
    lwc84_residual,
    sample_fixed_physics_residual,
)


def _case():
    velocity = torch.full((1, 1, 17, 17), 1500.0)
    observed = torch.randn(1, 2, 17, 17) * 1.0e-3
    source = torch.tensor([[80.0, 80.0, 12.0, 0.01, 1.0]])
    time_s = torch.arange(8, dtype=torch.float32) * 0.0025
    return velocity, observed, source, time_s


def test_bridge_is_marked_synthetic_and_never_reads_hdf5():
    velocity, observed, source, time_s = _case()
    result = make_onset_bridge(velocity, source, observed, (0, 1), time_s, steps=2)
    assert result.provenance.synthetic is True
    assert result.provenance.true_indices == (0, 1)
    assert result.time_indices == (2, 3)


def test_fixed_points_are_reproducible_and_future_only():
    left = build_fixed_physics_points(10, (2, 3), count=20, seed=17)
    right = build_fixed_physics_points(10, (2, 3), count=20, seed=17)
    assert torch.equal(left, right)
    assert bool((left[:, 0] > 3).all())


def _hotspot_residual(frames=20, height=32, width=32, spot=(5, 10, 20)):
    residual = torch.rand(1, frames, height, width) * 0.01
    residual[0, spot[0], spot[1], spot[2]] = 50.0
    return residual


def test_rad_points_concentrate_on_high_residual_and_are_future_only():
    # Residual index ``spot[0]`` maps to absolute time spot[0] + observed[1] + 1.
    observed = (2, 3)
    residual = _hotspot_residual(spot=(5, 10, 20))
    gen = torch.Generator().manual_seed(7)
    points = build_rad_physics_points(
        residual, observed, count=2000, k=2.0, c=0.5, time_tilt=0.0, generator=gen
    )
    # All collocation times strictly after the second observed frame.
    assert bool((points[:, 0] > observed[1]).all())
    # A dominant share lands on the hotspot frame (abs time 5 + 4 = 9); uniform
    # over 16 candidate frames would place only ~6% there.
    hotspot_frame = 5 + observed[1] + 1
    frac_hotspot_frame = (points[:, 0] == hotspot_frame).float().mean()
    assert float(frac_hotspot_frame) > 0.4
    # The gathered residual recovers the spike, proving the point maps correctly.
    gathered = sample_fixed_physics_residual(residual, points, observed)
    assert float(gathered.abs().max()) > 40.0


def test_rad_causal_time_tilt_biases_sampling_earlier():
    observed = (2, 3)
    flat = torch.ones(1, 20, 32, 32)
    no_tilt = build_rad_physics_points(
        flat, observed, count=5000, k=0.0, c=1.0, time_tilt=0.0,
        generator=torch.Generator().manual_seed(7),
    )
    tilted = build_rad_physics_points(
        flat, observed, count=5000, k=0.0, c=1.0, time_tilt=3.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert float(tilted[:, 0].mean()) < float(no_tilt[:, 0].mean())


def test_rad_points_are_deterministic_under_seed():
    residual = _hotspot_residual()
    a = build_rad_physics_points(residual, (2, 3), count=500, generator=torch.Generator().manual_seed(11))
    b = build_rad_physics_points(residual, (2, 3), count=500, generator=torch.Generator().manual_seed(11))
    assert torch.equal(a, b)


def test_rad_falls_back_to_uniform_on_degenerate_residual():
    # An all-zero residual with c=0 would give an all-zero density; the sampler
    # must fall back to uniform rather than raising or returning NaNs.
    residual = torch.zeros(1, 20, 32, 32)
    points = build_rad_physics_points(
        residual, (2, 3), count=256, k=1.0, c=0.0, time_tilt=0.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert points.shape == (256, 3)
    assert bool(torch.isfinite(points).all())
    assert bool((points[:, 0] > 3).all())
    # Points must be valid for the existing gather path.
    sample_fixed_physics_residual(residual, points, (2, 3))


def test_r3_rams_keeps_all_per_frame_hotspots_and_never_duplicates():
    observed = (2, 3)
    residual = torch.arange(3 * 10 * 10, dtype=torch.float32).reshape(1, 3, 10, 10)
    point_set = build_r3_rams_physics_points(
        residual,
        observed,
        count=30,
        hard_quantile=0.9,
        uniform_fraction=0.2,
        rams_fraction=0.2,
        generator=torch.Generator().manual_seed(13),
    )
    # q90 means ten mandatory cells from every frame, even when that already
    # exceeds the nominal stochastic budget.
    assert point_set.hard_count == 30
    gathered = sample_fixed_physics_residual(residual, point_set, observed)[0]
    expected = torch.cat([frame.flatten().topk(10).values for frame in residual[0]])
    assert torch.equal(gathered[: point_set.hard_count].sort().values, expected.sort().values)
    assert len(point_set.points) == len(torch.unique(point_set.points, dim=0))
    assert torch.isclose(point_set.weights.sum(), torch.tensor(1.0))


def test_r3_retains_previous_points_until_release_threshold():
    observed = (0, 1)
    initial = torch.arange(2 * 10 * 10, dtype=torch.float32).reshape(1, 2, 10, 10)
    first = build_r3_rams_physics_points(
        initial, observed, count=8, hard_quantile=0.95,
        uniform_fraction=0.0, rams_fraction=0.0,
    )
    changed = initial.clone()
    # Former top cells remain above q70 but no longer belong to the current q95.
    changed[:, :, :, -1] = changed[:, :, :, -1] * 0.8
    second = build_r3_rams_physics_points(
        changed, observed, count=8, hard_quantile=0.95,
        release_quantile=0.7, uniform_fraction=0.0, rams_fraction=0.0,
        retained_points=first,
    )
    old = {tuple(row.tolist()) for row in first.points}
    new = {tuple(row.tolist()) for row in second.points}
    assert old & new


def test_fixed_baseline_residual_scale_prevents_candidate_rescaling():
    velocity = torch.full((1, 1, 17, 17), 1500.0)
    field = torch.randn(1, 8, 17, 17)
    baseline, scale = lwc84_residual(
        field, velocity, dt=0.0025, dx=10.0, dz=10.0,
        observed_indices=(0, 1), return_scale=True,
    )
    rescaled = lwc84_residual(
        field * 10.0, velocity, dt=0.0025, dx=10.0, dz=10.0,
        observed_indices=(0, 1), normalization_scale=scale,
    )
    assert torch.allclose(rescaled, baseline * 10.0, rtol=2.0e-4, atol=2.0e-4)


def test_loss_terms_are_finite_and_dimensionless():
    velocity, observed, source, time_s = _case()
    prediction = torch.randn(1, 8, 17, 17) * 1.0e-3
    prediction.requires_grad_()
    bridge = make_onset_bridge(velocity, source, observed, (0, 1), time_s, steps=2)
    terms = instance_loss_terms(
        raw_prediction=prediction,
        target_observed=observed,
        bridge=bridge,
        velocity=velocity,
        source=source,
        points=build_fixed_physics_points(8, (0, 1), count=4),
        weights=LossWeights(),
        dt=0.0025,
        dx=10.0,
        dz=10.0,
        observed_indices=(0, 1),
    )
    assert all(torch.isfinite(value) for value in terms.values())
    terms["total"].backward()
    assert prediction.grad is not None


def test_constant_field_has_finite_lwc_residual():
    velocity = torch.full((1, 1, 17, 17), 1500.0)
    field = torch.ones(1, 6, 17, 17)
    residual = lwc84_residual(field, velocity, dt=0.0025, dx=10.0, dz=10.0, observed_indices=(0, 1))
    assert torch.isfinite(residual).all()


def test_fourth_order_time_residual_smaller_on_lwc_consistent_field():
    """The default (time_order=4) residual must be markedly smaller than the plain
    3-point (time_order=2) form on a field advanced by the LWC-84 recurrence, because
    it includes the dt^2/12 L^2 Lax-Wendroff term that cancels the time-truncation
    error the solver's field carries. This locks the high-order PDE-loss contract."""
    from saved_time_phase_operator_v4.instance_adaptation.bridge import _laplacian8

    torch.manual_seed(0)
    dt, dx, dz = 0.0025, 10.0, 10.0
    velocity = torch.full((1, 24, 24), 1500.0, dtype=torch.float64)
    c2 = velocity.square()

    def L(field):  # source-free L(p) = c^2 nabla^2_8 p
        return c2 * _laplacian8(field, dx_m=dx, dz_m=dz)

    # Manufacture a smooth initial field and advance it with the exact LWC-84
    # source-free recurrence p_{n+1} = 2p_n - p_{n-1} + dt^2 L(p_n) + dt^4/12 L^2(p_n).
    zz, xx = torch.meshgrid(
        torch.linspace(0, 3.14, 24, dtype=torch.float64),
        torch.linspace(0, 3.14, 24, dtype=torch.float64),
        indexing="ij",
    )
    p_prev = (torch.sin(zz) * torch.sin(xx))[None] * 1.0e-3
    p_cur = p_prev.clone()
    frames = [p_prev, p_cur]
    for _ in range(6):
        acc = L(p_cur)
        p_next = 2.0 * p_cur - p_prev + dt * dt * acc + (dt**4 / 12.0) * L(acc)
        frames.append(p_next)
        p_prev, p_cur = p_cur, p_next
    field = torch.stack(frames, dim=1)  # [1, T, 24, 24]

    r4 = lwc84_residual(field, velocity, dt=dt, dx=dx, dz=dz, observed_indices=(0, 1), time_order=4)
    r2 = lwc84_residual(field, velocity, dt=dt, dx=dx, dz=dz, observed_indices=(0, 1), time_order=2)
    rms4 = r4.square().mean().sqrt()
    rms2 = r2.square().mean().sqrt()
    # On the LWC-consistent field the 4th-order residual is near machine precision;
    # the plain 3-point form retains the dt^2/12 L^2 term as spurious residual.
    assert rms4 < rms2 * 0.1, (float(rms4), float(rms2))
