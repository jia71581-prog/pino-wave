import pytest
import torch

from saved_time_phase_operator_v4.losses import band_limited_residual_loss


def test_band_limited_loss_is_zero_for_an_exact_field():
    target = torch.randn(2, 4, 17, 19)
    assert band_limited_residual_loss(target, target) == 0.0


def test_band_limited_loss_is_finite_for_zero_energy_frames_and_has_gradients():
    prediction = torch.randn(2, 4, 17, 19, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 2:] = torch.randn_like(target[:, 2:])

    loss = band_limited_residual_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0


def test_band_limited_loss_rejects_non_field_shapes():
    with pytest.raises(ValueError, match="matching"):
        band_limited_residual_loss(torch.zeros(2, 3), torch.zeros(2, 3))


from saved_time_phase_operator_v4.losses import (
    relative_energy_squared_reference,
    relative_energy_squared_block_loss,
)


def _decaying_case(seed=0):
    torch.manual_seed(seed)
    # energy decays over time so late frames are near-zero (the long-time regime)
    decay = torch.linspace(1.0, 0.02, 10)[None, :, None, None]
    target = torch.randn(2, 10, 16, 16) * decay
    prediction = target + 0.05 * torch.randn_like(target)
    return prediction, target


def test_block_loss_default_path_is_unchanged():
    pred, tgt = _decaying_case()
    ref = relative_energy_squared_reference(tgt, energy_floor_fraction=0.05)
    # legacy call (no frame weighting) must match the hand-computed record-normalized mean
    out = relative_energy_squared_block_loss(pred, tgt, reference=ref, spectrum_weight=0.0)
    manual = ((pred.float() - tgt.float()).square().flatten(1).sum(-1) / ref.target_square).mean()
    torch.testing.assert_close(out.frame, manual, atol=1e-6, rtol=1e-5)


def test_per_frame_weighting_exposes_late_frames():
    pred, tgt = _decaying_case()
    ref = relative_energy_squared_reference(tgt, energy_floor_fraction=0.05)
    default = relative_energy_squared_block_loss(pred, tgt, reference=ref, spectrum_weight=0.0).frame
    uniform = relative_energy_squared_block_loss(
        pred, tgt, reference=ref, spectrum_weight=0.0,
        frame_time_weights=torch.ones(10), frame_index_range=(0, 10),
    ).frame
    late = relative_energy_squared_block_loss(
        pred, tgt, reference=ref, spectrum_weight=0.0,
        frame_time_weights=torch.linspace(1.0, 5.0, 10), frame_index_range=(0, 10),
    ).frame
    # per-frame normalization makes the (previously invisible) late error visible,
    # and up-weighting late frames increases the loss further
    assert float(uniform) > float(default)
    assert float(late) > float(uniform)


def test_per_frame_weighting_finite_on_near_zero_late_frame():
    pred, tgt = _decaying_case()
    tgt[:, -1] *= 1e-6  # a late frame with essentially no energy
    ref = relative_energy_squared_reference(tgt, energy_floor_fraction=0.05)
    out = relative_energy_squared_block_loss(
        pred, tgt, reference=ref, spectrum_weight=0.0,
        frame_time_weights=torch.linspace(1.0, 5.0, 10), frame_index_range=(0, 10),
    )
    assert torch.isfinite(out.frame)


def test_per_frame_weighting_matches_block_slicing():
    # streaming decomposition: two 5-frame blocks must weight-align to absolute indices
    pred, tgt = _decaying_case()
    ref = relative_energy_squared_reference(tgt, energy_floor_fraction=0.05)
    w = torch.linspace(1.0, 5.0, 10)
    b0 = relative_energy_squared_block_loss(
        pred[:, :5], tgt[:, :5], reference=ref, spectrum_weight=0.0,
        frame_time_weights=w, frame_index_range=(0, 5),
    ).frame
    b1 = relative_energy_squared_block_loss(
        pred[:, 5:], tgt[:, 5:], reference=ref, spectrum_weight=0.0,
        frame_time_weights=w, frame_index_range=(5, 10),
    ).frame
    assert torch.isfinite(b0) and torch.isfinite(b1)
    # block 1 (later, higher weight, lower energy) should carry more loss than block 0
    assert float(b1) > float(b0)


from saved_time_phase_operator_v4.losses import wave_pde_residual_loss


def _plane_wave(records, T, Z, X, *, dz, dx, dt, c, omega_factor):
    # u = cos(kz*z + kx*x - omega*t), omega = omega_factor * c*|k|.
    z = torch.arange(Z).float()[None, None, :, None] * dz
    x = torch.arange(X).float()[None, None, None, :] * dx
    t = torch.arange(T).float()[None, :, None, None] * dt
    kz, kx = 2.0 * torch.pi / (Z * dz) * 3.0, 2.0 * torch.pi / (X * dx) * 2.0
    kmag = (kz**2 + kx**2) ** 0.5
    omega = omega_factor * c * kmag
    u = torch.cos(kz * z + kx * x - omega * t).expand(records, T, Z, X).contiguous()
    return u


def test_wave_pde_residual_small_for_a_true_solution_large_for_a_non_solution():
    R, T, Z, X = 2, 24, 48, 48
    dz = dx = 2000.0 / (Z - 1)
    dt = 1.0 / (T - 1)
    c = 1500.0
    vel = torch.full((R, Z, X), c)
    time_s = (torch.arange(T).float() * dt)[None, :].expand(R, T).contiguous()
    u_sol = _plane_wave(R, T, Z, X, dz=dz, dx=dx, dt=dt, c=c, omega_factor=1.0)
    u_bad = _plane_wave(R, T, Z, X, dz=dz, dx=dx, dt=dt, c=c, omega_factor=2.5)
    r_sol = wave_pde_residual_loss(u_sol, vel, time_s, dz_m=dz, dx_m=dx)
    r_bad = wave_pde_residual_loss(u_bad, vel, time_s, dz_m=dz, dx_m=dx)
    assert 0.0 <= float(r_sol) < 0.1          # true dispersion => small residual (up to FD dispersion)
    assert float(r_bad) > 5.0 * float(r_sol)  # wrong dispersion => clearly larger
    assert float(r_bad) > 0.3


def test_wave_pde_residual_handles_nonuniform_time_and_short_blocks():
    R, T, Z, X = 1, 3, 16, 16
    dz = dx = 100.0
    vel = torch.full((R, Z, X), 1500.0)
    # non-uniform time samples
    time_s = torch.tensor([[0.0, 0.03, 0.10]])
    u = torch.randn(R, T, Z, X)
    val = wave_pde_residual_loss(u, vel, time_s, dz_m=dz, dx_m=dx)
    assert torch.isfinite(val) and float(val) >= 0.0
    # fewer than 3 frames => no interior time point => zero (no constraint)
    z2 = wave_pde_residual_loss(u[:, :2], vel, time_s[:, :2], dz_m=dz, dx_m=dx)
    assert float(z2) == 0.0


def test_wave_pde_residual_source_free_mask_and_gradients():
    R, T, Z, X = 1, 5, 20, 20
    dz = dx = 100.0
    vel = torch.full((R, Z, X), 1500.0)
    time_s = (torch.linspace(0, 0.1, T))[None, :]
    u = torch.randn(R, T, Z, X, requires_grad=True)
    mask = torch.ones(R, Z, X, dtype=torch.bool)
    mask[:, 8:12, 8:12] = False  # exclude a source disk
    val = wave_pde_residual_loss(u, vel, time_s, dz_m=dz, dx_m=dx, source_free_mask=mask)
    val.backward()
    assert torch.isfinite(val) and u.grad is not None and torch.isfinite(u.grad).all()
