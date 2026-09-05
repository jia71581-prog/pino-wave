"""CPU contract tests for B2-H PhysicalResidualPropagator (b2h.py), GPU-free.

Pins the structure-preserving contract the addendum requires:
  1. gate==0 is EXACTLY the coarse physical leapfrog step (bit-exact vs a manual
     wave_acceleration + explicit source), i.e. a physics-grounded warm-start.
  2. the KNOWN source waveform actually drives the field (no source -> no wave).
  3. source enters LINEARLY at gate==0 (acoustic linearity for fixed medium):
     rollout(a*s) == a*rollout(s) when p0=p1=0.
  4. gate>0 is live, K-step rollout finite, grads reach ALL residual params + gate.
  5. causality/Markov: rollout(K)[:, :j] == rollout(j) (two-frame state is Markov).
"""
import torch

from saved_time_phase_operator_v4.b2h import PhysicalResidualPropagator
from saved_time_phase_operator_v4.wave_operators import wave_acceleration


def _make(gate_init=0.0, **kw):
    torch.manual_seed(0)
    return PhysicalResidualPropagator(
        width=24, spectral_rank=12, modes=8, depth=2,
        dt_s=0.0025, dx_m=10.0, dz_m=10.0, gate_init=gate_init,
        substeps_per_saved_step=kw.pop("substeps_per_saved_step", 1),
        pressure_scale=0.1, velocity_reference_mps=1500.0,
        velocity_scale_mps=1500.0,
        activation_checkpointing=False, **kw
    )


def _inputs(b=2, z=32, x=32, steps=5):
    torch.manual_seed(1)
    p0 = torch.zeros(b, 1, z, x)
    p1 = torch.randn(b, 1, z, x) * 0.1
    velocity = torch.full((b, 1, z, x), 1500.0)
    source_map = torch.zeros(b, 1, z, x); source_map[:, :, z // 2, x // 2] = 1.0
    series = torch.randn(b, steps)
    return p0, p1, velocity, source_map, series


def test_gate0_is_exact_physical_leapfrog():
    m = _make(gate_init=0.0).eval()
    p0, p1, vel, smap, series = _inputs(steps=1)
    with torch.no_grad():
        out = m(p0, p1, vel, smap, series, steps=1)
    s0 = series[:, 0].view(-1, 1, 1, 1)
    accel = (
        wave_acceleration(p1, vel, dx_m=10.0, dz_m=10.0, free_surface_top=True)
        + s0 * smap / (10.0 * 10.0)
    )
    manual = 2.0 * p1 - p0 + (0.0025 ** 2) * accel
    assert torch.allclose(out[:, 0], manual, atol=0, rtol=0), "gate0 must be the exact physical step"


def test_source_map_is_converted_to_discrete_delta_density():
    m = _make(gate_init=0.0).eval()
    p0 = torch.zeros(1, 1, 32, 32)
    p1 = torch.zeros_like(p0)
    vel = torch.full_like(p0, 1500.0)
    smap = torch.zeros_like(p0)
    smap[:, :, 16, 16] = 1.0
    series = torch.ones(1, 1)
    with torch.no_grad():
        out = m(p0, p1, vel, smap, series, steps=1)
    expected_peak = (m.dt_s**2) / (m.dx_m * m.dz_m)
    assert torch.isclose(
        out[0, 0, 0, 16, 16],
        torch.tensor(expected_peak, dtype=out.dtype),
        atol=0,
        rtol=1.0e-6,
    )


def test_known_source_drives_field():
    m = _make(gate_init=0.0).eval()
    p0 = torch.zeros(2, 1, 32, 32); p1 = torch.zeros(2, 1, 32, 32)
    vel = torch.full((2, 1, 32, 32), 1500.0)
    smap = torch.zeros(2, 1, 32, 32); smap[:, :, 16, 16] = 1.0
    with torch.no_grad():
        no_src = m(p0, p1, vel, smap, torch.zeros(2, 4), steps=4)
        with_src = m(p0, p1, vel, smap, torch.ones(2, 4), steps=4)
    assert no_src.abs().sum() == 0.0, "zero IC + zero source => identically zero"
    assert with_src.abs().sum() > 0.0, "known source must inject a wave"


def test_source_linearity_at_gate0():
    m = _make(gate_init=0.0).eval()
    p0 = torch.zeros(2, 1, 32, 32); p1 = torch.zeros(2, 1, 32, 32)
    vel = torch.full((2, 1, 32, 32), 1500.0)
    smap = torch.zeros(2, 1, 32, 32); smap[:, :, 16, 16] = 1.0
    torch.manual_seed(3); s = torch.randn(2, 6)
    with torch.no_grad():
        r1 = m(p0, p1, vel, smap, s, steps=6)
        r2 = m(p0, p1, vel, smap, 2.5 * s, steps=6)
    assert torch.allclose(r2, 2.5 * r1, atol=1e-5, rtol=1e-4), "acoustic source linearity at gate0"


def test_gate_live_rollout_finite_and_grads():
    m = _make(gate_init=0.03).train()
    p0, p1, vel, smap, series = _inputs(steps=8)
    out = m(p0, p1, vel, smap, series, steps=8)
    assert out.shape == (2, 8, 1, 32, 32)
    assert torch.isfinite(out).all()
    out.pow(2).mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert m.gate.grad is not None and torch.isfinite(m.gate.grad).all()


def test_neural_closure_preserves_the_zero_state():
    m = _make(gate_init=0.4).train()
    p0 = torch.zeros(1, 1, 32, 32)
    velocity = torch.full_like(p0, 2500.0)
    source_map = torch.zeros_like(p0)
    source_map[:, :, 16, 16] = 1.0
    output = m(
        p0, p0, velocity, source_map, torch.zeros(1, 3), steps=3
    )
    assert torch.count_nonzero(output) == 0


def test_composed_lwc84_core_is_finite_at_dataset_vmax():
    m = _make(
        gate_init=0.0, substeps_per_saved_step=2
    ).eval()
    z = x = 48
    p0 = torch.zeros(1, 1, z, x)
    p1 = torch.zeros_like(p0)
    p1[:, :, 24, 24] = 1.0e-7
    velocity = torch.full_like(p0, 6000.0)
    source_map = torch.zeros_like(p0)
    with torch.no_grad():
        output = m(
            p0, p1, velocity, source_map, torch.zeros(1, 8), steps=8
        )
    assert torch.isfinite(output).all()
    assert output.abs().max() < 20.0 * p1.abs().max()


def test_b2hm_zero_channel_warmstart_matches_b2h():
    base = _make(gate_init=0.08, substeps_per_saved_step=2).eval()
    memory = PhysicalResidualPropagator(
        memory_steps=1,
        width=24,
        spectral_rank=12,
        modes=8,
        depth=2,
        dt_s=0.0025,
        dx_m=10.0,
        dz_m=10.0,
        gate_init=0.08,
        residual_scale_init=1.0,
        substeps_per_saved_step=2,
        pressure_scale=0.1,
        velocity_reference_mps=1500.0,
        velocity_scale_mps=1500.0,
        activation_checkpointing=False,
    ).eval()
    state = dict(base.state_dict())
    expanded = memory.state_dict()["lift.weight"].clone()
    expanded[:, :4].copy_(state["lift.weight"])
    expanded[:, 4:].zero_()
    state["lift.weight"] = expanded
    memory.load_state_dict(state, strict=True)
    p0, p1, velocity, source_map, series = _inputs(steps=4)
    history = torch.randn_like(p0)
    with torch.no_grad():
        expected = base(p0, p1, velocity, source_map, series)
        actual = memory(
            p0,
            p1,
            velocity,
            source_map,
            series,
            history=history,
        )
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_b2hm_memory_channel_receives_gradient():
    model = PhysicalResidualPropagator(
        memory_steps=1,
        width=16,
        spectral_rank=8,
        modes=6,
        depth=1,
        gate_init=0.08,
        residual_scale_init=1.0e-4,
        substeps_per_saved_step=2,
        pressure_scale=0.1,
        velocity_reference_mps=1500.0,
        velocity_scale_mps=1500.0,
        activation_checkpointing=False,
    ).train()
    p0, p1, velocity, source_map, series = _inputs(
        b=1, z=24, x=24, steps=2
    )
    model(
        p0,
        p1,
        velocity,
        source_map,
        series,
        history=torch.randn_like(p0),
    ).square().mean().backward()
    assert model.lift.weight.grad[:, 4].abs().sum() > 0


def test_causality_markov_prefix_invariance():
    m = _make(gate_init=0.2).eval()
    p0, p1, vel, smap, series = _inputs(steps=10)
    with torch.no_grad():
        long = m(p0, p1, vel, smap, series, steps=10)
        short = m(p0, p1, vel, smap, series[:, :4].contiguous(), steps=4)
    assert torch.allclose(long[:, :4], short, atol=1e-6, rtol=0)


import math  # noqa: E402


def _box_eigenmode(Z, X, dz, dx, mz, mx, dtype=torch.float64):
    """Dirichlet/free-surface box standing-wave mode Phi = sin(mz*pi z/Lz) sin(mx*pi x/Lx),
    which EXACTLY satisfies the module's BCs (odd about z=0 free surface, zero at all
    grid edges), with continuous frequency omega = c*sqrt(kz^2+kx^2)."""
    Lz = (Z - 1) * dz; Lx = (X - 1) * dx
    kz = mz * math.pi / Lz; kx = mx * math.pi / Lx
    zz = torch.arange(Z, dtype=dtype).view(Z, 1) * dz
    xx = torch.arange(X, dtype=dtype).view(1, X) * dx
    phi = torch.sin(kz * zz) * torch.sin(kx * xx)
    return phi, kz, kx


def test_standing_wave_mode_purity_and_stability():
    # gate0, source-free: the fixed core must keep a box eigenmode PURE (no mode mixing)
    # and BOUNDED (numerically stable) over a rollout -- the numerical-dispersion contract.
    torch.manual_seed(0)
    m = _make(gate_init=0.0).double().eval()
    Z = X = 64; dz = dx = m.dz_m
    phi, kz, kx = _box_eigenmode(Z, X, dz, dx, mz=3, mx=4)
    c = 1500.0
    dt = m.dt_s
    phi4 = phi.view(1, 1, Z, X)
    vel = torch.full((1, 1, Z, X), c, dtype=torch.float64)
    smap = torch.zeros(1, 1, Z, X, dtype=torch.float64)
    # exact DISCRETE eigenpair: lam = Rayleigh quotient of the audited core on phi, so
    # a_{k+1} = (2 + dt^2 lam) a_k - a_{k-1} is solved by a_k = cos(omega_d k dt) with
    # cos(omega_d dt) = 1 + 0.5 dt^2 lam.  This removes the continuous-vs-discrete beat.
    from saved_time_phase_operator_v4.wave_operators import wave_acceleration as _wa
    accel = _wa(phi4, vel, dx_m=dx, dz_m=dz, free_surface_top=True)
    lam = (accel * phi4).sum().item() / (phi4 * phi4).sum().item()  # < 0
    cos_wd = 1.0 + 0.5 * dt * dt * lam
    assert -1.0 < cos_wd < 1.0, "CFL: |cos(omega_d dt)| must be < 1 for stability"
    p0 = phi4.clone()
    p1 = phi4 * cos_wd
    with torch.no_grad():
        out = m(p0, p1, vel, smap, torch.zeros(1, 12, dtype=torch.float64), steps=12)
    phin = phi.view(-1) / phi.view(-1).norm()
    amp0 = phi.view(-1).norm().item()
    for k in range(12):
        fk = out[0, k].view(-1)
        amp = fk.norm().item()
        # STABILITY (the core CFL contract): standing-wave amplitude bounded, never grows.
        assert amp <= 1.02 * amp0, f"amplitude grew at step {k}: {amp / amp0:.4f}"
        # MODE PURITY as ABSOLUTE off-mode residual (node-robust): the continuous box mode
        # has a small fixed projection onto other discrete eigenmodes (8th-order stencil +
        # free-surface boundary). The physical contract is that this off-mode energy stays
        # SMALL and BOUNDED (no instability pumping energy into other modes) -- NOT the
        # amplitude RATIO, which is meaningless near a temporal node where amp->0.
        off = (fk - (fk @ phin) * phin).norm().item()
        assert off < 0.10 * amp0, f"off-mode content grew at step {k}: {off / amp0:.4f}"


def test_source_superposition_at_gate0():
    # gate0, zero IC: rollout is LINEAR in the source series -> rollout(s1+s2)==rollout(s1)+rollout(s2)
    m = _make(gate_init=0.0).eval()
    p0 = torch.zeros(2, 1, 32, 32); p1 = torch.zeros(2, 1, 32, 32)
    vel = torch.full((2, 1, 32, 32), 1500.0)
    smap = torch.zeros(2, 1, 32, 32); smap[:, :, 16, 16] = 1.0
    torch.manual_seed(4); s1 = torch.randn(2, 7); s2 = torch.randn(2, 7)
    with torch.no_grad():
        r_sum = m(p0, p1, vel, smap, s1 + s2, steps=7)
        r1 = m(p0, p1, vel, smap, s1, steps=7)
        r2 = m(p0, p1, vel, smap, s2, steps=7)
    assert torch.allclose(r_sum, r1 + r2, atol=1e-5, rtol=1e-4), "source superposition at gate0"


def test_symplectic_time_reversibility_gate0():
    # leapfrog is time-reversible: step(a,b)->c implies step(c,b)->a (exact, no source).
    m = _make(gate_init=0.0).double().eval()
    torch.manual_seed(2)
    Z = X = 40
    p0 = torch.randn(1, 1, Z, X, dtype=torch.float64) * 0.1
    p1 = torch.randn(1, 1, Z, X, dtype=torch.float64) * 0.1
    vel = torch.full((1, 1, Z, X), 1500.0, dtype=torch.float64)
    smap = torch.zeros(1, 1, Z, X, dtype=torch.float64)
    zero = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
    with torch.no_grad():
        # forward 5 steps
        frames = [p0, p1]
        for _ in range(5):
            frames.append(m.step(frames[-2], frames[-1], vel, smap, zero))
        # reverse: step(p_{k+1}, p_k) must recover p_{k-1}
        recovered = m.step(frames[-1], frames[-2], vel, smap, zero)
        assert torch.allclose(recovered, frames[-3], atol=1e-9, rtol=1e-7), "leapfrog not reversible"


def test_full_length_rollout_stable_K401():
    # constant-c CFL-stable full-length (T=401, matches GT) gate0 rollout stays finite + bounded.
    m = _make(gate_init=0.0).eval()
    Z = X = 48
    torch.manual_seed(5)
    p0 = torch.zeros(1, 1, Z, X); p1 = torch.zeros(1, 1, Z, X)
    vel = torch.full((1, 1, Z, X), 1500.0)   # CFL c*dt/dx = 0.375 (stable)
    smap = torch.zeros(1, 1, Z, X); smap[:, :, Z // 2, X // 2] = 1.0
    series = torch.zeros(1, 401); series[0, :40] = torch.randn(40) * 0.5  # brief source
    with torch.no_grad():
        out = m(p0, p1, vel, smap, series, steps=401)
    assert torch.isfinite(out).all(), "K=401 rollout produced non-finite values"
    peak_early = out[0, :60].abs().max().item()
    peak_late = out[0, 300:].abs().max().item()
    assert peak_late <= 3.0 * peak_early + 1e-6, "late amplitude blew up (unstable)"
