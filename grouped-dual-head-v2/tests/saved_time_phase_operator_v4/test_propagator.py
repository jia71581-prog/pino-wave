"""CPU contract tests for the B2 CausalSemigroupPropagator.

Pins the properties the propagator's scientific validity depends on, GPU-free:
  1. single-step: correct output shape, finite, gradients reach ALL parameters.
  2. K-step rollout is numerically stable (finite, bounded) at K=24.
  3. causality / Markov: rollout(K)[:, :j] == rollout(j) exactly (no future leakage).
  4. semigroup: two composed steps == a length-2 rollout latent.
  5. gate-0 warm-start no-op: emitted correction is identically zero at gate_init=0,
     and becomes live + finite once the gate opens (gate_init=0.03).
"""
import torch

from saved_time_phase_operator_v4.propagator import CausalSemigroupPropagator


def _make(gate_init=0.03, ckpt=False, **kw):
    torch.manual_seed(0)
    return CausalSemigroupPropagator(
        state_channels=2, cond_channels=6, width=32, spectral_rank=16,
        modes=8, depth=3, gate_init=gate_init, activation_checkpointing=ckpt, **kw
    )


def _inputs(b=2, z=24, x=24):
    torch.manual_seed(1)
    return torch.randn(b, 2, z, x), torch.randn(b, 6, z, x)


def test_single_step_shape_finite():
    # steps=1 emits only the decoded initial latent (before any step) -> valid IC frame.
    m = _make().eval()
    ic, cond = _inputs()
    with torch.no_grad():
        out = m(ic, cond, steps=1)
    assert out.shape == (2, 1, 1, 24, 24)
    assert torch.isfinite(out).all()


def test_gradients_reach_all_params():
    # need steps>=2 so the shared step operator contributes to an emitted frame.
    m = _make().train()
    ic, cond = _inputs()
    out = m(ic, cond, steps=3)
    assert out.shape == (2, 3, 1, 24, 24)
    out.pow(2).mean().backward()
    # base_projection is exercised only in anchored mode (covered separately)
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None and not n.startswith("base_projection")]
    assert not missing, f"no gradient reached: {missing}"


def test_kstep_rollout_stable():
    m = _make().eval()
    ic, cond = _inputs()
    with torch.no_grad():
        out = m(ic, cond, steps=24)
    assert out.shape == (2, 24, 1, 24, 24)
    assert torch.isfinite(out).all()
    # bounded: no explosion across the rollout (gate small, step_scale small)
    assert out.abs().max().item() < 1e3


def test_causality_markov_prefix_invariance():
    m = _make().eval()
    ic, cond = _inputs()
    with torch.no_grad():
        long = m(ic, cond, steps=10)
        short = m(ic, cond, steps=4)
    # earlier emitted frames must NOT depend on how many future steps are taken
    assert torch.allclose(long[:, :4], short, atol=1e-6, rtol=0)


def test_semigroup_shared_step():
    m = _make(gate_init=1.0).eval()
    ic, cond = _inputs()
    with torch.no_grad():
        hidden0 = m.encoder(torch.cat((ic, cond), dim=1))
        cond_latent = m.cond_projection(cond)
        h1 = m._step(hidden0, cond_latent)
        h2 = m._step(h1, cond_latent)
        # emitted frame index 2 uses h2; verify decode path matches a length-3 rollout
        roll = m(ic, cond, steps=3)
        assert torch.allclose(roll[:, 2], m.gate * m.decoder(h2), atol=1e-6, rtol=0)


def test_gate0_is_exact_noop_and_opens():
    ic, cond = _inputs()
    m0 = _make(gate_init=0.0).eval()
    assert m0.is_warmstart_noop()
    with torch.no_grad():
        out0 = m0(ic, cond, steps=8)
    assert torch.count_nonzero(out0) == 0, "gate=0 must emit an identically-zero correction"

    m1 = _make(gate_init=0.03).train()
    assert not m1.is_warmstart_noop()
    out1 = m1(ic, cond, steps=8)
    assert torch.isfinite(out1).all() and out1.abs().sum() > 0
    out1.pow(2).mean().backward()
    assert m1.gate.grad is not None and torch.isfinite(m1.gate.grad).all()


def test_hard_free_surface_is_exact_for_free_and_anchored_rollouts():
    ic, cond = _inputs()
    model = _make(gate_init=0.5, hard_free_surface=True).eval()
    base = _base_seq()
    with torch.no_grad():
        free = model(ic, cond, steps=4)
        anchored = model.forward_anchored(base, cond)

    assert torch.count_nonzero(free[..., 0, :]) == 0
    assert torch.count_nonzero(anchored[..., 0, :]) == 0


def test_zero_gated_dg_flux_transfers_parent_exactly_and_then_opens():
    parent = _make(gate_init=0.5).eval()
    candidate = _make(
        gate_init=0.5, dg_interface_rank=8, dg_cpml_margin=2
    ).eval()
    incompatible = candidate.load_state_dict(parent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(name.startswith("dg_interface.") for name in incompatible.missing_keys)
    ic, cond = _inputs(z=24, x=24)

    with torch.no_grad():
        baseline = parent(ic, cond, steps=5)
        transferred = candidate(ic, cond, steps=5)
    torch.testing.assert_close(transferred, baseline, rtol=0.0, atol=0.0)

    candidate.dg_interface.scale.data.fill_(0.1)
    with torch.no_grad():
        opened = candidate(ic, cond, steps=5)
    assert (opened - baseline).abs().max() > 1.0e-7


def test_dg_flux_gate_receives_gradient_on_an_interface_then_features_open():
    model = _make(
        gate_init=0.5,
        dg_interface_rank=8,
        dg_cpml_margin=2,
    ).train()
    ic, cond = _inputs(z=24, x=24)
    cond[:, 0, :, :12] = -0.6
    cond[:, 0, :, 12:] = 0.4

    model(ic, cond, steps=4).square().mean().backward()
    assert model.dg_interface.scale.grad is not None
    assert torch.isfinite(model.dg_interface.scale.grad)
    assert torch.count_nonzero(model.dg_interface.scale.grad) > 0
    assert torch.count_nonzero(model.dg_interface.channel_in.weight.grad) == 0

    model.zero_grad(set_to_none=True)
    model.dg_interface.scale.data.fill_(1.0e-3)
    model(ic, cond, steps=4).square().mean().backward()
    gradient = model.dg_interface.channel_in.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_activation_checkpointing_matches_plain():
    ic, cond = _inputs()
    m = _make(gate_init=0.5, ckpt=True).train()
    plain = m(ic, cond, steps=6)
    m.activation_checkpointing = False
    again = m(ic, cond, steps=6)
    assert torch.allclose(plain, again, atol=1e-5, rtol=0)


# ---- warp-anchored mode (the drift-free training path) --------------------------------

def _base_seq(b=2, k=6, z=24, x=24):
    torch.manual_seed(3)
    return torch.randn(b, k, 1, z, x)


def test_anchored_gate0_reproduces_warp_exactly():
    # gate=0 => p_k == base_k bit-exact (warm-start no-op on the frozen warp render).
    m = _make(gate_init=0.0).eval()
    _, cond = _inputs()
    base = _base_seq()
    with torch.no_grad():
        out = m.forward_anchored(base, cond)
    assert out.shape == base.shape
    assert torch.equal(out, base), "gate=0 anchored forward must reproduce warp base exactly"


def test_anchored_is_live_and_grads_reach_all():
    m = _make(gate_init=0.03).train()
    _, cond = _inputs()
    base = _base_seq()
    out = m.forward_anchored(base, cond)
    assert torch.isfinite(out).all()
    assert (out - base).abs().sum() > 0, "gate>0 must move the prediction off the warp base"
    out.pow(2).mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_anchored_causality_prefix_invariance():
    # emitted frame k must depend only on base_0..base_k, never future frames.
    m = _make(gate_init=0.5).eval()
    _, cond = _inputs()
    base = _base_seq(k=8)
    with torch.no_grad():
        full = m.forward_anchored(base, cond)
        prefix = m.forward_anchored(base[:, :5].contiguous(), cond)
    assert torch.allclose(full[:, :5], prefix, atol=1e-6, rtol=0)
