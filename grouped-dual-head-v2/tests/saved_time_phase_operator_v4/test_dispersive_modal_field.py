"""A5 (class-A escalation): per-pixel LEARNED-DISPERSION modal coarse field.

A5 is the escalation the A3 (continuous-temporal-latent) falsification gate points to.
Where A3 modulates query-independent anchors with a GENERIC fixed-frequency harmonic
trunk, A5 gives each location its own oscillatory time law with a LEARNED per-pixel
angular frequency ``omega_m(x,z)`` -- a discrete dispersive/Green-modal expansion:

    residual(x,z,t) = gate * (1/sqrt(M)) sum_m A_m(x,z) cos(omega_m(x,z) t_rel + phi_m(x,z)).

These tests pin the contract this module MUST satisfy to be a clean single-factor
class-A branch (mirroring the A3 / warp / multi_arrival test suites):
  * gate_init=0 is an EXACT no-op (byte-reproducible warm start),
  * per-frame QUERY INVARIANCE holds by construction (a single-frame query matches that
    frame in a batch), both in isolation and through the real forward,
  * the FRAME-CHUNKED memory-safety loop is numerically identical to an un-chunked pass
    (chunking exists only to bound the (chunk,M,H,W) intermediate at count=401 validation
    -- it must not change the result),
  * a small positive gate_init makes it LIVE (nonzero, finite) with real gradient to all
    three heads (amp/omega/phi) + the gate, so the deadlock A3 hit cannot recur,
  * a generator built with the A5 flag constructs the module, no-op at init.
"""
from __future__ import annotations

import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _DispersiveModalField,
)
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


def _inputs(*, records, count, width, height, grid_w, seed=0):
    torch.manual_seed(seed)
    conditioning = torch.randn(records, width, height, grid_w)
    time_s = (torch.rand(records, count) * 0.4 + 0.1).sort(dim=1).values
    source = torch.zeros(records, 5)
    source[:, 2] = 15.0   # frequency
    source[:, 3] = 0.05   # onset
    return conditioning, time_s, source


def test_gate0_is_exact_noop():
    mod = _DispersiveModalField(8, modes=16, max_frequency=8.0, gate_init=0.0)
    conditioning, time_s, source = _inputs(records=2, count=5, width=8, height=7, grid_w=9)
    out = mod(conditioning, time_s, source, domain_t_s=1.0)
    assert out.shape == (2, 5, 7, 9)
    assert torch.count_nonzero(out) == 0        # gate=0 -> identically zero


def test_query_invariance_isolated():
    """A single-frame query must equal that frame computed inside a full batch."""
    mod = _DispersiveModalField(8, modes=16, max_frequency=8.0, gate_init=0.03)
    mod.eval()
    conditioning, time_s, source = _inputs(records=1, count=4, width=8, height=6, grid_w=6)
    full = mod(conditioning, time_s, source, domain_t_s=1.0)
    frame = 2
    single = mod(conditioning, time_s[:, frame : frame + 1], source, domain_t_s=1.0)
    assert torch.allclose(full[:, frame], single[:, 0], atol=1e-6)


def test_frame_chunk_invariance():
    """The memory-safety frame loop must be numerically identical to one big pass."""
    conditioning, time_s, source = _inputs(records=2, count=37, width=8, height=6, grid_w=6)
    outs = []
    for chunk in (1, 8, 64, 1000):
        mod = _DispersiveModalField(8, modes=16, max_frequency=8.0, gate_init=0.05, frame_chunk=chunk)
        torch.manual_seed(123)                       # identical head weights across chunk sizes
        for m in (mod.amp_projection, mod.omega_projection, mod.phi_projection):
            for p in m.parameters():
                p.data.normal_()
        outs.append(mod(conditioning, time_s, source, domain_t_s=1.0))
    for o in outs[1:]:
        assert torch.allclose(o, outs[0], atol=1e-6)   # chunking changes nothing


def test_gate003_is_live_finite_and_gradients_reach_all_heads():
    mod = _DispersiveModalField(8, modes=16, max_frequency=8.0, gate_init=0.03)
    conditioning, time_s, source = _inputs(records=2, count=4, width=8, height=7, grid_w=9)
    conditioning.requires_grad_(True)
    out = mod(conditioning, time_s, source, domain_t_s=1.0)
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0                       # live once the gate is open
    out.square().mean().backward()
    assert mod.gate.grad is not None and torch.isfinite(mod.gate.grad).all()
    for head in (mod.amp_projection, mod.omega_projection, mod.phi_projection):
        g = head[0].weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    assert conditioning.grad is not None and torch.isfinite(conditioning.grad).all()


def test_omega_is_bounded():
    """Learned per-pixel frequency stays in (0, 2*pi*max_frequency): sigmoid-bounded."""
    mod = _DispersiveModalField(8, modes=16, max_frequency=8.0, gate_init=0.03)
    conditioning, _, _ = _inputs(records=3, count=1, width=8, height=6, grid_w=6)
    with torch.no_grad():
        omega = (2.0 * torch.pi * mod.max_frequency) * torch.sigmoid(mod.omega_projection(conditioning))
    assert (omega > 0).all() and (omega < 2.0 * torch.pi * mod.max_frequency).all()


# --- End-to-end through the REAL LocalPropagationFieldGenerator.forward -------------


def _a5_generator(*, gate_init, width=8, pyramid_levels=2, saved_time_count=401):
    return LocalPropagationFieldGenerator(
        width=width, pyramid_levels=pyramid_levels, saved_time_count=saved_time_count,
        domain_t_s=1.0, domain_diagonal_m=2828.0, residual=True,
        warp=True, warp_max_shift_cells=8.0,
        dispersive_modal=True, dispersive_modal_modes=16,
        dispersive_modal_max_frequency=8.0, adapter_gate_init=gate_init,
    )


def _forward_inputs(gen, *, records, count, medium_count, height, grid_w, seed=0):
    torch.manual_seed(seed)
    base = gen.width
    levels = gen.pyramid_levels
    pyramid = [torch.randn(medium_count, base, height, grid_w)]
    h, w = height, grid_w
    for _ in range(1, levels):
        h = max(1, h // 2); w = max(1, w // 2)
        pyramid.append(torch.randn(medium_count, base, h, w))
    medium = MediumEncoding(pyramid=tuple(pyramid), tokens=None, token_positions=None, rank=None)
    source = SourceEncoding(
        hidden=torch.randn(records, base), rank=None,
        map_field=torch.randn(records, base, height, grid_w), local_medium=None,
    )
    points = height * grid_w
    travel = RayTravelTime(
        seconds=torch.rand(records, points) * 0.3 + 0.05,
        distance_m=torch.rand(records, points) * 1000.0 + 1.0,
        path_velocity_mps=torch.full((records, points), 1500.0),
        endpoint_velocity_mps=torch.full((records, points), 1500.0),
        mean_slowness_s_per_m=torch.full((records, points), 1.0 / 1500.0),
    )
    velocity_mps = torch.randn(records, 1, height, grid_w)
    source_parameters = torch.zeros(records, 5)
    source_parameters[:, 2] = 15.0
    source_parameters[:, 3] = 0.05
    record_to_medium = torch.arange(records) % medium_count
    time_s = (torch.rand(records, count) * 0.4 + 0.1).sort(dim=1).values
    saved_time_indices = torch.randint(0, 401, (records, count))
    return dict(
        velocity_mps=velocity_mps, medium=medium, source=source,
        source_parameters=source_parameters, record_to_medium=record_to_medium,
        time_s=time_s, travel=travel, saved_time_indices=saved_time_indices,
    )


def test_generator_builds_a5_as_noop():
    gen = _a5_generator(gate_init=0.0)
    assert gen.dispersive_modal is not None
    assert float(gen.dispersive_modal.gate) == 0.0


def test_full_forward_a5_gate0_is_exact_noop_vs_disabled():
    """Within the REAL forward, the zero-gate A5 adds nothing -> output equals the
    identical model with A5 disabled (the warp-only parent path). Warm-start byte-repro."""
    gen = _a5_generator(gate_init=0.0)
    gen.eval()
    inputs = _forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16)
    with torch.no_grad():
        out_a5 = gen(**inputs)
        saved = gen.dispersive_modal
        gen.dispersive_modal = None
        out_disabled = gen(**inputs)
        gen.dispersive_modal = saved
    assert out_a5.shape == (2, 3, 16, 16)
    assert torch.isfinite(out_a5).all()
    assert torch.allclose(out_a5, out_disabled, atol=1e-6)


def test_full_forward_a5_is_live_once_gate_opens():
    gen = _a5_generator(gate_init=0.0)
    gen.eval()
    inputs = _forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16)
    with torch.no_grad():
        out_noop = gen(**inputs)
        gen.dispersive_modal.gate.fill_(0.3)
        out_active = gen(**inputs)
    assert torch.isfinite(out_active).all()
    assert not torch.allclose(out_active, out_noop, atol=1e-4)


def test_full_forward_a5_query_invariance():
    """A single-frame query through the full forward matches that frame in a batch."""
    gen = _a5_generator(gate_init=0.03)
    gen.eval()
    with torch.no_grad():
        gen.dispersive_modal.gate.fill_(0.2)
    records, count, height, grid_w = 1, 4, 16, 16
    inputs = _forward_inputs(gen, records=records, count=count, medium_count=1,
                             height=height, grid_w=grid_w)
    frame = 2
    with torch.no_grad():
        full = gen(**inputs)
        single_inputs = dict(inputs)
        single_inputs["time_s"] = inputs["time_s"][:, frame : frame + 1]
        single_inputs["saved_time_indices"] = inputs["saved_time_indices"][:, frame : frame + 1]
        single = gen(**single_inputs)
    assert torch.allclose(full[:, frame], single[:, 0], atol=1e-6)
