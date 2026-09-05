"""Unit tests for the A4 extra-arrival mixture warp (``_MultiArrivalWarp``).

A4 (class-A) targets the DOMINANT layered+marmousi *late* residual: late frames carry
reflections / multiples that the r4 single first-arrival warp cannot place.

Option B contract (warm-start reproduces the warp ceiling): this module does NOT own the
primary first-arrival path -- path 0 stays the external, parent-trained ``self.warp``.
``num_paths`` is the number of EXTRA delayed/displaced paths applied ADDITIVELY on top of
the already-warped input field:

    ``field_out = field_in + sum_{k} g_k * causal_k * warp_k(field_in)``

with every ``g_k`` zero-init.  These tests pin:
  * zero-init path gates -> exact identity no-op (byte-reproducible warm start: returns field_in),
  * the module is purely ADDITIVE on top of the input field and linear in the mixing gate,
  * a zero mixing gate makes extra paths contribute nothing regardless of their shift heads,
  * a learned nonzero gate + delay makes an extra path actually contribute (transport, not rescale),
  * per-frame query invariance (single frame matches its slot in a multi-frame batch),
  * the shift is bounded by ``max_shift_cells``,
  * gradients flow to shift heads, delay heads, and the mixing gate,
  * invalid bounds are rejected.
"""
from __future__ import annotations

import pytest
import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _MultiArrivalWarp,
)
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime

_CAUSAL_WIDTH_S = 5.0e-3
_DOMAIN_T_S = 0.6


def _inputs(*, records: int, count: int, width: int, height: int, grid_w: int):
    field = torch.randn(records, count, height, grid_w)
    rendered = torch.randn(records * count, width, height, grid_w)
    spatial = torch.randn(records, width, height, grid_w)
    time_s = torch.rand(records, count) * _DOMAIN_T_S
    onset = torch.rand(records, 1, 1, 1) * 0.05
    return field, rendered, spatial, time_s, onset


def _radial_arrival(records: int, height: int, grid_w: int) -> torch.Tensor:
    zs = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(grid_w, dtype=torch.float32)[None, :]
    dist = torch.sqrt(zs * zs + xs * xs)
    return dist[None].repeat(records, 1, 1) * 1e-3


def _call(warp, field, rendered, spatial, time_s, onset, arrival):
    return warp(
        field,
        rendered,
        arrival,
        spatial,
        time_s,
        onset,
        _CAUSAL_WIDTH_S,
        _DOMAIN_T_S,
    )


def test_zero_init_is_exact_identity_noop():
    """Every path_gate == 0 -> returns the input field byte-for-byte (warp ceiling)."""
    warp = _MultiArrivalWarp(8, num_paths=3, max_shift_cells=8.0)
    field, rendered, spatial, time_s, onset = _inputs(
        records=2, count=3, width=8, height=7, grid_w=9
    )
    arrival = _radial_arrival(2, 7, 9)
    out = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    assert out.shape == field.shape
    assert torch.allclose(out, field, atol=1e-6)


def test_path_gate_and_shift_heads_are_zero_initialized():
    warp = _MultiArrivalWarp(8, num_paths=3, max_shift_cells=8.0)
    assert warp.path_gate.numel() == 3          # one gate per EXTRA path
    assert torch.count_nonzero(warp.path_gate) == 0
    assert len(warp.shift_heads) == 3
    assert len(warp.delay_heads) == 3
    for head in warp.shift_heads:
        assert torch.count_nonzero(head[-1].weight) == 0
        assert torch.count_nonzero(head[-1].bias) == 0


def test_additive_and_linear_in_mixing_gate():
    """out - field == sum_k g_k * causal_k * warp_k(field); doubling the gate doubles the delta."""
    warp = _MultiArrivalWarp(4, num_paths=1, max_shift_cells=6.0)
    with torch.no_grad():
        warp.shift_heads[0][-1].bias.fill_(1.5)   # a real, non-saturating shift
        warp.delay_heads[0].bias.fill_(0.2)
    field, rendered, spatial, time_s, onset = _inputs(
        records=2, count=3, width=4, height=8, grid_w=8
    )
    # push frames late so the delayed causal gate is open
    time_s = torch.full_like(time_s, 0.5)
    arrival = _radial_arrival(2, 8, 8)
    with torch.no_grad():
        warp.path_gate.fill_(0.4)
        out1 = _call(warp, field, rendered, spatial, time_s, onset, arrival)
        warp.path_gate.fill_(0.8)
        out2 = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    delta1 = out1 - field
    delta2 = out2 - field
    assert delta1.abs().max() > 1e-4               # the extra path really contributes
    assert torch.allclose(delta2, 2.0 * delta1, atol=1e-6)   # linear in the gate


def test_zero_mixing_gate_ignores_extra_paths():
    """With path_gate == 0, extra paths contribute nothing even with large shift heads."""
    warp = _MultiArrivalWarp(8, num_paths=3, max_shift_cells=6.0)
    with torch.no_grad():
        warp.shift_heads[0][-1].bias.fill_(2.0)   # extra paths would warp a lot...
        warp.shift_heads[1][-1].bias.fill_(-2.0)
        warp.shift_heads[2][-1].bias.fill_(1.0)
        warp.delay_heads[0].bias.fill_(1.0)
        warp.delay_heads[1].bias.fill_(-1.0)
        # ...but path_gate stays 0.
    field, rendered, spatial, time_s, onset = _inputs(
        records=2, count=3, width=8, height=7, grid_w=9
    )
    arrival = _radial_arrival(2, 7, 9)
    out = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    assert torch.allclose(out, field, atol=1e-6)


def test_learned_gate_and_delay_make_extra_path_contribute():
    warp = _MultiArrivalWarp(4, num_paths=1, max_shift_cells=6.0)
    with torch.no_grad():
        warp.shift_heads[0][-1].bias.fill_(3.0)   # extra path warps
        warp.path_gate.fill_(0.8)                 # and is mixed in
    records, count, height, grid_w = 1, 4, 16, 16
    field = torch.zeros(records, count, height, grid_w)
    field[0, :, 8, 8] = 1.0
    rendered = torch.zeros(records * count, 4, height, grid_w)
    spatial = torch.zeros(records, 4, height, grid_w)
    # Late frames so the delayed causal gate is open.
    time_s = torch.full((records, count), 0.5)
    onset = torch.zeros(records, 1, 1, 1)
    arrival = _radial_arrival(records, height, grid_w)
    out = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    assert not torch.allclose(out, field, atol=1e-4)
    assert torch.isfinite(out).all()


def test_per_frame_query_invariance():
    """A single-frame query must match that frame inside a multi-frame batch."""
    warp = _MultiArrivalWarp(8, num_paths=3, max_shift_cells=8.0)
    with torch.no_grad():
        warp.shift_heads[0][-1].bias.fill_(1.0)
        warp.shift_heads[1][-1].bias.fill_(1.5)
        warp.path_gate.fill_(0.5)
    warp.eval()
    records, count, height, grid_w = 1, 4, 6, 6
    field, rendered, spatial, time_s, onset = _inputs(
        records=records, count=count, width=8, height=height, grid_w=grid_w
    )
    arrival = _radial_arrival(records, height, grid_w)
    full = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    frame = 2
    single = warp(
        field[:, frame : frame + 1],
        rendered[frame : frame + 1],
        arrival,
        spatial,
        time_s[:, frame : frame + 1],
        onset,
        _CAUSAL_WIDTH_S,
        _DOMAIN_T_S,
    )
    assert torch.allclose(full[:, frame], single[:, 0], atol=1e-6)


def test_shift_is_bounded_by_max_shift_cells():
    warp = _MultiArrivalWarp(4, num_paths=3, max_shift_cells=3.0)
    with torch.no_grad():
        for head in warp.shift_heads:
            head[-1].bias.fill_(100.0)  # saturate tanh
    rendered = torch.randn(6, 4, 10, 10)
    for head in warp.shift_heads:
        shift = warp.max_shift_cells * torch.tanh(head(rendered))
        assert shift.abs().max() <= 3.0 + 1e-5


def test_gradients_flow_to_all_learnable_parts():
    warp = _MultiArrivalWarp(8, num_paths=2, max_shift_cells=8.0)
    with torch.no_grad():
        warp.shift_heads[0][-1].bias.fill_(1.0)
        warp.shift_heads[1][-1].bias.fill_(1.0)
        warp.path_gate.fill_(0.3)
    field, rendered, spatial, time_s, onset = _inputs(
        records=2, count=2, width=8, height=8, grid_w=8
    )
    time_s = torch.full_like(time_s, 0.5)   # open the delayed causal gate
    rendered.requires_grad_(True)
    spatial.requires_grad_(True)
    arrival = _radial_arrival(2, 8, 8)
    out = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    out.square().mean().backward()
    for head in warp.shift_heads:
        for conv in (head[0], head[-1]):
            assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
    for delay in warp.delay_heads:
        assert delay.weight.grad is not None and torch.isfinite(delay.weight.grad).all()
    assert warp.path_gate.grad is not None and torch.isfinite(warp.path_gate.grad).all()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()
    assert spatial.grad is not None and torch.isfinite(spatial.grad).all()


@pytest.mark.parametrize("bad", [0.0, -1.0, -8.0])
def test_invalid_max_shift_rejected(bad):
    with pytest.raises(ValueError, match="max_shift_cells"):
        _MultiArrivalWarp(8, num_paths=3, max_shift_cells=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_num_paths_rejected(bad):
    with pytest.raises(ValueError, match="num_paths"):
        _MultiArrivalWarp(8, num_paths=bad, max_shift_cells=8.0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_max_delay_frac_rejected(bad):
    with pytest.raises(ValueError, match="max_delay_frac"):
        _MultiArrivalWarp(8, num_paths=3, max_shift_cells=8.0, max_delay_frac=bad)


@pytest.mark.parametrize("bad", [-0.01, -1.0])
def test_invalid_gate_init_rejected(bad):
    with pytest.raises(ValueError, match="gate_init"):
        _MultiArrivalWarp(8, num_paths=3, max_shift_cells=8.0, gate_init=bad)


def test_gate_init_nonzero_breaks_cold_start_deadlock():
    """A small positive gate_init lets the (zero-init) shift/delay heads receive real
    gradient at load -- the whole point of the knob.

    With gate_init == 0 AND zero-init shift heads, the extra paths are an exact no-op
    AND every shift-head gradient is identically zero (dL/d(head) = gate * ... == 0),
    so on a frozen parent the paths can never develop.  A positive gate_init both makes
    the paths contribute on late frames at load and delivers a NON-zero gradient to the
    shift heads, which is what unblocks training.
    """
    warp = _MultiArrivalWarp(4, num_paths=1, max_shift_cells=6.0, gate_init=0.05)
    assert torch.allclose(warp.path_gate, torch.full((1,), 0.05))
    field, rendered, spatial, time_s, onset = _inputs(
        records=2, count=3, width=4, height=8, grid_w=8
    )
    time_s = torch.full_like(time_s, 0.5)   # late frames -> delayed causal gate open
    rendered.requires_grad_(True)
    arrival = _radial_arrival(2, 8, 8)
    out = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    # Contributes at load even though the shift heads are still zero-init (identity warp).
    assert (out - field).abs().max() > 1e-5
    out.square().mean().backward()
    # The deadlock is broken: the shift head's final (zero-init) conv now has real grad.
    last_conv = warp.shift_heads[0][-1]
    assert last_conv.weight.grad is not None
    assert last_conv.weight.grad.abs().max() > 0.0
    assert warp.path_gate.grad is not None and warp.path_gate.grad.abs().max() > 0.0



def test_finite_output_on_random_inputs():
    warp = _MultiArrivalWarp(8, num_paths=3, max_shift_cells=8.0)
    with torch.no_grad():
        for head in warp.shift_heads:
            head[-1].weight.normal_(0.0, 0.1)
            head[-1].bias.normal_(0.0, 1.0)
        warp.path_gate.normal_(0.0, 0.5)
    field, rendered, spatial, time_s, onset = _inputs(
        records=3, count=2, width=8, height=11, grid_w=13
    )
    arrival = _radial_arrival(3, 11, 13)
    out = _call(warp, field, rendered, spatial, time_s, onset, arrival)
    assert out.shape == (3, 2, 11, 13)
    assert torch.isfinite(out).all()


def _generator(**kwargs):
    return LocalPropagationFieldGenerator(
        width=8,
        pyramid_levels=2,
        saved_time_count=16,
        domain_t_s=0.5,
        domain_diagonal_m=1000.0,
        channel_multipliers=(1, 1, 2, 2),
        residual=True,
        activation_checkpointing=False,
        warp=True,
        **kwargs,
    )


def test_generator_wires_adapter_gate_init_into_multi_arrival():
    """The config knob local_field_adapter_gate_init must reach the constructed
    module's path_gate through LocalPropagationFieldGenerator (plumbing coverage).

    gate_init == 0 -> byte-exact warm-start no-op (path_gate all zero); a positive
    value warms every path_gate so the cold-start deadlock is broken at launch.
    """
    cold = _generator(multi_arrival=True, multi_arrival_paths=3)
    warm = _generator(
        multi_arrival=True, multi_arrival_paths=3, adapter_gate_init=0.03
    )
    assert cold.multi_arrival is not None and warm.multi_arrival is not None
    assert torch.count_nonzero(cold.multi_arrival.path_gate) == 0        # default byte no-op
    assert torch.allclose(
        warm.multi_arrival.path_gate, torch.full((3,), 0.03)
    )
    # single-factor: A4 adds ONLY its own params on top of the warp-only generator
    without = _generator(multi_arrival=False)
    base = sum(p.numel() for p in without.parameters())
    grown = sum(p.numel() for p in warm.parameters())
    ma_params = sum(p.numel() for p in warm.multi_arrival.parameters())
    assert grown - base == ma_params


def test_generator_multi_arrival_params_are_prefixed_for_lr_routing():
    """Every A4 param must live under ``multi_arrival.`` so the staged optimizer
    routes it into its own LR group (fresh-adapter warmup)."""
    gen = _generator(multi_arrival=True, multi_arrival_paths=2, adapter_gate_init=0.02)
    for name, param in gen.named_parameters():
        if any(param is p for p in gen.multi_arrival.parameters()):
            assert name.startswith("multi_arrival."), name


# --- End-to-end through the REAL LocalPropagationFieldGenerator.forward -------------
# The unit tests above drive _MultiArrivalWarp in isolation with the same argument
# tensors the generator passes.  The tests below pin the two LAUNCH-CRITICAL claims at
# the WHOLE-MODEL level, inside the real render pipeline (local_field.py output ->
# temporal_latent -> temporal_operator -> warp -> MULTI_ARRIVAL -> causal gate):
#   * gate_init == 0 is an EXACT no-op vs the warp-only parent (byte-reproducible
#     warm-start) AND leaves the A4 shift heads DEADLOCKED (zero gradient) -- the exact
#     inert failure that killed A3 (frozen zero-init gate on a near-frozen parent);
#   * gate_init == 0.03 BREAKS that deadlock end-to-end: opening the gate changes the
#     forward AND delivers real gradient to the A4 shift/delay heads + path_gate, so the
#     optimizer can actually develop the extra arrival paths.


def _e2e_generator(*, gate_init, paths=2, width=8, height=16, grid_w=16):
    gen = LocalPropagationFieldGenerator(
        width=width, pyramid_levels=2, saved_time_count=401,
        domain_t_s=1.0, domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2), residual=True,
        activation_checkpointing=False,
        warp=True, warp_max_shift_cells=8.0,
        multi_arrival=True, multi_arrival_paths=paths,
        multi_arrival_max_shift_cells=8.0, multi_arrival_max_delay_frac=0.5,
        adapter_gate_init=gate_init,
    )
    # residual=True zero-inits self.output, so a FRESH generator renders an
    # identically-zero coarse field -- and warp_k(0)==0 makes multi_arrival a
    # trivial no-op for the wrong reason (zero field, not zero gate).  The real
    # A4 launch warm-starts self.output from the TRAINED warp_r1/best.pt, whose
    # coarse field is nonzero.  Seed self.output with nonzero weights here so the
    # e2e forward faithfully mirrors that trained parent (a nonzero field that the
    # extra paths can actually transport).
    torch.manual_seed(1234)
    with torch.no_grad():
        gen.output.weight.normal_(0.0, 0.3)
        gen.output.bias.normal_(0.0, 0.1)
    return gen


def _e2e_inputs(gen, *, records=2, count=3, medium_count=2, height=16, grid_w=16, seed=0):
    torch.manual_seed(seed)
    base = gen.width
    pyramid = [torch.randn(medium_count, base, height, grid_w)]
    h, w = height, grid_w
    for _ in range(1, gen.pyramid_levels):
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
    source_parameters = torch.zeros(records, 5)
    source_parameters[:, 2] = 15.0   # frequency
    source_parameters[:, 3] = 0.05   # onset
    # LATE frames so the delayed causal gate of the extra paths is open.
    time_s = torch.full((records, count), 0.9)
    return dict(
        velocity_mps=torch.randn(records, 1, height, grid_w),
        medium=medium, source=source, source_parameters=source_parameters,
        record_to_medium=torch.arange(records) % medium_count,
        time_s=time_s, travel=travel,
        saved_time_indices=torch.randint(0, 401, (records, count)),
    )


def test_full_forward_zero_gate_is_noop_and_shift_heads_are_deadlocked():
    """gate_init == 0: within the REAL forward the extra paths add nothing (output ==
    warp-only parent) AND the A4 TRANSPORT heads (shift + delay) receive IDENTICALLY
    ZERO gradient (dL/d(head) = path_gate * ... == 0) -- the cold-start deadlock that
    keeps the extra arrival paths from ever developing on a near-frozen parent.

    A4-specific nuance (why gate_init still matters even though path_gate is NOT itself
    deadlocked): at zero-init the shift heads make warp_k the IDENTITY, so on the trained
    parent's NONZERO field the extra term is path_gate * causal_k * field -- a pure
    RESCALE of the already-rendered field.  So path_gate DOES receive a (rescale-only)
    gradient here.  But that signal only ever says "scale the existing field," never
    "transport a displaced later arrival": the transport heads stay frozen at zero grad,
    so no genuinely new arrival path can form.  gate_init injects the transport signal.
    """
    gen = _e2e_generator(gate_init=0.0)
    inputs = _e2e_inputs(gen)
    # no-op vs warp-only parent (byte-reproducible warm-start)
    gen.eval()
    with torch.no_grad():
        out_a4 = gen(**inputs)
        saved = gen.multi_arrival
        gen.multi_arrival = None
        out_warp_only = gen(**inputs)
        gen.multi_arrival = saved
    assert out_a4.shape == (2, 3, 16, 16)
    assert torch.isfinite(out_a4).all()
    assert torch.allclose(out_a4, out_warp_only, atol=1e-6)
    # deadlock: EVERY transport (shift + delay) head gets exactly-zero gradient through
    # the real pipeline -- the paths can never learn to place a displaced arrival.
    gen.train()
    out = gen(**inputs)
    out.square().mean().backward()
    for head in gen.multi_arrival.shift_heads:
        last_conv = head[-1]
        assert last_conv.weight.grad is None or last_conv.weight.grad.abs().max() == 0.0
    for delay in gen.multi_arrival.delay_heads:
        assert delay.weight.grad is None or delay.weight.grad.abs().max() == 0.0


def test_full_forward_warm_gate_breaks_deadlock_end_to_end():
    """gate_init == 0.03: opening the gate makes the extra paths contribute WITHIN the
    real forward (output != warp-only) AND delivers real gradient to the A4 shift/delay
    heads + path_gate -- the end-to-end proof the deadlock fix unblocks A4 training."""
    gen = _e2e_generator(gate_init=0.03)
    inputs = _e2e_inputs(gen)
    # output differs from the warp-only parent by a bounded amount (contributes at load)
    gen.eval()
    with torch.no_grad():
        out_a4 = gen(**inputs)
        saved = gen.multi_arrival
        gen.multi_arrival = None
        out_warp_only = gen(**inputs)
        gen.multi_arrival = saved
    assert torch.isfinite(out_a4).all()
    assert (out_a4 - out_warp_only).abs().max() > 1e-5
    # deadlock broken: gradient reaches the zero-init shift/delay heads and the gate
    gen.train()
    out = gen(**inputs)
    out.square().mean().backward()
    last_conv = gen.multi_arrival.shift_heads[0][-1]
    assert last_conv.weight.grad is not None and last_conv.weight.grad.abs().max() > 0.0
    for delay in gen.multi_arrival.delay_heads:
        assert delay.weight.grad is not None and torch.isfinite(delay.weight.grad).all()
    assert gen.multi_arrival.path_gate.grad is not None
    assert gen.multi_arrival.path_gate.grad.abs().max() > 0.0

