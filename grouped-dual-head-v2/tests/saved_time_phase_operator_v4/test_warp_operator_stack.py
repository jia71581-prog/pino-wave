"""B1 (r5 stack): warp + query-invariant temporal operator, composed.

warp_r1 isolated the arrival-aligned warp (transport) and reached full-panel
agg 0.2401 / phase 0.7998 from the opt16 floor (0.2530 / 0.7847), beating opt16
strict on BOTH agg and phase.  The residual bottleneck is LAYERED media at LATE
time (family layered 0.289, time_bin late 0.343) -- multiple reflections /
reverberation that a single per-pixel advective shift (warp) cannot represent.
B1 stacks the per-frame receptive-field temporal operator ON TOP of the warp so
the warp positions the primary arrival while the operator adds the late scattered /
reverberant amplitude the warp cannot.  Both modules already exist and compose in
``LocalPropagationFieldGenerator.forward`` as ``field += operator(...); field =
warp(field, ...)`` (local_field.py:490-509).

These tests pin the COMPOSITION (the only new thing B1 introduces -- each module is
already unit-tested in isolation):
  * warp + operator, both zero-init, is an EXACT joint no-op at warm-start,
  * per-frame query invariance is preserved through the composition with BOTH gates
    open (the ordering operator-then-warp introduces no cross-frame coupling),
  * a generator built with both variant flags constructs both submodules, both no-op
    at init (so a warm-start from a parent predating them is byte-reproducible).
"""
from __future__ import annotations

import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _ContinuousTimePropagationBasis,
    _EikonalArrivalWarp,
)
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


def _radial_arrival(records: int, height: int, grid_w: int) -> torch.Tensor:
    zs = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(grid_w, dtype=torch.float32)[None, :]
    dist = torch.sqrt(zs * zs + xs * xs)
    return dist[None].repeat(records, 1, 1) * 1e-3


def _stack_inputs(*, records, count, width, height, grid_w, seed=0):
    torch.manual_seed(seed)
    field = torch.randn(records, count, height, grid_w)
    rendered = torch.randn(records * count, width, height, grid_w)
    time_s = (torch.rand(records, count) * 0.5 + 0.1).sort(dim=1).values
    source = torch.zeros(records, 5)
    source[:, 2] = 15.0   # frequency
    source[:, 3] = 0.05   # onset
    arrival = _radial_arrival(records, height, grid_w)
    return field, rendered, time_s, source, arrival


def _compose(op, warp, field, rendered, time_s, source, arrival, *, records, count):
    """Replicate LocalPropagationFieldGenerator.forward operator->warp ordering."""
    field = field + op(rendered, time_s, source, records=records, count=count, domain_t_s=1.0)
    field = warp(field, rendered, arrival)
    return field


def test_warp_operator_stack_is_exact_joint_noop_at_warmstart():
    """Both modules zero-init -> the stacked residual is an exact no-op on the field."""
    op = _ContinuousTimePropagationBasis(8, rank=16, spatial_kernel=3)   # gate=0 at init
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)                   # shift head zero-init
    field, rendered, time_s, source, arrival = _stack_inputs(
        records=2, count=3, width=8, height=7, grid_w=9
    )
    out = _compose(op, warp, field, rendered, time_s, source, arrival, records=2, count=3)
    assert out.shape == field.shape
    # operator adds zero (gate=0) and the warp is the identity resample (shift=0):
    assert torch.allclose(out, field, atol=1e-6)


def test_warp_operator_stack_preserves_per_frame_query_invariance():
    """With BOTH gates open, a single-frame query must match that frame in a batch."""
    op = _ContinuousTimePropagationBasis(8, rank=16, spatial_kernel=3)
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():
        op.gate.fill_(0.7)                       # operator active
        warp.shift_head[-1].bias.fill_(2.0)      # warp active (nonzero shift)
    op.eval(); warp.eval()
    records, count, height, grid_w = 1, 4, 6, 6
    field, rendered, time_s, source, arrival = _stack_inputs(
        records=records, count=count, width=8, height=height, grid_w=grid_w
    )
    full = _compose(op, warp, field, rendered, time_s, source, arrival,
                    records=records, count=count)
    frame = 2
    single = _compose(
        op, warp,
        field[:, frame : frame + 1],
        rendered[frame : frame + 1],
        time_s[:, frame : frame + 1],
        source,
        arrival,
        records=records, count=1,
    )
    assert torch.allclose(full[:, frame], single[:, 0], atol=1e-6)


def test_warp_operator_stack_is_nonzero_and_finite_once_gates_open():
    op = _ContinuousTimePropagationBasis(8, rank=16, spatial_kernel=3)
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():
        op.gate.fill_(1.0)
        warp.shift_head[-1].bias.fill_(2.0)
    field, rendered, time_s, source, arrival = _stack_inputs(
        records=2, count=3, width=8, height=7, grid_w=9
    )
    out = _compose(op, warp, field, rendered, time_s, source, arrival, records=2, count=3)
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0
    # the composition genuinely differs from the field once active (not a no-op).
    assert not torch.allclose(out, field, atol=1e-4)


def test_generator_builds_both_warp_and_operator_as_noops():
    """The B1 variant flags (warp + operator rank>0) construct both submodules, both
    exact no-ops at init so a warm-start from an opt16/warp parent is byte-reproducible."""
    g = LocalPropagationFieldGenerator(
        width=16, pyramid_levels=4, saved_time_count=401,
        domain_t_s=1.0, domain_diagonal_m=2828.0,
        residual=True,
        temporal_operator_rank=16, temporal_operator_spatial_kernel=3,
        warp=True, warp_max_shift_cells=8.0,
    )
    assert g.temporal_operator is not None
    assert g.warp is not None
    # operator gate zero-init -> no-op
    assert float(g.temporal_operator.gate) == 0.0
    # warp final conv zero-init -> identity resample
    assert torch.count_nonzero(g.warp.shift_head[-1].weight) == 0
    assert torch.count_nonzero(g.warp.shift_head[-1].bias) == 0


def test_stack_gradients_flow_to_both_modules():
    op = _ContinuousTimePropagationBasis(8, rank=16, spatial_kernel=3)
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():
        op.gate.fill_(1.0)
        warp.shift_head[-1].bias.fill_(1.0)
    field, rendered, time_s, source, arrival = _stack_inputs(
        records=2, count=2, width=8, height=8, grid_w=8
    )
    rendered.requires_grad_(True)
    out = _compose(op, warp, field, rendered, time_s, source, arrival, records=2, count=2)
    out.square().mean().backward()
    assert op.gate.grad is not None and torch.isfinite(op.gate.grad).all()
    assert warp.shift_head[0].weight.grad is not None
    assert torch.isfinite(warp.shift_head[0].weight.grad).all()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()


# --- End-to-end through the REAL LocalPropagationFieldGenerator.forward -------------
# The tests above replicate the op->warp ordering by hand (_compose).  The tests below
# drive the actual forward (output -> +operator -> warp -> causal gate, local_field.py
# :488-514) with synthetic-but-valid encodings, to pin the two claims the B1 config
# makes at the whole-model level:
#   * at zero-init the operator is an EXACT no-op WITHIN the real forward, so a warp
#     parent (which lacks the operator) is byte-reproducible at warm-start;
#   * the operator is wired LIVE into that forward -- opening its gate changes the output.


def _b1_generator(*, width=8, pyramid_levels=2, saved_time_count=401, rank=8, sk=3):
    return LocalPropagationFieldGenerator(
        width=width, pyramid_levels=pyramid_levels, saved_time_count=saved_time_count,
        domain_t_s=1.0, domain_diagonal_m=2828.0, residual=True,
        temporal_operator_rank=rank, temporal_operator_spatial_kernel=sk,
        warp=True, warp_max_shift_cells=8.0,
    )


def _forward_inputs(gen, *, records, count, medium_count, height, grid_w, seed=0):
    """Synthetic-but-shape-valid encodings for LocalPropagationFieldGenerator.forward."""
    torch.manual_seed(seed)
    base = gen.width
    levels = gen.pyramid_levels
    # pyramid[0] must be full-res; deeper levels are interpolated up in forward.
    pyramid = [torch.randn(medium_count, base, height, grid_w)]
    h, w = height, grid_w
    for _ in range(1, levels):
        h = max(1, h // 2); w = max(1, w // 2)
        pyramid.append(torch.randn(medium_count, base, h, w))
    medium = MediumEncoding(pyramid=tuple(pyramid), tokens=None, token_positions=None, rank=None)
    source = SourceEncoding(
        hidden=torch.randn(records, base),
        rank=None,
        map_field=torch.randn(records, base, height, grid_w),
        local_medium=None,
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
    source_parameters[:, 2] = 15.0   # frequency
    source_parameters[:, 3] = 0.05   # onset
    record_to_medium = torch.arange(records) % medium_count
    time_s = (torch.rand(records, count) * 0.4 + 0.1).sort(dim=1).values
    saved_time_indices = torch.randint(0, 401, (records, count))
    return dict(
        velocity_mps=velocity_mps, medium=medium, source=source,
        source_parameters=source_parameters, record_to_medium=record_to_medium,
        time_s=time_s, travel=travel, saved_time_indices=saved_time_indices,
    )


def test_full_forward_zero_init_operator_is_exact_noop_vs_warp_only():
    """Within the REAL forward, the zero-init operator adds nothing -> the B1 model's
    output equals the identical model with the operator disabled (the warp-only parent
    path). This is the warm-start byte-reproducibility the single-factor claim rests on."""
    gen = _b1_generator()
    gen.eval()
    inputs = _forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16)
    with torch.no_grad():
        out_b1 = gen(**inputs)                 # operator present, gate zero-init
        saved = gen.temporal_operator
        gen.temporal_operator = None           # emulate the warp-only parent forward
        out_warp_only = gen(**inputs)
        gen.temporal_operator = saved
    assert out_b1.shape == (2, 3, 16, 16)
    assert torch.isfinite(out_b1).all()
    assert torch.allclose(out_b1, out_warp_only, atol=1e-6)


def test_full_forward_operator_is_live_once_gate_opens():
    """Opening the operator gate changes the real forward output -> the operator is
    genuinely wired into the trained path (not dead code bypassed by the warp)."""
    gen = _b1_generator()
    gen.eval()
    inputs = _forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16)
    with torch.no_grad():
        out_noop = gen(**inputs)
        gen.temporal_operator.gate.fill_(0.8)   # operator now contributes
        out_active = gen(**inputs)
    assert torch.isfinite(out_active).all()
    assert not torch.allclose(out_active, out_noop, atol=1e-4)
