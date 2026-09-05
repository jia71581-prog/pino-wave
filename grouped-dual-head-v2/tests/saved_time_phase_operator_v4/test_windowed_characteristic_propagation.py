"""Contract tests for C1 ``_WindowedCharacteristicPropagation``.

The critical guarantee is per-frame query invariance: a single-frame (count=1)
query of frame ``i`` must be bit-identical to that frame inside a multi-frame
batch, because the deterministic neighbor-time window is a function of the saved
index alone -- never of which other frames are in the batch. These tests mirror
``test_temporal_operator_spatial_kernel.py`` and pin the A4 cold-start deadlock
lesson (gate=0 => internals get no gradient; gate>0 => they train).
"""
from __future__ import annotations

import math

import pytest
import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _WindowedCharacteristicPropagation,
)
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


def _inputs(*, records, count, width, height, grid_w, n_saved=401, seed=0):
    gen = torch.Generator().manual_seed(seed)
    field = torch.randn(records, count, height, grid_w, generator=gen)
    rendered = torch.randn(records * count, width, height, grid_w, generator=gen)
    arrival = torch.rand(records, height, grid_w, generator=gen) * 0.3
    # contiguous-ish saved indices well inside [0, n_saved)
    base = torch.arange(count) * 3 + 40
    saved_time_indices = base[None, :].expand(records, count).clone()
    saved_time_values = torch.linspace(0.0, 1.0, n_saved)
    time_s = saved_time_values[saved_time_indices]
    source_parameters = torch.zeros(records, 5)
    source_parameters[:, 2] = 15.0  # f0
    source_parameters[:, 3] = 0.05  # t0
    return (
        field,
        rendered,
        arrival,
        time_s,
        saved_time_indices,
        saved_time_values,
        source_parameters,
    )


def _call(op, field, rendered, arrival, time_s, sti, stv, sp):
    return op(
        field, rendered, arrival, time_s, sti, stv, sp, domain_t_s=1.0
    )


@pytest.mark.parametrize("window", [1, 2])
def test_zero_init_gate_is_exact_noop(window):
    op = _WindowedCharacteristicPropagation(8, window=window, rank=6, gate_init=0.0)
    args = _inputs(records=2, count=3, width=8, height=10, grid_w=10)
    out = _call(op, *args)
    field = args[0]
    assert out.shape == field.shape
    # center tap reproduces the field; gate=0 => out == field bit-for-bit.
    assert torch.equal(out, field)


def test_gate_open_makes_nonzero_finite_change():
    op = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.0)
    with torch.no_grad():
        op.gate.fill_(0.5)
    args = _inputs(records=2, count=3, width=8, height=10, grid_w=10)
    out = _call(op, *args)
    field = args[0]
    assert torch.isfinite(out).all()
    assert (out - field).abs().sum() > 0


@pytest.mark.parametrize("window", [1, 2])
def test_per_frame_query_invariance(window):
    """A single-frame query must match that frame inside a multi-frame batch."""
    op = _WindowedCharacteristicPropagation(8, window=window, rank=6, gate_init=0.0)
    with torch.no_grad():
        op.gate.fill_(0.7)
    op.eval()
    records, count, width, height, grid_w = 1, 5, 8, 9, 9
    field, rendered, arrival, time_s, sti, stv, sp = _inputs(
        records=records, count=count, width=width, height=height, grid_w=grid_w
    )
    full = _call(op, field, rendered, arrival, time_s, sti, stv, sp)
    rendered_by_frame = rendered.reshape(records, count, width, height, grid_w)
    for frame in range(count):  # includes the two edge frames
        single = op(
            field[:, frame : frame + 1],
            rendered_by_frame[:, frame].reshape(records, width, height, grid_w),
            arrival,
            time_s[:, frame : frame + 1],
            sti[:, frame : frame + 1],
            stv,
            sp,
            domain_t_s=1.0,
        )
        assert torch.allclose(full[:, frame : frame + 1], single, atol=1e-6), frame


def test_query_invariance_with_edge_and_noncontiguous_indices():
    """Clamped-window edge frames and non-contiguous saved indices stay invariant."""
    op = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.0)
    with torch.no_grad():
        op.gate.fill_(0.6)
    op.eval()
    records, count, width, height, grid_w, n = 1, 4, 8, 8, 8, 401
    field = torch.randn(records, count, height, grid_w)
    rendered = torch.randn(records * count, width, height, grid_w)
    arrival = torch.rand(records, height, grid_w) * 0.3
    stv = torch.linspace(0.0, 1.0, n)
    # k=0 (left edge, clamped), k=400 (right edge, clamped), plus non-contiguous.
    sti = torch.tensor([[0, 137, 251, 400]])
    time_s = stv[sti]
    sp = torch.zeros(records, 5)
    sp[:, 2] = 12.0
    sp[:, 3] = 0.05
    full = _call(op, field, rendered, arrival, time_s, sti, stv, sp)
    rbf = rendered.reshape(records, count, width, height, grid_w)
    for frame in range(count):
        single = op(
            field[:, frame : frame + 1],
            rbf[:, frame].reshape(records, width, height, grid_w),
            arrival,
            time_s[:, frame : frame + 1],
            sti[:, frame : frame + 1],
            stv,
            sp,
            domain_t_s=1.0,
        )
        assert torch.allclose(full[:, frame : frame + 1], single, atol=1e-6), frame


def test_query_axis_permutation_invariance():
    """Permuting other queried frames does not change a fixed frame's output."""
    op = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.0)
    with torch.no_grad():
        op.gate.fill_(0.5)
    op.eval()
    records, count, width, height, grid_w = 1, 4, 8, 8, 8
    field, rendered, arrival, time_s, sti, stv, sp = _inputs(
        records=records, count=count, width=width, height=height, grid_w=grid_w
    )
    full = _call(op, field, rendered, arrival, time_s, sti, stv, sp)
    perm = torch.tensor([2, 0, 3, 1])
    rbf = rendered.reshape(records, count, width, height, grid_w)
    permuted = op(
        field[:, perm],
        rbf[:, perm].reshape(records * count, width, height, grid_w),
        arrival,
        time_s[:, perm],
        sti[:, perm],
        stv,
        sp,
        domain_t_s=1.0,
    )
    # frame originally at index 3 is at position 2 in the permuted batch.
    assert torch.allclose(full[:, 3], permuted[:, 2], atol=1e-6)


def test_gate_zero_starves_internal_gradient_deadlock():
    op = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.0)
    args = _inputs(records=1, count=3, width=8, height=8, grid_w=8)
    out = _call(op, *args)
    (out.sum()).backward()
    # gate=0 => multiplicative residual gives internals zero gradient (deadlock pin).
    for p in op.spatial_basis.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0
    for p in op.time_trunk.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_positive_gate_init_breaks_deadlock_internals_get_gradient():
    op = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.03)
    args = _inputs(records=1, count=3, width=8, height=8, grid_w=8)
    out = _call(op, *args)
    (out.sum()).backward()
    basis_grad = sum(
        float(p.grad.abs().sum()) for p in op.spatial_basis.parameters() if p.grad is not None
    )
    trunk_grad = sum(
        float(p.grad.abs().sum()) for p in op.time_trunk.parameters() if p.grad is not None
    )
    assert basis_grad > 0.0
    assert trunk_grad > 0.0


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        _WindowedCharacteristicPropagation(8, window=0)
    with pytest.raises(ValueError):
        _WindowedCharacteristicPropagation(8, stride=0)
    with pytest.raises(ValueError):
        _WindowedCharacteristicPropagation(8, rank=0)
    with pytest.raises(ValueError):
        _WindowedCharacteristicPropagation(8, max_advect_cells=0.0)
    with pytest.raises(ValueError):
        _WindowedCharacteristicPropagation(8, gate_init=-0.1)


def test_float64_saved_time_values_are_cast():
    """The stored-time axis buffer is float64 in the real model; the module must cast
    it so grid_sample/einsum see float32 (regression for the smoke dtype crash)."""
    op = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.3)
    field, rendered, arrival, time_s, sti, stv, sp = _inputs(
        records=1, count=3, width=8, height=8, grid_w=8
    )
    stv64 = stv.double()  # float64 stored-time axis, as in the real buffer
    out = op(field, rendered, arrival, time_s, sti, stv64, sp, domain_t_s=1.0)
    assert out.dtype == field.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_frame_chunk_matches_unchunked():
    """Chunked validation path must equal the single-shot path bit-for-bit."""
    op_a = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.4, frame_chunk=64)
    op_b = _WindowedCharacteristicPropagation(8, window=2, rank=6, gate_init=0.4, frame_chunk=2)
    op_b.load_state_dict(op_a.state_dict())
    op_a.eval()
    op_b.eval()
    args = _inputs(records=1, count=7, width=8, height=8, grid_w=8)
    out_a = _call(op_a, *args)
    out_b = _call(op_b, *args)
    assert torch.allclose(out_a, out_b, atol=1e-6)


# --- End-to-end through the REAL LocalPropagationFieldGenerator.forward -------------
# Pins the two launch-critical whole-model claims: (1) gate_init=0 is an exact no-op
# vs the coupler-off parent (byte-reproducible warm-start) AND leaves the C1 internals
# deadlocked (zero grad); (2) gate_init>0 breaks the deadlock end-to-end; plus the
# contract枢纽: per-frame query invariance through the real render pipeline.


def _e2e_generator(*, gate_init, width=8, height=16, grid_w=16):
    gen = LocalPropagationFieldGenerator(
        width=width, pyramid_levels=2, saved_time_count=401,
        domain_t_s=1.0, domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2), residual=True,
        activation_checkpointing=False,
        warp=True, warp_max_shift_cells=8.0,
        windowed_propagation=True, windowed_propagation_window=2,
        windowed_propagation_rank=6, windowed_propagation_max_advect_cells=8.0,
        adapter_gate_init=gate_init,
    )
    # mirror the trained parent: nonzero coarse field the coupler can transport.
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
    source_parameters[:, 2] = 15.0
    source_parameters[:, 3] = 0.05
    saved_time_values = torch.linspace(0.0, 1.0, 401)
    saved_time_indices = (torch.arange(count) * 3 + 200)[None, :].expand(records, count).clone()
    time_s = saved_time_values[saved_time_indices]
    return dict(
        velocity_mps=torch.randn(records, 1, height, grid_w),
        medium=medium, source=source, source_parameters=source_parameters,
        record_to_medium=torch.arange(records) % medium_count,
        time_s=time_s, travel=travel,
        saved_time_indices=saved_time_indices,
        saved_time_values=saved_time_values,
    )


def test_full_forward_zero_gate_is_noop_and_internals_deadlocked():
    gen = _e2e_generator(gate_init=0.0)
    inputs = _e2e_inputs(gen)
    gen.eval()
    with torch.no_grad():
        out_c1 = gen(**inputs)
        saved = gen.windowed_propagation
        gen.windowed_propagation = None
        out_off = gen(**inputs)
        gen.windowed_propagation = saved
    assert out_c1.shape == (2, 3, 16, 16)
    assert torch.isfinite(out_c1).all()
    assert torch.allclose(out_c1, out_off, atol=1e-6)  # byte-repro warm-start
    gen.train()
    out = gen(**inputs)
    out.square().mean().backward()
    for p in gen.windowed_propagation.spatial_basis.parameters():
        assert p.grad is None or p.grad.abs().max() == 0.0
    for p in gen.windowed_propagation.time_trunk.parameters():
        assert p.grad is None or p.grad.abs().max() == 0.0


def test_full_forward_positive_gate_breaks_deadlock():
    gen = _e2e_generator(gate_init=0.02)
    inputs = _e2e_inputs(gen)
    gen.eval()
    with torch.no_grad():
        out_c1 = gen(**inputs)
        saved = gen.windowed_propagation
        gen.windowed_propagation = None
        out_off = gen(**inputs)
        gen.windowed_propagation = saved
    assert not torch.allclose(out_c1, out_off, atol=1e-6)  # gate open changes forward
    gen.train()
    out = gen(**inputs)
    out.square().mean().backward()
    basis_grad = sum(
        float(p.grad.abs().sum())
        for p in gen.windowed_propagation.spatial_basis.parameters()
        if p.grad is not None
    )
    assert basis_grad > 0.0


def test_full_forward_per_frame_query_invariance():
    """The contract枢纽 at whole-model level: single-frame query == batch frame."""
    gen = _e2e_generator(gate_init=0.02)
    gen.eval()
    inputs = _e2e_inputs(gen, records=1, count=5)
    with torch.no_grad():
        full = gen(**inputs)
        for frame in range(inputs["time_s"].shape[1]):
            single_inputs = dict(inputs)
            single_inputs["time_s"] = inputs["time_s"][:, frame : frame + 1]
            single_inputs["saved_time_indices"] = inputs["saved_time_indices"][:, frame : frame + 1]
            single = gen(**single_inputs)
            assert torch.allclose(full[:, frame : frame + 1], single, atol=1e-5), frame
