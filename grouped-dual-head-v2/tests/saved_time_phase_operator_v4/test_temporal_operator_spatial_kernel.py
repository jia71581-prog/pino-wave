"""Unit tests for the receptive-field pivot of the query-invariant temporal operator.

The plateau diagnosis (tempop_r2: agg flat at the parent floor 0.2449 over 4 epochs)
traced to `_ContinuousTimePropagationBasis.spatial_coefficient` being a 1x1 conv --
zero receptive field, so the gated residual can only rescale amplitude in place, never
translate the wavefront (the confirmed position/phase bottleneck). The pivot adds a
config-gated ``spatial_kernel`` that swaps the 1x1 conv for a two-layer k x k stack
(RF = 2k-1) so the operator can gather from a spatial neighbourhood. These tests pin:
  * ``spatial_kernel=1`` reproduces the exact 1x1 operator (backward compatibility),
  * ``spatial_kernel>1`` builds the receptive-field stack and stays an exact warm-start
    no-op (zero-init gate),
  * per-frame query invariance is preserved for both kernels,
  * invalid kernels are rejected.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from saved_time_phase_operator_v4.local_field import _ContinuousTimePropagationBasis


def _inputs(*, records: int, count: int, width: int, height: int, grid_w: int):
    rendered = torch.randn(records * count, width, height, grid_w)
    time_s = torch.rand(records, count) * 0.5 + 0.1
    # source_parameters columns: [_, _, frequency(2), onset(3), _]
    source_parameters = torch.zeros(records, 5)
    source_parameters[:, 2] = 15.0
    source_parameters[:, 3] = 0.05
    return rendered, time_s, source_parameters


def test_spatial_kernel_one_is_pointwise_conv_backward_compatible():
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=1)
    assert isinstance(op.spatial_coefficient, nn.Conv2d)
    assert op.spatial_coefficient.kernel_size == (1, 1)


def test_spatial_kernel_three_builds_receptive_field_stack():
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=3)
    assert isinstance(op.spatial_coefficient, nn.Sequential)
    convs = [m for m in op.spatial_coefficient if isinstance(m, nn.Conv2d)]
    assert len(convs) == 2
    assert all(c.kernel_size == (3, 3) for c in convs)
    # padding preserves spatial resolution (same-size convs).
    assert all(c.padding == (1, 1) for c in convs)
    # first conv bottlenecks width->rank (memory-frugal intermediate); both convs
    # then carry `rank` channels, with the last emitting the rank-channel basis.
    assert convs[0].in_channels == 8
    assert convs[0].out_channels == 4
    assert convs[-1].out_channels == 4


@pytest.mark.parametrize("spatial_kernel", [1, 3, 5])
def test_zero_init_gate_makes_operator_exact_noop_at_warmstart(spatial_kernel):
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=spatial_kernel)
    rendered, time_s, source = _inputs(records=2, count=3, width=8, height=7, grid_w=9)
    out = op(rendered, time_s, source, records=2, count=3, domain_t_s=1.0)
    assert out.shape == (2, 3, 7, 9)
    assert torch.count_nonzero(out) == 0  # gate is zero at init -> exact no-op


@pytest.mark.parametrize("spatial_kernel", [1, 3])
def test_forward_is_nonzero_once_gate_opens(spatial_kernel):
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=spatial_kernel)
    with torch.no_grad():
        op.gate.fill_(1.0)
    rendered, time_s, source = _inputs(records=2, count=3, width=8, height=7, grid_w=9)
    out = op(rendered, time_s, source, records=2, count=3, domain_t_s=1.0)
    assert out.shape == (2, 3, 7, 9)
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0


@pytest.mark.parametrize("spatial_kernel", [1, 3])
def test_per_frame_query_invariance(spatial_kernel):
    """A single-frame query must match that frame inside a multi-frame batch."""
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=spatial_kernel)
    with torch.no_grad():
        op.gate.fill_(0.7)
    op.eval()
    records, count = 1, 4
    rendered, time_s, source = _inputs(
        records=records, count=count, width=8, height=6, grid_w=6
    )
    full = op(rendered, time_s, source, records=records, count=count, domain_t_s=1.0)
    # query frame index 2 alone; its rendered slice is row (0*count + 2).
    frame = 2
    single = op(
        rendered[frame : frame + 1],
        time_s[:, frame : frame + 1],
        source,
        records=records,
        count=1,
        domain_t_s=1.0,
    )
    assert torch.allclose(full[:, frame], single[:, 0], atol=1e-6)


@pytest.mark.parametrize("bad_kernel", [0, -1, 2, 4])
def test_invalid_spatial_kernel_rejected(bad_kernel):
    with pytest.raises(ValueError, match="spatial_kernel"):
        _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=bad_kernel)


def test_gradients_flow_to_receptive_field_stack():
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=3)
    with torch.no_grad():
        op.gate.fill_(1.0)
    rendered, time_s, source = _inputs(records=2, count=2, width=8, height=5, grid_w=5)
    rendered.requires_grad_(True)
    out = op(rendered, time_s, source, records=2, count=2, domain_t_s=1.0)
    out.square().mean().backward()
    convs = [m for m in op.spatial_coefficient if isinstance(m, nn.Conv2d)]
    for c in convs:
        assert c.weight.grad is not None
        assert torch.isfinite(c.weight.grad).all()
    assert op.gate.grad is not None and torch.isfinite(op.gate.grad).all()


def test_default_gate_init_is_zero_byte_reproducible_noop():
    """Default gate_init=0.0 keeps the exact old behaviour (self.gate == 0 scalar)."""
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=3)
    assert op.gate.shape == torch.Size([])
    assert float(op.gate) == 0.0


@pytest.mark.parametrize("bad", [-0.01, -1.0])
def test_negative_gate_init_rejected(bad):
    with pytest.raises(ValueError, match="gate_init"):
        _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=3, gate_init=bad)


def test_gate_init_zero_starves_internal_gradient_deadlock():
    """The B1 cold-start deadlock, pinned at the weight level: with a zero multiplicative
    gate, residual = gate * f, so dL/d(time_trunk) = dL/d(spatial_coefficient) = gate * ...
    == 0.  The internals receive EXACTLY zero gradient and can never develop -- exactly why
    tempop_warp_b1's gate stalled at -9e-5 through ep8.  (The gate scalar itself still gets
    dL/dgate = <f, dL/dout>, which is why it drifts to ~-1e-4, not the trunk.)"""
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=3, gate_init=0.0)
    rendered, time_s, source = _inputs(records=2, count=3, width=8, height=7, grid_w=9)
    out = op(rendered, time_s, source, records=2, count=3, domain_t_s=1.0)
    out.square().mean().backward()
    convs = [m for m in op.spatial_coefficient if isinstance(m, nn.Conv2d)]
    trunk_linears = [m for m in op.time_trunk if isinstance(m, nn.Linear)]
    for c in convs:
        assert c.weight.grad is None or float(c.weight.grad.abs().sum()) == 0.0
    for lin in trunk_linears:
        assert lin.weight.grad is None or float(lin.weight.grad.abs().sum()) == 0.0


def test_positive_gate_init_breaks_deadlock_internals_get_gradient():
    """A small POSITIVE gate_init gives the time_trunk AND spatial_coefficient real,
    finite, nonzero gradient at step 0 -- the fix that makes B1-warm actually trainable."""
    op = _ContinuousTimePropagationBasis(8, rank=4, spatial_kernel=3, gate_init=0.03)
    assert float(op.gate) == pytest.approx(0.03)
    rendered, time_s, source = _inputs(records=2, count=3, width=8, height=7, grid_w=9)
    out = op(rendered, time_s, source, records=2, count=3, domain_t_s=1.0)
    assert out.abs().sum() > 0  # no longer a no-op
    out.square().mean().backward()
    convs = [m for m in op.spatial_coefficient if isinstance(m, nn.Conv2d)]
    trunk_linears = [m for m in op.time_trunk if isinstance(m, nn.Linear)]
    for c in convs:
        assert c.weight.grad is not None and torch.isfinite(c.weight.grad).all()
        assert float(c.weight.grad.abs().sum()) > 0.0
    for lin in trunk_linears:
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
        assert float(lin.weight.grad.abs().sum()) > 0.0
