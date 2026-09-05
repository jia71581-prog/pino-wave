"""Unit tests for the r4 arrival-aligned characteristic warp.

The isolated bottleneck is wavefront POSITION/PHASE: the model places the moving
front slightly off its true location, and an amplitude operator (even the r3
receptive-field one) cannot fix a *displacement*.  ``_EikonalArrivalWarp`` resamples
the field along the propagation characteristic ``d = grad(T)/||grad(T)||`` by a bounded
zero-init shift, so it can reposition the wavefront while staying an exact warm-start
no-op.  These tests pin:
  * zero-init shift head -> identity warp at warm-start (exact no-op, byte reproducible),
  * a learned nonzero shift actually moves energy along grad(T) (transport, not rescale),
  * the shift is bounded by ``max_shift_cells``,
  * per-frame query invariance (grid_sample acts per batch element; no cross-frame mix),
  * gradients flow to the shift head,
  * invalid bounds are rejected.
"""
from __future__ import annotations

import pytest
import torch

from saved_time_phase_operator_v4.local_field import _EikonalArrivalWarp


def _field_and_features(*, records: int, count: int, width: int, height: int, grid_w: int):
    field = torch.randn(records, count, height, grid_w)
    rendered = torch.randn(records * count, width, height, grid_w)
    return field, rendered


def _radial_arrival(records: int, height: int, grid_w: int) -> torch.Tensor:
    """Travel time increasing with distance from the top-left corner (grad(T) != 0)."""
    zs = torch.arange(height, dtype=torch.float32)[:, None]
    xs = torch.arange(grid_w, dtype=torch.float32)[None, :]
    dist = torch.sqrt(zs * zs + xs * xs)
    return dist[None].repeat(records, 1, 1) * 1e-3


def test_zero_init_is_exact_identity_noop():
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    field, rendered = _field_and_features(records=2, count=3, width=8, height=7, grid_w=9)
    arrival = _radial_arrival(2, 7, 9)
    out = warp(field, rendered, arrival)
    assert out.shape == field.shape
    # zero-init shift head -> shift == 0 -> identity sampling grid -> exact copy.
    assert torch.allclose(out, field, atol=1e-6)


def test_shift_head_final_conv_is_zero_initialized():
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    final = warp.shift_head[-1]
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0


def test_learned_shift_moves_energy_along_characteristic():
    """A nonzero shift must actually translate the field (not just rescale it)."""
    warp = _EikonalArrivalWarp(4, max_shift_cells=6.0)
    # Force a uniform positive shift by biasing the final conv output.
    with torch.no_grad():
        warp.shift_head[-1].bias.fill_(5.0)  # tanh saturates -> near-max positive shift
    records, count, height, grid_w = 1, 1, 24, 24
    # A localized bump field; shifting it along grad(T) moves the peak.
    field = torch.zeros(records, count, height, grid_w)
    field[0, 0, 12, 12] = 1.0
    rendered = torch.zeros(records * count, 4, height, grid_w)
    arrival = _radial_arrival(records, height, grid_w)
    out = warp(field, rendered, arrival)
    # energy is conserved-ish but the peak location shifts off (12,12).
    peak = torch.argmax(out[0, 0])
    pz, px = int(peak // grid_w), int(peak % grid_w)
    assert (pz, px) != (12, 12)
    assert out.abs().sum() > 0


def test_shift_is_bounded_by_max_shift_cells():
    warp = _EikonalArrivalWarp(4, max_shift_cells=3.0)
    with torch.no_grad():
        warp.shift_head[-1].bias.fill_(100.0)  # saturate tanh
    rendered = torch.randn(6, 4, 10, 10)
    shift = warp.max_shift_cells * torch.tanh(warp.shift_head(rendered))
    assert shift.abs().max() <= 3.0 + 1e-5


def test_per_frame_query_invariance():
    """A single-frame query must match that frame inside a multi-frame batch."""
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():
        warp.shift_head[-1].bias.fill_(2.0)  # nonzero shift so the test is meaningful
    warp.eval()
    records, count, height, grid_w = 1, 4, 6, 6
    field, rendered = _field_and_features(
        records=records, count=count, width=8, height=height, grid_w=grid_w
    )
    arrival = _radial_arrival(records, height, grid_w)
    full = warp(field, rendered, arrival)
    frame = 2
    single = warp(
        field[:, frame : frame + 1],
        rendered[frame : frame + 1],
        arrival,
        )
    assert torch.allclose(full[:, frame], single[:, 0], atol=1e-6)


def test_gradients_flow_to_shift_head():
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():
        warp.shift_head[-1].bias.fill_(1.0)
    field, rendered = _field_and_features(records=2, count=2, width=8, height=8, grid_w=8)
    rendered.requires_grad_(True)
    arrival = _radial_arrival(2, 8, 8)
    out = warp(field, rendered, arrival)
    out.square().mean().backward()
    for conv in (warp.shift_head[0], warp.shift_head[-1]):
        assert conv.weight.grad is not None
        assert torch.isfinite(conv.weight.grad).all()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()


@pytest.mark.parametrize("bad", [0.0, -1.0, -8.0])
def test_invalid_max_shift_rejected(bad):
    with pytest.raises(ValueError, match="max_shift_cells"):
        _EikonalArrivalWarp(8, max_shift_cells=bad)


def test_finite_output_on_random_inputs():
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():
        warp.shift_head[-1].weight.normal_(0.0, 0.1)
        warp.shift_head[-1].bias.normal_(0.0, 1.0)
    field, rendered = _field_and_features(records=3, count=2, width=8, height=11, grid_w=13)
    arrival = _radial_arrival(3, 11, 13)
    out = warp(field, rendered, arrival)
    assert out.shape == (3, 2, 11, 13)
    assert torch.isfinite(out).all()


# --- controlled A2: shift-head receptive-field dilation ------------------------
# The diagnosed bottleneck is wavefront POSITION.  A displacement of the front is
# a function of *surrounding* structure, so the shift head benefits from a wider
# receptive field.  ``shift_dilation`` widens it (RF 5/9/13 for dilation 1/2/3)
# at ZERO extra params/activation and unchanged 3x3 weight shapes, so a warp
# checkpoint trained at any dilation loads unchanged.  Default dilation=1 must be
# byte-identical to the original RF=5 head.


def test_default_dilation_is_one_byte_identical_construction():
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    assert warp.shift_dilation == 1
    for conv in (warp.shift_head[0], warp.shift_head[-1]):
        assert conv.dilation == (1, 1)
        assert conv.padding == (1, 1)
        assert tuple(conv.weight.shape[-2:]) == (3, 3)


@pytest.mark.parametrize("dilation", [2, 3])
def test_dilation_sets_conv_dilation_and_preserves_shapes(dilation):
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0, shift_dilation=dilation)
    assert warp.shift_dilation == dilation
    for conv in (warp.shift_head[0], warp.shift_head[-1]):
        assert conv.dilation == (dilation, dilation)
        assert conv.padding == (dilation, dilation)
        # weight SHAPE is unchanged (still 3x3): no extra params vs dilation=1.
        assert tuple(conv.weight.shape[-2:]) == (3, 3)


@pytest.mark.parametrize("dilation", [2, 3])
def test_dilated_head_loads_dilation1_checkpoint_unchanged(dilation):
    """A warp_r1 checkpoint (trained at dilation=1) loads into a dilated head
    with no shape mismatch -- the only difference is where the kernel taps."""
    trained = _EikonalArrivalWarp(8, max_shift_cells=8.0)
    with torch.no_grad():  # give it a nonzero trained-like state
        trained.shift_head[0].weight.normal_(0.0, 0.2)
        trained.shift_head[-1].weight.normal_(0.0, 0.2)
        trained.shift_head[-1].bias.normal_(0.0, 0.1)
    dilated = _EikonalArrivalWarp(8, max_shift_cells=8.0, shift_dilation=dilation)
    missing, unexpected = dilated.load_state_dict(trained.state_dict(), strict=True)
    assert not missing and not unexpected
    for name, p in dilated.state_dict().items():
        assert torch.equal(p, trained.state_dict()[name])


@pytest.mark.parametrize("dilation", [1, 2, 3])
def test_zero_init_noop_preserved_at_any_dilation(dilation):
    warp = _EikonalArrivalWarp(8, max_shift_cells=8.0, shift_dilation=dilation)
    field, rendered = _field_and_features(records=2, count=3, width=8, height=9, grid_w=11)
    arrival = _radial_arrival(2, 9, 11)
    out = warp(field, rendered, arrival)
    assert out.shape == field.shape
    assert torch.allclose(out, field, atol=1e-6)  # warm-start exact no-op


def test_dilation_widens_effective_receptive_field():
    """A single-pixel feature perturbation influences the shift response over a
    strictly wider spatial radius at dilation=2 than at dilation=1."""
    height = grid_w = 21
    center = 10

    def _response_radius(dilation: int) -> int:
        warp = _EikonalArrivalWarp(4, max_shift_cells=8.0, shift_dilation=dilation)
        with torch.no_grad():  # make the (zero-init) final conv sensitive
            warp.shift_head[-1].weight.fill_(1.0)
            warp.shift_head[0].weight.fill_(1.0)
        base = torch.zeros(1, 4, height, grid_w)
        bumped = base.clone()
        bumped[0, :, center, center] = 1.0
        with torch.no_grad():
            r0 = warp.shift_head(base)
            r1 = warp.shift_head(bumped)
        diff = (r1 - r0).abs()[0, 0]  # (H, W)
        nz = torch.nonzero(diff > 1e-8)
        radius = int((nz[:, 0] - center).abs().max().item())
        return radius

    r_d1 = _response_radius(1)
    r_d2 = _response_radius(2)
    assert r_d2 > r_d1  # dilation genuinely enlarges the receptive field


@pytest.mark.parametrize("bad", [0, -1, -3])
def test_invalid_dilation_rejected(bad):
    with pytest.raises(ValueError, match="shift_dilation"):
        _EikonalArrivalWarp(8, max_shift_cells=8.0, shift_dilation=bad)
