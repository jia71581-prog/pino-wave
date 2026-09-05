"""Late-time strengthening of the local-propagation field: extended features."""
from __future__ import annotations

import torch

from saved_time_phase_operator_v4.local_field import LocalPropagationFieldGenerator


def _gen(extended: bool) -> LocalPropagationFieldGenerator:
    return LocalPropagationFieldGenerator(
        width=16, pyramid_levels=4, saved_time_count=401,
        domain_t_s=1.0, domain_diagonal_m=2828.0,
        extended_late_features=extended,
    )


def test_extended_features_add_one_zero_initialized_channel():
    g = _gen(True)
    assert g.phase_projection.in_channels == 13
    # the 13th (travel-progress) input weight is zero so a warm start is a no-op
    assert torch.count_nonzero(g.phase_projection.weight[:, 12:]) == 0
    # the original 12 channels keep a normal (nonzero) initialization
    assert torch.count_nonzero(g.phase_projection.weight[:, :12]) > 0


def test_default_generator_keeps_twelve_channels():
    assert _gen(False).phase_projection.in_channels == 12


from saved_time_phase_operator_v4.local_field import _ContinuousTimePropagationBasis


def _basis():
    return _ContinuousTimePropagationBasis(width=16, rank=8, harmonics=4)


def _basis_inputs(records=2, count=6, height=5, width=5, seed=0):
    torch.manual_seed(seed)
    rendered = torch.randn(records * count, 16, height, width)
    time_s = torch.rand(records, count).sort(dim=1).values
    source = torch.tensor([[80.0, 80.0, 12.0, 0.05, 1.0], [80.0, 80.0, 20.0, 0.08, 1.0]])
    return rendered, time_s, source


def test_temporal_basis_zero_initialized_is_no_op():
    m = _basis()
    rendered, time_s, source = _basis_inputs()
    out = m(rendered, time_s, source, records=2, count=6, domain_t_s=1.0)
    assert float(out.abs().max()) == 0.0  # gate=0 -> exact no-op (warm-start safe)


def test_temporal_basis_is_query_invariant():
    m = _basis()
    with torch.no_grad():
        m.gate.fill_(1.0)
    rendered, time_s, source = _basis_inputs()
    full = m(rendered, time_s, source, records=2, count=6, domain_t_s=1.0)
    # query only frame index 2 (count=1) for each record
    r1 = rendered.reshape(2, 6, 16, 5, 5)[:, 2:3].reshape(2, 16, 5, 5)
    single = m(r1, time_s[:, 2:3], source, records=2, count=1, domain_t_s=1.0)
    # single-frame query must match the same frame inside the multi-frame batch
    assert torch.allclose(full[:, 2:3], single, atol=1e-6)


def test_temporal_basis_time_features_are_per_frame_smooth():
    m = _basis()
    source = torch.tensor([[80.0, 80.0, 12.0, 0.05, 1.0]])
    # continuity: feature change shrinks proportionally as the time gap shrinks
    small = m.time_features(torch.tensor([[0.2, 0.2001]]), source, domain_t_s=1.0)
    large = m.time_features(torch.tensor([[0.2, 0.21]]), source, domain_t_s=1.0)
    d_small = float((small[:, 0] - small[:, 1]).abs().max())
    d_large = float((large[:, 0] - large[:, 1]).abs().max())
    assert d_small < 0.1                 # tiny gap -> tiny feature change (no per-index jump)
    assert d_small < d_large             # monotone continuity in dt
