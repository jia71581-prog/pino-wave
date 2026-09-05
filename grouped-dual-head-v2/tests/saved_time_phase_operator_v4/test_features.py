import pytest
import torch

from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime
from saved_time_phase_operator_v4.features import dense_propagation_features


def _travel(seconds: torch.Tensor) -> RayTravelTime:
    return RayTravelTime(
        seconds=seconds,
        distance_m=torch.tensor([[300.0, 600.0]], dtype=torch.float32),
        path_velocity_mps=torch.tensor([[2000.0, 2400.0]], dtype=torch.float32),
        endpoint_velocity_mps=torch.tensor([[2100.0, 2500.0]], dtype=torch.float32),
        mean_slowness_s_per_m=torch.tensor([[0.0005, 1.0 / 2400.0]], dtype=torch.float32),
    )


def test_dense_features_are_causal_and_phase_aligned():
    source = torch.tensor([[0.0, 0.0, 10.0, 0.1, 1.0]])

    value = dense_propagation_features(
        _travel(torch.tensor([[0.2, 0.4]])),
        torch.tensor([[0.25]]),
        source,
        height=1,
        width=2,
        domain_t_s=1.0,
        domain_diagonal_m=3000.0,
    )

    assert value.shape == (1, 1, 12, 1, 2)
    assert value[0, 0, 5, 0, 0] > value[0, 0, 5, 0, 1]
    assert torch.isfinite(value).all()


def test_dense_features_have_unit_phase_at_local_arrival():
    source = torch.tensor([[0.0, 0.0, 20.0, 0.1, 1.0]])
    value = dense_propagation_features(
        _travel(torch.tensor([[0.2, 0.4]])),
        torch.tensor([[0.3]]),
        source,
        height=1,
        width=2,
        domain_t_s=1.0,
        domain_diagonal_m=3000.0,
    )

    torch.testing.assert_close(value[0, 0, 0, 0, 0], torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(value[0, 0, 6, 0, 0], torch.tensor(0.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(value[0, 0, 7, 0, 0], torch.tensor(1.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(value[0, 0, 8:, 0, 0], torch.ones(4), atol=1e-5, rtol=0)


def test_dense_features_support_multiple_requested_times():
    source = torch.tensor([[0.0, 0.0, 10.0, 0.1, 1.0]])
    value = dense_propagation_features(
        _travel(torch.tensor([[0.2, 0.4]])),
        torch.tensor([[0.25, 0.5]]),
        source,
        height=1,
        width=2,
        domain_t_s=1.0,
        domain_diagonal_m=3000.0,
    )

    assert value.shape == (1, 2, 12, 1, 2)
    assert value[0, 1, 5].mean() > value[0, 0, 5].mean()


def test_dense_features_reject_inconsistent_grid_shape():
    source = torch.tensor([[0.0, 0.0, 10.0, 0.1, 1.0]])
    with pytest.raises(ValueError, match="grid point count"):
        dense_propagation_features(
            _travel(torch.tensor([[0.2, 0.4]])),
            torch.tensor([[0.25]]),
            source,
            height=2,
            width=2,
            domain_t_s=1.0,
            domain_diagonal_m=3000.0,
        )


def test_travel_progress_channel_is_optional_and_backward_compatible():
    source = torch.tensor([[0.0, 0.0, 10.0, 0.1, 1.0]])
    common = dict(height=1, width=2, domain_t_s=1.0, domain_diagonal_m=3000.0)
    twelve = dense_propagation_features(
        _travel(torch.tensor([[0.2, 0.4]])), torch.tensor([[0.25, 0.5]]), source, **common
    )
    thirteen = dense_propagation_features(
        _travel(torch.tensor([[0.2, 0.4]])), torch.tensor([[0.25, 0.5]]), source,
        include_travel_progress=True, **common
    )
    # default is unchanged; opting in appends exactly one channel, first 12 identical
    assert twelve.shape == (1, 2, 12, 1, 2)
    assert thirteen.shape == (1, 2, 13, 1, 2)
    assert torch.equal(twelve, thirteen[:, :, :12])
    # the travel-progress clock increases with time and is finite
    progress = thirteen[0, :, 12].flatten(1).mean(dim=1)
    assert float(progress[1]) > float(progress[0])
    assert torch.isfinite(thirteen).all()
