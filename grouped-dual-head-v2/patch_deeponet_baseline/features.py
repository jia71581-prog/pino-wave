"""Target-free static and query features for the Patch-DeepONet baseline."""
from __future__ import annotations

import torch


STATIC_CHANNELS = (
    "velocity_normalized",
    "source_map",
    "travel_time_normalized",
    "slowness_contrast",
)
QUERY_CHANNELS = (
    "x_normalized",
    "z_normalized",
    "time_normalized",
    "source_x_normalized",
    "source_z_normalized",
    "source_frequency_normalized",
    "source_onset_normalized",
    "source_amplitude_normalized",
    "travel_time_normalized",
    "arrival_cycles",
)


def _finite(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


def build_static_features(
    velocity_normalized: torch.Tensor,
    source_map: torch.Tensor,
    travel_time_s: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    domain_t_s: float,
) -> torch.Tensor:
    """Return four target-free maps shaped ``[record,4,z,x]``.

    Slowness contrast is centered per medium, so it carries local heterogeneity
    without pressure data or a model prediction.
    """

    velocity_n = _finite(velocity_normalized, name="normalized velocity")
    source = _finite(source_map, name="source map")
    travel = _finite(travel_time_s, name="travel time")
    velocity = _finite(velocity_mps, name="physical velocity")
    if velocity_n.ndim != 4 or velocity_n.shape[1] != 1:
        raise ValueError("normalized velocity must be [record,1,z,x]")
    if source.shape != velocity_n.shape or velocity.shape != velocity_n.shape:
        raise ValueError("velocity and source maps must have the same [record,1,z,x] shape")
    if travel.shape == (velocity_n.shape[0], *velocity_n.shape[-2:]):
        travel = travel[:, None]
    if travel.shape != velocity_n.shape:
        raise ValueError("travel time must be [record,z,x] or [record,1,z,x]")
    if float(domain_t_s) <= 0.0 or torch.any(velocity <= 0.0):
        raise ValueError("domain time and velocity must be positive")
    if torch.any(source < 0.0):
        raise ValueError("source map must be nonnegative")
    mass = source.sum(dim=(-2, -1))
    if not torch.allclose(mass, torch.ones_like(mass), atol=2.0e-4, rtol=2.0e-4):
        raise ValueError("source map must have unit mass")
    slowness = velocity.reciprocal()
    mean = slowness.mean(dim=(-2, -1), keepdim=True)
    contrast = (slowness - mean) / mean.clamp_min(1.0e-12)
    return torch.cat((velocity_n, source, travel / float(domain_t_s), contrast), dim=1)


def build_query_descriptors(
    query_xyz_m: torch.Tensor,
    source_normalized: torch.Tensor,
    source_physical: torch.Tensor,
    travel_time_s: torch.Tensor,
    *,
    domain_x_m: float,
    domain_z_m: float,
    domain_t_s: float,
) -> torch.Tensor:
    """Return target-free query descriptors shaped ``[record,query,10]``."""

    query = _finite(query_xyz_m, name="query coordinates")
    source_n = _finite(source_normalized, name="normalized source")
    source = _finite(source_physical, name="physical source")
    travel = _finite(travel_time_s, name="query travel time")
    if query.ndim != 3 or query.shape[-1] != 3:
        raise ValueError("query coordinates must be [record,query,3]")
    records, queries = query.shape[:2]
    if source_n.shape != (records, 5) or source.shape != (records, 5):
        raise ValueError("source arrays must be [record,5]")
    if travel.shape != (records, queries):
        raise ValueError("query travel time must be [record,query]")
    scales = (float(domain_x_m), float(domain_z_m), float(domain_t_s))
    if any(value <= 0.0 for value in scales):
        raise ValueError("domain scales must be positive")
    xyz_n = torch.stack(
        (
            2.0 * query[..., 0] / scales[0] - 1.0,
            2.0 * query[..., 1] / scales[1] - 1.0,
            2.0 * query[..., 2] / scales[2] - 1.0,
        ),
        dim=-1,
    )
    expanded_source = source_n[:, None].expand(-1, queries, -1)
    arrival_cycles = (
        query[..., 2] - source[:, None, 3] - travel
    ) * source[:, None, 2]
    return torch.cat(
        (
            xyz_n,
            expanded_source,
            (travel / scales[2])[..., None],
            arrival_cycles[..., None],
        ),
        dim=-1,
    )


__all__ = [
    "QUERY_CHANNELS",
    "STATIC_CHANNELS",
    "build_query_descriptors",
    "build_static_features",
]
