"""Velocity-routed low-rank residual experts for complete saved-time fields."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FamilyExpertOutput:
    correction: torch.Tensor
    router_logits: torch.Tensor
    router_probabilities: torch.Tensor
    effective_router_probabilities: torch.Tensor


def _routed_expert_mix(
    experts: nn.ModuleList,
    shared: torch.Tensor,
    time_features: torch.Tensor,
    record_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Dispatch exact one-hot routes sparsely and preserve soft mixtures."""

    records, times = time_features.shape[:2]
    probabilities = torch.as_tensor(
        record_probabilities,
        dtype=shared.dtype,
        device=shared.device,
    )
    if probabilities.shape != (records, len(experts)):
        raise ValueError("expert probabilities do not match record routes")
    exact_one_hot = bool(
        torch.logical_or(probabilities == 0.0, probabilities == 1.0).all()
    )
    if not exact_one_hot:
        corrections = torch.stack(
            [expert(shared, time_features) for expert in experts], dim=2
        )
        return (
            corrections * probabilities[:, None, :, None, None]
        ).sum(dim=2)

    routes = probabilities.argmax(dim=-1)
    shared_by_record = shared.reshape(records, times, *shared.shape[1:])
    mixed = shared.new_zeros(
        (records, times, shared.shape[-2], shared.shape[-1])
    )
    for expert_index, expert in enumerate(experts):
        selected = torch.nonzero(routes == expert_index, as_tuple=False).flatten()
        if selected.numel() == 0:
            continue
        selected_shared = shared_by_record.index_select(0, selected).reshape(
            selected.numel() * times, *shared.shape[1:]
        )
        selected_times = time_features.index_select(0, selected)
        correction = expert(selected_shared, selected_times)
        mixed = mixed.index_copy(0, selected, correction)
    return mixed


class VelocityFamilyRouter(nn.Module):
    """Learned residual on a velocity-only structural family prior.

    The registered training split separates the three supported families with two
    velocity statistics: shallow mean speed and global standard deviation.  The
    smooth prior makes train-time teacher routing agree with inference from the
    first step, while the zero-initialized learned residual remains free to adapt.
    """

    uniform_std_threshold_mps = 40.0
    marmousi_shallow_threshold_mps = 1700.0
    structural_margin_scale_mps = 20.0

    def __init__(self, *, width: int, family_count: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6 + 2 * width, width),
            nn.GELU(),
            nn.Linear(width, family_count),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @classmethod
    def structural_logits(cls, velocity_stats: torch.Tensor) -> torch.Tensor:
        """Return smooth uniform/layered/Marmousi logits from physical statistics."""

        if velocity_stats.ndim != 2 or velocity_stats.shape[-1] != 6:
            raise ValueError("velocity family statistics must have shape [medium,6]")
        deviation = velocity_stats[:, 1]
        shallow = velocity_stats[:, 2]
        scale = float(cls.structural_margin_scale_mps)
        uniform = (float(cls.uniform_std_threshold_mps) - deviation) / scale
        marmousi = (
            float(cls.marmousi_shallow_threshold_mps) - shallow
        ) / scale
        layered = torch.minimum(-uniform, -marmousi)
        return torch.stack((uniform, layered, marmousi), dim=-1).clamp(-30.0, 30.0)

    def forward(
        self, velocity_mps: torch.Tensor, medium_features: torch.Tensor
    ) -> torch.Tensor:
        velocity = velocity_mps.float()
        shallow_depth = max(1, velocity.shape[-2] // 4)
        deep_depth = max(1, velocity.shape[-2] // 4)
        shallow = velocity[..., :shallow_depth, :].mean((1, 2, 3))
        deep = velocity[..., -deep_depth:, :].mean((1, 2, 3))
        velocity_stats = torch.stack(
            (
                velocity.mean((1, 2, 3)),
                velocity.std((1, 2, 3)),
                shallow,
                deep,
                velocity.diff(dim=-2).abs().mean((1, 2, 3)),
                velocity.diff(dim=-1).abs().mean((1, 2, 3)),
            ),
            dim=-1,
        )
        pooled = torch.cat(
            (
                medium_features.mean((2, 3)),
                medium_features.std((2, 3)),
            ),
            dim=-1,
        )
        learned = self.network(
            torch.cat((velocity_stats / 5000.0, pooled), dim=-1)
        )
        return self.structural_logits(velocity_stats) + learned


class LowRankResidualExpert(nn.Module):
    def __init__(self, *, width: int, rank: int):
        super().__init__()
        self.down = nn.Conv2d(width, rank, 1)
        self.spatial = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)
        self.time = nn.Sequential(
            nn.Linear(5, rank),
            nn.GELU(),
            nn.Linear(rank, rank),
        )
        self.output = nn.Conv2d(rank, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, shared: torch.Tensor, time_features: torch.Tensor
    ) -> torch.Tensor:
        records, times = time_features.shape[:2]
        hidden = F.gelu(self.spatial(F.gelu(self.down(shared))))
        modulation = 1.0 + self.time(time_features).reshape(
            records * times, -1, 1, 1
        )
        return self.output(hidden * modulation).reshape(
            records, times, *shared.shape[-2:]
        )


class FamilyRoutedResidualExperts(nn.Module):
    def __init__(self, *, width: int, rank: int, family_count: int = 3):
        super().__init__()
        if width <= 0 or rank <= 0 or family_count != 3:
            raise ValueError(
                "expert width/rank must be positive and family_count must be three"
            )
        self.router = VelocityFamilyRouter(width=width, family_count=family_count)
        self.experts = nn.ModuleList(
            LowRankResidualExpert(width=width, rank=rank)
            for _ in range(family_count)
        )

    def forward(
        self,
        velocity_mps: torch.Tensor,
        medium_features: torch.Tensor,
        shared: torch.Tensor,
        time_features: torch.Tensor,
        record_to_medium: torch.Tensor,
        *,
        route_override: torch.Tensor | None = None,
    ) -> FamilyExpertOutput:
        records, times = time_features.shape[:2]
        mapping = torch.as_tensor(
            record_to_medium, dtype=torch.long, device=shared.device
        )
        if (
            mapping.shape != (records,)
            or mapping.numel() == 0
            or mapping.min() < 0
            or mapping.max() >= velocity_mps.shape[0]
        ):
            raise ValueError("record_to_medium contains invalid indices")
        logits = self.router(velocity_mps, medium_features)
        router_probabilities = logits.softmax(dim=-1)
        if route_override is None:
            medium_probabilities = router_probabilities
        else:
            override = torch.as_tensor(
                route_override, dtype=torch.long, device=shared.device
            )
            if (
                override.shape != (velocity_mps.shape[0],)
                or override.numel() == 0
                or override.min() < 0
                or override.max() >= len(self.experts)
            ):
                raise ValueError("family expert route override is invalid")
            medium_probabilities = F.one_hot(
                override, num_classes=len(self.experts)
            ).to(dtype=shared.dtype)
        probabilities = medium_probabilities[mapping]
        mixed = _routed_expert_mix(
            self.experts,
            shared,
            time_features,
            probabilities,
        )
        return FamilyExpertOutput(
            mixed,
            logits,
            router_probabilities,
            medium_probabilities,
        )


__all__ = [
    "FamilyExpertOutput",
    "FamilyRoutedResidualExperts",
    "LowRankResidualExpert",
    "VelocityFamilyRouter",
]
