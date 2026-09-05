from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        data = values.detach().reshape(-1).to(torch.float64)
        data = data[torch.isfinite(data)]
        if data.numel() == 0:
            return
        batch_count = int(data.numel())
        batch_mean = float(data.mean().item())
        batch_m2 = float(((data - batch_mean) ** 2).sum().item())
        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean += delta * batch_count / new_count
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / new_count
        self.count = new_count

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / self.count

    @property
    def std(self) -> float:
        return float(self.variance ** 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {"mean": float(self.mean), "std": float(self.std), "count": int(self.count)}


def encode_standard(values: torch.Tensor, stats: dict[str, float], eps: float = 1e-6) -> torch.Tensor:
    std = max(float(stats["std"]), eps)
    return (values - float(stats["mean"])) / std


def decode_standard(values: torch.Tensor, stats: dict[str, float], eps: float = 1e-6) -> torch.Tensor:
    std = max(float(stats["std"]), eps)
    return values * std + float(stats["mean"])
