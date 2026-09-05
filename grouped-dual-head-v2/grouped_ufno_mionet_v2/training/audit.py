"""Fail-fast numerical and optimization audits."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def missing_gradients(model: torch.nn.Module, required_prefixes: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    found = {prefix: False for prefix in required_prefixes}
    for name, parameter in model.named_parameters():
        matches = [prefix for prefix in required_prefixes if name.startswith(prefix)]
        if not matches or not parameter.requires_grad:
            continue
        for prefix in matches:
            found[prefix] = True
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            missing.append(name)
    missing.extend(f"{prefix}.* (no trainable parameters)" for prefix, exists in found.items() if not exists)
    return missing


def require_gradients(model: torch.nn.Module, required_prefixes: tuple[str, ...]) -> dict[str, float]:
    missing = missing_gradients(model, required_prefixes)
    if missing:
        raise RuntimeError("missing or nonfinite required gradients: " + ", ".join(missing))
    norms: dict[str, float] = {}
    for prefix in required_prefixes:
        squared = [parameter.grad.detach().float().square().sum()
                   for name, parameter in model.named_parameters()
                   if name.startswith(prefix) and parameter.requires_grad]
        norms[prefix] = float(torch.stack(squared).sum().sqrt())
    return norms


@dataclass
class LossDominanceMonitor:
    max_ratio: float = 10.0
    patience: int = 5
    bad_steps: int = 0

    def update(self, data_loss: float, auxiliary_loss: float) -> float:
        ratio = float(auxiliary_loss) / max(abs(float(data_loss)), 1e-12)
        self.bad_steps = self.bad_steps + 1 if ratio > self.max_ratio else 0
        if self.bad_steps >= self.patience:
            raise RuntimeError(
                f"training is dominated by auxiliary loss: ratio={ratio:.3g} "
                f"for {self.bad_steps} consecutive steps"
            )
        return ratio
