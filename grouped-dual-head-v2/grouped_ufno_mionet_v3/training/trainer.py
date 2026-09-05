"""Deterministic guarded AdamW/L-BFGS primitives for V3 training."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .audit import audit_required_gradients
from .checkpoint import save_checkpoint_atomic


class PlateauDetector:
    def __init__(self, *, patience: int, min_delta: float) -> None:
        if patience <= 0 or min_delta < 0:
            raise ValueError("plateau patience must be positive and min_delta nonnegative")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = math.inf
        self.bad_updates = 0

    def update(self, metric: float) -> bool:
        value = float(metric)
        if not math.isfinite(value):
            raise ValueError("plateau metric must be finite")
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_updates = 0
            return False
        self.bad_updates += 1
        return self.bad_updates >= self.patience


def fresh_full_batch_lbfgs(
    model: nn.Module,
    *,
    learning_rate: float,
    max_iter: int,
    history_size: int,
) -> torch.optim.LBFGS:
    if learning_rate <= 0 or max_iter <= 0 or history_size <= 0:
        raise ValueError("L-BFGS learning rate, iterations, and history must be positive")
    return torch.optim.LBFGS(
        model.parameters(),
        lr=float(learning_rate),
        max_iter=int(max_iter),
        history_size=int(history_size),
        line_search_fn="strong_wolfe",
    )


class GuardedV3Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        checkpoint_dir: str | Path,
        manifest_digest: str,
        config_digest: str,
        gradient_clip: float = 1.0,
    ) -> None:
        if not manifest_digest or not config_digest or gradient_clip <= 0:
            raise ValueError("trainer identities and gradient clip are required")
        if not hasattr(model, "required_gradient_groups"):
            raise ValueError("V3 model must expose required_gradient_groups")
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifest_digest = str(manifest_digest)
        self.config_digest = str(config_digest)
        self.gradient_clip = float(gradient_clip)
        self.global_step = 0

    def train_step(self, loss_closure: Callable[[], torch.Tensor]) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError("loss closure must return a scalar tensor")
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite V3 training loss")
        loss.backward()
        audit_required_gradients(self.model.required_gradient_groups())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        if not torch.isfinite(torch.as_tensor(norm)):
            raise RuntimeError("nonfinite V3 gradient norm")
        self.optimizer.step()
        self.global_step += 1
        return loss.detach()

    def save_epoch(self, epoch: int, *, metrics: dict[str, float]) -> Path:
        if epoch <= 0:
            raise ValueError("checkpoint epoch must be positive")
        destination = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        return save_checkpoint_atomic(
            destination,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            global_step=self.global_step,
            manifest_digest=self.manifest_digest,
            config_digest=self.config_digest,
            metrics=metrics,
        )


__all__ = ["GuardedV3Trainer", "PlateauDetector", "fresh_full_batch_lbfgs"]
