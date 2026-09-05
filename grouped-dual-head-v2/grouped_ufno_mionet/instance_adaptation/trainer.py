"""Bounded adaptation acceptance policy."""
from __future__ import annotations

import torch
import time

from .losses import lwc84_residual


def accept_or_rollback(baseline: torch.Tensor, candidate: torch.Tensor, *, baseline_residual: float,
                       candidate_residual: float, energy_ratio: float) -> tuple[bool, torch.Tensor]:
    """Accept only finite candidates with bounded residual and physical-energy change."""
    finite = torch.isfinite(candidate).all() and torch.isfinite(torch.tensor([candidate_residual, energy_ratio])).all()
    accepted = bool(finite and candidate_residual <= 1.02 * baseline_residual and 0.25 <= energy_ratio <= 4.0)
    return accepted, candidate if accepted else baseline.clone()


def adapt_instance(model, *, velocity, source, time_s, observed_wavefield, observed_indices,
                   steps: int = 20, learning_rate: float = 1e-3, dx: float = 10.0, dz: float = 10.0):
    """Optimize only adapter weights from two frames and future-domain physics; never takes future labels."""
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    parameters = tuple(model.trainable_parameters())
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items() if value.requires_grad is False or True}
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate) if steps else None
    last_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        field = model(velocity, source, time_s, observed_wavefield, observed_indices)
        dt = float(torch.as_tensor(time_s)[1] - torch.as_tensor(time_s)[0])
        physics = lwc84_residual(field, velocity, dt=dt, dx=dx, dz=dz, observed_indices=observed_indices)
        anchor = sum(parameter.square().mean() for parameter in parameters) * 1e-6
        loss = physics + anchor
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite instance-adaptation loss")
        loss.backward(); optimizer.step(); last_loss = float(loss.detach())
    return {"steps": steps, "loss": last_loss, "elapsed_s": time.perf_counter() - start,
            "state_dict": model.state_dict(), "baseline_state_dict": baseline}
