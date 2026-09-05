from __future__ import annotations

import torch


class ResidualImportanceSampler:
    """Record-specific spatial residual EMA with a nonzero uniform floor."""
    def __init__(self, cells=(12, 12), decay=0.95, uniform_floor=0.1, seed=17):
        self.cells = tuple(cells)
        self.decay, self.uniform_floor = float(decay), float(uniform_floor)
        if min(self.cells) <= 0 or not 0 <= self.uniform_floor <= 1:
            raise ValueError("invalid sampler settings")
        self.ema = {}
        self.generator = torch.Generator().manual_seed(int(seed))
        self.steps = 0

    def update(self, sample_id, residual):
        value = torch.as_tensor(residual, dtype=torch.float32).detach().abs()
        if value.ndim != 2:
            raise ValueError("residual must be a 2D spatial array")
        pooled = torch.nn.functional.adaptive_avg_pool2d(value[None, None], self.cells).squeeze()
        old = self.ema.get(sample_id)
        self.ema[sample_id] = pooled if old is None else self.decay * old + (1 - self.decay) * pooled
        self.steps += 1

    def probabilities(self, sample_id):
        value = self.ema.get(sample_id, torch.ones(self.cells))
        p = value.clamp_min(0) + 1e-6
        p = p / p.sum()
        uniform = torch.full_like(p, 1.0 / p.numel())
        return (1 - self.uniform_floor) * p + self.uniform_floor * uniform

    def state_dict(self):
        return {"cells": self.cells, "decay": self.decay, "uniform_floor": self.uniform_floor,
                "ema": self.ema, "generator_state": self.generator.get_state(), "steps": self.steps}

    def load_state_dict(self, state):
        self.ema = {k: v.clone() for k, v in state["ema"].items()}
        self.generator.set_state(state["generator_state"])
        self.steps = int(state["steps"])
