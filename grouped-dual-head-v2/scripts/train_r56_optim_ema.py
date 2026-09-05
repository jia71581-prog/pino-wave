#!/usr/bin/env python3
"""R56: optimizer-side lever on the exact R54 recipe.

Delegates to the R54 wrapper (worker-safe loading, R28 entrypoint, r25 driver).
Two independent additions, both selected by environment variables so the r25
CLI stays byte-identical to R54 v2:

1. R56_OPTIMIZER=adamw|soap   (default adamw)
   soap swaps torch.optim.AdamW for pytorch_optimizer.SOAP with the exact
   hyperparameters the A+1 chain validated (betas (0.95, 0.95),
   precondition_frequency 10, precondition_1d), keeping lr / weight_decay /
   cosine schedule from the CLI. Everything else about the step (grad clip,
   scheduler, checkpoint state_dict) is untouched.

2. R56_EMA_DECAY=<float>      (default 0.999; 0 disables)
   Polyak weight averaging as an EVALUATION/CHECKPOINT track only. The EMA
   never feeds back into training: after each per-epoch evaluate() the EMA
   weights are swapped into the model (so the best.pt saved right after the
   evaluation contains exactly the weights that produced the metric) and the
   raw training weights are restored inside DistributedSampler.set_epoch,
   which the r25 driver calls before any forward of the next epoch. The raw
   track is still evaluated each epoch and attached as ``raw_shadow_*`` so
   arm A doubles as a replication check against R54 v2.

Requires torchrun (distributed) so that sampler.set_epoch exists; a
non-distributed launch would never restore the raw weights and is refused.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).resolve()
R54_PATH = SCRIPT_PATH.with_name("train_r54_scratch_fullpool.py")
SPEC = importlib.util.spec_from_file_location("r54_entry", R54_PATH)
r54 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r54
SPEC.loader.exec_module(r54)
r25 = r54.r25

OPTIMIZER_NAME = os.environ.get("R56_OPTIMIZER", "adamw").lower()
EMA_DECAY = float(os.environ.get("R56_EMA_DECAY", "0.999"))
if OPTIMIZER_NAME not in ("adamw", "soap"):
    raise SystemExit(f"R56_OPTIMIZER must be adamw or soap, got {OPTIMIZER_NAME!r}")
if "RANK" not in os.environ:
    raise SystemExit("R56 requires torchrun (EMA restore hooks into DistributedSampler.set_epoch)")

_ema_state: dict = {}   # param tensor -> fp32 running average
_raw_stash: dict = {}   # param tensor -> stashed raw weights while EMA is swapped in
_tracked_params: list = []


def _ema_update() -> None:
    if EMA_DECAY <= 0.0:
        return
    with torch.no_grad():
        for p in _tracked_params:
            avg = _ema_state.get(p)
            if avg is None:
                _ema_state[p] = p.detach().clone()
            else:
                avg.mul_(EMA_DECAY).add_(p.detach(), alpha=1.0 - EMA_DECAY)


_orig_adamw = torch.optim.AdamW


def _build_optimizer(params, *, lr, weight_decay):
    params = list(params)
    _tracked_params.extend(params)
    if OPTIMIZER_NAME == "soap":
        from pytorch_optimizer import SOAP

        opt = SOAP(
            params,
            lr=float(lr),
            betas=(0.95, 0.95),
            weight_decay=float(weight_decay),
            precondition_frequency=10,
            precondition_1d=True,
        )
    else:
        opt = _orig_adamw(params, lr=float(lr), weight_decay=float(weight_decay))
    inner_step = opt.step

    def step_with_ema(self, closure=None):
        result = inner_step() if closure is None else inner_step(closure)
        _ema_update()
        return result

    # bound method so LRScheduler's with_counter (which reads step.__self__) still works
    opt.step = types.MethodType(step_with_ema, opt)
    return opt


torch.optim.AdamW = _build_optimizer

_orig_evaluate = r25.evaluate


def _evaluate_ema_primary(model, collection, **kwargs):
    raw = _orig_evaluate(model, collection, **kwargs)
    if EMA_DECAY <= 0.0 or not _ema_state:
        return raw
    if _raw_stash:
        raise RuntimeError("R56 EMA swap requested while raw weights are already stashed")
    with torch.no_grad():
        for p in _tracked_params:
            _raw_stash[p] = p.detach().clone()
            p.copy_(_ema_state[p])
    ema_metrics = _orig_evaluate(model, collection, **kwargs)
    # EMA is the primary track: best-selection score and the checkpoint saved
    # immediately after this call both follow it. Raw stays as a shadow record.
    ema_metrics["evaluation_track"] = "ema"
    ema_metrics["ema_decay"] = EMA_DECAY
    ema_metrics["raw_shadow_aggregate"] = raw["aggregate"]
    ema_metrics["raw_shadow_per_family"] = raw["per_family"]
    ema_metrics["raw_shadow_absolute_goal"] = raw["absolute_goal"]
    return ema_metrics  # EMA weights stay in the model until the next set_epoch


r25.evaluate = _evaluate_ema_primary

_orig_sampler = r25.DistributedSampler


class _RestoreRawSampler(_orig_sampler):
    def set_epoch(self, epoch):
        if _raw_stash:
            with torch.no_grad():
                for p, raw in _raw_stash.items():
                    p.copy_(raw)
            _raw_stash.clear()
        super().set_epoch(epoch)


r25.DistributedSampler = _RestoreRawSampler


def main() -> None:
    r54.r28.SCRIPT_PATH = SCRIPT_PATH
    r54.r28.main()


if __name__ == "__main__":
    main()
