"""Explicit, auditable activation modes for a transferred residual head."""
from __future__ import annotations

import math
from typing import Mapping

import torch


def residual_activation_config(
    recovery: Mapping[str, object],
    transfer: Mapping[str, object],
) -> dict[str, object]:
    """Resolve explicit activation while preserving the legacy boolean contract."""

    resolved = dict(recovery)
    mode = resolved.get("activation_mode")
    if mode is None:
        mode = (
            "preserve"
            if bool(transfer.get("parent_residual_already_active", False))
            else "reset"
        )
    mode = str(mode)
    if mode not in {"preserve", "absorb", "rescale", "reset"}:
        raise ValueError(
            "residual activation mode must be preserve, absorb, rescale, or reset"
        )
    if mode != "preserve" and bool(transfer.get("parent_optimizer_state", False)):
        raise ValueError(
            "residual reset/rescale is incompatible with parent optimizer-state transfer"
        )
    resolved["activation_mode"] = mode
    return resolved


@torch.no_grad()
def activate_residual_head(
    decoder,
    recovery: Mapping[str, object],
    *,
    seed: int | None = None,
) -> dict[str, float | str]:
    """Preserve, rescale, or deterministically reset one residual output head."""

    mode = str(recovery.get("activation_mode", "preserve"))
    if mode not in {"preserve", "absorb", "rescale", "reset"}:
        raise ValueError(
            "residual activation mode must be preserve, absorb, rescale, or reset"
        )
    scale = float(recovery.get("correction_scale", 1.0))
    output_std = float(recovery.get("output_std", 1.0e-4))
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not math.isfinite(output_std)
        or output_std <= 0.0
    ):
        raise ValueError(
            "residual activation scale and output_std must be positive and finite"
        )

    before_std = float(decoder.output.weight.detach().float().std())
    before_scale = float(decoder.correction_scale.detach())
    if mode == "absorb":
        # Function-preserving wake-up for a collapsed scalar gate.  Fold the
        # transferred scale into the final affine readout, then lock the explicit
        # multiplier at one.  The update-0 prediction is function-equivalent (including
        # negative or exactly-zero transferred scales), while the output weight now
        # receives an unattenuated gradient instead of one multiplied by ~1e-6.
        if not math.isfinite(before_scale):
            raise ValueError("cannot absorb a non-finite residual correction scale")
        decoder.output.weight.mul_(before_scale)
        if decoder.output.bias is not None:
            decoder.output.bias.mul_(before_scale)
        decoder.correction_scale.fill_(1.0)
        decoder.correction_scale.requires_grad_(False)
    elif mode == "rescale":
        decoder.correction_scale.fill_(scale)
        decoder.correction_scale.requires_grad_(False)
    elif mode == "reset":
        devices: list[int] = []
        if decoder.output.weight.is_cuda:
            devices.append(int(decoder.output.weight.device.index or 0))
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(int(seed))
            decoder.activate_residual_correction(
                scale=scale,
                output_std=output_std,
            )

    report: dict[str, float | str] = {
        "mode": mode,
        "pre_output_std": before_std,
        "post_output_std": float(
            decoder.output.weight.detach().float().std()
        ),
        "pre_correction_scale": before_scale,
        "post_correction_scale": float(decoder.correction_scale.detach()),
        "post_bias_norm": float(decoder.output.bias.detach().float().norm()),
    }
    if not all(
        math.isfinite(float(report[name]))
        for name in (
            "pre_output_std",
            "post_output_std",
            "pre_correction_scale",
            "post_correction_scale",
            "post_bias_norm",
        )
    ):
        raise FloatingPointError("residual activation produced non-finite telemetry")
    return report


__all__ = ["activate_residual_head", "residual_activation_config"]
