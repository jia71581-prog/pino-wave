#!/usr/bin/env python3
"""Measure how much R29C updates context inputs versus the R28 backbone."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def square_norm(value: torch.Tensor) -> float:
    return float(value.detach().double().square().sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)["model_state_dict"]
    context = torch.load(args.context, map_location="cpu", weights_only=False)["model_state_dict"]

    backbone_delta_square = 0.0
    backbone_square = 0.0
    extra_square = 0.0
    rows: list[dict] = []
    for key, after in context.items():
        if key not in base:
            continue
        before = base[key]
        if (
            key in {"stem.conv.weight", "spectral_stem.weight"}
            and after.ndim == 4
            and before.ndim == 4
            and after.shape[1] > before.shape[1]
        ):
            common = after[:, : before.shape[1]]
            extra = after[:, before.shape[1] :]
            delta_square = square_norm(common - before)
            channel_square = square_norm(extra)
            backbone_delta_square += delta_square
            backbone_square += square_norm(before)
            extra_square += channel_square
            rows.append(
                {
                    "key": key,
                    "backbone_delta_l2": math.sqrt(delta_square),
                    "new_context_weight_l2": math.sqrt(channel_square),
                }
            )
        elif tuple(after.shape) == tuple(before.shape):
            backbone_delta_square += square_norm(after - before)
            backbone_square += square_norm(before)

    payload = {
        "backbone_delta_l2": math.sqrt(backbone_delta_square),
        "backbone_weight_l2": math.sqrt(backbone_square),
        "relative_backbone_delta": math.sqrt(backbone_delta_square)
        / max(math.sqrt(backbone_square), 1.0e-30),
        "new_context_weight_l2": math.sqrt(extra_square),
        "new_context_to_backbone_delta_ratio": math.sqrt(extra_square)
        / max(math.sqrt(backbone_delta_square), 1.0e-30),
        "input_stems": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
