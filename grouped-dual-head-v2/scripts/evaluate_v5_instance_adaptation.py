#!/usr/bin/env python3
"""Sealed evaluator for full saved-time instance adaptation outputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from saved_time_phase_operator_v4.instance_adaptation.contracts import future_indices
from saved_time_phase_operator_v4.instance_adaptation.visualization import (
    plot_receiver_comparison,
    plot_wavefield_comparison,
)
from saved_time_phase_operator_v4.streaming_metrics import ExactWavefieldMetricAccumulator


@dataclass(frozen=True)
class EvaluationReport:
    metrics: dict[str, object]
    future_fullfield_relative_l2: float
    receiver_relative_l2: float
    sealed: bool


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = (prediction.float() - target.float()).flatten(1).norm(dim=1)
    reference = target.float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    return float((error / reference).mean())


def evaluate_after_adaptation(
    adapted_artifact: dict[str, object],
    target: torch.Tensor,
    *,
    observed_indices: Sequence[tuple[int, int]],
    families: Sequence[str],
    group_ids: Sequence[str],
    sample_ids: Sequence[str],
    sealed: bool,
) -> dict[str, object]:
    """Evaluate only after the adapter artifact and access audit are sealed."""
    if not sealed:
        raise RuntimeError("future truth evaluation requires a sealed adaptation artifact")
    adapted = torch.as_tensor(adapted_artifact["adapted_field"])
    parent = torch.as_tensor(adapted_artifact["parent_field"], device=adapted.device)
    truth = torch.as_tensor(target, device=adapted.device)
    if adapted.shape != parent.shape or adapted.shape != truth.shape or adapted.ndim != 4:
        raise ValueError("parent, adapted, and target fields must match [record,time,z,x]")
    if len(observed_indices) != adapted.shape[0]:
        raise ValueError("one onset tuple is required per record")
    future_masks = torch.zeros(adapted.shape[:2], dtype=torch.bool, device=adapted.device)
    for record, indices in enumerate(observed_indices):
        future_masks[record, future_indices(adapted.shape[1], indices).to(adapted.device)] = True
    future_error = _relative_l2(adapted[future_masks], truth[future_masks])
    parent_future_error = _relative_l2(parent[future_masks], truth[future_masks])
    future_truth64 = truth[future_masks].double()
    future_adapted64 = adapted[future_masks].double()
    future_parent64 = parent[future_masks].double()
    future_truth_squared_norm = float(future_truth64.square().sum())
    future_adapted_squared_error = float(
        (future_adapted64 - future_truth64).square().sum()
    )
    future_parent_squared_error = float(
        (future_parent64 - future_truth64).square().sum()
    )
    receiver_points = tuple((max(1, adapted.shape[-2] // 20), x) for x in np.linspace(5, adapted.shape[-1] - 6, 9, dtype=int))
    def receiver_traces(field: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [field[:, :, int(z), int(x)] for z, x in receiver_points], dim=-1
        )
    parent_receiver = receiver_traces(parent)
    adapted_receiver = receiver_traces(adapted)
    truth_receiver = receiver_traces(truth)
    receiver_error = _relative_l2(adapted_receiver, truth_receiver)
    time_indices = torch.arange(adapted.shape[1], dtype=torch.long)[None].expand(adapted.shape[0], -1)
    def accumulate(field: torch.Tensor) -> dict[str, object]:
        accumulator = ExactWavefieldMetricAccumulator(
            require_unique=True, stored_time_count=adapted.shape[1]
        )
        accumulator.update(
            field,
            truth,
            families=families,
            group_ids=group_ids,
            sample_ids=sample_ids,
            time_indices=time_indices,
        )
        return accumulator.finalize()

    metrics = accumulate(adapted)
    parent_metrics = accumulate(parent)
    return {
        "metrics": metrics,
        "parent_metrics": parent_metrics,
        "future_fullfield_relative_l2": future_error,
        "parent_future_fullfield_relative_l2": parent_future_error,
        "future_truth_squared_norm": future_truth_squared_norm,
        "future_adapted_squared_error": future_adapted_squared_error,
        "future_parent_squared_error": future_parent_squared_error,
        "receiver_relative_l2": receiver_error,
        "sealed": True,
        "receiver_indices": receiver_points,
        "all_saved_time_indices": int(adapted.shape[1]),
    }


def write_report(report: dict[str, object], output: str | Path) -> None:
    def jsonable(value):
        if isinstance(value, torch.Tensor):
            data = value.detach().cpu().contiguous().numpy().tobytes()
            return {
                "tensor_shape": tuple(value.shape),
                "tensor_sha256": hashlib.sha256(data).hexdigest(),
            }
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True, default=float) + "\n")


__all__ = ["evaluate_after_adaptation", "write_report"]
