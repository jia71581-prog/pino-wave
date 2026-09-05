"""Publication-safe wavefield snapshots and receiver diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = ("#000000", "#0072B2", "#D55E00", "#56B4E9", "#009E73")


def derive_receivers_from_field(field, receiver_indices: Sequence[tuple[int, int]]) -> np.ndarray:
    """Extract [receiver,time] traces from [time,z,x] or [record,time,z,x]."""
    value = np.asarray(field)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError("field must be [time,z,x] or [record,time,z,x]")
    traces = [value[:, int(z), int(x)] for z, x in receiver_indices]
    return np.stack(traces, axis=0)


def _save_pair(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wavefield_comparison(
    target: np.ndarray,
    parent: np.ndarray,
    adapted: np.ndarray,
    time_indices: Sequence[int],
    *,
    output: str | Path,
    title: str,
) -> None:
    target = np.asarray(target)
    parent = np.asarray(parent)
    adapted = np.asarray(adapted)
    if target.shape != parent.shape or target.shape != adapted.shape or target.ndim != 3:
        raise ValueError("comparison fields must match [time,z,x]")
    rows = len(tuple(time_indices))
    if rows == 0:
        raise ValueError("at least one snapshot is required")
    vmax = float(np.percentile(np.abs(target[list(time_indices)]), 99.5))
    vmax = max(vmax, 1.0e-12)
    error_max = max(float(np.percentile(np.abs(adapted - target), 99.5)), 1.0e-12)
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3.2 * rows), squeeze=False)
    columns = ("Truth", "Parent V5", "Adapted", "Parent error", "Adapted error")
    for row, index in enumerate(time_indices):
        images = (target[index], parent[index], adapted[index], parent[index] - target[index], adapted[index] - target[index])
        for column, (label, image) in enumerate(zip(columns, images, strict=True)):
            is_error = column >= 3
            axes[row, column].imshow(image, cmap="RdBu_r", vmin=-error_max if is_error else -vmax, vmax=error_max if is_error else vmax)
            axes[row, column].set_title(f"{label} · t={index}")
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    fig.suptitle(title)
    _save_pair(fig, Path(output))


def plot_receiver_comparison(
    target: np.ndarray,
    parent: np.ndarray,
    adapted: np.ndarray,
    time_s: Sequence[float],
    receiver_indices: Sequence[tuple[int, int]],
    *,
    output: str | Path,
    title: str,
) -> None:
    times = np.asarray(time_s)
    traces = [derive_receivers_from_field(value, receiver_indices) for value in (target, parent, adapted)]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    for color, label, values in zip(OKABE_ITO[:3], ("Truth", "Parent V5", "Adapted"), traces, strict=True):
        axes[0].plot(times, values[0], color=color, label=label, linewidth=1.4)
    axes[0].set_title(f"{title} · representative receiver")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Pressure"); axes[0].legend(frameon=False)
    axes[1].imshow(traces[2] - traces[0], aspect="auto", cmap="RdBu_r", extent=(times[0], times[-1], len(receiver_indices), 0))
    axes[1].set_title("Adapted − truth receiver residual")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Receiver index")
    _save_pair(fig, Path(output))


__all__ = ["derive_receivers_from_field", "plot_receiver_comparison", "plot_wavefield_comparison"]
