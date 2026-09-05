from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import per_time_relative_l2
from .normalization import decode_standard
from .utils import ensure_dir


def denormalize_wavefield(values: torch.Tensor, stats: dict[str, Any] | None) -> torch.Tensor:
    if stats and "wavefield" in stats:
        return decode_standard(values, stats["wavefield"], eps=stats.get("eps", 1e-6))
    return values


def plot_sample(
    sample: dict[str, Any],
    pred: torch.Tensor,
    output_dir: str | Path,
    normalization_stats: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    target = denormalize_wavefield(sample["target"].detach().cpu(), normalization_stats)
    pred = denormalize_wavefield(pred.detach().cpu(), normalization_stats)
    velocity = sample["velocity"][0].detach().cpu()
    source_map = sample["source_map"][0].detach().cpu()
    time = sample["time"].detach().cpu().numpy()
    sample_index = int(sample["sample_index"])
    target_np = target.numpy()
    pred_np = pred.numpy()
    err_np = pred_np - target_np
    frames = sorted(set([0, target_np.shape[-1] // 2, target_np.shape[-1] - 1]))

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    im0 = axes[0, 0].imshow(velocity.numpy(), cmap="viridis")
    axes[0, 0].set_title(f"velocity sample {sample_index}")
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)
    im1 = axes[0, 1].imshow(source_map.numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("source map")
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    rel = per_time_relative_l2(pred[None], target[None])[0].numpy()
    axes[1, 0].plot(time, rel)
    axes[1, 0].set_title("per-time relative L2")
    axes[1, 0].set_xlabel("time (s)")
    center = (target_np.shape[0] // 2, target_np.shape[1] // 2)
    axes[1, 1].plot(time, target_np[center[0], center[1], :], label="target")
    axes[1, 1].plot(time, pred_np[center[0], center[1], :], label="pred")
    axes[1, 1].set_title(f"receiver trace {center}")
    axes[1, 1].legend()
    overview_path = output_dir / "overview.png"
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(frames), 3, figsize=(10, 3.2 * len(frames)), constrained_layout=True)
    if len(frames) == 1:
        axes = np.asarray([axes])
    for row, ti in enumerate(frames):
        vmax = max(float(np.max(np.abs(target_np[:, :, ti]))), float(np.max(np.abs(pred_np[:, :, ti]))), 1e-12)
        err_vmax = max(float(np.max(np.abs(err_np[:, :, ti]))), 1e-12)
        for col, (name, data, vmin, vmax_use, cmap) in enumerate(
            [
                ("target", target_np[:, :, ti], -vmax, vmax, "seismic"),
                ("pred", pred_np[:, :, ti], -vmax, vmax, "seismic"),
                ("error", err_np[:, :, ti], -err_vmax, err_vmax, "seismic"),
            ]
        ):
            im = axes[row, col].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_use)
            axes[row, col].set_title(f"{name} t={float(time[ti]):.6g}s range=[{vmin:.3g},{vmax_use:.3g}]")
            plt.colorbar(im, ax=axes[row, col], shrink=0.8)
    fields_path = output_dir / "target_pred_error.png"
    fig.savefig(fields_path, dpi=160)
    plt.close(fig)
    return {"overview": str(overview_path), "target_pred_error": str(fields_path)}
