from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .config import load_config
from .data import PinoHDF5Dataset, collate_pino
from .metrics import basic_metrics, per_time_relative_l2
from .model import AcousticFNO3D
from .normalization import decode_standard
from .utils import choose_device, ensure_dir, write_json
from .visualization import plot_sample


def run_evaluation(config_path: str | Path, checkpoint_path: str | Path, split: str = "val") -> dict[str, Any]:
    config = load_config(config_path)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    stats = ckpt["normalization_stats"]
    split_manifest = ckpt["split_manifest"]
    device = choose_device(config["train"].get("device", "auto"))
    model = AcousticFNO3D(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    indices = list(split_manifest[split])
    max_samples = config.get("evaluation", {}).get("max_samples")
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    dataset = PinoHDF5Dataset(config, indices, normalization_stats=stats, return_normalized=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_pino)
    output_dir = ensure_dir(config["evaluation"]["output_dir"])
    metric_rows = []
    per_time_rows = []
    figure_paths = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pred = model(batch["input"].to(device)).cpu()
            target = batch["target"]
            pred_physical = decode_standard(pred, stats["wavefield"], eps=stats.get("eps", 1e-6))
            target_physical = decode_standard(target, stats["wavefield"], eps=stats.get("eps", 1e-6))
            metrics = basic_metrics(pred_physical, target_physical, eps=float(config["loss"].get("eps", 1e-8)))
            metrics["sample_index"] = int(batch["sample_index"][0])
            metric_rows.append(metrics)
            rel_t = per_time_relative_l2(pred_physical, target_physical)[0].cpu().numpy()
            time = batch["time"][0].numpy()
            for ti, value in enumerate(rel_t.tolist()):
                per_time_rows.append({"sample_index": metrics["sample_index"], "time_index": ti, "time": float(time[ti]), "relative_l2": float(value)})
            sample = {k: (v[0] if isinstance(v, torch.Tensor) else v[0]) for k, v in batch.items() if k != "metadata"}
            sample["metadata"] = batch["metadata"][0]
            sample_dir = output_dir / f"sample_{batch_idx:03d}"
            paths = plot_sample(sample, pred[0], sample_dir, normalization_stats=stats)
            figure_paths.append(paths)

    mean_metrics = {k: float(sum(row[k] for row in metric_rows) / len(metric_rows)) for k in metric_rows[0] if k != "sample_index"}
    metrics_payload = {"split": split, "samples": metric_rows, "mean": mean_metrics, "figures": figure_paths}
    write_json(output_dir / "metrics.json", metrics_payload)
    with (output_dir / "per_time_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", "time_index", "time", "relative_l2"])
        writer.writeheader()
        writer.writerows(per_time_rows)
    # Also save one denormalized tensor metric sanity entry for audit.
    if metric_rows:
        metrics_payload["denormalization"] = {
            "wavefield_mean": stats["wavefield"]["mean"],
            "wavefield_std": stats["wavefield"]["std"],
        }
    return metrics_payload
