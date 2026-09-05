from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data_elastic_vti import PinoHDF5Dataset, collate_pino, create_splits, sample_count_from_config
from .losses import combined_loss
from .losses_dispersion import combined_dispersion_loss, drp_spectral_loss, group_velocity_consistency_loss, phase_velocity_consistency_loss
from .losses_elastic_vti import (
    elastic_energy_stability_loss,
    elastic_receiver_waveform_loss,
    elastic_vti_pde_residual_loss,
)
from .losses_spectral import spectral_band_loss
from .metrics import basic_metrics
from .model_elastic_vti import ElasticVTIFNO3D, build_model_config
from .normalization import RunningStats, decode_standard
from .utils import choose_device, ensure_dir, repo_commit, set_seed, write_json


def compute_normalization(config: dict[str, Any], train_indices: list[int]) -> dict[str, Any]:
    max_samples = config.get("normalization", {}).get("max_stats_samples")
    indices = train_indices if max_samples is None else train_indices[: int(max_samples)]
    dataset = PinoHDF5Dataset(config, indices, normalization_stats=None, return_normalized=False)
    velocity_stats = RunningStats()
    wavefield_stats = RunningStats()
    for i in range(len(dataset)):
        sample = dataset[i]
        velocity_stats.update(sample["velocity"])
        wavefield_stats.update(sample["target"])
    dataset.close()
    stats = {
        "computed_from_split": "train",
        "train_sample_count": len(indices),
        "velocity": velocity_stats.as_dict(),
        "wavefield": wavefield_stats.as_dict(),
        "source_map": {"normalization": "max_to_one"},
        "time": {"normalization": "zero_to_one"},
        "eps": float(config["normalization"].get("eps", 1e-6)),
    }
    if not torch.isfinite(torch.tensor(stats["velocity"]["std"])) or stats["velocity"]["std"] <= stats["eps"]:
        raise ValueError("invalid velocity std from train split")
    if not torch.isfinite(torch.tensor(stats["wavefield"]["std"])) or stats["wavefield"]["std"] <= stats["eps"]:
        raise ValueError("invalid wavefield std from train split")
    return stats


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_split_manifest(raw: dict[str, Any], manifest_path: str | Path | None = None) -> dict[str, Any]:
    missing = [key for key in ("train", "val", "test") if key not in raw]
    if missing:
        raise ValueError(f"split manifest is missing required keys: {missing}")

    splits = dict(raw)
    for key in ("train", "val", "test"):
        splits[key] = [int(index) for index in raw[key]]
        if not splits[key]:
            raise ValueError(f"split manifest has empty {key} split")

    counts = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }
    counts["total_used"] = counts["train"] + counts["val"] + counts["test"]
    splits["sample_counts"] = dict(raw.get("sample_counts") or counts)
    for key, value in counts.items():
        splits["sample_counts"][key] = int(splits["sample_counts"].get(key, value))
    if manifest_path is not None:
        splits["manifest_path"] = str(manifest_path)
    return splits


def _load_or_create_splits(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = config["data"].get("split_manifest")
    if manifest_path:
        return _normalize_split_manifest(_read_json(manifest_path), manifest_path)

    sample_count = sample_count_from_config(config)
    splits = create_splits(
        sample_count,
        config["data"].get("split", [0.8, 0.1, 0.1]),
        int(config["data"].get("split_seed", config.get("seed", 2026))),
        config["data"].get("max_samples"),
    )
    return _normalize_split_manifest(splits)


def _write_split_snapshot(config: dict[str, Any], splits: dict[str, Any]) -> None:
    output_path = config["data"].get("split_output_path")
    if output_path:
        write_json(output_path, splits)
        return
    if not config["data"].get("split_manifest"):
        ensure_dir("artifacts/pino_adaptation")
        write_json("artifacts/pino_adaptation/splits.json", splits)


def _validate_normalization_stats(stats: dict[str, Any], eps: float) -> None:
    for section in ("velocity", "wavefield"):
        if section not in stats or "std" not in stats[section]:
            raise ValueError(f"normalization stats missing {section}.std")
        std = float(stats[section]["std"])
        if not torch.isfinite(torch.tensor(std)) or std <= eps:
            raise ValueError(f"invalid {section} std from normalization stats")


def _load_or_compute_normalization(config: dict[str, Any], train_indices: list[int]) -> dict[str, Any]:
    norm_cfg = config["normalization"]
    stats_path = Path(norm_cfg["stats_path"])
    eps = float(norm_cfg.get("eps", 1e-6))
    if bool(norm_cfg.get("reuse_stats", False)):
        if not stats_path.exists():
            raise FileNotFoundError(f"normalization.reuse_stats=true but stats_path does not exist: {stats_path}")
        stats = _read_json(stats_path)
        _validate_normalization_stats(stats, eps)
        return stats

    stats = compute_normalization(config, train_indices)
    write_json(stats_path, stats)
    return stats


def _resolve_num_workers(config: dict[str, Any]) -> int:
    value = config["data"].get("num_workers", 0)
    if isinstance(value, str) and value.lower() in {"all", "auto"}:
        return int(os.cpu_count() or 1)
    return int(value)


def _loader(config: dict[str, Any], indices: list[int], stats: dict[str, Any], shuffle: bool) -> DataLoader:
    data_cfg = config["data"]
    dataset = PinoHDF5Dataset(config, indices, normalization_stats=stats, return_normalized=True)
    num_workers = _resolve_num_workers(config)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["train"].get("batch_size", 1)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", False)) and num_workers > 0,
        "collate_fn": collate_pino,
    }
    if num_workers > 0 and data_cfg.get("prefetch_factor") is not None:
        loader_kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])
    return DataLoader(**loader_kwargs)


def _decode_hdf5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "dtype") and getattr(value, "dtype", None).kind == "S":
        return bytes(value).decode("utf-8")
    return str(value)


def _model_types_for_indices(config: dict[str, Any], indices: list[int]) -> dict[int, str]:
    key = config.get("data", {}).get("model_type_key", "model_type")
    if not key:
        return {int(index): "unknown" for index in indices}
    path = config["data"]["path"]
    out: dict[int, str] = {}
    with h5py.File(path, "r") as h5:
        if key not in h5:
            return {int(index): "unknown" for index in indices}
        dataset = h5[key]
        for index in indices:
            raw = dataset[int(index)] if dataset.shape else dataset[()]
            out[int(index)] = _decode_hdf5_text(raw)
    return out


def _select_validation_indices(config: dict[str, Any], val_indices: list[int]) -> tuple[list[int], dict[str, Any]]:
    indices = [int(index) for index in val_indices]
    cfg = dict(config.get("validation", {}).get("category_balanced", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return indices, {
            "mode": "split_order",
            "selected_count": len(indices),
            "indices": indices,
        }

    model_types = _model_types_for_indices(config, indices)
    pools: dict[str, list[int]] = {}
    for index in indices:
        category = model_types.get(int(index), "unknown")
        pools.setdefault(category, []).append(int(index))
    categories = cfg.get("categories")
    if categories is None:
        categories = sorted(pools)
    else:
        categories = [str(category) for category in categories]
    samples_per_category = max(1, int(cfg.get("samples_per_category", 1)))
    selected: list[int] = []
    counts_by_category: dict[str, int] = {}
    available_by_category: dict[str, int] = {}
    for category in categories:
        pool = pools.get(category, [])
        available_by_category[category] = len(pool)
        chosen = pool[:samples_per_category]
        selected.extend(chosen)
        counts_by_category[category] = len(chosen)
    if not selected:
        raise ValueError("category-balanced validation selected zero samples")
    return selected, {
        "mode": "category_balanced",
        "samples_per_category": samples_per_category,
        "categories": categories,
        "selected_count": len(selected),
        "indices": selected,
        "counts_by_category": {key: counts_by_category[key] for key in sorted(counts_by_category)},
        "available_by_category": {key: available_by_category[key] for key in sorted(available_by_category)},
    }


def _epoch_shards(
    indices: list[int],
    shard_count: int,
    epoch: int,
    shuffle: bool,
    seed: int,
) -> list[list[int]]:
    ordered = [int(i) for i in indices]
    if not ordered:
        return []
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(epoch))
        permutation = torch.randperm(len(ordered), generator=generator).tolist()
        ordered = [ordered[i] for i in permutation]

    count = max(1, min(int(shard_count), len(ordered)))
    base = len(ordered) // count
    remainder = len(ordered) % count
    shards: list[list[int]] = []
    start = 0
    for shard_idx in range(count):
        size = base + (1 if shard_idx < remainder else 0)
        stop = start + size
        shards.append(ordered[start:stop])
        start = stop
    return [shard for shard in shards if shard]


def _cpu_preload_shards(config: dict[str, Any]) -> int:
    return max(1, int(config.get("data", {}).get("cpu_preload_shards", 1) or 1))


def _preload_batches(config: dict[str, Any], indices: list[int], stats: dict[str, Any]) -> list[dict[str, Any]]:
    loader = _loader(config, indices, stats, shuffle=False)
    return [batch for batch in loader]


def _train_batch_iterables_for_epoch(
    config: dict[str, Any],
    train_indices: list[int],
    stats: dict[str, Any],
    epoch: int,
) -> Iterable[Iterable[dict[str, Any]]]:
    shard_count = _cpu_preload_shards(config)
    if shard_count <= 1:
        yield _loader(config, train_indices, stats, shuffle=_train_shuffle_enabled(config))
        return

    shards = _epoch_shards(
        train_indices,
        shard_count=shard_count,
        epoch=epoch,
        shuffle=_train_shuffle_enabled(config),
        seed=int(config.get("seed", 2026)),
    )
    for shard_idx, shard_indices in enumerate(shards):
        print(
            f"[data] preload_shard={shard_idx + 1}/{len(shards)} samples={len(shard_indices)}",
            flush=True,
        )
        yield _preload_batches(config, shard_indices, stats)


def _should_save_epoch_checkpoint(epoch: int, total_epochs: int, interval: int | None) -> bool:
    if interval is None:
        interval = 1
    interval = max(1, int(interval))
    return (int(epoch) + 1) % interval == 0 or int(epoch) + 1 == int(total_epochs)


def _update_early_stopping_state(
    best_val: float,
    val_metric: float,
    epochs_without_improvement: int,
    patience: int | None,
    min_delta: float = 0.0,
) -> tuple[float, int, bool, bool]:
    improved = float(val_metric) <= float(best_val) - float(min_delta)
    if improved:
        return float(val_metric), 0, True, False
    stale_epochs = int(epochs_without_improvement) + 1
    should_stop = patience is not None and int(patience) > 0 and stale_epochs >= int(patience)
    return float(best_val), stale_epochs, False, should_stop


def _evaluate_early_failure_gate(
    config: dict[str, Any],
    val_metrics: dict[str, float] | None,
    epoch: int | None = None,
) -> dict[str, Any]:
    cfg = dict(config.get("train", {}).get("early_failure_gate", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "status": "disabled"}
    threshold = float(cfg.get("relative_l2_threshold", 0.98))
    margin = float(cfg.get("zero_baseline_margin", 0.0))
    require_better_than_zero = bool(cfg.get("require_better_than_zero_baseline", True))
    val_metrics = val_metrics or {}
    validation_l2 = float(val_metrics.get("relative_l2", float("inf")))
    zero_baseline = float(val_metrics.get("zero_baseline_relative_l2", float("inf")))
    threshold_failed = validation_l2 >= threshold
    zero_failed = validation_l2 >= (zero_baseline - margin) if require_better_than_zero else True
    failed = bool(threshold_failed and zero_failed)
    min_epochs = max(1, int(cfg.get("min_epochs", 1) or 1))
    current_epoch = None if epoch is None else int(epoch) + 1
    in_warmup = failed and current_epoch is not None and current_epoch < min_epochs
    return {
        "enabled": True,
        "status": "warmup_failed" if in_warmup else ("failed" if failed else "passed"),
        "relative_l2_threshold": threshold,
        "validation_relative_l2": validation_l2,
        "zero_baseline_relative_l2": zero_baseline,
        "zero_baseline_margin": margin,
        "require_better_than_zero_baseline": require_better_than_zero,
        "threshold_failed": bool(threshold_failed),
        "zero_baseline_failed": bool(zero_failed),
        "would_fail_without_warmup": bool(failed),
        "min_epochs": min_epochs,
        "current_epoch": current_epoch,
        "stop_on_failure": bool(cfg.get("stop_on_failure", False)) and not in_warmup,
    }


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    collected: list[dict[str, float]] = []
    zero_collected: list[torch.Tensor] = []
    category_sums: dict[str, float] = {}
    category_counts: dict[str, int] = {}
    eps = float(config["loss"].get("eps", 1e-8))
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            non_blocking = device.type == "cuda" and bool(config["data"].get("pin_memory", False))
            x = batch["input"].to(device, non_blocking=non_blocking)
            y = batch["target"].to(device, non_blocking=non_blocking)
            pred = model(x)
            metrics = basic_metrics(pred, y, eps=eps)
            collected.append(metrics)
            per_sample = _per_sample_relative_l2(pred, y, eps=eps).detach().cpu()
            zero_per_sample = _per_sample_relative_l2(torch.zeros_like(y), y, eps=eps).detach().cpu()
            zero_collected.append(zero_per_sample)
            for metadata, rel in zip(batch.get("metadata", []), per_sample.tolist()):
                category = str((metadata or {}).get("model_type", "unknown") or "unknown")
                category_sums[category] = category_sums.get(category, 0.0) + float(rel)
                category_counts[category] = category_counts.get(category, 0) + 1
    if not collected:
        raise RuntimeError("validation loader produced no batches")
    summary = {k: float(sum(item[k] for item in collected) / len(collected)) for k in collected[0]}
    if zero_collected:
        summary["zero_baseline_relative_l2"] = float(torch.cat(zero_collected).mean())
    for category in sorted(category_sums):
        count = max(1, category_counts[category])
        summary[f"relative_l2_by_category/{category}"] = float(category_sums[category] / count)
        summary[f"sample_count_by_category/{category}"] = float(category_counts[category])
    return summary


def estimate_memory(config: dict[str, Any]) -> dict[str, Any]:
    h = int(config["sampling"]["target_height"])
    w = int(config["sampling"]["target_width"])
    t = int(config["sampling"]["max_time_steps"])
    batch = int(config["train"]["batch_size"])
    model_cfg = build_model_config(config)
    model_name = str(model_cfg.get("name", "elastic_vti_fno")).lower()
    model_state_width = int(model_cfg.get("width", model_cfg.get("latent_dim", model_cfg.get("branch_width", 1))))
    branch_type = str(model_cfg.get("branch_type", "")).lower()
    if model_name in {"sdr_deeponet", "hybrid_sdr_deeponet", "sdrdeeponet"} and branch_type in {
        "factorized_temporal",
        "temporal_factorized",
        "factorized_time",
    }:
        branch_width = int(model_cfg.get("branch_width", model_state_width))
        bytes_fp32 = batch * (h * w + t) * branch_width * 4
        estimate_method = "factorized_temporal_branch"
    elif model_name in {"sdr_deeponet", "hybrid_sdr_deeponet", "sdrdeeponet"} and branch_type in {
        "local_spatial",
        "spatial_local",
        "local_temporal",
    }:
        branch_width = int(model_cfg.get("branch_width", model_state_width))
        temporal_rank = int(model_cfg.get("temporal_rank", 4))
        bytes_fp32 = batch * h * w * branch_width * 4 + (batch * h * w * temporal_rank + t * temporal_rank) * 4
        estimate_method = "local_spatial_temporal_branch"
    elif model_name in {"factorized_fno", "factorized_acoustic_fno", "fs_fno"}:
        spatial_w = int(model_cfg.get("spatial_width", model_state_width))
        temporal_w = int(model_cfg.get("temporal_width", model_state_width))
        # Spatial: per-frame features [B,H,W,spatial_w] streamed (only 1 frame active)
        # Temporal: per-point features [B*H*W,T,temporal_w]
        bytes_fp32 = batch * h * w * spatial_w * 4 + batch * h * w * t * temporal_w * 4
        estimate_method = "factorized_spatiotemporal"
    else:
        bytes_fp32 = batch * h * w * t * model_state_width * 4
        estimate_method = "dense_3d_proxy"
    output_bytes_fp32 = batch * h * w * t * 4
    info = {
        "selected_hwt": [h, w, t],
        "batch_size": batch,
        "model_name": model_name,
        "memory_estimate_method": estimate_method,
        "model_state_width": model_state_width,
        "width": model_state_width,
        "rough_hidden_tensor_mb": bytes_fp32 / (1024**2),
        "rough_output_tensor_mb": output_bytes_fp32 / (1024**2),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_mb"] = props.total_memory / (1024**2)
        try:
            free, total = torch.cuda.mem_get_info(0)
            info["gpu_free_mb"] = free / (1024**2)
        except Exception:
            info["gpu_free_mb"] = None
    return info


def _should_log_train_step(global_step: int, log_every_steps: int | None) -> bool:
    if log_every_steps is None:
        return False
    interval = int(log_every_steps)
    return interval > 0 and global_step % interval == 0


def _format_train_loss_log(
    epoch: int, batch_idx: int, global_step: int, loss_value: float, lr: float
) -> str:
    return (
        f"[train] step={global_step} epoch={epoch + 1} batch={batch_idx + 1} "
        f"loss={loss_value:.6e} lr={lr:.6e}"
    )


def _train_shuffle_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("data", {}).get("shuffle_train", True))


def _freeze_batchnorm_stats_if_requested(model: torch.nn.Module, config: dict[str, Any]) -> None:
    if not bool(config.get("train", {}).get("freeze_batchnorm_stats", False)):
        return
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _adaptive_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("train", {}).get("adaptive_importance_sampling") or {})


def _adaptive_sampling_enabled(config: dict[str, Any]) -> bool:
    if bool(config.get("train", {}).get("spatial_importance_sampling", {}).get("enabled", False)):
        return False
    return bool(_adaptive_sampling_config(config).get("enabled", False))


def _adaptive_sampling_weights(
    indices: list[int],
    residual_ema: dict[int, float],
    sampling_config: dict[str, Any],
) -> dict[int, float]:
    if not indices:
        return {}
    uniform_fraction = min(max(float(sampling_config.get("uniform_fraction", 0.2)), 0.0), 1.0)
    min_weight = max(float(sampling_config.get("min_weight", 1.0e-6)), 0.0)
    residual_power = max(float(sampling_config.get("residual_power", 1.0)), 0.0)
    default_residual = max(float(sampling_config.get("default_residual", 1.0)), 0.0)
    scores = []
    for index in indices:
        score = float(residual_ema.get(int(index), default_residual))
        if not torch.isfinite(torch.tensor(score)) or score < 0.0:
            score = default_residual
        scores.append(max(score, min_weight) ** residual_power)
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    if float(score_tensor.sum()) <= 0.0:
        adaptive = torch.full_like(score_tensor, 1.0 / len(indices))
    else:
        adaptive = score_tensor / score_tensor.sum()
    uniform = torch.full_like(adaptive, 1.0 / len(indices))
    probs = (1.0 - uniform_fraction) * adaptive + uniform_fraction * uniform
    probs = probs / probs.sum().clamp_min(1.0e-12)
    return {int(index): float(prob) for index, prob in zip(indices, probs.tolist())}


def _adaptive_epoch_indices(
    train_indices: list[int],
    residual_ema: dict[int, float],
    epoch: int,
    config: dict[str, Any],
) -> list[int]:
    if not _adaptive_sampling_enabled(config):
        return [int(index) for index in train_indices]
    sampling_config = _adaptive_sampling_config(config)
    samples_per_epoch = int(sampling_config.get("samples_per_epoch") or len(train_indices))
    samples_per_epoch = max(1, samples_per_epoch)
    indices = [int(index) for index in train_indices]
    weights = _adaptive_sampling_weights(indices, residual_ema, sampling_config)
    probabilities = torch.tensor([weights[index] for index in indices], dtype=torch.float64)
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 2026)) + int(epoch))
    chosen = torch.multinomial(probabilities, num_samples=samples_per_epoch, replacement=True, generator=generator)
    return [indices[int(i)] for i in chosen.tolist()]


def _per_sample_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    diff = (pred.detach() - target.detach()).reshape(pred.shape[0], -1)
    ref = target.detach().reshape(target.shape[0], -1)
    numerator = torch.linalg.norm(diff.float(), dim=1)
    denominator = torch.clamp(torch.linalg.norm(ref.float(), dim=1), min=float(eps))
    return numerator / denominator


def _component_balanced_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    component_weights: list[float] | tuple[float, ...],
    eps: float,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim < 3:
        raise ValueError("component-balanced loss expects batch and component dimensions")
    weights = torch.as_tensor(component_weights, device=pred.device, dtype=pred.dtype)
    if weights.ndim != 1 or weights.numel() != pred.shape[-1]:
        raise ValueError("component weights must match the final component dimension")
    if not bool(torch.isfinite(weights).all().item()) or bool((weights <= 0).any().item()):
        raise ValueError("component weights must be finite and positive")

    diff2 = (pred - target).pow(2)
    target2 = target.pow(2)
    if sample_weights is not None:
        try:
            diff2 = diff2 * sample_weights
            target2 = target2 * sample_weights
        except RuntimeError as exc:
            raise ValueError("sample weights are not broadcastable to prediction shape") from exc
    reduce_dims = tuple(range(1, pred.ndim - 1))
    numerator = torch.sqrt(diff2.sum(dim=reduce_dims).clamp_min(0.0))
    denominator = torch.sqrt(target2.sum(dim=reduce_dims).clamp_min(0.0)).clamp_min(float(eps))
    per_component = numerator / denominator
    per_sample = (per_component * weights.view(1, -1)).sum(dim=-1) / weights.sum()
    return per_sample.mean(), per_component.mean(dim=0)


def _elastic_supervised_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    config = _supervised_loss_config(loss_config)
    component_weights = config.pop("component_relative_l2_weights", None)
    if component_weights is None:
        return combined_loss(pred, target, **config)

    relative_weight = float(config.pop("relative_l2_weight", 1.0))
    base, parts = combined_loss(pred, target, relative_l2_weight=0.0, **config)
    balanced, components = _component_balanced_relative_l2(
        pred,
        target,
        component_weights=component_weights,
        eps=float(config.get("eps", 1.0e-8)),
    )
    parts["relative_l2_global"] = parts.pop("relative_l2")
    parts["relative_l2"] = float(balanced.detach().cpu())
    for index, value in enumerate(components.detach().cpu().tolist()):
        parts[f"relative_l2_component_{index}"] = float(value)
    return base + relative_weight * balanced, parts


def _update_residual_ema(
    residual_ema: dict[int, float],
    sample_indices: torch.Tensor,
    residuals: torch.Tensor,
    momentum: float,
) -> None:
    alpha = min(max(float(momentum), 0.0), 1.0)
    for sample_index, residual in zip(sample_indices.detach().cpu().tolist(), residuals.detach().cpu().tolist()):
        key = int(sample_index)
        value = float(residual)
        if not torch.isfinite(torch.tensor(value)):
            continue
        if key in residual_ema:
            residual_ema[key] = (1.0 - alpha) * float(residual_ema[key]) + alpha * value
        else:
            residual_ema[key] = value


def _spatial_importance_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("train", {}).get("spatial_importance_sampling") or {})


def _spatial_importance_sampling_enabled(config: dict[str, Any]) -> bool:
    return bool(_spatial_importance_sampling_config(config).get("enabled", False))


def _spatial_residual_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim not in {4, 5}:
        raise ValueError(
            "spatial residual sampling expects [batch,x,z,time] or [batch,x,z,time,component]"
        )
    residual = (pred.detach() - target.detach()).abs().float()
    reduce_dims = (0, 3) if pred.ndim == 4 else (0, 3, 4)
    return residual.mean(dim=reduce_dims).cpu()


def _update_spatial_residual_ema(
    spatial_residual_ema: torch.Tensor | None,
    residual_map: torch.Tensor,
    momentum: float,
) -> torch.Tensor:
    residual_map = residual_map.detach().cpu().float()
    alpha = min(max(float(momentum), 0.0), 1.0)
    if spatial_residual_ema is None:
        return residual_map.clone()
    ema = spatial_residual_ema.detach().cpu().float()
    if tuple(ema.shape) != tuple(residual_map.shape):
        raise ValueError(f"spatial residual EMA shape {tuple(ema.shape)} does not match {tuple(residual_map.shape)}")
    return (1.0 - alpha) * ema + alpha * residual_map


def _spatial_sampling_probabilities(
    height: int,
    width: int,
    spatial_residual_ema: torch.Tensor | None,
    sampling_config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    total = int(height) * int(width)
    if total <= 0:
        raise ValueError("spatial sampling requires positive height and width")
    uniform_fraction = min(max(float(sampling_config.get("uniform_fraction", 0.2)), 0.0), 1.0)
    min_weight = max(float(sampling_config.get("min_weight", 1.0e-6)), 0.0)
    residual_power = max(float(sampling_config.get("residual_power", 1.0)), 0.0)
    default_residual = max(float(sampling_config.get("default_residual", 1.0)), 0.0)

    if spatial_residual_ema is None:
        adaptive = torch.full((total,), 1.0 / total, dtype=torch.float64, device=device)
    else:
        if tuple(spatial_residual_ema.shape) != (int(height), int(width)):
            raise ValueError(
                f"spatial residual EMA shape {tuple(spatial_residual_ema.shape)} "
                f"does not match ({int(height)}, {int(width)})"
            )
        scores = spatial_residual_ema.to(device=device, dtype=torch.float64).reshape(-1)
        fallback = torch.full_like(scores, default_residual)
        scores = torch.where(torch.isfinite(scores) & (scores >= 0.0), scores, fallback)
        scores = torch.clamp(scores, min=min_weight) ** residual_power
        if float(scores.sum().detach().cpu()) <= 0.0:
            adaptive = torch.full((total,), 1.0 / total, dtype=torch.float64, device=device)
        else:
            adaptive = scores / scores.sum().clamp_min(1.0e-12)

    uniform = torch.full_like(adaptive, 1.0 / total)
    probabilities = (1.0 - uniform_fraction) * adaptive + uniform_fraction * uniform
    return probabilities / probabilities.sum().clamp_min(1.0e-12)


def _spatial_pixels_per_sample(total_spatial_points: int, sampling_config: dict[str, Any]) -> int:
    if sampling_config.get("pixels_per_sample") is not None:
        count = int(sampling_config["pixels_per_sample"])
    else:
        fraction = float(sampling_config.get("spatial_fraction", 1.0))
        count = int(round(float(total_spatial_points) * fraction))
    return max(1, int(count))


def _sample_spatial_indices(
    height: int,
    width: int,
    spatial_residual_ema: torch.Tensor | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampling_config = _spatial_importance_sampling_config(config)
    probabilities = _spatial_sampling_probabilities(
        height=height,
        width=width,
        spatial_residual_ema=spatial_residual_ema,
        sampling_config=sampling_config,
        device=torch.device("cpu"),
    )
    count = _spatial_pixels_per_sample(int(height) * int(width), sampling_config)
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 2026)) + 1_000_003 * int(epoch) + 9_176 * int(global_step))
    indices = torch.multinomial(probabilities, num_samples=count, replacement=True, generator=generator)
    return indices.to(device=device), probabilities[indices].to(device=device)


def _spatial_importance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    flat_indices: torch.Tensor,
    flat_probabilities: torch.Tensor,
    full_spatial_count: int,
    loss_config: dict[str, Any],
    reweight: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} != {tuple(target.shape)}")
    if pred.ndim == 4:
        batch, height, width, time_steps = pred.shape
        flat_shape = (batch, height * width, time_steps)
    elif pred.ndim == 5:
        batch, height, width, time_steps, components = pred.shape
        flat_shape = (batch, height * width, time_steps, components)
    else:
        raise ValueError(
            "spatial importance loss expects [batch,x,z,time] or [batch,x,z,time,component]"
        )
    flat_pred = pred.reshape(flat_shape)
    flat_target = target.reshape(flat_shape)
    flat_indices = flat_indices.to(device=pred.device, dtype=torch.long)
    selected_pred = flat_pred.index_select(1, flat_indices)
    selected_target = flat_target.index_select(1, flat_indices)
    if not reweight:
        sampled_loss_config = _supervised_loss_config(loss_config)
        sampled_loss_config["grad_weight"] = 0.0
        return _elastic_supervised_loss(selected_pred, selected_target, sampled_loss_config)

    probabilities = flat_probabilities.to(device=pred.device, dtype=pred.dtype).clamp_min(1.0e-12)
    weight_shape = [1, -1] + [1] * (selected_pred.ndim - 2)
    weights = (1.0 / (float(full_spatial_count) * probabilities)).view(*weight_shape)
    diff = selected_pred - selected_target
    weighted_diff2 = weights * diff.pow(2)
    weighted_target2 = weights * selected_target.pow(2)
    component_weights = loss_config.get("component_relative_l2_weights")
    component_parts: dict[str, float] = {}
    if component_weights is not None:
        rel, components = _component_balanced_relative_l2(
            selected_pred,
            selected_target,
            component_weights=component_weights,
            eps=float(loss_config.get("eps", 1.0e-8)),
            sample_weights=weights,
        )
        component_parts = {
            f"relative_l2_component_{index}": float(value)
            for index, value in enumerate(components.detach().cpu().tolist())
        }
    else:
        reduce_dims = tuple(range(1, selected_pred.ndim))
        numerator = torch.sqrt(weighted_diff2.sum(dim=reduce_dims).clamp_min(0.0))
        denominator = torch.clamp(
            torch.sqrt(weighted_target2.sum(dim=reduce_dims).clamp_min(0.0)),
            min=float(loss_config.get("eps", 1.0e-8)),
        )
        rel = (numerator / denominator).mean()
    mse = weighted_diff2.mean()
    loss = float(loss_config.get("relative_l2_weight", 1.0)) * rel + float(loss_config.get("mse_weight", 0.0)) * mse
    return loss, {
        "relative_l2": float(rel.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        **component_parts,
    }


PHYSICS_LOSS_KEYS = {"pde_weight", "energy_weight", "receiver_weight", "komega_weight", "drp_weight", "phase_vel_weight", "group_vel_weight"}
SUPERVISED_LOSS_KEYS = {
    "relative_l2_weight",
    "component_relative_l2_weights",
    "mse_weight",
    "grad_weight",
    "late_time_weight_alpha",
    "late_time_weight_beta",
    "eps",
}

# ── Gradient-based adaptive loss balancing (Wang et al. 2021) ──
_GRAD_EMA: dict[str, float] = {}
_GRAD_EMA_DECAY = 0.99
_ADAPTIVE_WARMUP_STEPS = 50


def _adaptive_physics_weights(
    sup_loss: torch.Tensor,
    phys_loss_terms: list[tuple[str, torch.Tensor]],
    model: torch.nn.Module,
    base_weights: dict[str, float],
    global_step: int,
) -> dict[str, float]:
    """Compute gradient-norm-adaptive physics loss weights.

    Uses the "learning to balance" approach: scale each physics loss weight
    so that its gradient norm matches the supervised loss gradient norm.
    This prevents physics losses in physical units from overwhelming
    the supervised loss in normalized units.
    """
    if global_step < _ADAPTIVE_WARMUP_STEPS or not phys_loss_terms:
        return base_weights

    # Compute supervised loss gradient norm (on a parameter subset for speed)
    # Use the last layer for efficiency
    last_param = None
    for name, p in model.named_parameters():
        if "head" in name or "fc2" in name or "output_proj" in name:
            last_param = p
            break
    if last_param is None:
        # Fallback: use last parameter
        for p in model.parameters():
            last_param = p
        last_param = list(model.parameters())[-1]

    # Supervised grad norm
    grad_sup = torch.autograd.grad(sup_loss, last_param, retain_graph=True, create_graph=False)[0]
    g_sup = grad_sup.norm().item() + 1e-12

    # Physics grad norms
    phys_norms: dict[str, float] = {}
    for name, phys_loss in phys_loss_terms:
        grad_phys = torch.autograd.grad(phys_loss, last_param, retain_graph=True, create_graph=False)[0]
        g_phys = grad_phys.norm().item() + 1e-12
        phys_norms[name] = g_phys

    # Compute adaptive weights
    adapted: dict[str, float] = {}
    for name, weight in base_weights.items():
        if weight == 0.0:
            adapted[name] = 0.0
            continue
        # Find matching physics loss name
        g_phys = phys_norms.get(name, g_sup)
        # Scale: λ_new = λ_old * (g_sup / g_phys) weighted by EMA
        raw_scale = g_sup / max(g_phys, 1e-12)
        ema_key = f"adapt_{name}"
        if ema_key not in _GRAD_EMA:
            _GRAD_EMA[ema_key] = raw_scale
        else:
            _GRAD_EMA[ema_key] = _GRAD_EMA_DECAY * _GRAD_EMA[ema_key] + (1 - _GRAD_EMA_DECAY) * raw_scale
        # Clamp to reasonable range
        adapted[name] = weight * min(max(_GRAD_EMA[ema_key], 1e-3), 1e3)
    return adapted


def _physics_loss_enabled(loss_config: dict[str, Any]) -> bool:
    return any(float(loss_config.get(key, 0.0) or 0.0) != 0.0 for key in PHYSICS_LOSS_KEYS)


def _supervised_loss_config(loss_config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in loss_config.items() if key in SUPERVISED_LOSS_KEYS}


def _batch_dx_dz_dt(batch: dict[str, Any]) -> tuple[float, float, float]:
    metadata = batch.get("metadata") or [{}]
    first = metadata[0] if metadata else {}
    dx = float(first.get("dx", 1.0) or 1.0)
    dz = float(first.get("dz", dx) or dx)
    time = batch["time"]
    if time.ndim == 2:
        time_values = time[0].detach().float()
    else:
        time_values = time.detach().float()
    dt_saved = float(torch.median(torch.diff(time_values)).item()) if time_values.numel() > 1 else 1.0
    return dx, dz, max(dt_saved, 1.0e-12)


def _target_time_steps(batch: dict[str, Any]) -> int:
    target = batch["target"]
    if target.ndim == 5 and target.shape[-1] == 2:
        return int(target.shape[-2])
    return int(target.shape[-1])


def _source_term_from_batch(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    source_map = batch["source_map"].to(device=device, dtype=dtype)
    if source_map.ndim == 4 and source_map.shape[1] == 1:
        source_map = source_map[:, 0]
    wavelet = batch.get("wavelet_saved_t")
    time_steps = _target_time_steps(batch)
    if wavelet is None:
        wavelet = torch.zeros(source_map.shape[0], time_steps, device=device, dtype=dtype)
    else:
        wavelet = wavelet.to(device=device, dtype=dtype)
        if wavelet.ndim == 1:
            wavelet = wavelet.view(1, -1).expand(source_map.shape[0], -1)
        if wavelet.shape[0] == 1 and source_map.shape[0] != 1:
            wavelet = wavelet.expand(source_map.shape[0], -1)
        if wavelet.shape[-1] != time_steps:
            wavelet = F.interpolate(
                wavelet.unsqueeze(1).float(),
                size=time_steps,
                mode="linear",
                align_corners=False,
            ).squeeze(1).to(dtype=dtype)
    scalar_source = source_map.unsqueeze(-1) * wavelet.view(wavelet.shape[0], 1, 1, time_steps)
    target = batch["target"]
    if target.ndim == 5 and target.shape[-1] == 2:
        vector_source = torch.zeros(
            scalar_source.shape[0],
            scalar_source.shape[1],
            scalar_source.shape[2],
            scalar_source.shape[3],
            2,
            device=device,
            dtype=dtype,
        )
        vector_source[..., 1] = scalar_source
        return vector_source
    return scalar_source


def _physics_loss_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, Any],
    normalization_stats: dict[str, Any],
    loss_config: dict[str, Any],
    physics_warmup_factor: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not _physics_loss_enabled(loss_config):
        return pred.new_tensor(0.0), {}
    eps = float(normalization_stats.get("eps", 1.0e-6))
    pred_phys = decode_standard(pred, normalization_stats["wavefield"], eps=eps)
    target_phys = decode_standard(target, normalization_stats["wavefield"], eps=eps)
    velocity = batch["velocity"].to(device=pred.device, dtype=pred_phys.dtype)
    source_mask = batch["source_map"].to(device=pred.device, dtype=pred_phys.dtype)
    if source_mask.ndim == 4 and source_mask.shape[1] == 1:
        source_mask = source_mask[:, 0]
    source_term = _source_term_from_batch(batch, device=pred.device, dtype=pred_phys.dtype)
    dx, dz, dt_saved = _batch_dx_dz_dt(batch)

    total = pred.new_tensor(0.0)
    parts: dict[str, float] = {}
    pde_weight = float(loss_config.get("pde_weight", 0.0) or 0.0)
    if pde_weight != 0.0:
        pde = elastic_vti_pde_residual_loss(
            pred_phys,
            velocity,
            source_term,
            dx=dx,
            dz=dz,
            dt_saved=dt_saved,
            source_mask=source_mask,
            normalized=bool(loss_config.get("pde_normalized", False)),
            rho_kg_m3=float(loss_config.get("rho_kg_m3", 1000.0)),
        )
        total = total + pde_weight * pde
        parts["pde_residual"] = float(pde.detach().cpu())
    energy_weight = float(loss_config.get("energy_weight", 0.0) or 0.0)
    if energy_weight != 0.0:
        energy = elastic_energy_stability_loss(pred_phys)
        total = total + energy_weight * energy
        parts["energy"] = float(energy.detach().cpu())
    receiver_weight = float(loss_config.get("receiver_weight", 0.0) or 0.0)
    if receiver_weight != 0.0:
        rec_cfg = dict(loss_config.get("receiver", {}) or {})
        receiver = elastic_receiver_waveform_loss(
            pred_phys,
            target_phys,
            z_indices=rec_cfg.get("z_indices"),
            x_stride=int(rec_cfg.get("x_stride", 20)),
            eps=float(loss_config.get("eps", 1.0e-8)),
        )
        total = total + receiver_weight * receiver
        parts["receiver"] = float(receiver.detach().cpu())
    komega_weight = float(loss_config.get("komega_weight", 0.0) or 0.0)
    if komega_weight != 0.0:
        if pred_phys.ndim == 5 and pred_phys.shape[-1] == 2:
            raise ValueError("elastic VTI training does not support acoustic komega loss; use pde/receiver/energy losses")
        band_weights = dict(loss_config.get("komega_band_weights", {}) or {})
        komega, band_parts = spectral_band_loss(
            pred_phys,
            target_phys,
            low_weight=float(band_weights.get("low", 1.0)),
            mid_weight=float(band_weights.get("mid", 1.0)),
            high_weight=float(band_weights.get("high", 1.0)),
            eps=float(loss_config.get("eps", 1.0e-8)),
        )
        total = total + komega_weight * komega
        parts["komega"] = float(komega.detach().cpu())
        parts.update(band_parts)
    # ── DRP (Dispersion-Relation-Preserving) Loss ──
    drp_weight = float(loss_config.get("drp_weight", 0.0) or 0.0)
    phase_vel_weight = float(loss_config.get("phase_vel_weight", 0.0) or 0.0)
    group_vel_weight = float(loss_config.get("group_vel_weight", 0.0) or 0.0)
    if drp_weight != 0.0 or phase_vel_weight != 0.0 or group_vel_weight != 0.0:
        if pred_phys.ndim == 5 and pred_phys.shape[-1] == 2:
            raise ValueError("elastic VTI training does not support acoustic DRP/velocity losses; use pde/receiver/energy losses")
        drp_loss, drp_diag = combined_dispersion_loss(
            pred_phys,
            target_phys,
            velocity,
            dx=dx,
            dz=dz,
            dt=dt_saved,
            drp_weight=drp_weight,
            phase_vel_weight=phase_vel_weight,
            group_vel_weight=group_vel_weight,
            eps=float(loss_config.get("eps", 1.0e-8)),
        )
        total = total + drp_loss
        parts.update({f"disp_{k}": v for k, v in drp_diag.items()})
    # Apply physics warmup factor (linear ramp from 0→1 over warmup epochs)
    if physics_warmup_factor < 1.0:
        total = total * physics_warmup_factor
        parts["physics_warmup_factor"] = float(physics_warmup_factor)
    return total, parts


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_model_from_checkpoint(checkpoint_path: str | Path, model: torch.nn.Module, device: torch.device) -> str:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return _sha256_file(checkpoint_path)


def _distillation_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("train", {}).get("distillation", {}) or {})


def _build_distillation_teacher(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module | None, dict[str, Any]]:
    distill_cfg = _distillation_config(config)
    if not bool(distill_cfg.get("enabled", False)):
        return None, {"enabled": False}
    checkpoint_path = distill_cfg.get("teacher_checkpoint")
    if checkpoint_path is None:
        raise ValueError("train.distillation.teacher_checkpoint is required when distillation is enabled")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    teacher_config = dict(checkpoint["model_config"])
    teacher = ElasticVTIFNO3D(**teacher_config)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher = teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, {
        "enabled": True,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": _sha256_file(checkpoint_path),
        "teacher_epoch": checkpoint.get("epoch"),
        "teacher_global_step": checkpoint.get("global_step"),
        "teacher_best_val_metric": checkpoint.get("best_val_metric"),
        "teacher_model_config": teacher_config,
        "weight": float(distill_cfg.get("weight", 0.0) or 0.0),
        "max_time_steps": distill_cfg.get("max_time_steps"),
        "input_feature_indices": list(distill_cfg.get("input_feature_indices", [0, 1, 2])),
    }


def _teacher_distillation_loss(
    pred: torch.Tensor,
    batch: dict[str, Any],
    teacher: torch.nn.Module | None,
    distillation_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if teacher is None or not bool(distillation_config.get("enabled", False)):
        return pred.new_tensor(0.0), {}
    weight = float(distillation_config.get("weight", 0.0) or 0.0)
    if weight == 0.0:
        return pred.new_tensor(0.0), {}
    feature_indices = [int(index) for index in distillation_config.get("input_feature_indices", [0, 1, 2])]
    max_time_steps = distillation_config.get("max_time_steps")
    time_steps = int(pred.shape[-1])
    if max_time_steps is not None:
        time_steps = min(time_steps, int(max_time_steps))
    student = pred[..., :time_steps]
    teacher_input = batch["input"].to(device=pred.device, dtype=torch.float32)[..., :time_steps, feature_indices]
    with torch.no_grad(), torch.autocast(device_type=pred.device.type, enabled=False):
        teacher_target = teacher(teacher_input).detach().float()
    shared_time = min(int(student.shape[-1]), int(teacher_target.shape[-1]))
    raw = F.mse_loss(student[..., :shared_time].float(), teacher_target[..., :shared_time])
    return pred.new_tensor(weight) * raw, {"teacher_distillation_mse": float(raw.detach().cpu())}


def _model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(model_cfg)
    kwargs.pop("name", None)
    return kwargs


def build_training_model(config: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_cfg = build_model_config(config)
    model_name = str(model_cfg.get("name", "elastic_vti_fno")).lower()
    if model_name in {"sdr_pino", "sdrpino"}:
        from .model_sdr import SDRPINO

        return SDRPINO(**_model_kwargs(model_cfg)), model_cfg
    if model_name in {"sdr_deeponet", "hybrid_sdr_deeponet", "sdrdeeponet"}:
        from .model_deeponet import SDRDeepONet

        return SDRDeepONet(**_model_kwargs(model_cfg)), model_cfg
    if model_name in {"factorized_fno", "factorized_acoustic_fno", "fs_fno"}:
        from .model_factorized import FactorizedAcousticFNO

        return FactorizedAcousticFNO(**_model_kwargs(model_cfg)), model_cfg
    if model_name in {"elastic_vti_fno", "elasticvtifno3d", "fno", "acoustic_fno", "acousticfno3d"}:
        return ElasticVTIFNO3D(**_model_kwargs(model_cfg)), model_cfg
    raise ValueError(f"unsupported model.name: {model_cfg.get('name')}")


def _restore_training_state(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, int, float, torch.Tensor | None]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val = float(checkpoint.get("best_val_metric", float("inf")))
    spatial_residual_ema = checkpoint.get("spatial_residual_ema")
    if spatial_residual_ema is not None:
        spatial_residual_ema = spatial_residual_ema.detach().cpu().float()
    return start_epoch, global_step, best_val, spatial_residual_ema


def run_training(
    config_path: str | Path,
    device_override: str | None = None,
    output_metrics: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if device_override:
        config["train"]["device"] = device_override
    set_seed(int(config.get("seed", 2026)))
    device = choose_device(config["train"].get("device", "auto"))
    splits = _load_or_create_splits(config)
    _write_split_snapshot(config, splits)
    stats = _load_or_compute_normalization(config, splits["train"])
    val_indices, validation_selection = _select_validation_indices(config, splits["val"])

    val_loader = _loader(config, val_indices, stats, shuffle=False)
    model, model_cfg = build_training_model(config)
    model = model.to(device)
    distillation_cfg = _distillation_config(config)
    distillation_teacher, distillation_info = _build_distillation_teacher(config, device)
    init_checkpoint = config["train"].get("init_checkpoint")
    init_checkpoint_sha256 = None
    if init_checkpoint is not None and resume_from_checkpoint is not None:
        raise ValueError("train.init_checkpoint cannot be combined with --resume")
    if init_checkpoint is not None:
        init_checkpoint_sha256 = _initialize_model_from_checkpoint(init_checkpoint, model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["learning_rate"]), weight_decay=float(config["train"]["weight_decay"]))
    scheduler = None
    if config["train"].get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["train"]["epochs"])))
    use_amp = bool(config["train"].get("amp", False)) and device.type == "cuda"
    # GradScaler does not support complex-valued spectral weights in this stack.
    # Autocast is still useful for the real-valued pointwise/projection paths.
    use_scaler = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and use_scaler)

    checkpoint_dir = ensure_dir(config["train"]["checkpoint_dir"])
    log_dir = ensure_dir(config["train"]["log_dir"])
    metrics_path = log_dir / "metrics.csv"
    best_val = float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    global_step = 0
    start_epoch = 0
    spatial_residual_ema: torch.Tensor | None = None
    last_val_metrics: dict[str, float] | None = None
    last_early_failure_gate = _evaluate_early_failure_gate(config, None)
    if resume_from_checkpoint is not None:
        start_epoch, global_step, best_val, spatial_residual_ema = _restore_training_state(
            resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
            device,
        )
    log_every_steps = config["train"].get("log_every_steps", 10)
    early_stopping_patience = config["train"].get("early_stopping_patience")
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0
    first_input_shape: list[int] | None = None
    last_train_loss = None
    last_supervised_parts: dict[str, float] = {}
    last_distillation_parts: dict[str, float] = {}
    residual_ema: dict[int, float] = {}
    residual_ema_path = log_dir / "residual_ema.json"
    adaptive_sampling_cfg = _adaptive_sampling_config(config)
    spatial_residual_ema_path = log_dir / "spatial_residual_ema.pt"
    spatial_sampling_cfg = _spatial_importance_sampling_config(config)
    metrics_mode = "a" if resume_from_checkpoint is not None and metrics_path.exists() else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_relative_l2", "val_mse", "lr"])
        if metrics_mode == "w":
            writer.writeheader()
        total_epochs = int(config["train"]["epochs"])
        checkpoint_every_epochs = config["train"].get("checkpoint_every_epochs", 1)
        if start_epoch >= total_epochs:
            raise ValueError(
                f"resume checkpoint epoch {start_epoch - 1} is already at or beyond configured total epochs {total_epochs}"
            )
        for epoch in range(start_epoch, total_epochs):
            # ── Physics loss warmup ──
            warmup_epochs = int(config["train"].get("physics_warmup_epochs", 0))
            if warmup_epochs > 0:
                physics_warmup_factor = min(1.0, float(epoch + 1) / float(max(1, warmup_epochs)))
            else:
                physics_warmup_factor = 1.0
            model.train()
            _freeze_batchnorm_stats_if_requested(model, config)
            train_losses: list[float] = []
            epoch_batch_idx = 0
            stop_epoch = False
            epoch_train_indices = _adaptive_epoch_indices(splits["train"], residual_ema, epoch, config)
            for train_batches in _train_batch_iterables_for_epoch(config, epoch_train_indices, stats, epoch):
                for batch in train_batches:
                    max_batches = config["train"].get("max_train_batches")
                    if max_batches is not None and epoch_batch_idx >= int(max_batches):
                        stop_epoch = True
                        break
                    non_blocking = device.type == "cuda" and bool(config["data"].get("pin_memory", False))
                    x = batch["input"].to(device, non_blocking=non_blocking)
                    y = batch["target"].to(device, non_blocking=non_blocking)
                    first_input_shape = list(x.shape) if first_input_shape is None else first_input_shape
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = model(x)
                        if _spatial_importance_sampling_enabled(config):
                            height, width = int(pred.shape[1]), int(pred.shape[2])
                            flat_indices, flat_probabilities = _sample_spatial_indices(
                                height=height,
                                width=width,
                                spatial_residual_ema=spatial_residual_ema,
                                epoch=epoch,
                                global_step=global_step,
                                config=config,
                                device=device,
                            )
                            loss, supervised_parts = _spatial_importance_loss(
                                pred,
                                y,
                                flat_indices=flat_indices,
                                flat_probabilities=flat_probabilities,
                                full_spatial_count=height * width,
                                loss_config=config["loss"],
                                reweight=bool(spatial_sampling_cfg.get("reweight_loss", True)),
                            )
                            physics_loss, phys_parts = _physics_loss_terms(pred, y, batch, stats, config["loss"], physics_warmup_factor)
                            # Adaptive balancing: scale physics loss to match supervised grad norm
                            if config["loss"].get("adaptive_balancing", False):
                                sup_loss, _ = _elastic_supervised_loss(pred, y, config["loss"])
                                base_w = {k: float(config["loss"].get(k, 0) or 0) for k in PHYSICS_LOSS_KEYS}
                                adaptive_w = _adaptive_physics_weights(
                                    sup_loss,
                                    [("physics", physics_loss)],
                                    model, base_w, global_step,
                                )
                                sum_adapted = sum(adaptive_w.values())
                                sum_base = max(sum(float(config["loss"].get(k, 0) or 0) for k in PHYSICS_LOSS_KEYS), 1e-12)
                                physics_loss = physics_loss * (sum_adapted / sum_base)
                            loss = loss + physics_loss
                        else:
                            loss, supervised_parts = _elastic_supervised_loss(pred, y, config["loss"])
                            physics_loss, phys_parts = _physics_loss_terms(pred, y, batch, stats, config["loss"], physics_warmup_factor)
                            # Adaptive balancing: scale physics loss to match supervised grad norm
                            if config["loss"].get("adaptive_balancing", False):
                                base_w = {k: float(config["loss"].get(k, 0) or 0) for k in PHYSICS_LOSS_KEYS}
                                adaptive_w = _adaptive_physics_weights(
                                    loss,
                                    [("physics", physics_loss)],
                                    model, base_w, global_step,
                                )
                                sum_adapted = sum(adaptive_w.values())
                                sum_base = max(sum(float(config["loss"].get(k, 0) or 0) for k in PHYSICS_LOSS_KEYS), 1e-12)
                                physics_loss = physics_loss * (sum_adapted / sum_base)
                            loss = loss + physics_loss
                        distillation_loss, distillation_parts = _teacher_distillation_loss(
                            pred,
                            batch,
                            distillation_teacher,
                            distillation_cfg,
                        )
                        loss = loss + distillation_loss
                        if distillation_parts:
                            last_distillation_parts = distillation_parts
                    if _adaptive_sampling_enabled(config):
                        residuals = _per_sample_relative_l2(pred, y, eps=float(config["loss"].get("eps", 1e-8)))
                        _update_residual_ema(
                            residual_ema,
                            batch["sample_index"],
                            residuals,
                            momentum=float(adaptive_sampling_cfg.get("ema_momentum", 0.2)),
                        )
                    if _spatial_importance_sampling_enabled(config):
                        spatial_residual_ema = _update_spatial_residual_ema(
                            spatial_residual_ema,
                            _spatial_residual_map(pred, y),
                            momentum=float(spatial_sampling_cfg.get("ema_momentum", 0.2)),
                        )
                    if use_scaler:
                        scaler.scale(loss).backward()
                        if config["train"].get("grad_clip") is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if config["train"].get("grad_clip") is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip"]))
                        optimizer.step()
                    global_step += 1
                    loss_value = float(loss.detach().cpu())
                    train_losses.append(loss_value)
                    last_train_loss = loss_value
                    last_supervised_parts = supervised_parts
                    if _should_log_train_step(global_step, log_every_steps):
                        message = _format_train_loss_log(
                            epoch=epoch,
                            batch_idx=epoch_batch_idx,
                            global_step=global_step,
                            loss_value=loss_value,
                            lr=float(optimizer.param_groups[0]["lr"]),
                        )
                        component_text = " ".join(
                            f"{key}={value:.6e}"
                            for key, value in sorted(supervised_parts.items())
                            if key.startswith("relative_l2_component_")
                        )
                        print(f"{message} {component_text}".rstrip(), flush=True)
                    if not torch.isfinite(torch.tensor(loss_value)):
                        raise RuntimeError("non-finite training loss")
                    epoch_batch_idx += 1
                del train_batches
                if stop_epoch:
                    break
            if scheduler:
                scheduler.step()
            val_metrics = validate(
                model,
                val_loader,
                device,
                config,
                max_batches=config["train"].get("max_val_batches"),
            )
            last_val_metrics = dict(val_metrics)
            last_early_failure_gate = _evaluate_early_failure_gate(config, val_metrics, epoch=epoch)
            train_loss = float(sum(train_losses) / len(train_losses)) if train_losses else float("nan")
            (
                best_val,
                epochs_without_improvement,
                improved,
                stop_training,
            ) = _update_early_stopping_state(
                best_val=best_val,
                val_metric=val_metrics["relative_l2"],
                epochs_without_improvement=epochs_without_improvement,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
            if (
                last_early_failure_gate.get("status") == "failed"
                and bool(last_early_failure_gate.get("stop_on_failure", False))
            ):
                stop_training = True
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_relative_l2": val_metrics["relative_l2"],
                    "val_mse": val_metrics["mse"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            f.flush()
            if _adaptive_sampling_enabled(config):
                write_json(residual_ema_path, residual_ema)
            if _spatial_importance_sampling_enabled(config) and spatial_residual_ema is not None:
                torch.save(
                    {
                        "ema": spatial_residual_ema.cpu(),
                        "config": spatial_sampling_cfg,
                        "shape": list(spatial_residual_ema.shape),
                    },
                    spatial_residual_ema_path,
                )
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_val_metric": best_val,
                "model_config": model_cfg,
                "data_config": config["data"],
                "full_config": config,
                "normalization_stats": stats,
                "split_manifest": splits,
                "validation_selection": validation_selection,
                "early_failure_gate": last_early_failure_gate,
                "distillation": {
                    **distillation_info,
                    "last_parts": last_distillation_parts,
                },
                "supervised_loss_parts": last_supervised_parts,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
                "init_checkpoint_sha256": init_checkpoint_sha256,
                "adaptive_importance_sampling": {
                    "enabled": _adaptive_sampling_enabled(config),
                    "config": adaptive_sampling_cfg,
                    "residual_ema_path": str(residual_ema_path) if _adaptive_sampling_enabled(config) else None,
                    "tracked_sample_count": len(residual_ema),
                },
                "spatial_importance_sampling": {
                    "enabled": _spatial_importance_sampling_enabled(config),
                    "config": spatial_sampling_cfg,
                    "residual_ema_path": (
                        str(spatial_residual_ema_path)
                        if _spatial_importance_sampling_enabled(config) and spatial_residual_ema is not None
                        else None
                    ),
                    "tracked_shape": list(spatial_residual_ema.shape) if spatial_residual_ema is not None else None,
                },
                "spatial_residual_ema": spatial_residual_ema.cpu() if spatial_residual_ema is not None else None,
                "source_repo_commit": repo_commit(),
                "adaptation_git_commit": repo_commit(),
            }
            if _should_save_epoch_checkpoint(epoch, total_epochs, checkpoint_every_epochs):
                save_checkpoint(last_path, payload)
            if improved:
                save_checkpoint(best_path, payload)
            if stop_training:
                if (
                    last_early_failure_gate.get("status") == "failed"
                    and bool(last_early_failure_gate.get("stop_on_failure", False))
                ):
                    print(
                        f"[early_failure] epoch={epoch + 1} "
                        f"validation_relative_l2={last_early_failure_gate['validation_relative_l2']:.6e} "
                        f"zero_baseline_relative_l2={last_early_failure_gate['zero_baseline_relative_l2']:.6e}",
                        flush=True,
                    )
                else:
                    print(
                        f"[early_stopping] epoch={epoch + 1} best_val={best_val:.6e} "
                        f"stale_epochs={epochs_without_improvement} patience={int(early_stopping_patience)}",
                        flush=True,
                    )
                break

    # Reload one batch after training to record output shape and finite forward gate.
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        non_blocking = device.type == "cuda" and bool(config["data"].get("pin_memory", False))
        pred = model(batch["input"].to(device, non_blocking=non_blocking))
        output_shape = list(pred.shape)
        prediction_finite_ratio = float(torch.isfinite(pred).float().mean().cpu())
    result = {
        "device": str(device),
        "input_shape": first_input_shape,
        "output_shape": output_shape,
        "train_loss": last_train_loss,
        "validation_relative_l2": best_val,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "resume_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "init_checkpoint_sha256": init_checkpoint_sha256,
        "metrics_csv": str(metrics_path),
        "splits": splits["sample_counts"],
        "normalization_stats": stats,
        "validation_selection": validation_selection,
        "validation_safety_metrics": last_val_metrics or {},
        "early_failure_gate": last_early_failure_gate,
        "distillation": {
            **distillation_info,
            "last_parts": last_distillation_parts,
        },
        "supervised_loss_parts": last_supervised_parts,
        "adaptive_importance_sampling": {
            "enabled": _adaptive_sampling_enabled(config),
            "config": adaptive_sampling_cfg,
            "residual_ema_path": str(residual_ema_path) if _adaptive_sampling_enabled(config) else None,
            "tracked_sample_count": len(residual_ema),
        },
        "spatial_importance_sampling": {
            "enabled": _spatial_importance_sampling_enabled(config),
            "config": spatial_sampling_cfg,
            "residual_ema_path": (
                str(spatial_residual_ema_path)
                if _spatial_importance_sampling_enabled(config) and spatial_residual_ema is not None
                else None
            ),
            "tracked_shape": list(spatial_residual_ema.shape) if spatial_residual_ema is not None else None,
        },
        "memory_estimate": estimate_memory(config),
        "prediction_finite_ratio": prediction_finite_ratio,
    }
    if device.type == "cuda":
        result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        result["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024**2)
    if output_metrics:
        write_json(output_metrics, result)
    return result
