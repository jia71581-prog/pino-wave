#!/usr/bin/env python3
"""Meta-train early-feature modulation from two post-onset snapshots.

Future ground truth is opened only for training-split episodes.  At deployment,
the resulting adapter receives velocity, source information, and exactly two
post-onset snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES, build_manifest
from saved_time_phase_operator_v4.instance_adaptation.data_guard import (
    GuardedOnsetDataset,
)
from saved_time_phase_operator_v4.instance_adaptation.feature_modulation import (
    EarlyFeatureOnsetAdapter,
    metric_aligned_relative_loss,
)
from scripts.run_v5_instance_adaptation import (
    _load_parent,
    resolve_saved_time_parent_config,
)


def build_balanced_meta_episodes(
    manifest,
    *,
    split: str,
    per_family: int,
    seed: int,
    allowed_sample_ids=None,
):
    """Select a reproducible, equally represented episode set."""

    count = int(per_family)
    if count <= 0:
        raise ValueError("meta episode count per family must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    allowed = (
        None
        if allowed_sample_ids is None
        else {str(value) for value in allowed_sample_ids}
    )
    selected: dict[str, tuple[object, ...]] = {}
    for family in ALLOWED_MEDIUM_TYPES:
        candidates = tuple(
            record
            for record in manifest.records
            if record.split == split and record.medium_type == family
            and (allowed is None or record.sample_id in allowed)
        )
        if len(candidates) < count:
            raise ValueError(
                f"meta episode family support is insufficient: {family}"
            )
        order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
        selected[family] = tuple(candidates[index] for index in order)
    return tuple(
        selected[family][index]
        for index in range(count)
        for family in ALLOWED_MEDIUM_TYPES
    )


def select_future_time_indices(
    *,
    length: int,
    observed_indices: tuple[int, int],
    count: int,
    seed: int,
    epoch: int,
) -> torch.Tensor:
    """Select deterministic, stratified saved times strictly after observations."""

    total = int(length)
    requested = int(count)
    start = int(observed_indices[1]) + 1
    if total <= start or requested <= 0 or requested > total - start:
        raise ValueError("future time selection has invalid support")
    generator = torch.Generator().manual_seed(int(seed) + 1_000_003 * int(epoch))
    support = torch.arange(start, total, dtype=torch.long)
    edges = torch.linspace(0, len(support), requested + 1, dtype=torch.float64)
    selected: list[int] = []
    for index in range(requested):
        low = int(torch.floor(edges[index]))
        high = int(torch.floor(edges[index + 1]))
        high = max(low + 1, min(high, len(support)))
        offset = int(torch.randint(low, high, (1,), generator=generator))
        selected.append(int(support[offset]))
    return torch.tensor(sorted(selected), dtype=torch.long)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_wavefield_times(
    handle: h5py.File,
    source_index: int,
    time_indices,
) -> torch.Tensor:
    """Read an explicit training-only time subset from the source HDF5."""

    indices = tuple(int(value) for value in time_indices)
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("training time indices must be nonempty, unique, and sorted")
    return torch.from_numpy(
        np.asarray(
            handle["wavefield"][int(source_index), list(indices), :, :],
            dtype=np.float32,
        )
    )


def _to_pinned(tensor: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    value = tensor.detach().contiguous().cpu()
    return value.pin_memory() if enabled else value


def _active_families(epoch: int, epochs: int) -> tuple[str, ...]:
    if int(epochs) < 3:
        return tuple(ALLOWED_MEDIUM_TYPES)
    stage = min(2, (3 * int(epoch)) // int(epochs))
    return (
        ("uniform",),
        ("uniform", "layered"),
        tuple(ALLOWED_MEDIUM_TYPES),
    )[stage]


def curriculum_stage_is_complete(active_families) -> bool:
    """Return whether a checkpoint has trained on every registered family."""

    active = tuple(str(value) for value in active_families)
    return len(active) == len(ALLOWED_MEDIUM_TYPES) and set(active) == set(
        ALLOWED_MEDIUM_TYPES
    )


def _batch_indices(
    cached: list[dict[str, object]],
    *,
    active_families: tuple[str, ...],
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    selected = [
        index
        for index, sample in enumerate(cached)
        if str(sample["medium_type"]) in set(active_families)
    ]
    if not selected:
        raise ValueError("curriculum stage contains no cached episodes")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(selected), generator=generator).tolist()
    shuffled = [selected[index] for index in order]
    size = int(batch_size)
    if size <= 0:
        raise ValueError("meta batch size must be positive")
    return [shuffled[start : start + size] for start in range(0, len(shuffled), size)]


def _stack_batch(
    cached: list[dict[str, object]],
    indices: list[int],
    *,
    epoch: int,
    device: torch.device,
) -> dict[str, object]:
    samples = [cached[index] for index in indices]
    non_blocking = device.type == "cuda"

    def stack(name: str) -> torch.Tensor:
        return torch.stack([sample[name] for sample in samples]).to(
            device, non_blocking=non_blocking
        )

    epoch_index = int(epoch)
    return {
        "velocity": stack("velocity"),
        "source": stack("source"),
        "source_map": stack("source_map"),
        "observed": stack("observed"),
        "observed_time_s": stack("observed_time_s"),
        "travel": (
            None
            if samples[0]["travel"] is None
            else stack("travel")
        ),
        "truth": torch.stack(
            [sample["truth_by_epoch"][epoch_index] for sample in samples]
        ).to(device, non_blocking=non_blocking),
        "time_s": torch.stack(
            [sample["time_by_epoch"][epoch_index] for sample in samples]
        ).to(device, non_blocking=non_blocking),
        "x_m": samples[0]["x_m"].to(device, non_blocking=non_blocking),
        "z_m": samples[0]["z_m"].to(device, non_blocking=non_blocking),
        "medium_type": tuple(str(sample["medium_type"]) for sample in samples),
        "sample_id": tuple(str(sample["sample_id"]) for sample in samples),
    }


def _cache_training_episodes(
    *,
    source_h5: Path,
    manifest,
    episodes,
    epochs: int,
    time_points: int,
    seed: int,
    travel_time_h5: str | Path | None,
    pin_memory: bool,
) -> list[dict[str, object]]:
    dataset = GuardedOnsetDataset(
        source_h5,
        manifest,
        split="train",
        travel_time_h5=travel_time_h5,
    )
    by_source = {
        record.source_index: index
        for index, record in enumerate(dataset.records.records)
    }
    cached: list[dict[str, object]] = []
    try:
        with h5py.File(source_h5, "r", swmr=True) as handle:
            for episode_index, metadata in enumerate(episodes):
                record = dataset[by_source[metadata.source_index]]
                truth_by_epoch: list[torch.Tensor] = []
                time_by_epoch: list[torch.Tensor] = []
                for epoch in range(int(epochs)):
                    selected = select_future_time_indices(
                        length=len(record.time_s),
                        observed_indices=record.observed_indices,
                        count=int(time_points),
                        seed=int(seed) + 10_007 * episode_index,
                        epoch=epoch,
                    )
                    truth = read_training_wavefield_times(
                        handle,
                        metadata.source_index,
                        selected.tolist(),
                    )
                    truth_by_epoch.append(_to_pinned(truth, enabled=pin_memory))
                    time_by_epoch.append(
                        _to_pinned(record.time_s[selected], enabled=pin_memory)
                    )
                cached.append(
                    {
                        "velocity": _to_pinned(
                            record.velocity_mps, enabled=pin_memory
                        ),
                        "source": _to_pinned(
                            record.source_parameters, enabled=pin_memory
                        ),
                        "source_map": _to_pinned(
                            record.source_map, enabled=pin_memory
                        ),
                        "observed": _to_pinned(
                            record.observed_wavefield, enabled=pin_memory
                        ),
                        "observed_time_s": _to_pinned(
                            record.time_s[list(record.observed_indices)],
                            enabled=pin_memory,
                        ),
                        "travel": (
                            None
                            if record.dense_travel_time_s is None
                            else _to_pinned(
                                record.dense_travel_time_s, enabled=pin_memory
                            )
                        ),
                        "truth_by_epoch": truth_by_epoch,
                        "time_by_epoch": time_by_epoch,
                        "x_m": _to_pinned(record.x_m, enabled=pin_memory),
                        "z_m": _to_pinned(record.z_m, enabled=pin_memory),
                        "medium_type": metadata.medium_type,
                        "sample_id": metadata.sample_id,
                        "group_id": metadata.group_id,
                        "observed_indices": record.observed_indices,
                    }
                )
    finally:
        dataset.close()
    return cached


def _predict_batch(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    batch: dict[str, object],
    *,
    time_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared, latent = adapter.prepare_sources(
        batch["velocity"],
        batch["source"],
        batch["source_map"],
        batch["observed"],
        normalizer,
        conditioner_wavefield=batch.get("conditioner_wavefield"),
    )
    dense_grid = adapter.parent.prepare_dense_grid(
        prepared,
        x_m=batch["x_m"],
        z_m=batch["z_m"],
        travel_time_s=batch["travel"],
    )
    prediction = adapter.parent.predict_wavefield(
        prepared,
        batch["time_s"],
        dense_grid=dense_grid,
        time_block=int(time_block),
    )
    return prediction, latent


def _parent_observed_residual_batch(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    batch: dict[str, object],
    *,
    time_block: int,
) -> torch.Tensor:
    """Return two measurable onset residuals from the correctly loaded parent."""

    parent = adapter.parent
    velocity = torch.as_tensor(batch["velocity"])
    source = torch.as_tensor(batch["source"])
    source_map = torch.as_tensor(batch["source_map"])
    mapping = torch.arange(source.shape[0], dtype=torch.long, device=source.device)
    with torch.no_grad():
        medium = parent.encode_medium(velocity, normalizer)
        prepared = parent.prepare_sources(
            medium,
            source,
            source_map,
            normalizer,
            record_to_medium=mapping,
        )
        dense_grid = parent.prepare_dense_grid(
            prepared,
            x_m=batch["x_m"],
            z_m=batch["z_m"],
            travel_time_s=batch["travel"],
        )
        prediction = parent.predict_wavefield(
            prepared,
            batch["observed_time_s"],
            dense_grid=dense_grid,
            time_block=int(time_block),
        )
    observed = torch.as_tensor(
        batch["observed"], dtype=prediction.dtype, device=prediction.device
    )
    if prediction.shape != observed.shape:
        raise ValueError("parent onset prediction does not match guarded observations")
    return (observed - prediction).detach()


def metric_aligned_energy_weights(
    error_energy: torch.Tensor,
    target_energy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the joint relative L2 value and its derivative with respect to SSE.

    The detached derivative lets training reproduce the exact full-time loss
    gradient by recomputing small time chunks instead of retaining the frozen
    parent's complete autograd graph in GPU memory.
    """

    error = torch.as_tensor(error_energy).float()
    target = torch.as_tensor(
        target_energy, dtype=error.dtype, device=error.device
    )
    if error.ndim != 1 or target.shape != error.shape:
        raise ValueError("metric energies must be matching per-record vectors")
    if bool(torch.any(error < 0.0)) or bool(torch.any(target < 0.0)):
        raise ValueError("metric energies must be nonnegative")
    error_norm = error.sqrt()
    target_norm = target.sqrt().clamp_min(1.0e-8)
    value = (error_norm / target_norm).mean()
    derivative = 0.5 / (
        float(error.numel())
        * error_norm.clamp_min(1.0e-12)
        * target_norm
    )
    return value, derivative


def _streaming_metric_aligned_backward(
    adapter: EarlyFeatureOnsetAdapter,
    normalizer,
    batch: dict[str, object],
    *,
    time_block: int,
    gradient_time_chunk: int,
) -> torch.Tensor:
    """Backpropagate exact joint relative-L2 gradients in bounded time chunks."""

    truth = torch.as_tensor(batch["truth"])
    time_s = torch.as_tensor(batch["time_s"])
    if truth.ndim != 4 or time_s.ndim != 2 or truth.shape[:2] != time_s.shape:
        raise ValueError("streaming truth and time tensors do not match")
    chunk = int(gradient_time_chunk)
    if chunk <= 0:
        raise ValueError("gradient time chunk must be positive")
    target_energy = truth.float().square().flatten(2).sum(dim=(1, 2))
    error_energy = torch.zeros_like(target_energy)
    with torch.no_grad():
        for start in range(0, truth.shape[1], chunk):
            stop = min(start + chunk, truth.shape[1])
            piece = dict(batch)
            piece["truth"] = truth[:, start:stop]
            piece["time_s"] = time_s[:, start:stop]
            prediction, _ = _predict_batch(
                adapter, normalizer, piece, time_block=time_block
            )
            error_energy.add_(
                (prediction.float() - piece["truth"].float())
                .square()
                .flatten(2)
                .sum(dim=(1, 2))
            )
    field_loss, weights = metric_aligned_energy_weights(
        error_energy, target_energy
    )
    weights = weights.detach()
    for start in range(0, truth.shape[1], chunk):
        stop = min(start + chunk, truth.shape[1])
        piece = dict(batch)
        piece["truth"] = truth[:, start:stop]
        piece["time_s"] = time_s[:, start:stop]
        prediction, _ = _predict_batch(
            adapter, normalizer, piece, time_block=time_block
        )
        piece_energy = (
            (prediction.float() - piece["truth"].float())
            .square()
            .flatten(2)
            .sum(dim=(1, 2))
        )
        (weights * piece_energy).sum().backward()
    return field_loss.detach()


def train_feature_meta(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    device_name: str = "cuda",
    epochs_override: int | None = None,
    per_family_override: int | None = None,
    batch_size_override: int | None = None,
    time_points_override: int | None = None,
    learning_rate_override: float | None = None,
    time_block_override: int | None = None,
    gradient_time_chunk_override: int | None = None,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    source_h5 = Path(config["source_h5"]).expanduser()
    manifest = build_manifest(source_h5)
    device = torch.device(
        device_name
        if device_name != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    parent, normalizer = _load_parent(config, manifest, device)
    adapter = EarlyFeatureOnsetAdapter(
        parent,
        latent_dim=int(config.get("latent_dim", 16)),
        lora_rank=int(config.get("lora_rank", 4)),
        max_scale=float(config.get("max_scale", 0.1)),
        max_bias=float(config.get("max_bias", 0.05)),
    ).to(device)
    adapter.train()
    adapter.parent.eval()
    epochs = int(
        config.get("epochs", 3) if epochs_override is None else epochs_override
    )
    per_family = int(
        config.get("meta_episodes_per_family", 12)
        if per_family_override is None
        else per_family_override
    )
    batch_size = int(
        config.get("batch_size", 2)
        if batch_size_override is None
        else batch_size_override
    )
    time_points = int(
        config.get("time_points", 24)
        if time_points_override is None
        else time_points_override
    )
    learning_rate = float(
        config.get("learning_rate", 2.0e-4)
        if learning_rate_override is None
        else learning_rate_override
    )
    time_block = int(
        config.get("time_block", 4)
        if time_block_override is None
        else time_block_override
    )
    gradient_time_chunk = int(
        config.get("gradient_time_chunk", 0)
        if gradient_time_chunk_override is None
        else gradient_time_chunk_override
    )
    conditioner_input = str(config.get("conditioner_input", "raw_observed"))
    if conditioner_input not in {"raw_observed", "parent_onset_residual"}:
        raise ValueError("unknown feature-meta conditioner input")
    if min(epochs, per_family, batch_size, time_points, time_block) <= 0:
        raise ValueError("feature meta-training counts must be positive")
    if gradient_time_chunk < 0:
        raise ValueError("gradient time chunk cannot be negative")
    if learning_rate <= 0.0:
        raise ValueError("feature meta-training learning rate must be positive")
    episodes = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=per_family,
        seed=int(config.get("seed", 17)),
    )
    saved_time_parent = resolve_saved_time_parent_config(config)
    travel_time_h5 = config.get("travel_time_h5")
    if travel_time_h5 is None and saved_time_parent is not None:
        travel_time_h5 = saved_time_parent.get("travel_time_h5")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    cache_started = time.perf_counter()
    cached = _cache_training_episodes(
        source_h5=source_h5,
        manifest=manifest,
        episodes=episodes,
        epochs=epochs,
        time_points=time_points,
        seed=int(config.get("seed", 17)),
        travel_time_h5=travel_time_h5,
        pin_memory=device.type == "cuda",
    )
    cache_elapsed = time.perf_counter() - cache_started
    parameters = adapter.adapter_parameters()
    optimizer_options = {"lr": learning_rate, "weight_decay": 1.0e-5}
    if device.type == "cuda":
        optimizer_options["fused"] = True
    optimizer = torch.optim.AdamW(parameters, **optimizer_options)
    best = float("inf")
    started = time.perf_counter()
    with metrics_path.open("a", buffering=1) as log:
        log.write(
            json.dumps(
                {
                    "event": "cache_ready",
                    "episodes": len(cached),
                    "cache_elapsed_s": cache_elapsed,
                    "batch_size": batch_size,
                    "time_points": time_points,
                    "time_block": time_block,
                    "gradient_time_chunk": gradient_time_chunk,
                    "conditioner_input": conditioner_input,
                    "trainable_parameters": sum(p.numel() for p in parameters),
                },
                sort_keys=True,
            )
            + "\n"
        )
        global_step = 0
        for epoch in range(epochs):
            active = _active_families(epoch, epochs)
            batches = _batch_indices(
                cached,
                active_families=active,
                batch_size=batch_size,
                seed=int(config.get("seed", 17)) + epoch,
            )
            epoch_losses: list[float] = []
            for batch_indices in batches:
                batch = _stack_batch(
                    cached, batch_indices, epoch=epoch, device=device
                )
                if conditioner_input == "parent_onset_residual":
                    batch["conditioner_wavefield"] = _parent_observed_residual_batch(
                        adapter,
                        normalizer,
                        batch,
                        time_block=time_block,
                    )
                optimizer.zero_grad(set_to_none=True)
                if gradient_time_chunk:
                    field_loss = _streaming_metric_aligned_backward(
                        adapter,
                        normalizer,
                        batch,
                        time_block=time_block,
                        gradient_time_chunk=gradient_time_chunk,
                    )
                    latent = adapter.conditioner(
                        batch["velocity"],
                        batch["source"],
                        batch.get("conditioner_wavefield", batch["observed"]),
                    )
                else:
                    prediction, latent = _predict_batch(
                        adapter, normalizer, batch, time_block=time_block
                    )
                    field_loss = metric_aligned_relative_loss(
                        prediction, batch["truth"]
                    )
                coefficients = adapter.modulator.affine(latent)
                modulation_loss = coefficients.square().mean()
                latent_loss = latent.square().mean()
                regularization = (
                    float(config.get("modulation_weight", 1.0e-4))
                    * modulation_loss
                    + float(config.get("latent_weight", 1.0e-6)) * latent_loss
                )
                loss = field_loss + regularization
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite feature meta-training loss")
                if gradient_time_chunk:
                    regularization.backward()
                else:
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(config.get("gradient_clip", 1.0))
                )
                optimizer.step()
                global_step += 1
                value = float(field_loss.detach())
                epoch_losses.append(value)
                memory = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                log.write(
                    json.dumps(
                        {
                            "event": "step",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "field_loss": value,
                            "loss": float(loss.detach()),
                            "grad_norm": float(grad_norm),
                            "families": batch["medium_type"],
                            "records": len(batch_indices),
                            "peak_cuda_bytes": memory,
                            "elapsed_s": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            epoch_loss = float(np.mean(epoch_losses))
            payload = {
                "adapter_kind": "early_feature_meta",
                "adapter_state": {
                    key: value.detach().cpu()
                    for key, value in adapter.state_dict().items()
                    if key.startswith("conditioner.")
                    or key.startswith("modulator.")
                },
                "optimizer_state": optimizer.state_dict(),
                "manifest_digest": manifest.digest,
                "source_h5_sha256": _sha256(source_h5),
                "parent_checkpoint": str(config["parent_checkpoint"]),
                "parent_checkpoint_sha256": _sha256(config["parent_checkpoint"]),
                "training_sample_ids": tuple(
                    str(sample["sample_id"]) for sample in cached
                ),
                "training_group_ids": tuple(
                    str(sample["group_id"]) for sample in cached
                ),
                "epoch": epoch + 1,
                "field_loss": epoch_loss,
                "latent_dim": int(config.get("latent_dim", 16)),
                "lora_rank": int(config.get("lora_rank", 4)),
                "max_scale": float(config.get("max_scale", 0.1)),
                "max_bias": float(config.get("max_bias", 0.05)),
                "conditioner_input": conditioner_input,
                "future_truth_used_only_for_train_episode": True,
                "field_loss_kind": "per_record_joint_space_time_relative_l2",
            }
            checkpoint = output / f"epoch_{epoch + 1:04d}.pt"
            torch.save(payload, checkpoint)
            best_eligible = curriculum_stage_is_complete(active)
            if best_eligible and epoch_loss < best:
                best = epoch_loss
                torch.save(payload, output / "best.pt")
            log.write(
                json.dumps(
                    {
                        "event": "epoch",
                        "epoch": epoch + 1,
                        "field_loss": epoch_loss,
                        "active_families": active,
                        "checkpoint": str(checkpoint),
                        "best_checkpoint_eligible": best_eligible,
                        "best_field_loss": None if not np.isfinite(best) else best,
                        "elapsed_s": time.perf_counter() - started,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return output / "best.pt"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--per-family", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--time-points", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--time-block", type=int)
    parser.add_argument("--gradient-time-chunk", type=int)
    args = parser.parse_args(argv)
    checkpoint = train_feature_meta(
        args.config,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs_override=args.epochs,
        per_family_override=args.per_family,
        batch_size_override=args.batch_size,
        time_points_override=args.time_points,
        learning_rate_override=args.learning_rate,
        time_block_override=args.time_block,
        gradient_time_chunk_override=args.gradient_time_chunk,
    )
    print(json.dumps({"checkpoint": str(checkpoint)}, sort_keys=True))
    return 0


__all__ = [
    "build_balanced_meta_episodes",
    "curriculum_stage_is_complete",
    "metric_aligned_energy_weights",
    "read_training_wavefield_times",
    "select_future_time_indices",
    "train_feature_meta",
]


if __name__ == "__main__":
    raise SystemExit(main())
