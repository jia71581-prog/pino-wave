#!/usr/bin/env python3
"""Meta-pretrain the onset conditioner and low-rank residual on train episodes only.

The full future wavefield is opened only for a train episode.  The resulting
checkpoint can be loaded at validation time, where the guarded adapter still
receives exactly the two onset snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.adapters import (
    ADAPTER_SCHEMA_VERSION,
    OnsetAdaptedV5,
)
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.losses import frame_relative_l2
from scripts.run_v5_instance_adaptation import _load_parent
from scripts.train_v5_onset_conditioner import build_onset_episodes


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_indices(length: int, onset: tuple[int, int], count: int) -> torch.Tensor:
    if length < 2:
        raise ValueError("time axis must contain at least two samples")
    count = max(4, min(int(count), length))
    grid = torch.linspace(0, length - 1, count, dtype=torch.long)
    values = sorted(set(grid.tolist()) | {int(onset[0]), int(onset[1])})
    return torch.tensor(values, dtype=torch.long)


def _predict_parent_at_times(model, normalizer, record, device, time_s: torch.Tensor) -> torch.Tensor:
    """Predict only the supervised times; full 401-step inference is unnecessary during meta-training."""
    velocity = record.velocity_mps.to(device).unsqueeze(0)
    source = record.source_parameters.to(device).unsqueeze(0)
    source_map = record.source_map.to(device).unsqueeze(0)
    prepared = model.prepare_sources(
        model.encode_medium(velocity, normalizer), source, source_map, normalizer,
        record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
    )
    with torch.no_grad():
        return model.predict_wavefield(
            prepared,
            time_s.to(device),
            x_m=record.x_m.to(device),
            z_m=record.z_m.to(device),
            time_block=1,
        )


def train_meta(
    config_path: str | Path,
    *,
    output: str | Path,
    device: str = "cuda",
    max_episodes: int = 12,
    epochs: int = 1,
    time_points: int = 48,
    learning_rate: float = 2.0e-4,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    source_h5 = Path(config["source_h5"]).expanduser()
    manifest = build_manifest(source_h5)
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    parent, normalizer = _load_parent(config, manifest, target_device)
    adapter = OnsetAdaptedV5(
        parent,
        latent_dim=int(config.get("latent_dim", 32)),
        lora_rank=int(config.get("lora_rank", 4)),
    ).to(target_device)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.adapter_parameters(), lr=float(learning_rate), weight_decay=1.0e-5)
    episodes = build_onset_episodes(manifest, split="train", seed=int(config.get("seed", 17)))
    episodes = episodes[: int(max_episodes)]
    dataset = GuardedOnsetDataset(source_h5, manifest, split="train")
    records_by_source = {record.source_index: index for index, record in enumerate(dataset.records.records)}
    history: list[dict[str, float | int]] = []
    cached_samples: list[dict[str, torch.Tensor | tuple[int, int]]] = []
    try:
        with h5py.File(source_h5, "r", swmr=True) as h5:
            for metadata in episodes:
                record = dataset[records_by_source[metadata.source_index]]
                velocity = record.velocity_mps.to(target_device).unsqueeze(0)
                source = record.source_parameters.to(target_device).unsqueeze(0)
                observed = record.observed_wavefield.to(target_device).unsqueeze(0)
                time_s = record.time_s.to(target_device)
                indices = _time_indices(len(time_s), record.observed_indices, time_points)
                truth = torch.as_tensor(
                    np.asarray(h5["wavefield"][metadata.source_index, indices.tolist()], dtype=np.float32),
                    device=target_device,
                ).unsqueeze(0)
                selected_time = time_s[indices.to(target_device)]
                selected_parent = _predict_parent_at_times(
                    parent, normalizer, record, target_device, selected_time
                )
                cached_samples.append({
                    "velocity": velocity,
                    "source": source,
                    "observed": observed,
                    "truth": truth,
                    "selected_parent": selected_parent,
                    "selected_time": selected_time,
                    "observed_indices": record.observed_indices,
                    "indices": indices,
                })
            for epoch in range(int(epochs)):
                for step, sample in enumerate(cached_samples):
                    velocity = sample["velocity"]
                    source = sample["source"]
                    observed = sample["observed"]
                    truth = sample["truth"]
                    selected_parent = sample["selected_parent"]
                    selected_time = sample["selected_time"]
                    observed_indices = sample["observed_indices"]
                    indices = sample["indices"]
                    prediction = adapter.raw_wavefield(
                        selected_parent, velocity, source, observed, selected_time
                    )
                    field_loss = frame_relative_l2(
                        prediction,
                        truth,
                        energy_floor_fraction=0.05,
                    )
                    onset_positions = [int((indices == value).nonzero(as_tuple=False)[0]) for value in observed_indices]
                    observed_pred = prediction[:, onset_positions]
                    observed_loss = frame_relative_l2(
                        observed_pred,
                        observed,
                        energy_floor_fraction=0.05,
                    )
                    target_energy = truth.square().mean().clamp_min(1.0e-12)
                    energy_loss = (prediction.square().mean() / target_energy - 1.0).square()
                    loss = field_loss + 0.25 * observed_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite meta-pretraining loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter.adapter_parameters(), 1.0)
                    optimizer.step()
                    history.append({
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.detach()),
                        "field_loss": float(field_loss.detach()),
                        "observed_loss": float(observed_loss.detach()),
                        "energy_loss": float(energy_loss.detach()),
                        "time_points": int(len(indices)),
                    })
    finally:
        dataset.close()
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite an onset residual meta checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parent_checkpoint = Path(str(config["parent_checkpoint"])).expanduser().resolve()
    torch.save(
        {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "conditioner_state": adapter.conditioner.state_dict(),
            "residual_state": adapter.residual.state_dict(),
            "parent_checkpoint": str(parent_checkpoint),
            "parent_checkpoint_sha256": _sha256(parent_checkpoint),
            "config_sha256": _sha256(config_path),
            "source_h5_sha256": _sha256(source_h5),
            "manifest_digest": manifest.digest,
            "future_truth_used_only_for_train_episode": True,
            "episodes": len(episodes),
            "epochs": int(epochs),
            "time_points": int(time_points),
            "history": history,
        },
        output_path,
    )
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time-points", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    args = parser.parse_args(argv)
    path = train_meta(
        args.config,
        output=args.output,
        device=args.device,
        max_episodes=args.max_episodes,
        epochs=args.epochs,
        time_points=args.time_points,
        learning_rate=args.learning_rate,
    )
    print(json.dumps({"checkpoint": str(path), "future_truth_opened": "train_only"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
