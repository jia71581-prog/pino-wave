#!/usr/bin/env python3
"""Train a small onset conditioner from train-split episodes only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.instance_adaptation.adapters import OnsetSnapshotConditioner
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_onset_episodes(manifest, *, split: str, seed: int = 17):
    records = [record for record in manifest.records if record.split == split]
    if not records:
        raise ValueError(f"split has no records: {split}")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(records), generator=generator).tolist()
    return tuple(records[index] for index in order)


def train_conditioner_smoke(
    config_path: str | Path,
    *,
    output: str | Path,
    device: str = "cpu",
    max_steps: int = 2,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text())
    source_h5 = Path(config["source_h5"]).expanduser()
    manifest = build_manifest(source_h5)
    model = OnsetSnapshotConditioner(
        latent_dim=int(config.get("latent_dim", 32)),
        width=int(config.get("width", 32)),
        lora_rank=int(config.get("lora_rank", 4)),
    ).to(device)
    head = torch.nn.Linear(model.latent_dim, 4).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-5)),
    )
    episodes = build_onset_episodes(manifest, split="train", seed=int(config.get("seed", 17)))
    train_dataset = GuardedOnsetDataset(source_h5, manifest, split="train")
    records_by_source = {record.source_index: index for index, record in enumerate(train_dataset.records.records)}
    losses: list[float] = []
    with h5py.File(source_h5, "r", swmr=True) as h5:
        for step, metadata in enumerate(episodes[: int(max_steps)]):
            record = train_dataset[records_by_source[metadata.source_index]]
            velocity = record.velocity_mps.to(device)
            source = record.source_parameters.to(device).unsqueeze(0)
            observed = record.observed_wavefield.to(device).unsqueeze(0)
            latent = model(velocity.unsqueeze(0), source, observed)
            future = torch.as_tensor(
                h5["wavefield"][metadata.source_index, record.observed_indices[1] + 1 :],
                dtype=torch.float32,
                device=device,
            )
            if future.numel() == 0:
                continue
            target = torch.stack((future.mean(), future.std(), future.amax(), future.amin())).view(1, 4)
            prediction = head(latent)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target / target.detach().abs().clamp_min(1.0e-8))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "conditioner_state": model.state_dict(),
            "summary_head_state": head.state_dict(),
            "source_h5_sha256": _sha256(source_h5),
            "manifest_digest": manifest.digest,
            "future_truth_used_only_for_train_episode": True,
            "losses": losses,
        },
        output_path,
    )
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/v5_instance_adaptation/conditioner.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    path = train_conditioner_smoke(
        args.config,
        output=args.output,
        device=args.device,
        max_steps=2 if args.smoke else 128,
    )
    print(json.dumps({"checkpoint": str(path), "future_truth_opened": "train_only"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
