from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import DomainConfig, ModelConfig
from ..data.dataset import ContinuousWaveDataset
from ..model import ContinuousWaveOperator


def main() -> None:
    parser = argparse.ArgumentParser(description="Query arbitrary receiver coordinates and times")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--receivers-npy", required=True, type=Path, help="[R,2] physical x,z metres")
    parser.add_argument("--times-npy", required=True, type=Path, help="[T] seconds")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_values = dict(raw["model"])
    model_values["spectral_modes"] = tuple(model_values["spectral_modes"])
    model = ContinuousWaveOperator(ModelConfig(**model_values), DomainConfig(**raw["domain"]))
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(args.device).eval()
    dataset = ContinuousWaveDataset(raw["dataset"]["path"], split="test_id")
    handle = dataset._h5()
    global_index = int(dataset.indices[args.sample_index])
    velocity = torch.from_numpy(np.asarray(handle["velocity_mps"][global_index], dtype=np.float32))[None, None]
    source_map = torch.from_numpy(np.asarray(handle["source_map"][global_index], dtype=np.float32))[None, None, None]
    params = torch.tensor([[[_value for _value in (
        float(handle["source_x_m"][global_index]), float(handle["source_z_m"][global_index]),
        float(handle["source_f0_hz"][global_index]), float(handle["source_t0_s"][global_index]),
        float(handle["source_amplitude"][global_index]),
    )]]], dtype=torch.float32)
    receivers = np.load(args.receivers_npy)
    times = np.load(args.times_npy)
    receiver_grid = np.broadcast_to(receivers[:, None, :], (receivers.shape[0], times.size, 2))
    time_grid = np.broadcast_to(times[None, :, None], (receivers.shape[0], times.size, 1))
    query = torch.from_numpy(np.concatenate((receiver_grid, time_grid), axis=-1).reshape(1, 1, -1, 3).astype(np.float32))
    with torch.no_grad():
        pressure = model(
            velocity.to(args.device), source_map.to(args.device), params.to(args.device),
            query.to(args.device), chunk_size=4096,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, pressure.cpu().numpy().reshape(receivers.shape[0], times.size))


if __name__ == "__main__":
    main()
