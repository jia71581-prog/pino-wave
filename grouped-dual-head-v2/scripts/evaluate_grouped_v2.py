#!/usr/bin/env python
"""Evaluate V2 structured fields and receiver traces without misleading shared scales."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v2.config import V2Config
from grouped_ufno_mionet_v2.data.batch import pack_v2_groups
from grouped_ufno_mionet_v2.data.cache import StructuredCacheDataset
from grouped_ufno_mionet_v2.model.operator import DualHeadWaveOperator
from grouped_ufno_mionet_v2.normalization import PhysicalNormalizer
from grouped_ufno_mionet_v2.training.checkpoint import load_state


def _relative_l2(prediction, target):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    return float(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), np.finfo(np.float64).tiny))


def _atomic_json(payload, path):
    path = Path(path); temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8"); os.replace(temporary, path)


def _predict(model, normalizer, record, device):
    batch = pack_v2_groups([record]); values = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in vars(batch).items()}
    source = values["source_parameters"]; amplitude = source[:, 4]
    shared = dict(velocity_mps=values["velocity_mps"], source=source, source_map=values["source_map"],
                  normalizer=normalizer, record_to_medium=values["record_to_medium"])
    dense_times = values["time_s"][values["dense_time_indices"]]
    block = model.dense_decoder.time_block
    with torch.inference_mode():
        dense_normalized = torch.cat([model.dense_normalized(time_s=dense_times[:, i:i + block], **shared)
                                      for i in range(0, dense_times.shape[1], block)], 1)
        query_normalized = model.query_normalized(coords=values["query_coords"], chunk_size=4096, **shared)
        indices = values["receiver_zx_indices"]; _, receivers, _ = indices.shape; nt = values["time_s"].numel()
        nz, nx = values["source_map"].shape[-2:]
        coords = torch.empty(1, receivers, nt, 3, device=device)
        coords[..., 0] = indices[..., 1, None] * model.domain_x_m / (nx - 1)
        coords[..., 1] = indices[..., 0, None] * model.domain_z_m / (nz - 1)
        coords[..., 2] = values["time_s"][None, None]
        receiver_normalized = model.query_normalized(coords=coords.reshape(1, -1, 3), chunk_size=4096, **shared).reshape(1, receivers, nt)
    return {
        "dense_prediction": normalizer.decode_pressure(dense_normalized.float(), amplitude[:, None, None, None]).cpu().numpy()[0],
        "dense_target": values["dense_target"].cpu().numpy()[0],
        "query_prediction": normalizer.decode_pressure(query_normalized.float(), amplitude[:, None]).cpu().numpy()[0],
        "query_target": values["query_target"].cpu().numpy()[0],
        "receiver_prediction": normalizer.decode_pressure(receiver_normalized.float(), amplitude[:, None, None]).cpu().numpy()[0],
        "receiver_target": values["receiver_target"].cpu().numpy()[0],
        "dense_times": dense_times.cpu().numpy()[0], "time_s": values["time_s"].cpu().numpy(),
    }


def _figures(values, output, sample_id):
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    selected = np.linspace(0, len(values["dense_times"]) - 1, 3, dtype=int)
    figure, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    for row, index in enumerate(selected):
        true = values["dense_target"][index]; pred = values["dense_prediction"][index]; error = pred - true
        for column, (array, title, cmap) in enumerate(((true, "true", "RdBu_r"), (pred, "prediction", "RdBu_r"), (error, "error", "coolwarm"))):
            vmax = max(float(np.max(np.abs(array))), np.finfo(np.float32).tiny)
            image = axes[row, column].imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax, origin="upper")
            axes[row, column].set_title(f"t={values['dense_times'][index]:.4f}s {title}\nscale=±{vmax:.2e}")
            figure.colorbar(image, ax=axes[row, column], shrink=.72)
    figure.savefig(output / f"{sample_id}_wavefield.png", dpi=170); plt.close(figure)
    true, pred = values["receiver_target"], values["receiver_prediction"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for axis, array, title in ((axes[0, 0], true, "true receiver gather"), (axes[0, 1], pred, "predicted receiver gather")):
        vmax = max(float(np.max(np.abs(array))), np.finfo(np.float32).tiny)
        image = axis.imshow(array, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                            extent=(values["time_s"][0], values["time_s"][-1], len(array), 0))
        axis.set_title(f"{title}; scale=±{vmax:.2e}"); axis.set_xlabel("time (s)"); figure.colorbar(image, ax=axis)
    for receiver, axis in enumerate(axes[1]):
        index = min(receiver * max(len(true) - 1, 1), len(true) - 1)
        axis.plot(values["time_s"], true[index], label="true", lw=1.2)
        axis.plot(values["time_s"], pred[index], label="prediction", lw=1.0)
        axis.set_title(f"receiver {index}"); axis.set_xlabel("time (s)"); axis.legend()
    figure.savefig(output / f"{sample_id}_receivers.png", dpi=170); plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache"); parser.add_argument("--split", default="validation")
    parser.add_argument("--output", required=True); parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv); config = V2Config.from_yaml(args.config); device = torch.device(args.device)
    cache = args.cache or config.data.validation_cache
    dataset = StructuredCacheDataset(cache, expected_split=args.split)
    state = load_state(args.checkpoint, map_location=device)
    normalizer = PhysicalNormalizer.from_dict(state["normalizer"])
    model = DualHeadWaveOperator(width=config.model.width, rank=config.model.rank, modes=config.model.modes,
                                 heads=config.model.heads, dense_time_block=config.model.dense_time_block).to(device)
    model.load_state_dict(state["model"]); model.eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); reports = []
    indices = np.linspace(0, len(dataset) - 1, min(args.records, len(dataset)), dtype=int)
    for index in indices:
        record = dataset[int(index)]; values = _predict(model, normalizer, record, device)
        report = {"sample_id": record.sample_id, "medium_type": record.medium_type,
                  "dense_relative_l2": _relative_l2(values["dense_prediction"], values["dense_target"]),
                  "query_relative_l2": _relative_l2(values["query_prediction"], values["query_target"]),
                  "receiver_relative_l2": _relative_l2(values["receiver_prediction"], values["receiver_target"]),
                  "zero_baseline_relative_l2": 1.0}
        reports.append(report)
        np.savez_compressed(output / f"{record.sample_id}_predictions.npz", **values)
        _figures(values, output, record.sample_id)
        print(json.dumps(report), flush=True)
    aggregate = {key: float(np.mean([row[key] for row in reports])) for key in ("dense_relative_l2", "query_relative_l2", "receiver_relative_l2")}
    _atomic_json({"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()), "records": reports,
                  "aggregate": aggregate}, output / "evaluation_report.json")


if __name__ == "__main__":
    main()
