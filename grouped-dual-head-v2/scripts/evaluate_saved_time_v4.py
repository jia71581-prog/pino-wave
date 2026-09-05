#!/usr/bin/env python
"""Run one identity-bound evaluation on all 480 held-out stored-time records."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.data.pilot import PilotStepSpec
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
from saved_time_phase_operator_v4.evaluation import (
    sha256_file,
    time_axis_sha256,
    validate_evaluation_identity,
)
from saved_time_phase_operator_v4.metrics import exact_wavefield_metrics
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_probe import _atomic_json, _model


def _census(manifest) -> dict[str, object]:
    validation = [record for record in manifest.records if record.split == "validation"]
    return {
        "validation": len(validation),
        "families": dict(sorted(Counter(record.medium_type for record in validation).items())),
        "sample_ids_sha256": __import__("hashlib").sha256(
            "\n".join(record.sample_id for record in validation).encode("utf8")
        ).hexdigest(),
    }


def _validation_schedule(record_count: int, records_per_batch: int) -> tuple[PilotStepSpec, ...]:
    if record_count % records_per_batch:
        raise ValueError("sealed record census must divide into complete batches")
    return tuple(
        PilotStepSpec(step=50_000 + start // records_per_batch, record_indices=tuple(range(start, start + records_per_batch)))
        for start in range(0, record_count, records_per_batch)
    )


def _plot_examples(examples: Mapping[str, Mapping[str, object]], output: Path) -> list[str]:
    paths: list[str] = []
    for family, example in examples.items():
        target = torch.as_tensor(example["target"])
        prediction = torch.as_tensor(example["prediction"])
        times = torch.as_tensor(example["time_s"])
        scale = float(torch.maximum(target.abs().max(), prediction.abs().max()).clamp_min(1.0e-8))
        figure, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
        for index in range(4):
            error = prediction[index] - target[index]
            axes[0, index].imshow(target[index], cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[1, index].imshow(prediction[index], cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[2, index].imshow(error, cmap="RdBu_r", vmin=-scale, vmax=scale)
            axes[0, index].set_title(f"t={float(times[index]):.4f} s")
            for row in range(3):
                axes[row, index].set_xticks([]); axes[row, index].set_yticks([])
        for row, label in enumerate(("truth", "prediction", "error")):
            axes[row, 0].set_ylabel(label)
        path = output / f"{family}_stored_time_wavefields.png"
        figure.suptitle(f"{family}: exact stored-time full fields")
        figure.savefig(path, dpi=160); plt.close(figure); paths.append(str(path))
    return paths


@torch.inference_mode()
def _evaluate(model, dataset, normalizer, device):
    model.eval(); predictions=[]; targets=[]; families=[]; examples={}
    for index in range(len(dataset)):
        batch = dataset[index]
        if not bool(batch.target_exact.all()) or not torch.equal(batch.left_index, batch.right_index):
            raise RuntimeError("sealed V4 evaluation encountered an interpolated target")
        tensors = _to_device(batch, device); source = tensors["source_parameters"]
        prepared = model.prepare_sources(
            model.encode_medium(tensors["velocity_mps"], normalizer), source,
            tensors["source_map"], normalizer,
            record_to_medium=tensors["record_to_medium"],
        )
        prediction = model.dense_normalized(
            prepared, tensors["requested_time_s"], x_m=tensors["x_m"],
            z_m=tensors["z_m"], time_block=1,
        )
        target = normalizer.encode_pressure(tensors["dense_target_physical"], source[:, 4])
        predictions.append(prediction.cpu()); targets.append(target.cpu()); families.extend(batch.medium_type)
        for offset, family in enumerate(batch.medium_type):
            if family not in examples:
                examples[family] = {
                    "prediction": normalizer.decode_pressure(prediction[offset].cpu(), source[offset, 4].cpu()),
                    "target": batch.dense_target_physical[offset],
                    "time_s": batch.requested_time_s[offset],
                    "sample_id": batch.sample_id[offset],
                }
    return torch.cat(predictions), torch.cat(targets), tuple(families), examples


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--stored-times-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not args.stored_times_only:
        raise ValueError("V4 sealed evaluation requires --stored-times-only")
    summary = json.loads((args.artifact_dir / "probe_summary.json").read_text())
    selected = summary.get("selected")
    output = args.artifact_dir / "sealed_evaluation"; output.mkdir(parents=True, exist_ok=True)
    if not selected:
        _atomic_json({"status": "selection_gate_failed", "selected": None}, output / "evaluation_report.json")
        return 2
    variant_root = args.artifact_dir / "variants" / str(selected)
    run_identity = json.loads((variant_root / "run_identity.json").read_text())
    checkpoint = Path(summary["results"][selected]["checkpoint"])
    config = run_identity["config"]
    base = V3Config.from_yaml(config["base_config"])
    manifest = build_manifest(base.data.source_h5)
    validate_expected_counts(manifest, {"train": base.data.expected_train_records, "validation": base.data.expected_validation_records})
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    binding = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_digest": manifest.digest,
        "time_axis_sha256": time_axis_sha256(manifest.time_s),
        "record_census": _census(manifest),
        "model_config_digest": checkpoint_payload.get("config_digest"),
    }
    identity_path = output / "evaluation_identity.json"
    if identity_path.exists():
        validate_evaluation_identity(json.loads(identity_path.read_text()), binding)
    else:
        _atomic_json(binding, identity_path)
    report_path = output / "evaluation_report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("status") == "complete":
            print(json.dumps(existing, sort_keys=True)); return 0
    variant = ProbeVariant(**run_identity["variant_config"])
    device = torch.device(args.device)
    model = _model(base, manifest, variant).to(device)
    load_checkpoint(
        checkpoint, model=model, expected_manifest_digest=manifest.digest,
        expected_config_digest=str(binding["model_config_digest"]), map_location=device,
    )
    normalizer = load_normalizer(base, manifest.digest)
    record_count = int(binding["record_census"]["validation"])
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5, manifest, split="validation",
        schedule=_validation_schedule(record_count, 6), query_points=1, seed=int(config["seed"]) + 31_337,
    )
    prediction, target, families, examples = _evaluate(model, dataset, normalizer, device)
    metrics = exact_wavefield_metrics(
        prediction, target, families=families,
        energy_floor_fraction=float(config["energy_floor_fraction"]),
    )
    if metrics["record_count"] != record_count:
        raise RuntimeError("sealed evaluation did not cover the bound validation census")
    figures = _plot_examples(examples, output)
    report = {
        "status": "complete", "selected": selected, "stored_times_only": True,
        "interpolated_targets": 0, "binding": binding, "metrics": metrics,
        "representatives": {family: value["sample_id"] for family, value in examples.items()},
        "figures": figures,
        "passes_ten_percent_gate": metrics["aggregate_floored_relative_l2"] < 0.10
        and all(value < 0.10 for value in metrics["family_floored_relative_l2"].values()),
    }
    _atomic_json(report, report_path); print(json.dumps(report, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
