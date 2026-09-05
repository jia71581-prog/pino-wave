#!/usr/bin/env python3
"""Train-only, same-record exact-32 audit of r5b and PI-DeepONet.

The selected records exclude the fixed 48-record r5b epoch gate. Both models
consume the same ``ExactStoredTimeBatchDataset`` batch, so targets and saved-time
indices are identical. This is a diagnostic, not held-out validation evidence:
all selected records still belong to the pretraining split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from grouped_ufno_mionet_v3.data.index import ALLOWED_MEDIUM_TYPES
from grouped_ufno_mionet_v3.training.checkpoint import load_checkpoint
from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from patch_deeponet_baseline.training import dense_training_pair
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    source_causality_onset_s,
)
from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_full_support import (
    _load_context,
    _load_parent_model,
    validation_panel_indices,
)
from scripts.train_v5_feature_meta import build_balanced_meta_episodes


FAMILIES = tuple(ALLOWED_MEDIUM_TYPES)
R5B_CHECKPOINT_SHA256 = (
    "9295d21f2e0858dd5d2bb2ab2c10a0a39e062148a796e7ab71eec016a2ae9a3b"
)
PI_CHECKPOINT_SHA256 = (
    "e2d6d7a9ec5481268bae2f41cff2400ee78b93a3a9216b2c338db572d405c4c1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _terms(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    predicted = prediction.double()
    reference = target.to(prediction.device).double()
    return (
        float((predicted - reference).square().sum()),
        float(reference.square().sum()),
    )


def _add(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return float(left[0] + right[0]), float(left[1] + right[1])


def _relative(values: tuple[float, float] | list[float]) -> float:
    return float(math.sqrt(float(values[0]) / max(float(values[1]), 1.0e-30)))


def _schedule(indices: tuple[int, ...]) -> tuple[FullSupportStepSpec, ...]:
    return tuple(
        FullSupportStepSpec(
            step=1_100_000 + int(index),
            epoch=0,
            record_indices=(int(index),),
            appearance_indices=(0,),
        )
        for index in indices
    )


def _r5b_prediction(model, batch, normalizer, device, config):
    tensors = _to_device(batch, device)
    source = tensors["source_parameters"]
    prepared = model.prepare_sources(
        model.encode_medium(tensors["velocity_mps"], normalizer),
        source,
        tensors["source_map"],
        normalizer,
        record_to_medium=tensors["record_to_medium"],
    )
    dense_grid = model.prepare_dense_grid(
        prepared,
        x_m=tensors["x_m"],
        z_m=tensors["z_m"],
        travel_time_s=(
            None
            if batch.dense_travel_time_s is None
            else batch.dense_travel_time_s.to(
                device, non_blocking=device.type == "cuda"
            )
        ),
    )
    prediction, _ = model.dense_normalized_with_coarse(
        prepared,
        tensors["requested_time_s"],
        dense_grid=dense_grid,
        time_block=1,
    )
    target = normalizer.encode_pressure(
        tensors["dense_target_physical"], source[:, 4]
    )
    if bool(config["loss"].get("hard_causality", False)):
        onset = source_causality_onset_s(
            source,
            lead_cycles=float(
                config["loss"].get("hard_causality_lead_cycles", 0.0)
            ),
        )
        prediction = apply_hard_causality(
            prediction, tensors["requested_time_s"], onset
        )
    return prediction, target, batch.left_index.to(device)


def _model_summary(
    accumulator: ExactWavefieldMetricAccumulator,
    total: tuple[float, float],
    family_terms: dict[str, tuple[float, float]],
    record_terms: dict[str, tuple[float, float]],
) -> dict[str, object]:
    streamed = accumulator.finalize()
    return {
        "global_energy_relative_l2": _relative(total),
        "family_global_energy_relative_l2": {
            family: _relative(family_terms[family]) for family in FAMILIES
        },
        "mean_record_relative_l2": float(
            np.mean([_relative(values) for values in record_terms.values()])
        ),
        "family_mean_record_relative_l2": streamed["family_relative_l2"],
        "time_bin_relative_l2": streamed["time_bin_relative_l2"],
        "family_time_bin_relative_l2": streamed[
            "family_time_bin_relative_l2"
        ],
        "spectrum_relative_l2": streamed["spectrum_relative_l2"],
        "phase_correlation": streamed["phase_correlation"],
        "centroid_shift_cells": streamed["centroid_shift_cells"],
        "xcorr_peak_shift_cells": streamed["xcorr_peak_shift_cells"],
        "near_zero_frame_count": streamed["near_zero_frame_count"],
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if args.per_family <= 0:
        raise ValueError("per-family record count must be positive")
    if args.frames_per_record != 32:
        raise ValueError("this audit is registered for exactly 32 frames")
    if _sha256(args.r5b_checkpoint) != R5B_CHECKPOINT_SHA256:
        raise ValueError("r5b checkpoint binding changed")
    if _sha256(args.pi_checkpoint) != PI_CHECKPOINT_SHA256:
        raise ValueError("PI-DeepONet checkpoint binding changed")

    r5b_identity = json.loads(args.r5b_identity.read_text(encoding="utf-8"))
    r5b_config = dict(r5b_identity["config"])
    base, manifest, parent_identity = _load_context(r5b_config)
    train_rows = tuple(record for record in manifest.records if record.split == "train")
    if len(train_rows) != int(base.data.expected_train_records):
        raise RuntimeError("train census changed")

    fixed_gate_indices = validation_panel_indices(
        validation_records=len(train_rows),
        panel_records=int(r5b_config["validation"]["panel_records"]),
        epoch=1,
        seed=int(r5b_config["seed"]),
    )
    fixed_gate_ids = {train_rows[index].sample_id for index in fixed_gate_indices}
    allowed = {
        record.sample_id for record in train_rows if record.sample_id not in fixed_gate_ids
    }
    selected_rows = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=int(args.per_family),
        seed=int(args.selection_seed),
        allowed_sample_ids=allowed,
    )
    selected_ids = {record.sample_id for record in selected_rows}
    if selected_ids.intersection(fixed_gate_ids):
        raise RuntimeError("audit selection overlaps the fixed r5b epoch gate")
    index_by_id = {record.sample_id: index for index, record in enumerate(train_rows)}
    selected_indices = tuple(index_by_id[record.sample_id] for record in selected_rows)

    pi_config = yaml.safe_load(args.pi_config.read_text(encoding="utf-8"))
    if pi_config.get("schema") != "pi_deeponet_training_config_v1":
        raise ValueError("unexpected PI-DeepONet training config")
    if int(pi_config["seed"]) != int(r5b_config["seed"]):
        raise ValueError("same-frame audit requires matching time-selector seeds")
    if Path(pi_config["base_config"]).name != Path(r5b_config["base_config"]).name:
        raise ValueError("r5b and PI base configs are not the same registered dataset")

    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split="train",
        schedule=_schedule(selected_indices),
        query_points=1,
        seed=int(pi_config["seed"]),
        time_policy="validation_fixed",
        frames_per_record=int(args.frames_per_record),
        travel_time_h5=pi_config["travel_time_h5"],
    )
    normalizer = load_normalizer(base, manifest.digest)
    device = torch.device(args.device)

    r5b = _load_parent_model(
        r5b_config, base, manifest, parent_identity, device
    )
    r5b_metadata = load_checkpoint(
        args.r5b_checkpoint,
        model=r5b,
        expected_manifest_digest=manifest.digest,
        expected_config_digest=str(r5b_identity["run_digest"]),
        map_location=device,
    )
    r5b.eval()
    pi = PatchDeepONet(
        PatchDeepONetConfig(**dict(pi_config.get("model", {})))
    ).to(device)
    pi_payload = torch.load(args.pi_checkpoint, map_location=device, weights_only=True)
    if pi_payload.get("schema") != "patch_deeponet_checkpoint_v1":
        raise ValueError("unexpected PI-DeepONet checkpoint schema")
    if pi_payload.get("manifest_digest") != manifest.digest:
        raise ValueError("PI-DeepONet manifest binding changed")
    pi.load_state_dict(pi_payload["model_state"], strict=True)
    pi.eval()

    accumulators = {
        name: ExactWavefieldMetricAccumulator(
            energy_floor_fraction=0.01,
            require_unique=True,
            stored_time_count=len(manifest.time_s),
        )
        for name in ("r5b", "pi_deeponet")
    }
    totals = {name: (0.0, 0.0) for name in accumulators}
    family_terms = {
        name: {family: (0.0, 0.0) for family in FAMILIES}
        for name in accumulators
    }
    record_terms: dict[str, dict[str, tuple[float, float]]] = {
        name: {} for name in accumulators
    }
    measurements: list[dict[str, object]] = []
    progress = args.output.with_name(f"{args.output.stem}.progress.json")
    started = time.perf_counter()
    execution = dict(pi_config["model_execution"])
    lead_cycles = float(pi_config["loss"]["hard_causality_lead_cycles"])
    for position, batch in enumerate(dataset, start=1):
        if len(batch.sample_id) != 1:
            raise RuntimeError("audit dataset must yield one record at a time")
        sample_id = str(batch.sample_id[0])
        expected_row = selected_rows[position - 1]
        if sample_id != expected_row.sample_id:
            raise RuntimeError("audit dataset order changed")
        family = str(batch.medium_type[0])
        if family != expected_row.medium_type:
            raise RuntimeError("audit family label changed")

        r5b_prediction, r5b_target, r5b_indices = _r5b_prediction(
            r5b, batch, normalizer, device, r5b_config
        )
        pi_prediction, pi_target, pi_indices = dense_training_pair(
            pi,
            batch,
            normalizer,
            device,
            time_block=int(execution["time_block"]),
            query_chunk=int(execution["query_chunk"]),
            hard_causality_lead_cycles=lead_cycles,
        )
        if not torch.equal(r5b_indices.cpu(), pi_indices.cpu()):
            raise RuntimeError("r5b and PI saved-time indices differ")
        if not torch.equal(r5b_target.cpu(), pi_target.cpu()):
            maximum = float((r5b_target - pi_target).abs().max())
            raise RuntimeError(f"r5b and PI targets differ: {maximum}")
        time_indices = [int(value) for value in pi_indices[0].cpu().tolist()]
        if len(time_indices) != 32 or len(set(time_indices)) != 32:
            raise RuntimeError("exact-32 time selection changed")

        onset_index = int(
            np.searchsorted(manifest.time_s, float(batch.source_parameters[0, 3]))
        )
        row_measurement = {
            "sample_id": sample_id,
            "source_index": int(expected_row.source_index),
            "family": family,
            "source_f0_hz": float(batch.source_parameters[0, 2]),
            "time_indices": time_indices,
        }
        for name, prediction in (
            ("r5b", r5b_prediction),
            ("pi_deeponet", pi_prediction),
        ):
            values = _terms(prediction, pi_target)
            totals[name] = _add(totals[name], values)
            family_terms[name][family] = _add(family_terms[name][family], values)
            record_terms[name][sample_id] = values
            accumulators[name].update(
                prediction,
                pi_target,
                families=(family,),
                group_ids=(str(batch.group_id[0]),),
                sample_ids=(sample_id,),
                time_indices=pi_indices,
                source_onset_indices=(onset_index,),
            )
            row_measurement[f"{name}_error_numerator"] = values[0]
            row_measurement[f"{name}_relative_l2"] = _relative(values)
        row_measurement["truth_denominator"] = float(_terms(pi_target, pi_target)[1])
        row_measurement["r5b_over_pi_error_ratio"] = (
            float(row_measurement["r5b_relative_l2"])
            / max(float(row_measurement["pi_deeponet_relative_l2"]), 1.0e-30)
        )
        measurements.append(row_measurement)
        _atomic_json(
            {
                "schema": "r5b_pi_train_exact32_progress_v1",
                "status": "running",
                "completed_records": position,
                "total_records": len(selected_rows),
                "elapsed_s": float(time.perf_counter() - started),
                "last_measurement": row_measurement,
            },
            progress,
        )
        print(
            f"[{position}/{len(selected_rows)}] {sample_id} "
            f"r5b={row_measurement['r5b_relative_l2']:.6g} "
            f"pi={row_measurement['pi_deeponet_relative_l2']:.6g}",
            flush=True,
        )

    summaries = {
        name: _model_summary(
            accumulators[name], totals[name], family_terms[name], record_terms[name]
        )
        for name in accumulators
    }
    r5b_global = float(summaries["r5b"]["global_energy_relative_l2"])
    pi_global = float(summaries["pi_deeponet"]["global_energy_relative_l2"])
    output = {
        "schema": "r5b_vs_pi_train_exact32_audit_v1",
        "status": "complete",
        "role": "train_only_diagnostic_not_validation_evidence",
        "selection": {
            "split": "train",
            "per_family": int(args.per_family),
            "record_count": len(selected_rows),
            "selection_seed": int(args.selection_seed),
            "frames_per_record": 32,
            "time_policy": "validation_fixed",
            "time_selector_seed": int(pi_config["seed"]),
            "excluded_r5b_fixed_gate_record_count": len(fixed_gate_ids),
            "overlap_with_r5b_fixed_gate": 0,
            "records_are_unseen_by_pretraining": False,
            "records_are_fresh_to_r5b_epoch_gate": True,
            "validation_truth_opened": False,
            "test_id_truth_opened": False,
        },
        "metrics": summaries,
        "comparison": {
            "r5b_minus_pi_global_energy_relative_l2": r5b_global - pi_global,
            "r5b_over_pi_global_error_ratio": r5b_global / max(pi_global, 1.0e-30),
            "r5b_better_than_pi_on_this_train_panel": r5b_global < pi_global,
        },
        "measurements": measurements,
        "bindings": {
            "manifest_digest": manifest.digest,
            "r5b_checkpoint": str(args.r5b_checkpoint.resolve()),
            "r5b_checkpoint_sha256": R5B_CHECKPOINT_SHA256,
            "r5b_checkpoint_epoch": int(r5b_metadata.epoch),
            "r5b_checkpoint_global_step": int(r5b_metadata.global_step),
            "r5b_identity": str(args.r5b_identity.resolve()),
            "r5b_identity_sha256": _sha256(args.r5b_identity),
            "pi_checkpoint": str(args.pi_checkpoint.resolve()),
            "pi_checkpoint_sha256": PI_CHECKPOINT_SHA256,
            "pi_checkpoint_epoch": int(pi_payload["epoch"]),
            "pi_checkpoint_global_step": int(pi_payload["global_step"]),
            "pi_config": str(args.pi_config.resolve()),
            "pi_config_sha256": _sha256(args.pi_config),
            "script_sha256": _sha256(Path(__file__)),
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    _atomic_json(output, args.output)
    _atomic_json(
        {
            "schema": "r5b_pi_train_exact32_progress_v1",
            "status": "complete",
            "completed_records": len(selected_rows),
            "total_records": len(selected_rows),
            "elapsed_s": output["elapsed_s"],
            "terminal_output": str(args.output.resolve()),
        },
        progress,
    )
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5b-identity", type=Path, required=True)
    parser.add_argument("--r5b-checkpoint", type=Path, required=True)
    parser.add_argument("--pi-config", type=Path, required=True)
    parser.add_argument("--pi-checkpoint", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=62315)
    parser.add_argument("--frames-per-record", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = run(args)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
