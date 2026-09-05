#!/usr/bin/env python3
"""Render the bound r34 train-only accuracy and wavefield comparison.

The script regenerates the deployed r5b plus residual-conditioned instance
fine-tuning output and the frozen full-data PI-DeepONet output on the first
preregistered record from each medium family.  It verifies complete-future
errors against the immutable r34 result before saving only the t=0.60 s
snapshots used by the manuscript figure.  Validation and test_id are never
opened.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from grouped_ufno_mionet_v3.data.index import build_manifest
from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from patch_deeponet_baseline.training import dense_training_pair
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.instance_adaptation.data_guard import (
    GuardedOnsetDataset,
)
from scripts.evaluate_v5_feature_meta import (
    _load_adapter,
    _predict_feature,
    _refine_residual_trust_gate,
)
from scripts.run_v5_instance_adaptation import _load_parent, _predict_parent
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context


R34_SHA256 = "22f9a22cd105995415ec66dcbb15a86bf3dc9c12108f1f1790a67c4667fc2a8b"
R33_SHA256 = "ea42d7370ac42533c58ae7cf86e554c7a0ead50f0e69cf0300565dbca01b0f97"
ADAPTER_SHA256 = "77b8866f8aaba20d66bf5c2e0d9c0a1b93d39ca95dda1edf62795183cea42162"
FEATURE_CONFIG_SHA256 = "c1f7682fe866f4695cced1059f2bad442904c4b702e4c0c18d5a12f138d758fb"
PI_CONFIG_SHA256 = "012f47073dc5b20a182b6caf6394487110a5714dc114cc2f1750a2b6bd7f8efc"
PI_CHECKPOINT_SHA256 = "e2d6d7a9ec5481268bae2f41cff2400ee78b93a3a9216b2c338db572d405c4c1"
R5B_IDENTITY_SHA256 = "a8141ab179e7cc18331c0878743d5ecaec106f0ff727e7a20ec198d126c03449"
SELECTED_SAMPLE_IDS = (
    "train_uniform_00146",
    "train_layered_00109",
    "train_marmousi_00107",
)
SNAPSHOT_INDEX = 240
REFINEMENT_STEPS = 12
REFINEMENT_LEARNING_RATE = 0.05


@dataclass(frozen=True)
class SnapshotRow:
    sample_id: str
    family: str
    source_index: int
    observed_indices: tuple[int, int]
    refinement_accepted: bool
    modulation_gate: float
    future_ours_relative_l2: float
    future_pi_relative_l2: float
    snapshot_ours_relative_l2: float
    snapshot_pi_relative_l2: float
    regenerated_adapted_sha256: str
    registered_adapted_sha256: str
    adapted_sha256_match: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _schedule(indices: tuple[int, ...]) -> tuple[FullSupportStepSpec, ...]:
    return tuple(
        FullSupportStepSpec(
            step=1_340_000 + offset,
            epoch=0,
            record_indices=(int(index),),
            appearance_indices=(0,),
        )
        for offset, index in enumerate(indices)
    )


def _relative(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    prediction64 = torch.as_tensor(prediction).double()
    truth64 = torch.as_tensor(truth).double()
    numerator = float((prediction64 - truth64).square().sum())
    denominator = float(truth64.square().sum())
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def _verify_digest(path: Path, expected: str, label: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} digest changed: {observed}")


def _render(
    snapshots: dict[str, dict[str, np.ndarray]],
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]],
    rows: list[SnapshotRow],
    output_base: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    figure = plt.figure(figsize=(7.16, 4.45), constrained_layout=True)
    grid = figure.add_gridspec(
        3,
        7,
        width_ratios=(1.0, 1.0, 1.0, 0.055, 1.0, 1.0, 0.055),
        wspace=0.05,
        hspace=0.11,
    )
    title_by_column = {
        0: "Reference",
        1: "DFO+RCFA",
        2: "PI-DeepONet",
        4: "DFO+RCFA abs. error",
        5: "PI abs. error",
    }
    family_label = {"uniform": "Uniform", "layered": "Layered", "marmousi": "Marmousi"}
    axes: list[list[plt.Axes]] = []
    for row_index, row in enumerate(rows):
        sample = snapshots[row.sample_id]
        truth = sample["reference"]
        ours = sample["ours"]
        pi_field = sample["pi_deeponet"]
        ours_error = np.abs(ours - truth)
        pi_error = np.abs(pi_field - truth)
        pressure_values = np.concatenate(
            [np.abs(truth).ravel(), np.abs(ours).ravel(), np.abs(pi_field).ravel()]
        )
        error_values = np.concatenate([ours_error.ravel(), pi_error.ravel()])
        pressure_limit = max(float(np.quantile(pressure_values, 0.997)), 1.0e-30)
        error_limit = max(float(np.quantile(error_values, 0.997)), 1.0e-30)
        x_m, z_m = coordinates[row.sample_id]
        extent = (
            float(x_m[0] / 1000.0),
            float(x_m[-1] / 1000.0),
            float(z_m[-1] / 1000.0),
            float(z_m[0] / 1000.0),
        )
        pressure_norm = Normalize(vmin=-pressure_limit, vmax=pressure_limit)
        error_norm = Normalize(vmin=0.0, vmax=error_limit)
        row_axes: list[plt.Axes] = []
        for column, field, cmap, norm in (
            (0, truth, "RdBu_r", pressure_norm),
            (1, ours, "RdBu_r", pressure_norm),
            (2, pi_field, "RdBu_r", pressure_norm),
            (4, ours_error, "magma", error_norm),
            (5, pi_error, "magma", error_norm),
        ):
            axis = figure.add_subplot(grid[row_index, column])
            image = axis.imshow(
                field,
                cmap=cmap,
                norm=norm,
                extent=extent,
                interpolation="nearest",
                aspect="equal",
            )
            if row_index == 0:
                axis.set_title(title_by_column[column], fontsize=7.2, pad=3.0)
            if column == 0:
                letter = chr(ord("a") + row_index)
                axis.set_ylabel(
                    f"({letter}) {family_label[row.family]}\nz (km)",
                    fontsize=7.0,
                )
            else:
                axis.set_yticklabels([])
            if row_index == len(rows) - 1:
                axis.set_xlabel("x (km)", fontsize=7.0, labelpad=1.5)
            else:
                axis.set_xticklabels([])
            axis.tick_params(labelsize=6.4, length=2.0, pad=1.2)
            row_axes.append(axis)
        pressure_color_axis = figure.add_subplot(grid[row_index, 3])
        pressure_colorbar = figure.colorbar(
            row_axes[2].images[0], cax=pressure_color_axis, orientation="vertical"
        )
        pressure_colorbar.set_label("p (Pa)", fontsize=6.2, labelpad=1.5)
        pressure_colorbar.ax.tick_params(labelsize=5.5, length=1.7, pad=1.0)
        pressure_colorbar.formatter.set_powerlimits((-2, 2))
        pressure_colorbar.update_ticks()
        error_color_axis = figure.add_subplot(grid[row_index, 6])
        error_colorbar = figure.colorbar(
            row_axes[4].images[0], cax=error_color_axis, orientation="vertical"
        )
        error_colorbar.set_label(r"$|\Delta p|$ (Pa)", fontsize=6.2, labelpad=1.5)
        error_colorbar.ax.tick_params(labelsize=5.5, length=1.7, pad=1.0)
        error_colorbar.formatter.set_powerlimits((-2, 2))
        error_colorbar.update_ticks()
        axes.append(row_axes)

    figure.suptitle(
        "Matched wavefield snapshots at t = 0.60 s (train-only development records)",
        fontsize=8.0,
        y=1.01,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"))
    figure.savefig(output_base.with_suffix(".png"), dpi=450)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    for path, expected, label in (
        (args.r34_result, R34_SHA256, "r34 result"),
        (args.r33_summary, R33_SHA256, "r33 summary"),
        (args.adapter_checkpoint, ADAPTER_SHA256, "feature adapter checkpoint"),
        (args.feature_config, FEATURE_CONFIG_SHA256, "feature config"),
        (args.pi_config, PI_CONFIG_SHA256, "PI-DeepONet config"),
        (args.pi_checkpoint, PI_CHECKPOINT_SHA256, "PI-DeepONet checkpoint"),
        (args.r5b_identity, R5B_IDENTITY_SHA256, "r5b identity"),
    ):
        _verify_digest(path, expected, label)

    r34 = json.loads(args.r34_result.read_text())
    r33 = json.loads(args.r33_summary.read_text())
    if r34.get("role") != "development_evidence_not_validation_or_test_evidence":
        raise ValueError("r34 role changed")
    if r33.get("selection_split") != "train" or not r33.get(
        "future_truth_opened_only_after_seal"
    ):
        raise ValueError("r33 is not the sealed train-only result")
    r34_by_id = {str(row["sample_id"]): row for row in r34["records"]}
    r33_by_id = {str(row["sample_id"]): row for row in r33["records"]}
    if any(sample_id not in r34_by_id or sample_id not in r33_by_id for sample_id in SELECTED_SAMPLE_IDS):
        raise ValueError("registered figure samples changed")

    feature_config = yaml.safe_load(args.feature_config.read_text())
    feature_manifest = build_manifest(feature_config["source_h5"])
    device = torch.device(args.device)
    parent, feature_normalizer = _load_parent(feature_config, feature_manifest, device)
    adapter, adapter_payload = _load_adapter(
        parent,
        args.adapter_checkpoint,
        device=device,
        expected_manifest_digest=feature_manifest.digest,
    )
    if adapter_payload.get("conditioner_input") != "parent_onset_residual":
        raise ValueError("adapter conditioner input changed")
    ours_dataset = GuardedOnsetDataset(
        feature_config["source_h5"],
        feature_manifest,
        split="train",
        sample_ids=SELECTED_SAMPLE_IDS,
        travel_time_h5=feature_config["travel_time_h5"],
    )

    r5b_identity = json.loads(args.r5b_identity.read_text())
    base, pi_manifest, _ = _load_context(dict(r5b_identity["config"]))
    if pi_manifest.digest != feature_manifest.digest:
        raise ValueError("r5b and PI-DeepONet manifests differ")
    train_rows = tuple(row for row in pi_manifest.records if row.split == "train")
    train_index = {row.sample_id: index for index, row in enumerate(train_rows)}
    selected_indices = tuple(train_index[value] for value in SELECTED_SAMPLE_IDS)
    pi_config = yaml.safe_load(args.pi_config.read_text())
    pi_dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        pi_manifest,
        split="train",
        schedule=_schedule(selected_indices),
        query_points=1,
        seed=int(pi_config["seed"]),
        time_policy="all_saved",
        frames_per_record=len(pi_manifest.time_s),
        travel_time_h5=pi_config["travel_time_h5"],
    )
    pi_normalizer = load_normalizer(base, pi_manifest.digest)
    pi_model = PatchDeepONet(
        PatchDeepONetConfig(**dict(pi_config.get("model", {})))
    ).to(device)
    checkpoint = torch.load(args.pi_checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("manifest_digest") != pi_manifest.digest:
        raise ValueError("PI-DeepONet checkpoint manifest changed")
    pi_model.load_state_dict(checkpoint["model_state"], strict=True)
    pi_model.eval()
    execution = dict(pi_config["model_execution"])
    lead_cycles = float(pi_config["loss"]["hard_causality_lead_cycles"])

    rows: list[SnapshotRow] = []
    snapshots: dict[str, dict[str, np.ndarray]] = {}
    coordinates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    started = time.perf_counter()
    for position, pi_batch in enumerate(pi_dataset):
        record = ours_dataset[position]
        sample_id = SELECTED_SAMPLE_IDS[position]
        if record.sample_id != sample_id or str(pi_batch.sample_id[0]) != sample_id:
            raise RuntimeError("paired figure order changed")

        parent_field = _predict_parent(parent, feature_normalizer, record, device)
        onset = list(record.observed_indices)
        conditioner = (
            record.observed_wavefield.to(device).unsqueeze(0) - parent_field[:, onset]
        ).detach()
        latent_delta, modulation_gate, refinement = _refine_residual_trust_gate(
            adapter,
            feature_normalizer,
            record,
            device,
            steps=REFINEMENT_STEPS,
            learning_rate=REFINEMENT_LEARNING_RATE,
            conditioner_wavefield=conditioner,
        )
        with torch.no_grad():
            ours_field = _predict_feature(
                adapter,
                feature_normalizer,
                record,
                device,
                time_s=record.time_s.to(device),
                latent_delta=latent_delta,
                conditioner_wavefield=conditioner,
                modulation_gate=modulation_gate,
            )
        regenerated_hash = _tensor_sha256(ours_field)
        registered_hash = str(r33_by_id[sample_id]["sealed_prediction_sha256"]["adapted"])

        with torch.no_grad():
            pi_prediction_n, _, time_indices = dense_training_pair(
                pi_model,
                pi_batch,
                pi_normalizer,
                device,
                time_block=int(execution["time_block"]),
                query_chunk=int(execution["query_chunk"]),
                hard_causality_lead_cycles=lead_cycles,
            )
            expected_indices = torch.arange(len(pi_manifest.time_s), dtype=torch.long)
            if not torch.equal(time_indices[0].cpu(), expected_indices):
                raise RuntimeError("PI figure did not materialize every stored time")
            tensors = _to_device(pi_batch, device)
            pi_field = pi_normalizer.decode_pressure(
                pi_prediction_n, tensors["source_parameters"][:, 4]
            )
            truth = tensors["dense_target_physical"]

        future = slice(int(record.observed_indices[1]) + 1, len(pi_manifest.time_s))
        ours_future_error = _relative(ours_field[:, future], truth[:, future])
        pi_future_error = _relative(pi_field[:, future], truth[:, future])
        expected = r34_by_id[sample_id]
        if not math.isclose(
            ours_future_error,
            float(expected["our_instance_adapted_relative_l2"]),
            rel_tol=2.0e-6,
            abs_tol=2.0e-8,
        ):
            raise RuntimeError(f"ours r34 error mismatch for {sample_id}")
        if not math.isclose(
            pi_future_error,
            float(expected["pi_deeponet_relative_l2"]),
            rel_tol=2.0e-6,
            abs_tol=2.0e-8,
        ):
            raise RuntimeError(f"PI r34 error mismatch for {sample_id}")
        time_value = float(record.time_s[SNAPSHOT_INDEX])
        if not math.isclose(time_value, 0.60, rel_tol=0.0, abs_tol=1.0e-7):
            raise RuntimeError(f"snapshot time changed: {time_value}")

        truth_frame = truth[0, SNAPSHOT_INDEX].detach().cpu()
        ours_frame = ours_field[0, SNAPSHOT_INDEX].detach().cpu()
        pi_frame = pi_field[0, SNAPSHOT_INDEX].detach().cpu()
        snapshots[sample_id] = {
            "reference": truth_frame.numpy().astype(np.float32, copy=False),
            "ours": ours_frame.numpy().astype(np.float32, copy=False),
            "pi_deeponet": pi_frame.numpy().astype(np.float32, copy=False),
        }
        coordinates[sample_id] = (
            record.x_m.detach().cpu().numpy().astype(np.float32, copy=False),
            record.z_m.detach().cpu().numpy().astype(np.float32, copy=False),
        )
        rows.append(
            SnapshotRow(
                sample_id=sample_id,
                family=str(record.medium_type),
                source_index=int(record.source_index),
                observed_indices=tuple(int(value) for value in record.observed_indices),
                refinement_accepted=bool(refinement["accepted"]),
                modulation_gate=float(refinement["modulation_gate"]),
                future_ours_relative_l2=ours_future_error,
                future_pi_relative_l2=pi_future_error,
                snapshot_ours_relative_l2=_relative(ours_frame, truth_frame),
                snapshot_pi_relative_l2=_relative(pi_frame, truth_frame),
                regenerated_adapted_sha256=regenerated_hash,
                registered_adapted_sha256=registered_hash,
                adapted_sha256_match=regenerated_hash == registered_hash,
            )
        )
        print(
            f"[{position + 1}/3] {sample_id} "
            f"ours={ours_future_error:.6f} pi={pi_future_error:.6f} "
            f"accepted={bool(refinement['accepted'])}",
            flush=True,
        )
        del parent_field, conditioner, ours_field, pi_prediction_n, pi_field, truth, tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

    source_arrays: dict[str, np.ndarray] = {
        "snapshot_index": np.asarray([SNAPSHOT_INDEX], dtype=np.int64),
        "snapshot_time_s": np.asarray([0.60], dtype=np.float64),
    }
    for row in rows:
        prefix = row.family
        for name, array in snapshots[row.sample_id].items():
            source_arrays[f"{prefix}_{name}"] = array
        source_arrays[f"{prefix}_x_m"] = coordinates[row.sample_id][0]
        source_arrays[f"{prefix}_z_m"] = coordinates[row.sample_id][1]
    args.source_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.source_data, **source_arrays)
    _render(snapshots, coordinates, rows, args.figure_base)

    report: dict[str, object] = {
        "schema": "r34_method_vs_pi_wavefield_figure_v1",
        "status": "complete_train_only_figure",
        "role": "development_evidence_not_validation_or_test_evidence",
        "figure_claim": (
            "On the first preregistered record from each family, the proposed "
            "r5b plus instance-fine-tuning policy has lower complete-future "
            "relative L2 than PI-DeepONet, and the t=0.60 s snapshots provide "
            "a qualitative same-case illustration."
        ),
        "selection": {
            "split": "train",
            "rule": "first preregistered r33 record in each family",
            "sample_ids": list(SELECTED_SAMPLE_IDS),
            "snapshot_index": SNAPSHOT_INDEX,
            "snapshot_time_s": 0.60,
            "validation_opened": False,
            "test_id_opened": False,
        },
        "method_identity": {
            "ours": "frozen_r5b_plus_parent_residual_conditioned_instance_finetuning",
            "pi_deeponet": "full_training_set_pi_deeponet_epoch_96",
            "physical_wavefield_propagator": False,
            "our_pde_loss_weight": 0.0,
            "pi_physics_loss_weight": 0.02,
            "our_shared_adapter_parameter_count": 40_848,
            "our_instance_trainable_parameter_count": 17,
        },
        "rows": [asdict(row) for row in rows],
        "r34_metrics": r34["metrics"],
        "r34_comparison": r34["comparison"],
        "artifacts": {
            "figure_pdf": str(args.figure_base.with_suffix(".pdf").resolve()),
            "figure_pdf_sha256": _sha256(args.figure_base.with_suffix(".pdf")),
            "figure_png": str(args.figure_base.with_suffix(".png").resolve()),
            "figure_png_sha256": _sha256(args.figure_base.with_suffix(".png")),
            "source_data": str(args.source_data.resolve()),
            "source_data_sha256": _sha256(args.source_data),
        },
        "bindings": {
            "r34_result_sha256": R34_SHA256,
            "r33_summary_sha256": R33_SHA256,
            "adapter_checkpoint_sha256": ADAPTER_SHA256,
            "feature_config_sha256": FEATURE_CONFIG_SHA256,
            "r5b_identity_sha256": R5B_IDENTITY_SHA256,
            "pi_config_sha256": PI_CONFIG_SHA256,
            "pi_checkpoint_sha256": PI_CHECKPOINT_SHA256,
            "script_sha256": _sha256(Path(__file__)),
            "manifest_digest": feature_manifest.digest,
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    _atomic_json(report, args.report)
    return report


def main(argv=None) -> int:
    evidence = (
        ROOT
        / "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813"
        / "15_pi_deeponet_train_development_comparison"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r34-result",
        type=Path,
        default=ROOT / "results/r5b_feature_meta_vs_pi_train_all401_r34_result_20260816.json",
    )
    parser.add_argument(
        "--r33-summary",
        type=Path,
        default=Path(
            "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
            "pretraining/instance_adaptation/"
            "r5b_feature_meta_parent_residual_disjoint_gate_r33_20260816/summary.json"
        ),
    )
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
            "pretraining/instance_adaptation/"
            "r5b_feature_meta_parent_residual_train_r32_20260816/best.pt"
        ),
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=ROOT / "configs/saved_time_v5/feature_meta_r5b_parent_residual_trust_no_propagator.yaml",
    )
    parser.add_argument(
        "--r5b-identity",
        type=Path,
        default=Path(
            "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
            "pretraining/local_field_w128_hicap/"
            "temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag/run/run_identity.json"
        ),
    )
    parser.add_argument(
        "--pi-config",
        type=Path,
        default=ROOT / "configs/baselines/pi_deeponet_lwc84_long100e_r7_authorized_20260814.yaml",
    )
    parser.add_argument(
        "--pi-checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
            "baselines/pi_deeponet_lwc84_long100e_r7/"
            "full_long100e_r1_resume1/best.pt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--figure-base",
        type=Path,
        default=ROOT / "paper/tgrs_helmholtz_operator/figs/method_vs_pi_train_wavefield_snapshot",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        default=evidence / "method_vs_pi_train_wavefield_snapshot_source.npz",
    )
    parser.add_argument("--report", type=Path, default=evidence / "report.json")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
