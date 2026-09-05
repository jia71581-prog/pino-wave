#!/usr/bin/env python3
"""Render the audited best Helmholtz A+1 checkpoint on three selected records.

The script reconstructs the architecture exclusively from ``run_identity.json``,
loads the identity-bound best checkpoint, adds the exact cached smoothed-background
field used by the evaluator, and queries all 401 stored times.  It writes, per family:

* true / prediction / error wavefield snapshots;
* the true velocity model used by the wave solve;
* nine near-surface receiver waveform overlays;
* true / prediction / error receiver gathers;
* a compact NPZ containing only the plotted fields and traces.

This is intentionally separate from the older capacity-ladder plotting scripts: those
scripts do not reconstruct the Helmholtz synthesis branch or the A+1 background field.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider
from saved_time_phase_operator_v4.data import split_pilot_batch
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from saved_time_phase_operator_v4.losses import (
    apply_hard_causality,
    source_causality_onset_s,
)
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.diagnose_capacity_ladder_overfit import (
    build_base_config,
    build_probe_config,
)
from scripts.diagnose_helmholtz_g3_heldout import warmstart_full_helmholtz
from scripts.diagnose_saved_time_temporal_three_record_overfit import _dataset
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_probe import _model


FAMILIES = ("uniform", "layered", "marmousi")
TRUE_COLOR = "#0072B2"
PRED_COLOR = "#D55E00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    delta = prediction.astype(np.float64) - target.astype(np.float64)
    denominator = np.linalg.norm(target.astype(np.float64))
    return float(np.linalg.norm(delta) / max(float(denominator), 1.0e-30))


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _snapshot_indices(time_s: np.ndarray, source_t0_s: float, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("snapshot count must be at least two")
    start = min(max(source_t0_s + 0.05, float(time_s[0])), float(time_s[-1]))
    requested = np.linspace(start, float(time_s[-1]), int(count))
    indices = np.asarray(
        [int(np.argmin(np.abs(time_s - value))) for value in requested],
        dtype=np.int64,
    )
    indices = np.unique(indices)
    if len(indices) != count:
        indices = np.linspace(
            int(np.searchsorted(time_s, start)), len(time_s) - 1, count
        ).round().astype(np.int64)
    return indices


def _receiver_indices(height: int, width: int) -> tuple[int, np.ndarray]:
    # Match the project's numerical diagnostics: a near-surface line at z=100 m
    # on the 10 m saved grid, spread across the physical aperture.
    receiver_z = min(max(int(round(0.05 * (height - 1))), 0), height - 1)
    # Keep a common, well-illuminated near-surface aperture for all three
    # families.  The extreme left edge of the Marmousi held-out sample has
    # essentially zero trace energy, which makes a per-trace relative error
    # numerically meaningless even though the plotted amplitudes are tiny.
    receiver_x = np.linspace(0.15, 0.95, 9)
    receiver_x = np.rint(receiver_x * (width - 1)).astype(np.int64)
    return receiver_z, receiver_x


def _plot_wavefields(
    target: np.ndarray,
    prediction: np.ndarray,
    time_s: np.ndarray,
    snapshot_indices: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    source_x_m: float,
    source_z_m: float,
    family: str,
    output: Path,
) -> list[float]:
    _configure_style()
    selected_true = target[snapshot_indices]
    selected_pred = prediction[snapshot_indices]
    selected_error = selected_pred - selected_true
    extent = (x_m[0] / 1000.0, x_m[-1] / 1000.0, z_m[-1] / 1000.0, z_m[0] / 1000.0)
    frame_errors: list[float] = []
    fig, axes = plt.subplots(
        3,
        len(snapshot_indices),
        figsize=(2.35 * len(snapshot_indices), 6.7),
        constrained_layout=True,
    )
    row_labels = ("True pressure", "Predicted pressure", "Prediction error")
    for column, frame_index in enumerate(snapshot_indices):
        true = selected_true[column]
        pred = selected_pred[column]
        error = selected_error[column]
        field_limit = max(float(np.max(np.abs(true))), float(np.max(np.abs(pred))), 1.0e-30)
        error_limit = max(float(np.max(np.abs(error))), 1.0e-30)
        frame_rel = _relative_l2(pred, true)
        frame_errors.append(frame_rel)
        panels = ((true, field_limit), (pred, field_limit), (error, error_limit))
        for row, (panel, limit) in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(
                panel,
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                extent=extent,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            ax.scatter(
                [source_x_m / 1000.0],
                [source_z_m / 1000.0],
                marker="*",
                s=32,
                c="yellow",
                edgecolors="black",
                linewidths=0.35,
            )
            if row == 0:
                ax.set_title(f"t={time_s[frame_index]:.3f} s\nscale ±{field_limit:.1e}")
            elif row == 2:
                ax.set_title(f"relL2={frame_rel:.2%}\nscale ±{error_limit:.1e}")
            if column == 0:
                ax.set_ylabel(f"{row_labels[row]}\nz (km)")
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("x (km)")
            else:
                ax.set_xticklabels([])
    fig.suptitle(
        f"Best Phase4b Helmholtz A+1 — {family} wavefield snapshots",
        fontsize=13,
        y=1.025,
    )
    fig.savefig(output)
    plt.close(fig)
    return frame_errors


def _plot_velocity_model(
    velocity_mps: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    source_x_m: float,
    source_z_m: float,
    family: str,
    sample_id: str,
    group_id: str,
    output: Path,
    color_limits_mps: tuple[float, float] | None = None,
) -> None:
    _configure_style()
    velocity = np.asarray(velocity_mps, dtype=np.float32)
    if velocity.shape != (len(z_m), len(x_m)):
        raise ValueError("velocity model and coordinate axes disagree")
    extent = (
        x_m[0] / 1000.0,
        x_m[-1] / 1000.0,
        z_m[-1] / 1000.0,
        z_m[0] / 1000.0,
    )
    data_min = float(np.min(velocity))
    data_max = float(np.max(velocity))
    if color_limits_mps is None:
        vmin = data_min
        vmax = data_max
        if np.isclose(vmin, vmax):
            padding = max(abs(vmin) * 0.01, 1.0)
            vmin -= padding
            vmax += padding
    else:
        vmin, vmax = (float(value) for value in color_limits_mps)
        if not np.isfinite([vmin, vmax]).all() or not vmin < vmax:
            raise ValueError("velocity color limits must be finite and increasing")
        tolerance = max(abs(vmin), abs(vmax), 1.0) * 1.0e-6
        if data_min < vmin - tolerance or data_max > vmax + tolerance:
            raise ValueError(
                "velocity values fall outside the requested color limits: "
                f"data={data_min}..{data_max}, limits={vmin}..{vmax}"
            )
    fig, ax = plt.subplots(figsize=(6.4, 5.35), constrained_layout=True)
    panel = ax.imshow(
        velocity,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.scatter(
        [source_x_m / 1000.0],
        [source_z_m / 1000.0],
        marker="*",
        s=90,
        c="white",
        edgecolors="black",
        linewidths=0.65,
        label="source",
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(
        f"True {family} velocity model\n{sample_id}\n{group_id}",
        fontsize=11,
    )
    ax.legend(loc="upper right")
    fig.colorbar(panel, ax=ax, shrink=0.84, label="velocity (m/s)")
    fig.savefig(output)
    plt.close(fig)


def _plot_receiver_waveforms(
    target: np.ndarray,
    prediction: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    family: str,
    output: Path,
) -> tuple[int, np.ndarray, list[float]]:
    _configure_style()
    receiver_z, receiver_x = _receiver_indices(target.shape[-2], target.shape[-1])
    trace_errors: list[float] = []
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 7.2), sharex=True, constrained_layout=True)
    for ax, ix in zip(axes.flat, receiver_x, strict=True):
        true = target[:, receiver_z, ix]
        pred = prediction[:, receiver_z, ix]
        trace_rel = _relative_l2(pred, true)
        trace_errors.append(trace_rel)
        ax.plot(time_s, true, color=TRUE_COLOR, lw=1.05, label="true")
        ax.plot(time_s, pred, color=PRED_COLOR, lw=0.9, ls="--", label="prediction")
        ax.set_title(f"x={x_m[ix]/1000:.2f} km, z={z_m[receiver_z]/1000:.2f} km\nrelL2={trace_rel:.2%}")
        ax.grid(alpha=0.25, lw=0.45)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0, 0].legend(loc="upper right")
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("pressure")
    fig.suptitle(
        f"Best Phase4b Helmholtz A+1 — {family} near-surface receiver waveforms",
        fontsize=13,
        y=1.025,
    )
    fig.savefig(output)
    plt.close(fig)
    return receiver_z, receiver_x, trace_errors


def _plot_receiver_gather(
    target: np.ndarray,
    prediction: np.ndarray,
    time_s: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    receiver_z: int,
    source_x_m: float,
    family: str,
    output: Path,
) -> None:
    _configure_style()
    true = target[:, receiver_z, :].T
    pred = prediction[:, receiver_z, :].T
    error = pred - true
    field_limit = max(float(np.max(np.abs(true))), float(np.max(np.abs(pred))), 1.0e-30)
    error_limit = max(float(np.max(np.abs(error))), 1.0e-30)
    extent = (float(time_s[0]), float(time_s[-1]), x_m[-1] / 1000.0, x_m[0] / 1000.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    panels = (
        (true, field_limit, "True gather"),
        (pred, field_limit, "Predicted gather"),
        (error, error_limit, "Prediction error"),
    )
    for ax, (panel, limit, title) in zip(axes, panels, strict=True):
        image = ax.imshow(
            panel,
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
            extent=extent,
            aspect="auto",
            interpolation="nearest",
        )
        ax.axhline(source_x_m / 1000.0, color="lime", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("receiver x (km)")
        fig.colorbar(image, ax=ax, shrink=0.84, label="pressure")
    fig.suptitle(
        f"Best Phase4b Helmholtz A+1 — {family} receiver gather at z={z_m[receiver_z]:.0f} m",
        fontsize=13,
        y=1.02,
    )
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument(
        "--travel-time-h5",
        type=Path,
        default=Path("/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snapshot-count", type=int, default=6)
    parser.add_argument("--time-block", type=int, default=1)
    parser.add_argument(
        "--background-cache-shard",
        type=Path,
        action="append",
        default=[],
        help="additional provider-readable P_bg cache; repeatable",
    )
    parser.add_argument(
        "--uniform-sample-id",
        default=None,
        help="optional manifest-backed Uniform sample used for visualization",
    )
    parser.add_argument(
        "--layered-sample-id",
        default=None,
        help="optional manifest-backed Layered sample used for visualization",
    )
    parser.add_argument(
        "--marmousi-sample-id",
        default=None,
        help="optional manifest-backed Marmousi sample used for visualization",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    identity = json.loads((run_dir / "run_identity.json").read_text())
    terminal = json.loads((run_dir / "terminal.json").read_text())
    if terminal.get("status") != "complete":
        raise ValueError("the selected Helmholtz run is not complete")
    checkpoint = Path(str(terminal["best_checkpoint"])).expanduser().resolve()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("manifest_digest") != identity["manifest_digest"]:
        raise ValueError("checkpoint and run identity manifest digests disagree")
    if checkpoint_payload.get("config_digest") != identity["run_digest"]:
        raise ValueError("checkpoint and run identity config digests disagree")
    checkpoint_metric = float(
        checkpoint_payload.get("metrics", {}).get("triplet_aggregate_relative_l2", float("nan"))
    )
    if not np.isclose(
        checkpoint_metric,
        float(terminal["best_fixed_heldout_relative_l2"]),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("checkpoint metric does not reproduce the terminal best metric")
    checkpoint_global_step = int(checkpoint_payload.get("global_step", -1))
    del checkpoint_payload

    device = torch.device(args.device)
    torch.manual_seed(int(identity["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(identity["seed"]))

    base = build_base_config(int(identity["width"]))
    base = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            normalization_json=str(args.normalization_json.expanduser().resolve()),
        ),
    )
    manifest = build_manifest(base.data.source_h5)
    if manifest.digest != identity["manifest_digest"]:
        raise ValueError("current dataset manifest does not match the completed run")
    normalizer = load_normalizer(base, manifest.digest)

    variant = ProbeVariant(
        depth=int(identity["dense_depth"]),
        use_local_phase=True,
        spectral_rank=int(identity["dense_spectral_rank"]),
        modes=int(identity["dense_modes"]),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(identity["helmholtz_frequencies"]),
        local_field_helmholtz_synthesis_wkb_phase=bool(identity["helmholtz_wkb_phase"]),
        local_field_helmholtz_synthesis_rank=int(identity["helmholtz_rank"]),
        local_field_helmholtz_synthesis_late_rank=int(identity.get("helmholtz_late_rank", 0)),
        local_field_helmholtz_synthesis_late_frequencies=int(
            identity.get("helmholtz_late_frequencies", 0)
        ),
        local_field_helmholtz_spectral_bypass=bool(
            identity.get("helmholtz_spectral_bypass", False)
        ),
        local_field_helmholtz_spectral_bypass_per_branch=bool(
            identity.get("helmholtz_spectral_bypass_per_branch", False)
        ),
        local_field_helmholtz_background_conditioning=bool(
            identity.get("helmholtz_background_conditioning", False)
        ),
        local_field_helmholtz_background_sigma_cells=float(
            identity.get("helmholtz_background_sigma_cells", 2.0)
        ),
        local_field_helmholtz_background_global_propagator=bool(
            identity.get("helmholtz_background_global_propagator", False)
        ),
        local_field_helmholtz_background_propagation_modes=int(
            identity.get("helmholtz_background_propagation_modes", 48)
        ),
    )
    model = _model(base, manifest, variant).to(device)
    transfer = warmstart_full_helmholtz(model, checkpoint)
    if int(transfer["transferred_tensor_count"]) != len(model.state_dict()):
        raise ValueError("best checkpoint did not populate every model tensor")
    model.eval()

    config = build_probe_config(
        dense_lr=float(identity["dense_learning_rate"]),
        backbone_lr=float(identity["backbone_learning_rate"]),
        temporal_lr=1.0e-5,
        seed=int(identity["seed"]),
        travel_time_h5=str(args.travel_time_h5.expanduser().resolve()),
        family_gradient_weights=identity.get("family_gradient_weights"),
        per_frame_frame=bool(identity.get("per_frame_frame", False)),
        frame_energy_floor_fraction=float(identity.get("frame_energy_floor_fraction", 0.0)),
        late_frame_gain=float(identity.get("late_frame_gain", 0.0)),
        late_frame_start_fraction=float(identity.get("late_frame_start_fraction", 0.4)),
    )
    background_paths = (
        identity["background_cache"],
        *identity.get("background_cache_shards", []),
        *(str(path.expanduser().resolve()) for path in args.background_cache_shard),
    )
    background_provider = BackgroundFieldProvider(background_paths)
    heldout_indices = tuple(int(value) for value in identity["heldout_indices"])
    heldout_ids = tuple(str(value) for value in identity["heldout_sample_ids"])
    records_by_split: dict[str, tuple[object, ...]] = {
        split: tuple(record for record in manifest.records if str(record.split) == split)
        for split in sorted({str(record.split) for record in manifest.records})
    }
    requested_ids = dict(zip(FAMILIES, heldout_ids, strict=True))
    explicit_ids = {
        "uniform": args.uniform_sample_id,
        "layered": args.layered_sample_id,
        "marmousi": args.marmousi_sample_id,
    }
    for family, sample_id in explicit_ids.items():
        if sample_id is not None:
            requested_ids[family] = str(sample_id)
    selection_basis = {
        "ood_uniform_c4000_f15": "canonical OOD uniform reference at 4000 m/s",
        "ood_layered_c3000_c5000_f15": (
            "canonical OOD physical layering: upper 3000 m/s, lower 5000 m/s"
        ),
        "ood_marmousi_holdout_f10": (
            "canonical split-isolated Marmousi holdout crop from the registered source"
        ),
    }
    selections: list[dict[str, object]] = []
    visualization_overrides: dict[str, dict[str, object]] = {}
    for slot, family in enumerate(FAMILIES):
        sample_id = requested_ids[family]
        matches = [
            record for record in manifest.records if str(record.sample_id) == sample_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"visualization sample must resolve exactly once: {sample_id!r}"
            )
        record = matches[0]
        if str(record.medium_type) != family:
            raise ValueError(
                f"visualization sample {sample_id!r} does not belong to {family}"
            )
        split = str(record.split)
        split_records = records_by_split[split]
        record_index = next(
            index
            for index, candidate in enumerate(split_records)
            if str(candidate.sample_id) == sample_id
        )
        selections.append(
            {
                "family": family,
                "sample_id": sample_id,
                "split": split,
                "record_index": int(record_index),
                "metadata": record,
            }
        )
        if sample_id != heldout_ids[slot]:
            visualization_overrides[family] = {
                "terminal_sample_id": heldout_ids[slot],
                "visualization_sample_id": sample_id,
                "split": split,
                "record_index_in_split": int(record_index),
                "source_index": int(record.source_index),
                "group_id": str(record.group_id),
                "selection_basis": selection_basis.get(
                    sample_id, "explicit manifest-backed visualization selection"
                ),
            }
        elif split != "validation" or record_index != heldout_indices[slot]:
            raise ValueError("identity-bound held-out record index no longer resolves")
    selected_ids_tuple = tuple(str(value["sample_id"]) for value in selections)
    if not background_provider.covers(
        selected_ids_tuple, range(len(manifest.time_s))
    ):
        raise ValueError("background cache set does not cover all selected 401-frame records")

    checkpoint_hash = _sha256(checkpoint)
    report: dict[str, object] = {
        "status": "complete",
        "protocol": (
            "phase4b_heldout_validation_triplet_all_401_stored_times"
            if not visualization_overrides
            else (
                "phase4b_best_checkpoint_canonical_ood_triplet_all_401_stored_times"
                if all(str(value["split"]) == "ood_canonical" for value in selections)
                else "phase4b_best_checkpoint_manifest_visualization_all_401_stored_times"
            )
        ),
        "run_dir": str(run_dir),
        "run_digest": identity["run_digest"],
        "manifest_digest": manifest.digest,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_global_step": checkpoint_global_step,
        "additional_background_caches": [
            {
                "path": str(path.expanduser().resolve()),
                "sha256": _sha256(path.expanduser().resolve()),
            }
            for path in args.background_cache_shard
        ],
        "terminal_fixed_aggregate_relative_l2": float(
            terminal["best_fixed_heldout_relative_l2"]
        ),
        "terminal_all_saved_aggregate_relative_l2": float(
            terminal["heldout_all_saved_aggregate_relative_l2"]
        ),
        "aggregate_metric_is_terminal_comparable": not visualization_overrides,
        "visualization_sample_overrides": visualization_overrides,
        "families": {},
    }

    for selection in selections:
        family = str(selection["family"])
        record_index = int(selection["record_index"])
        expected_id = str(selection["sample_id"])
        record_split = str(selection["split"])
        record_metadata = selection["metadata"]
        schedule = (
            FullSupportStepSpec(
                step=990_000,
                epoch=0,
                record_indices=(record_index,),
                appearance_indices=(0,),
            ),
        )
        dataset = _dataset(
            config,
            base,
            manifest,
            (record_index,),
            split=record_split,
            schedule=schedule,
            time_policy="all_saved",
            frames_per_record=len(manifest.time_s),
        )
        if len(dataset) != 1:
            raise RuntimeError("expected exactly one visualization batch per family")
        batch = dataset[0]
        micros = tuple(split_pilot_batch(batch, microbatch_records=1))
        if len(micros) != 1:
            raise RuntimeError("expected one visualization microbatch per family")
        micro = micros[0]
        if tuple(micro.sample_id) != (expected_id,):
            raise ValueError(
                f"selected identity mismatch for {family}: {micro.sample_id} != {expected_id}"
            )
        tensors = _to_device(micro, device)
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
                if micro.dense_travel_time_s is None
                else micro.dense_travel_time_s.to(device, non_blocking=True)
            ),
        )
        prediction_norm, background_reference_norm = model.dense_normalized_with_coarse(
            prepared,
            tensors["requested_time_s"],
            dense_grid=dense_grid,
            time_block=int(args.time_block),
        )
        background_physical = background_provider.physical(
            micro.sample_id,
            micro.left_index,
            device=device,
            dtype=tensors["dense_target_physical"].dtype,
        )
        background_norm = normalizer.encode_pressure(background_physical, source[:, 4])
        prediction_norm = prediction_norm + background_norm
        background_reference_norm = background_reference_norm + background_norm
        if bool(config["loss"].get("hard_causality", False)):
            onset = source_causality_onset_s(
                source,
                lead_cycles=float(config["loss"].get("hard_causality_lead_cycles", 0.0)),
            )
            prediction_norm = apply_hard_causality(
                prediction_norm, tensors["requested_time_s"], onset
            )
            background_reference_norm = apply_hard_causality(
                background_reference_norm, tensors["requested_time_s"], onset
            )
        prediction = (
            normalizer.decode_pressure(prediction_norm.float(), source[:, 4])
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        background_reference = (
            normalizer.decode_pressure(background_reference_norm.float(), source[:, 4])
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        target = (
            tensors["dense_target_physical"]
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        time_s = (
            tensors["requested_time_s"].squeeze(0).cpu().numpy().astype(np.float32)
        )
        order = np.argsort(time_s)
        time_s = time_s[order]
        target = target[order]
        prediction = prediction[order]
        background_reference = background_reference[order]
        x_m = tensors["x_m"].cpu().numpy().astype(np.float32)
        z_m = tensors["z_m"].cpu().numpy().astype(np.float32)
        source_values = source.squeeze(0).cpu().numpy().astype(np.float32)
        velocity_mps = (
            tensors["velocity_mps"]
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        source_x_m, source_z_m = float(source_values[0]), float(source_values[1])
        snap_idx = _snapshot_indices(
            time_s, float(source_values[3]), int(args.snapshot_count)
        )

        family_dir = output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        velocity_path = family_dir / f"{family}_true_velocity_model.png"
        wavefield_path = family_dir / f"{family}_wavefield_snapshots.png"
        receiver_path = family_dir / f"{family}_receiver_waveforms.png"
        gather_path = family_dir / f"{family}_receiver_gather.png"
        _plot_velocity_model(
            velocity_mps,
            x_m,
            z_m,
            source_x_m,
            source_z_m,
            family,
            expected_id,
            str(record_metadata.group_id),
            velocity_path,
        )
        frame_errors = _plot_wavefields(
            target,
            prediction,
            time_s,
            snap_idx,
            x_m,
            z_m,
            source_x_m,
            source_z_m,
            family,
            wavefield_path,
        )
        receiver_z, receiver_x, trace_errors = _plot_receiver_waveforms(
            target,
            prediction,
            time_s,
            x_m,
            z_m,
            family,
            receiver_path,
        )
        _plot_receiver_gather(
            target,
            prediction,
            time_s,
            x_m,
            z_m,
            receiver_z,
            source_x_m,
            family,
            gather_path,
        )
        np.savez_compressed(
            family_dir / f"{family}_plotted_data.npz",
            time_s=time_s,
            snapshot_indices=snap_idx,
            snapshot_target=target[snap_idx],
            snapshot_prediction=prediction[snap_idx],
            receiver_x_indices=receiver_x,
            receiver_z_index=np.asarray(receiver_z),
            receiver_target=target[:, receiver_z, receiver_x],
            receiver_prediction=prediction[:, receiver_z, receiver_x],
            x_m=x_m,
            z_m=z_m,
            velocity_mps=velocity_mps,
            source_parameters=source_values,
        )
        full_rel = _relative_l2(prediction, target)
        background_rel = _relative_l2(background_reference, target)
        correction_ratio = float(
            np.linalg.norm((prediction - background_reference).astype(np.float64))
            / max(np.linalg.norm(background_reference.astype(np.float64)), 1.0e-30)
        )
        report["families"][family] = {
            "sample_id": expected_id,
            "split": record_split,
            "record_index_in_split": record_index,
            "source_index": int(record_metadata.source_index),
            "group_id": str(record_metadata.group_id),
            "source_parameters": [float(value) for value in source_values],
            "full_wavefield_relative_l2": full_rel,
            "background_only_relative_l2": background_rel,
            "neural_correction_to_background_l2_ratio": correction_ratio,
            "snapshot_indices": [int(value) for value in snap_idx],
            "snapshot_times_s": [float(value) for value in time_s[snap_idx]],
            "snapshot_frame_relative_l2": frame_errors,
            "receiver_z_m": float(z_m[receiver_z]),
            "receiver_x_m": [float(x_m[value]) for value in receiver_x],
            "receiver_trace_relative_l2": trace_errors,
            "receiver_mean_relative_l2": float(np.mean(trace_errors)),
            "outputs": {
                "true_velocity_model": str(velocity_path),
                "wavefield_snapshots": str(wavefield_path),
                "receiver_waveforms": str(receiver_path),
                "receiver_gather": str(gather_path),
                "plotted_data": str(family_dir / f"{family}_plotted_data.npz"),
            },
        }
        print(
            json.dumps(
                {
                    "event": "family_complete",
                    "family": family,
                    "sample_id": expected_id,
                    "full_relative_l2": full_rel,
                    "receiver_mean_relative_l2": float(np.mean(trace_errors)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del tensors, prepared, dense_grid, prediction_norm, background_reference_norm
        del background_physical, background_norm
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reproduced = {
        family: float(report["families"][family]["full_wavefield_relative_l2"])
        for family in FAMILIES
    }
    expected = terminal["heldout_all_saved_metrics"]["family_relative_l2"]
    reproduced_terminal_families: list[str] = []
    for family in FAMILIES:
        if selected_ids_tuple[FAMILIES.index(family)] != heldout_ids[FAMILIES.index(family)]:
            continue
        observed = reproduced[family]
        if not np.isclose(observed, float(expected[family]), rtol=2.0e-5, atol=2.0e-7):
            raise RuntimeError(
                f"rendered {family} metric does not reproduce terminal: "
                f"{observed} != {expected[family]}"
            )
        reproduced_terminal_families.append(family)
    report["terminal_metrics_reproduced_for_families"] = reproduced_terminal_families
    report["selected_triplet_aggregate_relative_l2"] = float(
        np.mean([reproduced[family] for family in FAMILIES])
    )
    report_path = output_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    background_provider.close()
    print(json.dumps({"event": "complete", "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
