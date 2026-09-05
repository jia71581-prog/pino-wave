#!/usr/bin/env python3
"""Evaluate Phase4b on the sealed rank-15 fixed-19-Hz position panel.

Prediction uses only the locked velocity/source inputs.  All eight predictions
are sealed before the pre-existing LWC-84 reference wavefields are opened.
The output figure combines the true velocity model with reference, Phase4b,
and signed-error snapshots for the eight registered source positions.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from fno_acoustic.data_generation.config import (  # noqa: E402
    boundaries_from_config,
    grid_from_config,
    load_config,
    resolve_config,
)
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver  # noqa: E402
from grouped_ufno_mionet_v3.data.index import build_manifest  # noqa: E402
from grouped_ufno_mionet_v3.normalization import PhysicalNormalizer  # noqa: E402
from saved_time_phase_operator_v4.evaluation import sha256_file, time_axis_sha256  # noqa: E402
from saved_time_phase_operator_v4.hybrid_travel import straight_ray_grid_numpy  # noqa: E402
from saved_time_phase_operator_v4.losses import (  # noqa: E402
    apply_hard_causality,
    source_causality_onset_s,
)
from saved_time_phase_operator_v4.probe import ProbeVariant  # noqa: E402
from scripts.diagnose_capacity_ladder_overfit import (  # noqa: E402
    build_base_config,
    build_probe_config,
)
from scripts.diagnose_helmholtz_g3_heldout import warmstart_full_helmholtz  # noqa: E402
from scripts.train_saved_time_v4_probe import _model  # noqa: E402


CASE_ORDER = (
    "interp_left_shallow",
    "interp_left_deep",
    "interp_center",
    "interp_right_shallow",
    "interp_right_deep",
    "extrap_left",
    "extrap_right",
    "extrap_deep",
)
RANK = 15
SNAPSHOT_INDEX = 240
EXPECTED_CHECKPOINT_SHA256 = (
    "f6efbb81dd1e9baab0eb34b32b125e2cb58cd3b3292ee0db33a29194c84bfea1"
)
EXPECTED_G3_MARMOUSI_RELATIVE_L2 = 0.04507399697519936


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    denominator = max(float(np.linalg.norm(target64.ravel())), 1.0e-30)
    return float(np.linalg.norm((prediction64 - target64).ravel()) / denominator)


def _upsample_saved_velocity(velocity_mps: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tensor = torch.as_tensor(velocity_mps[None, None], dtype=torch.float32)
    return (
        torch.nn.functional.interpolate(
            tensor, size=shape, mode="bilinear", align_corners=True
        )[0, 0]
        .numpy()
        .astype(np.float64)
    )


def _phase4b_variant(identity: Mapping[str, Any]) -> ProbeVariant:
    return ProbeVariant(
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


def _resolved_solver_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf8"))
    return {key: raw[key] for key in ("grid", "time", "boundaries")}


def _plot_composite(
    output_stem: Path,
    *,
    velocity_mps: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    cases: list[dict[str, Any]],
    snapshot_time_s: float,
) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    extent = (
        float(x_m[0]) / 1000.0,
        float(x_m[-1]) / 1000.0,
        float(z_m[-1]) / 1000.0,
        float(z_m[0]) / 1000.0,
    )
    field_limit = max(
        max(float(np.max(np.abs(item["target_snapshot"]))) for item in cases),
        max(float(np.max(np.abs(item["prediction_snapshot"]))) for item in cases),
        1.0e-30,
    )
    error_limit = max(
        max(float(np.max(np.abs(item["error_snapshot"]))) for item in cases),
        1.0e-30,
    )
    field_norm = Normalize(vmin=-field_limit, vmax=field_limit)
    error_norm = Normalize(vmin=-error_limit, vmax=error_limit)
    fig = plt.figure(figsize=(20.8, 7.25))
    grid = fig.add_gridspec(
        3,
        10,
        width_ratios=(3.05, 1, 1, 1, 1, 1, 1, 1, 1, 0.13),
        left=0.035,
        right=0.985,
        bottom=0.12,
        top=0.885,
        wspace=0.24,
        hspace=0.16,
    )
    velocity_ax = fig.add_subplot(grid[:, 0])
    velocity_image = velocity_ax.imshow(
        velocity_mps,
        cmap="turbo",
        vmin=float(np.min(velocity_mps)),
        vmax=float(np.max(velocity_mps)),
        extent=extent,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    for number, item in enumerate(cases, start=1):
        sx, sz = item["source_parameters"][:2]
        velocity_ax.scatter(
            [sx / 1000.0],
            [sz / 1000.0],
            marker="*",
            s=82,
            c="white",
            edgecolors="black",
            linewidths=0.55,
            zorder=3,
        )
        velocity_ax.text(
            sx / 1000.0 + 0.035,
            sz / 1000.0 + 0.035,
            f"S{number}",
            color="white",
            fontsize=7.2,
            weight="bold",
            path_effects=[],
            zorder=4,
        )
    velocity_ax.set_title("True Marmousi velocity model and eight source positions", fontsize=10)
    velocity_ax.set_xlabel("x (km)")
    velocity_ax.set_ylabel("z (km)")
    velocity_cax = inset_axes(
        velocity_ax,
        width="96%",
        height="4.2%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.20, 1.0, 1.0),
        bbox_transform=velocity_ax.transAxes,
        borderpad=0,
    )
    velocity_cbar = fig.colorbar(velocity_image, cax=velocity_cax, orientation="horizontal")
    velocity_cbar.set_label("velocity (m s$^{-1}$)")

    field_mappable = None
    error_mappable = None
    field_axes: list[plt.Axes] = []
    error_axes: list[plt.Axes] = []
    row_labels = ("LWC-84 reference", "Phase4b hybrid", "Signed error")
    for column, item in enumerate(cases, start=1):
        role_tag = "in-range" if item["role"] == "interpolation" else "outside-range"
        panels = (
            (item["target_snapshot"], field_norm),
            (item["prediction_snapshot"], field_norm),
            (item["error_snapshot"], error_norm),
        )
        for row, (panel, norm) in enumerate(panels):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(
                panel,
                cmap="seismic",
                norm=norm,
                extent=extent,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
            )
            sx, sz = item["source_parameters"][:2]
            ax.scatter(
                [sx / 1000.0],
                [sz / 1000.0],
                marker="*",
                s=22,
                c="yellow",
                edgecolors="black",
                linewidths=0.25,
            )
            if row == 0:
                ax.set_title(
                    f"S{column} {role_tag}\n({sx:.0f}, {sz:.0f}) m\nall-time $e_r$={item['relative_l2']:.1%}"
                )
            if column == 1:
                ax.set_ylabel("z (km)", labelpad=1.0)
                ax.text(
                    0.03,
                    0.96,
                    row_labels[row],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=6.8,
                    weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
                )
            else:
                ax.set_yticklabels([])
            if row == 2:
                ax.set_xlabel("x (km)")
            else:
                ax.set_xticklabels([])
            if row < 2:
                field_mappable = image
                field_axes.append(ax)
            else:
                error_mappable = image
                error_axes.append(ax)
    if field_mappable is None or error_mappable is None:
        raise RuntimeError("composite plot did not create wavefield panels")
    del field_axes, error_axes
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    cbar_field = fig.colorbar(
        field_mappable, cax=fig.add_subplot(grid[0:2, 9]), orientation="vertical", format=formatter
    )
    cbar_field.set_label(f"pressure (global $\\pm${field_limit:.2e})")
    error_formatter = ScalarFormatter(useMathText=True)
    error_formatter.set_powerlimits((-2, 2))
    cbar_error = fig.colorbar(
        error_mappable,
        cax=fig.add_subplot(grid[2, 9]),
        orientation="vertical",
        format=error_formatter,
    )
    cbar_error.set_label(f"signed error (global $\\pm${error_limit:.2e})")
    fig.suptitle(
        f"Fixed 19-Hz source-position test on one locked velocity slice, t={snapshot_time_s:.3f} s",
        fontsize=12,
    )
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--fixed19-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase4b-source-h5", type=Path, required=True)
    parser.add_argument("--phase4b-normalization", type=Path, required=True)
    parser.add_argument("--phase4b-frozen-config", type=Path, required=True)
    parser.add_argument("--reference-frozen-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-block", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.run_dir.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    fixed19_root = args.fixed19_root.expanduser().resolve()
    source_h5 = args.phase4b_source_h5.expanduser().resolve()
    normalization_path = args.phase4b_normalization.expanduser().resolve()
    phase4b_frozen = args.phase4b_frozen_config.expanduser().resolve()
    reference_frozen = args.reference_frozen_config.expanduser().resolve()
    input_manifest_path = fixed19_root / "prediction_manifest.json"

    identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf8"))
    terminal = json.loads((run_dir / "terminal.json").read_text(encoding="utf8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf8"))
    if terminal.get("status") != "complete" or input_manifest.get("status") != "complete":
        raise ValueError("Phase4b run and fixed-19-Hz input seal must both be complete")
    if input_manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("fixed-19-Hz input seal no longer matches the registered protocol")
    if bool(input_manifest.get("source_frequency_varied", True)):
        raise ValueError("input seal does not satisfy the fixed-frequency contract")

    checkpoint = Path(str(terminal["best_checkpoint"])).expanduser().resolve()
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("selected Phase4b checkpoint hash changed")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("manifest_digest") != identity["manifest_digest"]:
        raise ValueError("checkpoint and Phase4b run manifest digests disagree")
    if checkpoint_payload.get("config_digest") != identity["run_digest"]:
        raise ValueError("checkpoint and Phase4b run config digests disagree")
    checkpoint_metric = float(
        checkpoint_payload.get("metrics", {}).get("triplet_aggregate_relative_l2", np.nan)
    )
    if not np.isclose(
        checkpoint_metric,
        float(terminal["best_fixed_heldout_relative_l2"]),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("checkpoint metric does not reproduce the terminal selection score")
    checkpoint_step = int(checkpoint_payload.get("global_step", -1))
    del checkpoint_payload

    if _resolved_solver_contract(phase4b_frozen) != _resolved_solver_contract(reference_frozen):
        raise ValueError("Phase4b background and sealed-reference solver contracts differ")
    input_row = next(row for row in input_manifest["inputs"] if int(row["slice_rank"]) == RANK)
    velocity_path = Path(str(input_row["input_path"])).expanduser().resolve()
    if sha256_file(velocity_path) != input_row["input_file_sha256"]:
        raise ValueError("rank-15 input file hash changed")
    with np.load(velocity_path) as values:
        velocity_mps = np.asarray(values["velocity_mps"], dtype=np.float32)
        time_s_exact = np.asarray(values["time_s"], dtype=np.float64)
        time_s = time_s_exact.astype(np.float32)
        x_m = np.asarray(values["x_m"], dtype=np.float32)
        z_m = np.asarray(values["z_m"], dtype=np.float32)
    if _array_sha256(velocity_mps) != input_row["velocity_array_sha256"]:
        raise ValueError("rank-15 velocity bytes changed")
    if velocity_mps.shape != (201, 201) or time_s.shape != (401,):
        raise ValueError("rank-15 input geometry changed")

    registered_positions = {row["case_id"]: row for row in protocol["positions_m"]}
    sealed_rows = {
        row["case_id"]: row
        for row in input_manifest["records"]
        if int(row["slice_rank"]) == RANK
    }
    if tuple(sealed_rows) != tuple(sorted(sealed_rows)):
        sealed_rows = dict(sorted(sealed_rows.items()))
    if set(sealed_rows) != set(CASE_ORDER):
        raise ValueError("rank-15 sealed cases differ from the eight registered positions")
    case_inputs: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        row = sealed_rows[case_id]
        registered = registered_positions[case_id]
        source_file = Path(str(row["prediction_path"])).expanduser().resolve()
        if sha256_file(source_file) != row["prediction_sha256"]:
            raise ValueError(f"sealed source-input file hash changed for {case_id}")
        with np.load(source_file) as values:
            source_parameters = np.asarray(values["source_parameters"], dtype=np.float32)
            source_map = np.asarray(values["source_map_zx"], dtype=np.float32)
        expected_source = np.asarray(
            [
                registered["x_m"],
                registered["z_m"],
                protocol["fixed_source_parameters"]["source_f0_hz"],
                protocol["fixed_source_parameters"]["source_t0_s"],
                protocol["fixed_source_parameters"]["source_amplitude"],
            ],
            dtype=np.float32,
        )
        if not np.allclose(source_parameters, expected_source, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"registered source parameters changed for {case_id}")
        if source_map.shape != (201, 201) or not np.isclose(float(source_map.sum()), 1.0):
            raise ValueError(f"invalid source map for {case_id}")
        case_inputs.append(
            {
                "case_id": case_id,
                "role": str(registered["role"]),
                "source_parameters": source_parameters,
                "source_map": source_map,
                "source_input_file": str(source_file),
                "source_input_sha256": row["prediction_sha256"],
            }
        )

    axis_manifest = build_manifest(source_h5)
    if not (
        np.array_equal(np.asarray(axis_manifest.time_s), time_s_exact)
        and np.allclose(np.asarray(axis_manifest.x_m), x_m, rtol=0.0, atol=1.0e-6)
        and np.allclose(np.asarray(axis_manifest.z_m), z_m, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("Phase4b architecture axes differ from the fixed-19-Hz panel")
    normalization_payload = json.loads(normalization_path.read_text(encoding="utf8"))
    normalizer = PhysicalNormalizer.from_dict(
        normalization_payload, expected_manifest=str(identity["manifest_digest"])
    )
    base = build_base_config(int(identity["width"]))
    if normalizer.metadata.record_count != base.data.expected_train_records:
        raise ValueError("Phase4b normalizer train-record count changed")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(identity["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(identity["seed"]))
    model = _model(base, axis_manifest, _phase4b_variant(identity)).to(device)
    transfer = warmstart_full_helmholtz(model, checkpoint)
    if int(transfer["transferred_tensor_count"]) != len(model.state_dict()):
        raise ValueError("Phase4b checkpoint did not populate every model tensor")
    model.eval()
    loss_config = build_probe_config(
        dense_lr=float(identity["dense_learning_rate"]),
        backbone_lr=float(identity["backbone_learning_rate"]),
        temporal_lr=1.0e-5,
        seed=int(identity["seed"]),
        travel_time_h5="",
        family_gradient_weights=identity.get("family_gradient_weights"),
        per_frame_frame=bool(identity.get("per_frame_frame", False)),
        frame_energy_floor_fraction=float(identity.get("frame_energy_floor_fraction", 0.0)),
        late_frame_gain=float(identity.get("late_frame_gain", 0.0)),
        late_frame_start_fraction=float(identity.get("late_frame_start_fraction", 0.4)),
    )["loss"]

    cfg = resolve_config(load_config(str(phase4b_frozen)))
    grid = grid_from_config(cfg)
    boundaries = boundaries_from_config(cfg)
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=float(cfg["time"]["dt_used_s"]),
        output_times_s=time_s_exact,
        c_ref_mps=6750.0,
        device=str(device),
        dtype=torch.float32,
        kappa_max=float(cfg["boundaries"]["kappa_max"]),
        minimum_frequency_hz=float(cfg["boundaries"]["minimum_frequency_hz"]),
    )
    sigma = float(identity.get("helmholtz_background_sigma_cells", 2.0))
    smoothed_saved = gaussian_filter(
        velocity_mps.astype(np.float64), sigma=sigma, mode="nearest"
    )
    background_velocity_fine = _upsample_saved_velocity(
        smoothed_saved.astype(np.float32), (grid.nz, grid.nx)
    )

    prediction_manifest: dict[str, Any] = {
        "schema": "phase4b_fixed19hz_rank15_multisource_prediction_seal_v1",
        "status": "building",
        "prediction_sealed_before_reference_access": True,
        "reference_wavefields_accessed": False,
        "frequency_generalization_claim_permitted": False,
        "varied_variable": "source_position_only",
        "fixed_source_frequency_hz": 19.0,
        "rank": RANK,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "velocity_input": str(velocity_path),
        "velocity_input_sha256": sha256_file(velocity_path),
        "velocity_array_sha256": _array_sha256(velocity_mps),
        "time_axis_sha256": time_axis_sha256(time_s_exact),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_global_step": checkpoint_step,
        "run_digest": identity["run_digest"],
        "training_manifest_digest": identity["manifest_digest"],
        "axis_source_h5": str(source_h5),
        "axis_source_manifest_observed_digest": axis_manifest.digest,
        "axis_source_used_only_for_architecture_axes": True,
        "normalization": str(normalization_path),
        "normalization_sha256": sha256_file(normalization_path),
        "background_sigma_saved_cells": sigma,
        "records": [],
    }
    print(json.dumps({"event": "input_contract_verified", "cases": 8}), flush=True)
    for number, item in enumerate(case_inputs, start=1):
        source_values = item["source_parameters"]
        result = solver.simulate(
            background_velocity_fine[None],
            source_x_m=float(source_values[0]),
            source_z_m=float(source_values[1]),
            source_f0_hz=float(source_values[2]),
            source_t0_s=float(source_values[3]),
            source_amplitude=float(source_values[4]),
        )
        background = np.asarray(result.wavefield[0], dtype=np.float32)
        solver_source_map = np.asarray(result.source_map_saved[0], dtype=np.float32)
        if background.shape != (401, 201, 201) or not np.isfinite(background).all():
            raise RuntimeError(f"invalid Phase4b background for {item['case_id']}")
        if not np.allclose(solver_source_map, item["source_map"], rtol=0.0, atol=1.0e-7):
            raise ValueError(f"solver/source seal map mismatch for {item['case_id']}")
        travel_time = straight_ray_grid_numpy(
            velocity_mps,
            source_x_m=source_values[None, 0],
            source_z_m=source_values[None, 1],
            x_m=x_m,
            z_m=z_m,
            samples=12,
        )[0]
        velocity_tensor = torch.from_numpy(velocity_mps)[None, None].to(device)
        source_tensor = torch.from_numpy(source_values)[None].to(device)
        source_map_tensor = torch.from_numpy(item["source_map"])[None, None].to(device)
        requested_time_tensor = torch.from_numpy(time_s)[None].to(device)
        prepared = model.prepare_sources(
            model.encode_medium(velocity_tensor, normalizer),
            source_tensor,
            source_map_tensor,
            normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        dense_grid = model.prepare_dense_grid(
            prepared,
            x_m=torch.from_numpy(x_m).to(device),
            z_m=torch.from_numpy(z_m).to(device),
            travel_time_s=torch.from_numpy(travel_time)[None].to(device),
        )
        prediction_norm, coarse_norm = model.dense_normalized_with_coarse(
            prepared,
            requested_time_tensor,
            dense_grid=dense_grid,
            time_block=int(args.time_block),
        )
        background_norm = normalizer.encode_pressure(
            torch.from_numpy(background)[None].to(device), source_tensor[:, 4]
        )
        prediction_norm = prediction_norm + background_norm
        coarse_norm = coarse_norm + background_norm
        if bool(loss_config.get("hard_causality", False)):
            onset = source_causality_onset_s(
                source_tensor,
                lead_cycles=float(loss_config.get("hard_causality_lead_cycles", 0.0)),
            )
            prediction_norm = apply_hard_causality(
                prediction_norm, requested_time_tensor, onset
            )
            coarse_norm = apply_hard_causality(coarse_norm, requested_time_tensor, onset)
        prediction = (
            normalizer.decode_pressure(prediction_norm.float(), source_tensor[:, 4])
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        coarse_hybrid = (
            normalizer.decode_pressure(coarse_norm.float(), source_tensor[:, 4])
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if not all(np.isfinite(value).all() for value in (prediction, coarse_hybrid)):
            raise RuntimeError(f"non-finite Phase4b output for {item['case_id']}")
        prediction_path = output_dir / "sealed_predictions" / f"marm_r15_{item['case_id']}.npz"
        _atomic_npz(
            prediction_path,
            prediction_tzx=prediction,
            coarse_hybrid_tzx=coarse_hybrid,
            numerical_background_tzx=background,
            source_parameters=source_values,
            source_map_zx=item["source_map"],
            travel_time_s=travel_time,
        )
        prediction_manifest["records"].append(
            {
                "case_id": item["case_id"],
                "role": item["role"],
                "source_parameters": [float(value) for value in source_values],
                "source_input_file": item["source_input_file"],
                "source_input_sha256": item["source_input_sha256"],
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )
        print(
            json.dumps(
                {"event": "prediction_sealed", "case": number, "case_id": item["case_id"]}
            ),
            flush=True,
        )
        del result, background, prediction, coarse_hybrid, prepared, dense_grid

    prediction_manifest["status"] = "complete"
    prediction_manifest_path = output_dir / "prediction_manifest.json"
    _atomic_json(prediction_manifest, prediction_manifest_path)
    print(json.dumps({"event": "all_predictions_sealed_before_truth_access"}), flush=True)

    reference_manifest_path = fixed19_root / "reference_manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf8"))
    if reference_manifest.get("status") != "complete":
        raise ValueError("fixed-19-Hz reference seal is incomplete")
    if reference_manifest.get("protocol_sha256") != sha256_file(protocol_path):
        raise ValueError("reference seal no longer matches the registered protocol")
    reference_rows = {
        row["case_id"]: row
        for row in reference_manifest["records"]
        if int(row["slice_rank"]) == RANK
    }
    if set(reference_rows) != set(CASE_ORDER):
        raise ValueError("rank-15 reference cases changed")

    scored_cases: list[dict[str, Any]] = []
    for record in prediction_manifest["records"]:
        case_id = record["case_id"]
        reference_row = reference_rows[case_id]
        reference_path = Path(str(reference_row["reference_path"])).expanduser().resolve()
        if sha256_file(reference_path) != reference_row["reference_sha256"]:
            raise ValueError(f"reference file hash changed for {case_id}")
        with np.load(record["prediction_path"]) as values:
            prediction = np.asarray(values["prediction_tzx"], dtype=np.float32)
            coarse_hybrid = np.asarray(values["coarse_hybrid_tzx"], dtype=np.float32)
            numerical_background = np.asarray(values["numerical_background_tzx"], dtype=np.float32)
            source_parameters = np.asarray(values["source_parameters"], dtype=np.float32)
        with np.load(reference_path) as values:
            target = np.asarray(values["target_tzx"], dtype=np.float32)
            reference_source = np.asarray(values["source_parameters"], dtype=np.float32)
        if not np.array_equal(source_parameters, reference_source):
            raise ValueError(f"prediction/reference source mismatch for {case_id}")
        relative_l2 = _relative_l2(prediction, target)
        background_relative_l2 = _relative_l2(numerical_background, target)
        coarse_relative_l2 = _relative_l2(coarse_hybrid, target)
        correction_ratio = float(
            np.linalg.norm((prediction - coarse_hybrid).astype(np.float64).ravel())
            / max(float(np.linalg.norm(coarse_hybrid.astype(np.float64).ravel())), 1.0e-30)
        )
        scored_cases.append(
            {
                "case_id": case_id,
                "role": record["role"],
                "source_parameters": source_parameters,
                "relative_l2": relative_l2,
                "background_only_relative_l2": background_relative_l2,
                "coarse_hybrid_relative_l2": coarse_relative_l2,
                "helmholtz_correction_to_coarse_l2_ratio": correction_ratio,
                "snapshot_relative_l2": _relative_l2(
                    prediction[SNAPSHOT_INDEX], target[SNAPSHOT_INDEX]
                ),
                "target_snapshot": target[SNAPSHOT_INDEX],
                "prediction_snapshot": prediction[SNAPSHOT_INDEX],
                "error_snapshot": prediction[SNAPSHOT_INDEX] - target[SNAPSHOT_INDEX],
                "reference_path": str(reference_path),
                "reference_sha256": reference_row["reference_sha256"],
            }
        )
        print(
            json.dumps(
                {"event": "case_scored", "case_id": case_id, "relative_l2": relative_l2}
            ),
            flush=True,
        )

    figures = _plot_composite(
        output_dir / "phase4b_marmousi_fixed19_multisource_composite",
        velocity_mps=velocity_mps,
        x_m=x_m,
        z_m=z_m,
        cases=scored_cases,
        snapshot_time_s=float(time_s[SNAPSHOT_INDEX]),
    )
    relative_values = np.asarray([item["relative_l2"] for item in scored_cases])
    interpolation_values = np.asarray(
        [item["relative_l2"] for item in scored_cases if item["role"] == "interpolation"]
    )
    outside_values = np.asarray(
        [item["relative_l2"] for item in scored_cases if item["role"] != "interpolation"]
    )
    report = {
        "schema": "phase4b_fixed19hz_rank15_multisource_evaluation_v1",
        "status": "complete",
        "claim_scope": {
            "varied_variable": "source_position_only",
            "fixed_frequency_hz": 19.0,
            "one_locked_velocity_slice": True,
            "case_study_not_population_estimate": True,
            "frequency_generalization_claim_permitted": False,
        },
        "prediction_sealed_before_reference_access": True,
        "prediction_manifest": str(prediction_manifest_path.resolve()),
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "reference_manifest": str(reference_manifest_path.resolve()),
        "reference_manifest_sha256": sha256_file(reference_manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_global_step": checkpoint_step,
        "separate_g3_best_result": {
            "sample_id": "validation_marmousi_00000",
            "all_401_frame_relative_l2": EXPECTED_G3_MARMOUSI_RELATIVE_L2,
            "percent": 100.0 * EXPECTED_G3_MARMOUSI_RELATIVE_L2,
            "not_the_same_velocity_or_source_panel": True,
        },
        "panel": {
            "rank": RANK,
            "source_sample_id": input_row["source_sample_id"],
            "source_group_id": input_row["source_group_id"],
            "velocity_array_sha256": _array_sha256(velocity_mps),
            "snapshot_index": SNAPSHOT_INDEX,
            "snapshot_time_s": float(time_s[SNAPSHOT_INDEX]),
            "mean_relative_l2": float(np.mean(relative_values)),
            "median_relative_l2": float(np.median(relative_values)),
            "min_relative_l2": float(np.min(relative_values)),
            "max_relative_l2": float(np.max(relative_values)),
            "interpolation_mean_relative_l2": float(np.mean(interpolation_values)),
            "outside_range_mean_relative_l2": float(np.mean(outside_values)),
            "records": [
                {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "target_snapshot",
                        "prediction_snapshot",
                        "error_snapshot",
                        "source_parameters",
                    }
                }
                | {"source_parameters": [float(value) for value in item["source_parameters"]]}
                for item in scored_cases
            ],
        },
        "figures": {
            key: {"path": value, "sha256": sha256_file(Path(value))}
            for key, value in figures.items()
        },
        "interpretation": (
            "Phase4b is an external smoothed-background numerical hybrid. The neural "
            "Helmholtz correction must be interpreted through the reported correction-to-coarse ratios."
        ),
    }
    report_path = output_dir / "report.json"
    _atomic_json(report, report_path)
    print(
        json.dumps(
            {
                "event": "complete",
                "mean_relative_l2": report["panel"]["mean_relative_l2"],
                "interpolation_mean_relative_l2": report["panel"]["interpolation_mean_relative_l2"],
                "outside_range_mean_relative_l2": report["panel"]["outside_range_mean_relative_l2"],
                "report": str(report_path.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
