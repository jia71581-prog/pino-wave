#!/usr/bin/env python3
"""Build compact Elastic VTI figures for the group-meeting Beamer deck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.checkpoint import load_checkpoint
from fno_acoustic.config import load_config
from fno_acoustic.data_elastic_vti import PinoHDF5Dataset
from fno_acoustic.model_elastic_vti import ElasticVTIFNO3D
from fno_acoustic.normalization import decode_standard


MODEL_SPECS = {
    "uniform": {
        "title": "Uniform",
        "config": ROOT / "configs/pino_elastic_vti_uniform_gpu.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_uniform/checkpoints/best.pt",
        "metrics": ROOT / "artifacts/elastic_vti_pino_uniform/evaluation/metrics/metrics.json",
        "sample_index": 6,
    },
    "layered": {
        "title": "Layered",
        "config": ROOT / "configs/pino_elastic_vti_layered_gpu.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_layered/checkpoints/best.pt",
        "metrics": ROOT / "artifacts/elastic_vti_pino_layered/evaluation/metrics/metrics.json",
        "sample_index": 3,
    },
    "marmousi": {
        "title": "Marmousi",
        "config": ROOT / "configs/pino_elastic_vti_marmousi_component_balanced_short.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_marmousi_component_balanced_short/checkpoints/best.pt",
        "metrics": ROOT
        / "artifacts/elastic_vti_pino_marmousi_component_balanced_short/evaluation_fixed4/metrics/metrics.json",
        "pretrain_metrics": ROOT / "artifacts/elastic_vti_pino_marmousi/evaluation/metrics/metrics.json",
        "sample_index": 300,
    },
}

PALETTE = {"blue": "#16345b", "red": "#b02330", "green": "#24684e", "gray": "#59636e"}
VELOCITY_DISPLAY_LIMITS = {
    "vp": (1400.0, 7000.0),
    "vs": (800.0, 4600.0),
}


def coordinate_edges_km(coordinates_m: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(coordinates_m, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("coordinates_m must contain at least two cell centres")
    spacing = float(np.median(np.diff(values)))
    return float((values[0] - 0.5 * spacing) / 1000.0), float((values[-1] + 0.5 * spacing) / 1000.0)


def receiver_gather_time_x(field_zxt: np.ndarray, receiver_depth_index: int) -> np.ndarray:
    field = np.asarray(field_zxt)
    if field.ndim != 3:
        raise ValueError("field_zxt must have shape [z, x, time]")
    return field[int(receiver_depth_index), :, :]


def select_nearest_frames(times_s: Sequence[float], requested_s: Sequence[float]) -> list[int]:
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("times_s must be a non-empty one-dimensional sequence")
    return [int(np.argmin(np.abs(times - float(value)))) for value in requested_s]


def summarize_evaluation_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ux = payload["per_component_mean"]["component_0"]
    uz = payload["per_component_mean"]["component_1"]
    return {
        "ux_relative_l2": float(ux["relative_l2"]),
        "uz_relative_l2": float(uz["relative_l2"]),
        "ux_receiver_relative_l2": float(ux["receiver_line_relative_l2"]),
        "uz_receiver_relative_l2": float(uz["receiver_line_relative_l2"]),
    }


def relative_l2_per_component(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(target, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim < 2 or pred.shape[-1] != 2:
        raise ValueError("prediction and target must have identical shape ending in two displacement components")
    values = []
    for component in range(2):
        numerator = np.linalg.norm((pred[..., component] - ref[..., component]).ravel())
        denominator = max(float(np.linalg.norm(ref[..., component].ravel())), 1.0e-16)
        values.append(float(numerator / denominator))
    return {"ux": values[0], "uz": values[1]}


def select_validation_indices(split_path: Path, count: int) -> list[int]:
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    indices = [int(value) for value in payload.get("val", [])]
    if len(indices) < int(count):
        raise ValueError(f"validation split has {len(indices)} instances, fewer than requested {count}")
    return indices[: int(count)]


def validate_distribution_metrics(
    payload: dict[str, Any], expected_count: int = 30, required_models: set[str] | None = None
) -> None:
    if int(payload.get("protocol", {}).get("instances_per_model", -1)) != int(expected_count):
        raise ValueError("distribution protocol has the wrong instance count")
    required = set(MODEL_SPECS) if required_models is None else set(required_models)
    if set(payload.get("models", {})) != required:
        raise ValueError(f"distribution metrics must contain {sorted(required)}")
    for name, metrics in payload["models"].items():
        for key in ("indices", "ux_relative_l2", "uz_relative_l2"):
            values = metrics.get(key, [])
            if len(values) != int(expected_count):
                raise ValueError(f"{name} {key} must contain {expected_count} values")
            if key != "indices" and not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
                raise ValueError(f"{name} {key} contains non-finite values")


def validate_manifest(manifest: dict[str, Any]) -> None:
    models = manifest.get("models", {})
    required_models = set(MODEL_SPECS)
    if set(models) != required_models:
        raise ValueError(f"manifest models must be {sorted(required_models)}")
    for name, payload in models.items():
        if set(payload.get("figures", {})) != {"ux", "uz", "receivers"}:
            raise ValueError(f"{name} must define ux, uz, and receivers figures")
        metrics = payload.get("metrics", {})
        if not metrics or not all(np.isfinite(float(value)) for value in metrics.values()):
            raise ValueError(f"{name} metrics must be finite and non-empty")


def load_case(model_name: str, sample_index: int | None = None, device: str = "cpu") -> dict[str, Any]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model {model_name!r}")
    spec = MODEL_SPECS[model_name]
    index = int(spec["sample_index"] if sample_index is None else sample_index)
    config = load_config(spec["config"])
    checkpoint = load_checkpoint(spec["checkpoint"], map_location="cpu")
    dataset = PinoHDF5Dataset(config, [index], normalization_stats=checkpoint["normalization_stats"], return_normalized=True)
    sample = dataset[0]
    model_config = {key: value for key, value in checkpoint["model_config"].items() if key != "name"}
    model = ElasticVTIFNO3D(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        pred_norm = model(sample["input"].unsqueeze(0).to(device)).cpu()[0]
    target = decode_standard(
        sample["target"], checkpoint["normalization_stats"]["wavefield"], eps=config["normalization"]["eps"]
    ).cpu()
    pred = decode_standard(
        pred_norm, checkpoint["normalization_stats"]["wavefield"], eps=config["normalization"]["eps"]
    ).cpu()
    metadata = sample["metadata"]
    with h5py.File(metadata["shard_path"], "r") as h5:
        x_m = np.asarray(h5[config["data"]["x_key"]], dtype=np.float64)
        z_m = np.asarray(h5[config["data"]["z_key"]], dtype=np.float64)
    x_edges_km = coordinate_edges_km(x_m)
    z_edges_km = coordinate_edges_km(z_m)
    nz, nx = target.shape[:2]
    x_cell_km = (x_edges_km[1] - x_edges_km[0]) / nx
    z_cell_km = (z_edges_km[1] - z_edges_km[0]) / nz
    x_km = np.linspace(x_edges_km[0] + 0.5 * x_cell_km, x_edges_km[1] - 0.5 * x_cell_km, nx)
    z_km = np.linspace(z_edges_km[0] + 0.5 * z_cell_km, z_edges_km[1] - 0.5 * z_cell_km, nz)
    sample_metrics = relative_l2_per_component(pred.numpy(), target.numpy())
    receiver_z_index = _receiver_depth(target.numpy())
    for component, label in ((0, "ux"), (1, "uz")):
        reference_receiver = receiver_gather_time_x(target.numpy()[..., component], receiver_z_index)
        prediction_receiver = receiver_gather_time_x(pred.numpy()[..., component], receiver_z_index)
        sample_metrics[f"{label}_receiver_relative_l2"] = float(
            np.linalg.norm(prediction_receiver - reference_receiver)
            / max(float(np.linalg.norm(reference_receiver)), 1.0e-16)
        )
    return {
        "name": model_name,
        "title": spec["title"],
        "sample_index": index,
        "checkpoint": str(spec["checkpoint"]),
        "config": str(spec["config"]),
        "velocity": sample["velocity"].cpu().numpy(),
        "source_map": sample["source_map"][0].cpu().numpy(),
        "time": sample["time"].cpu().numpy(),
        "target": target.numpy(),
        "prediction": pred.numpy(),
        "metrics": summarize_evaluation_metrics(spec["metrics"]),
        "sample_metrics": sample_metrics,
        "x_edges_km": x_edges_km,
        "z_edges_km": z_edges_km,
        "x_km": x_km,
        "z_km": z_km,
        "source_xy_km": (
            float(metadata["source_position_x"]) / 1000.0,
            float(metadata["source_position_z"]) / 1000.0,
        ),
    }


def _field_limit(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(max(np.percentile(np.abs(reference), 99.5), np.percentile(np.abs(prediction), 99.5), 1.0e-12))


def _error_limit(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(max(np.percentile(np.abs(prediction - reference), 99.5), 1.0e-12))


def _source_xy(source_map: np.ndarray) -> tuple[int, int]:
    z, x = np.unravel_index(int(np.argmax(source_map)), source_map.shape)
    return int(x), int(z)


def _plot_field(
    ax,
    data: np.ndarray,
    limit: float,
    title: str,
    source_xy: tuple[float, float],
    extent_km: tuple[float, float, float, float],
    cmap: str = "seismic",
):
    image = ax.imshow(
        data,
        origin="upper",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        extent=extent_km,
        aspect="equal",
    )
    ax.scatter([source_xy[0]], [source_xy[1]], marker="*", s=44, c="#ffd43b", edgecolors="black", linewidths=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (km)", fontsize=8)
    ax.set_ylabel("z (km)", fontsize=8)
    ax.set_xticks(np.linspace(extent_km[0], extent_km[1], 5))
    ax.set_yticks(np.linspace(extent_km[3], extent_km[2], 5))
    ax.tick_params(labelsize=7)
    return image


def plot_ux_context(case: dict[str, Any], output: Path) -> None:
    frames = select_nearest_frames(case["time"], [0.165, 0.335])
    source_xy = case["source_xy_km"]
    extent_km = (*case["x_edges_km"], case["z_edges_km"][1], case["z_edges_km"][0])
    fig = plt.figure(figsize=(14.5, 6.0), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[0.82, 1.0, 1.0, 1.0])
    velocity_panels = zip(case["velocity"][:2], ("$v_p$ (m/s)", "$v_s$ (m/s)"), ("vp", "vs"))
    for row, (velocity, label, component) in enumerate(velocity_panels):
        ax = fig.add_subplot(gs[row, 0])
        vmin, vmax = VELOCITY_DISPLAY_LIMITS[component]
        im = ax.imshow(
            velocity,
            origin="upper",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=extent_km,
            aspect="equal",
        )
        ax.scatter([source_xy[0]], [source_xy[1]], marker="*", s=80, c="#ffd43b", edgecolors="black")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (km)", fontsize=8)
        ax.set_ylabel("z (km)", fontsize=8)
        ax.set_xticks(np.linspace(extent_km[0], extent_km[1], 5))
        ax.set_yticks(np.linspace(extent_km[3], extent_km[2], 5))
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    last_images = []
    for row, frame in enumerate(frames):
        ref = case["target"][..., frame, 0]
        pred = case["prediction"][..., frame, 0]
        field_limit = _field_limit(ref, pred)
        error_limit = _error_limit(ref, pred)
        for col, (data, title, limit) in enumerate(
            ((ref, "Reference", field_limit), (pred, "PINO", field_limit), (pred - ref, "Error", error_limit)), start=1
        ):
            ax = fig.add_subplot(gs[row, col])
            im = _plot_field(ax, data, limit, f"{title}\nt={case['time'][frame]:.3f} s", source_xy, extent_km)
            last_images.append((im, ax))
    for im, ax in last_images:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{case['title']} sample {case['sample_index']}: $u_x$ wavefield", fontsize=14, color=PALETTE["blue"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_uz_comparison(case: dict[str, Any], output: Path) -> None:
    frames = select_nearest_frames(case["time"], [0.165, 0.335])
    source_xy = case["source_xy_km"]
    extent_km = (*case["x_edges_km"], case["z_edges_km"][1], case["z_edges_km"][0])
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 6.2), constrained_layout=True)
    for row, frame in enumerate(frames):
        ref = case["target"][..., frame, 1]
        pred = case["prediction"][..., frame, 1]
        field_limit = _field_limit(ref, pred)
        error_limit = _error_limit(ref, pred)
        for col, (data, title, limit) in enumerate(
            ((ref, "Reference", field_limit), (pred, "PINO", field_limit), (pred - ref, "Error", error_limit))
        ):
            im = _plot_field(
                axes[row, col], data, limit, f"{title} | t={case['time'][frame]:.3f} s", source_xy, extent_km
            )
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.suptitle(f"{case['title']} sample {case['sample_index']}: $u_z$ wavefield", fontsize=14, color=PALETTE["blue"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _receiver_depth(target: np.ndarray) -> int:
    energy = np.sqrt(np.mean(target[..., 0].square() if hasattr(target[..., 0], "square") else target[..., 0] ** 2, axis=(1, 2)))
    candidates = np.arange(max(1, target.shape[0] // 2))
    return int(candidates[np.argmax(energy[candidates])])


def plot_receiver_runtime(case: dict[str, Any], runtime: dict[str, Any], output: Path) -> None:
    target, pred = case["target"], case["prediction"]
    z_idx = _receiver_depth(target)
    time_s = case["time"]
    fig = plt.figure(figsize=(15.5, 6.2), constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.0, 1.0, 1.05, 0.82])
    labels = ("$u_x$", "$u_z$")
    for row, component in enumerate((0, 1)):
        ref_gather = receiver_gather_time_x(target[..., component], z_idx)
        pred_gather = receiver_gather_time_x(pred[..., component], z_idx)
        limit = _field_limit(ref_gather, pred_gather)
        for col, (gather, title) in enumerate(((ref_gather, "Reference gather"), (pred_gather, "PINO gather"))):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                gather,
                origin="upper",
                aspect="auto",
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                extent=[float(time_s[0]), float(time_s[-1]), case["x_edges_km"][0], case["x_edges_km"][1]],
            )
            ax.set_title(f"{labels[row]} {title}", fontsize=9)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("receiver x (km)")
            ax.set_xticks(np.linspace(float(time_s[0]), float(time_s[-1]), 6))
            ax.set_yticks(np.linspace(case["x_edges_km"][0], case["x_edges_km"][1], 5))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = fig.add_subplot(gs[row, 2])
        active_x = int(np.argmax(np.linalg.norm(ref_gather, axis=1)))
        ax.plot(time_s, target[z_idx, active_x, :, component], color=PALETTE["blue"], lw=1.5, label="Reference")
        ax.plot(time_s, pred[z_idx, active_x, :, component], color=PALETTE["red"], lw=1.2, ls="--", label="PINO")
        ax.set_title(
            f"{labels[row]} trace | x={case['x_km'][active_x]:.2f} km, z={case['z_km'][z_idx]:.2f} km",
            fontsize=9,
        )
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[:, 3])
    fd_s = float(runtime["fd8"]["median_s"])
    pino_s = float(runtime["pino"]["median_s"])
    bars = ax.bar(["FD8", "PINO"], [fd_s, pino_s], color=[PALETTE["gray"], PALETTE["green"]])
    ax.set_yscale("log")
    ax.set_ylabel("median runtime (s, log scale)")
    ax.set_title(f"RTX 3090\n{runtime['speedup']:.1f}x speedup", fontsize=11, color=PALETTE["blue"])
    ax.bar_label(bars, labels=[f"{fd_s:.2f}s", f"{pino_s:.3f}s"], padding=3, fontsize=9)
    ax.text(
        0.5,
        0.02,
        f"FD: {runtime['fd_physical_shape'][0]}x{runtime['fd_physical_shape'][1]}x{runtime['fd_time_steps']}\n"
        f"PINO output: {'x'.join(str(v) for v in runtime['pino_output_shape'][1:4])}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.suptitle(f"{case['title']} sample {case['sample_index']}: receiver agreement and runtime", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cross_model_overview(metrics: dict[str, Any], output: Path) -> None:
    names = list(metrics)
    x = np.arange(len(names))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    for ax, suffix, title in (
        (axes[0], "relative_l2", "Full-field relative L2"),
        (axes[1], "receiver_relative_l2", "Receiver-line relative L2"),
    ):
        ux = [metrics[name][f"ux_{suffix}"] for name in names]
        uz = [metrics[name][f"uz_{suffix}"] for name in names]
        ax.bar(x - width / 2, ux, width, label="$u_x$", color=PALETTE["red"])
        ax.bar(x + width / 2, uz, width, label="$u_z$", color=PALETTE["blue"])
        ax.set_xticks(x, names, rotation=12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Elastic VTI PINO: cross-model validation", fontsize=14, color=PALETTE["blue"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distribution_boxplots(
    payload: dict[str, Any], output: Path, marmousi_pretrain: dict[str, Any] | None = None
) -> None:
    expected_count = int(payload.get("protocol", {}).get("instances_per_model", 30))
    validate_distribution_metrics(payload, expected_count=expected_count)
    entries: list[tuple[str, dict[str, Any], str, str]] = [
        ("uniform", payload["models"]["uniform"], "Uniform", "#56B4E9"),
        ("layered", payload["models"]["layered"], "Layered", "#009E73"),
    ]
    if marmousi_pretrain is not None:
        validate_distribution_metrics(
            marmousi_pretrain, expected_count=expected_count, required_models={"marmousi_pretrain"}
        )
        entries.append(("marmousi_pretrain", marmousi_pretrain["models"]["marmousi_pretrain"], "Marmousi pre", "#E69F00"))
    entries.append(("marmousi", payload["models"]["marmousi"], "Marmousi FT", "#D55E00"))
    model_labels = [entry[2] for entry in entries]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    rng = np.random.default_rng(20260711)
    for ax, component, title in zip(axes, ("ux", "uz"), (r"$u_x$ full-field relative $L_2$", r"$u_z$ full-field relative $L_2$")):
        data = [np.asarray(entry[1][f"{component}_relative_l2"], dtype=np.float64) for entry in entries]
        boxes = ax.boxplot(
            data,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#263238", "markersize": 5},
            medianprops={"color": "#263238", "linewidth": 1.8},
            whiskerprops={"color": "#59636e", "linewidth": 1.2},
            capprops={"color": "#59636e", "linewidth": 1.2},
            flierprops={"marker": "", "markersize": 0},
        )
        for position, (values, patch, entry) in enumerate(zip(data, boxes["boxes"], entries), start=1):
            color = entry[3]
            patch.set_facecolor(color)
            patch.set_alpha(0.42)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
            jitter = rng.uniform(-0.11, 0.11, size=values.size)
            ax.scatter(
                position + jitter,
                values,
                s=17,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.80,
                zorder=3,
            )
            median = float(np.median(values))
            ax.annotate(
                f"median={median:.3f}",
                (position, median),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#263238",
            )
        ax.set_xticks(np.arange(1, len(model_labels) + 1), model_labels)
        ax.set_ylabel(r"Relative $L_2$ error")
        ax.set_title(title, fontsize=13, color=PALETTE["blue"])
        ax.grid(axis="y", alpha=0.22, linestyle="-")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.01, 0.98, f"n={expected_count} validation instances/model", transform=ax.transAxes, va="top", fontsize=9)
    fig.suptitle("Elastic VTI PINO: 30-instance distributions (pretrain vs fine-tune)", fontsize=15, color=PALETTE["blue"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--distribution-metrics",
        type=Path,
        default=ROOT / "reports/no_asvgd_group_meeting_20260710/elastic_vti_30instance_metrics.json",
    )
    parser.add_argument(
        "--marmousi-pretrain-metrics",
        type=Path,
        default=ROOT / "reports/no_asvgd_group_meeting_20260710/elastic_vti_marmousi_pretrain_30instance_metrics.json",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))["models"]
    manifest: dict[str, Any] = {"models": {}, "runtime": str(args.runtime)}
    overview_metrics: dict[str, dict[str, float]] = {}
    for name in MODEL_SPECS:
        case = load_case(name, device=args.device)
        figures = {
            "ux": args.output_dir / f"elastic_{name}_ux.png",
            "uz": args.output_dir / f"elastic_{name}_uz.png",
            "receivers": args.output_dir / f"elastic_{name}_receivers_runtime.png",
        }
        plot_ux_context(case, figures["ux"])
        plot_uz_comparison(case, figures["uz"])
        plot_receiver_runtime(case, runtime[name], figures["receivers"])
        overview_metrics[case["title"]] = case["metrics"]
        manifest["models"][name] = {
            "sample_index": case["sample_index"],
            "checkpoint": case["checkpoint"],
            "metrics": case["metrics"],
            "sample_metrics": case["sample_metrics"],
            "figures": {key: str(path) for key, path in figures.items()},
        }
    marm_pre = summarize_evaluation_metrics(MODEL_SPECS["marmousi"]["pretrain_metrics"])
    manifest["marmousi_pretrain_metrics"] = marm_pre
    marm_ft = overview_metrics.pop("Marmousi")
    overview_metrics["Marmousi pre"] = marm_pre
    overview_metrics["Marmousi FT"] = marm_ft
    distribution = json.loads(args.distribution_metrics.read_text(encoding="utf-8"))
    marmousi_pretrain = json.loads(args.marmousi_pretrain_metrics.read_text(encoding="utf-8"))
    validate_distribution_metrics(distribution, expected_count=30)
    overview = args.output_dir / "elastic_cross_model_boxplot.png"
    plot_distribution_boxplots(distribution, overview, marmousi_pretrain=marmousi_pretrain)
    manifest["overview"] = str(overview)
    manifest["distribution_metrics"] = str(args.distribution_metrics)
    manifest["marmousi_pretrain_distribution_metrics"] = str(args.marmousi_pretrain_metrics)
    validate_manifest(manifest)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
