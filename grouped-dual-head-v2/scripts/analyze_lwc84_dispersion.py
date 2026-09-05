#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COEFFICIENTS = np.asarray(
    [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560],
    dtype=np.float64,
)
OFFSETS = np.arange(-4, 5, dtype=np.float64)
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]


def spatial_symbol(theta: np.ndarray) -> np.ndarray:
    return np.sum(COEFFICIENTS * np.cos(theta[..., None] * OFFSETS), axis=-1)


def phase_velocity_ratio(*, frequency_hz: float, angle_deg: float, dx_m: float, velocity_mps: float, dt_s: float) -> tuple[float, float, float, bool]:
    omega = 2.0 * math.pi * float(frequency_hz)
    wavenumber = omega / float(velocity_mps)
    angle = math.radians(float(angle_deg))
    theta_x = wavenumber * math.cos(angle) * float(dx_m)
    theta_z = wavenumber * math.sin(angle) * float(dx_m)
    lambda_h = float(velocity_mps) ** 2 * (
        float(spatial_symbol(np.asarray(theta_x))) + float(spatial_symbol(np.asarray(theta_z)))
    ) / float(dx_m) ** 2
    q = -float(dt_s) ** 2 * lambda_h
    cosine = 1.0 - 0.5 * q + q**2 / 24.0
    stable = bool(-1.0 <= cosine <= 1.0)
    omega_numeric = math.acos(max(-1.0, min(1.0, cosine))) / float(dt_s)
    return omega_numeric / omega, q, float(velocity_mps) / (float(frequency_hz) * float(dx_m)), stable


def ricker_energy_bandwidth_95(f0_hz: float) -> tuple[float, float]:
    frequency = np.linspace(0.0, 5.0 * float(f0_hz), 200_001)
    energy = frequency**4 * np.exp(-2.0 * (frequency / float(f0_hz)) ** 2)
    cumulative = np.cumsum((energy[:-1] + energy[1:]) * 0.5 * np.diff(frequency))
    cumulative = np.concatenate([[0.0], cumulative])
    cumulative /= cumulative[-1]
    return (
        float(np.interp(0.025, cumulative, frequency)),
        float(np.interp(0.975, cumulative, frequency)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze eighth-space/fourth-time LWC-84 numerical dispersion.")
    parser.add_argument("--output", default="artifacts/lwc84_dispersion")
    parser.add_argument("--velocity-mps", type=float, default=1500.0)
    parser.add_argument("--dt-s", type=float, default=0.00001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    spacings = [5.0, 10.0, 20.0]
    frequencies = [10.0, 15.0, 25.0, 30.0]
    angles = np.linspace(0.0, 90.0, 91)
    rows: list[dict[str, float | bool]] = []
    for dx in spacings:
        for frequency in frequencies:
            for angle in angles:
                ratio, q, ppw, stable = phase_velocity_ratio(
                    frequency_hz=frequency,
                    angle_deg=float(angle),
                    dx_m=dx,
                    velocity_mps=args.velocity_mps,
                    dt_s=args.dt_s,
                )
                rows.append(
                    {
                        "dx_m": dx,
                        "frequency_hz": frequency,
                        "angle_deg": float(angle),
                        "velocity_mps": float(args.velocity_mps),
                        "dt_s": float(args.dt_s),
                        "courant_axis": float(args.velocity_mps * args.dt_s / dx),
                        "points_per_wavelength": ppw,
                        "lwc_q": q,
                        "phase_velocity_ratio": ratio,
                        "phase_error_percent": 100.0 * (ratio - 1.0),
                        "stable": stable,
                    }
                )
    csv_path = output / "lwc84_dispersion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bandwidth = {
        f"{frequency:g}": {
            "lower_hz": ricker_energy_bandwidth_95(frequency)[0],
            "upper_hz": ricker_energy_bandwidth_95(frequency)[1],
        }
        for frequency in frequencies
    }
    maxima: dict[str, dict[str, float]] = {}
    for dx in spacings:
        maxima[f"dx_{dx:g}m"] = {}
        for frequency in frequencies:
            selected = [
                abs(float(row["phase_error_percent"]))
                for row in rows
                if row["dx_m"] == dx and row["frequency_hz"] == frequency
            ]
            maxima[f"dx_{dx:g}m"][f"f_{frequency:g}hz"] = max(selected)
    summary = {
        "method": "LWC-84 fourth-order time with exact eighth-order centered spatial symbol",
        "velocity_mps": float(args.velocity_mps),
        "dt_s": float(args.dt_s),
        "frequencies_hz": frequencies,
        "angles_deg": [0.0, 90.0, 1.0],
        "ricker_95_percent_energy_bandwidth_hz": bandwidth,
        "maximum_absolute_phase_error_percent": maxima,
        "claim": "Numerical dispersion is quantified, not claimed to be eliminated.",
    }
    json_path = output / "lwc84_dispersion_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), sharey=False)
    for axis, dx in zip(axes, spacings, strict=True):
        for index, frequency in enumerate(frequencies):
            selected = [row for row in rows if row["dx_m"] == dx and row["frequency_hz"] == frequency]
            axis.plot(
                [row["angle_deg"] for row in selected],
                [row["phase_error_percent"] for row in selected],
                color=COLORS[index],
                marker=MARKERS[index],
                markevery=15,
                markersize=3,
                linewidth=1.4,
                label=f"{frequency:g} Hz",
            )
        axis.axhline(0.0, color="#333333", linewidth=0.7)
        axis.set_title(f"$\\Delta x=\\Delta z={dx:g}$ m")
        axis.set_xlabel("Propagation angle (deg)")
        axis.set_xticks([0, 30, 60, 90])
        axis.set_ylabel("Error (%)")
    axes[-1].legend(loc="best")
    fig.tight_layout(w_pad=0.8)
    pdf_path = output / "fig_lwc84_dispersion.pdf"
    png_path = output / "fig_lwc84_dispersion.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "pdf": str(pdf_path), "png": str(png_path), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
