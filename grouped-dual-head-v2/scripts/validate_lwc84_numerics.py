#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_ROOT / "src"), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver


DT = 0.00001
DT_OUT = 0.00125


def _simulate(
    *,
    nx: int,
    nz: int,
    velocity: np.ndarray,
    source_x_m: float,
    source_z_m: float,
    frequency_hz: float,
    t_end_s: float,
    device: str,
    npml: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = AcousticGrid(
        nx=nx,
        nz=nz,
        dx_m=5.0,
        dz_m=5.0,
        lx_m=(nx - 1) * 5.0,
        lz_m=(nz - 1) * 5.0,
        centering="node",
    )
    boundaries = BoundaryConfig(npml=npml, cpml_target_reflection=1.0e-8, cpml_polynomial_order=3)
    times = np.arange(int(round(t_end_s / DT_OUT)) + 1, dtype=np.float64) * DT_OUT
    solver = LWC84CPMLSolver(
        grid=grid,
        boundaries=boundaries,
        dt_s=DT,
        output_times_s=times,
        c_ref_mps=float(np.max(velocity)),
        device=device,
        dtype=torch.float32 if device == "cuda" else torch.float64,
        kappa_max=3.0,
        minimum_frequency_hz=8.0,
    )
    result = solver.simulate(
        velocity,
        source_x_m=source_x_m,
        source_z_m=source_z_m,
        source_f0_hz=frequency_hz,
    )
    return result.wavefield[0], times


def _trace(field: np.ndarray, *, x_m: float, z_m: float) -> np.ndarray:
    return field[:, int(round(z_m / 10.0)), int(round(x_m / 10.0))]


def _window(times: np.ndarray, center: float, half_width: float) -> np.ndarray:
    return np.abs(times - float(center)) <= float(half_width)


def cpml_reflection_case(frequency_hz: float, *, oblique: bool, device: str) -> dict[str, float | str]:
    nx, nz_small, nz_reference = 81, 81, 161
    c = 2000.0
    source_x, source_z = ((100.0, 75.0) if oblique else (200.0, 75.0))
    receiver_x, receiver_z = ((300.0, 250.0) if oblique else (200.0, 250.0))
    t_end = 0.65 if frequency_hz <= 10.0 else 0.55
    small, times = _simulate(
        nx=nx,
        nz=nz_small,
        velocity=np.full((nz_small, nx), c, dtype=np.float32),
        source_x_m=source_x,
        source_z_m=source_z,
        frequency_hz=frequency_hz,
        t_end_s=t_end,
        device=device,
        npml=16,
    )
    reference, _ = _simulate(
        nx=nx,
        nz=nz_reference,
        velocity=np.full((nz_reference, nx), c, dtype=np.float32),
        source_x_m=source_x,
        source_z_m=source_z,
        frequency_hz=frequency_hz,
        t_end_s=t_end,
        device=device,
        npml=16,
    )
    small_trace = _trace(small, x_m=receiver_x, z_m=receiver_z)
    reference_trace = _trace(reference, x_m=receiver_x, z_m=receiver_z)
    direct_distance = math.hypot(receiver_x - source_x, receiver_z - source_z)
    mirrored_receiver_z = 2.0 * ((nz_small - 1) * 5.0) - receiver_z
    reflection_distance = math.hypot(receiver_x - source_x, mirrored_receiver_z - source_z)
    t0 = 1.5 / frequency_hz
    direct_window = _window(times, t0 + direct_distance / c, max(0.025, 0.55 / frequency_hz))
    reflection_window = _window(times, t0 + reflection_distance / c, max(0.035, 0.75 / frequency_hz))
    direct_peak = float(np.max(np.abs(reference_trace[direct_window])))
    reflected_peak = float(np.max(np.abs((small_trace - reference_trace)[reflection_window])))
    ratio = reflected_peak / max(direct_peak, np.finfo(np.float64).tiny)
    return {
        "incidence": "oblique" if oblique else "vertical",
        "frequency_hz": float(frequency_hz),
        "direct_peak": direct_peak,
        "reflected_difference_peak": reflected_peak,
        "reflection_ratio": ratio,
        "reflection_db": 20.0 * math.log10(max(ratio, np.finfo(np.float64).tiny)),
    }


def free_surface_polarity(device: str) -> dict[str, float | bool]:
    nx, nz, c, frequency = 201, 161, 2000.0, 15.0
    field, times = _simulate(
        nx=nx,
        nz=nz,
        velocity=np.full((nz, nx), c, dtype=np.float32),
        source_x_m=300.0,
        source_z_m=200.0,
        frequency_hz=frequency,
        t_end_s=0.38,
        device=device,
        npml=20,
    )
    reflected = _trace(field, x_m=300.0, z_m=200.0)
    equal_distance_direct = _trace(field, x_m=700.0, z_m=200.0)
    expected = 1.5 / frequency + 400.0 / c
    selected = _window(times, expected, 0.045)
    a, b = reflected[selected], equal_distance_direct[selected]
    correlation = float(np.corrcoef(a, b)[0, 1])
    amplitude_ratio = float(np.max(np.abs(a)) / np.max(np.abs(b)))
    top_error = float(np.max(np.abs(field[:, 0, :])))
    return {
        "top_pressure_max_abs": top_error,
        "reflected_vs_equal_distance_direct_correlation": correlation,
        "reflection_amplitude_ratio": amplitude_ratio,
        "polarity_inverted": bool(correlation < -0.9),
    }


def uniform_wavefront(device: str) -> dict[str, float]:
    nx = nz = 161
    c, frequency = 2000.0, 15.0
    field, times = _simulate(
        nx=nx,
        nz=nz,
        velocity=np.full((nz, nx), c, dtype=np.float32),
        source_x_m=400.0,
        source_z_m=400.0,
        frequency_hz=frequency,
        t_end_s=0.23,
        device=device,
        npml=20,
    )
    target_time = 0.20
    snapshot = np.abs(field[int(np.argmin(np.abs(times - target_time)))])
    coordinates = np.arange((nx + 1) // 2, dtype=np.float64) * 10.0
    zz, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.hypot(xx - 400.0, zz - 400.0)
    radial_bins = np.arange(0.0, 401.0, 10.0)
    energy = np.asarray(
        [snapshot[(radius >= left) & (radius < left + 10.0)].mean() for left in radial_bins]
    )
    measured = float(radial_bins[int(np.argmax(energy))] + 5.0)
    theoretical = c * (target_time - 1.5 / frequency)
    return {
        "target_time_s": target_time,
        "theoretical_peak_radius_m": theoretical,
        "measured_radial_energy_peak_m": measured,
        "absolute_error_m": abs(measured - theoretical),
    }


def layered_arrivals(device: str) -> dict[str, float | bool]:
    nx = nz = 201
    interface, upper, lower, frequency = 500.0, 3000.0, 5000.0, 15.0
    velocity = np.full((nz, nx), upper, dtype=np.float32)
    velocity[int(interface / 5.0) :, :] = lower
    field, times = _simulate(
        nx=nx,
        nz=nz,
        velocity=velocity,
        source_x_m=500.0,
        source_z_m=150.0,
        frequency_hz=frequency,
        t_end_s=0.38,
        device=device,
        npml=20,
    )
    reflected_trace = _trace(field, x_m=500.0, z_m=300.0)
    transmitted_trace = _trace(field, x_m=500.0, z_m=700.0)
    t0 = 1.5 / frequency
    expected_reflected = t0 + ((interface - 150.0) + (interface - 300.0)) / upper
    expected_transmitted = t0 + (interface - 150.0) / upper + (700.0 - interface) / lower

    def observed(trace: np.ndarray, expected: float) -> float:
        selected = _window(times, expected, 0.035)
        indices = np.flatnonzero(selected)
        return float(times[indices[int(np.argmax(np.abs(trace[selected])))]] )

    observed_reflected = observed(reflected_trace, expected_reflected)
    observed_transmitted = observed(transmitted_trace, expected_transmitted)
    return {
        "expected_reflected_arrival_s": expected_reflected,
        "observed_reflected_peak_s": observed_reflected,
        "reflected_error_s": abs(observed_reflected - expected_reflected),
        "expected_transmitted_arrival_s": expected_transmitted,
        "observed_transmitted_peak_s": observed_transmitted,
        "transmitted_error_s": abs(observed_transmitted - expected_transmitted),
        "arrival_gate_passed": bool(
            abs(observed_reflected - expected_reflected) <= 0.035
            and abs(observed_transmitted - expected_transmitted) <= 0.035
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run numerical LWC84/free-surface/CPML acceptance cases.")
    parser.add_argument("--output", default="artifacts/lwc84_numerical_validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cpml = [
        cpml_reflection_case(frequency, oblique=oblique, device=args.device)
        for frequency in (10.0, 15.0, 25.0)
        for oblique in (False, True)
    ]
    payload = {
        "device": args.device,
        "dt_s": DT,
        "dt_out_s": DT_OUT,
        "cpml": cpml,
        "free_surface": free_surface_polarity(args.device),
        "uniform_wavefront": uniform_wavefront(args.device),
        "layered_arrivals": layered_arrivals(args.device),
    }
    payload["passed"] = bool(
        all(float(row["reflection_ratio"]) < 1.0e-2 for row in cpml)
        and payload["free_surface"]["top_pressure_max_abs"] <= 1.0e-7
        and payload["free_surface"]["polarity_inverted"]
        and payload["uniform_wavefront"]["absolute_error_m"] <= 20.0
        and payload["layered_arrivals"]["arrival_gate_passed"]
    )
    json_path = output / "lwc84_numerical_validation.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output / "cpml_reflection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cpml[0]))
        writer.writeheader()
        writer.writerows(cpml)
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
         "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"}
    )
    fig, axis = plt.subplots(figsize=(3.4, 2.5))
    for incidence, color, marker in (("vertical", "#0072B2", "o"), ("oblique", "#D55E00", "s")):
        selected = [row for row in cpml if row["incidence"] == incidence]
        axis.plot([row["frequency_hz"] for row in selected], [row["reflection_db"] for row in selected],
                  color=color, marker=marker, linewidth=1.6, label=incidence.capitalize())
    axis.axhline(-40.0, color="#333333", linestyle="--", linewidth=0.8, label="Acceptance (−40 dB)")
    axis.set_xlabel("Ricker dominant frequency (Hz)")
    axis.set_ylabel("Measured CPML reflection (dB)")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "fig_cpml_reflection.pdf")
    fig.savefig(output / "fig_cpml_reflection.png")
    plt.close(fig)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), **payload}, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
