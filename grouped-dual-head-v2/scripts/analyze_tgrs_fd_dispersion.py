#!/usr/bin/env python
"""Analytic FD2/FD4/LWC-84 numerical-dispersion curves for the manuscript.

Emits, under ``--output-dir``:
  * ``phase_velocity.csv``          -- one row per (scheme, ppw, angle);
  * ``phase_velocity_summary.json`` -- per-scheme worst-case phase error + claim string;
  * ``fig3a_phase_velocity.{pdf,png}`` -- phase error vs propagation angle at fixed ppw,
    one panel per scheme, contrasting the three discretization orders.

Analytic only -- no solver, no GPU.  Frames *why* an under-resolved traditional solve
disperses, independent of any neural result.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tgrs_dclp_no.dispersion import phase_velocity_ratio  # noqa: E402


SCHEMES = ("fd2", "fd4", "lwc84")
SCHEME_LABEL = {"fd2": "2nd-order (FD2)", "fd4": "4th-order (FD4)", "lwc84": "LWC-84 (8th/4th)"}
# Per-scheme axis Courant number near each scheme's stable operating point.
COURANT = {"fd2": 0.25, "fd4": 0.20, "lwc84": 0.10}
COLORS = {"fd2": "#D55E00", "fd4": "#56B4E9", "lwc84": "#009E73"}
MARKERS = {"fd2": "o", "fd4": "s", "lwc84": "^"}
# Points-per-wavelength values highlighted in the summary / figure panels.
PANEL_PPW = (4.0, 6.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analytic FD2/FD4/LWC-84 dispersion curves.")
    parser.add_argument("--output-dir", default="artifacts/tgrs_dclp_no/dispersion/analytic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ppw_values = np.linspace(4.0, 30.0, 261)
    angles = np.arange(0.0, 91.0, 5.0)

    rows: list[dict[str, float | str]] = []
    for scheme in SCHEMES:
        courant = COURANT[scheme]
        for ppw in ppw_values:
            for angle in angles:
                try:
                    ratio = phase_velocity_ratio(
                        scheme=scheme,
                        points_per_wavelength=float(ppw),
                        angle_deg=float(angle),
                        courant_axis=courant,
                    )
                    stable = True
                except ValueError:
                    ratio = float("nan")
                    stable = False
                rows.append(
                    {
                        "scheme": scheme,
                        "courant_axis": float(courant),
                        "points_per_wavelength": float(ppw),
                        "angle_deg": float(angle),
                        "phase_velocity_ratio": float(ratio),
                        "phase_error_percent": float(100.0 * (ratio - 1.0)),
                        "stable": bool(stable),
                    }
                )

    csv_path = output / "phase_velocity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Worst-case (over angle) absolute phase error at the highlighted ppw values.
    worst: dict[str, dict[str, float]] = {}
    for scheme in SCHEMES:
        worst[scheme] = {}
        for ppw in PANEL_PPW:
            selected = [
                abs(float(row["phase_error_percent"]))
                for row in rows
                if row["scheme"] == scheme
                and abs(float(row["points_per_wavelength"]) - ppw) < 0.06
                and bool(row["stable"])
            ]
            worst[scheme][f"ppw_{ppw:g}"] = max(selected) if selected else float("nan")

    summary = {
        "schemes": {s: {"label": SCHEME_LABEL[s], "courant_axis": COURANT[s]} for s in SCHEMES},
        "highlighted_points_per_wavelength": list(PANEL_PPW),
        "maximum_absolute_phase_error_percent": worst,
        "claim": "Finite-difference dispersion is quantified; no scheme is claimed dispersion free.",
    }
    json_path = output / "phase_velocity_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9,
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Main-text figure: the most under-resolved registered regime carries the
    # scientific claim.  Author directly at IEEE column width to preserve type size.
    fig, axis = plt.subplots(figsize=(3.45, 2.35))
    ppw = PANEL_PPW[0]
    for scheme in SCHEMES:
        selected = [
            row
            for row in rows
            if row["scheme"] == scheme
            and abs(float(row["points_per_wavelength"]) - ppw) < 0.06
            and bool(row["stable"])
        ]
        selected.sort(key=lambda r: float(r["angle_deg"]))
        axis.plot(
            [float(r["angle_deg"]) for r in selected],
            [float(r["phase_error_percent"]) for r in selected],
            color=COLORS[scheme], marker=MARKERS[scheme], markevery=3,
            markersize=3.2, linewidth=1.45, label=SCHEME_LABEL[scheme],
        )
    axis.axhline(0.0, color="#333333", linewidth=0.7)
    axis.set_title("4 points per wavelength")
    axis.set_xlabel("Propagation angle (deg)")
    axis.set_ylabel("Phase-velocity error (%)")
    axis.set_xticks([0, 30, 60, 90])
    axis.set_xlim(-4, 113)
    axis.set_ylim(-10.35, 0.48)
    # Direct labels keep every angular segment visible at one-column width.
    for label, y_value, scheme in (
        ("FD2", -9.50, "fd2"),
        ("FD4", -2.38, "fd4"),
        ("LWC-84", -0.31, "lwc84"),
    ):
        axis.text(94.0, y_value, label, color=COLORS[scheme], fontsize=6.8,
                  fontweight="bold", va="center")
    fig.tight_layout(pad=0.35)
    pdf_path = output / "fig3a_phase_velocity.pdf"
    png_path = output / "fig3a_phase_velocity.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    # Full three-regime comparison retained as a supplementary display.
    fig, axes = plt.subplots(1, len(PANEL_PPW), figsize=(7.0, 2.45), sharey=False)
    for axis, ppw in zip(axes, PANEL_PPW, strict=True):
        for scheme in SCHEMES:
            selected = [
                row
                for row in rows
                if row["scheme"] == scheme
                and abs(float(row["points_per_wavelength"]) - ppw) < 0.06
                and bool(row["stable"])
            ]
            selected.sort(key=lambda r: float(r["angle_deg"]))
            axis.plot(
                [float(r["angle_deg"]) for r in selected],
                [float(r["phase_error_percent"]) for r in selected],
                color=COLORS[scheme],
                marker=MARKERS[scheme],
                markevery=3,
                markersize=3,
                linewidth=1.4,
                label=SCHEME_LABEL[scheme],
            )
        axis.axhline(0.0, color="#333333", linewidth=0.7)
        axis.set_title(f"{ppw:g} points / wavelength")
        axis.set_xlabel("Propagation angle (deg)")
        axis.set_xticks([0, 30, 60, 90])
        axis.set_ylabel("Phase-velocity error (%)")
    axes[-1].legend(loc="best")
    fig.tight_layout(w_pad=0.8)
    supplement_pdf = output / "fig3a_phase_velocity_supplement.pdf"
    supplement_png = output / "fig3a_phase_velocity_supplement.png"
    fig.savefig(supplement_pdf)
    fig.savefig(supplement_png)
    plt.close(fig)

    print(
        json.dumps(
            {"csv": str(csv_path), "json": str(json_path), "pdf": str(pdf_path),
             "png": str(png_path), "supplement_pdf": str(supplement_pdf),
             "supplement_png": str(supplement_png), **summary},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
