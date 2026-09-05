#!/usr/bin/env python3
"""Render the sealed 240-case source-position distribution at IEEE column width."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE.parent
    / "experiment_evidence_bundle_20260813"
    / "05_relative_error"
    / "latest_fixed19hz_all_240_records.csv"
)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    groups = (
        ("All\n($n=240$)", rows),
        ("Interpolation\n($n=150$)", [r for r in rows if r["role"] == "interpolation"]),
        ("Outside range\n($n=90$)", [r for r in rows if r["role"] == "outside_train_position_range"]),
    )
    values = [np.asarray([float(r["record_relative_l2"]) for r in selected]) for _, selected in groups]
    colors = ("#99AABB", "#5185C0", "#E99D4E")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axis = plt.subplots(figsize=(3.45, 2.55))
    boxes = axis.boxplot(
        values, labels=[label for label, _ in groups], widths=0.55,
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.15},
        whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
    )
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set(facecolor=color, alpha=0.72, linewidth=0.8)
    rng = np.random.default_rng(20260813)
    for index, (array, color) in enumerate(zip(values, colors, strict=True), start=1):
        axis.scatter(
            np.full(len(array), index) + rng.uniform(-0.14, 0.14, len(array)),
            array, s=4.5, alpha=0.43, color=color, edgecolor="none",
        )
    axis.set_ylabel("Complete-transient relative $L_2$")
    axis.grid(axis="y", color="#B8B8B8", alpha=0.38, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=6.6)
    fig.tight_layout(pad=0.35)
    fig.savefig(HERE / "fixed19hz_relative_error_boxplot.pdf")
    fig.savefig(HERE / "fixed19hz_relative_error_boxplot.svg")
    fig.savefig(HERE / "fixed19hz_relative_error_boxplot.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
