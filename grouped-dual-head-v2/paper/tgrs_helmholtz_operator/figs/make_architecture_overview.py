#!/usr/bin/env python3
"""Render the deployed r5b architecture as an editable vector schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent

COLORS = {
    "data_fill": "#F4F4F4",
    "data_edge": "#4D4D4D",
    "learn_fill": "#E8F1FA",
    "learn_edge": "#3B6EA8",
    "fixed_fill": "#E8F5F0",
    "fixed_edge": "#2E7D66",
    "physics_fill": "#FFF0DE",
    "physics_edge": "#B56A1B",
    "text": "#222222",
    "primary": "#303030",
    "secondary": "#777777",
}


def node(ax, xy, wh, text, kind="learn", *, fontsize=8.2, lw=1.25, zorder=3):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.016,rounding_size=0.035",
        facecolor=COLORS[f"{kind}_fill"], edgecolor=COLORS[f"{kind}_edge"],
        linewidth=(1.35 if kind == "learn" else lw),
        linestyle={"learn": "-", "fixed": "--", "physics": "-.", "data": "-"}[kind],
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["text"], zorder=zorder + 1,
            linespacing=1.15)
    return (x, y, w, h)


def port(box, side, frac=0.5):
    x, y, w, h = box
    return {
        "left": (x, y + h * frac), "right": (x + w, y + h * frac),
        "top": (x + w * frac, y + h), "bottom": (x + w * frac, y),
    }[side]


def arrow(ax, start, end, *, color=None, lw=1.25, rad=0.0, style="-"):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
        color=color or COLORS["primary"], linestyle=style,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2.5, shrinkB=2.5, zorder=2,
    )
    ax.add_patch(patch)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    # Match IEEE's two-column text width so typography is authored at final size.
    fig, ax = plt.subplots(figsize=(7.15, 3.30))
    ax.set_xlim(0, 16.25)
    ax.set_ylim(0, 6.25)
    ax.axis("off")

    ax.text(0.20, 5.94, "Record-level encoding  (computed once)", fontsize=9.2,
            weight="bold", color=COLORS["learn_edge"])
    ax.text(0.20, 3.18, "Per-time query", fontsize=9.2,
            weight="bold", color=COLORS["physics_edge"])
    ax.plot([0.18, 16.08], [3.45, 3.45], color="#C7C7C7", lw=0.75)

    record = node(ax, (0.25, 4.34), (1.55, 1.02),
                  "Record inputs\n$v(x,z)$\n$q=(x_s,z_s,f_0,t_0,A)$", "data", fontsize=7.0)
    encoders = node(ax, (2.20, 4.66), (1.88, 0.76),
                    "Complex FNO\n+ source encoder", "learn", fontsize=7.2)
    geometry = node(ax, (2.20, 3.70), (1.88, 0.68),
                    "Travel geometry\n$\\tau_q(x,z),\\;\\nabla\\tau_q$", "fixed", fontsize=7.2)
    shared = node(ax, (4.60, 4.66), (1.88, 0.76),
                  "Shared record state\n$S_\\theta(v,q)$", "learn", fontsize=7.2)
    anchors = node(ax, (7.00, 4.66), (1.78, 0.76),
                   "Rank-32 anchors\n$L_m(x,z)$", "learn", fontsize=7.2)

    time = node(ax, (0.25, 1.30), (1.55, 0.72), "Saved time\n$t$", "data", fontsize=7.3)
    time_features = node(ax, (2.20, 1.30), (1.88, 0.72),
                         "Query features\n$t$, phase, index", "fixed", fontsize=7.2)
    global_path = node(ax, (4.60, 2.30), (1.88, 0.72),
                       "Global MIONet\ncoarse field", "learn", fontsize=7.2)
    local_path = node(ax, (4.60, 1.10), (2.00, 0.78),
                      "Local synthesis\n$\\ell_\\theta(x,t)+b_\\theta(x,t)$", "learn", fontsize=7.2)
    warp = node(ax, (7.10, 1.10), (1.76, 0.78),
                "Characteristic warp\nalong $\\nabla\\tau_q$", "physics", fontsize=7.1)
    causal = node(ax, (9.32, 1.10), (1.42, 0.78),
                  "Arrival gate\n$\\times\,c_q(x,t)$", "fixed", fontsize=7.1)
    coarse_plus = node(ax, (11.20, 2.06), (0.46, 0.46), "+", "data", fontsize=13)
    decoder = node(ax, (12.05, 1.92), (1.66, 0.74),
                   "Complex spectral\nrefiner $\\times 12$", "learn", fontsize=7.1)
    surface = node(ax, (14.08, 1.92), (0.82, 0.74),
                   "$\\times\,g_{\\mathrm{fs}}(z)$", "fixed", fontsize=7.0)
    output = node(ax, (15.25, 1.92), (0.78, 0.74),
                  "Output\n$\\widehat p$", "data", fontsize=7.1)

    arrow(ax, port(record, "right", 0.72), port(encoders, "left", 0.50))
    arrow(ax, port(record, "right", 0.28), port(geometry, "left", 0.50),
          color=COLORS["secondary"], lw=0.9)
    arrow(ax, port(encoders, "right"), port(shared, "left"))
    arrow(ax, port(shared, "right"), port(anchors, "left"))

    arrow(ax, port(time, "right"), port(time_features, "left"))
    arrow(ax, port(geometry, "bottom"), port(time_features, "top"),
          color=COLORS["secondary"], lw=0.9)
    arrow(ax, port(shared, "bottom", 0.30), port(global_path, "top", 0.50),
          color=COLORS["secondary"], lw=0.9, rad=0.02)
    arrow(ax, port(shared, "bottom", 0.70), port(local_path, "top", 0.50),
          color=COLORS["secondary"], lw=0.9, rad=0.06)
    arrow(ax, port(time_features, "right", 0.78), port(global_path, "left", 0.55),
          color=COLORS["secondary"], lw=0.9, rad=-0.08)
    arrow(ax, port(time_features, "right", 0.46), port(local_path, "left", 0.55),
          color=COLORS["secondary"], lw=0.9, rad=0.03)
    arrow(ax, port(anchors, "bottom", 0.50), port(local_path, "top", 0.75),
          color=COLORS["secondary"], lw=0.9, rad=0.08)
    arrow(ax, port(local_path, "right"), port(warp, "left"))
    arrow(ax, port(warp, "right"), port(causal, "left"))
    arrow(ax, port(causal, "right"), port(coarse_plus, "bottom", 0.45), rad=-0.11)
    arrow(ax, port(global_path, "right"), port(coarse_plus, "left", 0.62))
    arrow(ax, port(coarse_plus, "right"), port(decoder, "left"))
    arrow(ax, port(decoder, "right"), port(surface, "left"))
    arrow(ax, port(surface, "right"), port(output, "left"))

    # Visual key; shape and border preserve meaning in grayscale.
    key_y = 5.05
    entries = (("learned", "learn", 9.20), ("fixed", "fixed", 10.85),
               ("transport", "physics", 12.25), ("data", "data", 14.42))
    for label, kind, x in entries:
        ax.add_patch(Rectangle((x, key_y - 0.05), 0.24, 0.18,
                               facecolor=COLORS[f"{kind}_fill"],
                               edgecolor=COLORS[f"{kind}_edge"], linewidth=1.0,
                               linestyle={"learn": "-", "fixed": "--", "physics": "-.", "data": "-"}[kind]))
        ax.text(x + 0.32, key_y + 0.04, label, fontsize=6.8, va="center")

    ax.text(16.00, 4.10, "No target wavefield or background solve",
            ha="right", va="center", fontsize=6.7, color="#555555", style="italic")
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.98)
    fig.savefig(OUT / "architecture_overview.pdf", bbox_inches="tight")
    fig.savefig(OUT / "architecture_overview.svg", bbox_inches="tight")
    fig.savefig(OUT / "architecture_overview.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
