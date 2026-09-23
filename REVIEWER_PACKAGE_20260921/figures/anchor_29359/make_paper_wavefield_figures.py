#!/usr/bin/env python3
"""Wavefield-snapshot and receiver-waveform figures for the anchor-29359 paper.

Every number printed on these figures is produced by importing
``research/l1_gate_measure_20260913/measure_gates.py`` -- the same module, at the
same sha256 recorded in ``panel_29359_META.json``, that produced the paper's
Table 1 -- rather than by a second implementation of the metric.  The recomputed
``future_relative_l2`` and ``time_bands`` for each plotted record are asserted
against the published panel rows before anything is drawn, so a figure can only
be written if it agrees with the table it illustrates.

Record selection is a rule fixed before the measurement, not a choice made after
looking at the pictures: for each family, the record whose full-future relative
L2 is the lower median of that family's frozen list.

Red lines honoured: train split only (validation and test_id are never opened),
checkpoints are read-only and sha-asserted, no predicted wavefield is persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REL = Path("/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/release_ic8_40m_20260910")
GATE_DIR = REL / "research/l1_gate_measure_20260913"
# The anchor's own frozen code tree, bit-identical to formal/run_identity.json's
# 38 code_files.  The live release tree has drifted since the anchor was trained
# (its config.py no longer parses this run's lwc84 loss keys), so the snapshot is
# the only tree that reproduces the checkpoint's function.  measure_gates.py
# hard-codes the release tree on sys.path, so the snapshot modules are imported
# first and land in sys.modules before measure_gates can resolve its own imports.
SNAPSHOT = REL / "research/l2_lwc84_ddp4_20260915_v1/snapshot"
sys.path.insert(0, str(SNAPSHOT))

from grouped_ufno_mionet_v3.config import V3Config  # noqa: E402
from grouped_ufno_mionet_v3.data.index import build_manifest  # noqa: E402
from grouped_ufno_mionet_v3.data.pilot import PilotBatchDataset  # noqa: E402,F401
from grouped_ufno_mionet_v3.data.records import V3WavefieldDataset  # noqa: E402
from grouped_ufno_mionet_v3.evaluation import assert_truth_shape, pixel_extent  # noqa: E402
from scripts.train_grouped_v3 import build_model  # noqa: E402,F401
from scripts.train_grouped_v3_pilot import load_normalizer  # noqa: E402
from scripts.train_grouped_v3_ic8_fullfield_ddp import _ANCHOR_PARAM_KEYS  # noqa: E402,F401

for _module, _name in ((V3Config, "config"), (V3WavefieldDataset, "records")):
    _origin = Path(sys.modules[_module.__module__].__file__).resolve()
    if SNAPSHOT.resolve() not in _origin.parents:
        raise SystemExit(f"{_name} was imported from {_origin}, not the anchor snapshot")

sys.path.insert(0, str(GATE_DIR))
import measure_gates as mg  # noqa: E402  (snapshot modules are already bound)

CONFIG = REL / "research/l2_lwc84_ddp4_20260915_v1/train.yaml"
CHECKPOINT = Path(
    "/root/autodl-tmp/staging/l2_lwc84_ddp4_20260915_v1/formal/checkpoints/checkpoint_step_00029359.pt")
CHECKPOINT_SHA = "6a45c074bf0fcac335b6ac69c0bc1527d7824a25106e34be041d81b7f7c03f24"
PANEL_ROWS = Path(
    "/root/autodl-tmp/staging/coda_panel_eval_20260916/ckpt_29359/panel_29359_ROWS.jsonl")
PANEL_META = PANEL_ROWS.with_name("panel_29359_META.json")

FAMILIES = ("uniform", "layered", "marmousi")
BAND_NAMES = ("B1 (early)", "B2 (middle)", "B3 (late)")
# Receiver line: 100 m below the node-centred free surface (p(z=0)=0 identically,
# so a receiver at z=0 would be a plot of zero), mid-domain in x.
RECEIVER_Z_M = 100.0
RECEIVER_X_M = 1000.0
# Families whose late window the paper places in the near-zero-denominator class and
# gates no claim on ("One conditioning result constrains what may be reported": the
# homogeneous coda has left the domain through the CPML, so its relative L2 divides by
# near machine zero).  Named by family because the paper states the caveat per family,
# and flagged on the figure so a reader cannot mistake the panel for a physical result.
ILL_CONDITIONED_LATE_FAMILIES = ("uniform",)
# Both figures are authored at exactly the width they are printed at -- \textwidth is
# 6.27 in for an a4 page with 1 in margins -- so LaTeX applies no scaling and a point
# specified here is that many points on the page.  Authoring wider and letting LaTeX
# shrink the result is what made an earlier draft of these panels unreadable; K is kept
# so every size stays explicit about which space it lives in.
PRINT_WIDTH_IN = 6.27
AUTHORED_WIDTH_IN = PRINT_WIDTH_IN
K = AUTHORED_WIDTH_IN / PRINT_WIDTH_IN
# Agreement tolerance against the published panel.  The panel ran on cuda:1 of a
# four-GPU box and this script runs on the single remaining cuda:0, so bitwise
# equality is not claimed; the realised deviation is recorded in the provenance.
REL_TOL = 1e-6


def rc() -> dict:
    """Matplotlib defaults pre-scaled for the printed width."""
    return {
        "font.size": 7.5 * K,
        "axes.titlesize": 7.5 * K,
        "axes.labelsize": 7.5 * K,
        "xtick.labelsize": 6.5 * K,
        "ytick.labelsize": 6.5 * K,
        "axes.linewidth": 0.8 * K,
        "xtick.major.width": 0.8 * K,
        "ytick.major.width": 0.8 * K,
        "xtick.major.size": 3.5 * K,
        "ytick.major.size": 3.5 * K,
        "grid.linewidth": 0.8 * K,
        "legend.fontsize": 6.5 * K,
        "figure.titlesize": 8.0 * K,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_records() -> dict:
    """Lower-median record of each family, by full-future relative L2."""
    rows = [json.loads(line) for line in PANEL_ROWS.read_text().splitlines() if line.strip()]
    if {row["measurement"] for row in rows} != {"n1"}:
        raise SystemExit("panel rows carry measurements other than n1")
    chosen = {}
    for family in FAMILIES:
        group = sorted((r for r in rows if r["medium_type"] == family),
                       key=lambda r: r["future_relative_l2"])
        if not group:
            raise SystemExit(f"no panel rows for family {family}")
        chosen[family] = {
            "row": group[(len(group) - 1) // 2],
            "family_n": len(group),
            "family_mean_full": float(np.mean([r["future_relative_l2"] for r in group])),
            "family_mean_bands": [
                float(np.mean([r["time_bands"][i] for r in group])) for i in range(3)],
        }
    return chosen


def evaluate(model, normalizer, dataset, ds_h5, device, sample_id, panel_row):
    """Reproduce the panel's n1 measurement for one record and keep the fields."""
    (position,) = mg.resolve_positions(dataset, [sample_id])
    record = dataset[position]
    if record.source_index != panel_row["source_index"]:
        raise SystemExit(
            f"{sample_id} resolves to source_index {record.source_index}, "
            f"panel row says {panel_row['source_index']}")
    truth = ds_h5["wavefield"][record.source_index][:].astype(np.float64)
    assert_truth_shape(truth, record.time_s.numpy(), record.z_m.numpy(), record.x_m.numpy())

    n_frames = len(record.time_s)
    onset = mg.source_onset(record)
    if onset != panel_row["onset"]:
        raise SystemExit(f"{sample_id}: onset {onset} against panel {panel_row['onset']}")
    future = np.arange(onset + mg.IC_FRAMES, n_frames)

    ic = dataset.read_wavefield(position, record.time_s[onset:onset + mg.IC_FRAMES])
    if not bool(ic.exact.all()):
        raise SystemExit("IC frames must be exact stored snapshots")
    prepared = mg.prepare_record(model, normalizer, record, device, ic.values,
                                 float(record.time_s[onset]))
    pred = mg.predict_frames(model, normalizer, prepared, record, device, future)
    assert_truth_shape(pred, future, record.z_m.numpy(), record.x_m.numpy(), label="prediction")

    bands = mg.band_split(future)
    offset = int(future[0])
    full = mg.window_relative_l2(pred, truth, future)
    band_rel = [mg.window_relative_l2(pred[b - offset], truth, b) for b in bands]

    # The figure may only be drawn if it agrees with the table it illustrates.
    deltas = {"future_relative_l2": abs(full - panel_row["future_relative_l2"])
              / abs(panel_row["future_relative_l2"])}
    for i, value in enumerate(band_rel):
        deltas[f"time_bands[{i}]"] = abs(value - panel_row["time_bands"][i]) / abs(
            panel_row["time_bands"][i])
    worst = max(deltas.values())
    if worst > REL_TOL:
        raise SystemExit(
            f"{sample_id}: recomputed metrics disagree with the panel by {worst:.3e} "
            f"(tolerance {REL_TOL:.0e}); refusing to draw a figure that contradicts Table 1. "
            f"{json.dumps({k: f'{v:.3e}' for k, v in deltas.items()})}")

    # Axis-order guard.  Fields are stored [T, Z, X] with depth on axis 0; if that
    # were transposed the early-future pressure peak would not sit near the source.
    src_x, src_z = float(record.source_parameters[0]), float(record.source_parameters[1])
    first = np.abs(truth[future[0]])
    iz, ix = np.unravel_index(int(first.argmax()), first.shape)
    peak_x, peak_z = float(record.x_m[ix]), float(record.z_m[iz])
    peak_distance_m = float(np.hypot(peak_x - src_x, peak_z - src_z))
    if peak_distance_m > 300.0:
        raise SystemExit(
            f"{sample_id}: first future frame peaks {peak_distance_m:.0f} m from the source at "
            f"({src_x:.0f},{src_z:.0f}) m -- the [z,x] axis assumption does not hold")

    return {
        "sample_id": sample_id,
        "position": int(position),
        "source_index": int(record.source_index),
        "medium_type": record.medium_type,
        "record": record,
        "truth": truth,
        "pred": pred,
        "future": future,
        "bands": bands,
        "offset": offset,
        "onset": int(onset),
        "full_relative_l2": full,
        "band_relative_l2": band_rel,
        "panel_deltas": deltas,
        "source_xz_m": [src_x, src_z],
        "source_f0_hz": float(record.source_parameters[2]),
        "source_t0_s": float(record.source_parameters[3]),
        "axis_guard_peak_distance_m": peak_distance_m,
        "velocity_min_max": [float(record.velocity_mps.min()), float(record.velocity_mps.max())],
    }


def snapshot_figure(item, out_path: Path) -> dict:
    """Medium and error growth, then truth / prediction / error per time band."""
    record = item["record"]
    truth, pred, offset, future = item["truth"], item["pred"], item["offset"], item["future"]
    x_m, z_m = record.x_m.numpy(), record.z_m.numpy()
    x_lo, x_hi = pixel_extent(x_m)
    z_lo, z_hi = pixel_extent(z_m)
    # imshow rows are depth; the extent flips z so 0 m is at the top.
    extent = [x_lo, x_hi, z_hi, z_lo]
    src_x, src_z = item["source_xz_m"]
    t_axis = record.time_s.numpy()

    frames = [int(band[len(band) // 2]) for band in item["bands"]]
    # Full-page float: a4 leaves about 9.2 in of text height, and the 3x3 grid of
    # square panels needs most of it to stay legible at this width.
    fig = plt.figure(figsize=(AUTHORED_WIDTH_IN, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(4, 3, height_ratios=[0.82, 1.0, 1.0, 1.0])

    # --- row 0: the medium, and this record's per-frame error growth ---
    ax_vel = fig.add_subplot(grid[0, 0])
    velocity = record.velocity_mps.numpy()
    if velocity.ndim == 3:
        velocity = velocity[0]
    image = ax_vel.imshow(velocity, cmap="viridis", extent=extent, aspect="equal",
                          interpolation="nearest")
    ax_vel.plot([src_x], [src_z], marker="*", color="yellow", markersize=7 * K,
                markeredgecolor="black", markeredgewidth=0.5 * K, linestyle="none")
    ax_vel.set_title("medium $c(x,z)$, $\\star$ source", fontsize=7.5 * K)
    ax_vel.set_xlabel("$x$ (m)")
    ax_vel.set_ylabel("$z$ (m)")
    bar = fig.colorbar(image, ax=ax_vel, shrink=0.88, pad=0.02)
    bar.set_label("$c$ (m s$^{-1}$)", fontsize=7.0 * K)
    bar.ax.tick_params(labelsize=6.0 * K)

    ax_curve = fig.add_subplot(grid[0, 1:])
    truth_future = truth[future]
    per_frame_truth = np.sqrt((truth_future ** 2).reshape(len(future), -1).sum(axis=1))
    per_frame_err = np.sqrt(((pred - truth_future) ** 2).reshape(len(future), -1).sum(axis=1))
    per_frame_rel = per_frame_err / np.clip(per_frame_truth, 1e-300, None)
    ax_curve.plot(t_axis[future], per_frame_rel, color="black", lw=1.0 * K)
    for band, band_name, band_rel, shade in zip(
            item["bands"], BAND_NAMES, item["band_relative_l2"], (0.0, 0.07, 0.14)):
        t_lo, t_hi = float(t_axis[int(band[0])]), float(t_axis[int(band[-1])])
        ax_curve.axvspan(t_lo, t_hi, color="black", alpha=shade, lw=0)
        ax_curve.text(0.5 * (t_lo + t_hi), 0.96,
                      f"{band_name.split()[0]}  rel$L^2$={band_rel:.3f}",
                      transform=ax_curve.get_xaxis_transform(), ha="center", va="top",
                      fontsize=7.0 * K, color="black")
    for frame in frames:
        ax_curve.axvline(float(t_axis[frame]), color="crimson", ls=":", lw=0.8 * K)
    ax_curve.set_xlim(float(t_axis[future[0]]), float(t_axis[future[-1]]))
    ax_curve.set_ylim(bottom=0.0)
    ax_curve.set_xlabel("$t$ (s)")
    ax_curve.set_ylabel("per-frame rel$L^2$")
    ax_curve.set_title("error growth over the future window; "
                       "dotted lines are the snapshot times below", fontsize=7.5 * K)
    ax_curve.grid(alpha=0.25)

    # --- rows 1-3: reference / prediction / error at each band's mid-frame ---
    # Every panel is divided by its own row's reference-and-prediction peak, so the whole
    # grid shares one dimensionless colour scale and needs one colour bar rather than six.
    # That also makes the error panel directly readable: on this scale it shows what
    # fraction of the local signal the operator gets wrong, which is the point being made.
    # The divisor is printed in physical units on each row, so nothing is hidden by it.
    drawn = []
    field_axes = []
    image = None
    for r, (frame, band_name, band_rel) in enumerate(
            zip(frames, BAND_NAMES, item["band_relative_l2"]), start=1):
        t_field = truth[frame]
        p_field = pred[frame - offset]
        e_field = p_field - t_field
        scale = float(np.abs(np.stack([t_field, p_field])).max()) or 1.0
        err_scale = float(np.abs(e_field).max()) or 1.0
        frame_rel = float(np.linalg.norm(e_field) / max(np.linalg.norm(t_field), 1e-300))
        axes_row = [fig.add_subplot(grid[r, c]) for c in range(3)]
        field_axes.extend(axes_row)
        # Two-line titles in every column so the three rows keep equal heights.  The band
        # score is not repeated here: the curve above already carries all three.
        titles = (
            f"{band_name.split()[0]},  $t$ = {float(t_axis[frame]):.3f} s\n"
            "reference (LWC-84)",
            f"$\\max|p|$ = {scale:.1e} Pa\noperator prediction",
            f"rel$L^2$ {frame_rel:.3f};  $\\max|e|/\\max|p|$ {err_scale / scale:.2f}\n"
            "error (prediction $-$ reference)",
        )
        for c, (field, title) in enumerate(
                ((t_field, titles[0]), (p_field, titles[1]), (e_field, titles[2]))):
            ax = axes_row[c]
            image = ax.imshow(field / scale, cmap="seismic", vmin=-1.0, vmax=1.0,
                              extent=extent, aspect="equal", interpolation="nearest")
            ax.set_title(title, fontsize=6.8 * K)
            if r == 3:
                ax.set_xlabel("$x$ (m)", fontsize=7.0 * K)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.plot([src_x], [src_z], marker="*", color="yellow", markersize=6 * K,
                        markeredgecolor="black", markeredgewidth=0.5 * K, linestyle="none")
                ax.set_ylabel("$z$ (m)", fontsize=7.0 * K)
            else:
                ax.set_yticklabels([])
        drawn.append({
            "band": band_name, "frame": frame, "t_s": float(t_axis[frame]),
            "frame_relative_l2": frame_rel,
            "row_normalisation_pa": scale, "error_peak_pa": err_scale,
            "error_to_reference_peak_ratio": err_scale / scale,
        })
    bar = fig.colorbar(image, ax=field_axes, location="right", shrink=0.42, pad=0.015,
                       aspect=26)
    bar.set_label("$p\\,/\\,\\max|p|$ of the row", fontsize=7.0 * K)
    bar.ax.tick_params(labelsize=6.5 * K)
    fig.suptitle(
        f"Anchor step 29359; {item['medium_type']} record {item['sample_id']}, train split, "
        f"full-future rel$L^2$ {item['full_relative_l2']:.4f}\n"
        "(family median of "
        f"{item['family_context']['n']}); one shared colour scale, each row over its printed peak",
        fontsize=7.5 * K)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "frames": drawn,
        "extent_m": [x_lo, x_hi, z_lo, z_hi],
        "per_frame_relative_l2_first": float(per_frame_rel[0]),
        "per_frame_relative_l2_min": float(per_frame_rel.min()),
        "per_frame_relative_l2_min_t_s": float(t_axis[future][int(per_frame_rel.argmin())]),
        "per_frame_relative_l2_max": float(per_frame_rel.max()),
        "per_frame_relative_l2_max_t_s": float(t_axis[future][int(per_frame_rel.argmax())]),
        "per_frame_relative_l2_at_window_end": float(per_frame_rel[-1]),
    }


def trace_figure(items: list[dict], out_path: Path) -> dict:
    """Receiver waveforms: full future window, and the late band on its own scale.

    The late band is re-normalised because the coda amplitude is one to two orders
    of magnitude below the first arrival; on a single global scale the window the
    paper is about is a flat line at zero.
    """
    fig, axes = plt.subplots(len(items), 2, figsize=(AUTHORED_WIDTH_IN, 5.5),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    axes = np.atleast_2d(axes)
    recorded = []
    for row, item in enumerate(items):
        record = item["record"]
        x_m, z_m = record.x_m.numpy(), record.z_m.numpy()
        ix = int(np.argmin(np.abs(x_m - RECEIVER_X_M)))
        iz = int(np.argmin(np.abs(z_m - RECEIVER_Z_M)))
        if not (np.isclose(x_m[ix], RECEIVER_X_M) and np.isclose(z_m[iz], RECEIVER_Z_M)):
            raise SystemExit(
                f"receiver ({RECEIVER_X_M},{RECEIVER_Z_M}) m is not a stored node; "
                f"nearest is ({x_m[ix]},{z_m[iz]}) m")
        future, offset = item["future"], item["offset"]
        t_axis = record.time_s.numpy()
        t_window = t_axis[future]
        t_trace = item["truth"][future, iz, ix]
        p_trace = item["pred"][future - offset, iz, ix]
        peak = float(np.abs(t_trace).max())
        if peak <= 0.0:
            raise SystemExit(f"{item['sample_id']}: reference trace is identically zero")
        trace_rel = float(np.linalg.norm(p_trace - t_trace) / np.linalg.norm(t_trace))

        late = item["bands"][-1]
        late_mask = np.isin(future, late)
        late_peak = float(np.abs(t_trace[late_mask]).max())
        late_rel = float(np.linalg.norm(p_trace[late_mask] - t_trace[late_mask])
                         / max(np.linalg.norm(t_trace[late_mask]), 1e-300))

        ax = axes[row, 0]
        ax.plot(t_window, t_trace / peak, color="black", lw=1.0 * K, label="reference (LWC-84)")
        ax.plot(t_window, p_trace / peak, color="crimson", lw=0.9 * K, ls="--",
                label="operator prediction")
        for band, band_name, band_rel, shade in zip(
                item["bands"], BAND_NAMES, item["band_relative_l2"], (0.0, 0.07, 0.14)):
            t_lo, t_hi = float(t_axis[int(band[0])]), float(t_axis[int(band[-1])])
            ax.axvspan(t_lo, t_hi, color="black", alpha=shade, lw=0)
            ax.text(0.5 * (t_lo + t_hi), 0.965,
                    f"{band_name.split()[0]}  rel$L^2$={band_rel:.3f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.2 * K)
        ax.set_xlim(float(t_window[0]), float(t_window[-1]))
        ax.set_ylim(-1.45, 1.45)
        ax.set_ylabel("$p\\,/\\,\\max|p_{\\mathrm{ref}}|$")
        ax.grid(alpha=0.22)
        ax.set_title(f"{item['medium_type']}   {item['sample_id']}   "
                     f"trace rel$L^2$ {trace_rel:.3f}", fontsize=7.2 * K)
        if row == 0:
            ax.legend(loc="lower right", fontsize=6.5 * K, ncol=2, framealpha=0.92)

        ax = axes[row, 1]
        ax.plot(t_window[late_mask], t_trace[late_mask] / late_peak, color="black", lw=1.0 * K)
        ax.plot(t_window[late_mask], p_trace[late_mask] / late_peak, color="crimson", lw=0.9 * K,
                ls="--")
        ax.axvspan(float(t_axis[int(late[0])]), float(t_axis[int(late[-1])]),
                   color="black", alpha=0.14, lw=0)
        ax.set_xlim(float(t_axis[int(late[0])]), float(t_axis[int(late[-1])]))
        ax.set_ylim(-1.45, 1.45)
        ax.set_ylabel("$p\\,/\\,\\max|p_{\\mathrm{ref}}^{B3}|$")
        ax.grid(alpha=0.22)
        gain = peak / late_peak
        gain_text = f"{gain:.0f}" if gain < 1000 else f"{gain:.1e}".replace("e+0", "e")
        ax.set_title(f"B3, re-normalised $\\times${gain_text},  rel$L^2$ {late_rel:.3f}",
                     fontsize=7.2 * K)
        # The prediction can leave the re-normalised window by orders of magnitude; say so
        # rather than letting the reader read clipped near-vertical strokes as waveform.
        pred_excursion = float(np.abs(p_trace[late_mask]).max() / late_peak)
        if pred_excursion > 1.45:
            ax.text(0.5, 0.06, f"prediction off scale: {pred_excursion:.0f}$\\times$ this panel's peak",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5 * K,
                    color="crimson",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="crimson", lw=0.5 * K))
        ill_conditioned = item["medium_type"] in ILL_CONDITIONED_LATE_FAMILIES
        if ill_conditioned:
            ax.text(0.5, 0.94, "near-zero denominator: no claim is gated on this cell",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.5 * K, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ffe9c9", ec="black", lw=0.5 * K))

        recorded.append({
            "sample_id": item["sample_id"], "medium_type": item["medium_type"],
            "receiver_node_zx": [iz, ix],
            "receiver_xz_m": [float(x_m[ix]), float(z_m[iz])],
            "trace_relative_l2": trace_rel, "reference_peak_pa": peak,
            "late_band_trace_relative_l2": late_rel, "late_band_reference_peak_pa": late_peak,
            "late_band_renormalisation_gain": peak / late_peak,
            "late_band_prediction_excursion": pred_excursion,
            "late_band_flagged_ill_conditioned": ill_conditioned,
        })
    for ax in axes[-1]:
        ax.set_xlabel("$t$ (s)")
    fig.suptitle(
        f"Receiver at $x$ = {RECEIVER_X_M:.0f} m, $z$ = {RECEIVER_Z_M:.0f} m; "
        "anchor step 29359, train split, family-median records\n"
        "left: full future window over the first-arrival peak; "
        "right: late band B3 over its own peak",
        fontsize=7.5 * K)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {"receivers": recorded}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    args = parser.parse_args(argv)

    # Type and line widths are pre-scaled for the printed width; see K.
    plt.rcParams.update(rc())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    free_gib = mg.require_vram(device, args.min_free_gib, 0.0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = V3Config.from_yaml(str(CONFIG))
    manifest = build_manifest(cfg.data.source_h5)
    # split='train' only: the anchor table is a train-split measurement and
    # test_id stays sealed.
    dataset = V3WavefieldDataset(cfg.data.source_h5, manifest, split="train")
    normalizer = load_normalizer(cfg, manifest.digest)
    model, provenance = mg.load_checkpoint(CHECKPOINT, cfg, device, CHECKPOINT_SHA, False)
    if provenance["backfill"]["measures_untrained_parameters"]:
        raise SystemExit("checkpoint would measure untrained parameters")

    chosen = select_records()
    panel_meta = json.loads(PANEL_META.read_text())
    frozen_sha = file_sha256(mg.FROZEN_LISTS)
    if frozen_sha != panel_meta["frozen_lists_sha256"]:
        raise SystemExit(
            f"frozen record lists have changed since the panel run: {frozen_sha} against "
            f"{panel_meta['frozen_lists_sha256']}")
    if manifest.digest != panel_meta["manifest_digest"]:
        raise SystemExit("dataset manifest digest differs from the panel run")
    ds_h5 = h5py.File(cfg.data.source_h5, "r", swmr=True)
    items = []
    for family in FAMILIES:
        entry = chosen[family]
        item = evaluate(model, normalizer, dataset, ds_h5, device,
                        entry["row"]["sample_id"], entry["row"])
        item["family_context"] = {
            "n": entry["family_n"],
            "family_mean_full_relative_l2": entry["family_mean_full"],
            "family_mean_band_relative_l2": entry["family_mean_bands"],
        }
        items.append(item)
        print(f"{family:9s} {item['sample_id']:22s} full={item['full_relative_l2']:.6f} "
              f"bands={[round(v, 6) for v in item['band_relative_l2']]} "
              f"worst_panel_delta={max(item['panel_deltas'].values()):.2e}", flush=True)

    marmousi = next(i for i in items if i["medium_type"] == "marmousi")
    snap_path = out_dir / "fig_wavefield_snapshots_marmousi_29359.pdf"
    trace_path = out_dir / "fig_receiver_traces_threefamily_29359.pdf"
    snap_meta = snapshot_figure(marmousi, snap_path)
    trace_meta = trace_figure(items, trace_path)

    record_meta = []
    for item in items:
        record_meta.append({
            key: item[key] for key in (
                "sample_id", "position", "source_index", "medium_type", "onset",
                "full_relative_l2", "band_relative_l2", "panel_deltas", "source_xz_m",
                "source_f0_hz", "source_t0_s", "axis_guard_peak_distance_m",
                "velocity_min_max", "family_context")
        })
        record_meta[-1]["future_frames"] = [int(item["future"][0]), int(item["future"][-1]) + 1]

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "wavefield snapshot and receiver waveform figures for the anchor-29359 paper",
        "script_sha256": file_sha256(Path(__file__)),
        "code_tree": str(SNAPSHOT),
        "code_tree_note": ("the anchor run's frozen snapshot; all 38 files match "
                           "formal/run_identity.json code_files, and the live release tree has "
                           "since drifted"),
        "code_digest_recorded_by_run": json.loads(
            (Path("/root/autodl-tmp/staging/l2_lwc84_ddp4_20260915_v1/formal/run_identity.json")
             ).read_text())["code_digest"],
        "measure_gates_sha256": file_sha256(GATE_DIR / "measure_gates.py"),
        "measure_gates_sha256_in_panel_meta": panel_meta["script_sha256"],
        "config_file": str(CONFIG),
        "config_file_sha256": file_sha256(CONFIG),
        "config_digest": cfg.digest(),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "global_step": provenance["global_step"],
        "epoch": provenance["epoch"],
        "manifest_digest": manifest.digest,
        "source_h5": cfg.data.source_h5,
        "split": "train",
        "device": str(device),
        "free_vram_gib_at_start": round(free_gib, 3),
        "tf32": False,
        "time_block": 1,
        "panel_rows": str(PANEL_ROWS),
        "panel_rows_sha256": file_sha256(PANEL_ROWS),
        "selection_rule": (
            "per family, the record whose full-future relative L2 is the lower median "
            "(sorted index (n-1)//2) of that family's frozen list; fixed before measurement"),
        "frozen_lists_path": str(mg.FROZEN_LISTS),
        "frozen_lists_sha256": file_sha256(mg.FROZEN_LISTS),
        "metric": ("sqrt(sum((pred-truth)^2)/sum(truth^2)) in float64 over the future window, "
                   "computed by measure_gates.window_relative_l2"),
        "panel_agreement_tolerance_relative": REL_TOL,
        "panel_agreement_worst_relative": max(
            max(item["panel_deltas"].values()) for item in items),
        "receiver_xz_m": [RECEIVER_X_M, RECEIVER_Z_M],
        "receiver_note": ("the free surface is node-centred with p(z=0)=0 identically, so the "
                          "receiver line sits 100 m below it"),
        "records": record_meta,
        "figures": {
            snap_path.name: {"kind": "wavefield snapshots", **snap_meta,
                             "record": marmousi["sample_id"],
                             "sha256": file_sha256(snap_path)},
            trace_path.name: {"kind": "receiver waveforms", **trace_meta,
                              "sha256": file_sha256(trace_path)},
        },
        "persisted": "figures and scalar rows only; predicted wavefields are never written",
    }
    meta_path = out_dir / "FIGURE_PROVENANCE.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "wrote", "figures": [str(snap_path), str(trace_path)],
                      "provenance": str(meta_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
