#!/usr/bin/env python
"""Build a compact, evidence-labelled TGRS experiment result bundle.

Small paper assets are copied so the bundle is directly browsable. Large complete
wavefield arrays are linked in place to avoid duplicating hundreds of megabytes.
No model inference, GPU access, validation opening, or process control occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "paper"
    / "tgrs_helmholtz_operator"
    / "experiment_evidence_bundle_20260813"
)
R5B_RUN = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/local_field_w128_hicap/"
    "temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag/run"
)
R5D_RUN = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/local_field_w128_hicap/"
    "temporal_latent_a3_rank32_r5d_metric_aligned_time_train_diag/run"
)
POSITION_OUTPUT = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "evaluation/marmousi_fixed19hz_position_240_post_r5d_r1"
)
CPADC_VALIDATION_SUMMARY = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/cpadc/"
    "cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7/"
    "evaluation/summary.json"
)
CPADC_TEST_ID_SUMMARY = Path(
    "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
    "pretraining/cpadc/"
    "cpadc_rank16_family_calibrated_a3e1_marmousi1_4m_v2_r7_test_id_confirmation/"
    "evaluation/summary.json"
)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _link(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source, start=destination.parent.resolve())
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.symlink_to(relative, target_is_directory=source.is_dir())
    os.replace(temporary, destination)


def _copy_many(entries: Iterable[tuple[str, str]], destination: Path) -> None:
    for source, name in entries:
        _copy(ROOT / source, destination / name)


def _read_r5b_gate_metrics() -> tuple[dict[str, Any], Path]:
    """Read the r5b prediction from r5d update-zero, which reproduces it exactly."""

    path = R5D_RUN / "epoch_validation_control.jsonl"
    for raw in path.read_text(encoding="utf8").splitlines():
        row = json.loads(raw)
        if row.get("event") == "validation_baseline":
            metrics = row.get("metrics", {})
            if len(metrics.get("source_relative_l2", {})) != 48:
                raise ValueError("r5b fixed train-gate census must contain 48 records")
            if not np.isclose(
                float(metrics["aggregate_relative_l2"]),
                0.2772543673381241,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError("r5b reproduced gate score changed")
            return metrics, path
    raise ValueError("r5d update-zero r5b metric record is missing")


def _family(sample_id: str) -> str:
    for name in ("uniform", "layered", "marmousi"):
        if name in sample_id:
            return name
    raise ValueError(f"cannot infer family from sample id: {sample_id}")


def _write_relative_error_assets(destination: Path) -> None:
    metrics, source = _read_r5b_gate_metrics()
    rows = [
        {
            "sample_id": sample_id,
            "family": _family(sample_id),
            "relative_l2": float(value),
            "evidence_split": "train",
            "evidence_scope": "fixed_48_record_train_gate",
            "frames_per_record": 32,
        }
        for sample_id, value in sorted(metrics["source_relative_l2"].items())
    ]
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "r5b_fixed_train_gate_all_48_records.csv"
    with csv_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    families = ("all", "uniform", "layered", "marmousi")
    groups = [
        np.asarray(
            [row["relative_l2"] for row in rows if name == "all" or row["family"] == name],
            dtype=np.float64,
        )
        for name in families
    ]
    summary = {
        "schema": "r5b_fixed_train_gate_relative_error_census_v1",
        "evidence_status": "train_only_diagnostic",
        "selection_or_confirmatory": "selection",
        "source_jsonl": str(source),
        "record_count": len(rows),
        "aggregate_relative_l2": float(metrics["aggregate_relative_l2"]),
        "family_relative_l2": metrics["family_relative_l2"],
        "distributions": {
            name: {
                "count": int(len(values)),
                "minimum": float(np.min(values)),
                "q1": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q3": float(np.quantile(values, 0.75)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
            }
            for name, values in zip(families, groups, strict=True)
        },
        "warning": (
            "These are all 48 records in the frozen train-only checkpoint gate. "
            "They are not validation/test estimates and are not the pending 240-case "
            "fixed-19-Hz source-position experiment."
        ),
    }
    _write_json(destination / "r5b_fixed_train_gate_summary.json", summary)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ("#99AABB", "#5185C0", "#E99D4E", "#55966B")
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    boxes = axis.boxplot(
        groups,
        labels=("All (n=48)", "Uniform", "Layered", "Marmousi"),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor("#333333")
    rng = np.random.default_rng(372)
    for index, (values, color) in enumerate(zip(groups, colors, strict=True), start=1):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        axis.scatter(
            np.full_like(values, index, dtype=np.float64) + jitter,
            values,
            s=17,
            alpha=0.72,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    axis.set_ylabel("Per-record relative $L_2$")
    axis.set_title("r5b fixed train-only gate: all 48 evaluated records")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
    axis.text(
        0.01,
        0.98,
        "Selection diagnostic, not validation/test evidence",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#7f3b2e",
    )
    fig.tight_layout()
    fig.savefig(destination / "r5b_fixed_train_gate_all_48_records_boxplot.pdf")
    fig.savefig(destination / "r5b_fixed_train_gate_all_48_records_boxplot.png", dpi=300)
    plt.close(fig)


def _write_cpadc_relative_error_assets(destination: Path) -> None:
    """Export the complete sealed CPADC validation and test_id record census."""

    split_sources = {
        "validation": CPADC_VALIDATION_SUMMARY,
        "test_id": CPADC_TEST_ID_SUMMARY,
    }
    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    destination.mkdir(parents=True, exist_ok=True)
    for split, source in split_sources.items():
        payload = json.loads(source.read_text(encoding="utf8"))
        records = payload.get("records", [])
        if len(records) != 480 or not payload.get("future_truth_opened_only_after_seal"):
            raise ValueError(f"CPADC {split} census is incomplete or unsealed")
        if split == "test_id" and payload.get("evaluation_split") != "test_id":
            raise ValueError("CPADC test_id summary has the wrong split")
        family_count: dict[str, int] = {}
        for record in records:
            family = str(record["medium_type"])
            family_count[family] = family_count.get(family, 0) + 1
            parent = float(record["parent_metrics"]["aggregate_relative_l2"])
            adapted = float(record["metrics"]["aggregate_relative_l2"])
            if parent <= 0.0 or adapted <= 0.0 or not bool(record["sealed"]):
                raise ValueError("CPADC record is nonpositive or unsealed")
            rows.append(
                {
                    "split": split,
                    "sample_id": str(record["sample_id"]),
                    "family": family,
                    "parent_relative_l2": parent,
                    "cpadc_relative_l2": adapted,
                    "relative_improvement_fraction": (parent - adapted) / parent,
                    "nonworse": adapted <= parent,
                    "adaptation_accepted": bool(record["adaptation"]["accepted"]),
                    "all_saved_time_indices": int(record["all_saved_time_indices"]),
                    "prediction_sealed_before_truth": bool(record["sealed"]),
                }
            )
        summaries[split] = {
            "source": str(source),
            "record_count": len(records),
            "family_count": family_count,
            "promotion_gate": payload["promotion_gate"],
        }
        _copy(source, destination / f"cpadc_r7_{split}_complete_summary.json")

    with (destination / "cpadc_r7_all_960_records.csv").open(
        "w", encoding="utf8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        destination / "cpadc_r7_all_960_records_summary.json",
        {
            "schema": "cpadc_r7_complete_relative_error_census_v1",
            "evidence_status": "sealed_complete_validation_and_independent_test_id",
            "record_count": len(rows),
            "splits": summaries,
            "claim_limit": (
                "CPADC establishes a bounded relative correction benefit. It does "
                "not establish an absolute five-percent solver error."
            ),
        },
    )

    families = ("all", "uniform", "layered", "marmousi")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    colors = {"parent": "#C96144", "cpadc": "#5185C0"}
    for axis, split in zip(axes, ("validation", "test_id"), strict=True):
        split_rows = [row for row in rows if row["split"] == split]
        positions = []
        groups = []
        group_colors = []
        labels = []
        for index, family in enumerate(families, start=1):
            selected = [
                row
                for row in split_rows
                if family == "all" or row["family"] == family
            ]
            groups.extend(
                (
                    np.asarray([row["parent_relative_l2"] for row in selected]),
                    np.asarray([row["cpadc_relative_l2"] for row in selected]),
                )
            )
            positions.extend((index - 0.16, index + 0.16))
            group_colors.extend((colors["parent"], colors["cpadc"]))
            labels.append("All" if family == "all" else family.title())
        boxes = axis.boxplot(
            groups,
            positions=positions,
            widths=0.27,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
        )
        for box, color in zip(boxes["boxes"], group_colors, strict=True):
            box.set_facecolor(color)
            box.set_edgecolor("#333333")
            box.set_alpha(0.72)
        axis.set_xticks(range(1, len(families) + 1), labels)
        axis.set_yscale("log")
        axis.grid(axis="y", which="both", color="#dddddd", linewidth=0.6, alpha=0.65)
        gate = summaries[split]["promotion_gate"]
        display_split = "Validation" if split == "validation" else "Independent test_id"
        axis.set_title(
            f"{display_split} (n=480)\nmean relative improvement "
            f"{100.0 * float(gate['mean_relative_improvement']):.2f}%"
        )
    axes[0].set_ylabel("Per-record relative $L_2$ (log scale)")
    fig.suptitle("CPADC R7: complete sealed validation and independent test_id censuses")
    handles = [
        plt.Line2D([], [], color=colors["parent"], linewidth=8, alpha=0.72, label="Parent"),
        plt.Line2D([], [], color=colors["cpadc"], linewidth=8, alpha=0.72, label="CPADC R7"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(destination / "cpadc_r7_all_960_records_boxplot.pdf")
    fig.savefig(destination / "cpadc_r7_all_960_records_boxplot.png", dpi=300)
    plt.close(fig)


def _write_hyperparameters(destination: Path) -> None:
    r5b_audit = json.loads((ROOT / "results/r5b_latest_model_architecture_audit_20260813.json").read_text())
    r5d_audit = json.loads((ROOT / "results/r5d_candidate_architecture_audit_20260813.json").read_text())
    patch_audit = json.loads((ROOT / "results/patch_deeponet_training_static_audit_20260813.json").read_text())
    r5b_config_path = ROOT / "configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag.yaml"
    r5b_config = yaml.safe_load(r5b_config_path.read_text())
    r5e_config_path = ROOT / "configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5e_true_metric_aligned_time_train_diag.yaml"
    r5e_config = yaml.safe_load(r5e_config_path.read_text())
    parent_identity = json.loads((R5B_RUN / "run_identity.json").read_text())
    base_model = parent_identity["config"].get("base_config")
    payload = {
        "schema": "tgrs_network_hyperparameter_design_v1",
        "deployed_r5b": {
            "evidence_status": "selected_train_gate_checkpoint",
            "parameter_count": r5b_audit["parameter_count"],
            "checkpoint_sha256": r5b_audit["checkpoint_sha256"],
            "base_model_config": base_model,
            "variant": r5b_audit["variant"],
            "training": {
                "frames_per_record": r5b_config["training_frames_per_record"],
                "effective_records_per_update": r5b_config["macro_records"]
                * r5b_config["macros_per_update"],
                "microbatch_records": r5b_config["microbatch_records"],
                "epochs": r5b_config["epochs"],
                "optimizer": r5b_config["optimizer"],
                "loss": r5b_config["loss"],
                "time_policy": r5b_config["time_policy"],
            },
            "active_components": r5b_audit["active"],
        },
        "diagnostic_r5d": {
            "evidence_status": "diagnostic_not_promotion_eligible",
            "parameter_count": r5d_audit["parameter_count"],
            "variant": r5d_audit["variant"],
            "known_selector_defect": "training seed 372; gate seed 8291",
        },
        "registered_r5e": {
            "evidence_status": "registered_not_launched",
            "time_selector_seed_offset": r5e_config["epoch_validation_control"]["time_selector_seed_offset"],
            "optimizer": r5e_config["optimizer"],
            "loss": r5e_config["loss"],
        },
        "patch_deeponet": {
            "evidence_status": "static_audit_pass_not_gpu_trained",
            "model_config": patch_audit["model_config"],
            "parameter_match": patch_audit["parameter_match"],
        },
        "claim_scope": {
            "source_generalization_variable": "position_only",
            "fixed_frequency_hz": 19.0,
            "frequency_generalization_permitted": False,
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "network_hyperparameters.json", payload)
    rows = [
        ("deployed_r5b", "parameter_count", r5b_audit["parameter_count"]),
        ("deployed_r5b", "width", 128),
        ("deployed_r5b", "decoder_depth", r5b_audit["variant"]["depth"]),
        ("deployed_r5b", "spectral_modes", r5b_audit["variant"]["modes"]),
        ("deployed_r5b", "spectral_rank", r5b_audit["variant"]["spectral_rank"]),
        ("deployed_r5b", "temporal_latent_rank", r5b_audit["variant"]["local_field_temporal_latent_rank"]),
        ("deployed_r5b", "temporal_harmonics", r5b_audit["variant"]["local_field_temporal_latent_harmonics"]),
        ("deployed_r5b", "warp_max_shift_cells", r5b_audit["variant"]["local_field_warp_max_shift_cells"]),
        ("deployed_r5b", "local_channels", "1|1|2|2"),
        ("deployed_r5b", "optimizer", r5b_config["optimizer"]["name"]),
        ("deployed_r5b", "frames_per_record", r5b_config["training_frames_per_record"]),
        ("deployed_r5b", "effective_records_per_update", r5b_config["macro_records"] * r5b_config["macros_per_update"]),
        ("patch_deeponet", "parameter_count", patch_audit["parameter_match"]["parameter_count"]),
        *(('patch_deeponet', key, value) for key, value in patch_audit["model_config"].items()),
    ]
    with (destination / "network_hyperparameters.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "hyperparameter", "value"))
        writer.writerows(rows)
    markdown = """# Network hyperparameter design

## Deployed r5b operator

The selected train-gate checkpoint contains 32,420,564 parameters. Its dominant
settings are width 128, a depth-12 factorized complex spectral decoder, 32 retained
modes, spectral rank 112, a four-level local field with channel multipliers
`1,1,2,2`, an 8-cell first-arrival-gradient warp, and a rank-32 temporal latent bank
with four harmonics. The active components and every low-level variant field are
recorded in `network_hyperparameters.json`.

Training uses 24 exact frames per record and an effective 32 records per optimizer
update, accumulated from one-record physical microbatches. r5b uses Muon for matrix
parameters and AdamW for the remaining groups. Its selection metric is mean
per-record relative L2 on a fixed 48-record train-only gate.

## Diagnostic continuation

r5d adds a shared dynamic multiscale spectral adapter and has 33,241,115 parameters,
but its training/gate time selectors were discovered to use different seeds. It is
diagnostic only. r5e is registered but not launched and fixes only this seed offset.

## Parameter-matched Patch-DeepONet

Patch-DeepONet has 32,420,465 parameters, 99 fewer than r5b. It combines a compact
local CNN, a high-capacity pooled branch MLP, and a width-48 query trunk. It has
passed static and CPU tests but has not been GPU-trained, so no accuracy comparison
is claimed.

## Claim scope

The source experiment fixes the Ricker frequency at 19 Hz and varies source
position only. No frequency-generalization claim is permitted.
"""
    _write_text(destination / "NETWORK_HYPERPARAMETERS.md", markdown)


def _write_status_and_readme(output: Path) -> None:
    status_rows = [
        ("complete wavefields", "available", "historical development triplet", "01_complete_wavefields"),
        ("wavefield snapshots", "available", "historical development examples", "02_wavefield_snapshots"),
        ("receiver waveforms and gathers", "available", "historical development examples", "03_receiver_waveforms"),
        ("analytic numerical dispersion", "available", "FD2/FD4/LWC-84 analytic context", "04_numerical_dispersion"),
        ("learned dispersion suppression", "not established", "claim gate failed because receiver evidence is incomplete", "04_numerical_dispersion"),
        ("all-record relative-error boxplot", "available", "all 48 records of train-only r5b selection gate", "05_relative_error"),
        ("complete CPADC relative-error distributions", "available", "480 validation plus 480 independent test_id records", "05_relative_error"),
        ("240-case fixed-19-Hz position boxplot", "pending", "queued sealed evaluation; no result fabricated", "09_pending_position_evaluation"),
        ("network hyperparameters", "available", "r5b, r5d/r5e and Patch-DeepONet", "06_network_hyperparameters"),
    ]
    with (output / "EVIDENCE_STATUS.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact", "status", "evidence_scope", "directory"))
        writer.writerows(status_rows)

    readme = """# TGRS experiment evidence bundle

This folder consolidates the current acoustic-operator experiment evidence without
duplicating large wavefield arrays or changing any running process. Open
`EVIDENCE_STATUS.csv` first: every asset is labelled as historical development,
train-only selection, confirmatory, or pending.

## Directory map

1. `01_complete_wavefields`: full 401-frame truth/coarse/prediction arrays for one
   historical development example per family. The NPZ files are read-only relative
   links to their canonical location.
2. `02_wavefield_snapshots`: prediction/reference/error snapshot panels. Their main
   role is qualitative illustration, not population inference.
3. `03_receiver_waveforms`: receiver traces and common-receiver gathers for the same
   development examples, plus PDE-adaptation diagnostic plots.
4. `04_numerical_dispersion`: analytic phase-velocity curves and numeric CSV for
   FD2, FD4 and LWC-84, plus the learned-dispersion claim gate. The analytic figure
   supports numerical context only; the current claim gate does not establish
   learned dispersion suppression.
5. `05_relative_error`: all 48 per-record errors from the frozen r5b train-only gate,
   plus paired parent/CPADC values for all 480 validation and 480 independent
   `test_id` records. CSV, summary JSON, PNG and vector PDF are provided. The r5b
   panel is selection evidence; the CPADC panel is sealed full-split evidence for a
   relative correction benefit, not an absolute solver-accuracy claim.
6. `06_network_hyperparameters`: proposed r5b, diagnostic r5d/r5e and parameter-
   matched Patch-DeepONet design/configuration files.
7. `07_protocols_and_figure_plan`: frozen fixed-19-Hz, position-only protocol,
   preregistration and figure plan.
8. `08_reproducibility_and_diagnostics`: metric diagnosis, VDS dependency audit and
   architecture audits.
9. `09_pending_position_evaluation`: read-only link to the queued 30-slice x
   8-position evaluation. Its official snapshots, receiver gathers and 240-record
   score table will appear there only after the active training ends naturally.
10. `99_historical_experiment_index`: one-row-per-top-level-result registry. Large
    checkpoints are indexed, not copied.
11. `10_training_runs_read_only`: live read-only links to the selected r5b,
    naturally completed r5c and active r5d run directories. They preserve training
    curves, retry histories and checkpoint metadata without copying model weights.

## Figure logic

- Complete-wavefield figure claim: the model reconstructs propagation structure
  across early, middle and late times on selected development cases.
- Receiver figure claim: time-series diagnostics expose arrival and phase errors
  that field snapshots alone can conceal.
- Dispersion figure claim: LWC-84 has lower analytic phase-velocity error than FD2
  and FD4 at equal points per wavelength; no method is described as dispersion-free.
- Relative-error figure claim: r5b error remains strongly family dependent on the
  fixed train-only gate, with Marmousi as the hardest family.
- Position-generalization figure claim, pending: at one fixed 19-Hz wavelet, error
  must remain controlled across eight unseen source positions on 30 unseen
  Marmousi velocity slices.

Main-paper candidates are the fixed-19-Hz position panel, receiver comparison,
all-sample distribution and analytic dispersion context after their evidence gates
pass. Historical single-case panels and adaptation diagnostics belong in the
supplement. Legends must state the split, sample count, source frequency, and
whether the result is selection or confirmatory evidence.
"""
    _write_text(output / "README.md", readme)
    readme_zh = """# TGRS 实验结果证据包

本文件夹集中整理当前声学算子研究的图、逐样本数据、网络超参数、实验协议和
可复现性审计。大型完整波场采用相对符号链接，避免重复占用约 440 MB；其余
图表和报告均实体复制。先查看 `EVIDENCE_STATUS.csv`，其中明确区分历史开发
证据、训练集选模证据、完整确认实验和等待中的实验。

## 内容导航

- `01_complete_wavefields`：Uniform、Layered、Marmousi 各一个历史开发样本的
  401 帧真值、粗解和预测数组。
- `02_wavefield_snapshots`：完整波场快照以及参考、预测、误差对比图。
- `03_receiver_waveforms`：接收器时间波形与浅层接收线炮集图。
- `04_numerical_dispersion`：FD2、FD4、LWC-84 的解析相速度误差 PDF/PNG/CSV，
  以及 learned-dispersion claim gate。当前只能声称量化了数值频散，不能声称
  神经网络已经克服或消除了频散。
- `05_relative_error`：两套完整数据。第一套是 r5b 固定训练门控全部 48 条记录；
  第二套是 CPADC R7 全部 480 条 validation 和全部 480 条独立 `test_id`，包含
  960 行配对 CSV、箱线图 PDF/PNG 和原始 summary JSON。
- `06_network_hyperparameters`：r5b、r5d/r5e、参数匹配 Patch-DeepONet 的
  超参数 JSON/CSV/Markdown，以及原始 YAML 配置。
- `07_protocols_and_figure_plan`：固定 19 Hz、仅震源位置泛化的冻结协议和图版计划。
- `08_reproducibility_and_diagnostics`：相对误差平台期、selector seed、VDS 文件依赖、
  网络结构和 Patch-DeepONet 审计。
- `09_pending_position_evaluation`：30 个未见 Marmousi 切片乘 8 个位置的实时只读
  输出链接。正式结果未完成前保持 pending，绝不以协议文件冒充实验结果。
- `10_training_runs_read_only`：r5b/r5c/r5d 规范训练目录的只读链接，便于检查训练曲线、
  retry 记录和 checkpoint 元数据。
- `99_historical_experiment_index`：`results/` 下每个顶层实验的一层注册表。大型历史
  checkpoint 仅登记路径，不递归复制。

## 当前可引用的数值

- r5b 固定 48 条训练门控平均相对误差为 0.277254；Uniform、Layered、Marmousi
  分别为 0.108413、0.252105、0.573991。该结果仅用于训练集选模诊断。
- CPADC R7 在完整 validation 上平均相对改善 2.7006%，97.2917% 的记录不变差；
  在独立 `test_id` 上平均改善 2.8420%，97.0833% 的记录不变差。它支持受限的
  相对校正收益，不支持绝对 5% 求解器误差声明。
- 在每波长 4 个网格点处，FD2、FD4、LWC-84 的最大解析相速度误差分别为
  9.49%、2.37%、0.337%。不得写成“无频散”。

## 论文图组织原则

主文候选应包括固定 19 Hz 多位置结果、接收器相位/延迟、完整样本误差分布和
解析频散背景；单样本历史快照、早期 adaptation 图以及失败基线放补充材料。
每个图注必须注明 split、样本数、震源频率以及它属于选模还是确认实验。

本研究的震源泛化变量只允许是位置，频率固定为 19 Hz，不允许频率泛化结论。
"""
    _write_text(output / "README_zh.md", readme_zh)


def _write_historical_registry(output: Path) -> None:
    rows: list[tuple[str, str, int, str]] = []
    for path in sorted((ROOT / "results").iterdir()):
        if path.name == output.name:
            continue
        if path.is_file():
            rows.append((str(path.relative_to(ROOT)), "file", path.stat().st_size, ""))
            continue
        terminal = path / "terminal.json"
        status = ""
        if terminal.is_file() and terminal.stat().st_size <= 2_000_000:
            try:
                status = str(json.loads(terminal.read_text()).get("status", ""))
            except (json.JSONDecodeError, OSError):
                status = "unreadable_terminal"
        rows.append((str(path.relative_to(ROOT)), "directory", 0, status))
    destination = output / "99_historical_experiment_index"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "top_level_results_registry.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "kind", "byte_count_if_file", "terminal_status_if_available"))
        writer.writerows(rows)
    _write_text(
        destination / "README.md",
        "# Historical result index\n\nThis bounded, one-level registry inventories every top-level entry under `results/`. It deliberately does not copy checkpoints or recursively hash the archive while four-GPU training is active.\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.name == "FILE_MANIFEST.csv" or path.is_dir():
            continue
        relative = str(path.relative_to(output))
        linked = path.is_symlink()
        target = path.resolve()
        size = target.stat().st_size
        digest = _sha256(target) if size <= 20 * 1024 * 1024 else "not_hashed_over_20MiB"
        rows.append((relative, "symlink" if linked else "file", str(target), size, digest))
    with (output / "FILE_MANIFEST.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bundle_path", "kind", "resolved_source", "byte_count", "sha256_or_policy"))
        writer.writerows(rows)


def build(output: Path) -> None:
    if (output / "LATEST_MODEL_PROMOTION.json").is_file():
        raise FileExistsError(
            "refusing to overwrite a promoted latest-model evidence bundle; "
            "use promote_latest_model_evidence_bundle.py for a new atomic release"
        )
    output.mkdir(parents=True, exist_ok=True)
    wavefield = output / "01_complete_wavefields"
    for family in ("uniform", "layered", "marmousi"):
        _link(
            ROOT / f"results/wavefield_viz/full_frames_{family}_w192.npz",
            wavefield / f"full_401_frames_{family}.npz",
        )
    _write_text(
        wavefield / "README.md",
        "# Complete wavefields\n\nEach NPZ contains `times`, `true`, `coarse`, `pred`, coordinates and source metadata for all 401 stored frames. These are historical one-record-per-family development assets, not population estimates. The files are relative links to avoid duplicating about 440 MB.\n",
    )

    snapshot_entries = []
    receiver_entries = []
    for family in ("uniform", "layered", "marmousi"):
        snapshot_entries.extend(
            [
                (f"results/wavefield_viz/pretrained_snapshots_{family}.png", f"historical_pretrained_{family}.png"),
                (
                    f"results/current_best_phase4b_figures_no_water_complex_marmousi_20260808T0342Z/{family}/{family}_wavefield_snapshots.png",
                    f"historical_phase4b_{family}_snapshots.png",
                ),
            ]
        )
        receiver_entries.extend(
            [
                (f"results/wavefield_viz/traces_{family}.png", f"historical_pretrained_{family}_traces.png"),
                (f"results/wavefield_viz/gather_{family}.png", f"historical_pretrained_{family}_gather.png"),
                (
                    f"results/current_best_phase4b_figures_no_water_complex_marmousi_20260808T0342Z/{family}/{family}_receiver_waveforms.png",
                    f"historical_phase4b_{family}_waveforms.png",
                ),
                (
                    f"results/current_best_phase4b_figures_no_water_complex_marmousi_20260808T0342Z/{family}/{family}_receiver_gather.png",
                    f"historical_phase4b_{family}_gather.png",
                ),
            ]
        )
    _copy_many(snapshot_entries, output / "02_wavefield_snapshots")
    _copy(ROOT / "results/wavefield_viz/pretrained_snapshots_relL2.json", output / "02_wavefield_snapshots/pretrained_snapshot_relative_l2.json")
    _copy_many(receiver_entries, output / "03_receiver_waveforms")
    for family in ("uniform", "layered", "marmousi"):
        for suffix in ("wavefield_comparison.pdf", "wavefield_comparison.png"):
            _copy(
                ROOT / f"results/instance_adaptation/pilot_pde_rad/validation_{family}_{'00069' if family == 'uniform' else '00048' if family == 'layered' else '00109'}/{suffix}",
                output / "02_wavefield_snapshots" / f"pde_adaptation_{family}_{suffix}",
            )
        for suffix in ("receiver_waveforms.pdf", "receiver_waveforms.png"):
            _copy(
                ROOT / f"results/instance_adaptation/pilot_pde_rad/validation_{family}_{'00069' if family == 'uniform' else '00048' if family == 'layered' else '00109'}/{suffix}",
                output / "03_receiver_waveforms" / f"pde_adaptation_{family}_{suffix}",
            )

    _copy_many(
        (
            ("artifacts/tgrs_dclp_no/dispersion/analytic/fig3a_phase_velocity.pdf", "phase_velocity.pdf"),
            ("artifacts/tgrs_dclp_no/dispersion/analytic/fig3a_phase_velocity.png", "phase_velocity.png"),
            ("artifacts/tgrs_dclp_no/dispersion/analytic/phase_velocity.csv", "phase_velocity.csv"),
            ("artifacts/tgrs_dclp_no/dispersion/analytic/phase_velocity_summary.json", "phase_velocity_summary.json"),
            ("artifacts/tgrs_dclp_no/comparison/comparison.json", "historical_three_record_comparison.json"),
            ("artifacts/tgrs_dclp_no/comparison/claim_gate.json", "learned_dispersion_claim_gate.json"),
        ),
        output / "04_numerical_dispersion",
    )
    _write_text(
        output / "04_numerical_dispersion/DISPERSION_CONCLUSION.md",
        "# Numerical-dispersion conclusion\n\nAt four points per wavelength, the analytic maximum phase-velocity errors are 9.49% for FD2, 2.37% for FD4 and 0.337% for LWC-84. This supports using LWC-84 as the numerical reference, but no scheme is dispersion-free. The existing learned-dispersion gate is `false`: receiver-level confirmatory evidence is missing, and the three-record high-band comparison is development evidence only.\n",
    )

    _write_relative_error_assets(output / "05_relative_error")
    _write_cpadc_relative_error_assets(output / "05_relative_error")
    _write_hyperparameters(output / "06_network_hyperparameters")
    _copy_many(
        (
            ("configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5b_train_gate_record_l2_diag.yaml", "r5b_selected_config.yaml"),
            ("configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5d_metric_aligned_time_train_diag.yaml", "r5d_diagnostic_config.yaml"),
            ("configs/saved_time_v4/generated/local_field_w128_temporal_latent_a3_rank32_r5e_true_metric_aligned_time_train_diag.yaml", "r5e_registered_config.yaml"),
            ("configs/baselines/patch_deeponet_fixed_train_gate_20260813.yaml", "patch_deeponet_registered_config.yaml"),
        ),
        output / "06_network_hyperparameters/configs",
    )
    _copy_many(
        (
            ("paper/tgrs_helmholtz_operator/marmousi_fixed_frequency_source_ood_protocol_20260813.json", "fixed19hz_position_protocol.json"),
            ("paper/tgrs_helmholtz_operator/marmousi_multisource_figure_plan_20260813.md", "fixed19hz_position_figure_plan.md"),
            ("paper/tgrs_helmholtz_operator/confirmatory_experiment_protocol_20260813.json", "confirmatory_experiment_protocol.json"),
            ("results/marmousi_fixed19hz_position_240_post_r5d_preregistration_20260813.json", "position_240_preregistration.json"),
            ("paper/tgrs_helmholtz_operator/patch_deeponet_confirmatory_protocol_20260813.json", "patch_deeponet_protocol.json"),
        ),
        output / "07_protocols_and_figure_plan",
    )
    _copy_many(
        (
            ("results/relative_error_plateau_diagnosis_20260813.md", "relative_error_plateau_diagnosis.md"),
            ("results/r5d_time_selector_alignment_audit_20260813.json", "r5d_time_selector_alignment_audit.json"),
            ("results/current_acoustic_vds_dependency_audit_20260813.json", "current_vds_dependency_audit.json"),
            ("results/current_dataset_missing_file_diagnosis_20260813.md", "dataset_missing_file_diagnosis.md"),
            ("results/r5b_latest_model_architecture_audit_20260813.json", "r5b_architecture_audit.json"),
            ("results/r5d_candidate_architecture_audit_20260813.json", "r5d_architecture_audit.json"),
            ("results/patch_deeponet_static_audit_20260813.json", "patch_deeponet_static_audit.json"),
            ("results/patch_deeponet_training_static_audit_20260813.json", "patch_deeponet_training_audit.json"),
        ),
        output / "08_reproducibility_and_diagnostics",
    )
    _link(POSITION_OUTPUT, output / "09_pending_position_evaluation/live_output_read_only")
    _write_text(
        output / "09_pending_position_evaluation/README.md",
        "# Pending fixed-19-Hz position evaluation\n\nThe linked output is intentionally pending while r5d finishes naturally. The registered pipeline will evaluate 30 unseen Marmousi slices at eight source positions (240 cases), render snapshots at 0.2/0.6/1.0 s and shallow receiver gathers, and write per-record scores only after predictions are sealed. Frequency is fixed at 19 Hz; frequency generalization is outside scope.\n",
    )
    training_links = output / "10_training_runs_read_only"
    r5c_run = Path(
        "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/"
        "pretraining/local_field_w128_hicap/"
        "temporal_latent_a3_rank32_r5c_shared_dynamic_adapter_train_diag/run"
    )
    _link(R5B_RUN, training_links / "r5b_selected_run")
    _link(r5c_run, training_links / "r5c_completed_diagnostic_run")
    _link(R5D_RUN, training_links / "r5d_active_diagnostic_run")
    _write_text(
        training_links / "README.md",
        "# Training-run links\n\nThese relative links expose the canonical training logs and checkpoint metadata without duplicating weights. r5b is the selected train-gate checkpoint. r5c completed naturally with no accepted epoch. r5d remains diagnostic because its time-selector seeds were misaligned; its link may continue to acquire logs while the job runs naturally. No process is controlled by this bundle.\n",
    )
    _write_historical_registry(output)
    _write_status_and_readme(output)
    _write_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
