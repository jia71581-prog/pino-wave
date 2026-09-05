#!/usr/bin/env python3
"""Atomically promote a better latest-model evidence bundle.

The existing bundle is left untouched unless the fixed-15-Hz same-sample gate
passes and the complete fixed-19-Hz position score is bound to the identical
checkpoint.  On promotion the previous bundle is retained as a sibling archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813"
DEFAULT_ARCHIVE = ROOT / "paper/tgrs_helmholtz_operator/experiment_evidence_bundle_20260813_pre_latest_archive"
PROMOTION_ELIGIBLE_SELECTION_SOURCES = {"r5b_preregistered_fallback"}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _relative_link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def validate_promotion_inputs(
    comparison: Mapping[str, Any],
    position_score: Mapping[str, Any],
    checkpoint_selection: Mapping[str, Any],
) -> dict[str, Any]:
    if comparison.get("schema") != "latest_vs_phase4b_fixed15_same_sample_comparison_v1":
        raise ValueError("unexpected same-sample comparison schema")
    if comparison.get("status") != "complete":
        raise ValueError("same-sample comparison is incomplete")
    decision = comparison.get("replacement_decision", {})
    if not isinstance(decision, Mapping):
        raise ValueError("same-sample replacement decision is missing")
    if position_score.get("schema") != "marmousi_fixed_frequency_source_position_scores_v1":
        raise ValueError("unexpected fixed-position score schema")
    if position_score.get("status") != "complete":
        raise ValueError("fixed-position score is incomplete")
    if position_score.get("source_generalization_variable") != "position_only":
        raise ValueError("position score varies more than source position")
    if position_score.get("frequency_generalization_claim_permitted") is not False:
        raise ValueError("position score does not forbid frequency-generalization claims")
    if not np.isclose(float(position_score.get("fixed_source_frequency_hz")), 19.0):
        raise ValueError("position score is not fixed at 19 Hz")
    records = position_score.get("records", [])
    if len(records) != 240 or len({row.get("record_id") for row in records}) != 240:
        raise ValueError("position score is not the complete 240-record census")
    hashes = {
        str(comparison.get("checkpoint_sha256")),
        str(position_score.get("checkpoint_sha256")),
        str(checkpoint_selection.get("checkpoint_sha256")),
    }
    if len(hashes) != 1 or "None" in hashes:
        raise ValueError("comparison, position score, and selection use different checkpoints")
    selection_source = str(checkpoint_selection.get("selection_source"))
    same_sample_passed = bool(decision.get("replacement_gate_passed"))
    selection_eligible = selection_source in PROMOTION_ELIGIBLE_SELECTION_SOURCES
    return {
        "promotion_permitted": bool(same_sample_passed and selection_eligible),
        "checkpoint_sha256": hashes.pop(),
        "selection_source": selection_source,
        "selection_promotion_eligible": selection_eligible,
        "promotion_block_reason": (
            None
            if same_sample_passed and selection_eligible
            else (
                "same_sample_accuracy_gate_failed"
                if not same_sample_passed
                else "selected_checkpoint_is_diagnostic_only"
            )
        ),
        "same_sample_gate": decision,
        "position_record_count": len(records),
    }


def _write_position_distribution(score: Mapping[str, Any], destination: Path) -> None:
    rows = score["records"]
    destination.mkdir(parents=True, exist_ok=True)
    fields = (
        "record_id",
        "slice_rank",
        "case_id",
        "role",
        "record_relative_l2",
        "late_relative_l2",
        "spectrum_high_relative_l2",
        "receiver_relative_l2",
    )
    with (destination / "latest_fixed19hz_all_240_records.csv").open(
        "w", encoding="utf8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "slice_rank": row["slice_rank"],
                    "case_id": row["case_id"],
                    "role": row["role"],
                    **{name: row["metrics"][name] for name in fields[4:]},
                }
            )
    groups = {
        "All (n=240)": rows,
        "Interpolation": [row for row in rows if row["role"] == "interpolation"],
        "Outside range": [
            row for row in rows if row["role"] == "outside_train_position_range"
        ],
    }
    summary = {
        "schema": "latest_fixed19hz_position_relative_error_census_v1",
        "status": "complete",
        "checkpoint_sha256": score["checkpoint_sha256"],
        "fixed_source_frequency_hz": 19.0,
        "source_generalization_variable": "position_only",
        "frequency_generalization_claim_permitted": False,
        "independent_velocity_slices": 30,
        "repeated_source_positions_per_slice": 8,
        "groups": {},
    }
    values = []
    for label, selected in groups.items():
        array = np.asarray(
            [row["metrics"]["record_relative_l2"] for row in selected],
            dtype=np.float64,
        )
        values.append(array)
        summary["groups"][label] = {
            "count": len(array),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "q1": float(np.quantile(array, 0.25)),
            "q3": float(np.quantile(array, 0.75)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
        }
    _atomic_json(summary, destination / "latest_fixed19hz_all_240_records_summary.json")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    boxes = axis.boxplot(
        values,
        labels=list(groups),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
    )
    colors = ("#99AABB", "#5185C0", "#E99D4E")
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.72)
    rng = np.random.default_rng(20260813)
    for index, (array, color) in enumerate(zip(values, colors, strict=True), start=1):
        jitter = rng.uniform(-0.15, 0.15, len(array))
        axis.scatter(
            np.full(len(array), index) + jitter,
            array,
            s=9,
            alpha=0.5,
            color=color,
            edgecolor="none",
        )
    axis.set_ylabel("Complete-transient relative $L_2$")
    axis.set_title("Latest model: fixed-19-Hz source-position control")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination / "latest_fixed19hz_all_240_records_boxplot.png", dpi=300)
    fig.savefig(destination / "latest_fixed19hz_all_240_records_boxplot.pdf")
    plt.close(fig)


def _manifest(bundle: Path) -> None:
    rows = []
    for path in sorted(bundle.rglob("*")):
        if path.name == "FILE_MANIFEST.csv" or path.is_dir():
            continue
        target = path.resolve()
        if path.is_symlink() and not target.exists():
            rows.append(
                (
                    str(path.relative_to(bundle)),
                    "symlink",
                    str(target),
                    0,
                    "future_archive_target_created_by_atomic_swap",
                )
            )
            continue
        size = target.stat().st_size
        rows.append(
            (
                str(path.relative_to(bundle)),
                "symlink" if path.is_symlink() else "file",
                str(target),
                size,
                _sha256(target) if size <= 20 * 1024 * 1024 else "not_hashed_over_20MiB",
            )
        )
    with (bundle / "FILE_MANIFEST.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("bundle_path", "kind", "resolved_source", "byte_count", "sha256_or_policy"))
        writer.writerows(rows)


def stage_bundle(
    *,
    current_bundle: Path,
    archive_bundle: Path,
    comparison_dir: Path,
    position_dir: Path,
    selection: Mapping[str, Any],
    decision: Mapping[str, Any],
    staging: Path,
) -> None:
    shutil.copytree(current_bundle, staging, symlinks=True)
    # Only the Phase4b wavefield/receiver assets are superseded.  The existing
    # relative-error directory also contains the still-current r5b train gate
    # and the separately scoped CPADC validation/test_id census; the network
    # directory contains parameter-matched baseline designs.  Preserve those
    # records and add the sealed position result plus selected identity below.
    for name in (
        "01_complete_wavefields",
        "02_wavefield_snapshots",
        "03_receiver_waveforms",
        "09_pending_position_evaluation",
    ):
        shutil.rmtree(staging / name, ignore_errors=False)
    for family in ("uniform", "layered", "marmousi"):
        source = comparison_dir / family
        _relative_link(
            source / f"{family}_complete_wavefield.npz",
            staging / "01_complete_wavefields" / f"latest_full_401_frames_{family}.npz",
        )
        for suffix in ("png", "pdf"):
            _copy(
                source / f"{family}_wavefield_snapshots.{suffix}",
                staging / "02_wavefield_snapshots" / f"latest_{family}_wavefield_snapshots.{suffix}",
            )
            _copy(
                source / f"{family}_receiver_waveforms.{suffix}",
                staging / "03_receiver_waveforms" / f"latest_{family}_receiver_waveforms.{suffix}",
            )
            _copy(
                source / f"{family}_receiver_gather.{suffix}",
                staging / "03_receiver_waveforms" / f"latest_{family}_receiver_gather.{suffix}",
            )
    _copy(comparison_dir / "comparison_report.json", staging / "01_complete_wavefields/latest_vs_phase4b_comparison_report.json")
    _copy(comparison_dir / "prediction_manifest.json", staging / "01_complete_wavefields/latest_prediction_manifest.json")
    (staging / "01_complete_wavefields/README.md").write_text(
        "# Latest-model complete wavefields\n\nEach linked NPZ contains all 401 reference and latest-model frames for the exact fixed-15-Hz sample used in the Phase4b comparison. Prediction was sealed before target access. These are same-sample development comparisons, not population estimates.\n",
        encoding="utf8",
    )
    _write_position_distribution(
        json.loads((position_dir / "score_summary.json").read_text(encoding="utf8")),
        staging / "05_relative_error",
    )
    _copy(position_dir / "score_summary.json", staging / "05_relative_error/latest_fixed19hz_score_summary.json")
    _copy(position_dir / "checkpoint_selection.json", staging / "06_network_hyperparameters/checkpoint_selection.json")
    selected_config = Path(selection["config"])
    selected_identity = Path(selection["checkpoint_identity"])
    _copy(selected_config, staging / "06_network_hyperparameters/latest_selected_config.yaml")
    _copy(selected_identity, staging / "06_network_hyperparameters/latest_selected_run_identity.json")
    (staging / "06_network_hyperparameters/LATEST_SELECTED_MODEL.md").write_text(
        f"# Latest selected model\n\nSelection source: `{selection['selection_source']}`.\n\nCheckpoint SHA-256: `{selection['checkpoint_sha256']}`.\n\nSelection used train-only gate evidence. Detailed configuration and run identity are copied alongside this file.\n",
        encoding="utf8",
    )
    evaluation = staging / "09_fixed19_position_evaluation"
    _relative_link(position_dir, evaluation / "complete_output_read_only")
    _copy(position_dir / "score_summary.json", evaluation / "score_summary.json")
    _copy(position_dir / "checkpoint_selection.json", evaluation / "checkpoint_selection.json")
    figures = position_dir / "figures"
    for path in sorted(figures.glob("*.pdf")):
        _copy(path, evaluation / "figures" / path.name)
    (evaluation / "README.md").write_text(
        "# Complete fixed-19-Hz position evaluation\n\nThis is the complete 30-slice by 8-position census for the same checkpoint used by every latest-model asset in this bundle. Source frequency is fixed at 19 Hz; only position generalization is in scope.\n",
        encoding="utf8",
    )
    old_link = staging / "99_historical_experiment_index/pre_latest_bundle_read_only"
    old_link.parent.mkdir(parents=True, exist_ok=True)
    old_link.symlink_to(
        os.path.relpath(archive_bundle.resolve(), old_link.parent.resolve())
    )
    status_rows = [
        ("complete wavefields", "latest model", "three fixed-15-Hz same-sample development cases", "01_complete_wavefields"),
        ("wavefield snapshots", "latest model", "same checkpoint and samples as complete wavefields", "02_wavefield_snapshots"),
        ("receiver waveforms and gathers", "latest model", "same checkpoint and samples as complete wavefields", "03_receiver_waveforms"),
        ("analytic numerical dispersion", "available", "FD2/FD4/LWC-84 analytic context", "04_numerical_dispersion"),
        ("all-record relative-error boxplot", "latest model", "240 fixed-19-Hz records; 30 independent slices", "05_relative_error"),
        ("r5b train-gate relative errors", "retained", "48 train-only selection records", "05_relative_error"),
        ("CPADC relative errors", "retained separate contribution", "480 validation plus 480 independent test_id records", "05_relative_error"),
        ("network hyperparameters", "latest selected model", str(selection["selection_source"]), "06_network_hyperparameters"),
        ("fixed-19-Hz position evaluation", "complete", "position only; 30 slices x 8 positions", "09_fixed19_position_evaluation"),
        ("historical model assets", "archived", "recoverable sibling bundle", "99_historical_experiment_index"),
    ]
    with (staging / "EVIDENCE_STATUS.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("artifact", "status", "evidence_scope", "directory"))
        writer.writerows(status_rows)
    (staging / "README.md").write_text(
        f"# TGRS latest-model experiment evidence bundle\n\nAll active surrogate wavefield, receiver, and source-position assets in this directory use one checkpoint: `{selection['checkpoint_sha256']}` selected as `{selection['selection_source']}`. The strict same-sample replacement gate passed before publication. Historical model figures and statistics remain recoverable in `{archive_bundle.name}` and are not mixed into the active figure directories. The CPADC validation/test_id census is retained as a separately labelled deployment contribution.\n\nThe main figure claim is source-position generalization on 30 unseen Marmousi slices at eight positions with frequency fixed at 19 Hz. The three-family fixed-15-Hz panels are same-sample qualitative comparisons and belong in the supplement. No frequency-generalization claim is permitted.\n",
        encoding="utf8",
    )
    (staging / "README_zh.md").write_text(
        f"# TGRS 最新模型实验结果证据包\n\n本目录有效的代理波场、接收器与震源位置结果均来自同一个检查点：`{selection['checkpoint_sha256']}`，选模来源为 `{selection['selection_source']}`。只有在三类同样本完整波场和接收器严格替换门槛通过后才发布。历史模型资产完整保存在同级目录 `{archive_bundle.name}`，不会与最新模型图混用。CPADC 验证与独立 test_id 统计作为单独标注的部署贡献保留。\n\n主文图的单一主张是：固定 19 Hz 时，模型在 30 个未见 Marmousi 切片、每个 8 个震源位置上的位置泛化。固定 15 Hz 的三类同样本图是补充材料的定性比较。不得声称频率泛化。\n",
        encoding="utf8",
    )
    _atomic_json(
        {
            "schema": "latest_model_bundle_promotion_v1",
            "status": "promoted",
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "selection_source": selection["selection_source"],
            "decision": decision,
            "historical_bundle_archive": str(archive_bundle.resolve()),
        },
        staging / "LATEST_MODEL_PROMOTION.json",
    )
    _manifest(staging)


def record_nonpromotion_evidence(
    *,
    bundle: Path,
    comparison_dir: Path,
    position_dir: Path,
    selection: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> None:
    """Record a completed test without replacing the better historical assets."""

    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    if (bundle / "LATEST_MODEL_PROMOTION.json").exists():
        raise FileExistsError("refusing to mix a non-promotion report into a promoted bundle")
    score = json.loads((position_dir / "score_summary.json").read_text(encoding="utf8"))
    _write_position_distribution(score, bundle / "05_relative_error")

    evaluation = bundle / "09_pending_position_evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    live_link = evaluation / "live_output_read_only"
    if not live_link.exists():
        _relative_link(position_dir, live_link)
    for name in (
        "checkpoint_selection.json",
        "prediction_manifest.json",
        "reference_manifest.json",
        "score_summary.json",
    ):
        _copy(position_dir / name, evaluation / name)
    for path in sorted((position_dir / "figures").glob("*.pdf")):
        _copy(path, evaluation / "figures" / path.name)
    _atomic_text(
        "# Complete fixed-19-Hz position evaluation\n\n"
        "The directory name is retained for path stability, but the registered "
        "evaluation is complete: 30 unseen Marmousi slices by eight source "
        "positions (240 records), all at fixed 19 Hz. Predictions were sealed "
        "before reference generation. These results characterize the tested r5b "
        "checkpoint; they did not trigger replacement of the more accurate "
        "historical same-sample model.\n",
        evaluation / "README.md",
    )

    comparison = bundle / "11_latest_candidate_nonpromotion_test"
    comparison.mkdir(parents=True, exist_ok=True)
    _relative_link(comparison_dir, comparison / "complete_output_read_only")
    _copy(comparison_dir / "comparison_report.json", comparison / "comparison_report.json")
    _copy(comparison_dir / "prediction_manifest.json", comparison / "prediction_manifest.json")
    _atomic_json(
        {
            "schema": "latest_model_bundle_promotion_decision_v1",
            "status": "complete",
            **decision,
        },
        comparison / "nonpromotion_decision.json",
    )
    _atomic_text(
        "# Latest-candidate non-promotion test\n\n"
        "The selected train-gate checkpoint was evaluated on the exact fixed-15-Hz "
        "Uniform, Layered, and Marmousi samples used by the historical Phase4b "
        "figures. It failed the strict all-family full-wavefield and receiver gate, "
        "so active historical assets were not replaced. The linked output retains "
        "all complete fields, snapshots, receiver figures, prediction seals, and "
        "metrics for audit.\n",
        comparison / "README.md",
    )

    status_path = bundle / "EVIDENCE_STATUS.csv"
    rows = list(csv.DictReader(status_path.open(encoding="utf8")))
    rows = [
        {
            **row,
            "status": "complete",
            "evidence_scope": "240 fixed-19-Hz position-only records; not a promotion gate",
        }
        if row["artifact"] == "240-case fixed-19-Hz position boxplot"
        else row
        for row in rows
    ]
    rows.append(
        {
            "artifact": "latest-candidate same-sample comparison",
            "status": "complete; not promoted",
            "evidence_scope": "three fixed-15-Hz development samples; strict replacement gate failed",
            "directory": "11_latest_candidate_nonpromotion_test",
        }
    )
    with status_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("artifact", "status", "evidence_scope", "directory"))
        writer.writeheader()
        writer.writerows(rows)

    report = json.loads((comparison_dir / "comparison_report.json").read_text(encoding="utf8"))
    position_mean = float(score["overall"]["record_relative_l2"]["mean"])
    summary = (
        "\n## Completed latest-candidate tests (2026-08-13)\n\n"
        f"The fixed-19-Hz position-only census is complete for 240 records and has "
        f"mean complete-transient relative L2 `{position_mean:.12g}`. The exact "
        "same-sample replacement gate against Phase4b failed for all three medium "
        "families, so no historical model asset was replaced. See "
        "`09_pending_position_evaluation` (stable legacy directory name) and "
        "`11_latest_candidate_nonpromotion_test`.\n"
    )
    readme = (bundle / "README.md").read_text(encoding="utf8")
    if "## Completed latest-candidate tests (2026-08-13)" not in readme:
        _atomic_text(readme.rstrip() + "\n" + summary, bundle / "README.md")
    summary_zh = (
        "\n## 最新候选模型测试已完成（2026-08-13）\n\n"
        f"固定 19 Hz、仅改变震源位置的 240 条测试已完成，完整瞬态相对 L2 "
        f"均值为 `{position_mean:.12g}`。与 Phase4b 的三类同样本严格替换门槛 "
        "全部未通过，因此没有替换历史优胜模型资产。完整结果见 "
        "`09_pending_position_evaluation`（为保持路径稳定而保留的旧目录名）和 "
        "`11_latest_candidate_nonpromotion_test`。\n"
    )
    readme_zh = (bundle / "README_zh.md").read_text(encoding="utf8")
    if "## 最新候选模型测试已完成（2026-08-13）" not in readme_zh:
        _atomic_text(readme_zh.rstrip() + "\n" + summary_zh, bundle / "README_zh.md")
    _manifest(bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--position-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--decision-output", type=Path)
    args = parser.parse_args()
    comparison_dir = args.comparison_dir.resolve()
    position_dir = args.position_dir.resolve()
    bundle = args.bundle.resolve()
    archive = args.archive.resolve()
    comparison = json.loads((comparison_dir / "comparison_report.json").read_text(encoding="utf8"))
    score = json.loads((position_dir / "score_summary.json").read_text(encoding="utf8"))
    selection = json.loads((position_dir / "checkpoint_selection.json").read_text(encoding="utf8"))
    decision = validate_promotion_inputs(comparison, score, selection)
    decision_output = args.decision_output or comparison_dir / "bundle_promotion_decision.json"
    _atomic_json(
        {"schema": "latest_model_bundle_promotion_decision_v1", "status": "complete", **decision},
        decision_output,
    )
    if not decision["promotion_permitted"]:
        record_nonpromotion_evidence(
            bundle=bundle,
            comparison_dir=comparison_dir,
            position_dir=position_dir,
            selection=selection,
            decision=decision,
        )
        print(json.dumps({"status": "not_promoted", **decision}, sort_keys=True))
        return
    if not bundle.is_dir():
        raise FileNotFoundError(bundle)
    if archive.exists():
        raise FileExistsError(f"historical bundle archive already exists: {archive}")
    staging = bundle.with_name(f".{bundle.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        stage_bundle(
            current_bundle=bundle,
            archive_bundle=archive,
            comparison_dir=comparison_dir,
            position_dir=position_dir,
            selection=selection,
            decision=decision,
            staging=staging,
        )
        os.replace(bundle, archive)
        try:
            os.replace(staging, bundle)
        except BaseException:
            os.replace(archive, bundle)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"status": "promoted", **decision, "bundle": str(bundle), "archive": str(archive)}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["record_nonpromotion_evidence", "validate_promotion_inputs"]
