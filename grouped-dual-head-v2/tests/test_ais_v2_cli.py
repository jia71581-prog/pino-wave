"""Contracts for the serial AIS normalization-v2 screen runner."""

from __future__ import annotations

import hashlib
import csv
import copy
import io
import json
import os
import random
import sys
from pathlib import Path

import pytest
import torch
import yaml
import subprocess
import h5py
import numpy as np

import scripts.run_ais_v2_screen as screen_cli
import scripts.train_ais_mqfno as train_cli
import scripts.prepare_ais_b1_baseline as baseline_cli
import scripts.evaluate_ais_mqfno as evaluate_cli
from fno_acoustic.ais_dataset_binding import (
    canonical_json_bytes as canonical_manifest_bytes,
    compute_dataset_manifest,
    verified_ids_sha256,
)
from fno_acoustic.query_census import (
    SAMPLE_COLUMNS,
    SCREEN_NUMERIC_COLUMNS,
    SCREEN_SAMPLE_COLUMNS,
    aggregate_category_metrics,
    aggregate_screen_category_metrics,
)
from scripts.train_ais_mqfno import _load_validation_site_manifest
from scripts.run_ais_v2_screen import (
    REQUIRED_ARTIFACT_FIELDS,
    ScreenLock,
    atomic_write_json,
    build_screen_plan,
    canonical_sha256,
    census_candidate_artifacts,
    validate_completed_artifacts,
)
from scripts.train_ais_mqfno import _load_overfit_sites


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs/ais_zero_collapse_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_binding_dataset(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        handle["tensor"] = np.zeros((2, 160, 2, 2), dtype=np.float32)
        handle["nu"] = np.ones((2, 2, 2), dtype=np.float32)
        handle["source_mask"] = np.ones((2, 2, 2), dtype=np.float32)
        handle["model_type"] = np.asarray([b"uniform", b"layered"])
        handle["t-coordinate"] = np.linspace(0.0, 1.2, 160)
        handle["x-coordinate"] = np.asarray([0.1, 0.2])
        handle["y-coordinate"] = np.asarray([0.3, 0.4])
    return path


def test_dry_run_orders_candidates_and_never_mentions_test_split(tmp_path: Path) -> None:
    plan = build_screen_plan(CONFIG_DIR, tmp_path)

    assert [item.candidate_id for item in plan.gate_o] == [f"N{i}" for i in range(8)]
    assert plan.review_boundary == "O"
    assert all("test" not in " ".join(item.command).lower() for item in plan.all_commands)
    assert all("--device" in item.command and "cuda" in item.command for item in plan.all_commands)
    assert all("--max-train-batches" in item.command and "400" in item.command for item in plan.gate_o)
    assert all("--overfit-sample-id" in item.command and "2" in item.command for item in plan.gate_o)
    assert all("--overfit-site-manifest" in item.command for item in plan.gate_o)
    assert not list(tmp_path.iterdir())


def test_completed_row_rejects_missing_or_extra_fields(tmp_path: Path) -> None:
    row = _fake_completed_row(tmp_path, "H1", "N0", 600)
    for mutation in ({key: value for key, value in row.items() if key != "command"}, {**row, "forged": True}):
        with pytest.raises(ValueError, match="schema"):
            validate_completed_artifacts(mutation)


def test_resume_rejects_changed_checkpoint_or_metrics(tmp_path: Path) -> None:
    row = _fake_completed_row(tmp_path, "H1", "N0", 600)
    checkpoint = Path(row["last_checkpoint"])
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        validate_completed_artifacts(row)


def test_resume_rejects_hash_drifted_screen_samples_csv(tmp_path: Path) -> None:
    row = _fake_completed_row(tmp_path, "H1", "N0", 600)
    samples = Path(row["samples_path"])
    samples.write_bytes(samples.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="hash mismatch: samples_path"):
        validate_completed_artifacts(row)


def test_resume_rejects_hash_valid_external_or_symlinked_artifact_path(
    tmp_path: Path,
) -> None:
    row = _fake_completed_row(tmp_path, "O", "N0", 400)
    manifest = {
        "artifacts": {"O": {"N0": row}},
        "decisions": {},
        "stage": {"gate": "O", "status": "in_progress"},
    }
    external = tmp_path / "outside.pt"
    external.write_bytes(Path(row["last_checkpoint"]).read_bytes())
    row["last_checkpoint"] = str(external)
    row["last_checkpoint_sha256"] = _sha(external)
    with pytest.raises(ValueError, match="canonical"):
        build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

    row = _fake_completed_row(tmp_path, "O", "N0", 400)
    canonical = Path(row["last_checkpoint"])
    target = tmp_path / "target.pt"
    canonical.replace(target)
    canonical.symlink_to(target)
    manifest["artifacts"] = {"O": {"N0": row}}
    with pytest.raises(ValueError, match="symlink"):
        build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)


def test_atomic_json_failure_does_not_replace_published_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "screen_manifest.json"
    atomic_write_json(path, {"generation": 1})
    original = path.read_bytes()

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(path, {"generation": 2})

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".screen_manifest.json.*.tmp"))


def test_screen_lock_rejects_second_live_child(tmp_path: Path) -> None:
    path = tmp_path / ".screen.lock"
    with ScreenLock(path):
        with pytest.raises(RuntimeError, match="live"):
            with ScreenLock(path):
                pass


def test_screen_lock_is_inherited_by_child_across_runner_death(tmp_path: Path) -> None:
    path = tmp_path / ".screen.lock"
    lock = ScreenLock(path)
    lock.__enter__()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.3)"],
        pass_fds=(lock.fd,),
    )
    lock.__exit__()

    with pytest.raises(RuntimeError, match="held"):
        with ScreenLock(path):
            pass
    child.wait()
    with ScreenLock(path):
        pass
    assert path.is_file()


def test_manifest_row_requires_complete_provenance() -> None:
    required = {
        "source_commit",
        "source_tree_sha256",
        "dirty_entries_sha256",
        "command",
        "candidate_id",
        "config_sha256",
        "split_sha256",
        "normalization_stats_sha256",
        "last_checkpoint_sha256",
        "best_checkpoint_sha256",
        "metrics_sha256",
        "runtime_seed",
        "parent_checkpoint_sha256",
        "optimizer_updates",
        "parameter_count",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "wall_seconds",
        "gpu_name",
    }
    assert required <= REQUIRED_ARTIFACT_FIELDS


def _registered_overfit_inputs() -> tuple[dict[str, object], dict[str, list[int]], Path]:
    config = yaml.safe_load((CONFIG_DIR / "n0_norm.yaml").read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["data"]["split_manifest"]
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    manifest = PROJECT_ROOT / config["screen"]["gate_o_site_manifest"]
    return config, splits, manifest


def test_gate_o_overfit_uses_exact_registered_train_scene_and_sites() -> None:
    config, splits, manifest = _registered_overfit_inputs()

    sites = _load_overfit_sites(config, splits, 2, manifest, 400, None)

    assert set(sites) == {2}
    assert isinstance(sites[2], torch.Tensor)
    assert sites[2].dtype == torch.long
    assert sites[2].numel() == 2048
    assert 2 in splits["train"] and 2 not in splits["val"] + splits["test"]


@pytest.mark.parametrize("batches", [1, 399, 401])
def test_gate_o_overfit_rejects_nonexact_update_budget(batches: int) -> None:
    config, splits, manifest = _registered_overfit_inputs()

    with pytest.raises(ValueError, match="400"):
        _load_overfit_sites(config, splits, 2, manifest, batches, None)


def test_gate_o_overfit_rejects_validation_limit_or_nonregistered_scene() -> None:
    config, splits, manifest = _registered_overfit_inputs()

    with pytest.raises(ValueError, match="max-val"):
        _load_overfit_sites(config, splits, 2, manifest, 400, 1)
    with pytest.raises(ValueError, match="sample"):
        _load_overfit_sites(config, splits, splits["val"][0], manifest, 400, None)


def test_gate_o_overfit_rejects_manifest_hash_tamper(tmp_path: Path) -> None:
    config, splits, manifest = _registered_overfit_inputs()
    copied = copy.deepcopy(config)
    changed = tmp_path / manifest.name
    changed.write_bytes(manifest.read_bytes() + b" ")
    copied["screen"]["gate_o_site_manifest"] = str(changed)

    with pytest.raises(ValueError, match="SHA-256"):
        _load_overfit_sites(copied, splits, 2, changed, 400, None)


def test_h1_h3_validation_loads_exact_registered_64x64_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, splits, _ = _registered_overfit_inputs()
    manifest_path = PROJECT_ROOT / config["screen"]["validation_site_manifest"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        torch,
        "linspace",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("registered validation must not regenerate sites")
        ),
    )

    sites, digest = _load_validation_site_manifest(config, splits)

    assert digest == config["screen"]["validation_site_manifest_sha256"]
    assert set(sites) == set(splits["val"])
    assert torch.equal(sites[splits["val"][0]], torch.tensor(payload["site_indices"]))
    assert sites[splits["val"][0]].numel() == 2048


def test_h1_h3_validation_manifest_hash_tamper_is_fatal(tmp_path: Path) -> None:
    config, splits, _ = _registered_overfit_inputs()
    copied = copy.deepcopy(config)
    source = PROJECT_ROOT / config["screen"]["validation_site_manifest"]
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b" ")
    copied["screen"]["validation_site_manifest"] = str(changed)

    with pytest.raises(ValueError, match="validation site manifest SHA-256"):
        _load_validation_site_manifest(copied, splits)


def test_halving_and_native_census_commands_are_distinct() -> None:
    screen = screen_cli._command("N0", "H1", "evaluate", 600, "cpu")
    assert "--screen-validation-site-manifest" in screen.command
    assert "screen64_fixed2048" in screen.command
    assert "native400_final_census" not in screen.command

    native = screen_cli._command("N0", "H3", "evaluate_native", 3000, "cpu")
    assert "--screen-validation-site-manifest" not in native.command
    assert "native400_final_census" in native.command
    assert "evaluation/native400" in " ".join(native.command)


def test_h3_plan_evaluates_both_survivors_on_screen_and_native_paths(
    tmp_path: Path,
) -> None:
    commands = screen_cli._candidate_stage_commands(
        "N0", "H3", "cpu", CONFIG_DIR, tmp_path, "a" * 64,
        "<OUTPUT_DIR>/h2/N0/checkpoints/last.pt",
    )
    assert [item.kind for item in commands] == [
        "train", "evaluate", "evaluate_native"
    ]


@pytest.mark.parametrize("gate", ["H1", "H2"])
def test_recovery_complete_screen_publishes_without_rerunning_evaluation(
    tmp_path: Path, gate: str,
) -> None:
    update = screen_cli.GATE_UPDATES[gate]
    _fake_completed_row(tmp_path, gate, "N0", update)

    commands = screen_cli._candidate_stage_commands(
        "N0", gate, "cpu", CONFIG_DIR, tmp_path, None, None,
    )

    assert [item.kind for item in commands] == ["publish"]


def test_recovery_h3_complete_screen_runs_only_missing_native_census(
    tmp_path: Path,
) -> None:
    _fake_completed_row(tmp_path, "H3", "N0", 3000, "a" * 64)
    native = tmp_path / "h3/N0/evaluation/native400"
    for path in native.iterdir():
        path.unlink()
    native.rmdir()

    commands = screen_cli._candidate_stage_commands(
        "N0", "H3", "cpu", CONFIG_DIR, tmp_path, "a" * 64, None,
    )

    assert [item.kind for item in commands] == ["evaluate_native"]


def test_recovery_h3_complete_screen_and_native_publishes_without_child(
    tmp_path: Path,
) -> None:
    _fake_completed_row(tmp_path, "H3", "N0", 3000, "a" * 64)

    commands = screen_cli._candidate_stage_commands(
        "N0", "H3", "cpu", CONFIG_DIR, tmp_path, "a" * 64, None,
    )

    assert [item.kind for item in commands] == ["publish"]


def test_recovery_existing_incomplete_evaluation_directory_is_fatal(
    tmp_path: Path,
) -> None:
    _fake_completed_row(tmp_path, "H1", "N0", 600)
    (tmp_path / "h1/N0/evaluation/screen64/samples.csv").unlink()

    with pytest.raises(ValueError, match="incomplete canonical.*screen64"):
        screen_cli._candidate_stage_commands(
            "N0", "H1", "cpu", CONFIG_DIR, tmp_path, None, None,
        )


def test_recovery_evaluation_drift_is_fatal_instead_of_overwritten(
    tmp_path: Path,
) -> None:
    _fake_completed_row(tmp_path, "H1", "N0", 600)
    samples = tmp_path / "h1/N0/evaluation/screen64/samples.csv"
    samples.write_bytes(samples.read_bytes().replace(b"0.2", b"0.8", 1))

    with pytest.raises(ValueError, match="summary metrics differ"):
        screen_cli._candidate_stage_commands(
            "N0", "H1", "cpu", CONFIG_DIR, tmp_path, None, None,
        )


def test_screen_samples_snapshot_is_reaggregated_and_summary_tamper_rejected() -> None:
    provenance = {
        "config_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "normalization_sha256": "d" * 64,
        "validation_site_manifest_sha256": "e" * 64,
    }
    rows = []
    for sample_id, category, value in (
        (1, "uniform", 0.2), (2, "layered", 0.4), (3, "marmousi", 0.6)
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "split": "val",
                "predictor_family": "ais_mqfno",
                **provenance,
                **{
                    name: value for name in SCREEN_NUMERIC_COLUMNS
                },
            }
        )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=SCREEN_SAMPLE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    samples_raw = stream.getvalue().encode()
    aggregated = aggregate_screen_category_metrics(rows)
    summary = {
        "schema_version": 1,
        "purpose": "screen64_fixed2048",
        "split": "val",
        "sample_count": 3,
        "provenance": provenance,
        "category_metrics": aggregated,
    }

    assert screen_cli._validated_evaluation_evidence(
        json.dumps(summary).encode(), samples_raw,
        purpose="screen64_fixed2048", expected_ids={1, 2, 3},
        expected_provenance=provenance,
    ) == aggregated

    rows[1]["relative_l2"] = 9.0
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=SCREEN_SAMPLE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    with pytest.raises(ValueError, match="summary.*CSV"):
        screen_cli._validated_evaluation_evidence(
            json.dumps(summary).encode(), stream.getvalue().encode(),
            purpose="screen64_fixed2048", expected_ids={1, 2, 3},
            expected_provenance=provenance,
        )


def _fake_completed_row(
    root: Path,
    gate: str,
    candidate_id: str,
    update: int,
    parent_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    run = root / gate.lower() / candidate_id
    checkpoint = run / "checkpoints/last.pt"
    best = run / "checkpoints/best.pt"
    lineage = run / "checkpoints/screen_lineage.json"
    metrics = (
        run / "screen_metrics.json"
        if gate == "O"
        else run / "evaluation/screen64/summary.json"
    )
    config_path = CONFIG_DIR / screen_cli.CONFIG_FILENAMES[int(candidate_id[1:])]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["data"]["split_manifest"]
    stats_path = PROJECT_ROOT / config["normalization"]["stats_path"]
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 5,
            "global_step": update,
            "config_sha256": canonical_sha256(config),
            "split_manifest_sha256": _sha(split_path),
            "normalization_stats_sha256": _sha(stats_path),
            "model_state_dict": {"weight": torch.ones(1)},
            "screen_candidate_id": candidate_id,
            "screen_gate": gate,
            "runtime_seed": config["seed"],
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "dataset_binding_sha256": "d" * 64,
            "execution_binding_sha256": "e" * 64,
        },
        checkpoint,
    )
    best.write_bytes(b"best")
    atomic_write_json(
        lineage,
        {
            "runtime_seed": config["seed"],
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
        },
    )
    if gate == "O":
        values = {
            "relative_l2": 0.2,
            "relative_l2_q4": 0.3,
            "prediction_target_norm_ratio": 1.0,
            "prediction_target_pearson": 0.9,
        }
    else:
        values = {
            "relative_l2": 0.2,
            "relative_l2_q4": 0.3,
            "prediction_target_norm_ratio": 1.0,
            "zero_relative_l2": 1.0,
            "zero_relative_l2_q4": 1.0,
        }
    native_values = {
            "relative_l2": 0.2,
            "relative_l2_q4": 0.3,
            "prediction_target_norm_ratio": 1.0,
            "prediction_target_pearson": 0.9,
            **{name: 0.1 for name in screen_cli.LOWER_GUARDS},
            **{name: 0.9 for name in screen_cli.HIGHER_GUARDS},
    }
    metrics_payload = {
        "finite": True,
        "category_metrics": {
            name: dict(values) for name in ("uniform", "layered", "marmousi")
        },
    }
    if gate == "O":
        metrics_payload.update(
            {
                "global_step": update,
                "candidate_id": candidate_id,
                "last_checkpoint_sha256": _sha(checkpoint),
            }
        )
    else:
        split = json.loads(split_path.read_text())
        site_path = PROJECT_ROOT / config["screen"]["validation_site_manifest"]
        screen_provenance = {
            "config_sha256": canonical_sha256(config),
            "checkpoint_sha256": _sha(checkpoint),
            "split_manifest_sha256": _sha(split_path),
            "normalization_sha256": _sha(stats_path),
            "validation_site_manifest_sha256": _sha(site_path),
        }
        screen_rows = [
            {
                "sample_id": sample_id,
                "category": ("uniform", "layered", "marmousi")[index % 3],
                "split": "val",
                "predictor_family": "ais_mqfno",
                **screen_provenance,
                **values,
            }
            for index, sample_id in enumerate(split["val"])
        ]
        screen_aggregated = aggregate_screen_category_metrics(screen_rows)
        samples = run / "evaluation/screen64/samples.csv"
        samples.parent.mkdir(parents=True, exist_ok=True)
        with samples.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCREEN_SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(screen_rows)
        metrics_payload = {
            "schema_version": 1,
            "purpose": "screen64_fixed2048",
            "split": "val",
            "sample_count": len(screen_rows),
            "provenance": screen_provenance,
            "category_metrics": screen_aggregated,
            "dataset_content_root": "f" * 64,
            "verified_sample_count": len(split["val"]),
            "verified_sample_ids_sha256": verified_ids_sha256(split["val"]),
            "verified_sample_scope": "process_unique",
        }
    atomic_write_json(metrics, metrics_payload)
    row = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "dirty_entries_sha256": "c" * 64,
        "dataset_binding_sha256": "d" * 64,
        "execution_binding_sha256": "e" * 64,
        "command": ["fake"],
        "candidate_id": candidate_id,
        "config_path": str(config_path),
        "config_sha256": canonical_sha256(config),
        "split_path": str(split_path),
        "split_sha256": _sha(split_path),
        "normalization_stats_path": str(stats_path),
        "normalization_stats_sha256": _sha(stats_path),
        "optimizer_updates": update,
        "last_checkpoint": str(checkpoint),
        "last_checkpoint_sha256": _sha(checkpoint),
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": _sha(best),
        "metrics_path": str(metrics),
        "metrics_sha256": _sha(metrics),
        "lineage_path": str(lineage),
        "lineage_sha256": _sha(lineage),
        "runtime_seed": config["seed"],
        "parameter_count": 1,
        "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0,
        "wall_seconds": 0.0,
        "gpu_name": "fake",
        "category_metrics": {
            category: {
                name: float(metrics_payload["category_metrics"][category][name])
                for name in (
                    screen_cli.GATE_O_METRICS
                    if gate == "O"
                    else screen_cli.HALVING_METRICS
                )
            }
            for category in ("uniform", "layered", "marmousi")
        },
        "finite": True,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    if gate != "O":
        row.update(
            {
                "evaluation_purpose": "screen64_fixed2048",
                "validation_site_manifest_path": str(site_path),
                "validation_site_manifest_sha256": _sha(site_path),
                "samples_path": str(samples),
                "samples_sha256": _sha(samples),
            }
        )
    if gate == "H3":
        native_summary = run / "evaluation/native400/summary.json"
        native_samples = run / "evaluation/native400/samples.csv"
        native_samples.parent.mkdir(parents=True, exist_ok=True)
        native_provenance = {
            "config_sha256": canonical_sha256(config),
            "checkpoint_sha256": _sha(checkpoint),
            "split_manifest_sha256": _sha(split_path),
            "normalization_sha256": _sha(stats_path),
            "receiver_geometry_sha256": "f" * 64,
        }
        native_rows = []
        for index, sample_id in enumerate(split["val"]):
            native_rows.append(
                {
                    **{name: 0.0 for name in SAMPLE_COLUMNS},
                    "sample_id": sample_id,
                    "category": ("uniform", "layered", "marmousi")[index % 3],
                    "split": "val",
                    "seed": str(config["seed"]),
                    "predictor_family": "ais_mqfno",
                    **native_provenance,
                    **native_values,
                }
            )
        native_aggregated = aggregate_category_metrics(native_rows)
        with native_samples.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS)
            writer.writeheader()
            writer.writerows(native_rows)
        atomic_write_json(
            native_summary,
            {
                "schema_version": 1,
                "purpose": "native400_final_census",
                "split": "val",
                "sample_count": len(native_rows),
                "provenance": native_provenance,
                "category_metrics": native_aggregated,
                "dataset_content_root": "f" * 64,
                "verified_sample_count": len(split["val"]),
                "verified_sample_ids_sha256": verified_ids_sha256(split["val"]),
                "verified_sample_scope": "process_unique",
            },
        )
        row.update(
            {
                "native_metrics_path": str(native_summary),
                "native_metrics_sha256": _sha(native_summary),
                "native_samples_path": str(native_samples),
                "native_samples_sha256": _sha(native_samples),
                "native_category_metrics": {
                    category: {
                        name: float(native_aggregated[category][name])
                        for name in screen_cli.FINAL_CANDIDATE_METRICS
                    }
                    for category in ("uniform", "layered", "marmousi")
                },
            }
        )
    training_summary = {
        "dataset_binding_sha256": "d" * 64,
        "execution_binding_sha256": "e" * 64,
        "parameter_count": 1,
        "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0,
        "wall_seconds": 0.0,
        "gpu_name": "fake",
        "dataset_content_root": "f" * 64,
        "verified_sample_count": 0,
        "verified_sample_ids_sha256": verified_ids_sha256([]),
        "verified_sample_scope": "process_unique",
    }
    if gate != "O":
        training_summary.update(
            {
                "validation_purpose": "screen64_fixed2048",
                "validation_site_manifest_sha256": _sha(site_path),
            }
        )
    atomic_write_json(run / "summary.json", training_summary)
    return row


def _fake_smoke_row(root: Path, candidate_id: str) -> dict[str, object]:
    config_path = CONFIG_DIR / screen_cli.CONFIG_FILENAMES[int(candidate_id[1:])]
    config = yaml.safe_load(config_path.read_text())
    split = PROJECT_ROOT / config["data"]["split_manifest"]
    stats = PROJECT_ROOT / config["normalization"]["stats_path"]
    run = root / "smoke" / candidate_id
    last, best, metrics = (
        run / "checkpoints/last.pt", run / "checkpoints/best.pt", run / "metrics.jsonl"
    )
    last.parent.mkdir(parents=True)
    torch.save(
        {"schema_version": 4, "global_step": 1,
         "config_sha256": canonical_sha256(config),
         "split_manifest_sha256": _sha(split),
         "normalization_stats_sha256": _sha(stats)},
        last,
    )
    best.write_bytes(b"best")
    metrics.write_text(json.dumps({"global_step": 1, "loss": 0.1}) + "\n")
    return {
        "source_commit": "a" * 40, "source_tree_sha256": "b" * 64,
        "dirty_entries_sha256": "c" * 64, "command": ["fake"],
        "candidate_id": candidate_id, "config_sha256": canonical_sha256(config),
        "split_sha256": _sha(split), "normalization_stats_sha256": _sha(stats),
        "last_checkpoint": str(last), "last_checkpoint_sha256": _sha(last),
        "best_checkpoint": str(best), "best_checkpoint_sha256": _sha(best),
        "metrics_path": str(metrics), "metrics_sha256": _sha(metrics),
        "runtime_seed": config["seed"], "optimizer_updates": 1,
        "parameter_count": 1, "peak_gpu_allocated_bytes": 0,
        "peak_gpu_reserved_bytes": 0, "wall_seconds": 0.0, "gpu_name": "fake",
    }


def test_resume_runs_only_next_approved_boundary_and_skips_valid_partial(tmp_path: Path) -> None:
    gate_o_rows = {
        candidate_id: _fake_completed_row(tmp_path, "O", candidate_id, 400)
        for candidate_id in (f"N{i}" for i in range(8))
    }
    partial_h1 = {"N0": _fake_completed_row(tmp_path, "H1", "N0", 600)}
    manifest = {
        "artifacts": {"O": gate_o_rows, "H1": partial_h1},
        "decisions": {},
    }

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

    assert plan.review_boundary == "H1"
    assert not plan.smoke
    assert {item.gate for item in plan.all_commands} == {"H1"}
    assert {item.candidate_id for item in plan.all_commands} == {f"N{i}" for i in range(1, 8)}
    assert len(plan.all_commands) == 14


def test_resume_partial_gate_o_skips_hash_valid_completed_candidate(
    tmp_path: Path,
) -> None:
    manifest = {
        "artifacts": {"O": {"N0": _fake_completed_row(tmp_path, "O", "N0", 400)}},
        "decisions": {},
        "stage": {"gate": "O", "status": "in_progress"},
    }

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

    assert plan.review_boundary == "O"
    assert not plan.smoke
    assert [item.candidate_id for item in plan.gate_o] == [f"N{i}" for i in range(1, 8)]


def test_resume_partial_smoke_skips_completed_smoke_and_still_plans_gate_o(
    tmp_path: Path,
) -> None:
    manifest = {
        "artifacts": {},
        "smokes": {"N0": _fake_smoke_row(tmp_path, "N0")},
        "decisions": {},
        "stage": {"gate": "smoke", "status": "in_progress"},
    }

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

    assert [item.candidate_id for item in plan.smoke] == [f"N{i}" for i in range(1, 8)]
    assert [item.candidate_id for item in plan.gate_o] == [f"N{i}" for i in range(8)]


def test_smoke_census_product_is_directly_resumable(tmp_path: Path) -> None:
    _fake_smoke_row(tmp_path, "N0")
    run = tmp_path / "smoke/N0"
    atomic_write_json(
        run / "summary.json",
        {
            "parameter_count": 1, "peak_gpu_allocated_bytes": 0,
            "peak_gpu_reserved_bytes": 0, "wall_seconds": 0.1, "gpu_name": "cpu",
        },
    )
    item = screen_cli._smoke_command("N0", "cpu")
    row = screen_cli._census_smoke_artifacts(
        item, ["fake"], CONFIG_DIR, tmp_path, "a" * 40, "b" * 64, "c" * 64
    )
    assert set(row) == screen_cli.REQUIRED_SMOKE_FIELDS
    manifest = {
        "artifacts": {}, "smokes": {"N0": row}, "decisions": {},
        "stage": {"gate": "smoke", "status": "in_progress"},
    }
    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest, device="cpu")
    assert "N0" not in {command.candidate_id for command in plan.smoke}


def test_partial_smoke_directory_with_last_checkpoint_plans_finalize(
    tmp_path: Path,
) -> None:
    _fake_smoke_row(tmp_path, "N0")
    manifest = {
        "artifacts": {}, "smokes": {}, "decisions": {},
        "stage": {"gate": "smoke", "status": "in_progress"},
    }
    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest, device="cpu")
    n0 = [command for command in plan.smoke if command.candidate_id == "N0"]
    assert len(n0) == 1
    assert n0[0].kind == "finalize_training"
    assert "--finalize-training-only" in n0[0].command


def test_cli_dry_run_never_writes_or_launches_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run launched a child")

    monkeypatch.setattr(screen_cli.subprocess, "run", forbidden)
    output = tmp_path / "not-created"

    assert screen_cli.main(
        ["--config-dir", str(CONFIG_DIR), "--output-dir", str(output), "--dry-run"]
    ) == 0
    assert not output.exists()
    assert '"review_boundary": "O"' in capsys.readouterr().out


def test_first_child_failure_leaves_resumable_in_progress_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-run"
    source = screen_cli.SourceBinding("a" * 40, "b" * 64, "c" * 64)
    monkeypatch.setattr(screen_cli, "compute_source_binding", lambda: source)
    monkeypatch.setattr(
        screen_cli,
        "load_dataset_binding",
        lambda path: {
            "sha256": "d" * 64,
            "dataset_content_root": "f" * 64,
            "sample_count": 2500,
        },
    )
    monkeypatch.setattr(screen_cli, "compute_execution_binding", lambda device: {"sha256": "e" * 64})

    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(7, ["fake-child"])

    monkeypatch.setattr(screen_cli, "_run_child", fail)
    with pytest.raises(subprocess.CalledProcessError):
        screen_cli.main(["--config-dir", str(CONFIG_DIR), "--output-dir", str(output)])

    manifest = json.loads((output / "screen_manifest.json").read_text())
    assert manifest["stage"] == {"gate": "smoke", "status": "in_progress"}
    assert manifest["artifacts"] == {}


def test_candidate_success_is_durable_before_next_child_failure_and_resume_skips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "durable-run"
    source = screen_cli.SourceBinding("a" * 40, "b" * 64, "c" * 64)
    original_builder = screen_cli.build_screen_plan
    full_plan = original_builder(CONFIG_DIR, output)
    two_candidates = tuple(full_plan.gate_o[:2])
    monkeypatch.setattr(
        screen_cli,
        "build_screen_plan",
        lambda *args, **kwargs: screen_cli.ScreenPlan("O", (), two_candidates),
    )
    monkeypatch.setattr(screen_cli, "compute_source_binding", lambda: source)
    monkeypatch.setattr(
        screen_cli,
        "load_dataset_binding",
        lambda path: {
            "sha256": "d" * 64,
            "dataset_content_root": "f" * 64,
            "sample_count": 2500,
        },
    )
    monkeypatch.setattr(screen_cli, "compute_execution_binding", lambda device: {"sha256": "e" * 64})
    launches = 0

    def launch(command: object, lock: object, expected: object) -> None:
        nonlocal launches
        launches += 1
        if launches == 2:
            raise subprocess.CalledProcessError(9, ["N1"])

    def census(item: object, *args: object, **kwargs: object) -> dict[str, object]:
        return _fake_completed_row(output, "O", item.candidate_id, 400)

    monkeypatch.setattr(screen_cli, "_run_child", launch)
    monkeypatch.setattr(screen_cli, "census_candidate_artifacts", census)

    with pytest.raises(subprocess.CalledProcessError):
        screen_cli.main(["--config-dir", str(CONFIG_DIR), "--output-dir", str(output)])

    manifest_path = output / "screen_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]["O"]) == {"N0"}
    assert manifest["stage"]["completed_candidates"] == ["N0"]
    monkeypatch.setattr(screen_cli, "build_screen_plan", original_builder)
    resumed = original_builder(CONFIG_DIR, output, resume_manifest=manifest)
    assert "N0" not in {item.candidate_id for item in resumed.all_commands}


def test_artifact_census_records_complete_exact_last_provenance(tmp_path: Path) -> None:
    item = build_screen_plan(CONFIG_DIR, tmp_path).gate_o[0]
    config_path = CONFIG_DIR / "n0_norm.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["data"]["split_manifest"]
    stats_path = PROJECT_ROOT / config["normalization"]["stats_path"]
    run = tmp_path / "o/N0"
    last = run / "checkpoints/last.pt"
    best = run / "checkpoints/best.pt"
    last.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 5,
            "global_step": 400,
            "config_sha256": canonical_sha256(config),
            "split_manifest_sha256": _sha(split_path),
            "normalization_stats_sha256": _sha(stats_path),
            "screen_candidate_id": "N0",
            "screen_gate": "O",
            "parent_checkpoint_sha256": None,
            "model_state_dict": {"weight": torch.ones(1)},
            "dataset_binding_sha256": "d" * 64,
            "execution_binding_sha256": "e" * 64,
        },
        last,
    )
    best.write_bytes(b"best-is-recorded-but-never-ranked")
    atomic_write_json(
        run / "checkpoints/screen_lineage.json",
        {"runtime_seed": config["seed"], "parent_checkpoint_sha256": None},
    )
    physical = {
        "relative_l2": 0.2,
        "relative_l2_q4": 0.3,
        "prediction_target_norm_ratio": 1.0,
        "prediction_target_pearson": 0.9,
    }
    atomic_write_json(
        run / "screen_metrics.json",
        {
            "global_step": 400,
            "candidate_id": "N0",
            "last_checkpoint_sha256": _sha(last),
            "finite": True,
            "category_metrics": {
                category: dict(physical)
                for category in ("uniform", "layered", "marmousi")
            },
        },
    )
    atomic_write_json(
        run / "summary.json",
        {
            "parameter_count": 123,
            "peak_gpu_allocated_bytes": 456,
            "peak_gpu_reserved_bytes": 789,
            "wall_seconds": 1.25,
            "gpu_name": "fake-gpu",
            "dataset_binding_sha256": "d" * 64,
            "execution_binding_sha256": "e" * 64,
        },
    )

    row = census_candidate_artifacts(
        item,
        ["fake", "--device", "cuda"],
        CONFIG_DIR,
        tmp_path,
        "a" * 40,
        "b" * 64,
        None,
        "c" * 64,
        "d" * 64,
        "e" * 64,
    )

    assert set(row) == REQUIRED_ARTIFACT_FIELDS
    assert row["optimizer_updates"] == 400
    assert row["last_checkpoint_sha256"] == _sha(last)
    assert row["best_checkpoint_sha256"] == _sha(best)
    assert row["parameter_count"] == 123
    assert row["gpu_name"] == "fake-gpu"

    atomic_write_json(
        run / "screen_metrics.json",
        {
            "global_step": 400,
            "candidate_id": "N0",
            "last_checkpoint_sha256": "0" * 64,
            "finite": True,
            "category_metrics": {
                category: dict(physical)
                for category in ("uniform", "layered", "marmousi")
            },
        },
    )
    with pytest.raises(ValueError, match="exact last checkpoint"):
        census_candidate_artifacts(
            item,
            ["fake", "--device", "cuda"],
            CONFIG_DIR,
            tmp_path,
            "a" * 40,
            "b" * 64,
            None,
            "c" * 64,
            "d" * 64,
            "e" * 64,
        )


def test_source_binding_includes_relevant_untracked_files_and_detects_stage_change(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "src/fno_acoustic").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "configs/ais_zero_collapse_v2").mkdir(parents=True)
    tracked = repo / "scripts/runner.py"
    tracked.write_text("version = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=repo,
        check=True,
    )
    before = screen_cli.compute_source_binding(repo)
    (repo / "scripts/untracked.py").write_text("changed = True\n", encoding="utf-8")
    after = screen_cli.compute_source_binding(repo)

    assert after.source_commit == before.source_commit
    assert after.source_tree_sha256 != before.source_tree_sha256
    assert after.dirty_entries_sha256 != before.dirty_entries_sha256
    with pytest.raises(RuntimeError, match="source tree changed"):
        screen_cli.require_source_unchanged(before, after)


def test_dataset_binding_is_versioned_and_missing_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="dataset_content_manifest"):
        screen_cli.load_dataset_binding(tmp_path)
    manifest = tmp_path / "registration/dataset_content_manifest.json"
    manifest.parent.mkdir()
    config = yaml.safe_load((CONFIG_DIR / "n0_norm.yaml").read_text())
    dataset = _tiny_binding_dataset(tmp_path / "tiny.h5")
    config["data"]["path"] = str(dataset)
    (tmp_path / "n0_norm.yaml").write_text(yaml.safe_dump(config))
    manifest.write_bytes(
        canonical_manifest_bytes(
            {
            "schema_version": 1,
            "dataset_path": config["data"]["path"],
            "sample_count": 2,
            "sample_merkle_root": "a" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="version 2"):
        screen_cli.load_dataset_binding(tmp_path)

    manifest.write_bytes(canonical_manifest_bytes(compute_dataset_manifest(dataset)))
    binding = screen_cli.load_dataset_binding(tmp_path)
    assert binding["sha256"] == _sha(manifest)
    assert binding["dataset_content_root"] == compute_dataset_manifest(dataset)[
        "dataset_content_root"
    ]


def test_runner_census_requires_exact_process_verified_validation_ids(
    tmp_path: Path,
) -> None:
    _fake_completed_row(tmp_path, "H1", "N0", 600)
    item = screen_cli._command("N0", "H1", "evaluate", 600, "cpu")

    row = census_candidate_artifacts(
        item, ["fake"], CONFIG_DIR, tmp_path, "a" * 40, "b" * 64, None,
        "c" * 64, "d" * 64, "e" * 64, "f" * 64, 2500,
    )
    assert row["candidate_id"] == "N0"

    summary = tmp_path / "h1/N0/evaluation/screen64/summary.json"
    payload = json.loads(summary.read_bytes())
    payload["verified_sample_count"] -= 1
    atomic_write_json(summary, payload)
    with pytest.raises(ValueError, match="verified sample"):
        census_candidate_artifacts(
            item, ["fake"], CONFIG_DIR, tmp_path, "a" * 40, "b" * 64, None,
            "c" * 64, "d" * 64, "e" * 64, "f" * 64, 2500,
        )


@pytest.mark.parametrize("entrypoint", ["train", "evaluate"])
def test_formal_v1_manifest_fails_before_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str,
) -> None:
    manifest = tmp_path / "v1.json"
    manifest.write_bytes(
        canonical_manifest_bytes(
            {
                "schema_version": 1,
                "dataset_path": "/does/not/matter.h5",
                "sample_count": 1,
                "sample_merkle_root": "a" * 64,
            }
        )
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(
            AssertionError("CUDA probed before dataset manifest preflight")
        ),
    )
    common = [
        "--config", str(CONFIG_DIR / "n0_norm.yaml"),
        "--output-dir", str(tmp_path / entrypoint),
        "--device", "cuda",
        "--dataset-binding-sha256", _sha(manifest),
        "--execution-binding-sha256", "e" * 64,
        "--dataset-content-manifest", str(manifest),
    ]
    argv = (
        [*common, "--screen-gate", "O", "--max-train-batches", "1"]
        if entrypoint == "train"
        else [
            *common,
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--split", "val",
            "--evaluation-purpose", "native400_final_census",
        ]
    )

    with pytest.raises(ValueError, match="version 2"):
        (train_cli.main if entrypoint == "train" else evaluate_cli.main)(argv)


def test_execution_binding_changes_with_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    before = screen_cli.compute_execution_binding("cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    after = screen_cli.compute_execution_binding("cpu")
    assert before["sha256"] != after["sha256"]


def test_child_source_binding_is_rechecked_after_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = screen_cli.SourceBinding("a" * 40, "b" * 64, "c" * 64)
    after = screen_cli.SourceBinding("a" * 40, "d" * 64, "e" * 64)
    bindings = iter((before, after))

    class FakeProcess:
        pid = 999_999_998

        @staticmethod
        def wait() -> int:
            return 0

    monkeypatch.setattr(screen_cli, "compute_source_binding", lambda: next(bindings))
    monkeypatch.setattr(screen_cli.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    with ScreenLock(tmp_path / ".screen.lock") as lock:
        with pytest.raises(RuntimeError, match="source tree changed"):
            screen_cli._run_child(["fake-child"], lock, before)


def _completed_h3_manifest(root: Path) -> dict[str, object]:
    gate_o = {
        candidate_id: _fake_completed_row(root, "O", candidate_id, 400)
        for candidate_id in (f"N{i}" for i in range(8))
    }
    h1 = {
        candidate_id: _fake_completed_row(root, "H1", candidate_id, 600)
        for candidate_id in (f"N{i}" for i in range(8))
    }
    h2: dict[str, dict[str, object]] = {}
    for candidate_id in (f"N{i}" for i in range(4)):
        row = _fake_completed_row(
            root,
            "H2",
            candidate_id,
            1500,
            h1[candidate_id]["last_checkpoint_sha256"],
        )
        h2[candidate_id] = row
    h3: dict[str, dict[str, object]] = {}
    for candidate_id in ("N0", "N1"):
        row = _fake_completed_row(
            root,
            "H3",
            candidate_id,
            3000,
            h2[candidate_id]["last_checkpoint_sha256"],
        )
        h3[candidate_id] = row
    return {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "dirty_entries_sha256": "c" * 64,
        "artifacts": {"O": gate_o, "H1": h1, "H2": h2, "H3": h3},
        "decisions": {},
        "stage": {"gate": "H3", "status": "awaiting_baseline"},
    }


@pytest.mark.parametrize("gate", ["O", "H1", "H2", "H3"])
def test_complete_records_missing_decision_finalize_same_review_boundary(
    tmp_path: Path, gate: str
) -> None:
    full = _completed_h3_manifest(tmp_path)
    gates = ("O", "H1", "H2", "H3")
    selected = gates[: gates.index(gate) + 1]
    manifest = {
        "artifacts": {name: full["artifacts"][name] for name in selected},
        "decisions": {},
        "stage": {"gate": gate, "status": "in_progress"},
    }

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

    assert plan.review_boundary == gate
    assert plan.action == "finalize_current_stage"
    assert not plan.all_commands
    finalized = screen_cli.finalize_current_stage(manifest, gate)
    if gate == "H3":
        assert finalized["stage"] == {"gate": "H3", "status": "awaiting_baseline"}
        assert "H3" not in finalized["decisions"]
    else:
        assert finalized["stage"] == {"gate": gate, "status": "complete"}
        assert gate in finalized["decisions"]


def _write_unpublished_last(
    root: Path,
    gate: str,
    candidate_id: str,
    global_step: int,
    *,
    config_hash: str | None = None,
    parent_sha256: str | None = None,
) -> Path:
    config = yaml.safe_load(
        (CONFIG_DIR / screen_cli.CONFIG_FILENAMES[int(candidate_id[1:])]).read_text(
            encoding="utf-8"
        )
    )
    split = PROJECT_ROOT / config["data"]["split_manifest"]
    stats = PROJECT_ROOT / config["normalization"]["stats_path"]
    run = root / gate.lower() / candidate_id
    last = run / "checkpoints/last.pt"
    last.parent.mkdir(parents=True)
    torch.save(
        {
            "schema_version": 5,
            "global_step": global_step,
            "config_sha256": config_hash or canonical_sha256(config),
            "split_manifest_sha256": _sha(split),
            "normalization_stats_sha256": _sha(stats),
            "runtime_seed": config["seed"],
            "screen_candidate_id": candidate_id,
            "screen_gate": gate,
            "parent_checkpoint_sha256": parent_sha256,
        },
        last,
    )
    atomic_write_json(
        run / "checkpoints/screen_lineage.json",
        {
            "runtime_seed": config["seed"],
            "parent_checkpoint_sha256": parent_sha256,
        },
    )
    return last


def _h1_in_progress_manifest(root: Path) -> dict[str, object]:
    return {
        "artifacts": {
            "O": {
                candidate_id: _fake_completed_row(root, "O", candidate_id, 400)
                for candidate_id in (f"N{i}" for i in range(8))
            }
        },
        "decisions": {},
        "stage": {"gate": "H1", "status": "in_progress"},
    }


def test_unpublished_partial_last_resumes_only_remaining_updates(tmp_path: Path) -> None:
    manifest = _h1_in_progress_manifest(tmp_path)
    last = _write_unpublished_last(tmp_path, "H1", "N0", 300)

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)
    n0 = [item for item in plan.all_commands if item.candidate_id == "N0"]

    assert [item.kind for item in n0] == ["train", "evaluate"]
    assert "--resume" in n0[0].command
    assert str(last.relative_to(tmp_path)).replace("h1/", "<OUTPUT_DIR>/h1/") in n0[0].command
    limit = n0[0].command.index("--max-train-batches")
    assert n0[0].command[limit + 1] == "300"


def test_unpublished_target_last_finalizes_summary_without_retraining(
    tmp_path: Path,
) -> None:
    manifest = _h1_in_progress_manifest(tmp_path)
    _write_unpublished_last(tmp_path, "H1", "N0", 600)

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)
    n0 = [item for item in plan.all_commands if item.candidate_id == "N0"]

    assert [item.kind for item in n0] == ["finalize_training", "evaluate"]
    assert "--finalize-training-only" in n0[0].command
    assert "--resume" in n0[0].command
    assert "--max-train-batches" not in n0[0].command


def test_unpublished_target_last_with_summary_runs_evaluation_only(
    tmp_path: Path,
) -> None:
    manifest = _h1_in_progress_manifest(tmp_path)
    _write_unpublished_last(tmp_path, "H1", "N0", 600)
    atomic_write_json(tmp_path / "h1/N0/summary.json", {"global_step": 600})

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)
    n0 = [item for item in plan.all_commands if item.candidate_id == "N0"]

    assert [item.kind for item in n0] == ["evaluate"]


def test_unpublished_incompatible_last_is_fatal(tmp_path: Path) -> None:
    manifest = _h1_in_progress_manifest(tmp_path)
    _write_unpublished_last(
        tmp_path, "H1", "N0", 300, config_hash="0" * 64
    )

    with pytest.raises(ValueError, match="config.*hash"):
        build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)


def test_gate_o_target_last_uses_finalize_only_without_optimizer_budget(
    tmp_path: Path,
) -> None:
    manifest = {
        "artifacts": {},
        "decisions": {},
        "stage": {"gate": "O", "status": "in_progress"},
    }
    _write_unpublished_last(tmp_path, "O", "N0", 400)

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)
    n0 = [item for item in plan.all_commands if item.candidate_id == "N0"]

    assert [item.kind for item in n0] == ["finalize_overfit"]
    assert "--finalize-overfit-only" in n0[0].command
    assert "--max-train-batches" not in n0[0].command


@pytest.mark.parametrize(
    ("finalize_flag", "global_step", "overfit"),
    [
        ("--finalize-training-only", 600, False),
        ("--finalize-overfit-only", 400, True),
    ],
)
def test_finalize_cli_never_constructs_or_restores_optimizer_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finalize_flag: str,
    global_step: int,
    overfit: bool,
) -> None:
    config_path = CONFIG_DIR / screen_cli.CONFIG_FILENAMES[0]
    config = train_cli._load_configuration(config_path)
    normalization = train_cli._load_normalization_binding(config)
    split_path = PROJECT_ROOT / config["data"]["split_manifest"]
    run_dir = tmp_path / ("overfit" if overfit else "halving")
    checkpoint = run_dir / "checkpoints/last.pt"
    checkpoint.parent.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    torch.save(
        {
            "schema_version": 4,
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "epoch_batch_cursor": 0,
            "epoch_scene_order": [0],
            "global_step": global_step,
            "phase_index": 0,
            "phase_update": global_step,
            "config_sha256": train_cli._config_sha256(config),
            "split_manifest_sha256": _sha(split_path),
            "runtime_seed": config["seed"],
            "normalization_stats_sha256": normalization.stats_sha256,
            "normalization_contract": normalization.contract_id,
        },
        checkpoint,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("finalize-only must not touch optimizer or scheduler state")

    monkeypatch.setattr(train_cli, "_new_optimizer", forbidden)
    monkeypatch.setattr(train_cli, "load_query_checkpoint", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(train_cli.np.random, "seed", forbidden)
    monkeypatch.setattr(train_cli.random, "seed", forbidden)
    monkeypatch.setattr(train_cli, "AdaptiveSpatialSampler", forbidden)
    monkeypatch.setattr(torch, "Generator", forbidden)
    monkeypatch.setattr(train_cli, "AISMQFNO", lambda **kwargs: torch.nn.Linear(2, 1))
    monkeypatch.setattr(train_cli, "DenseCPUQueryStore", lambda path: object())
    monkeypatch.setattr(
        train_cli,
        "_overfit_physical_metrics",
        lambda *args, **kwargs: {
            "relative_l2": 0.1,
            "relative_l2_q4": 0.2,
            "prediction_target_norm_ratio": 1.0,
            "prediction_target_pearson": 0.9,
        },
    )
    argv = [
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--device",
        "cpu",
        "--resume",
        str(checkpoint),
        finalize_flag,
    ]
    if overfit:
        argv.extend(
            [
                "--overfit-sample-id",
                "2",
                "--overfit-site-manifest",
                str(
                    CONFIG_DIR
                    / "registration/gate_o_sample_0002_sites_2048.json"
                ),
            ]
        )

    torch_rng_before = torch.random.get_rng_state().clone()
    python_rng_before = random.getstate()
    numpy_rng_before = train_cli.np.random.get_state()

    assert train_cli.main(argv) == 0
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)
    assert random.getstate() == python_rng_before
    numpy_rng_after = train_cli.np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    assert train_cli.np.array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert json.loads((run_dir / "summary.json").read_text())["global_step"] == global_step
    assert json.loads((run_dir / "training_state.json").read_text())["global_step"] == global_step
    assert (run_dir / "screen_metrics.json").is_file() is overfit


@pytest.mark.parametrize(
    ("current_gate", "same_run", "stored_gate"),
    [("H1", True, "H1"), ("H2", False, "H1"), ("H3", False, "H2")],
)
def test_schema5_resume_preflight_binds_gate_parent_data_and_environment(
    current_gate: str, same_run: bool, stored_gate: str
) -> None:
    payload = {
        "screen_candidate_id": "N0", "screen_gate": stored_gate,
        "parent_checkpoint_sha256": "a" * 64,
        "dataset_binding_sha256": "d" * 64,
        "execution_binding_sha256": "e" * 64,
    }
    train_cli._validate_screen_resume_identity(
        payload, candidate_id="N0", current_gate=current_gate, same_run=same_run,
        dataset_binding_sha256="d" * 64, execution_binding_sha256="e" * 64,
        lineage_parent_sha256="a" * 64,
    )
    for key in ("screen_gate", "parent_checkpoint_sha256", "dataset_binding_sha256", "execution_binding_sha256"):
        forged = dict(payload)
        forged[key] = "f" * 64
        if key == "parent_checkpoint_sha256" and not same_run:
            continue
        with pytest.raises(ValueError, match="identity|binding"):
            train_cli._validate_screen_resume_identity(
                forged, candidate_id="N0", current_gate=current_gate,
                same_run=same_run, dataset_binding_sha256="d" * 64,
                execution_binding_sha256="e" * 64,
                lineage_parent_sha256="a" * 64,
            )


def test_h3_records_publish_before_baseline_then_finalize_without_gpu_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _completed_h3_manifest(tmp_path)

    plan = build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=manifest)

    assert plan.review_boundary == "H3"
    assert not plan.all_commands
    baseline_values = {
        "relative_l2": 0.4,
        "relative_l2_q4": 0.5,
        **{name: 0.1 for name in screen_cli.LOWER_GUARDS},
        **{name: 0.9 for name in screen_cli.HIGHER_GUARDS},
    }
    checkpoint = tmp_path / "baseline_b1/checkpoints/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"registered-b1")
    monkeypatch.setattr(screen_cli, "B1_CHECKPOINT_SHA256", _sha(checkpoint))
    b1_config = yaml.safe_load(screen_cli.B1_CONFIG_PATH.read_text())
    b1_split = PROJECT_ROOT / b1_config["data"]["split_manifest"]
    b1_stats = PROJECT_ROOT / b1_config["normalization"]["stats_path"]
    baseline = tmp_path / "baseline_b1/summary.json"
    samples = tmp_path / "baseline_b1/samples.csv"
    samples.write_text("sample_id\n" + "".join(f"{item}\n" for item in json.loads(b1_split.read_text())["val"]))
    atomic_write_json(
        baseline,
        {
            "schema_version": 1,
            "purpose": "ais_b1_validation_baseline",
            "baseline_id": "B1",
            "checkpoint_sha256": _sha(checkpoint),
            "config_sha256": screen_cli.B1_CONFIG_SHA256,
            "split": "val",
            "sample_count": len(json.loads(b1_split.read_text())["val"]),
            "representative_manifest": None,
            "split_sha256": _sha(b1_split),
            "normalization_stats_sha256": _sha(b1_stats),
            "source_commit": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "dirty_entries_sha256": "c" * 64,
            "samples_path": str(samples),
            "samples_sha256": _sha(samples),
            "category_metrics": {
                category: dict(baseline_values)
                for category in ("uniform", "layered", "marmousi")
            }
        },
    )

    finalized = screen_cli.finalize_h3_with_baseline(manifest, baseline)

    assert finalized["baseline_metrics_sha256"] == _sha(baseline)
    assert finalized["baseline_metrics_path"] == str(baseline)
    assert finalized["stage"] == {"gate": "H3", "status": "complete"}
    assert finalized["decisions"]["H3"]["status"] in {"promote", "stop"}
    baseline.write_bytes(baseline.read_bytes() + b" ")
    with pytest.raises(ValueError, match="baseline.*hash"):
        build_screen_plan(CONFIG_DIR, tmp_path, resume_manifest=finalized)


def test_b1_producer_writes_canonical_consumer_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "input.pt"
    checkpoint.write_bytes(b"pinned")
    monkeypatch.setattr(baseline_cli, "B1_CHECKPOINT_SHA256", _sha(checkpoint))
    monkeypatch.setattr(
        baseline_cli, "compute_source_binding",
        lambda: screen_cli.SourceBinding("a" * 40, "b" * 64, "c" * 64),
    )
    config = yaml.safe_load(baseline_cli.B1_CONFIG_PATH.read_text())
    dataset = _tiny_binding_dataset(tmp_path / "tiny.h5")
    split_path = tmp_path / "split.json"
    atomic_write_json(split_path, {"train": [], "val": [0, 1], "test": []})
    stats_path = tmp_path / "normalization.json"
    stats_path.write_text("{}")
    config["data"]["path"] = str(dataset)
    config["data"]["split_manifest"] = str(split_path)
    config["normalization"]["stats_path"] = str(stats_path)
    config_path = tmp_path / "b1.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(baseline_cli, "B1_CONFIG_PATH", config_path)
    monkeypatch.setattr(baseline_cli, "B1_CONFIG_SHA256", canonical_sha256(config))
    split = json.loads(split_path.read_text())
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_bytes(
        canonical_manifest_bytes(compute_dataset_manifest(dataset))
    )
    provenance = {
        "config_sha256": canonical_sha256(config),
        "checkpoint_sha256": _sha(checkpoint),
        "split_manifest_sha256": _sha(split_path),
        "normalization_sha256": _sha(stats_path),
        "receiver_geometry_sha256": "f" * 64,
    }
    samples = tmp_path / "samples.csv"
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=baseline_cli.SAMPLE_COLUMNS)
        writer.writeheader()
        for item, category in zip(split["val"], ("uniform", "layered"), strict=True):
            writer.writerow(
                {
                    **{column: "0" for column in baseline_cli.SAMPLE_COLUMNS},
                    "sample_id": item, "category": category, "split": "val",
                    **provenance,
                }
            )
    evaluation = tmp_path / "evaluation.json"
    atomic_write_json(
        evaluation,
        {
            "split": "val", "sample_count": len(split["val"]),
            "provenance": provenance,
            "mean": {
                metric: 0.0 for metric in baseline_cli.REQUIRED_NUMERIC_COLUMNS
            },
            "categories": {
                category: {
                    "sample_count": 1,
                    "mean": {
                        metric: 0.0
                        for metric in baseline_cli.REQUIRED_NUMERIC_COLUMNS
                    },
                }
                for category in ("uniform", "layered")
            },
            "category_metrics": {
                name: {"sample_count": count, **{
                    metric: 0.0 for metric in baseline_cli.REQUIRED_NUMERIC_COLUMNS
                }}
                for name, count in (("global", 2), ("uniform", 1), ("layered", 1))
            },
        },
    )
    wrong_manifest = tmp_path / "wrong_dataset_manifest.json"
    wrong_payload = compute_dataset_manifest(dataset)
    wrong_payload["dataset_path"] = str(tmp_path / "wrong.h5")
    wrong_manifest.write_bytes(canonical_manifest_bytes(wrong_payload))
    with pytest.raises(ValueError, match="dataset_path"):
        baseline_cli.main(
            ["--checkpoint", str(checkpoint), "--evaluation-summary", str(evaluation),
             "--samples", str(samples), "--output-dir", str(tmp_path / "wrong"),
             "--dataset-content-manifest", str(wrong_manifest)]
        )
    assert baseline_cli.main(
        ["--checkpoint", str(checkpoint), "--evaluation-summary", str(evaluation),
         "--samples", str(samples), "--output-dir", str(tmp_path / "formal"),
         "--dataset-content-manifest", str(dataset_manifest)]
    ) == 0
    output = tmp_path / "formal/baseline_b1"
    assert (output / "summary.json").is_file()
    assert (output / "samples.csv").is_file()
    assert (output / "checkpoints/best.pt").read_bytes() == b"pinned"
    b1_summary = json.loads((output / "summary.json").read_bytes())
    assert b1_summary["dataset_content_root"] == compute_dataset_manifest(dataset)[
        "dataset_content_root"
    ]
    assert b1_summary["verified_sample_count"] == 2
    assert b1_summary["verified_sample_ids_sha256"] == verified_ids_sha256([0, 1])
    rows = list(csv.DictReader(samples.open()))
    rows[1]["category"] = "uniform"
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=baseline_cli.SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="category.*registered"):
        baseline_cli.main(
            ["--checkpoint", str(checkpoint), "--evaluation-summary", str(evaluation),
             "--samples", str(samples), "--output-dir", str(tmp_path / "forged"),
             "--dataset-content-manifest", str(dataset_manifest)]
        )


def test_b1_producer_rejects_wrong_evaluator_provenance_ids_and_categories(
    tmp_path: Path,
) -> None:
    expected = {
        "checkpoint_sha256": "a" * 64, "config_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64, "normalization_sha256": "d" * 64,
    }
    summary_path, samples_path = tmp_path / "summary.json", tmp_path / "samples.csv"
    zero_mean = {
        metric: 0.0 for metric in baseline_cli.REQUIRED_NUMERIC_COLUMNS
    }
    zero_aggregate = {
        "global": {"sample_count": 2, **zero_mean},
        "uniform": {"sample_count": 2, **zero_mean},
    }

    def write_summary(**updates: object) -> None:
        payload = {
            "split": "val", "sample_count": 2, "provenance": dict(expected),
            "mean": dict(zero_mean),
            "categories": {
                "uniform": {"sample_count": 2, "mean": dict(zero_mean)}
            },
            "category_metrics": copy.deepcopy(zero_aggregate),
        }
        payload.update(updates)
        atomic_write_json(summary_path, payload)

    def write_rows(rows: list[dict[str, object]]) -> None:
        with samples_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=baseline_cli.SAMPLE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {**{column: "0" for column in baseline_cli.SAMPLE_COLUMNS},
                     "category": "uniform", "split": "val",
                     "receiver_geometry_sha256": "e" * 64,
                     **expected, **row}
                )

    def validate() -> None:
        baseline_cli._validated_evaluator_artifacts(
            summary_path, samples_path, checkpoint_sha256="a" * 64,
            config_sha256="b" * 64, split_sha256="c" * 64,
            normalization_sha256="d" * 64, expected_ids={1, 2},
        )

    write_summary()
    write_rows([{"sample_id": 1}, {"sample_id": 2}])
    validate()

    tampered_metrics = copy.deepcopy(zero_aggregate)
    tampered_metrics["global"]["relative_l2"] = 0.5
    write_summary(category_metrics=tampered_metrics)
    with pytest.raises(ValueError, match="category metrics"):
        validate()

    write_summary()
    write_rows([{"sample_id": 1}, {"sample_id": 2, "relative_l2": 1.0}])
    with pytest.raises(ValueError, match="summary metrics"):
        validate()

    swapped_aggregate = {
        "global": {"sample_count": 2, **zero_mean},
        "uniform": {"sample_count": 1, **zero_mean},
        "layered": {"sample_count": 1, **zero_mean},
    }
    write_summary(
        categories={
            category: {"sample_count": 1, "mean": dict(zero_mean)}
            for category in ("uniform", "layered")
        },
        category_metrics=swapped_aggregate,
    )
    write_rows([
        {"sample_id": 1, "category": "layered"},
        {"sample_id": 2, "category": "uniform"},
    ])
    with pytest.raises(ValueError, match="category.*registered split"):
        baseline_cli._validated_evaluator_artifacts(
            summary_path, samples_path, checkpoint_sha256="a" * 64,
            config_sha256="b" * 64, split_sha256="c" * 64,
            normalization_sha256="d" * 64, expected_ids={1, 2},
            expected_categories={1: "uniform", 2: "layered"},
        )
    for update in (
        {"split": "train"}, {"sample_count": 1},
        {"provenance": {**expected, "checkpoint_sha256": "f" * 64}},
        {"provenance": {**expected, "config_sha256": "f" * 64}},
        {"provenance": {**expected, "split_manifest_sha256": "f" * 64}},
        {"provenance": {**expected, "normalization_sha256": "f" * 64}},
    ):
        write_summary(**update)
        with pytest.raises(ValueError, match="mismatch"):
            validate()
    write_summary()
    for rows in (
        [{"sample_id": 1}, {"sample_id": 1}],
        [{"sample_id": 1}],
        [{"sample_id": 1}, {"sample_id": 3}],
        [{"sample_id": 1}, {"sample_id": 2, "category": "forged"}],
        [{"sample_id": 1}, {"sample_id": 2, "category": "layered"}],
        [{"sample_id": 1}, {"sample_id": 2, "config_sha256": "f" * 64}],
    ):
        write_rows(rows)
        with pytest.raises(ValueError, match="duplicate|IDs|categor|provenance|metrics"):
            validate()


def test_baseline_metrics_cli_is_resume_only_and_documented(tmp_path: Path) -> None:
    parser = screen_cli.build_parser()
    help_text = parser.format_help()

    assert "--baseline-metrics" in help_text
    assert "Task 12" in help_text
    with pytest.raises(ValueError, match="only.*--resume"):
        screen_cli.main(
            [
                "--config-dir",
                str(CONFIG_DIR),
                "--output-dir",
                str(tmp_path / "run"),
                "--baseline-metrics",
                str(tmp_path / "baseline.json"),
            ]
        )


def test_baseline_shortcut_rejects_noncanonical_h3_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _completed_h3_manifest(tmp_path)
    manifest["schema_version"] = screen_cli.MANIFEST_SCHEMA_VERSION
    row = manifest["artifacts"]["H3"]["N0"]
    external = tmp_path / "forged-last.pt"
    external.write_bytes(Path(row["last_checkpoint"]).read_bytes())
    row["last_checkpoint"] = str(external)
    row["last_checkpoint_sha256"] = _sha(external)
    atomic_write_json(tmp_path / "screen_manifest.json", manifest)
    source = screen_cli.SourceBinding("a" * 40, "b" * 64, "c" * 64)
    monkeypatch.setattr(screen_cli, "compute_source_binding", lambda: source)

    with pytest.raises(ValueError, match="canonical"):
        screen_cli.main(
            [
                "--config-dir", str(CONFIG_DIR), "--output-dir", str(tmp_path),
                "--resume", "--baseline-metrics", str(tmp_path / "baseline_b1/summary.json"),
            ]
        )
