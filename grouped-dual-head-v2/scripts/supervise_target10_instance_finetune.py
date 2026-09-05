#!/usr/bin/env python3
"""Wait for accepted pretraining, then run accuracy-gated ASAM and CPADC."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import h5py
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILIES = ("uniform", "layered", "marmousi")
MAXIMUM_RELATIVE_L2 = 0.05
MINIMUM_SPEEDUP = 10.0


def _atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf8") as handle:
            handle.write(str(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_vds_shard_inventory(dataset: Path) -> list[dict[str, object]]:
    """Record the sidecars just verified by strict VDS validation."""

    with h5py.File(dataset, "r", swmr=True) as handle:
        try:
            values = json.loads(str(handle.attrs["vds_source_shards"]))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("external evaluation VDS shard list is invalid") from error
    if not isinstance(values, list) or not values:
        raise ValueError("external evaluation VDS has no physical shards")
    inventory: list[dict[str, object]] = []
    for value in values:
        shard = Path(str(value)).resolve()
        sidecar = Path(str(shard) + ".sha256")
        if not shard.is_file() or not sidecar.is_file():
            raise FileNotFoundError("external evaluation shard/sidecar is missing")
        digest = sidecar.read_text(encoding="utf-8").strip()
        if len(digest) != 64 or _sha256_file(shard) != digest:
            raise ValueError("external evaluation shard sidecar digest is invalid")
        inventory.append(
            {
                "path": str(shard),
                "byte_count": int(shard.stat().st_size),
                "sha256": digest,
                "sidecar": str(sidecar),
                "sidecar_sha256": _sha256_file(sidecar),
            }
        )
    return inventory


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _sealed_full_support_target_met(
    report: Mapping[str, object], *, expected_split: str | None = None
) -> bool:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    family = metrics.get("family_relative_l2")
    if not isinstance(family, Mapping) or set(family) != set(FAMILIES):
        return False
    try:
        aggregate = float(metrics["aggregate_relative_l2"])
        family_values = tuple(float(family[name]) for name in FAMILIES)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        report.get("status") == "complete"
        and (
            expected_split is None
            or str(report.get("evaluation_split", "validation"))
            == str(expected_split)
        )
        and report.get("stored_times_only") is True
        and int(report.get("interpolated_targets", -1)) == 0
        and int(metrics.get("record_count", -1)) == 480
        and int(metrics.get("unique_time_index_count", -1)) == 401
        and aggregate <= MAXIMUM_RELATIVE_L2
        and all(value <= MAXIMUM_RELATIVE_L2 for value in family_values)
    )


def _cpadc_target_met(
    terminal: Mapping[str, object], *, expected_split: str | None = None
) -> bool:
    promotion = terminal.get("promotion_gate")
    if not isinstance(promotion, Mapping):
        return False
    checks = promotion.get("checks")
    families = promotion.get("families")
    if not isinstance(checks, Mapping) or not isinstance(families, Mapping):
        return False
    try:
        aggregate = float(promotion["aggregate_adapted_future_relative_l2"])
        mean_speedup = float(
            promotion["mean_end_to_end_speedup_vs_traditional"]
        )
        p95_speedup = float(
            promotion["p95_end_to_end_speedup_vs_traditional"]
        )
        family_values = tuple(
            float(families[name]["aggregate_adapted_future_relative_l2"])
            for name in FAMILIES
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        terminal.get("status") == "complete"
        and int(promotion.get("record_count", -1)) == 480
        and (
            expected_split is None
            or str(terminal.get("evaluation_split", "validation"))
            == str(expected_split)
        )
        and terminal.get("same_protocol_validation_passed") is True
        and promotion.get("passed") is True
        and checks.get("full_validation_protocol") is True
        and checks.get("future_truth_sealed") is True
        and checks.get("cpu_coefficient_finetune") is True
        and checks.get("absolute_mean_accuracy") is True
        and checks.get("absolute_family_accuracy") is True
        and checks.get("runtime_measurement_synchronized") is True
        and checks.get("end_to_end_speedup_mean") is True
        and checks.get("end_to_end_speedup_p95") is True
        and mean_speedup >= MINIMUM_SPEEDUP
        and p95_speedup >= MINIMUM_SPEEDUP
        and aggregate <= MAXIMUM_RELATIVE_L2
        and all(value <= MAXIMUM_RELATIVE_L2 for value in family_values)
    )


def _validation_authorizes_test_id(terminal: Mapping[str, object]) -> bool:
    """Open test_id only after the complete frozen CPADC validation gate passes."""

    return _cpadc_target_met(terminal, expected_split="validation")


def _holdout_generation_command(
    *,
    config: Path,
    output: Path,
    split: str,
    confirm_token: str,
) -> list[str]:
    normalized = str(split)
    if normalized not in {"validation", "test_id"}:
        raise ValueError("frozen holdout generation split is invalid")
    return [
        sys.executable,
        "-u",
        "scripts/launch_gpu_dataset_workers.py",
        "--config",
        str(config),
        "--output",
        str(output),
        "--confirm-production",
        str(confirm_token),
        "--splits",
        normalized,
        "--devices",
        "0,1,2,3",
        "--batch-size",
        "128",
        "--resume",
        "--frozen-manifest",
    ]


def _external_evaluation_contract(
    *,
    role: str,
    training_manifest_digest: str,
    evaluation_manifest_digest: str,
    sample_hash_audit: Path,
    pretruth_overlap_audit: Path,
    internal_split_overlap_audit: Path,
    evaluation_dataset: Path,
    travel_time_h5: Path,
) -> dict[str, object]:
    if role not in {"frozen_validation", "independent_test_id"}:
        raise ValueError("external evaluation role is invalid")
    paths = {
        "postgeneration_sample_sha256_audit": sample_hash_audit.resolve(),
        "pretruth_historical_overlap_audit": pretruth_overlap_audit.resolve(),
        "internal_split_overlap_audit": internal_split_overlap_audit.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"external evaluation {name} is missing: {path}")
    dataset = evaluation_dataset.resolve()
    travel = travel_time_h5.resolve()
    if not dataset.is_file() or not travel.is_file():
        raise FileNotFoundError("external evaluation dataset/travel cache is missing")
    with h5py.File(travel, "r", swmr=True) as handle:
        travel_source = Path(str(handle.attrs.get("source_h5", ""))).resolve()
        travel_content_sha256 = str(handle.attrs.get("content_sha256", ""))
        travel_schema = str(handle.attrs.get("schema", ""))
    if travel_source != dataset:
        raise ValueError("external evaluation travel cache source is not the dataset")
    if len(travel_content_sha256) != 64:
        raise ValueError("external evaluation travel cache content digest is invalid")
    return {
        "role": role,
        "training_manifest_digest": str(training_manifest_digest),
        "evaluation_manifest_digest": str(evaluation_manifest_digest),
        "evaluation_only": True,
        "training_forbidden": True,
        **{name: str(path) for name, path in paths.items()},
        **{
            f"{name}_sha256": _sha256_file(path)
            for name, path in paths.items()
        },
        "evaluation_dataset": str(dataset),
        "evaluation_vds_shards": _validated_vds_shard_inventory(dataset),
        "travel_time_h5": str(travel),
        "travel_time_h5_sha256": _sha256_file(travel),
        "travel_time_content_sha256": travel_content_sha256,
        "travel_time_schema": travel_schema,
    }


def _traditional_runtime_reference(report: Mapping[str, object]) -> float:
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("traditional runtime report lacks a protocol")
    try:
        reference = float(report["conservative_reference_runtime_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("traditional runtime reference is invalid") from error
    if not reference > 0.0:
        raise ValueError("traditional runtime reference must be positive")
    if not (
        report.get("status") == "complete"
        and report.get("schema") == "lwc84_traditional_runtime_reference_v1"
        and protocol.get("solver_grid") == [401, 401]
        and protocol.get("saved_grid") == [201, 201]
        and int(protocol.get("saved_frames", -1)) == 401
        and float(protocol.get("propagation_time_s", -1.0)) == 1.0
        and protocol.get("single_instance") is True
        and protocol.get("same_gpu_as_deployment") is True
        and protocol.get("cuda_synchronized_timing") is True
        and protocol.get("includes_output_materialization") is True
        and protocol.get("excludes_disk_io") is True
    ):
        raise ValueError("traditional runtime protocol is not deployment comparable")
    return reference


def _run_logged(
    command: Sequence[str],
    *,
    log: Path,
    environment: Mapping[str, str],
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf8") as handle:
        handle.write(
            json.dumps(
                {"event": "launch", "command": list(command), "time": time.time()},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        result = subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(
            json.dumps(
                {"event": "exit", "return_code": result.returncode, "time": time.time()},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        return int(result.returncode)


def _write_bound_config(template: Path, output: Path, updates: Mapping[str, object]) -> None:
    config = yaml.safe_load(template.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"config template is not a mapping: {template}")
    config.update(dict(updates))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    os.replace(temporary, output)


def _ensure_frozen_holdout_split(
    *,
    holdout_config: Path,
    split: str,
    environment: Mapping[str, str],
    pipeline_output: Path,
    status_path: Path,
) -> dict[str, Path]:
    """Generate one preregistered split, content-audit it, then derive travel inputs."""
    payload = yaml.safe_load(holdout_config.read_text())
    if not isinstance(payload, dict):
        raise ValueError("frozen holdout config must contain a mapping")
    data_root = Path(str(payload["paths"]["output_root"])).resolve()
    filename = Path(str(payload["storage"]["vds_filename"]))
    dataset = data_root / f"{filename.stem}_{split}{filename.suffix}"
    sample_audit = data_root / f"postgeneration_sample_sha256_audit_{split}.json"
    pretruth_audit = data_root / "historical_overlap_audit.json"
    internal_split_audit = data_root / "internal_split_overlap_audit.json"
    if not dataset.is_file() or not sample_audit.is_file():
        _atomic_json(
            {
                "status": f"generating_frozen_{split}",
                "future_truth_opened_for_evaluation": False,
            },
            status_path,
        )
        command = _holdout_generation_command(
            config=holdout_config,
            output=data_root,
            split=split,
            confirm_token=str(payload["production"]["confirm_token"]),
        )
        code = _run_logged(
            command,
            log=pipeline_output / f"frozen_{split}_generation.log",
            environment=environment,
        )
        if code != 0:
            raise RuntimeError(
                f"frozen {split} generation failed with return code {code}"
            )
    for path in (dataset, sample_audit, pretruth_audit, internal_split_audit):
        if not path.is_file():
            raise FileNotFoundError(f"frozen {split} artifact is missing: {path}")
    from fno_acoustic.data_generation.hdf5_lwc84 import validate_lwc84_dataset_vds

    dataset_summary = validate_lwc84_dataset_vds(
        dataset, strict=True, expected_n=480
    )
    if dataset_summary.get("included_splits") != [split]:
        raise ValueError(f"frozen {split} VDS includes another split")
    sample_audit_payload = _read_json(sample_audit)
    pretruth_payload = _read_json(pretruth_audit)
    internal_payload = _read_json(internal_split_audit)
    if not (
        sample_audit_payload is not None
        and sample_audit_payload.get("status") == "passed"
        and sample_audit_payload.get("passed") is True
        and int(sample_audit_payload.get("intersection_count", -1)) == 0
        and pretruth_payload is not None
        and pretruth_payload.get("status") == "passed"
        and pretruth_payload.get("passed") is True
        and internal_payload is not None
        and internal_payload.get("status") == "passed"
        and internal_payload.get("passed") is True
    ):
        raise ValueError(f"frozen {split} identity audits did not all pass")

    travel_root = data_root / "derived"
    travel_root.mkdir(parents=True, exist_ok=True)
    eikonal = travel_root / f"eikonal_{split}.h5"
    hybrid = travel_root / f"hybrid_travel_{split}.h5"
    if not eikonal.is_file():
        _atomic_json({"status": f"building_{split}_eikonal_cache"}, status_path)
        code = _run_logged(
            [
                sys.executable,
                "-u",
                "scripts/build_eikonal_travel_cache.py",
                "--source-h5",
                str(dataset),
                "--output",
                str(eikonal),
                "--splits",
                split,
                "--families",
                "uniform,layered,marmousi",
                "--workers",
                "16",
            ],
            log=pipeline_output / f"frozen_{split}_eikonal.log",
            environment=environment,
        )
        if code != 0:
            raise RuntimeError(f"{split} Eikonal cache failed with return code {code}")
    if not hybrid.is_file():
        _atomic_json({"status": f"building_{split}_hybrid_travel_cache"}, status_path)
        code = _run_logged(
            [
                sys.executable,
                "-u",
                "scripts/build_hybrid_travel_cache.py",
                "--eikonal-cache",
                str(eikonal),
                "--output",
                str(hybrid),
                "--workers",
                "16",
            ],
            log=pipeline_output / f"frozen_{split}_hybrid_travel.log",
            environment=environment,
        )
        if code != 0:
            raise RuntimeError(
                f"{split} hybrid travel cache failed with return code {code}"
            )
    return {
        "dataset": dataset,
        "sample_hash_audit": sample_audit,
        "pretruth_audit": pretruth_audit,
        "internal_split_audit": internal_split_audit,
        "travel_cache": hybrid,
    }


def _run_parallel_cpadc_evaluation(
    *,
    config: Path,
    basis_checkpoint: Path,
    root: Path,
    split: str,
    environment: Mapping[str, str],
    pipeline_output: Path,
) -> dict[str, object]:
    """Run four mutually exclusive evaluation shards and one exact merge."""
    normalized = str(split)
    if normalized not in {"validation", "test_id"}:
        raise ValueError("CPADC frozen evaluation split is invalid")
    selection_flag = "--all-validation" if normalized == "validation" else "--all-test-id"
    shard_dirs: list[Path] = []
    processes: list[tuple[subprocess.Popen, object, Path]] = []
    for index in range(4):
        shard_dir = root / f"{normalized}_shards" / f"shard_{index}"
        shard_dirs.append(shard_dir)
        existing = _read_json(shard_dir / "terminal.json")
        if existing is not None and existing.get("status") == "complete":
            continue
        log_path = pipeline_output / f"frozen_{normalized}_cpadc_shard_{index}.log"
        handle = log_path.open("a", encoding="utf8")
        command = [
            sys.executable,
            "-u",
            "scripts/run_causal_defect_adaptation.py",
            "--config",
            str(config),
            "--basis-checkpoint",
            str(basis_checkpoint),
            "--output-dir",
            str(shard_dir),
            "--device",
            "cuda",
            "--adaptation-device",
            "cpu",
            selection_flag,
            "--shard-index",
            str(index),
            "--shard-count",
            "4",
            "--no-fields",
        ]
        handle.write(json.dumps({"event": "launch", "command": command}) + "\n")
        handle.flush()
        shard_environment = dict(environment)
        shard_environment["CUDA_VISIBLE_DEVICES"] = str(index)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=shard_environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, log_path))
    failures: list[dict[str, object]] = []
    for process, handle, log_path in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append({"log": str(log_path), "return_code": code})
    if failures:
        raise RuntimeError(f"frozen {normalized} CPADC shards failed: {failures}")

    merged = root / f"{normalized}_evaluation"
    terminal = _read_json(merged / "terminal.json")
    if terminal is None or terminal.get("status") != "complete":
        command = [
            sys.executable,
            "-u",
            "scripts/merge_causal_defect_evaluations.py",
            "--config",
            str(config),
            "--output-dir",
            str(merged),
            "--evaluation-split",
            normalized,
        ]
        for shard_dir in shard_dirs:
            command.extend(("--shard-dir", str(shard_dir)))
        code = _run_logged(
            command,
            log=pipeline_output / f"frozen_{normalized}_cpadc_merge.log",
            environment=environment,
        )
        terminal = _read_json(merged / "terminal.json")
        if code != 0 or terminal is None:
            raise RuntimeError(
                f"frozen {normalized} CPADC merge failed with return code {code}"
            )
    return terminal


def _run_independent_confirmation_pipeline(
    *,
    training_cpadc_config: Path,
    basis_checkpoint: Path,
    cpadc_root: Path,
    pipeline_output: Path,
    status_path: Path,
    terminal_path: Path,
    environment: Mapping[str, str],
    context: Mapping[str, object],
) -> int:
    """Freeze the model, then open fresh validation and test in strict order."""
    from grouped_ufno_mionet_v3.data.index import build_manifest

    holdout_config = (
        ROOT
        / "configs/datasets/acoustic_lwc84_target5_frozen_validation_r1.yaml"
    )
    training_payload = yaml.safe_load(training_cpadc_config.read_text())
    training_manifest = build_manifest(training_payload["source_h5"])

    validation_artifacts = _ensure_frozen_holdout_split(
        holdout_config=holdout_config,
        split="validation",
        environment=environment,
        pipeline_output=pipeline_output,
        status_path=status_path,
    )
    validation_manifest = build_manifest(validation_artifacts["dataset"])
    validation_config = pipeline_output / "bound_cpadc_frozen_validation.yaml"
    _write_bound_config(
        training_cpadc_config,
        validation_config,
        {
            "source_h5": str(validation_artifacts["dataset"]),
            "travel_time_h5": str(validation_artifacts["travel_cache"]),
            "external_evaluation_contract": _external_evaluation_contract(
                role="frozen_validation",
                training_manifest_digest=training_manifest.digest,
                evaluation_manifest_digest=validation_manifest.digest,
                sample_hash_audit=validation_artifacts["sample_hash_audit"],
                pretruth_overlap_audit=validation_artifacts["pretruth_audit"],
                internal_split_overlap_audit=validation_artifacts[
                    "internal_split_audit"
                ],
                evaluation_dataset=validation_artifacts["dataset"],
                travel_time_h5=validation_artifacts["travel_cache"],
            ),
        },
    )
    _atomic_json({"status": "running_new_frozen_validation"}, status_path)
    validation_terminal = _run_parallel_cpadc_evaluation(
        config=validation_config,
        basis_checkpoint=basis_checkpoint,
        root=cpadc_root / "independent_confirmation",
        split="validation",
        environment=environment,
        pipeline_output=pipeline_output,
    )
    validation_target = _validation_authorizes_test_id(validation_terminal)
    if not validation_target:
        terminal = {
            **dict(context),
            "status": "target_not_met",
            "instance_validation_target_met": False,
            "instance_test_id_target_met": False,
            "test_id_opened": False,
            "test_id_authorization": "denied_by_new_frozen_validation_gate",
            "cpadc_terminal": validation_terminal,
            "cpadc_test_id_terminal": None,
            "claim": (
                "new frozen validation did not pass; independent test_id remains ungenerated"
            ),
        }
        _atomic_json(terminal, terminal_path)
        _atomic_json(terminal, status_path)
        return 5

    _atomic_json(
        {
            "status": "new_validation_passed_generating_independent_test_id",
            "test_id_opened": False,
        },
        status_path,
    )
    test_artifacts = _ensure_frozen_holdout_split(
        holdout_config=holdout_config,
        split="test_id",
        environment=environment,
        pipeline_output=pipeline_output,
        status_path=status_path,
    )
    test_manifest = build_manifest(test_artifacts["dataset"])
    test_config = pipeline_output / "bound_cpadc_independent_test_id.yaml"
    _write_bound_config(
        training_cpadc_config,
        test_config,
        {
            "source_h5": str(test_artifacts["dataset"]),
            "travel_time_h5": str(test_artifacts["travel_cache"]),
            "external_evaluation_contract": _external_evaluation_contract(
                role="independent_test_id",
                training_manifest_digest=training_manifest.digest,
                evaluation_manifest_digest=test_manifest.digest,
                sample_hash_audit=test_artifacts["sample_hash_audit"],
                pretruth_overlap_audit=test_artifacts["pretruth_audit"],
                internal_split_overlap_audit=test_artifacts[
                    "internal_split_audit"
                ],
                evaluation_dataset=test_artifacts["dataset"],
                travel_time_h5=test_artifacts["travel_cache"],
            ),
        },
    )
    _atomic_json(
        {"status": "running_new_independent_test_id", "test_id_opened": True},
        status_path,
    )
    test_terminal = _run_parallel_cpadc_evaluation(
        config=test_config,
        basis_checkpoint=basis_checkpoint,
        root=cpadc_root / "independent_confirmation",
        split="test_id",
        environment=environment,
        pipeline_output=pipeline_output,
    )
    test_target = _cpadc_target_met(test_terminal, expected_split="test_id")
    instance_target = bool(validation_target and test_target)
    validation_gate = dict(validation_terminal["promotion_gate"])
    test_gate = dict(test_terminal["promotion_gate"])
    speed_target = bool(
        float(validation_gate["mean_end_to_end_speedup_vs_traditional"]) >= 10.0
        and float(validation_gate["p95_end_to_end_speedup_vs_traditional"]) >= 10.0
        and float(test_gate["mean_end_to_end_speedup_vs_traditional"]) >= 10.0
        and float(test_gate["p95_end_to_end_speedup_vs_traditional"]) >= 10.0
    )
    terminal = {
        **dict(context),
        "status": "complete" if instance_target else "target_not_met",
        "joint_global_and_instance_target_met": instance_target,
        "global_operator_target_required": False,
        "instance_same_protocol_target_met": instance_target,
        "instance_validation_target_met": validation_target,
        "instance_test_id_target_met": test_target,
        "instance_end_to_end_speed_target_met": speed_target,
        "test_id_opened": True,
        "test_id_authorization": "new_frozen_validation_gate_passed",
        "cpadc_terminal": validation_terminal,
        "cpadc_test_id_terminal": test_terminal,
        "claim": (
            "instance <=5% accuracy and >=10x end-to-end speed passed new validation and independent test_id"
            if instance_target
            else "independent confirmation complete; instance target not yet achieved"
        ),
    }
    _atomic_json(terminal, terminal_path)
    _atomic_json(terminal, status_path)
    return 0 if instance_target else 5


def _wait_for_pretraining(
    *,
    terminal_path: Path,
    training_pid: int,
    poll_seconds: int,
    status_path: Path,
) -> dict[str, object]:
    while True:
        terminal = _read_json(terminal_path)
        if terminal is not None and terminal.get("status") in {"complete", "failed"}:
            return terminal
        if not _pid_alive(training_pid):
            raise RuntimeError(
                f"pretraining process {training_pid} exited without a terminal record"
            )
        _atomic_json(
            {
                "status": "waiting_for_pretraining",
                "training_pid": int(training_pid),
                "updated_unix_s": time.time(),
            },
            status_path,
        )
        time.sleep(int(poll_seconds))


def _accepted_pretraining_parent(
    artifact: Path,
    terminal: Mapping[str, object],
) -> tuple[Path, Path, Path | None, str]:
    run = artifact / "run"
    identity = run / "run_identity.json"
    if not identity.is_file():
        raise FileNotFoundError("pretraining run identity is unavailable")
    if terminal.get("status") == "complete":
        checkpoint = run / "best.pt"
        report = run / "best.json"
        if not checkpoint.is_file() or not report.is_file():
            raise FileNotFoundError("completed pretraining best artifacts are incomplete")
        return checkpoint, identity, report, "completed_best"
    is_plateau = (
        terminal.get("status") == "failed"
        and "validation did not improve"
        in str(terminal.get("error", ""))
    )
    if not is_plateau:
        raise RuntimeError("pretraining did not finish successfully")
    control = _read_json(run / "epoch_validation_control.json")
    if control is None:
        raise FileNotFoundError("plateaued pretraining lacks accepted control state")
    checkpoint = Path(str(control.get("checkpoint", ""))).resolve()
    accepted_epoch = int(control.get("accepted_epoch", -1))
    if not checkpoint.is_file() or accepted_epoch < 0:
        raise FileNotFoundError("plateaued pretraining accepted checkpoint is invalid")
    return checkpoint, identity, None, f"plateau_accepted_epoch_{accepted_epoch}"


def _wait_for_free_gpus(*, poll_seconds: int, status_path: Path) -> None:
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        pids = tuple(value.strip() for value in result.stdout.splitlines() if value.strip())
        if result.returncode == 0 and not pids:
            return
        _atomic_json(
            {
                "status": "waiting_for_exclusive_gpus",
                "compute_pids": pids,
                "updated_unix_s": time.time(),
            },
            status_path,
        )
        time.sleep(int(poll_seconds))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-config", type=Path, required=True)
    parser.add_argument("--pretraining-artifact", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")

    artifact = args.pretraining_artifact.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / "pipeline.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another target10 fine-tuning supervisor owns the lock") from error
    _atomic_text(f"{os.getpid()}\n", output / "supervisor.pid")
    status_path = output / "pipeline_status.json"
    terminal_path = output / "pipeline_terminal.json"
    environment = dict(os.environ)
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["OMP_NUM_THREADS"] = "8"

    try:
        pretraining_terminal = _wait_for_pretraining(
            terminal_path=artifact / "run" / "terminal.json",
            training_pid=int(args.training_pid),
            poll_seconds=int(args.poll_seconds),
            status_path=status_path,
        )
        (
            parent_checkpoint,
            parent_identity,
            parent_report,
            parent_selection,
        ) = _accepted_pretraining_parent(artifact, pretraining_terminal)
        _wait_for_free_gpus(
            poll_seconds=int(args.poll_seconds), status_path=status_path
        )

        # Establish the speed denominator first, on the same now-exclusive GPU
        # that will execute deployment inference.  The fastest observed
        # traditional single-shot solve is deliberately used as the conservative
        # reference for the required 10x speedup.
        cpadc_template = (
            ROOT
            / "configs/saved_time_v5/causal_defect_basis_marmousi1_4m_v2_target10_rank32.yaml"
        )
        cpadc_template_payload = yaml.safe_load(cpadc_template.read_text())
        if not isinstance(cpadc_template_payload, dict):
            raise ValueError("CPADC template must contain a mapping")
        runtime_config = dict(
            cpadc_template_payload.get("runtime_benchmark", {}) or {}
        )
        minimum_speedup = float(
            runtime_config.get(
                "minimum_end_to_end_speedup_vs_traditional", 10.0
            )
        )
        if minimum_speedup < 10.0:
            raise ValueError("deployment speedup target must be at least 10x")
        runtime_report_path = output / "traditional_lwc84_runtime.json"
        runtime_report = _read_json(runtime_report_path)
        if runtime_report is None:
            _atomic_json(
                {"status": "benchmarking_traditional_lwc84_runtime"}, status_path
            )
            runtime_environment = dict(environment)
            runtime_environment["CUDA_VISIBLE_DEVICES"] = "0"
            code = _run_logged(
                [
                    sys.executable,
                    "-u",
                    "scripts/benchmark_lwc84_traditional_runtime.py",
                    "--source-h5",
                    str(cpadc_template_payload["source_h5"]),
                    "--solver-config",
                    str(runtime_config["solver_config"]),
                    "--output",
                    str(runtime_report_path),
                    "--device",
                    "cuda",
                    "--repeats-per-family",
                    str(runtime_config.get("repeats_per_family", 2)),
                    "--warmup-runs",
                    str(runtime_config.get("warmup_runs", 1)),
                ],
                log=output / "traditional_lwc84_runtime.log",
                environment=runtime_environment,
            )
            runtime_report = _read_json(runtime_report_path)
            if code != 0 or runtime_report is None:
                raise RuntimeError(
                    f"traditional LWC-84 runtime benchmark failed with return code {code}"
                )
        traditional_reference = _traditional_runtime_reference(runtime_report)
        deployment_gate = dict(
            cpadc_template_payload.get("deployment_gate", {}) or {}
        )
        deployment_gate.update(
            {
                "minimum_end_to_end_speedup_vs_traditional": minimum_speedup,
                "traditional_solver_reference_runtime_s": traditional_reference,
            }
        )

        asam_root = output / "asam"
        asam_config = output / "bound_asam.yaml"
        _write_bound_config(
            ROOT / "configs/saved_time_v4/asam_full_support_adaptive_v2.yaml",
            asam_config,
            {
                "parent_checkpoint": str(parent_checkpoint),
                "parent_identity": str(parent_identity),
                "parent_best_report": (
                    str(parent_report) if parent_report is not None else ""
                ),
                "artifact_dir": str(asam_root),
                "recompute_parent_baseline": True,
            },
        )
        asam_terminal = _read_json(asam_root / "run" / "terminal.json")
        if asam_terminal is None or asam_terminal.get("status") != "complete":
            _atomic_json({"status": "running_asam"}, status_path)
            asam_environment = dict(environment)
            asam_environment["CUDA_VISIBLE_DEVICES"] = "0"
            code = _run_logged(
                [
                    sys.executable,
                    "-u",
                    "scripts/refine_saved_time_v4_asam.py",
                    "--config",
                    str(asam_config),
                ],
                log=output / "asam.log",
                environment=asam_environment,
            )
            asam_terminal = _read_json(asam_root / "run" / "terminal.json")
            if code != 0 or asam_terminal is None:
                raise RuntimeError(f"ASAM refinement failed with return code {code}")

        refined_checkpoint = asam_root / "run" / "best.pt"
        refined_identity = asam_root / "run" / "run_identity.json"
        if not refined_checkpoint.is_file() or not refined_identity.is_file():
            raise FileNotFoundError("ASAM refinement checkpoint identity is incomplete")
        sealed_output = asam_root / "run" / "sealed_evaluation_validation"
        sealed_report = _read_json(sealed_output / "evaluation_report.json")
        if sealed_report is None or sealed_report.get("status") != "complete":
            _atomic_json({"status": "running_full_480x401_evaluation"}, status_path)
            eval_environment = dict(environment)
            eval_environment["CUDA_VISIBLE_DEVICES"] = "0"
            code = _run_logged(
                [
                    sys.executable,
                    "-u",
                    "scripts/evaluate_saved_time_v4_full_support.py",
                    "--config",
                    str(args.pretraining_config.resolve()),
                    "--checkpoint",
                    str(refined_checkpoint),
                    "--checkpoint-identity",
                    str(refined_identity),
                    "--output",
                    str(sealed_output),
                    "--time-block",
                    "16",
                ],
                log=output / "sealed_evaluation.log",
                environment=eval_environment,
            )
            sealed_report = _read_json(sealed_output / "evaluation_report.json")
            if code != 0 or sealed_report is None:
                raise RuntimeError(
                    f"full 480x401 evaluation failed with return code {code}"
                )

        cpadc_root = output / "cpadc"
        cpadc_config = output / "bound_cpadc.yaml"
        _write_bound_config(
            cpadc_template,
            cpadc_config,
            {
                "parent_operator_config": str(args.pretraining_config.resolve()),
                "parent_checkpoint": str(refined_checkpoint),
                "parent_checkpoint_identity": str(refined_identity),
                "deployment_gate": deployment_gate,
                "traditional_runtime_report": str(runtime_report_path),
            },
        )
        raw_basis_checkpoint = cpadc_root / "run" / "best.pt"
        basis_terminal = _read_json(cpadc_root / "run" / "terminal.json")
        if basis_terminal is None or basis_terminal.get("status") != "complete":
            _atomic_json({"status": "running_cpadc_rank32_4096"}, status_path)
            train_environment = dict(environment)
            train_environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            code = _run_logged(
                [
                    "torchrun",
                    "--standalone",
                    "--nproc_per_node=4",
                    "scripts/train_causal_defect_basis.py",
                    "--config",
                    str(cpadc_config),
                    "--output-dir",
                    str(cpadc_root / "run"),
                    "--device",
                    "cuda",
                ],
                log=output / "cpadc_training.log",
                environment=train_environment,
            )
            basis_terminal = _read_json(cpadc_root / "run" / "terminal.json")
            if code != 0 or basis_terminal is None or not raw_basis_checkpoint.is_file():
                raise RuntimeError(f"CPADC basis training failed with return code {code}")

        calibration_per_family = 64
        calibration_seed = int(
            yaml.safe_load(cpadc_config.read_text()).get("seed", 372)
        ) + 10_000
        _atomic_json({"status": "running_cpadc_train_only_risk_calibration"}, status_path)
        calibration_shard_dirs: list[Path] = []
        calibration_processes: list[tuple[subprocess.Popen, object, Path]] = []
        for index in range(4):
            shard_dir = cpadc_root / "calibration_shards" / f"shard_{index}"
            calibration_shard_dirs.append(shard_dir)
            existing = _read_json(shard_dir / "terminal.json")
            if existing is not None and existing.get("status") == "complete":
                continue
            log_path = output / f"cpadc_calibration_shard_{index}.log"
            handle = log_path.open("a", encoding="utf8")
            command = [
                sys.executable,
                "-u",
                "scripts/run_causal_defect_adaptation.py",
                "--config",
                str(cpadc_config),
                "--basis-checkpoint",
                str(raw_basis_checkpoint),
                "--output-dir",
                str(shard_dir),
                "--device",
                "cuda",
                "--adaptation-device",
                "cpu",
                "--calibration-per-family",
                str(calibration_per_family),
                "--calibration-seed",
                str(calibration_seed),
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
                "--no-fields",
            ]
            handle.write(json.dumps({"event": "launch", "command": command}) + "\n")
            handle.flush()
            shard_environment = dict(environment)
            shard_environment["CUDA_VISIBLE_DEVICES"] = str(index)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=shard_environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            calibration_processes.append((process, handle, log_path))
        calibration_failures: list[dict[str, object]] = []
        for process, handle, log_path in calibration_processes:
            code = process.wait()
            handle.close()
            if code != 0:
                calibration_failures.append(
                    {"log": str(log_path), "return_code": code}
                )
        if calibration_failures:
            raise RuntimeError(
                f"CPADC calibration shards failed: {calibration_failures}"
            )
        calibration_dir = cpadc_root / "calibration"
        calibration_terminal = _read_json(calibration_dir / "terminal.json")
        basis_checkpoint = calibration_dir / "calibrated.pt"
        if calibration_terminal is None or calibration_terminal.get("status") != "complete":
            command = [
                sys.executable,
                "-u",
                "scripts/calibrate_causal_defect_risk.py",
                "--config",
                str(cpadc_config),
                "--basis-checkpoint",
                str(raw_basis_checkpoint),
                "--output-dir",
                str(calibration_dir),
                "--calibration-per-family",
                str(calibration_per_family),
                "--calibration-seed",
                str(calibration_seed),
                "--minimum-nonworse-fraction",
                "0.95",
                "--minimum-family-nonworse-fraction",
                "0.95",
                "--minimum-mean-improvement",
                "0.01",
                "--family-specific",
            ]
            for shard_dir in calibration_shard_dirs:
                command.extend(("--shard-dir", str(shard_dir)))
            code = _run_logged(
                command,
                log=output / "cpadc_calibration_merge.log",
                environment=environment,
            )
            calibration_terminal = _read_json(calibration_dir / "terminal.json")
            if code != 0 or calibration_terminal is None:
                raise RuntimeError(
                    f"CPADC risk calibration failed with return code {code}"
                )
        if not basis_checkpoint.is_file():
            raise FileNotFoundError("calibrated CPADC checkpoint is unavailable")

        # From this point onward all learned parameters and risk thresholds are
        # frozen.  The historical validation/test_id remain retrospective only;
        # generate and open the preregistered independent splits in strict order.
        return _run_independent_confirmation_pipeline(
            training_cpadc_config=cpadc_config,
            basis_checkpoint=basis_checkpoint,
            cpadc_root=cpadc_root,
            pipeline_output=output,
            status_path=status_path,
            terminal_path=terminal_path,
            environment=environment,
            context={
                "joint_global_and_instance_target_met": False,
                "global_operator_target_required": False,
                "maximum_required_relative_l2": MAXIMUM_RELATIVE_L2,
                "minimum_required_speedup_vs_traditional": MINIMUM_SPEEDUP,
                "traditional_runtime_benchmark": runtime_report,
                "pretraining_terminal": pretraining_terminal,
                "pretraining_parent_selection": parent_selection,
                "asam_terminal": asam_terminal,
                "historical_sealed_evaluation_retrospective_only": sealed_report,
                "cpadc_risk_calibration": calibration_terminal,
                "legacy_test_id_excluded_from_promotion": True,
            },
        )

        _atomic_json({"status": "running_cpadc_full_validation"}, status_path)
        shard_processes: list[tuple[subprocess.Popen, object, Path]] = []
        shard_dirs: list[Path] = []
        for index in range(4):
            shard_dir = cpadc_root / "validation_shards" / f"shard_{index}"
            shard_dirs.append(shard_dir)
            existing = _read_json(shard_dir / "terminal.json")
            if existing is not None and existing.get("status") == "complete":
                continue
            log_path = output / f"cpadc_validation_shard_{index}.log"
            handle = log_path.open("a", encoding="utf8")
            command = [
                sys.executable,
                "-u",
                "scripts/run_causal_defect_adaptation.py",
                "--config",
                str(cpadc_config),
                "--basis-checkpoint",
                str(basis_checkpoint),
                "--output-dir",
                str(shard_dir),
                "--device",
                "cuda",
                "--adaptation-device",
                "cpu",
                "--all-validation",
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
                "--no-fields",
            ]
            handle.write(json.dumps({"event": "launch", "command": command}) + "\n")
            handle.flush()
            shard_environment = dict(environment)
            shard_environment["CUDA_VISIBLE_DEVICES"] = str(index)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=shard_environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            shard_processes.append((process, handle, log_path))
        shard_failures: list[dict[str, object]] = []
        for process, handle, log_path in shard_processes:
            code = process.wait()
            handle.close()
            if code != 0:
                shard_failures.append({"log": str(log_path), "return_code": code})
        if shard_failures:
            raise RuntimeError(f"CPADC validation shards failed: {shard_failures}")

        merged_dir = cpadc_root / "evaluation"
        merge_terminal = _read_json(merged_dir / "terminal.json")
        if merge_terminal is None or merge_terminal.get("status") != "complete":
            command = [
                sys.executable,
                "-u",
                "scripts/merge_causal_defect_evaluations.py",
                "--config",
                str(cpadc_config),
                "--output-dir",
                str(merged_dir),
            ]
            for shard_dir in shard_dirs:
                command.extend(("--shard-dir", str(shard_dir)))
            code = _run_logged(
                command,
                log=output / "cpadc_merge.log",
                environment=environment,
            )
            merge_terminal = _read_json(merged_dir / "terminal.json")
            if code != 0 or merge_terminal is None:
                raise RuntimeError(f"CPADC validation merge failed with return code {code}")

        global_validation_target = _sealed_full_support_target_met(
            sealed_report, expected_split="validation"
        )
        instance_validation_target = _validation_authorizes_test_id(merge_terminal)
        if not instance_validation_target:
            terminal = {
                "status": "target_not_met",
                "joint_global_and_instance_target_met": False,
                "global_480x401_target_met": False,
                "global_validation_target_met": global_validation_target,
                "global_test_id_target_met": False,
                "instance_same_protocol_target_met": False,
                "instance_validation_target_met": False,
                "instance_test_id_target_met": False,
                "instance_end_to_end_speed_target_met": False,
                "test_id_opened": False,
                "test_id_authorization": (
                    "denied_by_complete_frozen_cpadc_validation_gate"
                ),
                "maximum_required_relative_l2": MAXIMUM_RELATIVE_L2,
                "minimum_required_speedup_vs_traditional": MINIMUM_SPEEDUP,
                "traditional_runtime_benchmark": runtime_report,
                "pretraining_terminal": pretraining_terminal,
                "pretraining_parent_selection": parent_selection,
                "asam_terminal": asam_terminal,
                "sealed_evaluation": sealed_report,
                "test_id_evaluation": None,
                "cpadc_terminal": merge_terminal,
                "cpadc_test_id_terminal": None,
                "cpadc_risk_calibration": calibration_terminal,
                "claim": (
                    "frozen validation did not pass; test_id remains sealed and "
                    "the target is not met"
                ),
            }
            _atomic_json(terminal, terminal_path)
            _atomic_json(terminal, status_path)
            return 5

        _atomic_json(
            {
                "status": "validation_passed_test_id_authorized",
                "test_id_opened": True,
            },
            status_path,
        )
        test_output = asam_root / "run" / "sealed_evaluation_test_id"
        test_report = _read_json(test_output / "evaluation_report.json")
        if test_report is None or test_report.get("status") != "complete":
            _atomic_json({"status": "running_test_id_480x401_evaluation"}, status_path)
            test_environment = dict(environment)
            test_environment["CUDA_VISIBLE_DEVICES"] = "0"
            code = _run_logged(
                [
                    sys.executable,
                    "-u",
                    "scripts/evaluate_saved_time_v4_full_support.py",
                    "--config",
                    str(args.pretraining_config.resolve()),
                    "--checkpoint",
                    str(refined_checkpoint),
                    "--checkpoint-identity",
                    str(refined_identity),
                    "--output",
                    str(test_output),
                    "--time-block",
                    "16",
                    "--evaluation-split",
                    "test_id",
                ],
                log=output / "test_id_evaluation.log",
                environment=test_environment,
            )
            test_report = _read_json(test_output / "evaluation_report.json")
            if code != 0 or test_report is None:
                raise RuntimeError(
                    f"test_id 480x401 evaluation failed with return code {code}"
                )

        _atomic_json({"status": "running_cpadc_test_id_confirmation"}, status_path)
        test_shard_dirs: list[Path] = []
        test_shard_processes: list[tuple[subprocess.Popen, object, Path]] = []
        for index in range(4):
            shard_dir = cpadc_root / "test_id_shards" / f"shard_{index}"
            test_shard_dirs.append(shard_dir)
            existing = _read_json(shard_dir / "terminal.json")
            if existing is not None and existing.get("status") == "complete":
                continue
            log_path = output / f"cpadc_test_id_shard_{index}.log"
            handle = log_path.open("a", encoding="utf8")
            command = [
                sys.executable,
                "-u",
                "scripts/run_causal_defect_adaptation.py",
                "--config",
                str(cpadc_config),
                "--basis-checkpoint",
                str(basis_checkpoint),
                "--output-dir",
                str(shard_dir),
                "--device",
                "cuda",
                "--adaptation-device",
                "cpu",
                "--all-test-id",
                "--shard-index",
                str(index),
                "--shard-count",
                "4",
                "--no-fields",
            ]
            handle.write(json.dumps({"event": "launch", "command": command}) + "\n")
            handle.flush()
            shard_environment = dict(environment)
            shard_environment["CUDA_VISIBLE_DEVICES"] = str(index)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=shard_environment,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            test_shard_processes.append((process, handle, log_path))
        test_shard_failures: list[dict[str, object]] = []
        for process, handle, log_path in test_shard_processes:
            code = process.wait()
            handle.close()
            if code != 0:
                test_shard_failures.append(
                    {"log": str(log_path), "return_code": code}
                )
        if test_shard_failures:
            raise RuntimeError(
                f"CPADC test_id shards failed: {test_shard_failures}"
            )
        test_merged_dir = cpadc_root / "test_id_evaluation"
        test_terminal = _read_json(test_merged_dir / "terminal.json")
        if test_terminal is None or test_terminal.get("status") != "complete":
            command = [
                sys.executable,
                "-u",
                "scripts/merge_causal_defect_evaluations.py",
                "--config",
                str(cpadc_config),
                "--output-dir",
                str(test_merged_dir),
                "--evaluation-split",
                "test_id",
            ]
            for shard_dir in test_shard_dirs:
                command.extend(("--shard-dir", str(shard_dir)))
            code = _run_logged(
                command,
                log=output / "cpadc_test_id_merge.log",
                environment=environment,
            )
            test_terminal = _read_json(test_merged_dir / "terminal.json")
            if code != 0 or test_terminal is None:
                raise RuntimeError(
                    f"CPADC test_id merge failed with return code {code}"
                )

        global_test_target = _sealed_full_support_target_met(
            test_report, expected_split="test_id"
        )
        instance_test_target = _cpadc_target_met(
            test_terminal, expected_split="test_id"
        )
        global_target = global_validation_target and global_test_target
        instance_target = instance_validation_target and instance_test_target
        instance_speed_target = bool(
            float(
                dict(merge_terminal["promotion_gate"])[
                    "p95_end_to_end_speedup_vs_traditional"
                ]
            )
            >= 10.0
            and float(
                dict(test_terminal["promotion_gate"])[
                    "p95_end_to_end_speedup_vs_traditional"
                ]
            )
            >= 10.0
        )
        terminal = {
            "status": "complete" if instance_target else "target_not_met",
            "joint_global_and_instance_target_met": global_target and instance_target,
            "global_480x401_target_met": global_target,
            "global_validation_target_met": global_validation_target,
            "global_test_id_target_met": global_test_target,
            "instance_same_protocol_target_met": instance_target,
            "instance_validation_target_met": instance_validation_target,
            "instance_test_id_target_met": instance_test_target,
            "instance_end_to_end_speed_target_met": instance_speed_target,
            "test_id_opened": True,
            "test_id_authorization": "complete_frozen_cpadc_validation_gate_passed",
            "maximum_required_relative_l2": MAXIMUM_RELATIVE_L2,
            "minimum_required_speedup_vs_traditional": MINIMUM_SPEEDUP,
            "traditional_runtime_benchmark": runtime_report,
            "pretraining_terminal": pretraining_terminal,
            "pretraining_parent_selection": parent_selection,
            "asam_terminal": asam_terminal,
            "sealed_evaluation": sealed_report,
            "test_id_evaluation": test_report,
            "cpadc_terminal": merge_terminal,
            "cpadc_test_id_terminal": test_terminal,
            "cpadc_risk_calibration": calibration_terminal,
            "claim": (
                "instance <=5% accuracy and 10x speed targets achieved on validation and test_id"
                if instance_target
                else "pipeline complete; instance <=5% accuracy and 10x speed targets not yet achieved"
            ),
        }
        _atomic_json(terminal, terminal_path)
        _atomic_json(terminal, status_path)
        return 0 if instance_target else 5
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_unix_s": time.time(),
        }
        _atomic_json(failure, terminal_path)
        _atomic_json(failure, status_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_cpadc_target_met",
    "_sealed_full_support_target_met",
    "_traditional_runtime_reference",
    "_validation_authorizes_test_id",
]
