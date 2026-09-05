#!/usr/bin/env python3
"""Train-only K4 variance-reduction mechanism audit for the frozen r3 model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import h5py


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

import scripts.audit_transfer_dg_r3_exact_gradient_sampling as exact  # noqa: E402
from scripts.train_transfer_dg_parent_anchored_k4 import (  # noqa: E402
    FAMILIES,
    GuardedParentCache,
    LOSS_CONFIG,
    MECHANISM_PANEL,
    OFFSETS,
    STARTS,
    TruthAccessGuard,
    atomic_json,
    k4_indices,
    sha256,
    training_window,
)
from saved_time_phase_operator_v4.parent_anchored_relative_loss import (  # noqa: E402
    unbiased_window_weights,
)
from saved_time_phase_operator_v4.parent_anchored_block import (  # noqa: E402
    render_retained_rfft_frames,
)


SCHEMA = "transfer_dg_parent_anchored_k4_mechanism_v1"
EXPECTED_LEDGER = {
    "authorized_call_count": 132,
    "purpose_call_counts": {"mechanism": 132},
    "role_call_counts": {"fit": 132},
    "unique_sample_count": 4,
    "unique_group_count": 4,
    "unique_sample_hash_count": 4,
    "unique_sample_id_digest": "89bd437aab87501634e95699ed9fcd5b76b58cc3db38bda2d0360bac6452f2e0",
    "unique_group_digest": "b6e9651fddd8b55adfd95ba0b114a0be94a95646cc562455ec7cc41e429fef2b",
    "unique_sample_hash_digest": "4a15dfcfb6e0e1609567d80afc2f1b5f069a69d581e1d6fcfaac73f4ff10a90e",
    "slice_start_min": 0,
    "slice_stop_max_exclusive": 401,
    "confirmation_opened": False,
    "validation_opened": False,
    "test_id_opened": False,
}


def mechanism_environment() -> dict[str, Any]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }


def mechanism_ledger_passed(ledger: Mapping[str, Any]) -> bool:
    return all(ledger.get(key) == value for key, value in EXPECTED_LEDGER.items())


def partial_truth_ledger(guard: TruthAccessGuard) -> dict[str, Any]:
    """Expose the pre-created guard ledger even when cache construction fails."""

    return guard.summary()


def validate_mechanism_contract(
    args: argparse.Namespace, prereg: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, Any], str]:
    if prereg.get("schema") != "transfer_dg_parent_anchored_k4_preregistration_v1":
        raise RuntimeError("mechanism preregistration schema drift")
    if prereg.get("candidate") != "transfer_dg_parent_anchored_k4_r4_20260904":
        raise RuntimeError("mechanism candidate drift")
    if prereg.get("status") != "draft_pending_audit":
        raise RuntimeError("mechanism preregistration status is not draft_pending_audit")
    frozen = prereg["paths"]
    supplied = {
        "preregistration": args.preregistration.resolve(),
        "mechanism_selection": args.selection_manifest.resolve(),
        "cache": args.cache.resolve(),
        "source_h5": args.source_h5.resolve(),
        "r3_best": args.checkpoint.resolve(),
        "mechanism_output": args.output.resolve(),
    }
    for key, supplied_path in supplied.items():
        frozen_path = Path(frozen[key])
        if not frozen_path.is_absolute():
            frozen_path = ROOT / frozen_path
        if supplied_path != frozen_path.resolve():
            raise RuntimeError(f"mechanism path override rejected: {key}")
    environment = mechanism_environment()
    if environment["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8" or environment["CUDA_VISIBLE_DEVICES"] != "0":
        raise RuntimeError("mechanism environment contract drift")
    bindings = prereg["bindings"]
    binding_paths = {
        "mechanism_audit_sha256": Path(__file__),
        "trainer_sha256": ROOT / "scripts/train_transfer_dg_parent_anchored_k4.py",
        "launcher_sha256": ROOT / "scripts/launch_transfer_dg_parent_anchored_k4.py",
        "test_sha256": ROOT / "tests/saved_time_phase_operator_v4/test_parent_anchored_k4.py",
        "corrector_sha256": ROOT / "saved_time_phase_operator_v4/parent_anchored_block.py",
        "relative_loss_sha256": ROOT / "saved_time_phase_operator_v4/parent_anchored_relative_loss.py",
        "r3_trainer_sha256": ROOT / frozen["r3_trainer"],
        "exact_gradient_audit_sha256": ROOT / "scripts/audit_transfer_dg_r3_exact_gradient_sampling.py",
        "exact_gradient_report_sha256": ROOT / frozen["exact_gradient_report"],
        "exact_gradient_selection_sha256": ROOT / frozen["exact_gradient_selection"],
        "cache_sha256": args.cache,
        "source_h5_sha256": args.source_h5,
        "selection_136_manifest_sha256": ROOT / frozen["selection_136_manifest"],
        "full_2800_manifest_sha256": ROOT / frozen["full_2800_manifest"],
        "baseline_metrics_sha256": ROOT / frozen["baseline_metrics"],
        "r3_best_sha256": args.checkpoint,
        "parent_checkpoint_sha256": ROOT / frozen["parent_checkpoint"],
    }
    hashes = {key: sha256(path) for key, path in binding_paths.items()}
    drift = {key: value for key, value in hashes.items() if bindings.get(key) != value}
    if drift:
        raise RuntimeError(f"mechanism binding drift: {sorted(drift)}")
    baseline_lines = (ROOT / frozen["baseline_metrics"]).read_bytes().splitlines()
    line3_sha256 = hashlib.sha256(baseline_lines[2]).hexdigest() if len(baseline_lines) >= 3 else ""
    if line3_sha256 != bindings["baseline_line3_sha256"]:
        raise RuntimeError("mechanism baseline line-3 digest drift")
    hashes["baseline_line3_sha256"] = line3_sha256
    preregistration_sha256 = sha256(args.preregistration)
    return hashes, environment, preregistration_sha256


def cosine_strict(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm, right_norm = float(left.norm()), float(right.norm())
    if min(left_norm, right_norm) < 1.0e-12:
        raise RuntimeError("degenerate gradient cosine")
    return float(torch.dot(left.double(), right.double()) / (left_norm * right_norm))


def relative_norm_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.double() - right.double()).norm()) / max(float(right.norm()), 1.0e-12)


def sampling_cv(gradients: Sequence[torch.Tensor], mean: torch.Tensor) -> float:
    variance = torch.stack([(gradient.double() - mean.double()).square().sum() for gradient in gradients]).mean()
    return math.sqrt(float(variance)) / max(float(mean.norm()), 1.0e-12)


def k4_means(k1_gradients: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if len(k1_gradients) != len(STARTS):
        raise ValueError("K4 mechanism requires exactly 13 K1 gradients")
    return [
        torch.stack([k1_gradients[index] for index in k4_indices(anchor)]).mean(0)
        for anchor in range(len(STARTS))
    ]


def mechanism_classification(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "per_record_k4_mean_algebra": all(
            row["k4_mean_vs_ht_cosine"] >= 0.999999
            and row["k4_mean_vs_ht_relative_norm_difference"] <= 1.0e-5
            for row in metrics["records"]
        ),
        "median_cv_ratio": float(metrics["median_cv_k4_over_k1"]) <= 0.70,
        "three_of_four_cv_k4": sum(row["cv_k4"] <= 1.0 for row in metrics["records"]) >= 3,
        "k4_vs_full_main_median": float(metrics["k4_vs_own_full_main_cosine_median"]) >= 0.95,
        "aggregate_cosine": float(metrics["aggregate_mean_k4_vs_full_main_cosine"]) >= 0.995,
        "clip_fraction": float(metrics["k4_clip_fraction"]) < 0.25,
        "finite": bool(metrics["finite"]),
        "rollback": bool(metrics["rollback"]),
            "truth_ledger": bool(metrics["truth_ledger_passed"]),
    }
    return {
        "status": "passed" if all(gates.values()) else "rejected",
        "classification": "k4_sampling_mechanism_supported" if all(gates.values()) else "k4_sampling_mechanism_rejected",
        "gates": gates,
    }


def gradient_for_window(
    model: torch.nn.Module,
    cache: GuardedParentCache,
    position: int,
    start: int,
    parameters: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> torch.Tensor:
    coefficients, condition = cache.parent_and_condition(position, device)
    prediction, target, parent = training_window(
        model,
        coefficients,
        condition,
        cache,
        position,
        loss_start=start,
        purpose="mechanism",
    )
    frame, delta = unbiased_window_weights(
        loss_start=start,
        sample_length=int(prediction.shape[1]),
        time_count=cache.time_count,
        block_size=32,
        rollout_blocks=2,
        device=device,
        dtype=prediction.dtype,
    )
    energy, delta_energy = cache.full_future_energies(position, purpose="mechanism")
    objectives = exact.component_objectives(
        prediction,
        target,
        parent,
        frame_weights=frame,
        delta_weights=delta,
        full_target_energy=energy,
        full_target_delta_energy=delta_energy,
    )
    gradients = exact.gradients_for_objectives(objectives, parameters)
    return gradients["total"]


def full_main_gradient(
    model: torch.nn.Module,
    cache: GuardedParentCache,
    position: int,
    parameters: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> torch.Tensor:
    coefficients, condition = cache.parent_and_condition(position, device)
    parent = render_retained_rfft_frames(
        coefficients,
        time_count=cache.time_count,
        frame_indices=torch.arange(cache.time_count, device=device),
    ).clone()
    parent[..., 0, :] = 0.0
    target = cache.truth(position, 0, cache.time_count, device, purpose="mechanism")
    prediction = model.rollout(parent, condition)
    energy, delta_energy = cache.full_future_energies(position, purpose="mechanism")
    objectives = exact.component_objectives(
        prediction[:, 2:],
        target[:, 2:],
        parent[:, 2:],
        frame_weights=torch.ones(399, device=device, dtype=prediction.dtype),
        delta_weights=torch.ones(398, device=device, dtype=prediction.dtype),
        full_target_energy=energy,
        full_target_delta_energy=delta_energy,
    )
    gradients = exact.gradients_for_objectives(objectives, parameters)
    return gradients["main"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    fixed_output = Path(prereg["paths"]["mechanism_output"])
    if not fixed_output.is_absolute():
        fixed_output = ROOT / fixed_output
    fixed_output = fixed_output.resolve()
    if fixed_output.exists():
        raise FileExistsError("refusing to overwrite mechanism report")
    started = time.time()
    cache = None
    model = None
    initial_state = None
    guard = TruthAccessGuard(mechanism_panel=MECHANISM_PANEL)
    hashes_before: dict[str, str] = {}
    environment = mechanism_environment()
    preregistration_sha256_before = sha256(args.preregistration)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate": "transfer_dg_parent_anchored_k4_r4_20260904",
        "status": "running",
        "argv": list(sys.argv),
        "observed_environment": environment,
        "preregistration_sha256_observed_before": preregistration_sha256_before,
        "promotion_authorized": False,
        "confirmation_opened": False,
        "validation_opened": False,
        "test_id_opened": False,
    }
    try:
        hashes_before, environment, preregistration_sha256_before = validate_mechanism_contract(
            args, prereg
        )
        report["input_hashes_before"] = hashes_before
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        records = selection["records"]
        if [record["family"] for record in records] != list(FAMILIES):
            raise RuntimeError("mechanism family order drift")
        if {record["sample_id"] for record in records} != MECHANISM_PANEL:
            raise RuntimeError("mechanism selection drift")
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(372)
        torch.cuda.manual_seed_all(372)
        np.random.seed(372)
        device = torch.device("cuda:0")
        cache = GuardedParentCache(args.cache, args.source_h5, purpose="mechanism", guard=guard)
        model = exact.load_model(args.checkpoint, device)
        _, parameters = exact.declared_parameters(model)
        initial_state = exact.capture_complete_state(model)
        rows = []
        all_k4_cosines = []
        all_k4_norms = []
        mean_k4_per_record = []
        full_per_record = []
        for record in records:
            position = int(record["cache_position"])
            if str(cache.sample_ids[position]) != record["sample_id"]:
                raise RuntimeError("mechanism cache-position drift")
            k1 = [gradient_for_window(model, cache, position, start, parameters, device) for start in STARTS]
            ght = torch.stack(k1).mean(0)
            if float(ght.norm()) < 1.0e-12:
                raise RuntimeError("degenerate per-record HT gradient")
            k4 = k4_means(k1)
            mean_k4 = torch.stack(k4).mean(0)
            full = full_main_gradient(model, cache, position, parameters, device)
            k4_cosines = [cosine_strict(gradient, full) for gradient in k4]
            k4_norms = [float(gradient.norm()) for gradient in k4]
            cv1 = sampling_cv(k1, ght)
            cv4 = sampling_cv(k4, mean_k4)
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "family": record["family"],
                    "cv_k1": cv1,
                    "cv_k4": cv4,
                    "cv_ratio": cv4 / max(cv1, 1.0e-12),
                    "k4_mean_vs_ht_cosine": cosine_strict(mean_k4, ght),
                    "k4_mean_vs_ht_relative_norm_difference": relative_norm_difference(mean_k4, ght),
                    "k4_vs_full_main_cosines": k4_cosines,
                    "k4_norms": k4_norms,
                }
            )
            all_k4_cosines.extend(k4_cosines)
            all_k4_norms.extend(k4_norms)
            mean_k4_per_record.append(mean_k4)
            full_per_record.append(full)
            exact.assert_complete_state_equal(model, initial_state)
        aggregate_k4 = torch.stack(mean_k4_per_record).mean(0)
        aggregate_full = torch.stack(full_per_record).mean(0)
        ledger = guard.summary()
        metrics = {
            "records": rows,
            "median_cv_k4_over_k1": float(np.median([row["cv_ratio"] for row in rows])),
            "k4_vs_own_full_main_cosine_median": float(np.median(all_k4_cosines)),
            "aggregate_mean_k4_vs_full_main_cosine": cosine_strict(aggregate_k4, aggregate_full),
            "k4_clip_fraction": float(np.mean(np.asarray(all_k4_norms) > 1.0)),
            "finite": all(
                math.isfinite(value)
                for row in rows
                for value in [row["cv_k1"], row["cv_k4"], row["cv_ratio"], *row["k4_vs_full_main_cosines"], *row["k4_norms"]]
            ),
            "rollback": True,
            "truth_ledger_passed": mechanism_ledger_passed(ledger),
        }
        decision = mechanism_classification(metrics)
        exact.restore_complete_state(model, initial_state)
        exact.assert_complete_state_equal(model, initial_state)
        hashes_after, environment_after, preregistration_sha256_after = validate_mechanism_contract(
            args, prereg
        )
        if hashes_after != hashes_before:
            raise RuntimeError("mechanism input mutated")
        if preregistration_sha256_after != preregistration_sha256_before:
            raise RuntimeError("mechanism preregistration mutated")
        if environment_after != environment:
            raise RuntimeError("mechanism environment mutated")
        resources = {
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        if resources["elapsed_seconds"] > 300 or resources["peak_allocated_bytes"] >= 8 * 2**30:
            raise RuntimeError("mechanism resource budget exceeded")
        report.update(
            {
                "status": decision["status"],
                "classification": decision["classification"],
                "gates": decision["gates"],
                "metrics": metrics,
                "truth_ledger": ledger,
                "input_hashes_before": hashes_before,
                "input_hashes_after": hashes_after,
                "preregistration_sha256_observed_after": preregistration_sha256_after,
                "preregistration_hash_unchanged": True,
                "resources": resources,
                "promotion_authorized": False,
                "confirmation_opened": False,
                "validation_opened": False,
                "test_id_opened": False,
            }
        )
        encoded = json.dumps(report, sort_keys=True, allow_nan=False).encode()
        if len(encoded) >= 16 * 2**20:
            raise RuntimeError("mechanism disk budget exceeded")
        atomic_json(report, fixed_output)
        return 0 if decision["status"] == "passed" else 2
    except Exception as error:
        rollback_passed = initial_state is None
        if model is not None and initial_state is not None:
            try:
                exact.restore_complete_state(model, initial_state)
                exact.assert_complete_state_equal(model, initial_state)
                rollback_passed = True
            except Exception as rollback_error:
                rollback_passed = False
                report["rollback_error"] = repr(rollback_error)
        report["partial_truth_ledger"] = partial_truth_ledger(guard)
        try:
            hashes_after, environment_after, preregistration_sha256_after = validate_mechanism_contract(
                args, prereg
            )
            report["input_hashes_after"] = hashes_after
            report["preregistration_sha256_observed_after"] = preregistration_sha256_after
            report["preregistration_hash_unchanged"] = (
                preregistration_sha256_after == preregistration_sha256_before
            )
            report["observed_environment_after"] = environment_after
        except Exception as after_error:
            report["after_audit_error"] = repr(after_error)
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_initialized() else 0
        report["resources"] = {
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": peak,
        }
        report["rollback_passed"] = rollback_passed
        report.update(
            {
                "status": "invalid",
                "classification": "invalid",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "promotion_authorized": False,
                "confirmation_opened": False,
                "validation_opened": False,
                "test_id_opened": False,
            }
        )
        atomic_json(report, fixed_output)
        raise
    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
