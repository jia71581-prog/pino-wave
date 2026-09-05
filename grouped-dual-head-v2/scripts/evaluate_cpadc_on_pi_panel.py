#!/usr/bin/env python3
"""Evaluate frozen CPADC on the exact records and frames used by frozen PI-DeepONet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts import run_causal_defect_adaptation as cpadc
from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    ConvexDefectCorrectionResult,
)


FAMILIES = ("uniform", "layered", "marmousi")
PI_CHECKPOINT_SHA256 = (
    "e2d6d7a9ec5481268bae2f41cff2400ee78b93a3a9216b2c338db572d405c4c1"
)
CPADC_CHECKPOINT_SHA256 = (
    "6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b"
)
CPADC_PARENT_CHECKPOINT_SHA256 = (
    "395b83560db3342b1e2331de2d8dde2994640d56a53dfbec8c99ece4bbb177bf"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative(terms: tuple[float, float] | list[float]) -> float:
    return float(math.sqrt(float(terms[0]) / max(float(terms[1]), 1.0e-30)))


def _add(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float]:
    return float(left[0] + right[0]), float(left[1] + right[1])


def _terms(
    prediction: torch.Tensor, target: torch.Tensor, time_indices: list[int]
) -> tuple[float, float]:
    indices = torch.as_tensor(time_indices, dtype=torch.long, device=prediction.device)
    prediction64 = prediction.index_select(1, indices).double()
    target64 = target.to(prediction.device).index_select(1, indices).double()
    return (
        float((prediction64 - target64).square().sum()),
        float(target64.square().sum()),
    )


def load_pi_panel(
    paths: list[Path], *, split: str, expected_records: int = 480
) -> tuple[dict[str, dict], dict[str, object]]:
    workers = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(workers) != 4:
        raise ValueError("the frozen PI panel requires exactly four worker reports")
    if any(worker.get("schema") != "frozen_pi_deeponet_split_worker_v1" for worker in workers):
        raise ValueError("unexpected PI worker schema")
    if any(worker.get("status") != "complete" for worker in workers):
        raise ValueError("all PI workers must be complete")
    if any(worker.get("split") != split for worker in workers):
        raise ValueError("PI worker split mismatch")
    if any(int(worker.get("frames_per_record", 0)) != 32 for worker in workers):
        raise ValueError("PI comparison panel must contain 32 exact frames")
    shards = {(int(worker["shard_index"]), int(worker["num_shards"])) for worker in workers}
    if shards != {(index, 4) for index in range(4)}:
        raise ValueError(f"incomplete PI worker shard set: {sorted(shards)}")
    checkpoint_hashes = {str(worker["checkpoint"]["sha256"]) for worker in workers}
    if checkpoint_hashes != {PI_CHECKPOINT_SHA256}:
        raise ValueError("frozen PI checkpoint binding mismatch")
    manifest_digests = {str(worker["bindings"]["manifest_digest"]) for worker in workers}
    if len(manifest_digests) != 1:
        raise ValueError("PI worker manifest binding mismatch")

    panel: dict[str, dict] = {}
    for worker in workers:
        for row in worker["measurements"]:
            sample_id = str(row["sample_id"])
            indices = [int(value) for value in row["time_indices"]]
            if sample_id in panel:
                raise ValueError(f"duplicate PI panel sample: {sample_id}")
            if len(indices) != 32 or len(indices) != len(set(indices)):
                raise ValueError(f"invalid PI frame panel for {sample_id}")
            panel[sample_id] = {
                "sample_id": sample_id,
                "source_index": int(row["source_index"]),
                "family": str(row["family"]),
                "time_indices": indices,
                "pi_error_numerator": float(row["error_numerator"]),
                "truth_denominator": float(row["truth_denominator"]),
                "pi_relative_l2": float(row["relative_l2"]),
            }
    if len(panel) != expected_records:
        raise ValueError(f"PI panel coverage is {len(panel)}, expected {expected_records}")
    if any(row["family"] not in FAMILIES for row in panel.values()):
        raise ValueError("PI panel contains an unsupported family")
    metadata = {
        "checkpoint_sha256": PI_CHECKPOINT_SHA256,
        "manifest_digest": next(iter(manifest_digests)),
        "worker_paths": [str(path.resolve()) for path in paths],
        "worker_sha256": {str(path.resolve()): _sha256(path) for path in paths},
    }
    return panel, metadata


def _comparison_evaluator(
    panel: dict[str, dict],
    original: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    def evaluate(adapted_artifact, target, **kwargs):
        report = original(adapted_artifact, target, **kwargs)
        sample_ids = tuple(str(value) for value in kwargs["sample_ids"])
        families = tuple(str(value) for value in kwargs["families"])
        if len(sample_ids) != 1 or len(families) != 1:
            raise ValueError("matched CPADC comparison expects one record at a time")
        sample_id = sample_ids[0]
        if sample_id not in panel:
            raise ValueError(f"CPADC sample is absent from the PI panel: {sample_id}")
        pi_row = panel[sample_id]
        if families[0] != pi_row["family"]:
            raise ValueError(f"family mismatch for {sample_id}")
        adapted = torch.as_tensor(adapted_artifact["adapted_field"])
        parent = torch.as_tensor(adapted_artifact["parent_field"], device=adapted.device)
        truth = torch.as_tensor(target, device=adapted.device)
        adapted_terms = _terms(adapted, truth, pi_row["time_indices"])
        parent_terms = _terms(parent, truth, pi_row["time_indices"])
        denominator = float(adapted_terms[1])
        if not math.isclose(
            denominator,
            float(parent_terms[1]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-24,
        ):
            raise RuntimeError("CPADC matched-panel truth denominator changed")
        pi_denominator = float(pi_row["truth_denominator"])
        normalization_scale_squared = pi_denominator / max(denominator, 1.0e-30)
        adapted_error_pi_normalization = (
            float(adapted_terms[0]) * normalization_scale_squared
        )
        parent_error_pi_normalization = (
            float(parent_terms[0]) * normalization_scale_squared
        )
        if not math.isclose(
            _relative((adapted_error_pi_normalization, pi_denominator)),
            _relative(adapted_terms),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("CPADC normalization conversion changed relative L2")
        report["pi_matched_panel"] = {
            "frames_per_record": 32,
            "time_indices": list(pi_row["time_indices"]),
            "adapted_error_numerator": float(adapted_terms[0]),
            "parent_error_numerator": float(parent_terms[0]),
            "truth_denominator": denominator,
            "adapted_error_numerator_in_pi_normalization": (
                adapted_error_pi_normalization
            ),
            "parent_error_numerator_in_pi_normalization": (
                parent_error_pi_normalization
            ),
            "truth_denominator_in_pi_normalization": pi_denominator,
            "physical_to_pi_normalization_scale_squared": (
                normalization_scale_squared
            ),
            "adapted_relative_l2": _relative(adapted_terms),
            "parent_relative_l2": _relative(parent_terms),
            "pi_error_numerator": float(pi_row["pi_error_numerator"]),
            "pi_truth_denominator": pi_denominator,
            "pi_relative_l2": float(pi_row["pi_relative_l2"]),
        }
        return report

    return evaluate


def run(args: argparse.Namespace) -> dict[str, object]:
    if _sha256(args.basis_checkpoint) != CPADC_CHECKPOINT_SHA256:
        raise ValueError("frozen CPADC-R7 checkpoint binding mismatch")
    if _sha256(args.parent_checkpoint) != CPADC_PARENT_CHECKPOINT_SHA256:
        raise ValueError("frozen CPADC-R7 parent checkpoint binding mismatch")
    panel, pi_metadata = load_pi_panel(args.pi_workers, split=args.split)
    original_evaluator = cpadc.evaluate_after_adaptation
    original_cpml_loader = cpadc.saved_grid_cpml_config
    original_dataset_validator = cpadc.validate_dataset_cpml_contract
    original_parent_loader = cpadc._load_parent
    original_cpadc_sha256 = cpadc._sha256
    original_solver = cpadc.solve_causal_defect_correction

    class _LegacyEmptyCPML:
        @staticmethod
        def as_dict() -> dict[str, object]:
            return {}

    def legacy_cpml_loader(payload):
        if payload is None:
            return _LegacyEmptyCPML()
        return original_cpml_loader(payload)

    def legacy_dataset_validator(source_h5, *, saved_grid_cpml, **kwargs):
        if saved_grid_cpml == {}:
            return {}
        return original_dataset_validator(
            source_h5, saved_grid_cpml=saved_grid_cpml, **kwargs
        )

    configured_parent_path: Path | None = None

    def legacy_parent_loader(config, manifest, device):
        nonlocal configured_parent_path
        configured_parent_path = Path(str(config["parent_checkpoint"])).resolve()
        load_config = dict(config)
        load_config["parent_checkpoint"] = str(args.parent_checkpoint.resolve())
        return original_parent_loader(load_config, manifest, device)

    def legacy_cpadc_sha256(path):
        resolved = Path(path).expanduser().resolve()
        if configured_parent_path is not None and resolved == configured_parent_path:
            return CPADC_PARENT_CHECKPOINT_SHA256
        return original_cpadc_sha256(path)

    selected_sample_ids = tuple(args.sample_id or ())
    archive_shard_name = (
        "validation_shards" if args.split == "validation" else "test_id_shards"
    )
    archive_shard = (
        args.archived_cpadc_root.resolve()
        / archive_shard_name
        / f"shard_{args.shard_index}"
    )
    if not archive_shard.is_dir():
        raise FileNotFoundError(archive_shard)
    if selected_sample_ids:
        replay_queue = list(selected_sample_ids)
    else:
        archived_manifest = json.loads(
            (archive_shard / "instance_manifest.json").read_text(encoding="utf-8")
        )
        replay_queue = [str(row["sample_id"]) for row in archived_manifest]
    replay_position = 0
    archived_adaptation: dict[str, dict[str, object]] = {}

    def replay_archived_solver(basis, parent_field, *solver_args, **solver_kwargs):
        nonlocal replay_position
        if replay_position >= len(replay_queue):
            raise RuntimeError("archived CPADC replay queue was exhausted")
        sample_id = replay_queue[replay_position]
        replay_position += 1
        artifact_path = archive_shard / sample_id / "adaptation.pt"
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        metadata = dict(payload["adaptation"])
        if metadata.get("future_truth_used") is not False:
            raise RuntimeError(f"archived CPADC adaptation is not sealed: {sample_id}")
        archived_basis = dict(metadata["basis"])
        if archived_basis.get("checkpoint_sha256") != CPADC_CHECKPOINT_SHA256:
            raise RuntimeError(f"archived CPADC basis mismatch: {sample_id}")
        coefficients = torch.as_tensor(
            metadata["coefficients"],
            dtype=parent_field.dtype,
            device=parent_field.device,
        ).reshape(1, -1)
        if coefficients.shape[1] != basis.rank:
            raise RuntimeError(f"archived coefficient rank mismatch: {sample_id}")
        if not math.isclose(
            float(basis.trust_fraction[0]),
            float(metadata["trust_fraction"]),
            rel_tol=2.0e-5,
            abs_tol=2.0e-6,
        ):
            raise RuntimeError(f"archived CPADC basis replay drifted for {sample_id}")
        archived_adaptation[sample_id] = metadata
        field = parent_field + basis.combine(coefficients)
        solve_device = str(parent_field.device)

        def scalar(key: str) -> torch.Tensor:
            return torch.tensor(
                [float(metadata[key])],
                dtype=parent_field.dtype,
                device=parent_field.device,
            )

        return ConvexDefectCorrectionResult(
            field=field,
            coefficients=coefficients,
            accepted=torch.tensor(
                [bool(metadata["accepted"])],
                dtype=torch.bool,
                device=parent_field.device,
            ),
            rollback_reasons=(metadata.get("rollback_reason"),),
            objective_before=scalar("objective_before"),
            objective_after=scalar("objective_after"),
            condition_number=scalar("condition_number"),
            effective_design_rank=torch.tensor(
                [basis.rank], dtype=torch.long, device=parent_field.device
            ),
            design_row_count=0,
            correction_ratio=scalar("correction_ratio"),
            unconstrained_correction_ratio=scalar("unconstrained_correction_ratio"),
            projection_scale=scalar("projection_scale"),
            trust_fraction=scalar("trust_fraction"),
            effective_correction_ratio_limit=scalar(
                "effective_correction_ratio_limit"
            ),
            cholesky_jitter=0.0,
            coefficient_solve_elapsed_s=0.0,
            coefficient_solve_device=solve_device,
            coefficient_objective_device=solve_device,
            correction_materialization_device=solve_device,
            future_truth_used=False,
        )

    # R7 is a legacy schema-5 checkpoint. Its archived online solve explicitly
    # used cpml_config=None; the current evaluator later made CPML metadata
    # mandatory for newer schemas. Restore only the legacy empty-metadata path.
    cpadc.evaluate_after_adaptation = _comparison_evaluator(panel, original_evaluator)
    cpadc.saved_grid_cpml_config = legacy_cpml_loader
    cpadc.validate_dataset_cpml_contract = legacy_dataset_validator
    cpadc._load_parent = legacy_parent_loader
    cpadc._sha256 = legacy_cpadc_sha256
    cpadc.solve_causal_defect_correction = replay_archived_solver
    if selected_sample_ids and args.split != "validation":
        raise ValueError("sample-limited comparison smoke is validation-only")
    try:
        reports = cpadc.run(
            args.config,
            basis_checkpoint=args.basis_checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
            adaptation_device_name="cpu",
            sample_ids=selected_sample_ids or None,
            all_validation=args.split == "validation" and not selected_sample_ids,
            all_test_id=args.split == "test_id" and not selected_sample_ids,
            shard_index=args.shard_index,
            shard_count=args.num_shards,
            save_fields=False,
            travel_time_h5=args.travel_time_h5,
        )
    finally:
        cpadc.evaluate_after_adaptation = original_evaluator
        cpadc.saved_grid_cpml_config = original_cpml_loader
        cpadc.validate_dataset_cpml_contract = original_dataset_validator
        cpadc._load_parent = original_parent_loader
        cpadc._sha256 = original_cpadc_sha256
        cpadc.solve_causal_defect_correction = original_solver
    if replay_position != len(replay_queue):
        raise RuntimeError(
            f"archived CPADC replay coverage is {replay_position}/{len(replay_queue)}"
        )

    total = (0.0, 0.0)
    parent_total = (0.0, 0.0)
    family_terms = {family: (0.0, 0.0) for family in FAMILIES}
    parent_family_terms = {family: (0.0, 0.0) for family in FAMILIES}
    measurements = []
    for report in reports:
        sample_id = str(report["sample_id"])
        family = str(report["medium_type"])
        matched = dict(report["pi_matched_panel"])
        adapted_terms = (
            float(matched["adapted_error_numerator_in_pi_normalization"]),
            float(matched["truth_denominator_in_pi_normalization"]),
        )
        parent_terms = (
            float(matched["parent_error_numerator_in_pi_normalization"]),
            float(matched["truth_denominator_in_pi_normalization"]),
        )
        total = _add(total, adapted_terms)
        parent_total = _add(parent_total, parent_terms)
        family_terms[family] = _add(family_terms[family], adapted_terms)
        parent_family_terms[family] = _add(parent_family_terms[family], parent_terms)
        archived_evaluation = json.loads(
            (archive_shard / sample_id / "evaluation.json").read_text(
                encoding="utf-8"
            )
        )
        replay_error = float(report["future_fullfield_relative_l2"])
        archived_error = float(archived_evaluation["future_fullfield_relative_l2"])
        replay_parent_error = float(report["parent_future_fullfield_relative_l2"])
        archived_parent_error = float(
            archived_evaluation["parent_future_fullfield_relative_l2"]
        )
        if max(
            abs(replay_error - archived_error),
            abs(replay_parent_error - archived_parent_error),
        ) > 1.0e-3:
            raise RuntimeError(f"archived CPADC full-field replay drifted for {sample_id}")
        measurements.append(
            {
                "sample_id": sample_id,
                "source_index": int(panel[sample_id]["source_index"]),
                "family": family,
                **matched,
                "adaptation_accepted": bool(archived_adaptation[sample_id]["accepted"]),
                "adaptation_elapsed_s": float(
                    archived_adaptation[sample_id]["adaptation_elapsed_s"]
                ),
                "total_inference_elapsed_s_for_401_frames": float(
                    archived_adaptation[sample_id]["total_inference_elapsed_s"]
                ),
                "archived_full401_adapted_relative_l2": archived_error,
                "replayed_full401_adapted_relative_l2": replay_error,
                "adapted_full401_replay_delta": replay_error - archived_error,
                "archived_full401_parent_relative_l2": archived_parent_error,
                "replayed_full401_parent_relative_l2": replay_parent_error,
                "parent_full401_replay_delta": (
                    replay_parent_error - archived_parent_error
                ),
            }
        )
    expected_shard_count = (
        len(selected_sample_ids) if selected_sample_ids else len(panel) // args.num_shards
    )
    if len(measurements) != expected_shard_count:
        raise RuntimeError(
            f"CPADC shard coverage is {len(measurements)}, expected {expected_shard_count}"
        )
    output = {
        "schema": "frozen_cpadc_on_pi_panel_worker_v1",
        "status": "complete",
        "role": "comparison_only",
        "split": args.split,
        "selection_scope": (
            "limited_validation_smoke_on_frozen_pi_panel"
            if selected_sample_ids
            else "same_records_and_32_exact_frames_as_frozen_pi_deeponet"
        ),
        "frames_per_record": 32,
        "cpadc_full_output_frames_per_record": 401,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "global_record_count": len(panel) if not selected_sample_ids else len(selected_sample_ids),
        "record_count": len(measurements),
        "aggregate_relative_l2": _relative(total),
        "parent_aggregate_relative_l2": _relative(parent_total),
        "error_terms": {
            "normalization": "pi_deeponet_pressure_normalization_per_record",
            "aggregate": list(total),
            "parent_aggregate": list(parent_total),
            "family": {key: list(value) for key, value in family_terms.items()},
            "parent_family": {
                key: list(value) for key, value in parent_family_terms.items()
            },
        },
        "measurements": measurements,
        "cpadc_checkpoint": {
            "path": str(args.basis_checkpoint.resolve()),
            "sha256": CPADC_CHECKPOINT_SHA256,
        },
        "pi_panel": pi_metadata,
        "bindings": {
            "config_path": str(args.config.resolve()),
            "config_sha256": _sha256(args.config),
            "comparison_script_sha256": _sha256(Path(__file__)),
            "actual_parent_checkpoint": str(args.parent_checkpoint.resolve()),
            "actual_parent_checkpoint_sha256": CPADC_PARENT_CHECKPOINT_SHA256,
            "archived_cpadc_root": str(args.archived_cpadc_root.resolve()),
            "prediction_reconstruction": (
                "frozen_parent_plus_frozen_basis_times_archived_sealed_coefficients"
            ),
            "legacy_schema5_compatibility": (
                "empty_cpml_metadata_only_online_solve_still_passes_cpml_config_none"
            ),
        },
    }
    _atomic_json(output, args.output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--basis-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--archived-cpadc-root", type=Path, required=True)
    parser.add_argument("--travel-time-h5", type=Path)
    parser.add_argument("--pi-workers", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("validation", "test_id"), required=True)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
