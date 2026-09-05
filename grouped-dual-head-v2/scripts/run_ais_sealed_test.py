#!/usr/bin/env python3
"""Run the hash-bound native sealed test exactly once."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


Evaluator = Callable[[dict[str, Any], Path, str], dict[str, Any]]


def evaluate_scene_major(
    sample_ids: Sequence[int],
    load_scene: Callable[[int], Any],
    evaluators: Mapping[str, Callable[[Any], Any]],
) -> dict[str, list[Any]]:
    """Evaluate every registered model for one scene before loading the next scene."""
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("scene-major evaluation requires unique nonempty sample IDs")
    if not evaluators or any(not isinstance(name, str) or not name for name in evaluators):
        raise ValueError("scene-major evaluators require unique nonempty names")
    outputs = {name: [] for name in evaluators}
    for sample_id in sample_ids:
        scene = load_scene(sample_id)
        for name, evaluator in evaluators.items():
            outputs[name].append(evaluator(scene))
    return outputs


def compute_sealed_gate_bundle(
    baseline_rows, candidate_rows, *, threshold: float = 0.30,
    bootstrap_replicates: int = 10000, seed: int = 20260714, gate_evaluator=None,
):
    """Mirror compare_native400_accuracy individual and aggregate seed semantics."""
    from fno_acoustic.native400_gate import (
        aggregate_candidate_rows_by_sample, aggregate_seed_gates, evaluate_native400_gate,
    )
    evaluate = gate_evaluator or evaluate_native400_gate
    published = [
        evaluate(baseline_rows, rows, threshold, bootstrap_replicates, seed)
        for rows in candidate_rows
    ]
    aggregate_seed_results = [
        evaluate(baseline_rows, rows, threshold, bootstrap_replicates, seed + index)
        for index, rows in enumerate(candidate_rows)
    ]
    aggregate_rows = aggregate_candidate_rows_by_sample(candidate_rows)
    aggregate = evaluate(
        baseline_rows, aggregate_rows, threshold, bootstrap_replicates, seed
    )
    combined = aggregate_seed_gates(aggregate_seed_results, aggregate)
    return published, aggregate_seed_results, aggregate, combined


def _require_recipe_artifact_binding(
    authorization: dict[str, Any], recipe: dict[str, Any], recipe_path: Path,
    authorization_name: str, filename: str,
) -> None:
    files = recipe.get("files")
    binding = authorization.get(authorization_name)
    expected_path = (recipe_path.parent / filename).resolve()
    if not isinstance(files, dict) or not isinstance(binding, dict) or \
       binding.get("sha256") != files.get(filename) or \
       Path(binding.get("path", "")).resolve() != expected_path:
        raise ValueError(f"authorization {authorization_name} differs from frozen recipe inventory")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def _ids(value: object, name: str, exact: int | None = None) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                                         for item in value):
        raise ValueError(f"{name} must contain nonnegative integer IDs")
    if len(set(value)) != len(value) or (exact is not None and len(value) != exact):
        raise ValueError(f"{name} must contain exactly {exact or len(value)} unique IDs")
    return list(value)


def _bound_snapshot(binding: object, name: str) -> tuple[dict[str, str], bytes]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"authorization {name} binding is invalid")
    path, expected = binding["path"], binding["sha256"]
    if not isinstance(path, str) or not isinstance(expected, str):
        raise ValueError(f"authorization {name} binding is invalid")
    data = Path(path).read_bytes()
    if _sha(data) != expected:
        raise ValueError(f"authorization {name} hash mismatch")
    return {"path": path, "sha256": expected}, data


def _validate_authorization(path: Path) -> tuple[dict[str, Any], str]:
    authorization_bytes = Path(path).read_bytes()
    authorization = _json_bytes(authorization_bytes, "authorization")
    if authorization.get("schema_version") != 1:
        raise ValueError("unsupported sealed authorization schema")
    snapshots: dict[str, bytes] = {}
    for name in ("recipe", "experiment_manifest", "split_manifest", "b0_config",
                 "candidate_config", "b0_checkpoint", "aggregate_gate"):
        _, snapshots[name] = _bound_snapshot(authorization.get(name), name)
    if authorization.get("normalization_stats") is not None:
        _, snapshots["normalization_stats"] = _bound_snapshot(
            authorization.get("normalization_stats"), "normalization stats"
        )
    candidates = authorization.get("candidate_checkpoints")
    individual = authorization.get("individual_gates")
    if not isinstance(candidates, list) or len(candidates) != 3 or \
       not isinstance(individual, list) or len(individual) != 3:
        raise ValueError("authorization requires three checkpoints and three individual gates")
    for index, binding in enumerate(candidates):
        _, snapshots[f"candidate_checkpoint_{index}"] = _bound_snapshot(
            binding, f"candidate checkpoint {index}")
    for index, binding in enumerate(individual):
        _, snapshots[f"individual_gate_{index}"] = _bound_snapshot(
            binding, f"individual gate {index}")
    recipe = _json_bytes(snapshots["recipe"], "recipe")
    if _sha(snapshots["recipe"]) != authorization["recipe"]["sha256"] or \
       recipe.get("sealed_test_authorized") is not False:
        raise ValueError("recipe is not the frozen sealed recipe")
    experiment = _json_bytes(snapshots["experiment_manifest"], "experiment manifest")
    if experiment.get("recipe_sha256") != authorization["recipe"]["sha256"] or \
       experiment.get("confirmation_seeds") != authorization.get("confirmation_seeds"):
        raise ValueError("authorization experiment provenance mismatch")
    import yaml
    try:
        candidate_config = yaml.safe_load(snapshots["candidate_config"].decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("candidate config is not valid UTF-8 YAML") from exc
    canonical_config_hash = _sha(
        json.dumps(candidate_config, sort_keys=True, separators=(",", ":")).encode()
    )
    if authorization.get("candidate_config_canonical_sha256") != canonical_config_hash:
        raise ValueError("authorization canonical candidate config hash mismatch")
    split = _json_bytes(snapshots["split_manifest"], "split manifest")
    split_ids = _ids(split.get("test"), "split test", 250)
    authorized_ids = _ids(authorization.get("test_sample_ids"), "authorized test", 250)
    if split_ids != authorized_ids or authorization.get("test_sample_count") != 250:
        raise ValueError("authorization does not bind the exact 250 test IDs")
    if authorization.get("split_manifest_bytes_sha256") != _sha(snapshots["split_manifest"]):
        raise ValueError("authorization split manifest bytes hash mismatch")
    seeds = authorization.get("confirmation_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("authorization does not contain three independent seeds")
    if authorization.get("threshold") != 0.30 or \
       authorization.get("aggregation") != "per_sample_median_then_paired_bootstrap" or \
       authorization.get("required_individual_seed_passes") != 2:
        raise ValueError("authorization gate contract mismatch")
    if "files" in recipe:
        recipe_path = Path(authorization["recipe"]["path"])
        for authorization_name, filename in (
            ("b0_config", "config_b0.yaml"),
            ("candidate_config", "config_400.yaml"),
            ("split_manifest", "split_manifest.json"),
            ("normalization_stats", "normalization_stats.json"),
        ):
            _require_recipe_artifact_binding(
                authorization, recipe, recipe_path, authorization_name, filename
            )
        for descriptor_name, filename in (
            ("split_manifest", "split_manifest.json"),
            ("normalization_stats", "normalization_stats.json"),
        ):
            descriptor = recipe.get(descriptor_name)
            if not isinstance(descriptor, dict) or descriptor != {
                "file": filename, "sha256": recipe["files"][filename]
            }:
                raise ValueError(f"frozen recipe {descriptor_name} descriptor mismatch")
        if experiment.get("split_manifest_sha256") != authorization["split_manifest"]["sha256"] or \
           experiment.get("b0_checkpoint") != authorization.get("b0_checkpoint") or \
           experiment.get("candidate_checkpoints") != authorization.get("candidate_checkpoints"):
            raise ValueError("authorization future artifact bindings differ from experiment manifest")
        b0_config = yaml.safe_load(snapshots["b0_config"].decode("utf-8"))
        b0_payload = _load_torch_checkpoint(snapshots["b0_checkpoint"])
        normalization_payload = _json_bytes(
            snapshots["normalization_stats"], "normalization stats"
        )
        validate_legacy_checkpoint_binding(
            b0_payload, b0_config, split, normalization_payload
        )
        for index, seed in enumerate(seeds):
            payload = _load_torch_checkpoint(snapshots[f"candidate_checkpoint_{index}"])
            phases = candidate_config.get("train", {}).get("phases", [])
            expected_updates = phases[0].get("optimizer_updates") if len(phases) == 1 else None
            if payload.get("schema_version") != 4 or payload.get("runtime_seed") != seed or \
               payload.get("config_sha256") != canonical_config_hash or \
               payload.get("split_manifest_sha256") != authorization["split_manifest"]["sha256"] or \
               payload.get("normalization_stats_sha256") != authorization["normalization_stats"]["sha256"] or \
               payload.get("normalization_contract") != "ais_normalization_v2" or \
               payload.get("global_step") != expected_updates or \
               payload.get("phase_index") != 0 or payload.get("phase_update") != expected_updates:
                raise ValueError("authorization candidate checkpoint semantic binding mismatch")
        gate_payloads = [
            _json_bytes(snapshots[f"individual_gate_{index}"], f"individual gate {index}")
            for index in range(3)
        ]
        aggregate_payload = _json_bytes(snapshots["aggregate_gate"], "aggregate gate")
        if all("input_sha256" in gate for gate in gate_payloads):
            from authorize_ais_sealed_test import (
                _canonical_config_sha256, _validate_actual_gate_artifacts,
            )
            _validate_actual_gate_artifacts(
                Path(authorization["individual_gates"][0]["path"]).parents[1],
                gate_payloads, aggregate_payload, seeds,
                authorization["registered_validation_sample_ids"], canonical_config_hash,
                _canonical_config_sha256(Path(authorization["b0_config"]["path"])),
                authorization["split_manifest"]["sha256"],
                authorization["normalization_stats"]["sha256"],
                authorization["b0_checkpoint"]["sha256"],
                [item["sha256"] for item in authorization["candidate_checkpoints"]],
            )
            if sum(bool(gate.get("passed")) for gate in gate_payloads) < 2 or \
               aggregate_payload.get("passed") is not True:
                raise ValueError("authorization validation gates no longer pass")
    authorization = deepcopy(authorization)
    authorization["_bound_bytes"] = snapshots
    return authorization, _sha(authorization_bytes)


def _create_opened_marker(output_dir: Path, authorization_hash: str) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"sealed output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    marker = output_dir / "test_opened.json"
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = (json.dumps({"schema_version": 1, "authorization_sha256": authorization_hash,
                            "state": "opened_irreversible"}, sort_keys=True) + "\n").encode()
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return marker


def _load_torch_checkpoint(data: bytes) -> dict[str, Any]:
    import torch
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def validate_legacy_checkpoint_binding(
    payload: object, expected_config: dict[str, Any], expected_split: dict[str, Any],
    expected_normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the provenance fields actually written by train_pino.py."""
    if not isinstance(payload, dict):
        raise ValueError("legacy checkpoint must contain a mapping")
    if payload.get("full_config") != expected_config:
        raise ValueError("legacy checkpoint full config mismatch")
    if payload.get("model_config") != expected_config.get("model"):
        raise ValueError("legacy checkpoint model config mismatch")
    split = payload.get("split_manifest")
    if not isinstance(split, dict) or any(
        split.get(name) != expected_split.get(name) for name in ("train", "val", "test")
    ):
        raise ValueError("legacy checkpoint split manifest mismatch")
    if expected_normalization is not None and \
       payload.get("normalization_stats") != expected_normalization:
        raise ValueError("legacy checkpoint normalization metadata mismatch")
    if expected_normalization is not None and payload.get("global_step") != 12000:
        raise ValueError("legacy checkpoint must be complete at global_step 12000")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("legacy checkpoint model state is invalid")
    return state


def _load_bound_normalization(
    stats_bytes: bytes, authorized_path: Path,
):
    """Load the exact authorized bytes, retaining their frozen public path."""

    from dataclasses import replace
    from fno_acoustic.ais_normalization import load_ais_normalization

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(stats_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        binding = load_ais_normalization(temporary_path)
        return replace(binding, path=Path(authorized_path))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_candidate_predictor(
    config: dict[str, Any], normalization, state: dict[str, Any], device,
):
    """Construct one formal normalized AIS candidate from preflighted state."""

    from fno_acoustic.model_ais_mqfno import AISMQFNO
    from evaluate_ais_mqfno import NormalizedQueryPredictorAdapter
    from train_ais_mqfno import _model_kwargs

    model = AISMQFNO(**_model_kwargs(config, normalization)).to(device)
    model.load_state_dict(state, strict=True)
    predictor = NormalizedQueryPredictorAdapter(
        model,
        device,
        int(config["train"].get("query_chunk_size", 2048)),
        normalization,
        global_size=int(config["sampling"]["global_size"]),
    )
    return model, predictor


def build_sealed_census_row(
    scene: Any,
    native: Any,
    predictor_family: str,
    provenance: Mapping[str, str],
    receiver: Any,
) -> dict[str, object]:
    """Build one full-native sealed row using the shared census metric contract."""

    from fno_acoustic.long_horizon_metrics import long_horizon_metrics
    from fno_acoustic.query_census import (
        AIS_CATEGORIES,
        native_enumeration_sampler_diagnostics,
    )

    if native.field_cpu.shape != (1, *scene.target_cpu.shape) or \
       (native.coverage_min, native.coverage_max) != (1, 1):
        raise ValueError("sealed native coverage/shape failed")
    category = scene.metadata.get("model_type")
    if category not in AIS_CATEGORIES:
        raise ValueError("sealed scene category is invalid")
    metrics = long_horizon_metrics(
        native.field_cpu,
        scene.target_cpu[None].to(native.field_cpu.dtype),
        scene.time_s,
        receiver,
    )
    metrics.update(
        native_enumeration_sampler_diagnostics(
            native.field_cpu.shape[1], native.field_cpu.shape[2]
        )
    )
    return {
        "sample_id": scene.sample_id,
        "category": category,
        "split": "test",
        "seed": provenance["seed"],
        "predictor_family": predictor_family,
        **{
            key: provenance[key]
            for key in (
                "config_sha256",
                "checkpoint_sha256",
                "split_manifest_sha256",
                "normalization_sha256",
                "receiver_geometry_sha256",
            )
        },
        **metrics,
    }


def _default_evaluator(authorization: dict[str, Any], staging: Path, device_name: str) -> dict[str, Any]:
    """Evaluate B0 and three candidates serially using the Task9/Task10 contracts."""
    from dataclasses import asdict
    import torch
    import yaml
    from fno_acoustic.model_factorized import FactorizedAcousticFNO
    from fno_acoustic.query_census import (
        LegacyFullFieldPredictorAdapter,
        validate_provenance, write_census_summary, write_rows_atomically,
    )
    from fno_acoustic.query_data import DenseCPUQueryStore
    from evaluate_ais_mqfno import _receiver_manifest, validate_checkpoint_binding
    from train_ais_mqfno import _config_sha256

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    bound = authorization["_bound_bytes"]
    b0_config = yaml.safe_load(bound["b0_config"].decode("utf-8"))
    candidate_config = yaml.safe_load(bound["candidate_config"].decode("utf-8"))
    if not isinstance(b0_config, dict) or not isinstance(candidate_config, dict):
        raise ValueError("bound configs must be mappings")
    sample_ids = authorization["test_sample_ids"]
    split_payload = _json_bytes(bound["split_manifest"], "split manifest")
    store = DenseCPUQueryStore(Path(candidate_config["data"]["path"]))
    receiver = _receiver_manifest(candidate_config, store, sample_ids[0])
    split_hash = authorization["split_manifest"]["sha256"]

    stats = (
        _json_bytes(bound["normalization_stats"], "normalization stats")
        if "normalization_stats" in bound
        else json.loads(Path(candidate_config["normalization"]["stats_path"]).read_text(encoding="utf-8"))
    )
    if "normalization_stats" not in bound:
        raise ValueError("sealed AIS candidates require bound normalization stats")
    normalization = _load_bound_normalization(
        bound["normalization_stats"],
        Path(authorization["normalization_stats"]["path"]),
    )
    b0_hash = _config_sha256(b0_config)
    b0_payload = _load_torch_checkpoint(bound["b0_checkpoint"])
    b0_state = validate_legacy_checkpoint_binding(
        b0_payload, b0_config, split_payload, stats
    )
    b0_kwargs = {key: value for key, value in b0_config["model"].items() if key != "name"}
    b0_model = FactorizedAcousticFNO(**b0_kwargs)
    b0_model.load_state_dict(b0_state, strict=True)
    b0_predictor = LegacyFullFieldPredictorAdapter(b0_model, device, stats)
    common_provenance = {
        "split_manifest_sha256": split_hash,
        "normalization_sha256": normalization.stats_sha256,
        "receiver_geometry_sha256": receiver.sha256,
    }
    candidate_hash = _config_sha256(candidate_config)
    predictors = {"b0": b0_predictor}
    models = {"b0": b0_model}
    provenances = {
        "b0": {"seed": "b0", "config_sha256": b0_hash,
               "checkpoint_sha256": authorization["b0_checkpoint"]["sha256"],
               **common_provenance}
    }
    for index, seed in enumerate(authorization["confirmation_seeds"]):
        payload = _load_torch_checkpoint(bound[f"candidate_checkpoint_{index}"])
        state = validate_checkpoint_binding(
            payload,
            candidate_hash,
            split_hash,
            normalization.stats_sha256,
            normalization.contract_id,
            normalization.velocity_mean,
            normalization.velocity_std,
        )
        model, predictor = _build_candidate_predictor(
            candidate_config, normalization, state, device
        )
        name = f"seed{seed}"
        predictors[name] = predictor
        models[name] = model
        provenances[name] = {
            "seed": str(seed), "config_sha256": candidate_hash,
            "checkpoint_sha256": authorization["candidate_checkpoints"][index]["sha256"],
            **common_provenance,
        }
    bound_provenance = {
        name: validate_provenance(payload, receiver) for name, payload in provenances.items()
    }

    def row_evaluator(name: str):
        predictor, model, provenance = predictors[name], models[name], bound_provenance[name]

        def evaluate(scene):
            model.to(device)
            try:
                native = predictor.predict_scene(scene)
                return build_sealed_census_row(
                    scene, native, predictor.predictor_family, provenance, receiver
                )
            finally:
                model.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        return evaluate

    rows_by_name = evaluate_scene_major(
        sample_ids, store.read_scene,
        {name: row_evaluator(name) for name in predictors},
    )
    for name, rows in rows_by_name.items():
        census_dir = staging / name
        census_dir.mkdir()
        write_rows_atomically(census_dir / "samples.csv", rows)
        write_census_summary(
            census_dir / "summary.json", rows, "test", bound_provenance[name]
        )

    # Task10's CLI parser performs numeric conversion; import and use it to preserve the schema.
    from compare_native400_accuracy import read_native400_rows
    baseline_rows = read_native400_rows(staging / "b0/samples.csv")
    candidate_rows = [
        read_native400_rows(staging / f"seed{seed}/samples.csv")
        for seed in authorization["confirmation_seeds"]
    ]
    gates, aggregate_seed_results, aggregate, combined = compute_sealed_gate_bundle(
        baseline_rows, candidate_rows
    )
    gates_dir = staging / "gates"
    gates_dir.mkdir()
    baseline_csv = staging / "b0/samples.csv"
    baseline_csv_hash = _sha(baseline_csv.read_bytes())
    candidate_csv_hashes = [
        _sha((staging / f"seed{seed}/samples.csv").read_bytes())
        for seed in authorization["confirmation_seeds"]
    ]
    for seed, gate, candidate_csv_hash in zip(
        authorization["confirmation_seeds"], gates, candidate_csv_hashes, strict=True
    ):
        (gates_dir / f"seed{seed}.json").write_text(
            json.dumps({**asdict(gate), "threshold": 0.30,
                        "bootstrap_replicates": 10000, "seed": 20260714,
                        "input_sha256": {"baseline": baseline_csv_hash,
                                         "candidates": [candidate_csv_hash]}},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (gates_dir / "aggregate.json").write_text(
        json.dumps({"passed": combined.passed,
                    "passed_seed_count": combined.passed_seed_count,
                    "seed_results": [asdict(gate) for gate in aggregate_seed_results],
                    "aggregate": asdict(combined.aggregate),
                    "aggregation": "per_sample_median_then_paired_bootstrap",
                    "threshold": 0.30, "bootstrap_replicates": 10000,
                    "seed": 20260714,
                    "input_sha256": {"baseline": baseline_csv_hash,
                                     "candidates": candidate_csv_hashes}},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"passed": combined.passed, "passed_seed_count": combined.passed_seed_count,
            "test_sample_count": 250}


def run_sealed_test(
    authorization_path: Path, device: str, output_dir: Path, *, evaluator: Evaluator | None = None,
    fail_before_publish: bool = False,
) -> dict[str, Any]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    authorization, authorization_hash = _validate_authorization(Path(authorization_path))
    output_dir = Path(output_dir)
    compatibility_names = (
        "b0", *(f"seed{seed}" for seed in authorization["confirmation_seeds"]),
        "gates", "sealed_test_summary.json",
    )
    if (output_dir / "results").exists() or any(
        (output_dir / name).exists() or (output_dir / name).is_symlink()
        for name in compatibility_names
    ):
        raise FileExistsError("sealed results or compatibility links already exist")
    marker = _create_opened_marker(output_dir, authorization_hash)
    authorization["opened_marker"] = str(marker)
    staging = Path(tempfile.mkdtemp(prefix=".results.", suffix=".tmp", dir=output_dir))
    try:
        result = (evaluator or _default_evaluator)(authorization, staging, device)
        if not isinstance(result, dict) or result.get("test_sample_count") != 250:
            raise ValueError("sealed evaluator did not report the exact 250 test samples")
        summary = staging / "sealed_test_summary.json"
        with summary.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if fail_before_publish:
            raise RuntimeError("injected publish failure")
        results_dir = output_dir / "results"
        os.replace(staging, results_dir)
        directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        for name in compatibility_names:
            if not (results_dir / name).exists():
                continue
            temporary = output_dir / f".{name}.link.tmp"
            os.symlink(str(Path("results") / name), temporary)
            os.replace(temporary, output_dir / name)
        directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_sealed_test(args.authorization, args.device, args.output_dir)
    except (OSError, ValueError) as exc:
        build_parser().error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
