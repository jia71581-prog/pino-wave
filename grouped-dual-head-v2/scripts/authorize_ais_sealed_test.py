#!/usr/bin/env python3
"""Authorize the one-shot sealed test only after preregistered validation passes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml


THRESHOLD = 0.30
AGGREGATION = "per_sample_median_then_paired_bootstrap"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_snapshot(path: Path, name: str) -> tuple[dict[str, Any], bytes, str]:
    data = Path(path).read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload, data, _sha(data)


def _hash_binding(binding: object, name: str) -> dict[str, str]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"{name} must contain exactly path and sha256")
    path, expected = binding["path"], binding["sha256"]
    if not isinstance(path, str) or not path or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{name} has an invalid path or SHA-256")
    data = Path(path).read_bytes()
    if _sha(data) != expected.lower():
        raise ValueError(f"{name} hash mismatch")
    return {"path": path, "sha256": expected.lower()}


def _canonical_config_sha256(path: Path) -> str:
    data = Path(path).read_bytes()
    try:
        payload = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("candidate config is not valid UTF-8 YAML") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate config must contain a mapping")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha(canonical)


def _recipe_file_binding(
    recipe: dict[str, Any], recipe_path: Path, filename: str, name: str
) -> dict[str, str]:
    files = recipe.get("files")
    if not isinstance(files, dict) or filename not in files:
        raise ValueError(f"frozen recipe does not bind {filename}")
    return _hash_binding(
        {"path": str(recipe_path.parent / filename), "sha256": files[filename]}, name
    )


def _verify_recipe_inventory(recipe: dict[str, Any], recipe_path: Path) -> None:
    files = recipe.get("files")
    if files is None:
        return  # compatibility with the explicit pre-freeze unit fixture
    if not isinstance(files, dict) or not files:
        raise ValueError("frozen recipe file inventory is invalid")
    for filename, digest in files.items():
        _hash_binding({"path": str(recipe_path.parent / filename), "sha256": digest},
                      f"frozen recipe file {filename}")


def _load_checkpoint(path: str, name: str) -> dict[str, Any]:
    import torch
    try:
        payload = torch.load(io.BytesIO(Path(path).read_bytes()), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{name} is not a readable checkpoint") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a mapping")
    return payload


def _validate_completed_checkpoint_metadata(
    b0: dict[str, str], candidates: list[dict[str, str]], b0_config_path: Path,
    candidate_config_path: Path, candidate_config_hash: str,
    normalization_payload: dict[str, Any], normalization_hash: str,
    split_payload: dict[str, Any], split_hash: str, seeds: list[int],
) -> None:
    b0_config = yaml.safe_load(b0_config_path.read_text(encoding="utf-8"))
    b0_payload = _load_checkpoint(b0["path"], "B0 checkpoint")
    checkpoint_split = b0_payload.get("split_manifest")
    split_matches = isinstance(checkpoint_split, dict) and all(
        checkpoint_split.get(name) == split_payload.get(name)
        for name in ("train", "val", "test")
    )
    if b0_payload.get("full_config") != b0_config or \
       b0_payload.get("model_config") != b0_config.get("model") or not split_matches or \
       b0_payload.get("normalization_stats") != normalization_payload or \
       b0_payload.get("global_step") != 12000:
        raise ValueError("B0 checkpoint config or split provenance mismatch")
    if not isinstance(b0_payload.get("model_state_dict"), dict):
        raise ValueError("B0 checkpoint model state is invalid")
    candidate_config = yaml.safe_load(candidate_config_path.read_text(encoding="utf-8"))
    expected_updates = candidate_config["train"]["phases"][0]["optimizer_updates"]
    for index, (binding, seed) in enumerate(zip(candidates, seeds, strict=True)):
        payload = _load_checkpoint(binding["path"], f"candidate checkpoint {index}")
        if payload.get("schema_version") != 4 or payload.get("runtime_seed") != seed or \
           payload.get("config_sha256") != candidate_config_hash or \
           payload.get("split_manifest_sha256") != split_hash or \
           payload.get("normalization_stats_sha256") != normalization_hash or \
           payload.get("normalization_contract") != "ais_normalization_v2" or \
           payload.get("global_step") != expected_updates or \
           payload.get("phase_index") != 0 or payload.get("phase_update") != expected_updates or \
           not isinstance(payload.get("model_state_dict"), dict):
            raise ValueError(f"candidate checkpoint {index} config or split provenance mismatch")


def _ids(value: object, name: str, *, exact_count: int | None = None) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                                         for item in value):
        raise ValueError(f"{name} must be nonnegative integer IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} IDs must be unique")
    if exact_count is not None and len(value) != exact_count:
        raise ValueError(f"{name} requires exactly {exact_count} IDs")
    return list(value)


def _seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"authorization already exists: {path}")
        os.link(temporary, path)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _census_snapshot(path: Path) -> tuple[list[dict[str, str]], str]:
    data = path.read_bytes()
    rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
    if not rows:
        raise ValueError(f"validation census is empty: {path}")
    return rows, _sha(data)


def _validate_actual_gate_artifacts(
    gate_root: Path, gates: list[dict[str, Any]], aggregate: dict[str, Any],
    seeds: list[int], registered_ids: list[int], candidate_config_hash: str,
    b0_config_hash: str, split_hash: str, normalization_hash: str,
    b0_checkpoint_hash: str, candidate_checkpoint_hashes: list[str],
) -> None:
    experiment_root = gate_root.parent
    baseline_path = experiment_root / "baselines/b0/val/samples.csv"
    baseline_rows, baseline_hash = _census_snapshot(baseline_path)
    baseline_ids = [int(row["sample_id"]) for row in baseline_rows]
    if baseline_ids != registered_ids or len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError("baseline census sample IDs differ from registered validation IDs")
    if {row["config_sha256"] for row in baseline_rows} != {b0_config_hash} or \
       {row["checkpoint_sha256"] for row in baseline_rows} != {b0_checkpoint_hash} or \
       {row["split_manifest_sha256"] for row in baseline_rows} != {split_hash} or \
       {row["normalization_sha256"] for row in baseline_rows} != {normalization_hash}:
        raise ValueError("baseline census config/checkpoint/split provenance mismatch")
    candidate_hashes = []
    for index, (seed, gate) in enumerate(zip(seeds, gates, strict=True)):
        candidate_path = experiment_root / f"evaluations/400_final_seed{seed}_val/samples.csv"
        rows, candidate_hash = _census_snapshot(candidate_path)
        candidate_hashes.append(candidate_hash)
        ids = [int(row["sample_id"]) for row in rows]
        if ids != registered_ids or {row["seed"] for row in rows} != {str(seed)}:
            raise ValueError("candidate census sample IDs or runtime seed mismatch")
        if {row["config_sha256"] for row in rows} != {candidate_config_hash} or \
           {row["checkpoint_sha256"] for row in rows} != {candidate_checkpoint_hashes[index]} or \
           {row["split_manifest_sha256"] for row in rows} != {split_hash} or \
           {row["normalization_sha256"] for row in rows} != {normalization_hash}:
            raise ValueError("candidate census config/checkpoint/split provenance mismatch")
        if gate.get("threshold") != 0.30 or gate.get("bootstrap_replicates") != 10000 or \
           gate.get("seed") != 20260714 or gate.get("input_sha256") != {
               "baseline": baseline_hash, "candidates": [candidate_hash]}:
            raise ValueError("individual gate inputs or bootstrap contract mismatch")
    if aggregate.get("threshold") != 0.30 or aggregate.get("bootstrap_replicates") != 10000 or \
       aggregate.get("seed") != 20260714 or aggregate.get("aggregation") != AGGREGATION or \
       aggregate.get("input_sha256") != {"baseline": baseline_hash, "candidates": candidate_hashes}:
        raise ValueError("aggregate gate inputs or bootstrap contract mismatch")


def authorize_sealed_test(
    recipe_path: Path,
    experiment_manifest_path: Path,
    individual_gate_paths: Iterable[Path],
    aggregate_gate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"authorization already exists: {output_path}")
    if (output_path.parent / "test_opened.json").exists():
        raise FileExistsError("sealed test has already been opened")
    recipe, _, recipe_hash = _json_snapshot(recipe_path, "recipe")
    experiment, _, experiment_hash = _json_snapshot(experiment_manifest_path, "experiment manifest")
    _verify_recipe_inventory(recipe, Path(recipe_path))
    seeds = [_seed(value, "confirmation seed") for value in recipe.get("confirmation_seeds", [])]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("recipe requires exactly three independent confirmation seeds")
    if recipe.get("required_individual_seed_passes") != 2 or recipe.get("threshold") != THRESHOLD:
        raise ValueError("recipe must register the two-of-three rule at threshold 0.30")
    if recipe.get("seed_aggregation") != AGGREGATION or recipe.get("sealed_test_authorized") is not False:
        raise ValueError("recipe aggregation or sealed state is invalid")
    required_bindings: dict[str, dict[str, str]] = {}
    if "files" in recipe:
        required_bindings["b0_config"] = _recipe_file_binding(
            recipe, Path(recipe_path), "config_b0.yaml", "b0 config"
        )
        required_bindings["candidate_config"] = _recipe_file_binding(
            recipe, Path(recipe_path), "config_400.yaml", "candidate config"
        )
        required_bindings["normalization_stats"] = _recipe_file_binding(
            recipe, Path(recipe_path), "normalization_stats.json", "normalization stats"
        )
        split_binding = recipe.get("split_manifest")
        if not isinstance(split_binding, dict) or split_binding.get("file") != "split_manifest.json":
            raise ValueError("frozen recipe split manifest binding is invalid")
        required_bindings["split_manifest"] = _recipe_file_binding(
            recipe, Path(recipe_path), "split_manifest.json", "split manifest"
        )
        required_bindings["b0_checkpoint"] = _hash_binding(
            experiment.get("b0_checkpoint"), "b0 checkpoint"
        )
        checkpoints_raw = experiment.get("candidate_checkpoints")
    else:
        for name in ("split_manifest", "b0_config", "candidate_config", "b0_checkpoint"):
            required_bindings[name] = _hash_binding(recipe.get(name), name)
        if recipe.get("normalization_stats") is not None:
            required_bindings["normalization_stats"] = _hash_binding(
                recipe.get("normalization_stats"), "normalization stats"
            )
        checkpoints_raw = recipe.get("candidate_checkpoints")
    if not isinstance(checkpoints_raw, list) or len(checkpoints_raw) != 3:
        raise ValueError("recipe requires exactly three candidate checkpoints")
    checkpoints = [_hash_binding(item, f"candidate checkpoint {index}")
                   for index, item in enumerate(checkpoints_raw)]
    if len({item["sha256"] for item in checkpoints}) != 3:
        raise ValueError("candidate checkpoint hashes must be independent")
    candidate_config_canonical_hash = _canonical_config_sha256(
        Path(required_bindings["candidate_config"]["path"])
    )

    split, _, split_hash = _json_snapshot(Path(required_bindings["split_manifest"]["path"]), "split manifest")
    if split_hash != required_bindings["split_manifest"]["sha256"]:
        raise ValueError("split manifest hash mismatch")
    test_ids = _ids(split.get("test"), "test split", exact_count=250)
    validation_ids = _ids(split.get("val"), "validation split")
    if not validation_ids:
        raise ValueError("validation split must be nonempty")
    if "files" in recipe:
        _validate_completed_checkpoint_metadata(
            required_bindings["b0_checkpoint"], checkpoints,
            Path(required_bindings["b0_config"]["path"]),
            Path(required_bindings["candidate_config"]["path"]),
            candidate_config_canonical_hash,
            _json_snapshot(Path(required_bindings["normalization_stats"]["path"]),
                           "normalization stats")[0],
            required_bindings["normalization_stats"]["sha256"],
            split, split_hash, seeds,
        )

    if experiment.get("recipe_sha256") != recipe_hash or \
       experiment.get("split_manifest_sha256") != split_hash or \
       experiment.get("confirmation_seeds") != seeds:
        raise ValueError("experiment provenance does not match recipe")
    registered_ids = _ids(experiment.get("registered_validation_sample_ids"),
                          "registered validation samples")
    if registered_ids != validation_ids:
        raise ValueError("registered validation samples must equal the frozen validation split")

    gate_paths = [Path(path) for path in individual_gate_paths]
    if len(gate_paths) != 3:
        raise ValueError("authorization requires exactly three individual seed gates")
    gates = []
    gate_bindings = []
    for path in gate_paths:
        gate, _, gate_hash = _json_snapshot(path, "individual gate")
        gates.append(gate)
        gate_bindings.append({"path": str(path), "sha256": gate_hash})
    aggregate, _, aggregate_hash = _json_snapshot(aggregate_gate_path, "aggregate gate")
    actual_gate_schema = all("input_sha256" in gate and "kind" not in gate for gate in gates)
    if actual_gate_schema:
        _validate_actual_gate_artifacts(
            Path(gate_paths[0]).parents[1], gates, aggregate, seeds, registered_ids,
            candidate_config_canonical_hash,
            _canonical_config_sha256(Path(required_bindings["b0_config"]["path"])),
            split_hash, required_bindings["normalization_stats"]["sha256"],
            required_bindings["b0_checkpoint"]["sha256"],
            [item["sha256"] for item in checkpoints],
        )
        for seed, gate, checkpoint in zip(seeds, gates, checkpoints, strict=True):
            gate.update({"kind": "individual", "seed": seed, "aggregation": "paired_bootstrap",
                         "registered_validation_sample_ids": registered_ids,
                         "recipe_sha256": recipe_hash,
                         "config_sha256": candidate_config_canonical_hash,
                         "checkpoint_sha256": checkpoint["sha256"],
                         "split_manifest_sha256": split_hash})
        aggregate.update({"kind": "aggregate",
                          "registered_validation_sample_ids": registered_ids,
                          "recipe_sha256": recipe_hash,
                          "config_sha256": candidate_config_canonical_hash,
                          "checkpoint_sha256": hashlib.sha256("\n".join(sorted(
                              item["sha256"] for item in checkpoints)).encode()).hexdigest(),
                          "split_manifest_sha256": split_hash})
    gate_seeds = [_seed(gate.get("seed"), "gate seed") for gate in gates]
    if set(gate_seeds) != set(seeds) or len(set(gate_seeds)) != 3:
        raise ValueError("individual gates must cover three independent registered seeds")
    by_seed = {gate["seed"]: gate for gate in gates}
    checkpoint_by_seed = dict(zip(seeds, checkpoints, strict=True))
    common = {
        "threshold": THRESHOLD, "registered_validation_sample_ids": registered_ids,
        "recipe_sha256": recipe_hash, "config_sha256": candidate_config_canonical_hash,
        "split_manifest_sha256": split_hash,
    }
    for seed in seeds:
        gate = by_seed[seed]
        if gate.get("kind") != "individual" or gate.get("aggregation") != "paired_bootstrap":
            raise ValueError("invalid individual gate kind or aggregation")
        for key, expected in common.items():
            if gate.get(key) != expected:
                raise ValueError(f"individual gate provenance mismatch: {key}")
        if gate.get("checkpoint_sha256") != checkpoint_by_seed[seed]["sha256"]:
            raise ValueError("individual gate checkpoint hash mismatch")
        if not isinstance(gate.get("passed"), bool):
            raise ValueError("individual gate passed must be boolean")
    passed_seed_count = sum(bool(gate["passed"]) for gate in gates)
    if passed_seed_count < 2:
        raise ValueError("fewer than two individual validation seeds passed")

    aggregate_checkpoint_hash = hashlib.sha256(
        "\n".join(sorted(item["sha256"] for item in checkpoints)).encode()
    ).hexdigest()
    aggregate_expected = {
        **common, "kind": "aggregate", "passed": True,
        "passed_seed_count": passed_seed_count,
        "aggregation": AGGREGATION, "checkpoint_sha256": aggregate_checkpoint_hash,
    }
    for key, expected in aggregate_expected.items():
        if aggregate.get(key) != expected:
            raise ValueError(f"aggregate validation gate mismatch: {key}")

    authorization: dict[str, Any] = {
        "schema_version": 1, "recipe": {"path": str(recipe_path), "sha256": recipe_hash},
        "experiment_manifest": {"path": str(experiment_manifest_path), "sha256": experiment_hash},
        **required_bindings, "candidate_checkpoints": checkpoints,
        "individual_gates": gate_bindings,
        "aggregate_gate": {"path": str(aggregate_gate_path), "sha256": aggregate_hash},
        "confirmation_seeds": seeds, "threshold": THRESHOLD,
        "aggregation": AGGREGATION, "required_individual_seed_passes": 2,
        "registered_validation_sample_ids": registered_ids,
        "candidate_config_canonical_sha256": candidate_config_canonical_hash,
        "test_sample_ids": test_ids, "test_sample_count": 250,
        "split_manifest_bytes_sha256": split_hash,
    }
    _atomic_json(output_path, authorization)
    return authorization


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--validation-gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe, _, _ = _json_snapshot(args.recipe, "recipe")
    seeds = recipe.get("confirmation_seeds", [])
    individual = [args.validation_gates / f"confirmation_val_seed{seed}/gate.json"
                  for seed in seeds]
    aggregate = args.validation_gates / "confirmation_val_aggregate/gate.json"
    if not all(path.is_file() for path in [*individual, aggregate]):
        individual = sorted(args.validation_gates.glob("individual*.json"))
        aggregate = args.validation_gates / "aggregate.json"
    try:
        authorize_sealed_test(args.recipe, args.experiment_manifest, individual, aggregate, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
