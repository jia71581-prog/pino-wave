#!/usr/bin/env python
"""Static audit for the preregistered parameter-matched Patch-DeepONet."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from patch_deeponet_baseline.model import PatchDeepONet


FORBIDDEN_TOKENS = (
    "read_wavefield",
    "dense_target_physical",
    "target_pressure",
    "parent_checkpoint",
    "proposed_prediction",
)

PREDICTOR_FORBIDDEN_CALLS = {
    "dense_target_physical",
    "generate_references",
    "read_wavefield",
    "reconstruct_locked_fine_velocities",
    "score",
    "_reference_solver",
}


def called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    return names


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(protocol_path: Path) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf8"))
    model_path = Path(protocol["model"]["source_file"])
    feature_path = Path(protocol["model"]["feature_file"])
    predictor_path = Path(protocol["model"]["prediction_file"])
    training_primitives_path = Path(protocol["model"]["training_primitives_file"])
    training_entry_path = Path(protocol["model"]["training_entry_file"])
    training_config_path = Path(protocol["model"]["training_config"])
    for path in (model_path, feature_path, training_primitives_path, training_entry_path):
        ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    predictor_tree = ast.parse(
        predictor_path.read_text(encoding="utf8"), filename=str(predictor_path)
    )
    forbidden_predictor_calls = sorted(called_names(predictor_tree) & PREDICTOR_FORBIDDEN_CALLS)
    combined = model_path.read_text(encoding="utf8") + feature_path.read_text(encoding="utf8")
    forbidden_found = [token for token in FORBIDDEN_TOKENS if token in combined]
    model = PatchDeepONet()
    match = model.parameter_match()
    expected = int(protocol["model"]["resolved_parameters"])
    if int(match["parameter_count"]) != expected or not bool(match["within_tolerance"]):
        raise ValueError("Patch-DeepONet parameter contract changed")
    position = protocol["position_generalization"]
    if (
        float(position["fixed_source_frequency_hz"]) != 19.0
        or bool(position["source_frequency_generalization_in_scope"])
        or bool(position["frequency_sweep_permitted"])
        or int(position["record_count"]) != 240
        or int(position["independent_velocity_slice_count"]) != 30
    ):
        raise ValueError("Patch-DeepONet position scope changed")
    if forbidden_found:
        raise ValueError("forbidden baseline input token found: " + ", ".join(forbidden_found))
    if forbidden_predictor_calls:
        raise ValueError(
            "forbidden baseline predictor call found: " + ", ".join(forbidden_predictor_calls)
        )
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf8"))
    if (
        training_config.get("launch_authorized") is not False
        or training_config["schedule"]["split"] != "train"
        or training_config["selection"]["split"] != "train"
        or training_config["claim_scope"]["frequency_generalization_permitted"] is not False
    ):
        raise ValueError("Patch-DeepONet training launch/split/scope contract changed")
    return {
        "schema": "patch_deeponet_static_audit_v1",
        "status": "pass",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "model_file": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "feature_file": str(feature_path.resolve()),
        "feature_sha256": sha256(feature_path),
        "prediction_file": str(predictor_path.resolve()),
        "prediction_sha256": sha256(predictor_path),
        "training_primitives_file": str(training_primitives_path.resolve()),
        "training_primitives_sha256": sha256(training_primitives_path),
        "training_entry_file": str(training_entry_path.resolve()),
        "training_entry_sha256": sha256(training_entry_path),
        "training_config": str(training_config_path.resolve()),
        "training_config_sha256": sha256(training_config_path),
        "full_training_launch_authorized": False,
        "forbidden_input_tokens_found": forbidden_found,
        "forbidden_predictor_calls_found": forbidden_predictor_calls,
        "parameter_match": match,
        "fixed_source_frequency_hz": 19.0,
        "source_frequency_generalization_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="paper/tgrs_helmholtz_operator/patch_deeponet_confirmatory_protocol_20260813.json",
    )
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.protocol)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
