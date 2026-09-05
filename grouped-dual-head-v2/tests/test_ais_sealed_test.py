from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

import scripts.run_ais_sealed_test as sealed_runner
from scripts.authorize_ais_sealed_test import authorize_sealed_test
from scripts.freeze_ais_recipe import freeze_recipe
from scripts.generate_ais_configs import build_configs
from scripts.run_ais_sealed_test import (
    compute_sealed_gate_bundle, evaluate_scene_major, run_sealed_test,
    validate_legacy_checkpoint_binding,
)
from fno_acoustic.long_horizon_metrics import ReceiverSiteManifest
from fno_acoustic.query_census import NativePrediction, SAMPLE_COLUMNS, write_rows_atomically
from fno_acoustic.query_data import QueryScene
from fno_acoustic.native400_gate import evaluate_native400_gate
from fno_acoustic.model_ais_mqfno import AISMQFNO
from evaluate_ais_mqfno import NormalizedQueryPredictorAdapter
from scripts.train_ais_mqfno import _model_kwargs


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_ais_sealed_test.py"


def test_sealed_production_row_satisfies_atomic_census_contract(tmp_path: Path) -> None:
    target = torch.linspace(-1.0, 1.0, 160)[None, None].expand(3, 3, 160)
    scene = QueryScene(
        7,
        target,
        torch.full((3, 3), 3000.0),
        torch.nn.functional.pad(torch.ones(1, 1), (1, 1, 1, 1)),
        torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        {"model_type": "uniform"},
    )
    receiver = ReceiverSiteManifest(
        (3, 3),
        torch.tensor([0, 4, 8]),
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        "5" * 64,
    )
    provenance = {
        "seed": "17",
        "config_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "normalization_sha256": "4" * 64,
        "receiver_geometry_sha256": receiver.sha256,
    }
    native = NativePrediction(target[None].clone(), 1, 1)

    row = sealed_runner.build_sealed_census_row(
        scene, native, "ais_mqfno", provenance, receiver
    )
    output = write_rows_atomically(tmp_path / "samples.csv", [row])

    assert output.is_file()
    assert row["sampler_ess"] == 9.0
    assert row["sampler_duplicate_fraction"] == 0.0
    assert row["sampler_coverage"] == 1.0
    assert row["sampler_max_median_inverse_weight"] == 1.0


@pytest.mark.parametrize("dispersion_head", ["none", "phase_residual_24"])
def test_sealed_candidate_uses_bound_normalization_and_physical_adapter(
    tmp_path: Path, dispersion_head: str
) -> None:
    stats_path = tmp_path / "normalization_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "computed_from_split": "train",
                "velocity": {"mean": 3000.0, "std": 500.0},
                "wavefield": {"mean": 7.0, "std": 2.0},
                "eps": 1.0e-6,
            }
        )
    )
    config = {
        "normalization": {
            "contract": "ais_normalization_v2",
            "stats_path": str(stats_path),
        },
        "sampling": {"global_size": 3},
        "train": {"query_chunk_size": 4},
        "model": {
            "name": "ais_mqfno",
            "global_in_features": 6,
            "native_in_channels": 5,
            "spatial_width": 2,
            "spatial_modes": 1,
            "spatial_layers": 1,
            "temporal_modes": 1,
            "local_dim": 24,
            "fusion_dim": 48,
            "halo_size": 25,
            "local_encoder_kind": "multiscale_9_25",
            "dispersion_head": dispersion_head,
        },
    }
    from fno_acoustic.ais_normalization import load_ais_normalization

    reference_binding = load_ais_normalization(stats_path)
    reference = AISMQFNO(**_model_kwargs(config, reference_binding))
    for parameter in reference.parameters():
        parameter.data.zero_()

    binding = sealed_runner._load_bound_normalization(
        stats_path.read_bytes(),
        stats_path,
    )
    model, predictor = sealed_runner._build_candidate_predictor(
        config,
        binding,
        reference.state_dict(),
        torch.device("cpu"),
    )

    assert isinstance(model, AISMQFNO)
    assert isinstance(predictor, NormalizedQueryPredictorAdapter)
    assert predictor.normalization is binding
    assert binding.stats_sha256 == _sha(stats_path)
    if dispersion_head == "phase_residual_24":
        assert model.dispersion_residual_head is not None
        assert model.dispersion_residual_head.velocity_mean == 3000.0
    scene = QueryScene(
        sample_id=1,
        target_cpu=torch.zeros(3, 3, 160),
        velocity_cpu=torch.full((3, 3), 3000.0),
        source_cpu=torch.nn.functional.one_hot(
            torch.tensor(4), num_classes=9
        ).reshape(3, 3).float(),
        time_s=torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        x_m=torch.arange(3, dtype=torch.float64) * 30.0,
        z_m=torch.arange(3, dtype=torch.float64) * 40.0,
        metadata={"model_type": "uniform"},
    )
    prediction = predictor.predict_scene(scene)
    assert prediction.field_cpu.shape == (1, 3, 3, 160)
    assert torch.equal(prediction.field_cpu, torch.full_like(prediction.field_cpu, 7.0))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: bytes) -> dict[str, str]:
    path.write_bytes(data)
    return {"path": str(path), "sha256": _sha(path)}


def _canonical_config_sha(path: Path) -> str:
    payload = yaml.safe_load(path.read_text())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, list[Path], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": [300], "val": [301, 302], "test": list(range(250))}))
    config_b0 = _write(tmp_path / "b0.yaml", b'{"model":{"name":"factorized_fno"}}')
    config = _write(tmp_path / "candidate.yaml", b'{ "model": {"name": "ais_mqfno"} }\n')
    canonical_config_hash = _canonical_config_sha(Path(config["path"]))
    b0_checkpoint = _write(tmp_path / "b0.pt", b"b0-checkpoint")
    candidate_checkpoints = [_write(tmp_path / f"seed{i}.pt", f"seed-{i}".encode()) for i in range(3)]
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "confirmation_seeds": [20260714, 20260715, 20260716],
        "seed_aggregation": "per_sample_median_then_paired_bootstrap",
        "required_individual_seed_passes": 2,
        "threshold": 0.30,
        "sealed_test_authorized": False,
        "split_manifest": {"path": str(split), "sha256": _sha(split)},
        "b0_config": config_b0, "candidate_config": config,
        "b0_checkpoint": b0_checkpoint,
        "candidate_checkpoints": candidate_checkpoints,
    }))
    validation_ids = [301, 302]
    gates = []
    for index, passed in enumerate((True, True, False)):
        path = tmp_path / f"gate{index}.json"
        path.write_text(json.dumps({
            "kind": "individual", "seed": 20260714 + index, "passed": passed,
            "threshold": 0.30, "registered_validation_sample_ids": validation_ids,
            "aggregation": "paired_bootstrap", "recipe_sha256": _sha(recipe),
            "config_sha256": canonical_config_hash,
            "checkpoint_sha256": candidate_checkpoints[index]["sha256"],
            "split_manifest_sha256": _sha(split),
        }))
        gates.append(path)
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps({
        "kind": "aggregate", "passed": True, "passed_seed_count": 2,
        "threshold": 0.30, "registered_validation_sample_ids": validation_ids,
        "aggregation": "per_sample_median_then_paired_bootstrap",
        "recipe_sha256": _sha(recipe), "config_sha256": canonical_config_hash,
        "checkpoint_sha256": hashlib.sha256("\n".join(sorted(
            item["sha256"] for item in candidate_checkpoints)).encode()).hexdigest(),
        "split_manifest_sha256": _sha(split),
    }))
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({
        "recipe_sha256": _sha(recipe), "split_manifest_sha256": _sha(split),
        "registered_validation_sample_ids": validation_ids,
        "confirmation_seeds": [20260714, 20260715, 20260716],
    }))
    return recipe, experiment, gates, aggregate


def test_authorization_requires_three_independent_seeds_two_passes_and_aggregate(tmp_path: Path) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    output = tmp_path / "authorization.json"
    auth = authorize_sealed_test(recipe, experiment, gates, aggregate, output)
    assert len(auth["test_sample_ids"]) == len(set(auth["test_sample_ids"])) == 250
    assert auth["threshold"] == 0.30
    assert auth["candidate_config_canonical_sha256"] == _canonical_config_sha(
        Path(auth["candidate_config"]["path"])
    )
    assert output.is_file()
    with pytest.raises(FileExistsError):
        authorize_sealed_test(recipe, experiment, gates, aggregate, output)


def test_authorization_accepts_all_three_individual_seed_passes(tmp_path: Path) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    for gate_path in gates:
        payload = json.loads(gate_path.read_text())
        payload["passed"] = True
        gate_path.write_text(json.dumps(payload))
    payload = json.loads(aggregate.read_text())
    payload["passed_seed_count"] = 3
    aggregate.write_text(json.dumps(payload))
    assert authorize_sealed_test(
        recipe, experiment, gates, aggregate, tmp_path / "authorization.json"
    )["required_individual_seed_passes"] == 2


@pytest.mark.parametrize("mutation", ["one_seed", "aggregate_fail", "checkpoint_hash", "split_249"])
def test_authorization_rejects_invalid_gate_or_bound_file(tmp_path: Path, mutation: str) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    if mutation == "one_seed":
        gates = gates[:1]
    elif mutation == "aggregate_fail":
        payload = json.loads(aggregate.read_text())
        payload["passed"] = False
        aggregate.write_text(json.dumps(payload))
    elif mutation == "checkpoint_hash":
        Path(json.loads(recipe.read_text())["candidate_checkpoints"][0]["path"]).write_bytes(b"tamper")
    else:
        split = Path(json.loads(recipe.read_text())["split_manifest"]["path"])
        payload = json.loads(split.read_text())
        payload["test"] = payload["test"][:-1]
        split.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        authorize_sealed_test(recipe, experiment, gates, aggregate, tmp_path / "authorization.json")


def test_runner_parser_has_only_authorization_device_output_dir() -> None:
    help_result = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd="/", text=True, capture_output=True)
    assert help_result.returncode == 0
    assert "--authorization" in help_result.stdout and "--device" in help_result.stdout
    assert "--output-dir" in help_result.stdout and "--max-samples" not in help_result.stdout


def test_runner_rehashes_then_opens_once_before_label_reader(tmp_path: Path) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    authorization_path = tmp_path / "authorization.json"
    authorize_sealed_test(recipe, experiment, gates, aggregate, authorization_path)
    calls: list[tuple[str, bool]] = []

    def evaluator(authorization: dict, staging: Path, device: str) -> dict:
        opened = Path(authorization["opened_marker"])
        calls.append((device, opened.exists()))
        (staging / "b0").mkdir()
        for seed in authorization["confirmation_seeds"]:
            (staging / f"seed{seed}").mkdir()
        (staging / "gates").mkdir()
        return {"passed": True, "test_sample_count": 250}

    output = tmp_path / "test-output"
    result = run_sealed_test(authorization_path, "cpu", output, evaluator=evaluator)
    assert result["passed"] is True and calls == [("cpu", True)]
    assert (output / "test_opened.json").is_file()
    assert (output / "results/sealed_test_summary.json").is_file()
    for name in ("b0", *(f"seed{seed}" for seed in (20260714, 20260715, 20260716)),
                 "gates", "sealed_test_summary.json"):
        assert (output / name).is_symlink()
        assert (output / name).resolve() == (output / "results" / name).resolve()
    with pytest.raises((FileExistsError, ValueError)):
        run_sealed_test(authorization_path, "cpu", output, evaluator=evaluator)


def test_failure_after_opening_is_permanently_sealed_and_publishes_no_results(tmp_path: Path) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    authorization = tmp_path / "authorization.json"
    authorize_sealed_test(recipe, experiment, gates, aggregate, authorization)

    def fail(_authorization: dict, _staging: Path, _device: str) -> dict:
        raise RuntimeError("injected evaluation failure")

    output = tmp_path / "failed-test"
    with pytest.raises(RuntimeError, match="injected"):
        run_sealed_test(authorization, "cpu", output, evaluator=fail)
    assert (output / "test_opened.json").is_file()
    assert not (output / "results").exists()
    with pytest.raises((FileExistsError, ValueError)):
        run_sealed_test(authorization, "cpu", output, evaluator=fail)


def test_publish_failure_leaves_no_results_or_compatibility_links(tmp_path: Path) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    authorization = tmp_path / "authorization.json"
    authorize_sealed_test(recipe, experiment, gates, aggregate, authorization)

    def evaluator(auth: dict, staging: Path, _device: str) -> dict:
        (staging / "b0").mkdir()
        for seed in auth["confirmation_seeds"]:
            (staging / f"seed{seed}").mkdir()
        (staging / "gates").mkdir()
        return {"passed": True, "test_sample_count": 250}

    output = tmp_path / "publish-failure"
    with pytest.raises(RuntimeError, match="publish"):
        run_sealed_test(
            authorization, "cpu", output, evaluator=evaluator, fail_before_publish=True
        )
    assert not (output / "results").exists()
    assert not any((output / name).exists() or (output / name).is_symlink() for name in (
        "b0", "seed20260714", "seed20260715", "seed20260716", "gates",
        "sealed_test_summary.json",
    ))
    assert (output / "test_opened.json").is_file()
    with pytest.raises((FileExistsError, ValueError)):
        run_sealed_test(authorization, "cpu", output, evaluator=evaluator)


def test_runner_refuses_hash_mismatch_before_opening_labels(tmp_path: Path) -> None:
    recipe, experiment, gates, aggregate = _inputs(tmp_path)
    authorization = tmp_path / "authorization.json"
    payload = authorize_sealed_test(recipe, experiment, gates, aggregate, authorization)
    Path(payload["candidate_checkpoints"][1]["path"]).write_bytes(b"changed")
    output = tmp_path / "never-opened"
    with pytest.raises(ValueError, match="hash"):
        run_sealed_test(authorization, "cpu", output, evaluator=lambda *_: pytest.fail("must not evaluate"))
    assert not (output / "test_opened.json").exists()


def _frozen_completed_validation(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    split = tmp_path / "splits.json"
    validation_ids = list(range(301, 307))
    split.write_text(json.dumps({"train": [300], "val": validation_ids,
                                 "test": list(range(250))}))
    selected = build_configs()["ais_mqfno_64x160_ai_energy.yaml"]
    selected["model"] = {
        "name": "ais_mqfno",
        "global_max_size": 400,
        "global_in_features": 6,
        "native_in_channels": 5,
        "spatial_width": 2,
        "spatial_modes": 1,
        "spatial_layers": 1,
        "temporal_modes": 1,
        "local_dim": 2,
        "fusion_dim": 2,
        "halo_size": 1,
        "dispersion_head": "phase_residual_24",
    }
    selected["data"]["split_manifest"] = str(split)
    selected_path = tmp_path / "selected.yaml"
    selected_path.write_text(yaml.safe_dump(selected, sort_keys=True))
    selected_checkpoint = tmp_path / "selected64.pt"
    selected_checkpoint.write_bytes(b"selected-64")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "selected_config": str(selected_path),
        "selected_checkpoint": str(selected_checkpoint),
        "selected_config_sha256": _sha(selected_path),
        "selected_checkpoint_sha256": _sha(selected_checkpoint),
        "diagnostic_only": False,
    }))
    frozen = freeze_recipe(selection, tmp_path / "frozen")
    recipe_hash = _sha(frozen.manifest_path)
    frozen_b0 = yaml.safe_load(frozen.config_b0.read_text())
    canonical = _canonical_config_sha(frozen.config_400)
    split_payload = json.loads((frozen.root / "split_manifest.json").read_text())
    b0_path = tmp_path / "b0-final.pt"
    torch.save({"model_state_dict": {"weight": torch.ones(1)},
                "full_config": frozen_b0, "model_config": frozen_b0["model"],
                "split_manifest": split_payload,
                "normalization_stats": json.loads(
                    (frozen.root / "normalization_stats.json").read_text()
                ),
                "global_step": 12000}, b0_path)
    b0_checkpoint = {"path": str(b0_path), "sha256": _sha(b0_path)}
    candidates = []
    for seed in frozen.manifest["confirmation_seeds"]:
        path = tmp_path / f"candidate-{seed}.pt"
        torch.save({"schema_version": 4, "config_sha256": canonical,
                    "split_manifest_sha256": frozen.manifest["split_manifest"]["sha256"],
                    "normalization_stats_sha256": frozen.manifest["normalization_stats"]["sha256"],
                    "normalization_contract": "ais_normalization_v2",
                    "model_state_dict": {"weight": torch.ones(1)},
                    "runtime_seed": seed, "global_step": 6000,
                    "phase_index": 0, "phase_update": 6000}, path)
        candidates.append({"path": str(path), "sha256": _sha(path)})
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({
        "recipe_sha256": recipe_hash,
        "split_manifest_sha256": frozen.manifest["split_manifest"]["sha256"],
        "registered_validation_sample_ids": validation_ids,
        "confirmation_seeds": frozen.manifest["confirmation_seeds"],
        "b0_checkpoint": b0_checkpoint,
        "candidate_checkpoints": candidates,
    }))
    candidate_config = frozen.config_400
    canonical = _canonical_config_sha(candidate_config)
    gates = []
    for index, seed in enumerate(frozen.manifest["confirmation_seeds"]):
        gate = tmp_path / f"individual{index}.json"
        gate.write_text(json.dumps({
            "kind": "individual", "seed": seed, "passed": index < 2,
            "threshold": 0.30, "registered_validation_sample_ids": validation_ids,
            "aggregation": "paired_bootstrap", "recipe_sha256": recipe_hash,
            "config_sha256": canonical, "checkpoint_sha256": candidates[index]["sha256"],
            "split_manifest_sha256": frozen.manifest["split_manifest"]["sha256"],
        }))
        gates.append(gate)
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps({
        "kind": "aggregate", "passed": True, "passed_seed_count": 2,
        "threshold": 0.30, "registered_validation_sample_ids": validation_ids,
        "aggregation": "per_sample_median_then_paired_bootstrap",
        "recipe_sha256": recipe_hash, "config_sha256": canonical,
        "checkpoint_sha256": hashlib.sha256("\n".join(sorted(
            item["sha256"] for item in candidates)).encode()).hexdigest(),
        "split_manifest_sha256": frozen.manifest["split_manifest"]["sha256"],
    }))
    return frozen, experiment, gates, aggregate


def test_actual_frozen_recipe_bridges_future_completed_validation_artifacts(tmp_path: Path) -> None:
    frozen, experiment, gates, aggregate = _frozen_completed_validation(tmp_path)
    authorization = authorize_sealed_test(
        frozen.manifest_path, experiment, gates, aggregate, tmp_path / "authorization.json"
    )
    assert authorization["candidate_config"]["sha256"] == frozen.manifest["files"]["config_400.yaml"]
    assert authorization["b0_config"]["sha256"] == frozen.manifest["files"]["config_b0.yaml"]
    assert len(authorization["candidate_checkpoints"]) == 3


def test_schema4_authorization_runs_real_normalized_dispersion_candidate(
    tmp_path: Path,
) -> None:
    frozen, experiment, gates, aggregate = _frozen_completed_validation(tmp_path)
    config = yaml.safe_load(frozen.config_400.read_text())
    binding = sealed_runner._load_bound_normalization(
        (frozen.root / "normalization_stats.json").read_bytes(),
        frozen.root / "normalization_stats.json",
    )
    reference = AISMQFNO(**_model_kwargs(config, binding))
    for parameter in reference.parameters():
        parameter.data.zero_()
    state = reference.state_dict()

    experiment_payload = json.loads(experiment.read_text())
    candidate_bindings = []
    for checkpoint_binding, gate_path in zip(
        experiment_payload["candidate_checkpoints"], gates, strict=True
    ):
        checkpoint = Path(checkpoint_binding["path"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["model_state_dict"] = state
        torch.save(payload, checkpoint)
        updated = {"path": str(checkpoint), "sha256": _sha(checkpoint)}
        candidate_bindings.append(updated)
        gate = json.loads(gate_path.read_text())
        gate["checkpoint_sha256"] = updated["sha256"]
        gate_path.write_text(json.dumps(gate))
    experiment_payload["candidate_checkpoints"] = candidate_bindings
    experiment.write_text(json.dumps(experiment_payload))
    aggregate_payload = json.loads(aggregate.read_text())
    aggregate_payload["checkpoint_sha256"] = hashlib.sha256(
        "\n".join(sorted(item["sha256"] for item in candidate_bindings)).encode()
    ).hexdigest()
    aggregate.write_text(json.dumps(aggregate_payload))
    authorization_path = tmp_path / "authorization.json"
    authorize_sealed_test(
        frozen.manifest_path, experiment, gates, aggregate, authorization_path
    )

    scene = QueryScene(
        1,
        torch.zeros(3, 3, 160),
        torch.full((3, 3), binding.velocity_mean),
        torch.nn.functional.one_hot(torch.tensor(4), 9).reshape(3, 3).float(),
        torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        torch.arange(3, dtype=torch.float64) * 30.0,
        torch.arange(3, dtype=torch.float64) * 40.0,
        {"model_type": "uniform"},
    )

    def evaluator(authorization: dict, staging: Path, _device: str) -> dict:
        from evaluate_ais_mqfno import validate_checkpoint_binding

        bound = authorization["_bound_bytes"]
        loaded_config = yaml.safe_load(bound["candidate_config"].decode())
        loaded_binding = sealed_runner._load_bound_normalization(
            bound["normalization_stats"],
            Path(authorization["normalization_stats"]["path"]),
        )
        for index, seed in enumerate(authorization["confirmation_seeds"]):
            payload = sealed_runner._load_torch_checkpoint(
                bound[f"candidate_checkpoint_{index}"]
            )
            loaded_state = validate_checkpoint_binding(
                payload,
                authorization["candidate_config_canonical_sha256"],
                authorization["split_manifest"]["sha256"],
                loaded_binding.stats_sha256,
                loaded_binding.contract_id,
            )
            _, predictor = sealed_runner._build_candidate_predictor(
                loaded_config, loaded_binding, loaded_state, torch.device("cpu")
            )
            prediction = predictor.predict_scene(scene)
            assert prediction.field_cpu.shape == (1, 3, 3, 160)
            assert torch.isfinite(prediction.field_cpu).all()
            (staging / f"seed{seed}").mkdir()
        (staging / "b0").mkdir()
        (staging / "gates").mkdir()
        return {"passed": True, "test_sample_count": 250}

    output = tmp_path / "sealed-output"
    result = run_sealed_test(
        authorization_path, "cpu", output, evaluator=evaluator
    )
    assert result["passed"] is True
    assert (output / "test_opened.json").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 3),
        ("normalization_stats_sha256", "0" * 64),
        ("normalization_contract", "wrong_contract"),
    ],
)
def test_authorizer_rejects_candidate_without_exact_normalization_binding(
    tmp_path: Path, field: str, value: object
) -> None:
    frozen, experiment, gates, aggregate = _frozen_completed_validation(tmp_path)
    experiment_payload = json.loads(experiment.read_text())
    checkpoint = Path(experiment_payload["candidate_checkpoints"][0]["path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, checkpoint)
    experiment_payload["candidate_checkpoints"][0] = {
        "path": str(checkpoint),
        "sha256": _sha(checkpoint),
    }
    experiment.write_text(json.dumps(experiment_payload))

    with pytest.raises(ValueError, match="candidate|normalization|schema"):
        authorize_sealed_test(
            frozen.manifest_path,
            experiment,
            gates,
            aggregate,
            tmp_path / "must-not-authorize.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 3),
        ("normalization_stats_sha256", "0" * 64),
        ("normalization_contract", "wrong_contract"),
    ],
)
def test_runner_rejects_candidate_semantics_before_opening_marker(
    tmp_path: Path, field: str, value: object
) -> None:
    frozen, experiment, gates, aggregate = _frozen_completed_validation(tmp_path)
    authorization_path = tmp_path / "authorization.json"
    authorization = authorize_sealed_test(
        frozen.manifest_path, experiment, gates, aggregate, authorization_path
    )
    checkpoint = Path(authorization["candidate_checkpoints"][0]["path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, checkpoint)
    updated_binding = {"path": str(checkpoint), "sha256": _sha(checkpoint)}
    experiment_payload = json.loads(experiment.read_text())
    experiment_payload["candidate_checkpoints"][0] = updated_binding
    experiment.write_text(json.dumps(experiment_payload))
    authorization["candidate_checkpoints"][0] = updated_binding
    authorization["experiment_manifest"]["sha256"] = _sha(experiment)
    authorization_path.write_text(json.dumps(authorization))

    output = tmp_path / "must-remain-unopened"
    with pytest.raises(ValueError, match="candidate|normalization|schema"):
        run_sealed_test(
            authorization_path,
            "cpu",
            output,
            evaluator=lambda *_: pytest.fail("semantic preflight must reject"),
        )
    assert not (output / "test_opened.json").exists()


def test_frozen_bridge_rejects_config_and_future_checkpoint_tampering(tmp_path: Path) -> None:
    frozen, experiment, gates, aggregate = _frozen_completed_validation(tmp_path)
    frozen.config_400.write_text(frozen.config_400.read_text() + "# tamper\n")
    with pytest.raises(ValueError, match="hash"):
        authorize_sealed_test(
            frozen.manifest_path, experiment, gates, aggregate, tmp_path / "bad-config.json"
        )

    other = tmp_path / "checkpoint-case"
    frozen, experiment, gates, aggregate = _frozen_completed_validation(other)
    checkpoint = Path(json.loads(experiment.read_text())["candidate_checkpoints"][0]["path"])
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        authorize_sealed_test(
            frozen.manifest_path, experiment, gates, aggregate, other / "bad-checkpoint.json"
        )


def test_authorizer_rejects_b0_normalization_semantic_tampering(tmp_path: Path) -> None:
    frozen, experiment, gates, aggregate = _frozen_completed_validation(tmp_path)
    experiment_payload = json.loads(experiment.read_text())
    checkpoint = Path(experiment_payload["b0_checkpoint"]["path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["normalization_stats"] = {**payload["normalization_stats"], "tampered": True}
    torch.save(payload, checkpoint)
    experiment_payload["b0_checkpoint"]["sha256"] = _sha(checkpoint)
    experiment.write_text(json.dumps(experiment_payload))
    with pytest.raises(ValueError, match="B0 checkpoint"):
        authorize_sealed_test(
            frozen.manifest_path, experiment, gates, aggregate,
            tmp_path / "bad-normalization.json",
        )


def test_scene_major_helper_evaluates_all_models_before_next_scene() -> None:
    order: list[tuple[int, str]] = []

    def load_scene(sample_id: int) -> int:
        return sample_id

    evaluators = {
        name: (lambda scene, model=name: order.append((scene, model)) or f"{scene}:{model}")
        for name in ("b0", "seed1", "seed2", "seed3")
    }
    rows = evaluate_scene_major([10, 20], load_scene, evaluators)
    assert order == [
        (10, "b0"), (10, "seed1"), (10, "seed2"), (10, "seed3"),
        (20, "b0"), (20, "seed1"), (20, "seed2"), (20, "seed3"),
    ]
    assert rows["seed2"] == ["10:seed2", "20:seed2"]


def test_legacy_b0_checkpoint_uses_actual_config_split_normalization_and_completion() -> None:
    config = {"model": {"name": "factorized_fno"}, "data": {"split_manifest": "s.json"}}
    split = {"train": [1], "val": [2], "test": [3]}
    payload = {
        "model_state_dict": {"weight": torch.ones(1)}, "full_config": config,
        "model_config": config["model"], "split_manifest": split,
        "normalization_stats": {"mean": 1.0}, "global_step": 12000,
    }
    state = validate_legacy_checkpoint_binding(payload, config, split, {"mean": 1.0})
    assert torch.equal(state["weight"], torch.ones(1))
    with pytest.raises(ValueError, match="config"):
        validate_legacy_checkpoint_binding(
            {**payload, "full_config": {"model": {}}}, config, split, {"mean": 1.0}
        )
    with pytest.raises(ValueError, match="normalization"):
        validate_legacy_checkpoint_binding(payload, config, split, {"mean": 2.0})
    with pytest.raises(ValueError, match="12000"):
        validate_legacy_checkpoint_binding(
            {**payload, "global_step": 11999}, config, split, {"mean": 1.0}
        )


def _gate_row(sample_id: int, category: str, scale: float, *, seed: str,
              config_hash: str, checkpoint_hash: str, split_hash: str,
              normalization_hash: str) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in SAMPLE_COLUMNS}
    row.update({
        "sample_id": sample_id, "category": category, "split": "val", "seed": seed,
        "predictor_family": "factorized_fno_b0" if seed == "b0" else "ais_mqfno",
        "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
        "split_manifest_sha256": split_hash, "normalization_sha256": normalization_hash,
        "receiver_geometry_sha256": "f" * 64,
    })
    for name in ("relative_l2", "relative_l2_q1", "relative_l2_q2", "relative_l2_q3",
                 "relative_l2_q4", "receiver_relative_l2", "receiver_relative_l2_q1",
                 "receiver_relative_l2_q2", "receiver_relative_l2_q3",
                 "receiver_relative_l2_q4", "komega_relative_l2", "komega_relative_l2_q4",
                 "komega_high", "komega_high_q4"):
        row[name] = scale
    row.update({
        "field_q4_q1_ratio": 1.0, "receiver_q4_q1_ratio": 1.0,
        "active_time_coverage": 1.0, "active_time_error_slope_per_s": 0.0,
        "active_time_error_max": scale, "arrival_mae_s": 0.01,
        "arrival_miss_rate": 0.0, "arrival_target_coverage": 1.0,
        "receiver_lag_abs_s": 0.01, "receiver_xcorr_peak": 0.9,
        "receiver_phase_error": 0.1, "receiver_phase_coherence": 0.9,
        "energy_log_ratio": 0.1, "zero_relative_l2": 2.0,
        "zero_relative_l2_q4": 2.0, "zero_receiver_relative_l2": 2.0,
        "zero_receiver_relative_l2_q4": 2.0,
        "prediction_target_norm_ratio": 1.0, "prediction_target_pearson": 0.9,
        "sampler_ess": 160000.0, "sampler_duplicate_fraction": 0.0,
        "sampler_coverage": 1.0, "sampler_max_median_inverse_weight": 1.0,
        "prediction_finite": 1.0,
        "prediction_nonzero": 1.0, "output_height": 400.0,
        "output_width": 400.0, "output_time_steps": 160.0,
    })
    return row


def _write_census(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_actual_compare_cli_gate_tree_authorizes_with_documented_paths(tmp_path: Path) -> None:
    frozen, experiment, _, _ = _frozen_completed_validation(tmp_path)
    exp = json.loads(experiment.read_text())
    split_hash = frozen.manifest["split_manifest"]["sha256"]
    norm_hash = frozen.manifest["normalization_stats"]["sha256"]
    categories = ("uniform", "uniform", "layered", "layered", "marmousi", "marmousi")
    sample_ids = list(range(301, 307))
    baseline_rows = [
        _gate_row(sample_id, category, 1.0, seed="b0",
                  config_hash=_canonical_config_sha(frozen.config_b0),
                  checkpoint_hash=exp["b0_checkpoint"]["sha256"], split_hash=split_hash,
                  normalization_hash=norm_hash)
        for sample_id, category in zip(sample_ids, categories, strict=True)
    ]
    baseline_csv = tmp_path / "baselines/b0/val/samples.csv"
    _write_census(baseline_csv, baseline_rows)
    candidate_csvs = []
    for seed, checkpoint in zip(frozen.manifest["confirmation_seeds"],
                                exp["candidate_checkpoints"], strict=True):
        rows = [_gate_row(sample_id, category, 0.6, seed=str(seed),
                          config_hash=_canonical_config_sha(frozen.config_400),
                          checkpoint_hash=checkpoint["sha256"], split_hash=split_hash,
                          normalization_hash=norm_hash)
                for sample_id, category in zip(sample_ids, categories, strict=True)]
        path = tmp_path / f"evaluations/400_final_seed{seed}_val/samples.csv"
        _write_census(path, rows)
        candidate_csvs.append(path)
        result = subprocess.run([
            sys.executable, str(ROOT / "scripts/compare_native400_accuracy.py"),
            "--baseline-csv", str(baseline_csv), "--candidate-csv", str(path),
            "--threshold", "0.30", "--bootstrap-replicates", "10000",
            "--seed", "20260714", "--output-dir",
            str(tmp_path / f"gates/confirmation_val_seed{seed}"),
        ], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    command = [sys.executable, str(ROOT / "scripts/compare_native400_accuracy.py"),
               "--baseline-csv", str(baseline_csv)]
    for path in candidate_csvs:
        command.extend(("--candidate-csv", str(path)))
    command.extend(("--aggregate-seeds", "--threshold", "0.30",
                    "--bootstrap-replicates", "10000", "--seed", "20260714",
                    "--output-dir", str(tmp_path / "gates/confirmation_val_aggregate")))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    authorization_path = tmp_path / "authorization.json"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/authorize_ais_sealed_test.py"),
        "--recipe", str(frozen.manifest_path), "--experiment-manifest", str(experiment),
        "--validation-gates", str(tmp_path / "gates"), "--output", str(authorization_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    authorization = json.loads(authorization_path.read_text())
    assert authorization["test_sample_count"] == 250
    tampered = dict(authorization)
    tampered["candidate_checkpoints"] = list(reversed(authorization["candidate_checkpoints"]))
    tampered_path = tmp_path / "tampered-authorization.json"
    tampered_path.write_text(json.dumps(tampered))
    sealed_output = tmp_path / "must-stay-sealed"
    with pytest.raises(ValueError, match="experiment|semantic"):
        run_sealed_test(
            tampered_path, "cpu", sealed_output,
            evaluator=lambda *_: pytest.fail("preflight must refuse before evaluation"),
        )
    assert not (sealed_output / "test_opened.json").exists()
    for field in ("candidate_config", "split_manifest", "normalization_stats"):
        coordinated = dict(authorization)
        source = Path(authorization[field]["path"])
        replacement = tmp_path / f"replacement-{source.name}"
        replacement.write_bytes(source.read_bytes())
        coordinated[field] = {"path": str(replacement), "sha256": _sha(replacement)}
        path = tmp_path / f"tampered-{field}.json"
        path.write_text(json.dumps(coordinated))
        output = tmp_path / f"sealed-{field}"
        with pytest.raises(ValueError, match="recipe inventory"):
            run_sealed_test(path, "cpu", output, evaluator=lambda *_: pytest.fail("must refuse"))
        assert not (output / "test_opened.json").exists()


def test_sealed_gate_bundle_matches_compare_cli_bootstrap_seed_schedule() -> None:
    categories = ("uniform", "uniform", "layered", "layered", "marmousi", "marmousi")
    ids = list(range(6))
    baseline = [_gate_row(sample_id, category, 1.0, seed="b0",
                          config_hash="a" * 64, checkpoint_hash="b" * 64,
                          split_hash="c" * 64, normalization_hash="d" * 64)
                for sample_id, category in zip(ids, categories, strict=True)]
    candidates = [[_gate_row(sample_id, category, 0.6, seed=str(seed),
                             config_hash="e" * 64, checkpoint_hash=str(index + 1) * 64,
                             split_hash="c" * 64, normalization_hash="d" * 64)
                   for sample_id, category in zip(ids, categories, strict=True)]
                  for index, seed in enumerate((20260714, 20260715, 20260716))]
    calls: list[int] = []

    def recording_evaluator(base, candidate, threshold, replicates, seed):
        calls.append(seed)
        return evaluate_native400_gate(base, candidate, threshold, replicates, seed)

    published, aggregate_seeds, _, combined = compute_sealed_gate_bundle(
        baseline, candidates, bootstrap_replicates=20, gate_evaluator=recording_evaluator
    )
    assert calls == [20260714] * 4 + [20260715, 20260716, 20260714]
    assert len(published) == len(aggregate_seeds) == 3
    assert combined.passed
