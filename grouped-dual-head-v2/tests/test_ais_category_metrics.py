from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from fno_acoustic.ais_normalization import AISNormalizationBinding
from fno_acoustic.long_horizon_metrics import (
    ReceiverSiteManifest,
    long_horizon_metrics,
)
from fno_acoustic.query_census import (
    REQUIRED_NUMERIC_COLUMNS,
    SAMPLE_COLUMNS,
    SCREEN_NUMERIC_COLUMNS,
    SCREEN_SAMPLE_COLUMNS,
    aggregate_category_metrics,
    aggregate_screen_category_metrics,
)
from fno_acoustic.query_data import QueryScene
import scripts.evaluate_ais_mqfno as evaluate_ais_mqfno
from scripts.evaluate_ais_mqfno import (
    PINNED_LEGACY_RAW_B1_CONFIG_SHA256,
    PINNED_LEGACY_RAW_B1_SHA256,
    NormalizedQueryPredictorAdapter,
    load_legacy_raw_b1,
    validate_legacy_raw_b1_config,
    validate_legacy_raw_b1_operation,
)
from scripts.train_ais_mqfno import _config_sha256


ROOT = Path(__file__).resolve().parents[1]
PINNED_CONFIG = ROOT / "configs/ais_mqfno_64x160_b1_uniform.yaml"


def _row(sample_id: int, category: str, relative_l2: float) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in SAMPLE_COLUMNS}
    row.update(
        {
            "sample_id": sample_id,
            "category": category,
            "split": "val",
            "seed": "7",
            "predictor_family": "ais_mqfno",
            "config_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64,
            "split_manifest_sha256": "3" * 64,
            "normalization_sha256": "4" * 64,
            "receiver_geometry_sha256": "5" * 64,
            "relative_l2": relative_l2,
            "relative_l2_q4": relative_l2,
            "zero_relative_l2": 1.0,
            "zero_relative_l2_q4": 1.0,
            "prediction_target_norm_ratio": 1.0,
            "prediction_target_pearson": 1.0,
            "prediction_finite": 1.0,
            "prediction_nonzero": 1.0,
            "sampler_ess": 2048.0,
            "sampler_duplicate_fraction": 0.0,
            "sampler_coverage": 1.0,
            "sampler_max_median_inverse_weight": 1.0,
        }
    )
    return row


def test_category_aggregation_is_sample_first_not_concatenated_energy() -> None:
    rows = [_row(1, "uniform", 1.0), _row(2, "uniform", 0.0)]

    report = aggregate_category_metrics(rows)

    concatenated_ratio = 1.0 / (1.0 + 100.0**2) ** 0.5
    assert report["uniform"]["relative_l2"] == pytest.approx(0.5)
    assert report["global"]["relative_l2"] == pytest.approx(0.5)
    assert report["uniform"]["relative_l2"] != pytest.approx(concatenated_ratio)


def test_three_category_report_has_zero_norm_pearson_and_sampler_fields() -> None:
    report = aggregate_category_metrics(
        [_row(index, category, 0.25) for index, category in enumerate(
            ("uniform", "layered", "marmousi"), start=1
        )]
    )

    for category in ("global", "uniform", "layered", "marmousi"):
        assert report[category]["zero_relative_l2"] == pytest.approx(1.0)
        assert report[category]["zero_relative_l2_q4"] == pytest.approx(1.0)
        assert report[category]["prediction_target_norm_ratio"] == pytest.approx(1.0)
        assert report[category]["prediction_target_pearson"] == pytest.approx(1.0)
        assert report[category]["sampler_ess"] == pytest.approx(2048.0)


def test_category_aggregation_rejects_unknown_or_missing_metadata_category() -> None:
    with pytest.raises(ValueError, match="category|model_type"):
        aggregate_category_metrics([_row(1, "unknown", 0.2)])
    missing = _row(1, "uniform", 0.2)
    missing.pop("category")
    with pytest.raises(ValueError, match="category"):
        aggregate_category_metrics([missing])


def test_screen64_aggregation_uses_strict_sparse_metric_schema() -> None:
    rows = []
    for sample_id, category, value in ((1, "uniform", 0.2), (2, "layered", 0.4)):
        row = {name: "a" * 64 for name in SCREEN_SAMPLE_COLUMNS}
        row.update(
            sample_id=sample_id,
            category=category,
            split="val",
            predictor_family="ais_mqfno",
            **{name: value for name in SCREEN_NUMERIC_COLUMNS},
        )
        rows.append(row)

    report = aggregate_screen_category_metrics(rows)

    assert report["global"]["relative_l2"] == pytest.approx(0.3)
    assert report["uniform"]["sample_count"] == 1
    with pytest.raises(ValueError, match="screen.*schema"):
        aggregate_screen_category_metrics([{**rows[0], "forged": 1}])


def test_long_horizon_statistics_are_computed_from_physical_fields() -> None:
    time_s = torch.linspace(0.0, 1.0, 160, dtype=torch.float64)
    target = torch.linspace(-2.0, 3.0, 160)[None, None, None].expand(1, 3, 3, -1)
    prediction = 2.0 * target
    receivers = ReceiverSiteManifest(
        (3, 3), torch.tensor([0, 4, 8]),
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), "a" * 64,
    )

    metrics = long_horizon_metrics(prediction, target, time_s, receivers)

    assert metrics["prediction_target_norm_ratio"] == pytest.approx(2.0)
    assert metrics["prediction_target_pearson"] == pytest.approx(1.0)
    assert metrics["zero_relative_l2"] == pytest.approx(1.0)
    assert metrics["zero_relative_l2_q4"] == pytest.approx(1.0)


def test_normalized_adapter_decodes_complete_field_before_metrics() -> None:
    class ConstantStandardizedModel(torch.nn.Module):
        def encode_global(self, global_inputs, time_s):
            return global_inputs.mean(dim=(1, 2, 4))

        def decode_queries(self, context, native_static, query_xz, time_s):
            return torch.full((1, query_xz.shape[1], 160), 2.0)

    scene = QueryScene(
        1,
        torch.full((3, 3, 160), 5.0),
        torch.full((3, 3), 3000.0),
        torch.nn.functional.pad(torch.ones(1, 1), (1, 1, 1, 1)),
        torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        {"model_type": "uniform"},
    )
    binding = AISNormalizationBinding(
        Path("unused.json"), "a" * 64, 3000.0, 500.0, 1.0, 2.0, 1.0e-6
    )

    native = NormalizedQueryPredictorAdapter(
        ConstantStandardizedModel(), torch.device("cpu"), 4, binding, global_size=3
    ).predict_scene(scene)

    assert native.field_cpu.shape == (1, 3, 3, 160)
    assert torch.equal(native.field_cpu, torch.full_like(native.field_cpu, 5.0))


def test_pinned_legacy_config_hash_and_forbidden_operations_are_fixed() -> None:
    config = yaml.safe_load(PINNED_CONFIG.read_text(encoding="utf-8"))
    assert PINNED_LEGACY_RAW_B1_SHA256 == (
        "91546ba21c31e0e875c0ed7051068c279b455a94f38206274094e1a9a9233653"
    )
    assert PINNED_LEGACY_RAW_B1_CONFIG_SHA256 == (
        "f9a8b81e116f3ba4c463ee0875b67f90ab46eba413a8e78f10d834ebc8e8dd81"
    )
    assert _config_sha256(config) == PINNED_LEGACY_RAW_B1_CONFIG_SHA256
    assert validate_legacy_raw_b1_config(config) == PINNED_LEGACY_RAW_B1_CONFIG_SHA256
    config["seed"] += 1
    with pytest.raises(ValueError, match="pinned original config"):
        validate_legacy_raw_b1_config(config)
    for operation in (
        "representative_tuning", "test", "resume", "initialization",
        "recipe_freeze", "authorization", "sealed_test",
    ):
        with pytest.raises(ValueError, match="legacy raw B1|validation baseline"):
            validate_legacy_raw_b1_operation(operation)
    assert validate_legacy_raw_b1_operation("validation_baseline") == "validation_baseline"


def test_load_legacy_raw_b1_rejects_every_unpinned_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "different.pt"
    path.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="pinned legacy B1"):
        load_legacy_raw_b1(path)


def test_load_legacy_raw_b1_exposes_only_read_only_model_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path = tmp_path / "splits.json"
    split_path.write_text('{"train":[1],"val":[2],"test":[3]}', encoding="utf-8")
    split_sha256 = hashlib.sha256(split_path.read_bytes()).hexdigest()
    config = {"seed": 17, "data": {"split_manifest": str(split_path)}}
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    config_sha256 = _config_sha256(config)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema_version": 3,
            "config_sha256": config_sha256,
            "split_manifest_sha256": split_sha256,
            "runtime_seed": 17,
            "model_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {"forbidden": True},
            "scheduler_state_dict": {"forbidden": True},
            "sampler_state_dict": {"forbidden": True},
            "rng_state": {"forbidden": True},
            "training_state": {"forbidden": True},
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(
        evaluate_ais_mqfno, "PINNED_LEGACY_RAW_B1_CONFIG", config_path
    )
    monkeypatch.setattr(
        evaluate_ais_mqfno, "PINNED_LEGACY_RAW_B1_SHA256", checkpoint_sha256
    )
    monkeypatch.setattr(
        evaluate_ais_mqfno,
        "PINNED_LEGACY_RAW_B1_CONFIG_SHA256",
        config_sha256,
    )

    preflight_binding = evaluate_ais_mqfno.preflight_ais_evaluation_checkpoint(
        checkpoint,
        config,
        split_sha256,
        legacy_raw_b1_baseline=True,
    )
    binding = load_legacy_raw_b1(checkpoint)

    for candidate in (preflight_binding, binding):
        assert candidate.checkpoint_sha256 == checkpoint_sha256
        assert candidate.legacy_raw_b1 is True
        assert candidate.normalization is None
        for training_attribute in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "sampler_state_dict",
            "rng_state",
            "training_state",
        ):
            assert not hasattr(candidate, training_attribute)
        with pytest.raises(TypeError):
            candidate.model_state_dict["forbidden"] = torch.ones(1)
        candidate.model_state_dict["weight"].zero_()
        assert torch.equal(candidate.model_state_dict["weight"], torch.ones(1))
        assert candidate.checkpoint_sha256 == checkpoint_sha256


def test_evaluation_config_paths_resolve_from_repository_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = Path("data/native400.h5")
    absolute = tmp_path / "absolute.h5"

    assert evaluate_ais_mqfno.resolve_repository_config_path(relative) == ROOT / relative
    assert evaluate_ais_mqfno.resolve_repository_config_path(absolute) == absolute


def test_category_numeric_contract_includes_all_new_physical_fields() -> None:
    for name in (
        "zero_relative_l2", "prediction_target_norm_ratio",
        "prediction_target_pearson", "sampler_ess", "sampler_duplicate_fraction",
        "sampler_coverage", "sampler_max_median_inverse_weight",
    ):
        assert name in REQUIRED_NUMERIC_COLUMNS
