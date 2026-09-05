from __future__ import annotations

import csv
import hashlib
import h5py
import json
import io
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from fno_acoustic.long_horizon_metrics import ReceiverSiteManifest
from fno_acoustic.query_census import (
    REQUIRED_NUMERIC_COLUMNS,
    SAMPLE_COLUMNS,
    LegacyFullFieldPredictorAdapter,
    NativePrediction,
    QueryPredictorAdapter,
    run_query_census,
    write_rows_atomically,
)
from fno_acoustic.query_data import QueryScene


def _scene(sample_id: int = 3, category: str = "layered") -> QueryScene:
    h, w = 9, 7
    time = torch.linspace(0, 1, 160, dtype=torch.float64)
    target = torch.sin(time.float())[None, None].expand(h, w, -1).clone()
    return QueryScene(sample_id, target, torch.ones(h, w), torch.zeros(h, w), time,
                      torch.arange(h, dtype=torch.float64),
                      torch.arange(w, dtype=torch.float64), {"model_type": category})


class Store:
    def read_scene(self, sample_id: int) -> QueryScene:
        return _scene(sample_id, "layered" if sample_id % 2 else "uniform")


class Predictor:
    predictor_family = "ais_mqfno"

    def predict_scene(self, scene: QueryScene) -> NativePrediction:
        return NativePrediction(scene.target_cpu[None].clone(), 1, 1)


def _manifest() -> ReceiverSiteManifest:
    return ReceiverSiteManifest((9, 7), torch.tensor([0, 20, 62]),
                                torch.tensor([[0., 0.], [2., 6.], [8., 6.]]), "f" * 64)


def _provenance() -> dict[str, str]:
    return {"seed": "20260714", "config_sha256": "1" * 64,
            "checkpoint_sha256": "2" * 64, "split_manifest_sha256": "3" * 64,
            "normalization_sha256": "4" * 64, "receiver_geometry_sha256": "f" * 64}


def _valid_row() -> dict[str, object]:
    row = {name: 0.0 for name in SAMPLE_COLUMNS}
    row.update({"sample_id": 1, "category": "layered", "split": "val", "seed": "1",
                "predictor_family": "ais_mqfno", "config_sha256": "1" * 64,
                "checkpoint_sha256": "2" * 64, "split_manifest_sha256": "3" * 64,
                "normalization_sha256": "4" * 64, "receiver_geometry_sha256": "5" * 64})
    return row


def _identity_stats() -> dict[str, object]:
    return {"eps": 1e-6, "velocity": {"mean": 0.0, "std": 1.0},
            "wavefield": {"mean": 0.0, "std": 1.0}}


def test_sample_columns_are_exact_and_numeric_suffix_is_stable() -> None:
    assert SAMPLE_COLUMNS[:10] == (
        "sample_id", "category", "split", "seed", "predictor_family",
        "config_sha256", "checkpoint_sha256", "split_manifest_sha256",
        "normalization_sha256", "receiver_geometry_sha256",
    )
    assert REQUIRED_NUMERIC_COLUMNS == SAMPLE_COLUMNS[10:]


def test_census_uses_each_physical_scene_once_and_writes_summary(tmp_path: Path) -> None:
    report = run_query_census(Predictor(), Store(), [2, 3], "val", tmp_path / "census",
                              _manifest(), _provenance())
    assert report.coverage_min == report.coverage_max == 1
    assert report.sample_count == 2
    rows = list(csv.DictReader(report.samples_csv.open()))
    assert len(rows) == 2 and tuple(rows[0]) == SAMPLE_COLUMNS
    summary = json.loads(report.summary_json.read_text())
    assert summary["sample_count"] == 2
    assert summary["categories"]["layered"]["sample_count"] == 1
    assert summary["categories"]["uniform"]["sample_count"] == 1


def test_bad_coverage_and_split_or_provenance_are_rejected(tmp_path: Path) -> None:
    class Bad(Predictor):
        def predict_scene(self, scene: QueryScene) -> NativePrediction:
            return NativePrediction(scene.target_cpu[None], 0, 1)

    with pytest.raises(ValueError, match="coverage"):
        run_query_census(Bad(), Store(), [3], "val", tmp_path / "bad", _manifest(), _provenance())
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="split"):
        run_query_census(Predictor(), Store(), [3], "holdout", tmp_path / "test", _manifest(), _provenance())
    invalid = _provenance()
    invalid["checkpoint_sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA-256"):
        run_query_census(Predictor(), Store(), [3], "val", tmp_path / "prov", _manifest(), invalid)


def test_atomic_writer_rejects_nonfinite_and_preserves_old_file(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    path.write_text("old\n")
    row = _valid_row()
    row["relative_l2"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_rows_atomically(path, [row])
    assert path.read_text() == "old\n"
    assert not list(tmp_path.glob("*.tmp"))
    row = _valid_row()
    row.pop("relative_l2")
    with pytest.raises(ValueError, match="columns"):
        write_rows_atomically(path, [row])


def test_atomic_replace_failure_preserves_old_file_and_cleans_tmp(tmp_path: Path, monkeypatch) -> None:
    import fno_acoustic.query_census as module

    path = tmp_path / "samples.csv"
    path.write_text("old\n")
    monkeypatch.setattr(module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("replace")))
    with pytest.raises(OSError, match="replace"):
        write_rows_atomically(path, [_valid_row()])
    assert path.read_text() == "old\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_census_scene_and_summary_failures_publish_no_directory(tmp_path: Path, monkeypatch) -> None:
    import fno_acoustic.query_census as module

    class FailingStore(Store):
        def read_scene(self, sample_id: int) -> QueryScene:
            if sample_id == 3:
                raise RuntimeError("scene failure")
            return super().read_scene(sample_id)

    final = tmp_path / "scene-failure"
    with pytest.raises(RuntimeError, match="scene failure"):
        run_query_census(Predictor(), FailingStore(), [2, 3], "val", final,
                         _manifest(), _provenance())
    assert not final.exists() and not list(tmp_path.glob(".scene-failure.*.tmp"))

    monkeypatch.setattr(module, "write_census_summary",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("summary failure")))
    final = tmp_path / "summary-failure"
    with pytest.raises(RuntimeError, match="summary failure"):
        run_query_census(Predictor(), Store(), [3], "val", final,
                         _manifest(), _provenance())
    assert not final.exists() and not list(tmp_path.glob(".summary-failure.*.tmp"))


def test_census_predictions_and_directory_publish_are_transactional(tmp_path: Path, monkeypatch) -> None:
    import fno_acoustic.query_census as module

    final = tmp_path / "prediction-failure"
    monkeypatch.setattr(module, "_write_prediction_atomically",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("prediction failure")),
                        raising=False)
    with pytest.raises(RuntimeError, match="prediction failure"):
        run_query_census(Predictor(), Store(), [3], "val", final, _manifest(), _provenance(),
                         representative_sample_ids=[3],
                         representative_provenance={"representative_manifest_sha256": "a" * 64})
    assert not final.exists() and not list(tmp_path.glob(".prediction-failure.*.tmp"))

    monkeypatch.undo()
    original_replace = module.os.replace
    def fail_directory_publish(source: object, target: object) -> None:
        if Path(source).is_dir():
            raise OSError("directory publish")
        original_replace(source, target)
    monkeypatch.setattr(module.os, "replace", fail_directory_publish)
    final = tmp_path / "publish-failure"
    with pytest.raises(OSError, match="directory publish"):
        run_query_census(Predictor(), Store(), [3], "val", final, _manifest(), _provenance())
    assert not final.exists() and not list(tmp_path.glob(".publish-failure.*.tmp"))


def test_census_success_publishes_predictions_with_single_final_directory(tmp_path: Path) -> None:
    final = tmp_path / "published"
    report = run_query_census(
        Predictor(), Store(), [3], "val", final, _manifest(), _provenance(),
        representative_sample_ids=[3],
        representative_provenance={"representative_manifest_sha256": "a" * 64},
    )
    assert report.samples_csv == final / "samples.csv"
    assert (final / "predictions/sample_3.npy").is_file()
    assert (final / "predictions/provenance.json").is_file()
    with pytest.raises(FileExistsError):
        run_query_census(Predictor(), Store(), [3], "val", final, _manifest(), _provenance())


def test_census_summary_includes_formal_dataset_verification_scope(tmp_path: Path) -> None:
    binding = {
        "dataset_content_root": "a" * 64,
        "verified_sample_count": 1,
        "verified_sample_ids_sha256": "b" * 64,
        "verified_sample_scope": "process_unique",
    }
    report = run_query_census(
        Predictor(), Store(), [3], "val", tmp_path / "bound", _manifest(),
        _provenance(), binding_summary=binding,
    )

    assert json.loads(report.summary_json.read_bytes()) | binding == json.loads(
        report.summary_json.read_bytes()
    )


def test_legacy_adapter_uses_same_native_prediction_contract() -> None:
    class Legacy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[:4], device=x.device)

    prediction = LegacyFullFieldPredictorAdapter(
        Legacy(), torch.device("cpu"), _identity_stats()
    ).predict_scene(_scene())
    assert prediction.field_cpu.shape == (1, 9, 7, 160)
    assert prediction.coverage_min == prediction.coverage_max == 1


def test_query_adapter_streams_every_site_once() -> None:
    class QueryModel(torch.nn.Module):
        def encode_global(self, global_inputs: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
            return global_inputs

        def decode_queries(self, context: torch.Tensor, native_static: torch.Tensor,
                           query_xz: torch.Tensor, time_s: torch.Tensor) -> torch.Tensor:
            return torch.zeros(query_xz.shape[0], query_xz.shape[1], 160, device=query_xz.device)

    prediction = QueryPredictorAdapter(
        QueryModel(), torch.device("cpu"), chunk_sites=13, global_size=5
    ).predict_scene(_scene())
    assert prediction.field_cpu.shape == (1, 9, 7, 160)
    assert prediction.coverage_min == prediction.coverage_max == 1


def test_checkpoint_binding_requires_v2_config_and_split_hashes() -> None:
    from scripts.evaluate_ais_mqfno import validate_checkpoint_binding

    payload = {"schema_version": 2, "config_sha256": "1" * 64,
               "split_manifest_sha256": "2" * 64, "model_state_dict": {"x": torch.ones(1)}}
    assert validate_checkpoint_binding(payload, "1" * 64, "2" * 64) == {"x": torch.ones(1)}
    with pytest.raises(ValueError, match="config"):
        validate_checkpoint_binding(payload, "3" * 64, "2" * 64)


def test_evaluator_supports_legacy_b0_binding_and_runtime_seed_sidecar(tmp_path: Path) -> None:
    from scripts.evaluate_ais_mqfno import (
        checkpoint_runtime_seed, validate_legacy_checkpoint_binding,
    )

    config = {"model": {"name": "factorized_fno"}}
    split = {"train": [1], "val": [2], "test": [3]}
    payload = {"full_config": config, "model_config": config["model"],
               "split_manifest": split, "model_state_dict": {"x": torch.ones(1)},
               "normalization_stats": _identity_stats()}
    state, stats = validate_legacy_checkpoint_binding(payload, config, split)
    assert torch.equal(state["x"], torch.ones(1)) and stats == _identity_stats()
    assert checkpoint_runtime_seed(
        {"schema_version": 3, "runtime_seed": 20260716}, 1
    ) == "20260716"
    assert checkpoint_runtime_seed({"schema_version": 2}, 1) == "1"


def test_checkpoint_snapshot_reads_once_and_hashes_loaded_bytes(tmp_path: Path, monkeypatch) -> None:
    from scripts.evaluate_ais_mqfno import load_checkpoint_snapshot

    path = tmp_path / "checkpoint.pt"
    buffer = io.BytesIO()
    torch.save({"schema_version": 2, "model_state_dict": {"x": torch.ones(1)}}, buffer)
    raw = buffer.getvalue()
    path.write_bytes(raw)
    original = Path.read_bytes
    calls = 0
    def read_once(self: Path) -> bytes:
        nonlocal calls
        calls += 1
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", read_once)
    payload, digest = load_checkpoint_snapshot(path)
    assert calls == 1 and payload["schema_version"] == 2
    assert digest == hashlib.sha256(raw).hexdigest()


def test_cli_exposes_val_only_and_required_controls() -> None:
    from scripts.evaluate_ais_mqfno import build_parser

    parser = build_parser()
    base = ["--config", "c.yaml", "--checkpoint", "m.pt", "--output-dir", "out"]
    args = parser.parse_args([*base, "--split", "val", "--max-samples", "2", "--chunk-sites", "13"])
    assert args.split == "val" and args.max_samples == 2 and args.chunk_sites == 13
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--split", "test"])


def test_cli_help_runs_directly_without_pythonpath() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(Path("/home/jiayh/miniforge3/envs/qwen/bin/python")),
         str(repo / "scripts/evaluate_ais_mqfno.py"), "--help"],
        cwd=repo.parent, text=True, capture_output=True, check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "--representative-manifest" in result.stdout


def test_normalization_hash_prefers_formal_path_and_rejects_legacy_conflict(tmp_path: Path) -> None:
    from scripts.evaluate_ais_mqfno import normalization_hash

    stats = tmp_path / "stats.json"
    stats.write_text('{"eps":1e-6}')
    expected = hashlib.sha256(stats.read_bytes()).hexdigest()
    config = {"normalization": {"stats_path": str(stats)}, "data": {}}
    assert normalization_hash(config) == expected
    other = tmp_path / "other.json"
    other.write_text("{}")
    config["data"]["normalization_stats"] = str(other)
    with pytest.raises(ValueError, match="conflict"):
        normalization_hash(config)


def test_task13_representative_manifest_binds_split_hash_and_categories(tmp_path: Path) -> None:
    from scripts.evaluate_ais_mqfno import load_representative_manifest

    class RepresentativeStore:
        def read_scene(self, sample_id: int) -> QueryScene:
            return _scene(sample_id, {2: "uniform", 3: "layered", 5: "marmousi"}[sample_id])

    payload = {"split": "val", "split_manifest_sha256": "3" * 64,
               "representatives": {"uniform": 2, "layered": 3, "marmousi": 5},
               "selection_rule": "lowest registered validation id", "metric_source_sha256": "9" * 64,
               "seed": 31, "extra_provenance": {"frozen": True}}
    path = tmp_path / "representatives.json"
    path.write_text(json.dumps(payload))
    selection = load_representative_manifest(path, RepresentativeStore(), [2, 3, 5], "3" * 64)
    assert selection.sample_ids == (2, 3, 5)
    assert selection.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    payload["representatives"]["marmousi"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique|category"):
        load_representative_manifest(path, RepresentativeStore(), [2, 3, 5], "3" * 64)
    payload["representatives"] = {"uniform": 2, "layered": 3, "marmousi": 5}
    payload["metric_source_sha256"] = "bad"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metric"):
        load_representative_manifest(path, RepresentativeStore(), [2, 3, 5], "3" * 64)
    payload["metric_source_sha256"] = "9" * 64
    payload["split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="split"):
        load_representative_manifest(path, RepresentativeStore(), [2, 3, 5], "3" * 64)
    payload["split"] = "val"
    payload["representatives"] = {"uniform": 3, "layered": 2, "marmousi": 5}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="category"):
        load_representative_manifest(path, RepresentativeStore(), [2, 3, 5], "3" * 64)


def test_representative_manifest_is_hashed_and_parsed_from_one_read(tmp_path: Path, monkeypatch) -> None:
    from scripts.evaluate_ais_mqfno import load_representative_manifest

    class RepresentativeStore:
        def read_scene(self, sample_id: int) -> QueryScene:
            return _scene(sample_id, {2: "uniform", 3: "layered", 5: "marmousi"}[sample_id])

    payload = {"split": "val", "split_manifest_sha256": "3" * 64,
               "representatives": {"uniform": 2, "layered": 3, "marmousi": 5},
               "selection_rule": "frozen", "metric_source_sha256": "9" * 64, "seed": 31}
    path = tmp_path / "representatives.json"
    path.write_text(json.dumps(payload))
    original = Path.read_bytes
    calls = 0
    def read_once(self: Path) -> bytes:
        nonlocal calls
        calls += 1
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", read_once)
    selection = load_representative_manifest(path, RepresentativeStore(), [2, 3, 5], "3" * 64)
    assert calls == 1
    assert selection.sha256 == hashlib.sha256(original(path)).hexdigest()


def test_legacy_adapter_normalizes_velocity_and_decodes_physical_wavefield() -> None:
    seen: dict[str, torch.Tensor] = {}
    class Legacy(torch.nn.Module):
        in_features = 3
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen["velocity"] = x[..., 2].detach().clone()
            return torch.zeros(x.shape[:4])
    stats = {"eps": 1e-6, "velocity": {"mean": 1.0, "std": 2.0},
             "wavefield": {"mean": 7.0, "std": 3.0}}
    result = LegacyFullFieldPredictorAdapter(
        Legacy(), torch.device("cpu"), stats
    ).predict_scene(_scene())
    assert torch.allclose(seen["velocity"], torch.zeros_like(seen["velocity"]))
    assert torch.all(result.field_cpu == 7.0)


def test_legacy_stats_accept_real_artifact_provenance_schema() -> None:
    class Legacy(torch.nn.Module):
        in_features = 3
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[:4])

    stats = {
        "computed_from_split": "train", "eps": 1e-6,
        "source_map": {"normalization": "max_to_one"},
        "time": {"normalization": "zero_to_one"}, "train_sample_count": 64,
        "velocity": {"count": 262144, "mean": 1.0, "std": 2.0},
        "wavefield": {"count": 20971520, "mean": 7.0, "std": 3.0},
    }
    result = LegacyFullFieldPredictorAdapter(
        Legacy(), torch.device("cpu"), stats
    ).predict_scene(_scene())
    assert torch.all(result.field_cpu == 7.0)


def test_tiny_cli_subprocess_runs_without_pythonpath(tmp_path: Path) -> None:
    from fno_acoustic.model_ais_mqfno import AISMQFNO
    from scripts.train_ais_mqfno import _config_sha256

    data_path = tmp_path / "tiny.h5"
    with h5py.File(data_path, "w") as h5:
        time = torch.linspace(0, 1, 160).numpy()
        target = torch.sin(torch.from_numpy(time))[None, :, None, None].expand(2, -1, 3, 3).numpy()
        h5.create_dataset("tensor", data=target)
        h5.create_dataset("nu", data=torch.ones(2, 3, 3).numpy())
        source = torch.zeros(2, 3, 3)
        source[:, 1, 1] = 1
        h5.create_dataset("source_mask", data=source.numpy())
        h5.create_dataset("t-coordinate", data=time)
        h5.create_dataset("x-coordinate", data=torch.linspace(0, 2, 3).numpy())
        h5.create_dataset("y-coordinate", data=torch.linspace(0, 2, 3).numpy())
        h5.create_dataset("model_type", data=[b"uniform", b"layered"])
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"train": [0], "val": [1], "test": []}))
    stats_path = tmp_path / "stats.json"
    stats = {**_identity_stats(), "computed_from_split": "train"}
    stats_path.write_text(json.dumps(stats))
    config = {
        "seed": 7,
        "data": {"path": str(data_path), "split_manifest": str(split_path)},
        "normalization": {
            "contract": "ais_normalization_v2",
            "stats_path": str(stats_path),
        },
        "sampling": {"target_height": 3, "target_width": 3, "global_size": 3,
                     "max_time_steps": 160, "mixture": [1 / 6] * 6},
        "model": {"name": "ais_mqfno", "global_in_features": 6,
                  "native_in_channels": 5, "spatial_width": 2, "spatial_modes": 1,
                  "temporal_modes": 2, "local_dim": 2, "fusion_dim": 2,
                  "halo_size": 3, "spatial_layers": 1,
                  "activation_checkpointing": False, "spatial_chunk_size": 0},
        "receiver": {"x_start_m": 0.0, "x_stop_m": 2000.0,
                     "x_stride_m": 1000.0, "z_m": [0.0]},
        "train": {"amp": False, "phases": [{"name": "tiny", "optimizer_updates": 1,
                                                "init_from": "random"}]},
        "loss": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    checkpoint = tmp_path / "model.pt"
    model = AISMQFNO(global_in_features=6, native_in_channels=5, spatial_width=2,
                     spatial_modes=1, temporal_modes=2, local_dim=2, fusion_dim=2,
                     halo_size=3, spatial_layers=1)
    torch.save({
        "schema_version": 4,
        "config_sha256": _config_sha256(config),
        "split_manifest_sha256": split_hash,
        "normalization_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "normalization_contract": "ais_normalization_v2",
        "runtime_seed": 7,
        "model_state_dict": model.state_dict(),
    }, checkpoint)
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "evaluation"
    result = subprocess.run([
        "/home/jiayh/miniforge3/envs/qwen/bin/python", str(repo / "scripts/evaluate_ais_mqfno.py"),
        "--config", str(config_path), "--checkpoint", str(checkpoint), "--split", "val",
        "--output-dir", str(output), "--device", "cpu", "--max-samples", "1",
        "--chunk-sites", "9",
    ], cwd=repo.parent, text=True, capture_output=True, check=False, env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    assert (output / "samples.csv").is_file() and (output / "summary.json").is_file()
