from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import torch
import yaml

from fno_acoustic.ais_normalization import AISNormalizationBinding
from fno_acoustic.ais_sampler import AdaptiveSpatialSampler
from fno_acoustic.query_data import QueryScene
from fno_acoustic.query_training import (
    QUERY_CHECKPOINT_BOUNDARY,
    load_query_checkpoint,
    save_query_checkpoint,
    validate_query_checkpoint_preflight,
)


CONFIG_HASH = "a" * 64
SPLIT_HASH = "b" * 64
NORMALIZATION_HASH = "c" * 64
NORMALIZATION_CONTRACT = "ais_normalization_v2"
PINNED_B1_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/ais_mqfno_full160_native400_20260714/runs/"
    "64_b1_seed20260714/checkpoints/best.pt"
)


def make_objects(seed: int = 17) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    sampler = AdaptiveSpatialSampler(
        4, 4, [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], seed=seed, tile_size=2
    )
    generator = torch.Generator().manual_seed(seed + 1)
    return model, optimizer, scheduler, sampler, generator


class _DispersionCheckpointModel(torch.nn.Module):
    def __init__(self, velocity_mean: float, velocity_std: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.dispersion_residual_head = torch.nn.Module()
        self.dispersion_residual_head.register_buffer(
            "velocity_mean", torch.tensor(velocity_mean, dtype=torch.float64)
        )
        self.dispersion_residual_head.register_buffer(
            "velocity_std", torch.tensor(velocity_std, dtype=torch.float64)
        )


def _dispersion_objects(
    velocity_mean: float = 3000.0, velocity_std: float = 500.0
) -> tuple[Any, ...]:
    model = _DispersionCheckpointModel(velocity_mean, velocity_std)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    sampler = AdaptiveSpatialSampler(
        4, 4, [0.2, 0.2, 0.2, 0.2, 0.1, 0.1], seed=17, tile_size=2
    )
    generator = torch.Generator().manual_seed(18)
    return model, optimizer, scheduler, sampler, generator


def _save_dispersion_screen_checkpoint(
    path: Path,
    objects: tuple[Any, ...],
    *,
    expected_mean: float = 3000.0,
    expected_std: float = 500.0,
) -> None:
    model, optimizer, scheduler, sampler, generator = objects
    save_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        epoch=1,
        epoch_batch_cursor=1,
        epoch_scene_order=[0, 1],
        data_generator=generator,
        global_step=2,
        config_sha256=CONFIG_HASH,
        split_manifest_sha256=SPLIT_HASH,
        checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY,
        phase_index=0,
        phase_update=2,
        runtime_seed=17,
        normalization_stats_sha256=NORMALIZATION_HASH,
        normalization_contract=NORMALIZATION_CONTRACT,
        normalization_velocity_mean=expected_mean,
        normalization_velocity_std=expected_std,
        screen_candidate_id="N6",
        screen_gate="H1",
        dataset_binding_sha256="d" * 64,
        execution_binding_sha256="e" * 64,
    )


def save_checkpoint(path: Path, objects: tuple[Any, ...], *, normalized: bool) -> None:
    model, optimizer, scheduler, sampler, generator = objects
    kwargs: dict[str, object] = {}
    if normalized:
        kwargs.update(
            normalization_stats_sha256=NORMALIZATION_HASH,
            normalization_contract=NORMALIZATION_CONTRACT,
        )
    save_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        epoch=1,
        epoch_batch_cursor=1,
        epoch_scene_order=[0, 1],
        data_generator=generator,
        global_step=2,
        config_sha256=CONFIG_HASH,
        split_manifest_sha256=SPLIT_HASH,
        checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY,
        phase_index=0,
        phase_update=2,
        runtime_seed=17,
        **kwargs,
    )


def load_checkpoint(path: Path, objects: tuple[Any, ...], *, normalized: bool) -> None:
    model, optimizer, scheduler, sampler, generator = objects
    kwargs: dict[str, object] = {}
    if normalized:
        kwargs.update(
            expected_normalization_stats_sha256=NORMALIZATION_HASH,
            expected_normalization_contract=NORMALIZATION_CONTRACT,
        )
    load_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        data_generator=generator,
        expected_config_sha256=CONFIG_HASH,
        expected_split_manifest_sha256=SPLIT_HASH,
        expected_runtime_seed=17,
        **kwargs,
    )


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_schema_v4_checkpoint_binds_normalization_hash_and_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "last.pt"
    save_checkpoint(path, make_objects(), normalized=True)

    payload = torch.load(path, weights_only=True)

    assert payload["schema_version"] == 4
    assert payload["normalization_stats_sha256"] == NORMALIZATION_HASH
    assert payload["normalization_contract"] == NORMALIZATION_CONTRACT
    assert payload["phase_index"] == 0
    assert payload["runtime_seed"] == 17


def test_schema_v5_save_rejects_dispersion_buffers_not_matching_bound_stats(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="dispersion.*velocity_mean"):
        _save_dispersion_screen_checkpoint(
            tmp_path / "last.pt", _dispersion_objects(velocity_mean=3001.0)
        )


def test_schema_v5_tampered_dispersion_buffer_is_fatal_for_preflight_eval_and_resume(
    tmp_path: Path,
) -> None:
    from scripts.evaluate_ais_mqfno import validate_checkpoint_binding

    path = tmp_path / "last.pt"
    source = _dispersion_objects()
    _save_dispersion_screen_checkpoint(path, source)
    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"][
        "dispersion_residual_head.velocity_std"
    ] = torch.tensor(501.0, dtype=torch.float64)
    torch.save(payload, path)

    expected = {
        "expected_normalization_velocity_mean": 3000.0,
        "expected_normalization_velocity_std": 500.0,
    }
    with pytest.raises(ValueError, match="dispersion.*velocity_std"):
        validate_query_checkpoint_preflight(
            payload,
            expected_config_sha256=CONFIG_HASH,
            expected_split_manifest_sha256=SPLIT_HASH,
            expected_runtime_seed=17,
            expected_normalization_stats_sha256=NORMALIZATION_HASH,
            expected_normalization_contract=NORMALIZATION_CONTRACT,
            **expected,
        )
    with pytest.raises(ValueError, match="dispersion.*velocity_std"):
        validate_checkpoint_binding(
            payload,
            CONFIG_HASH,
            SPLIT_HASH,
            NORMALIZATION_HASH,
            NORMALIZATION_CONTRACT,
            normalization_velocity_mean=3000.0,
            normalization_velocity_std=500.0,
        )

    target = _dispersion_objects()
    target_before = copy.deepcopy(target[0].state_dict())
    with pytest.raises(ValueError, match="dispersion.*velocity_std"):
        load_query_checkpoint(
            path,
            model=target[0],
            optimizer=target[1],
            scheduler=target[2],
            sampler=target[3],
            data_generator=target[4],
            expected_config_sha256=CONFIG_HASH,
            expected_split_manifest_sha256=SPLIT_HASH,
            expected_runtime_seed=17,
            expected_normalization_stats_sha256=NORMALIZATION_HASH,
            expected_normalization_contract=NORMALIZATION_CONTRACT,
            **expected,
        )
    assert_nested_equal(target[0].state_dict(), target_before)


def test_schema_v4_allows_base_metadata_without_phase_or_runtime(
    tmp_path: Path,
) -> None:
    objects = make_objects()
    model, optimizer, scheduler, sampler, generator = objects
    path = tmp_path / "base-v4.pt"
    save_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        epoch=0,
        epoch_batch_cursor=0,
        epoch_scene_order=[0],
        data_generator=generator,
        global_step=0,
        config_sha256=CONFIG_HASH,
        split_manifest_sha256=SPLIT_HASH,
        checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY,
        normalization_stats_sha256=NORMALIZATION_HASH,
        normalization_contract=NORMALIZATION_CONTRACT,
    )
    payload = torch.load(path, weights_only=True)
    assert payload["schema_version"] == 4
    assert "phase_index" not in payload and "runtime_seed" not in payload

    target = make_objects(seed=33)
    model, optimizer, scheduler, sampler, generator = target
    metadata = load_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        data_generator=generator,
        expected_config_sha256=CONFIG_HASH,
        expected_split_manifest_sha256=SPLIT_HASH,
        expected_normalization_stats_sha256=NORMALIZATION_HASH,
        expected_normalization_contract=NORMALIZATION_CONTRACT,
    )
    assert metadata.phase_index is None


def test_expected_runtime_seed_rejects_v4_checkpoint_with_missing_field(
    tmp_path: Path,
) -> None:
    objects = make_objects()
    model, optimizer, scheduler, sampler, generator = objects
    path = tmp_path / "missing-runtime.pt"
    save_query_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        epoch=0,
        epoch_batch_cursor=0,
        epoch_scene_order=[0],
        data_generator=generator,
        global_step=0,
        config_sha256=CONFIG_HASH,
        split_manifest_sha256=SPLIT_HASH,
        checkpoint_boundary=QUERY_CHECKPOINT_BOUNDARY,
        normalization_stats_sha256=NORMALIZATION_HASH,
        normalization_contract=NORMALIZATION_CONTRACT,
    )

    with pytest.raises(ValueError, match="runtime seed.*missing"):
        load_query_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            data_generator=generator,
            expected_config_sha256=CONFIG_HASH,
            expected_split_manifest_sha256=SPLIT_HASH,
            expected_runtime_seed=17,
            expected_normalization_stats_sha256=NORMALIZATION_HASH,
            expected_normalization_contract=NORMALIZATION_CONTRACT,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy_v3",
        "missing_hash",
        "changed_hash",
        "missing_contract",
        "changed_contract",
    ],
)
def test_formal_resume_rejects_normalization_mismatch_before_any_mutation(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "last.pt"
    save_checkpoint(path, make_objects(), normalized=mutation != "legacy_v3")
    payload = torch.load(path, weights_only=True)
    if mutation == "missing_hash":
        del payload["normalization_stats_sha256"]
    elif mutation == "changed_hash":
        payload["normalization_stats_sha256"] = "d" * 64
    elif mutation == "missing_contract":
        del payload["normalization_contract"]
    elif mutation == "changed_contract":
        payload["normalization_contract"] = "ais_normalization_v1"
    torch.save(payload, path)

    target = make_objects(seed=91)
    state_before = [copy.deepcopy(item.state_dict()) for item in target[:4]]
    generator_before = target[4].get_state().clone()
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    with pytest.raises(ValueError, match="normalization"):
        load_checkpoint(path, target, normalized=True)

    for item, state in zip(target[:4], state_before, strict=True):
        assert_nested_equal(item.state_dict(), state)
    assert torch.equal(target[4].get_state(), generator_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert_nested_equal(torch.cuda.get_rng_state_all(), cuda_before)
    assert_nested_equal(np.random.get_state(), numpy_before)
    assert random.getstate() == python_before


def test_generic_historical_resume_still_reads_v3_without_expected_binding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, make_objects(), normalized=False)

    load_checkpoint(path, make_objects(seed=29), normalized=False)


def make_cli_case(tmp_path: Path) -> tuple[Path, Path]:
    data_path = tmp_path / "tiny.h5"
    with h5py.File(data_path, "w") as h5:
        h5["tensor"] = np.ones((2, 160, 5, 5), dtype=np.float32)
        h5["nu"] = np.full((2, 5, 5), 3000.0, dtype=np.float32)
        source = np.zeros((2, 5, 5), dtype=np.float32)
        source[:, 2, 2] = 1.0
        h5["source_mask"] = source
        h5["t-coordinate"] = np.linspace(0.0, 1.0, 160)
        h5["x-coordinate"] = np.linspace(0.0, 0.004, 5)
        h5["y-coordinate"] = np.linspace(0.0, 0.004, 5)
        h5["model_type"] = np.array([b"uniform", b"layered"])
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": [0], "val": [1], "test": []}))
    stats_path = tmp_path / "normalization.json"
    stats_path.write_text(
        json.dumps(
            {
                "computed_from_split": "train",
                "velocity": {"mean": 3000.0, "std": 500.0},
                "wavefield": {"mean": 0.2, "std": 0.5},
                "eps": 1.0e-6,
            }
        )
    )
    config = {
        "seed": 11,
        "data": {"path": str(data_path), "split_manifest": str(split_path)},
        "normalization": {
            "contract": NORMALIZATION_CONTRACT,
            "stats_path": str(stats_path),
        },
        "sampling": {
            "target_height": 5,
            "target_width": 5,
            "global_size": 3,
            "mixture": [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],
            "tile_size": 2,
        },
        "model": {
            "global_in_features": 6,
            "native_in_channels": 5,
            "spatial_width": 2,
            "spatial_modes": 1,
            "temporal_modes": 2,
            "local_dim": 2,
            "fusion_dim": 2,
            "halo_size": 3,
            "spatial_layers": 1,
        },
        "receiver": {
            "x_start_m": 0,
            "x_stop_m": 4,
            "x_stride_m": 2,
            "z_m": [0],
        },
        "train": {
            "query_sites_per_scene": 3,
            "query_chunk_size": 2,
            "patch_size": 3,
            "patch_centers_per_scene": 1,
            "grad_clip": 1.0,
            "amp": False,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "phases": [
                {"name": "field_pretrain", "updates": 2, "init_from": "random"}
            ],
        },
        "loss": {
            "hh_reweight": True,
            "receiver_weight": 0.0,
            "phase_weight": 0.0,
            "local_spectrum_weight": 0.0,
            "energy_weight": 0.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, stats_path


def test_train_cli_writes_v4_and_rejects_changed_stats_on_resume(
    tmp_path: Path,
) -> None:
    config_path, stats_path = make_cli_case(tmp_path)
    output = tmp_path / "run"
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "scripts/train_ais_mqfno.py",
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
        "--device",
        "cpu",
        "--max-train-batches",
        "1",
        "--max-val-batches",
        "1",
    ]
    first = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=120
    )
    assert first.returncode == 0, first.stderr
    checkpoint = output / "checkpoints/last.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["schema_version"] == 4
    assert payload["normalization_contract"] == NORMALIZATION_CONTRACT
    assert payload["normalization_stats_sha256"] == hashlib.sha256(
        stats_path.read_bytes()
    ).hexdigest()

    stats_path.write_bytes(stats_path.read_bytes() + b"\n")
    resumed = subprocess.run(
        [*command, "--resume", str(checkpoint)],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert resumed.returncode != 0
    assert "normalization" in resumed.stderr.lower()


def test_train_cli_requires_normalization_before_creating_output(tmp_path: Path) -> None:
    config_path, _ = make_cli_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    del config["normalization"]
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "missing-normalization"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/train_ais_mqfno.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--max-train-batches",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert process.returncode != 0
    assert "normalization" in process.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["split", "runtime_seed"])
def test_init_preflight_rejects_binding_before_rng_output_or_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from scripts import train_ais_mqfno

    config_path, _ = make_cli_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["train"]["phases"][0]["init_from"] = "external_checkpoint"
    config_path.write_text(yaml.safe_dump(config))
    split_hash = hashlib.sha256(
        Path(config["data"]["split_manifest"]).read_bytes()
    ).hexdigest()
    checkpoint = tmp_path / "init.pt"
    save_checkpoint(checkpoint, make_objects(), normalized=True)
    payload = torch.load(checkpoint, weights_only=True)
    payload["split_manifest_sha256"] = split_hash
    payload["runtime_seed"] = int(config["seed"])
    payload["normalization_stats_sha256"] = hashlib.sha256(
        Path(config["normalization"]["stats_path"]).read_bytes()
    ).hexdigest()
    if mutation == "split":
        payload["split_manifest_sha256"] = "d" * 64
    else:
        payload["runtime_seed"] += 1
    torch.save(payload, checkpoint)
    model_load_called = False

    def forbidden_model_load(*args: object, **kwargs: object) -> None:
        nonlocal model_load_called
        model_load_called = True
        raise AssertionError("model load must not run before checkpoint preflight")

    monkeypatch.setattr(train_ais_mqfno, "_load_model_weights", forbidden_model_load)
    torch_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    output = tmp_path / "init-output"

    with pytest.raises(ValueError, match="split|runtime seed"):
        train_ais_mqfno.main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
                "--device",
                "cpu",
                "--init-checkpoint",
                str(checkpoint),
                "--max-train-batches",
                "1",
            ]
        )

    assert model_load_called is False
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert_nested_equal(torch.cuda.get_rng_state_all(), cuda_before)
    assert_nested_equal(np.random.get_state(), numpy_before)
    assert random.getstate() == python_before
    assert not output.exists()


def test_preloaded_resume_snapshot_reads_once_and_survives_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import train_ais_mqfno

    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, make_objects(), normalized=True)
    original_read_bytes = Path.read_bytes
    original_torch_load = torch.load
    reads = 0
    deserializations = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == checkpoint:
            reads += 1
        return original_read_bytes(path)

    def counted_torch_load(*args: object, **kwargs: object) -> object:
        nonlocal deserializations
        deserializations += 1
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(train_ais_mqfno.torch, "load", counted_torch_load)
    snapshot = train_ais_mqfno.load_training_checkpoint_snapshot(checkpoint)
    checkpoint.write_bytes(b"replaced after immutable snapshot")
    target = make_objects(seed=29)
    model, optimizer, scheduler, sampler, generator = target

    load_query_checkpoint(
        payload=snapshot.payload,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        data_generator=generator,
        expected_config_sha256=CONFIG_HASH,
        expected_split_manifest_sha256=SPLIT_HASH,
        expected_runtime_seed=17,
        expected_normalization_stats_sha256=NORMALIZATION_HASH,
        expected_normalization_contract=NORMALIZATION_CONTRACT,
    )

    assert reads == 1
    assert deserializations == 1


@pytest.mark.parametrize("mutation", ["legacy_v3", "changed_hash", "changed_contract"])
def test_evaluation_cli_rejects_non_v4_or_changed_binding_before_data_and_output(
    tmp_path: Path, mutation: str
) -> None:
    from scripts.train_ais_mqfno import _config_sha256

    config_path, _ = make_cli_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["data"]["path"] = str(tmp_path / "must-not-be-opened.h5")
    config_path.write_text(yaml.safe_dump(config))
    split_path = Path(config["data"]["split_manifest"])
    checkpoint_path = tmp_path / f"{mutation}.pt"
    payload = {
        "schema_version": 3 if mutation == "legacy_v3" else 4,
        "config_sha256": _config_sha256(config),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "normalization_stats_sha256": hashlib.sha256(
            Path(config["normalization"]["stats_path"]).read_bytes()
        ).hexdigest(),
        "normalization_contract": NORMALIZATION_CONTRACT,
        "model_state_dict": {},
    }
    if mutation == "changed_hash":
        payload["normalization_stats_sha256"] = "d" * 64
    elif mutation == "changed_contract":
        payload["normalization_contract"] = "ais_normalization_v1"
    torch.save(payload, checkpoint_path)
    output = tmp_path / "evaluation-output"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_ais_mqfno.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "val",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--max-samples",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert process.returncode != 0
    assert "normalization" in process.stderr.lower()
    assert "must-not-be-opened" not in process.stderr
    assert not output.exists()


def test_evaluation_cli_resolves_config_paths_from_repo_when_called_from_tmp(
    tmp_path: Path,
) -> None:
    from scripts.train_ais_mqfno import _config_sha256

    root = Path(__file__).resolve().parents[1]
    config_path, stats_path = make_cli_case(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    split_path = Path(config["data"]["split_manifest"])
    for section, key in (
        ("data", "path"),
        ("data", "split_manifest"),
        ("normalization", "stats_path"),
    ):
        config[section][key] = os.path.relpath(config[section][key], root)
    config_path.write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / "invalid-model-state.pt"
    torch.save(
        {
            "schema_version": 4,
            "config_sha256": _config_sha256(config),
            "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "normalization_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "normalization_contract": NORMALIZATION_CONTRACT,
            "model_state_dict": {},
        },
        checkpoint,
    )

    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/evaluate_ais_mqfno.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert process.returncode != 0
    assert "model_state_dict" in process.stderr
    assert "No such file" not in process.stderr


def test_legacy_raw_b1_flag_rejects_every_unpinned_checkpoint(tmp_path: Path) -> None:
    config_path, _ = make_cli_case(tmp_path)
    checkpoint = tmp_path / "not-pinned.pt"
    torch.save({"schema_version": 3}, checkpoint)
    output = tmp_path / "legacy-output"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_ais_mqfno.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--legacy-raw-b1-baseline",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=120,
    )

    assert process.returncode != 0
    assert "pinned legacy b1" in process.stderr.lower()
    assert not output.exists()


@pytest.mark.skipif(
    not PINNED_B1_PATH.exists(), reason="requires local pinned B1 checkpoint artifact"
)
def test_legacy_raw_b1_preflight_accepts_only_repository_pinned_artifact() -> None:
    from scripts.evaluate_ais_mqfno import preflight_ais_evaluation_checkpoint

    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/ais_mqfno_64x160_b1_uniform.yaml"
    config = yaml.safe_load(config_path.read_text())
    split_path = root / config["data"]["split_manifest"]
    binding = preflight_ais_evaluation_checkpoint(
        PINNED_B1_PATH,
        config,
        hashlib.sha256(split_path.read_bytes()).hexdigest(),
        legacy_raw_b1_baseline=True,
    )

    assert binding.checkpoint_sha256 == (
        "91546ba21c31e0e875c0ed7051068c279b455a94f38206274094e1a9a9233653"
    )
    assert binding.legacy_raw_b1 is True
    assert binding.normalization is None
    assert not hasattr(binding, "optimizer_state_dict")


def test_evaluation_checkpoint_binding_requires_normalization_v2() -> None:
    from scripts.evaluate_ais_mqfno import validate_checkpoint_binding

    payload = {
        "schema_version": 4,
        "config_sha256": CONFIG_HASH,
        "split_manifest_sha256": SPLIT_HASH,
        "normalization_stats_sha256": NORMALIZATION_HASH,
        "normalization_contract": NORMALIZATION_CONTRACT,
        "model_state_dict": {"weight": torch.ones(1)},
    }
    state = validate_checkpoint_binding(
        payload,
        CONFIG_HASH,
        SPLIT_HASH,
        NORMALIZATION_HASH,
        NORMALIZATION_CONTRACT,
    )
    assert torch.equal(state["weight"], torch.ones(1))
    with pytest.raises(ValueError, match="normalization"):
        validate_checkpoint_binding(
            payload,
            CONFIG_HASH,
            SPLIT_HASH,
            "d" * 64,
            NORMALIZATION_CONTRACT,
        )


def test_evaluation_adapter_uses_normalized_inputs_and_decodes_output() -> None:
    from scripts.evaluate_ais_mqfno import NormalizedQueryPredictorAdapter

    class RecordingModel(torch.nn.Module):
        def encode_global(self, global_inputs, time_s):
            self.global_inputs = global_inputs.detach().clone()
            return global_inputs.mean(dim=(1, 2, 4))

        def decode_queries(self, context, native_static, query_xz, time_s):
            self.native_static = native_static.detach().clone()
            return torch.full(
                (query_xz.shape[0], query_xz.shape[1], 160),
                0.8,
                device=query_xz.device,
            )

    scene = QueryScene(
        0,
        torch.zeros(3, 3, 160),
        torch.full((3, 3), 3000.0),
        torch.nn.functional.pad(torch.ones(1, 1), (1, 1, 1, 1)),
        torch.linspace(0.0, 1.0, 160, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        torch.linspace(0.0, 2.0, 3, dtype=torch.float64),
        {},
    )
    binding = AISNormalizationBinding(
        Path("unused.json"),
        NORMALIZATION_HASH,
        3000.0,
        500.0,
        0.2,
        0.5,
        1.0e-6,
    )
    model = RecordingModel()

    prediction = NormalizedQueryPredictorAdapter(
        model, torch.device("cpu"), 9, binding, global_size=3
    ).predict_scene(scene)

    assert torch.equal(model.native_static[:, 0], torch.zeros(1, 3, 3))
    assert torch.equal(model.global_inputs[..., 0], torch.zeros(1, 3, 3, 160))
    assert torch.allclose(prediction.field_cpu, torch.full((1, 3, 3, 160), 0.6))
