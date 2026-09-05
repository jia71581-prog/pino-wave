from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.coarse_lwc84 import (
    accumulate_exact_metrics,
    decide_coarse_gate,
    seal_prediction,
    select_validation_records,
)


def _index_h5(path: Path) -> None:
    families = np.asarray(
        ["uniform", "layered", "marmousi"] * 2 + ["uniform"], dtype="S16"
    )
    splits = np.asarray(["validation"] * 6 + ["train"], dtype="S16")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("medium_type", data=families)
        handle.create_dataset("split", data=splits)
        handle.create_dataset(
            "sample_id", data=np.asarray([f"s{i}" for i in range(7)], dtype="S8")
        )
        handle.create_dataset(
            "group_id", data=np.asarray([f"g{i}" for i in range(7)], dtype="S8")
        )
        handle.create_dataset(
            "sample_sha256", data=np.asarray([f"h{i}" for i in range(7)], dtype="S8")
        )


def test_selection_is_fixed_balanced_and_validation_only(tmp_path: Path) -> None:
    path = tmp_path / "index.h5"
    _index_h5(path)
    first = select_validation_records(path, seed=17, per_family=1)
    second = select_validation_records(path, seed=17, per_family=1)
    assert first == second
    assert [row.medium_type for row in first] == ["uniform", "layered", "marmousi"]
    assert all(row.split == "validation" for row in first)
    assert len({row.group_id for row in first}) == 3


def test_exact_metrics_are_block_size_invariant() -> None:
    target = torch.arange(2 * 7 * 5 * 5, dtype=torch.float32).reshape(2, 7, 5, 5)
    prediction = target * 0.9
    metadata = {
        "families": ("uniform", "layered"),
        "group_ids": ("g0", "g1"),
        "sample_ids": ("s0", "s1"),
        "source_onset_indices": (2, 3),
    }
    one = accumulate_exact_metrics(prediction, target, block_size=1, **metadata)
    all_at_once = accumulate_exact_metrics(prediction, target, block_size=7, **metadata)
    assert one["aggregate_relative_l2"] == pytest.approx(0.1, abs=1.0e-6)
    assert one["aggregate_relative_l2"] == pytest.approx(
        all_at_once["aggregate_relative_l2"]
    )
    assert one["unique_time_index_count"] == 7


def test_prediction_is_atomically_sealed_with_content_hash(tmp_path: Path) -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    sealed = seal_prediction(
        tmp_path / "prediction.pt", tensor, metadata={"sample_id": "s0"}
    )
    assert sealed.path.is_file()
    assert sealed.byte_count == sealed.path.stat().st_size
    assert sealed.sha256 == hashlib.sha256(sealed.path.read_bytes()).hexdigest()
    payload = torch.load(sealed.path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(payload["wavefield"], tensor)
    assert payload["sealed"] is True


@pytest.mark.parametrize(
    ("aggregate", "families", "expected"),
    [
        (
            0.099,
            {"uniform": 0.11, "layered": 0.10, "marmousi": 0.119},
            "direct_baseline",
        ),
        (
            0.10,
            {"uniform": 0.20, "layered": 0.30, "marmousi": 0.44},
            "neural_corrector",
        ),
        (
            0.35,
            {"uniform": 0.20, "layered": 0.30, "marmousi": 0.44},
            "neural_corrector",
        ),
        (
            0.351,
            {"uniform": 0.20, "layered": 0.30, "marmousi": 0.44},
            "reject",
        ),
        (
            0.20,
            {"uniform": 0.20, "layered": 0.45, "marmousi": 0.30},
            "reject",
        ),
    ],
)
def test_registered_gate_boundaries(aggregate, families, expected) -> None:
    decision = decide_coarse_gate(aggregate, families)
    assert decision.action == expected
    assert decision.thresholds == {
        "direct_aggregate_lt": 0.10,
        "direct_family_lt": 0.12,
        "corrector_aggregate_lte": 0.35,
        "corrector_family_lt": 0.45,
    }
