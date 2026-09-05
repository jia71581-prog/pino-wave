from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_marmousi_multisource_sealed import (
    _load_verified_predictions,
    _atomic_json,
    _atomic_npz,
)
from saved_time_phase_operator_v4.evaluation import sha256_file


def test_predict_function_has_no_target_reader_call() -> None:
    source_path = Path("scripts/evaluate_marmousi_multisource_sealed.py")
    tree = ast.parse(source_path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "predict")
    calls = [
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert "read_wavefield" not in calls


def test_prediction_seal_detects_changed_bytes(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.npz"
    _atomic_npz(prediction, prediction_tzx=np.zeros((4, 2, 2), dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    _atomic_json(
        {
            "status": "complete",
            "truth_wavefield_access": False,
            "records": [
                {
                    "sample_id": "sample",
                    "prediction_path": str(prediction),
                    "prediction_sha256": sha256_file(prediction),
                }
            ],
        },
        manifest,
    )
    payload, arrays = _load_verified_predictions(manifest)
    assert payload["status"] == "complete"
    assert arrays["sample"]["prediction_tzx"].shape == (4, 2, 2)
    with prediction.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="seal mismatch"):
        _load_verified_predictions(manifest)
