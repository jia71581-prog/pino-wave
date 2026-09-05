from __future__ import annotations

from pathlib import Path

import pytest
import torch

from grouped_ufno_mionet_v3.config import V3Config
from scripts.smoke_grouped_v3 import run_structural_smoke


def test_cpu_structural_smoke_updates_query_dense_and_all_required_branches(tmp_path: Path):
    config = V3Config.from_yaml("configs/grouped_v3/smoke.yaml")
    report = run_structural_smoke(config, device="cpu", checkpoint_dir=tmp_path)
    assert report["device"] == "cpu"
    assert report["query_shape"] == [2, 6]
    assert report["dense_shape"] == [2, 3, 17, 17]
    assert report["loss"] > 0
    assert report["global_step"] == 1
    assert report["parameter_count"] > 0
    assert {
        "medium_spectral",
        "source_parameters",
        "travel_branch",
        "periodic_trunk",
        "query_local_residual",
        "dense_spectral",
        "dense_film",
    } <= set(report["gradient_groups"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_structural_smoke_is_finite(tmp_path: Path):
    config = V3Config.from_yaml("configs/grouped_v3/smoke.yaml")
    report = run_structural_smoke(config, device="cuda", checkpoint_dir=tmp_path)
    assert report["device"] == "cuda"
    assert report["loss"] > 0
    assert report["peak_memory_mib"] > 0
