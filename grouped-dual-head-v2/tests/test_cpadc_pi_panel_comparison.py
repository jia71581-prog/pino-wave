import json
from pathlib import Path

import pytest
import torch

from scripts.aggregate_cpadc_vs_pi_panel import _nearest_rank
from scripts.evaluate_cpadc_on_pi_panel import _relative, _terms, load_pi_panel


def _pi_worker(path: Path, shard: int, measurements: list[dict]) -> Path:
    payload = {
        "schema": "frozen_pi_deeponet_split_worker_v1",
        "status": "complete",
        "split": "validation",
        "frames_per_record": 32,
        "shard_index": shard,
        "num_shards": 4,
        "checkpoint": {
            "sha256": "e2d6d7a9ec5481268bae2f41cff2400ee78b93a3a9216b2c338db572d405c4c1"
        },
        "bindings": {"manifest_digest": "manifest"},
        "measurements": measurements,
    }
    path.write_text(json.dumps(payload))
    return path


def test_terms_select_only_registered_exact_frames():
    target = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    prediction = target.clone()
    prediction[:, 1] += 2.0
    values = _terms(prediction, target, [0, 2])
    assert values[0] == 0.0
    assert _relative(values) == 0.0
    selected = _terms(prediction, target, [1])
    assert selected[0] == pytest.approx(16.0)


def test_load_pi_panel_checks_complete_shards(tmp_path):
    paths = []
    for shard in range(4):
        rows = []
        for offset in range(2):
            sample = shard * 2 + offset
            rows.append(
                {
                    "sample_id": f"validation_uniform_{sample:05d}",
                    "source_index": sample,
                    "family": "uniform",
                    "time_indices": list(range(32)),
                    "error_numerator": 1.0,
                    "truth_denominator": 4.0,
                    "relative_l2": 0.5,
                }
            )
        paths.append(_pi_worker(tmp_path / f"worker_{shard}.json", shard, rows))
    panel, metadata = load_pi_panel(paths, split="validation", expected_records=8)
    assert len(panel) == 8
    assert metadata["manifest_digest"] == "manifest"
    assert panel["validation_uniform_00007"]["time_indices"] == list(range(32))


def test_nearest_rank_uses_noninterpolated_p95():
    assert _nearest_rank(list(range(1, 21)), 0.95) == 19.0
