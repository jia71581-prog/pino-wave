import json
import os

import torch

from scripts.select_current_helmholtz_parent import select_current_parent


def _complete_run(path, *, step, mtime):
    path.mkdir(parents=True)
    checkpoint = path / f"update_{step:04d}.pt"
    torch.save(
        {
            "global_step": step,
            "metrics": {"triplet_aggregate_relative_l2": 0.25},
            "model_state": {},
        },
        checkpoint,
    )
    terminal = path / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "status": "complete",
                "best_checkpoint": str(checkpoint),
                "best_fixed_heldout_relative_l2": 0.25,
                "train_record_count": 3203,
                "training_splits": [
                    "train",
                    "validation",
                    "test_id",
                    "ood_canonical",
                ],
                "training_time_pool_count": 96,
                "anomaly_excluded": True,
            }
        )
    )
    os.utime(terminal, (mtime, mtime))
    return terminal


def test_parent_selector_uses_newest_complete_audited_lineage(tmp_path):
    fallback = _complete_run(tmp_path / "fallback", step=1, mtime=1)
    older = _complete_run(
        tmp_path / "helmholtz_full3203_time96_vrba_ep5_old", step=10, mtime=10
    )
    newer = _complete_run(
        tmp_path / "helmholtz_full3203_time96_vrba_ep5_new", step=20, mtime=20
    )

    assert select_current_parent(tmp_path, fallback) == newer.resolve()
    assert select_current_parent(tmp_path / "empty", fallback) == fallback.resolve()
    assert older != newer
