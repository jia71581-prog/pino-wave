import json
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnose_helmholtz_g3_heldout import (
    resolve_best_warmstart,
    select_family_balanced_pool,
    select_one_index_per_family_after_skip,
)


def _write_parent(tmp_path, *, terminal_metric=0.25, checkpoint_metric=0.25):
    checkpoint = tmp_path / "update_0017.pt"
    torch.save(
        {
            "global_step": 17,
            "metrics": {"triplet_aggregate_relative_l2": checkpoint_metric},
            "model_state": {},
        },
        checkpoint,
    )
    terminal = tmp_path / "terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "status": "complete",
                "best_checkpoint": str(checkpoint),
                "best_fixed_heldout_relative_l2": terminal_metric,
            }
        )
    )
    return terminal, checkpoint


def test_resolve_best_warmstart_uses_terminal_selected_checkpoint(tmp_path):
    terminal, checkpoint = _write_parent(tmp_path)

    selected, report = resolve_best_warmstart(terminal)

    assert selected == checkpoint.resolve()
    assert report["mode"] == "terminal_best_checkpoint"
    assert report["checkpoint_global_step"] == 17
    assert report["selection_value"] == pytest.approx(0.25)


def test_resolve_best_warmstart_rejects_terminal_checkpoint_metric_mismatch(tmp_path):
    terminal, _ = _write_parent(
        tmp_path,
        terminal_metric=0.25,
        checkpoint_metric=0.30,
    )

    with pytest.raises(ValueError, match="metric mismatch"):
        resolve_best_warmstart(terminal)


def test_family_pool_accepts_auditable_unequal_limits():
    records = [
        SimpleNamespace(split="train", medium_type=family)
        for family, count in (("uniform", 3), ("layered", 4), ("marmousi", 2))
        for _ in range(count)
    ]
    pool = select_family_balanced_pool(
        records,
        splits=("train",),
        per_family={"uniform": 2, "layered": 3, "marmousi": 1},
    )
    selected = [records[index].medium_type for index in pool]
    assert selected == ["uniform", "uniform", "layered", "layered", "layered", "marmousi"]


def test_family_pool_and_train_heldout_selection_are_disjoint():
    records = [
        SimpleNamespace(split="train", medium_type=family, sample_id=f"{family}:{index}")
        for family in ("uniform", "layered", "marmousi")
        for index in range(5)
    ]
    heldout = select_one_index_per_family_after_skip(
        records, split="train", skip_per_family=1
    )
    heldout_ids = tuple(records[index].sample_id for index in heldout)
    pool = select_family_balanced_pool(
        records,
        splits=("train",),
        per_family=3,
        excluded_sample_ids=heldout_ids,
    )
    pool_ids = {records[index].sample_id for index in pool}
    assert not pool_ids.intersection(heldout_ids)
    assert heldout_ids == ("uniform:1", "layered:1", "marmousi:1")
