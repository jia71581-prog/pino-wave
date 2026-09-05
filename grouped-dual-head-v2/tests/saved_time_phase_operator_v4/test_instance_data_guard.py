from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetRecord
from saved_time_phase_operator_v4.instance_adaptation.contracts import SnapshotAccessAudit


def _record() -> GuardedOnsetRecord:
    observed_indices = (2, 3)
    observed = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    velocity = torch.ones(1, 4, 4)
    source = torch.tensor([20.0, 10.0, 12.0, 0.011, 1.0])
    source_map = torch.zeros(1, 4, 4)
    source_map[..., 1, 1] = 1.0
    return GuardedOnsetRecord(
        velocity_mps=velocity,
        source_parameters=source,
        source_map=source_map,
        time_s=torch.arange(6, dtype=torch.float32) * 0.01,
        x_m=torch.arange(4, dtype=torch.float32),
        z_m=torch.arange(4, dtype=torch.float32),
        sample_id="sample",
        group_id="medium",
        medium_type="uniform",
        source_index=0,
        observed_indices=observed_indices,
        observed_wavefield=observed,
        input_digest="digest",
        audit=SnapshotAccessAudit(observed_indices),
    )


def test_guarded_record_exposes_metadata_and_two_frames_only():
    record = _record()
    assert record.observed_indices == (2, 3)
    assert record.observed_wavefield.shape[0] == 2
    assert set(record.public_keys) == {
        "velocity_mps", "source_parameters", "source_map", "time_s",
        "x_m", "z_m", "observed_wavefield", "observed_indices",
        "dense_travel_time_s",
    }


def test_guarded_record_cannot_read_k1_plus_one():
    record = _record()
    try:
        record.read_truth_forbidden((4,))
    except PermissionError:
        pass
    else:
        raise AssertionError("future truth access was not rejected")


def test_future_truth_replacement_does_not_change_adapter_inputs():
    left = _record()
    right = _record()
    right.observed_wavefield[0, 0, 0] = left.observed_wavefield[0, 0, 0]
    assert torch.equal(left.observed_wavefield, right.observed_wavefield)
    assert left.input_digest == right.input_digest
