from __future__ import annotations

import pytest
import torch

from saved_time_phase_operator_v4.instance_adaptation.contracts import (
    SnapshotAccessAudit,
    future_indices,
    onset_indices,
)


def test_onset_indices_are_exactly_first_two_saved_frames_after_t0():
    axis = torch.tensor([0.00, 0.01, 0.02, 0.03])
    assert onset_indices(axis, 0.011, 30.0, lead_cycles=0.0) == (2, 3)


def test_rejects_k1_out_of_range():
    with pytest.raises(ValueError, match="two onset frames"):
        onset_indices(torch.tensor([0.0]), 0.0, 30.0, lead_cycles=0.0)


def test_future_indices_are_strictly_after_k1():
    assert future_indices(6, (2, 3)).tolist() == [4, 5]


def test_snapshot_access_audit_rejects_later_true_frame():
    audit = SnapshotAccessAudit(allowed_indices=(2, 3))
    audit.read((2, 3))
    with pytest.raises(PermissionError, match="future truth"):
        audit.read((4,))
