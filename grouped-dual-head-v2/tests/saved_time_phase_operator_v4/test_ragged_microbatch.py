from types import SimpleNamespace

import torch

from grouped_ufno_mionet_v3.data.pilot import PilotBatch
from saved_time_phase_operator_v4.data import split_pilot_batch
from scripts.train_saved_time_v4_full_support import _microbatch_record_weights


def _batch(records: int = 12) -> PilotBatch:
    mapping = torch.arange(records, dtype=torch.long)
    return PilotBatch(
        step=0,
        velocity_mps=torch.zeros(records, 1, 2, 2),
        record_to_medium=mapping,
        source_parameters=torch.zeros(records, 6),
        source_map=torch.zeros(records, 1, 2, 2),
        requested_time_s=torch.zeros(records, 2),
        dense_target_physical=torch.zeros(records, 2, 2, 2),
        target_exact=torch.ones(records, 2, dtype=torch.bool),
        interpolation_alpha=torch.zeros(records, 2),
        left_index=torch.zeros(records, 2, dtype=torch.long),
        right_index=torch.zeros(records, 2, dtype=torch.long),
        query_coords=torch.zeros(records, 1, 3),
        query_target_physical=torch.zeros(records, 1),
        query_probability=torch.ones(records, 1),
        x_m=torch.arange(2, dtype=torch.float32),
        z_m=torch.arange(2, dtype=torch.float32),
        sample_id=tuple(f"sample-{index}" for index in range(records)),
        group_id=tuple(f"group-{index}" for index in range(records)),
        medium_type=("uniform",) * records,
    )


def test_ragged_microbatch_uses_largest_safe_physical_batch():
    batch = _batch()

    pieces = split_pilot_batch(batch, microbatch_records=5)

    assert tuple(len(piece.sample_id) for piece in pieces) == (5, 5, 2)
    assert tuple(value for piece in pieces for value in piece.sample_id) == batch.sample_id


def test_ragged_microbatch_loss_weights_preserve_per_record_mean():
    pieces = tuple(
        SimpleNamespace(sample_id=("x",) * size)
        for size in (5, 5, 2)
    )

    assert _microbatch_record_weights(pieces) == (5 / 12, 5 / 12, 2 / 12)
