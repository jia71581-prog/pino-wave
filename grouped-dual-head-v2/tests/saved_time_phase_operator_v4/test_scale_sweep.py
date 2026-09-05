from types import SimpleNamespace

import torch

from scripts.evaluate_saved_time_v6_scale_sweep import (
    prepare_dense_grid_for_micro,
    scaled_correction_prediction,
    update_scale_metric_accumulators,
)


def test_scaled_correction_prediction_rescales_only_the_learned_delta():
    coarse = torch.tensor([[[[2.0, -1.0]]]])
    prediction = torch.tensor([[[[3.0, 1.0]]]])

    scaled = scaled_correction_prediction(prediction, coarse, scale=2.5)

    assert torch.equal(scaled, torch.tensor([[[[4.5, 4.0]]]]))
    assert torch.equal(scaled_correction_prediction(prediction, coarse, scale=0.0), coarse)


def test_scaled_correction_prediction_rejects_negative_scale():
    coarse = torch.zeros(1, 1, 1, 1)
    prediction = torch.ones_like(coarse)

    try:
        scaled_correction_prediction(prediction, coarse, scale=-1.0)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative correction scale must be rejected")


def test_prepare_dense_grid_for_micro_forwards_cached_eikonal_travel_time():
    class RecordingModel:
        def prepare_dense_grid(self, prepared, **kwargs):
            self.prepared = prepared
            self.kwargs = kwargs
            return "dense-grid"

    model = RecordingModel()
    prepared = object()
    travel = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    tensors = {
        "x_m": torch.tensor([0.0, 1.0, 2.0]),
        "z_m": torch.tensor([0.0, 1.0]),
    }

    result = prepare_dense_grid_for_micro(
        model,
        prepared,
        tensors,
        SimpleNamespace(dense_travel_time_s=travel),
        torch.device("cpu"),
    )

    assert result == "dense-grid"
    assert model.prepared is prepared
    assert model.kwargs["travel_time_s"] is travel
    assert torch.equal(model.kwargs["x_m"], tensors["x_m"])
    assert torch.equal(model.kwargs["z_m"], tensors["z_m"])


def test_prepare_dense_grid_for_micro_preserves_ray_fallback_without_cache():
    class RecordingModel:
        def prepare_dense_grid(self, prepared, **kwargs):
            self.kwargs = kwargs
            return kwargs

    model = RecordingModel()
    tensors = {"x_m": torch.tensor([0.0]), "z_m": torch.tensor([0.0])}
    result = prepare_dense_grid_for_micro(
        model,
        object(),
        tensors,
        SimpleNamespace(dense_travel_time_s=None),
        torch.device("cpu"),
    )

    assert result["travel_time_s"] is None


def test_scale_metric_update_forwards_exact_candidate_and_record_metadata():
    class Recorder:
        def update(self, prediction, target, **metadata):
            self.prediction = prediction.clone()
            self.target = target.clone()
            self.metadata = metadata

    coarse = torch.tensor([[[[1.0, 2.0]]]])
    prediction = torch.tensor([[[[2.0, 4.0]]]])
    target = torch.tensor([[[[3.0, 6.0]]]])
    record = SimpleNamespace(
        medium_type=("layered",),
        group_id=("medium-1",),
        sample_id=("sample-1",),
        left_index=torch.tensor([[17]]),
    )
    accumulators = {0.0: Recorder(), 2.0: Recorder()}

    update_scale_metric_accumulators(
        accumulators,
        prediction,
        coarse,
        target,
        micro=record,
        onset_indices=(11,),
    )

    torch.testing.assert_close(accumulators[0.0].prediction, coarse)
    torch.testing.assert_close(
        accumulators[2.0].prediction,
        coarse + 2.0 * (prediction - coarse),
    )
    assert accumulators[2.0].metadata == {
        "families": record.medium_type,
        "group_ids": record.group_id,
        "sample_ids": record.sample_id,
        "time_indices": record.left_index,
        "source_onset_indices": (11,),
    }
