import numpy as np
import torch
import h5py

from saved_time_phase_operator_v4.b2h_training import (
    SequenceWindowDataset,
    relative_window_loss,
    summarize_metric_rows,
)


def test_relative_window_loss_is_zero_for_exact_prediction():
    target = torch.randn(2, 4, 1, 8, 8) * 1.0e-7
    assert relative_window_loss(
        target.clone(), target, energy_floor_fraction=0.05
    ).item() == 0.0


def test_relative_window_loss_is_scale_invariant():
    target = torch.randn(2, 4, 1, 8, 8)
    prediction = 0.8 * target
    first = relative_window_loss(
        prediction, target, energy_floor_fraction=0.05
    )
    second = relative_window_loss(
        1.0e-7 * prediction, 1.0e-7 * target, energy_floor_fraction=0.05
    )
    assert torch.allclose(first, second, atol=1.0e-6, rtol=1.0e-5)


def test_metric_summary_uses_mean_record_relative_and_energy_bins():
    rows = [
        {
            "family": "uniform",
            "relative_l2": 0.1,
            "phase_sum": 1.0,
            "phase_count": 1,
            "bins": {"early": [1.0, 100.0]},
        },
        {
            "family": "layered",
            "relative_l2": 0.3,
            "phase_sum": 0.5,
            "phase_count": 1,
            "bins": {"early": [3.0, 300.0]},
        },
    ]
    result = summarize_metric_rows(rows)
    assert np.isclose(result["aggregate_relative_l2"], 0.2)
    assert np.isclose(result["time_bin_relative_l2"]["early"], 0.1)
    assert np.isclose(result["phase_correlation"], 0.75)


def test_dataset_returns_exact_source_parameters(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("split", data=np.asarray([b"validation"] * 3))
        h5.create_dataset(
            "medium_type",
            data=np.asarray([b"uniform", b"layered", b"marmousi"]),
        )
        h5.create_dataset("wavefield", data=np.zeros((3, 5, 9, 9), np.float32))
        h5.create_dataset("velocity_mps", data=np.ones((3, 9, 9), np.float32))
        source_map = np.zeros((3, 9, 9), np.float32)
        source_map[:, 4, 4] = 1.0
        h5.create_dataset("source_map", data=source_map)
        h5.create_dataset("source_wavelet", data=np.zeros((3, 5), np.float32))
        h5.create_dataset("source_f0_hz", data=np.asarray([10.0, 20.0, 30.0]))
        h5.create_dataset("source_t0_s", data=np.asarray([0.15, 0.075, 0.05]))
        h5.create_dataset("source_amplitude", data=np.ones(3))
        h5.create_dataset("time_s", data=np.arange(5) * 0.0025)
    dataset = SequenceWindowDataset(
        str(path),
        split="validation",
        rollout_steps=2,
        seed=1,
        records_per_family=1,
        fixed_starts=(0,),
    )
    item = dataset[0]
    assert tuple(item["source_parameters"].shape) == (3,)
    assert item["initial_time_s"].item() == np.float32(0.0025)


def test_dataset_history_preserves_the_same_prediction_origin(tmp_path):
    path = tmp_path / "history.h5"
    wavefield = np.arange(3 * 6 * 9 * 9, dtype=np.float32).reshape(
        3, 6, 9, 9
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("split", data=np.asarray([b"validation"] * 3))
        h5.create_dataset(
            "medium_type",
            data=np.asarray([b"uniform", b"layered", b"marmousi"]),
        )
        h5.create_dataset("wavefield", data=wavefield)
        h5.create_dataset("velocity_mps", data=np.ones((3, 9, 9), np.float32))
        h5.create_dataset("source_map", data=np.zeros((3, 9, 9), np.float32))
        h5.create_dataset("source_wavelet", data=np.zeros((3, 6), np.float32))
        h5.create_dataset("source_f0_hz", data=np.ones(3) * 10.0)
        h5.create_dataset("source_t0_s", data=np.ones(3) * 0.15)
        h5.create_dataset("source_amplitude", data=np.ones(3))
        h5.create_dataset("time_s", data=np.arange(6) * 0.0025)
    dataset = SequenceWindowDataset(
        str(path),
        split="validation",
        rollout_steps=2,
        seed=1,
        records_per_family=1,
        history_steps=1,
        fixed_starts=(1,),
    )
    item = dataset[0]
    record = item["record"]
    assert torch.equal(item["history"][0], torch.from_numpy(wavefield[record, 0]))
    assert torch.equal(item["p0"][0], torch.from_numpy(wavefield[record, 1]))
    assert torch.equal(item["p1"][0], torch.from_numpy(wavefield[record, 2]))
    assert torch.equal(item["target"][0, 0], torch.from_numpy(wavefield[record, 3]))
