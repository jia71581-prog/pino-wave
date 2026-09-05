from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.multifidelity import (
    NumericalTeacherCache,
    fixed_teacher_time_indices,
    multifidelity_distillation_block_loss,
    multifidelity_energy_reference,
    numerical_teacher_pool_indices,
    residual_adaptive_teacher_pool_indices,
)
from saved_time_phase_operator_v4.background_field import BackgroundFieldProvider


def _write_cache(path: Path) -> tuple[str, ...]:
    sample_ids = ("train_uniform_00000", "train_layered_00000")
    time_indices = np.asarray([0, 4, 9, 15], dtype=np.int64)
    values = np.arange(2 * 4 * 3 * 2, dtype=np.float32).reshape(2, 4, 3, 2)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "lwc84_multifidelity_teacher_v1"
        handle.attrs["status"] = "complete"
        handle.attrs["source_manifest_sha256"] = "manifest-17"
        handle.create_dataset("sample_id", data=np.asarray(sample_ids, dtype="S32"))
        handle.create_dataset("source_index", data=np.asarray([7, 11], dtype=np.int64))
        handle.create_dataset("time_indices", data=time_indices)
        handle.create_dataset("time_s", data=time_indices.astype(np.float64) * 0.0025)
        handle.create_dataset("wavefield", data=values)
    return sample_ids


def test_fixed_teacher_time_indices_span_complete_exact_axis():
    indices = fixed_teacher_time_indices(stored_time_count=401, count=64)

    assert len(indices) == len(set(indices)) == 64
    assert indices[0] == 0
    assert indices[-1] == 400
    assert all(left < right for left, right in zip(indices, indices[1:]))


def test_teacher_pool_rotates_deterministically_and_covers_pool():
    pool = fixed_teacher_time_indices(stored_time_count=401, count=64)
    selections = [
        numerical_teacher_pool_indices(
            pool,
            sample_id="train_marmousi_00003",
            appearance=appearance,
            seed=372,
            count=16,
        )
        for appearance in range(4)
    ]

    assert all(len(row) == 16 for row in selections)
    assert all(tuple(sorted(row)) == row for row in selections)
    assert set().union(*(set(row) for row in selections)) == set(pool)
    assert selections[0] == numerical_teacher_pool_indices(
        pool,
        sample_id="train_marmousi_00003",
        appearance=0,
        seed=372,
        count=16,
    )


def test_residual_adaptive_teacher_pool_prefers_high_residual_times():
    pool = fixed_teacher_time_indices(stored_time_count=401, count=128)
    high = set(pool[-16:])
    probabilities = np.asarray(
        [20.0 if value in high else 1.0 for value in pool], dtype=np.float64
    )
    selections = [
        residual_adaptive_teacher_pool_indices(
            pool,
            sample_id=f"sample-{sample}",
            appearance=appearance,
            seed=372,
            count=32,
            probabilities=probabilities,
        )
        for sample in range(32)
        for appearance in range(4)
    ]

    high_fraction = sum(len(set(row) & high) for row in selections) / (
        len(selections) * len(selections[0])
    )
    assert high_fraction > 0.35
    assert all(len(row) == len(set(row)) == 32 for row in selections)
    assert selections[0] == residual_adaptive_teacher_pool_indices(
        pool,
        sample_id="sample-0",
        appearance=0,
        seed=372,
        count=32,
        probabilities=probabilities,
    )


def test_numerical_teacher_cache_reads_sample_specific_exact_times(tmp_path):
    path = tmp_path / "teacher.h5"
    sample_ids = _write_cache(path)
    cache = NumericalTeacherCache(
        path,
        expected_source_manifest_sha256="manifest-17",
        expected_sample_ids=sample_ids,
    )

    values = cache.read(
        (sample_ids[1], sample_ids[0]),
        torch.tensor([[0, 9], [4, 15]], dtype=torch.long),
    )

    with h5py.File(path, "r") as handle:
        expected = np.stack(
            [handle["wavefield"][1, [0, 2]], handle["wavefield"][0, [1, 3]]]
        )
    assert torch.equal(values, torch.from_numpy(expected))
    assert values.dtype == torch.float32


def test_numerical_teacher_cache_rejects_identity_or_unregistered_time(tmp_path):
    path = tmp_path / "teacher.h5"
    sample_ids = _write_cache(path)

    with pytest.raises(ValueError, match="manifest"):
        NumericalTeacherCache(path, expected_source_manifest_sha256="wrong")

    cache = NumericalTeacherCache(path, expected_sample_ids=sample_ids)
    with pytest.raises(KeyError, match="time"):
        cache.read((sample_ids[0],), torch.tensor([[3]], dtype=torch.long))


def test_background_provider_routes_union_of_full_and_sparse_caches(tmp_path):
    full = tmp_path / "full.h5"
    sparse = tmp_path / "sparse.h5"

    def write(path, sample_id, time_indices, offset):
        times = np.asarray(time_indices, dtype=np.int64)
        values = (
            np.arange(len(times) * 3 * 2, dtype=np.float32).reshape(1, len(times), 3, 2)
            + float(offset)
        )
        with h5py.File(path, "w") as handle:
            handle.attrs["schema"] = "lwc84_multifidelity_teacher_v1"
            handle.attrs["status"] = "complete"
            handle.attrs["source_manifest_sha256"] = ""
            handle.create_dataset("sample_id", data=np.asarray([sample_id], dtype="S32"))
            handle.create_dataset("source_index", data=np.asarray([offset], dtype=np.int64))
            handle.create_dataset("time_indices", data=times)
            handle.create_dataset("time_s", data=times.astype(np.float64) * 0.0025)
            handle.create_dataset("wavefield", data=values)
        return values

    full_values = write(full, "existing", [0, 4, 9, 15], 10)
    sparse_values = write(sparse, "missing", [0, 9, 15], 20)
    provider = BackgroundFieldProvider((full, sparse))

    assert provider.sample_ids == ("existing", "missing")
    assert provider.time_indices == (0, 9, 15)
    assert provider.covers(("existing", "missing"), (0, 9, 15))
    assert not provider.covers(("missing",), (4,))
    values = provider.physical(
        ("existing", "missing"),
        torch.tensor([[15, 0, 15], [9, 0, 15]], dtype=torch.long),
    )
    torch.testing.assert_close(values[0, 0], torch.from_numpy(full_values[0, 3]))
    torch.testing.assert_close(values[0, 1], torch.from_numpy(full_values[0, 0]))
    torch.testing.assert_close(values[1, 0], torch.from_numpy(sparse_values[0, 1]))
    torch.testing.assert_close(values[1, 2], torch.from_numpy(sparse_values[0, 2]))


def test_multifidelity_loss_is_exactly_time_decomposable_and_targets_detached():
    generator = torch.Generator().manual_seed(17)
    high = torch.randn(2, 5, 4, 3, generator=generator, requires_grad=True)
    low = torch.randn(2, 5, 4, 3, generator=generator, requires_grad=True)
    prediction_full = torch.randn(
        2, 5, 4, 3, generator=generator, requires_grad=True
    )
    coarse_full = torch.randn(2, 5, 4, 3, generator=generator, requires_grad=True)
    prediction_block = prediction_full.detach().clone().requires_grad_(True)
    coarse_block = coarse_full.detach().clone().requires_grad_(True)
    reference = multifidelity_energy_reference(
        high, low, residual_energy_floor_fraction=0.1
    )

    full = multifidelity_distillation_block_loss(
        prediction_full,
        coarse_full,
        high,
        low,
        reference=reference,
        low_fidelity_weight=0.5,
        residual_weight=0.25,
    )
    blocked = sum(
        multifidelity_distillation_block_loss(
            prediction_block[:, start:stop],
            coarse_block[:, start:stop],
            high[:, start:stop],
            low[:, start:stop],
            reference=reference,
            low_fidelity_weight=0.5,
            residual_weight=0.25,
        ).total
        for start, stop in ((0, 2), (2, 4), (4, 5))
    )

    assert torch.allclose(blocked, full.total, rtol=1.0e-6, atol=1.0e-7)
    full.total.backward()
    blocked.backward()
    assert torch.allclose(
        prediction_block.grad, prediction_full.grad, rtol=1.0e-6, atol=1.0e-7
    )
    assert torch.allclose(
        coarse_block.grad, coarse_full.grad, rtol=1.0e-6, atol=1.0e-7
    )
    assert high.grad is None
    assert low.grad is None


def test_multifidelity_residual_reference_uses_registered_high_energy_floor():
    high = torch.ones(1, 2, 3, 4)
    low = high.clone()
    reference = multifidelity_energy_reference(
        high, low, residual_energy_floor_fraction=0.2
    )

    assert torch.allclose(reference.residual_energy, 0.2 * reference.high_energy)
