"""Unit tests for the read-only train-only spectral capacity diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fno_acoustic.data import PinoHDF5Dataset
from scripts.diagnose_trainonly_spectral_capacity import (
    LoaderWavefieldReader,
    build_loader_config,
    parse_comma_ints,
    parse_nonnegative_comma_ints,
    per_record_cumulative_energies,
    real_coefficients_matrix,
    resolve_train_only_rows,
    rfft_coefficient_weights,
    rfft_energy_weights,
    run_diagnostic,
    select_fit_candidate,
    spatial_mask,
)


# --------------------------------------------------------------------------- #
# Small synthetic NTZX train-only VDS
# --------------------------------------------------------------------------- #
def _make_vds(
    path: Path,
    *,
    n_records: int,
    time_steps: int,
    height: int,
    width: int,
    waves: np.ndarray | None = None,
    split_values: list[str] | None = None,
    id_prefix: str = "train_",
    dt_s: float = 0.0025,
) -> Path:
    rng = np.random.default_rng(0)
    if waves is None:
        waves = rng.standard_normal((n_records, time_steps, height, width)).astype(
            np.float32
        )
    split_values = split_values or ["train"] * n_records
    with h5py.File(path, "w") as h5:
        h5.attrs["axis_order"] = "NTZX"
        h5.attrs["schema_version"] = "acoustic_lwc84_401_to_201_v1"
        h5.attrs["included_splits"] = ["train"]
        h5.attrs["saved_grid_shape"] = json.dumps([height, width])
        h5.create_dataset(
            "sample_id",
            data=np.array(
                [f"{id_prefix}{i:05d}".encode() for i in range(n_records)], dtype="S32"
            ),
        )
        h5.create_dataset(
            "split",
            data=np.array([value.encode() for value in split_values], dtype="S16"),
        )
        h5.create_dataset(
            "sample_sha256",
            data=np.array(
                [
                    hashlib.sha256(f"s{i}".encode()).hexdigest().encode()
                    for i in range(n_records)
                ],
                dtype="S64",
            ),
        )
        h5.create_dataset(
            "time_s", data=np.linspace(0.0, (time_steps - 1) * dt_s, time_steps)
        )
        h5.create_dataset("x_m", data=np.arange(width) * 10.0)
        h5.create_dataset("z_m", data=np.arange(height) * 10.0)
        h5.create_dataset("wavefield", data=waves.astype(np.float32))
    return path


def _manifest(
    *,
    scope: str = "train_only",
    fit: tuple[str, ...] = ("train_00000", "train_00001"),
    calibration: tuple[str, ...] = ("train_00002",),
    confirmation: tuple[str, ...] = ("train_00003",),
    index_rows: dict[str, list[int]] | None = None,
) -> dict:
    panels = {"train": fit, "val": calibration, "test": confirmation}
    manifest = {
        "schema": "factorized_pbg_train_only_split_v1",
        "project_data_scope": scope,
        "sample_ids": {name: list(ids) for name, ids in panels.items()},
        "validation_opened": False,
        "test_id_opened": False,
    }
    if index_rows is not None:
        manifest.update(index_rows)
    return manifest


# --------------------------------------------------------------------------- #
# rFFT single-sided energy weights (odd/even T, Parseval)
# --------------------------------------------------------------------------- #
def test_rfft_energy_weights_odd_even_and_parseval() -> None:
    rng = np.random.default_rng(7)
    np.testing.assert_array_equal(rfft_energy_weights(5), [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(rfft_energy_weights(8), [1.0, 2.0, 2.0, 2.0, 1.0])
    assert rfft_coefficient_weights(8).tolist() == [1, 2, 2, 2, 1]
    assert rfft_energy_weights(401).size == 201
    assert rfft_energy_weights(401)[0] == 1.0 and rfft_energy_weights(401)[-1] == 2.0
    for total_time in (5, 8, 9, 16, 401):
        weights = rfft_energy_weights(total_time)
        signal = rng.standard_normal(total_time)
        spectrum = np.fft.rfft(signal, norm="ortho")
        single_sided = float(np.sum(weights * np.abs(spectrum) ** 2))
        assert single_sided == pytest.approx(float(np.sum(signal**2)), rel=1e-12)


# --------------------------------------------------------------------------- #
# Full time-frequency and full spatial retention recovers the record exactly
# --------------------------------------------------------------------------- #
def test_full_frequency_full_space_retention_error_near_zero() -> None:
    rng = np.random.default_rng(3)
    height = width = 8
    total_time = 9  # odd T: rFFT has no Nyquist bin
    wave = rng.standard_normal((height, width, total_time)).astype(np.float32)
    weights = rfft_energy_weights(total_time)
    full_mask = spatial_mask(height, width, height // 2)
    cumulative, total = per_record_cumulative_energies(wave, weights, [full_mask])
    retained = float(cumulative[-1, 0])  # nf = n_rfft, k = Nyquist
    # Two orthonormal FFT passes accumulate float64 roundoff at the machine
    # epsilon level in energy, so the relative-L2 error is O(sqrt(eps)).
    assert retained == pytest.approx(total, rel=1e-9)
    error = math.sqrt(max(0.0, 1.0 - retained / total))
    assert error < 1e-6


# --------------------------------------------------------------------------- #
# Known single-frequency, low-spatial-mode retention / discard
# --------------------------------------------------------------------------- #
def test_known_single_frequency_low_spatial_mode_retention_and_discard() -> None:
    height = width = 8
    total_time = 16  # even T exercises the Nyquist weighting path
    f0 = 2
    kx0 = ky0 = 1
    t = np.arange(total_time)
    x = np.arange(height)
    z = np.arange(width)
    wave = (
        np.cos(2 * np.pi * f0 * t / total_time)[None, None, :]
        * np.cos(2 * np.pi * kx0 * x / height)[:, None, None]
        * np.cos(2 * np.pi * ky0 * z / width)[None, :, None]
    ).astype(np.float32)

    weights = rfft_energy_weights(total_time)
    cumulative, total = per_record_cumulative_energies(
        wave, weights, [spatial_mask(height, width, 1), spatial_mask(height, width, 0)]
    )

    def relative_error(nf: int, k_index: int) -> float:
        retained = float(cumulative[nf - 1, k_index])
        return math.sqrt(max(0.0, 1.0 - retained / total))

    # Keeping nf=3 time bins (DC..2) and |ky|,|kx| <= 1 retains the full mode.
    assert relative_error(3, 0) < 2e-3  # float32 storage leakage
    # Dropping time bin f0=2 (nf=2) discards the mode entirely.
    assert relative_error(2, 0) == pytest.approx(1.0, abs=1e-5)
    # Dropping the spatial mode (k=0 keeps only the DC bin) discards it.
    assert relative_error(3, 1) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# Manifest / row-metadata enforcement (train-only, disjoint panels, prefixes)
# --------------------------------------------------------------------------- #
def test_resolve_rejects_non_train_only_scope(tmp_path: Path) -> None:
    path = _make_vds(tmp_path / "v.h5", n_records=4, time_steps=9, height=8, width=8)
    manifest = _manifest(scope="full")
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="project_data_scope"):
            resolve_train_only_rows(manifest, h5, fit_limit=2, calibration_limit=1)


def test_resolve_rejects_overlapping_fit_calibration(tmp_path: Path) -> None:
    path = _make_vds(tmp_path / "v.h5", n_records=4, time_steps=9, height=8, width=8)
    manifest = _manifest(
        fit=("train_00000", "train_00001"), calibration=("train_00001",)
    )
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="pairwise disjoint"):
            resolve_train_only_rows(manifest, h5, fit_limit=2, calibration_limit=1)


def test_resolve_rejects_confirmation_overlap(tmp_path: Path) -> None:
    path = _make_vds(tmp_path / "v.h5", n_records=4, time_steps=9, height=8, width=8)
    manifest = _manifest(
        fit=("train_00000",), calibration=("train_00001",), confirmation=("train_00000",)
    )
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="pairwise disjoint"):
            resolve_train_only_rows(manifest, h5, fit_limit=1, calibration_limit=1)


def test_resolve_rejects_non_train_split_metadata(tmp_path: Path) -> None:
    path = _make_vds(
        tmp_path / "v.h5",
        n_records=4,
        time_steps=9,
        height=8,
        width=8,
        split_values=["train", "validation", "train", "train"],
    )
    manifest = _manifest(fit=("train_00000",), calibration=("train_00001",))
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="split metadata"):
            resolve_train_only_rows(manifest, h5, fit_limit=1, calibration_limit=1)


def test_resolve_rejects_sample_id_not_starting_with_train(tmp_path: Path) -> None:
    path = _make_vds(
        tmp_path / "v.h5",
        n_records=4,
        time_steps=9,
        height=8,
        width=8,
        id_prefix="val_",
    )
    manifest = _manifest(fit=("val_00000",), calibration=("val_00001",))
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="train_"):
            resolve_train_only_rows(manifest, h5, fit_limit=1, calibration_limit=1)


def test_resolve_rejects_sample_id_missing_from_vds(tmp_path: Path) -> None:
    path = _make_vds(tmp_path / "v.h5", n_records=4, time_steps=9, height=8, width=8)
    manifest = _manifest(fit=("train_00000", "train_00099"))
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="missing from the VDS"):
            resolve_train_only_rows(manifest, h5, fit_limit=2, calibration_limit=1)


def test_resolve_rejects_manifest_indices_disagreeing_with_mapping(
    tmp_path: Path,
) -> None:
    path = _make_vds(tmp_path / "v.h5", n_records=4, time_steps=9, height=8, width=8)
    # sample_ids train = [train_00000, train_00001] map to rows [0, 1], but the
    # manifest integer array claims rows [0, 2].
    manifest = _manifest(
        fit=("train_00000", "train_00001"),
        calibration=("train_00002",),
        confirmation=("train_00003",),
        index_rows={"train": [0, 2], "val": [2], "test": [3]},
    )
    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="integer indices"):
            resolve_train_only_rows(manifest, h5, fit_limit=2, calibration_limit=1)


def test_resolve_locates_rows_via_sample_id_not_manifest_order(tmp_path: Path) -> None:
    path = _make_vds(tmp_path / "v.h5", n_records=5, time_steps=9, height=8, width=8)
    # Manifest order is deliberately shuffled relative to VDS row order.
    manifest = _manifest(
        fit=("train_00004", "train_00001"),
        calibration=("train_00003",),
        confirmation=("train_00002",),
        index_rows={"train": [4, 1], "val": [3], "test": [2]},
    )
    with h5py.File(path, "r") as h5:
        resolved = resolve_train_only_rows(manifest, h5, fit_limit=2, calibration_limit=1)
    assert resolved["fit_rows"] == [4, 1]
    assert resolved["calibration_rows"] == [3]
    assert resolved["confirmation_ids"] == ["train_00002"]
    assert resolved["index_audit"]["consistent"] is True


# --------------------------------------------------------------------------- #
# Loader reuse: deterministic spatial sampling, all stored time points
# --------------------------------------------------------------------------- #
def test_reader_reuses_loader_deterministic_spatial_sampling(tmp_path: Path) -> None:
    stored_z, stored_x = 16, 12
    total_time = 13  # odd T
    path = _make_vds(
        tmp_path / "v.h5",
        n_records=2,
        time_steps=total_time,
        height=stored_z,
        width=stored_x,
    )
    reader = LoaderWavefieldReader(path, spatial_size=8)
    assert reader.time_indices.tolist() == list(range(total_time))
    assert reader.total_time_steps == total_time
    with h5py.File(path, "r") as h5:
        got = reader.read(h5, 1)
    assert got.shape == (8, 8, total_time)

    # Independent loader instance built from the same config gives the same row.
    dataset = PinoHDF5Dataset(build_loader_config(path, 8), [0], return_normalized=False)
    with h5py.File(path, "r") as h5:
        raw = dataset._read_wavefield(h5, 1)
        expected = dataset._resize_wavefield(raw).detach().numpy()
    np.testing.assert_array_equal(got, expected)

    # Explicit bilinear-antialias reference with the loader's exact arguments.
    tensor = torch.as_tensor(np.transpose(raw, (2, 0, 1)), dtype=torch.float32)[:, None]
    reference = F.interpolate(
        tensor, size=(8, 8), mode="bilinear", align_corners=False, antialias=True
    )
    reference = reference[:, 0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(got, reference)


# --------------------------------------------------------------------------- #
# Full pipeline: candidate freezing and calibration verdict
# --------------------------------------------------------------------------- #
def test_pipeline_selects_frozen_candidate_and_accepts(tmp_path: Path) -> None:
    height = width = 8
    total_time = 16
    t = np.arange(total_time)
    x = np.arange(height)
    z = np.arange(width)
    wave = (
        np.cos(2 * np.pi * 2 * t / total_time)[None, None, :]
        * np.cos(2 * np.pi * x / height)[:, None, None]
        * np.cos(2 * np.pi * z / width)[None, :, None]
    ).astype(np.float32)  # [X, Z, T]
    waves = np.stack([wave] * 5).transpose(0, 3, 1, 2)  # [N, T, Z, X] = NTZX
    path = _make_vds(
        tmp_path / "v.h5",
        n_records=5,
        time_steps=total_time,
        height=height,
        width=width,
        waves=waves,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                fit=("train_00000", "train_00001"),
                calibration=("train_00002", "train_00003"),
                confirmation=("train_00004",),
                index_rows={"train": [0, 1], "val": [2, 3], "test": [4]},
            )
        )
    )
    output = tmp_path / "out.json"
    payload = run_diagnostic(
        data_path=path,
        split_manifest=manifest_path,
        output_path=output,
        fit_limit=2,
        calibration_limit=2,
        spatial_size=8,
        temporal_bins=(1, 2, 3, 4, 16),
        spatial_half_widths=(0, 1, 2, 4),
    )

    assert payload["schema"] == "trainonly_spectral_capacity_v1"
    assert payload["project_data_scope"] == "train_only"
    audit = payload["access_audit"]
    assert audit["confirmation_opened"] is False
    assert audit["validation_opened"] is False
    assert audit["test_id_opened"] is False
    assert audit["fit_opened"] is True and audit["calibration_opened"] is True
    assert audit["row_metadata_contract"]["used_rows_split_metadata_all_train"] is True
    assert payload["wavefield_usage"]["train_labels_only"] is True
    assert payload["wavefield_usage"]["model_input"] is False

    # The single low mode needs nf=3 (time bins 0..2) and k=1 (spatial ±1);
    # (nf=3, k=1) has the fewest real coefficients among qualifying cells.
    selection = payload["candidate_selection"]
    assert selection["selected"] is not None
    assert selection["selected"]["nf"] == 3
    assert selection["selected"]["k"] == 1
    assert selection["selected"]["real_coefficients"] == 45
    assert selection["selected"]["fit_aggregate_relative_l2"] <= 0.05
    assert selection["selected"]["calibration_aggregate_relative_l2"] <= 0.05
    assert selection["calibration_accepted"] is True
    assert selection["calibration_verdict"] == "accepted"
    assert payload["temporal_bins"] == [1, 2, 3, 4, 9]  # 16 clamped to n_rfft
    assert payload["temporal_bins_clamped_to_rfft_count"] is True

    # Atomic JSON output round-trips exactly and stdout-equivalent content.
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["used_sample_identities"]["fit"][0]["sample_id"] == "train_00000"
    assert payload["used_sample_identities"]["fit"][0]["vds_row"] == 0
    assert len(payload["used_sample_identities"]["fit"][0]["sample_sha256"]) == 64
    assert payload["confirmation_sample_ids_manifest_only"] == ["train_00004"]
    assert payload["fit"]["aggregate_relative_l2"]
    assert payload["calibration"]["record_relative_l2"]["p90"]


def test_pipeline_no_fit_candidate_rejected_curves_still_emitted(tmp_path: Path) -> None:
    height = width = 8
    total_time = 16
    # Checkerboard Nyquist spatial mode with a high time bin: no low (nf, k)
    # cell can capture it, so no fit candidate exists.
    t = np.arange(total_time)
    x = np.arange(height)
    z = np.arange(width)
    wave = (
        np.cos(2 * np.pi * 6 * t / total_time)[None, None, :]
        * np.cos(2 * np.pi * 4 * x / height)[:, None, None]
        * np.cos(2 * np.pi * 4 * z / width)[None, :, None]
    ).astype(np.float32)  # [X, Z, T]
    waves = np.stack([wave] * 4).transpose(0, 3, 1, 2)  # [N, T, Z, X] = NTZX
    path = _make_vds(
        tmp_path / "v.h5",
        n_records=4,
        time_steps=total_time,
        height=height,
        width=width,
        waves=waves,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                fit=("train_00000", "train_00001"),
                calibration=("train_00002",),
                confirmation=("train_00003",),
                index_rows={"train": [0, 1], "val": [2], "test": [3]},
            )
        )
    )
    payload = run_diagnostic(
        data_path=path,
        split_manifest=manifest_path,
        output_path=tmp_path / "out.json",
        fit_limit=2,
        calibration_limit=1,
        spatial_size=8,
        temporal_bins=(1, 2, 3, 4),
        spatial_half_widths=(0, 1, 2),
    )
    selection = payload["candidate_selection"]
    assert selection["selected"] is None
    assert selection["fit_candidate_count"] == 0
    assert selection["calibration_accepted"] is False
    assert selection["calibration_verdict"] == "rejected"
    # Curves are still emitted for every (nf, k) cell.
    assert len(payload["fit"]["aggregate_relative_l2"]) == 4
    assert len(payload["fit"]["aggregate_relative_l2"][0]) == 3
    assert payload["real_coefficients_per_record"]["matrix"]
    assert payload["compression_ratio"]["matrix"]


# --------------------------------------------------------------------------- #
# CLI list parsing
# --------------------------------------------------------------------------- #
def test_cli_list_parsing() -> None:
    assert parse_comma_ints("16,32,48", name="temporal-bins") == (16, 32, 48)
    assert parse_nonnegative_comma_ints("0,8,16", name="k") == (0, 8, 16)
    with pytest.raises(ValueError, match="unique"):
        parse_comma_ints("16,16", name="temporal-bins")
    with pytest.raises(ValueError, match="positive"):
        parse_comma_ints("0,8", name="temporal-bins")


# --------------------------------------------------------------------------- #
# Real-coefficient counting / compression ratio at full retention
# --------------------------------------------------------------------------- #
def test_real_coefficients_full_retention_equals_stored_reals() -> None:
    height = width = 8
    total_time = 16
    coefficients = real_coefficients_matrix(
        (9,), (4,), height=height, width=width, total_time=total_time
    )
    assert int(coefficients[0, 0]) == total_time * height * width
    odd = real_coefficients_matrix(
        (5,), (4,), height=height, width=width, total_time=9
    )
    assert int(odd[0, 0]) == 9 * height * width
    # (nf=3, k=1): mask has 9 bins, coefficient weights sum to 1 + 2 + 2.
    partial = real_coefficients_matrix(
        (3,), (1,), height=height, width=width, total_time=16
    )
    assert int(partial[0, 0]) == 9 * 5
