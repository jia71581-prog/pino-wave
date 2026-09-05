#!/usr/bin/env python3
"""Read-only train-only spectral capacity floor for the stored P_bg truth.

Measures, on project-train records only, how much of the true P_bg wavefield
energy is captured by keeping the lowest ``nf`` real-time frequencies and a
centered spatial wavenumber rectangle ``|ky|, |kx| <= k``.

Access contract
---------------
* The split manifest must declare ``project_data_scope == "train_only"``.
  Its ``train`` / ``val`` / ``test`` panels are interpreted as the project-train
  fit / calibration / confirmation panels.
* Only the fit and calibration panels are opened.  The confirmation panel is
  read from the manifest text only and its HDF5 rows are never opened.
* Every opened HDF5 row must carry split metadata ``train`` and a sample id
  starting with ``train_``; rows are located through the sample-id -> row
  mapping (never through manifest order).
* Numerical wavefields are used only as train labels for this capacity floor;
  they are never used as model inputs.

Numerics
--------
Each record's true P_bg is read with the current loader's deterministic
spatial sampling (bilinear + antialias to ``--spatial-size``) over all stored
time points (stride 1, no interpolation).  A real time rFFT (norm="ortho") is
followed by a 2D spatial FFT (norm="ortho") of every complex frequency slice.
Parseval is used to compute, without any explicit iFFT, the relative L2 error
after retaining the lowest ``nf`` time bins and ``|ky|,|kx| <= k``:

    rel_L2(nf, k) = sqrt(1 - E_retained(nf, k) / E_total)

The single-sided rFFT energy weights are 1 at DC (and at Nyquist when T is
even) and 2 for every conjugate-paired bin.  A complex coefficient is counted
as two real values, so a paired time bin contributes two real values per
spatial mask point while DC/Nyquist contribute one; at full retention the real
coefficient count equals ``T * H * W`` exactly.

Candidate selection
-------------------
A candidate is the (nf, k) cell with the fewest real coefficients among fit
cells satisfying aggregate relative L2 <= 0.05 and nf <= 64.  The calibration
panel only evaluates that frozen candidate and reports accepted/rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data import PinoHDF5Dataset  # noqa: E402

_OUTPUT_SCHEMA = "trainonly_spectral_capacity_v1"
_SELECTION_THRESHOLD = 0.05
_SELECTION_MAX_NF = 64
_WAVEFIELD_DECLARATION = (
    "Numerical wavefields were opened only for project-train records (fit and "
    "calibration panels) and used exclusively as train labels for this "
    "read-only capacity floor diagnostic; they were never used as model "
    "inputs. No validation, test_id, or confirmation wavefield truth was read."
)


# --------------------------------------------------------------------------- #
# Small pure helpers (testable)
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_column(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_attr_list(value: Any) -> list[str] | None:
    """Normalize an HDF5 attr that may be a JSON string, list, or ndarray."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return _normalize_attr_list(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            if isinstance(parsed, (list, tuple)):
                return [_attr_text(item) for item in parsed]
        return [stripped]
    if isinstance(value, np.ndarray):
        return [_attr_text(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_attr_text(item) for item in value]
    return [_attr_text(value)]


def parse_comma_ints(text: str, *, name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers."""
    items = tuple(int(part) for part in str(text).split(",") if part.strip())
    if not items:
        raise ValueError(f"{name} must be a nonempty comma-separated integer list")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} values must be unique")
    if any(value <= 0 for value in items):
        raise ValueError(f"{name} values must be positive")
    return tuple(sorted(items))


def parse_nonnegative_comma_ints(text: str, *, name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of nonnegative integers (k = 0 is valid)."""
    items = tuple(int(part) for part in str(text).split(",") if part.strip())
    if not items:
        raise ValueError(f"{name} must be a nonempty comma-separated integer list")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} values must be unique")
    if any(value < 0 for value in items):
        raise ValueError(f"{name} values must be nonnegative")
    return tuple(sorted(items))


def rfft_energy_weights(total_time: int) -> np.ndarray:
    """Parseval weights of a single-sided real rFFT (norm="ortho").

    DC and (for even T) Nyquist bins hold a single real coefficient and carry
    energy weight 1; every other bin represents a conjugate pair and carries
    energy weight 2:  sum_t x(t)^2 = sum_f w[f] |rfft(x)[f]|^2.
    """
    if int(total_time) < 1:
        raise ValueError("total_time must be positive")
    n_freq = int(total_time) // 2 + 1
    weights = np.full(n_freq, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if int(total_time) % 2 == 0:
        weights[-1] = 1.0
    return weights


def rfft_coefficient_weights(total_time: int) -> np.ndarray:
    """Real values needed to store each single-sided rFFT bin of a real signal.

    Identical structure to the energy weights: 1 for DC/Nyquist, 2 for a
    conjugate-paired bin (one complex amplitude = two real values).
    """
    return rfft_energy_weights(total_time)


def spatial_mask(height: int, width: int, half_width: int) -> np.ndarray:
    """Boolean [height, width] mask of centered spatial DFT bins |ky|,|kx|<=k."""
    if int(half_width) < 0:
        raise ValueError("half_width must be nonnegative")
    ky = np.fft.fftfreq(int(height)) * int(height)
    kx = np.fft.fftfreq(int(width)) * int(width)
    return (np.abs(ky)[:, None] <= int(half_width)) & (np.abs(kx)[None, :] <= int(half_width))


def real_coefficients_matrix(
    temporal_bins: Sequence[int],
    spatial_half_widths: Sequence[int],
    *,
    height: int,
    width: int,
    total_time: int,
) -> np.ndarray:
    """Real coefficients per record for each (nf, k) cell.

    ``mask_size(k) * sum_{f<nf} coefficient_weight(f)`` with coefficient
    weight 1 at DC/Nyquist and 2 elsewhere.  At full retention this equals
    ``total_time * height * width``.
    """
    coefficient_cumulative = np.cumsum(rfft_coefficient_weights(total_time))
    mask_sizes = [int(spatial_mask(height, width, k).sum()) for k in spatial_half_widths]
    out = np.empty((len(temporal_bins), len(spatial_half_widths)), dtype=np.int64)
    for ni, nf in enumerate(temporal_bins):
        bin_count = min(int(nf), int(coefficient_cumulative.size))
        for ki, size in enumerate(mask_sizes):
            out[ni, ki] = int(coefficient_cumulative[bin_count - 1]) * size
    return out


def per_record_cumulative_energies(
    wave: np.ndarray,
    weights: np.ndarray,
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Cumulative retained energy over time bins for each spatial mask.

    Exact Parseval accounting: time rFFT with norm="ortho" and single-sided
    weights, then a 2D spatial FFT (norm="ortho") of every complex frequency
    slice.  Returns (cumulative [n_freq, n_masks], total energy).
    """
    wave64 = np.asarray(wave, dtype=np.float64)
    total = float(np.sum(wave64 * wave64))
    y = np.fft.rfft(wave64, axis=-1, norm="ortho")  # [H, W, n_freq]
    n_freq = y.shape[-1]
    per_freq = np.empty((n_freq, len(masks)), dtype=np.float64)
    for f in range(n_freq):
        y2 = np.fft.fft2(y[..., f], norm="ortho")  # [H, W] complex
        energy = np.abs(y2) ** 2
        for ki, mask in enumerate(masks):
            per_freq[f, ki] = float(energy[mask].sum())
    if weights.size < n_freq:
        raise ValueError("energy weights are shorter than the rFFT bin count")
    cumulative = np.cumsum(per_freq * weights[:n_freq, None], axis=0)
    return cumulative, total


def panel_residual_truth(
    reader: "LoaderWavefieldReader",
    h5: h5py.File,
    rows: Sequence[int],
    *,
    temporal_bins: Sequence[int],
    masks: Sequence[np.ndarray],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual [n_records, n_nf, n_k] and truth [n_records] energies."""
    residual = np.empty((len(rows), len(temporal_bins), len(masks)), dtype=np.float64)
    truth = np.empty((len(rows),), dtype=np.float64)
    for i, row in enumerate(rows):
        wave = reader.read(h5, int(row))
        cumulative, total = per_record_cumulative_energies(wave, weights, masks)
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"row {int(row)} has nonpositive total wavefield energy")
        retained = cumulative[np.asarray(temporal_bins, dtype=np.int64) - 1, :]
        residual[i] = np.maximum(total - retained, 0.0)
        truth[i] = total
    return residual, truth


def aggregate_table(residual: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Aggregate and per-record relative L2 tables over the (nf, k) grid."""
    denominator = max(float(truth.sum()), 1.0e-300)
    aggregate = np.sqrt(np.maximum(residual.sum(axis=0), 0.0) / denominator)
    per_record = np.sqrt(
        np.maximum(residual, 0.0) / np.maximum(truth[:, None, None], 1.0e-300)
    )
    return {
        "aggregate_relative_l2": aggregate.tolist(),
        "record_relative_l2": {
            "mean": per_record.mean(axis=0).tolist(),
            "median": np.median(per_record, axis=0).tolist(),
            "p90": np.quantile(per_record, 0.9, axis=0).tolist(),
            "max": per_record.max(axis=0).tolist(),
        },
        "truth_energy_total": float(truth.sum()),
    }


def select_fit_candidate(
    fit_aggregate: np.ndarray,
    *,
    temporal_bins: Sequence[int],
    spatial_half_widths: Sequence[int],
    real_coefficients: np.ndarray,
    threshold: float = _SELECTION_THRESHOLD,
    max_nf: int = _SELECTION_MAX_NF,
) -> list[dict[str, Any]]:
    """Qualifying fit cells, sorted by fewest real coefficients then (nf, k)."""
    cells: list[dict[str, Any]] = []
    for ni, nf in enumerate(temporal_bins):
        if int(nf) > int(max_nf):
            continue
        for ki, k in enumerate(spatial_half_widths):
            value = float(fit_aggregate[ni, ki])
            if value <= float(threshold):
                cells.append(
                    {
                        "nf": int(nf),
                        "k": int(k),
                        "real_coefficients": int(real_coefficients[ni, ki]),
                        "fit_aggregate_relative_l2": value,
                    }
                )
    cells.sort(key=lambda cell: (cell["real_coefficients"], cell["nf"], cell["k"]))
    return cells


# --------------------------------------------------------------------------- #
# Manifest / row resolution and loader reuse
# --------------------------------------------------------------------------- #
def resolve_train_only_rows(
    manifest: dict[str, Any],
    h5: h5py.File,
    *,
    fit_limit: int,
    calibration_limit: int,
) -> dict[str, Any]:
    """Validate the train-only contract and resolve fit/calibration rows.

    Rows are located through the sample-id -> row mapping (never through
    manifest order).  Confirmation ids are read from the manifest text only;
    their HDF5 rows are never resolved or opened.
    """
    if manifest.get("project_data_scope") != "train_only":
        raise ValueError(
            "split manifest must declare project_data_scope='train_only', got "
            f"{manifest.get('project_data_scope')!r}"
        )
    if str(h5.attrs.get("axis_order", "")) != "NTZX":
        raise ValueError(
            f"dataset axis_order must be NTZX, got {h5.attrs.get('axis_order')!r}"
        )
    included_splits = _normalize_attr_list(h5.attrs.get("included_splits"))
    if included_splits is not None and "train" not in included_splits:
        raise ValueError(
            f"dataset attrs included_splits={included_splits!r} does not include 'train'"
        )
    sample_ids = manifest.get("sample_ids")
    if not isinstance(sample_ids, dict):
        raise ValueError("split manifest must contain a sample_ids mapping")
    missing_panels = [key for key in ("train", "val", "test") if key not in sample_ids]
    if missing_panels:
        raise ValueError(f"split manifest sample_ids is missing panels {missing_panels}")

    fit_ids = [str(value) for value in sample_ids["train"]]
    calibration_ids = [str(value) for value in sample_ids["val"]]
    confirmation_ids = [str(value) for value in sample_ids["test"]]
    for panel_name, ids in (
        ("fit", fit_ids),
        ("calibration", calibration_ids),
        ("confirmation", confirmation_ids),
    ):
        if len(set(ids)) != len(ids):
            raise ValueError(f"manifest {panel_name} sample ids contain duplicates")
    if (
        set(fit_ids) & set(calibration_ids)
        or set(fit_ids) & set(confirmation_ids)
        or set(calibration_ids) & set(confirmation_ids)
    ):
        raise ValueError(
            "manifest fit/calibration/confirmation sample ids must be pairwise disjoint"
        )

    id_column = _decode_column(h5["sample_id"][:])
    split_column = _decode_column(h5["split"][:])
    if len(id_column) != len(split_column):
        raise ValueError("sample_id and split columns must have equal length")
    id_to_row = {sample_id: row for row, sample_id in enumerate(id_column)}
    if len(id_to_row) != len(id_column):
        raise ValueError("VDS sample_id column contains duplicate values")

    # Optional integer index arrays must agree with the sample-id mapping.
    index_audit: dict[str, Any] = {"present": False, "consistent": None}
    if all(isinstance(manifest.get(key), list) for key in ("train", "val", "test")):
        index_audit["present"] = True
        for panel_name, ids, rows in (
            ("train", fit_ids, manifest["train"]),
            ("val", calibration_ids, manifest["val"]),
            ("test", confirmation_ids, manifest["test"]),
        ):
            expected = [id_to_row[sample_id] for sample_id in ids if sample_id in id_to_row]
            actual = [int(value) for value in rows]
            if len(expected) != len(actual) or expected != actual:
                raise ValueError(
                    f"manifest {panel_name} integer indices disagree with the "
                    "sample_id -> row mapping"
                )
        index_audit["consistent"] = True

    if int(fit_limit) <= 0:
        raise ValueError("fit-limit must be positive")
    if int(calibration_limit) <= 0:
        raise ValueError("cal-limit must be positive")
    fit_used_ids = fit_ids[: int(fit_limit)]
    calibration_used_ids = calibration_ids[: int(calibration_limit)]
    if not fit_used_ids:
        raise ValueError("fit panel is empty after applying fit-limit")
    if not calibration_used_ids:
        raise ValueError("calibration panel is empty after applying calibration-limit")

    resolved: dict[str, list[int]] = {"fit": [], "calibration": []}
    for panel_name, ids in (("fit", fit_used_ids), ("calibration", calibration_used_ids)):
        for sample_id in ids:
            if sample_id not in id_to_row:
                raise ValueError(
                    f"{panel_name} sample_id {sample_id!r} is missing from the "
                    "VDS sample_id column"
                )
            row = id_to_row[sample_id]
            if split_column[row] != "train":
                raise ValueError(
                    f"{panel_name} sample_id {sample_id!r} maps to VDS row {row} "
                    f"with split metadata {split_column[row]!r} instead of 'train'"
                )
            if not sample_id.startswith("train_"):
                raise ValueError(
                    f"{panel_name} sample_id {sample_id!r} does not start with 'train_'"
                )
            resolved[panel_name].append(row)

    return {
        "fit_ids": fit_used_ids,
        "calibration_ids": calibration_used_ids,
        "confirmation_ids": confirmation_ids,
        "fit_rows": resolved["fit"],
        "calibration_rows": resolved["calibration"],
        "fit_limit_requested": int(fit_limit),
        "calibration_limit_requested": int(calibration_limit),
        "fit_available": len(fit_ids),
        "calibration_available": len(calibration_ids),
        "index_audit": index_audit,
    }


def build_loader_config(data_path: str | Path, spatial_size: int) -> dict[str, Any]:
    """Minimal PinoHDF5Dataset config selecting every stored time point."""
    return {
        "data": {
            "path": str(Path(data_path).resolve()),
            "wavefield_key": "wavefield",
            "wavefield_axes": ["sample", "time", "z", "x"],
            "time_key": "time_s",
        },
        "sampling": {
            "time_start": 0,
            "time_stop": None,
            "time_stride": 1,
            "max_time_steps": None,
            "target_height": int(spatial_size),
            "target_width": int(spatial_size),
        },
        "source_map": {},
        "normalization": {"eps": 1.0e-12},
    }


class LoaderWavefieldReader:
    """Reads stored wavefield rows through PinoHDF5Dataset's own sampling.

    Spatial resampling reuses the loader's deterministic bilinear-antialias
    rule (PinoHDF5Dataset._resize_wavefield); temporal sampling uses every
    stored time point (stride 1, no interpolation).  Only the loader's own
    read/resize helpers are used so the diagnostic always observes exactly the
    same deterministic sampling the current loader applies.
    """

    def __init__(self, data_path: str | Path, spatial_size: int) -> None:
        self.config = build_loader_config(data_path, spatial_size)
        self._dataset = PinoHDF5Dataset(self.config, [0], return_normalized=False)
        self.time_indices = np.asarray(self._dataset.time_indices, dtype=np.int64)
        self.total_time_steps = int(self._dataset.total_time_steps)
        self.target_height = int(self._dataset.target_height)
        self.target_width = int(self._dataset.target_width)

    def read(self, h5: h5py.File, row: int) -> np.ndarray:
        raw = self._dataset._read_wavefield(h5, int(row))  # [X, Z, T] float32
        resized = self._dataset._resize_wavefield(raw)  # [H, W, T] float32
        return resized.detach().numpy()


# --------------------------------------------------------------------------- #
# Main diagnostic
# --------------------------------------------------------------------------- #
def run_diagnostic(
    *,
    data_path: Path,
    split_manifest: Path,
    output_path: Path,
    fit_limit: int,
    calibration_limit: int,
    spatial_size: int,
    temporal_bins: Sequence[int],
    spatial_half_widths: Sequence[int],
) -> dict[str, Any]:
    started = time.perf_counter()
    if int(spatial_size) < 2:
        raise ValueError("spatial-size must be at least 2")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing output: {output_path}")

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    data_path = data_path.resolve()
    split_manifest = split_manifest.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as h5:
        time_s = np.asarray(h5["time_s"][:], dtype=np.float64)
        x_m = np.asarray(h5["x_m"][:], dtype=np.float64)
        z_m = np.asarray(h5["z_m"][:], dtype=np.float64)
        attrs = {
            "axis_order": _attr_text(h5.attrs.get("axis_order", "")),
            "schema_version": _attr_text(h5.attrs.get("schema_version", "")),
            "included_splits": _normalize_attr_list(h5.attrs.get("included_splits")),
            "dt_output_s": (
                float(h5.attrs["dt_output_s"]) if "dt_output_s" in h5.attrs else None
            ),
            "saved_grid_shape": (
                [
                    int(value)
                    for value in (
                        _normalize_attr_list(h5.attrs.get("saved_grid_shape")) or []
                    )
                ]
                if "saved_grid_shape" in h5.attrs
                else None
            ),
        }
        resolution = resolve_train_only_rows(
            manifest,
            h5,
            fit_limit=int(fit_limit),
            calibration_limit=int(calibration_limit),
        )

        reader = LoaderWavefieldReader(data_path, int(spatial_size))
        n_rfft = reader.total_time_steps // 2 + 1
        effective_bins = tuple(sorted({min(int(nf), n_rfft) for nf in temporal_bins}))
        temporal_clamped = effective_bins != tuple(int(nf) for nf in temporal_bins)
        weights = rfft_energy_weights(reader.total_time_steps)
        masks = [
            spatial_mask(reader.target_height, reader.target_width, int(k))
            for k in spatial_half_widths
        ]

        fit_residual, fit_truth = panel_residual_truth(
            reader,
            h5,
            resolution["fit_rows"],
            temporal_bins=effective_bins,
            masks=masks,
            weights=weights,
        )
        calibration_residual, calibration_truth = panel_residual_truth(
            reader,
            h5,
            resolution["calibration_rows"],
            temporal_bins=effective_bins,
            masks=masks,
            weights=weights,
        )
        fit_table = aggregate_table(fit_residual, fit_truth)
        calibration_table = aggregate_table(calibration_residual, calibration_truth)
        fit_aggregate = np.asarray(fit_table["aggregate_relative_l2"], dtype=np.float64)
        calibration_aggregate = np.asarray(
            calibration_table["aggregate_relative_l2"], dtype=np.float64
        )

        real_coefficients = real_coefficients_matrix(
            effective_bins,
            spatial_half_widths,
            height=reader.target_height,
            width=reader.target_width,
            total_time=reader.total_time_steps,
        )
        total_stored_reals = int(reader.total_time_steps) * reader.target_height * reader.target_width
        compression_ratio = np.divide(
            float(total_stored_reals),
            real_coefficients.astype(np.float64),
            out=np.full_like(real_coefficients, np.nan, dtype=np.float64),
            where=real_coefficients > 0,
        )

        candidates = select_fit_candidate(
            fit_aggregate,
            temporal_bins=effective_bins,
            spatial_half_widths=spatial_half_widths,
            real_coefficients=real_coefficients,
        )
        selected = candidates[0] if candidates else None
        calibration_accepted = False
        if selected is not None:
            ni = effective_bins.index(int(selected["nf"]))
            ki = spatial_half_widths.index(int(selected["k"]))
            cal_value = float(calibration_aggregate[ni, ki])
            selected["calibration_aggregate_relative_l2"] = cal_value
            calibration_accepted = bool(cal_value <= _SELECTION_THRESHOLD)
        calibration_verdict = "accepted" if calibration_accepted else "rejected"

        sample_sha256_column = (
            _decode_column(h5["sample_sha256"][:]) if "sample_sha256" in h5 else None
        )
        fit_identities = [
            {
                "sample_id": sample_id,
                "vds_row": row,
                "sample_sha256": (
                    sample_sha256_column[row] if sample_sha256_column is not None else None
                ),
            }
            for sample_id, row in zip(resolution["fit_ids"], resolution["fit_rows"], strict=True)
        ]
        calibration_identities = [
            {
                "sample_id": sample_id,
                "vds_row": row,
                "sample_sha256": (
                    sample_sha256_column[row] if sample_sha256_column is not None else None
                ),
            }
            for sample_id, row in zip(
                resolution["calibration_ids"], resolution["calibration_rows"], strict=True
            )
        ]

        dt_s = float(np.median(np.diff(time_s))) if time_s.size > 1 else 0.0
        rfft_frequencies = np.fft.rfftfreq(reader.total_time_steps, d=dt_s)
        manifest_vds = manifest.get("vds")
        data_matches_manifest = bool(
            manifest_vds is not None
            and Path(str(manifest_vds)).resolve() == data_path
        )

        payload: dict[str, Any] = {
            "schema": _OUTPUT_SCHEMA,
            "status": "completed",
            "project_data_scope": "train_only",
            "data": {
                "path": str(data_path),
                "byte_count": data_path.stat().st_size,
                "sha256": _sha256(data_path),
                "attrs": attrs,
            },
            "split_manifest": {
                "path": str(split_manifest),
                "byte_count": split_manifest.stat().st_size,
                "sha256": _sha256(split_manifest),
                "schema": manifest.get("schema"),
                "source_identity_run_digest": manifest.get("source_identity_run_digest"),
                "declared_vds": manifest_vds,
                "data_path_matches_manifest_vds": data_matches_manifest,
            },
            "output": {"path": str(output_path)},
            "limits": {
                "fit_requested": int(fit_limit),
                "fit_used": len(resolution["fit_rows"]),
                "fit_available": resolution["fit_available"],
                "calibration_requested": int(calibration_limit),
                "calibration_used": len(resolution["calibration_rows"]),
                "calibration_available": resolution["calibration_available"],
            },
            "sampling": {
                "time": {
                    "stored_count": reader.total_time_steps,
                    "used_count": int(reader.time_indices.size),
                    "used_indices": {
                        "start": int(reader.time_indices[0]),
                        "stop": int(reader.time_indices[-1]),
                        "stride": 1,
                    },
                    "interpolated": False,
                    "rule": "all stored time points, stride 1, no interpolation",
                },
                "spatial": {
                    "stored_shape": [int(z_m.size), int(x_m.size)],
                    "used_shape": [reader.target_height, reader.target_width],
                    "rule": "PinoHDF5Dataset deterministic bilinear-antialias resize",
                },
                "dt_s": dt_s,
                "time_frequency_hz": {
                    "bin_count": n_rfft,
                    "min_hz": float(rfft_frequencies[0]),
                    "max_hz": float(rfft_frequencies[-1]),
                    "bin_width_hz": (
                        float(rfft_frequencies[1]) if rfft_frequencies.size > 1 else None
                    ),
                },
                "spatial_half_width_units": (
                    "DFT bins of the resized spatial grid (|ky|,|kx| <= k)"
                ),
                "spatial_nyquist_bin": reader.target_height // 2,
            },
            "temporal_bins": list(effective_bins),
            "temporal_bins_requested": [int(nf) for nf in temporal_bins],
            "temporal_bins_clamped_to_rfft_count": bool(temporal_clamped),
            "spatial_half_widths": [int(k) for k in spatial_half_widths],
            "access_audit": {
                "fit_opened": True,
                "calibration_opened": True,
                "confirmation_opened": False,
                "validation_opened": False,
                "test_id_opened": False,
                "fit_record_count": len(resolution["fit_rows"]),
                "calibration_record_count": len(resolution["calibration_rows"]),
                "confirmation_record_count": 0,
                "confirmation_ids_read_from_manifest_only": len(
                    resolution["confirmation_ids"]
                ),
                "row_metadata_contract": {
                    "used_rows_split_metadata_all_train": True,
                    "used_rows_sample_id_prefix_train_": True,
                    "fit_calibration_confirmation_ids_pairwise_disjoint": True,
                    "manifest_index_arrays_match_sample_id_mapping": resolution[
                        "index_audit"
                    ],
                },
                "manifest_declared_validation_opened": manifest.get("validation_opened"),
                "manifest_declared_test_id_opened": manifest.get("test_id_opened"),
                "hdf5_datasets_read": [
                    "sample_id",
                    "split",
                    "sample_sha256",
                    "time_s",
                    "x_m",
                    "z_m",
                    "wavefield (fit/calibration rows only)",
                ],
            },
            "used_sample_identities": {
                "fit": fit_identities,
                "calibration": calibration_identities,
            },
            "confirmation_sample_ids_manifest_only": resolution["confirmation_ids"],
            "wavefield_usage": {
                "train_labels_only": True,
                "model_input": False,
                "declaration": _WAVEFIELD_DECLARATION,
            },
            "real_coefficients_per_record": {
                "matrix": real_coefficients.tolist(),
                "total_stored_reals": total_stored_reals,
            },
            "compression_ratio": {"matrix": compression_ratio.tolist()},
            "fit": fit_table,
            "calibration": calibration_table,
            "candidate_selection": {
                "threshold_relative_l2": _SELECTION_THRESHOLD,
                "max_time_frequency_bins": _SELECTION_MAX_NF,
                "rule": (
                    "fewest real coefficients among fit aggregate relative L2 "
                    "<= 0.05 with nf <= 64"
                ),
                "fit_candidate_count": len(candidates),
                "fit_candidates": candidates,
                "selected": selected,
                "calibration_accepted": calibration_accepted,
                "calibration_verdict": calibration_verdict,
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        }

    payload["elapsed_seconds"] = float(time.perf_counter() - started)
    partial = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="NTZX train-only VDS (.h5)")
    parser.add_argument("--split-manifest", required=True, help="train-only split manifest JSON")
    parser.add_argument("--output", required=True, help="output JSON path (must not exist)")
    parser.add_argument("--fit-limit", type=int, default=70)
    parser.add_argument("--cal-limit", type=int, default=70)
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument(
        "--temporal-bins",
        default="16,32,48,64,96,128,201",
        help="comma-separated lowest time-frequency bin counts to retain",
    )
    parser.add_argument(
        "--spatial-half-widths",
        default="8,12,16,24,32",
        help="comma-separated centered spatial half-widths k (|ky|,|kx| <= k)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    temporal_bins = parse_comma_ints(args.temporal_bins, name="temporal-bins")
    spatial_half_widths = parse_nonnegative_comma_ints(
        args.spatial_half_widths, name="spatial-half-widths"
    )
    payload = run_diagnostic(
        data_path=Path(args.data_path),
        split_manifest=Path(args.split_manifest),
        output_path=Path(args.output),
        fit_limit=int(args.fit_limit),
        calibration_limit=int(args.cal_limit),
        spatial_size=int(args.spatial_size),
        temporal_bins=temporal_bins,
        spatial_half_widths=spatial_half_widths,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
