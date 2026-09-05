#!/usr/bin/env python3
"""Build a train-only local frequency-domain trace-to-flux dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from fno_acoustic.data_generation.stencils import first_derivatives8
from saved_time_phase_operator_v4.transfer_dg_elements import (
    cartesian_element_origins,
    extract_element_normal_flux_traces,
    extract_element_traces,
    orthonormal_cosine_trace_basis,
    project_trace_modes,
)


FAMILIES = ("uniform", "layered", "marmousi")
FREQUENCY_RATIOS = (0.50, 0.75, 1.00, 1.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def final_cosine_taper(count: int, *, fraction: float = 0.10) -> np.ndarray:
    """Keep the causal start intact and taper only the terminal FFT boundary."""
    size = int(count)
    if size < 4 or not 0.0 < float(fraction) < 0.5:
        raise ValueError("invalid terminal taper")
    edge = max(2, int(round(size * float(fraction))))
    window = np.ones(size, dtype=np.float32)
    window[-edge:] = 0.5 * (
        1.0 + np.cos(np.linspace(0.0, np.pi, edge, dtype=np.float32))
    )
    return window


def select_frequency_bins(
    frequencies_hz: np.ndarray,
    source_f0_hz: float,
    *,
    minimum_hz: float = 4.0,
    maximum_hz: float = 40.0,
) -> np.ndarray:
    axis = np.asarray(frequencies_hz, dtype=np.float64)
    allowed = np.flatnonzero((axis >= minimum_hz) & (axis <= maximum_hz))
    if allowed.size < len(FREQUENCY_RATIOS) or source_f0_hz <= 0.0:
        raise ValueError("frequency axis cannot support the Transfer DG bins")
    selected = []
    for ratio in FREQUENCY_RATIOS:
        target = float(source_f0_hz) * ratio
        order = allowed[np.argsort(np.abs(axis[allowed] - target))]
        chosen = next((int(index) for index in order if int(index) not in selected), None)
        if chosen is None:
            raise RuntimeError("failed to select unique frequency bins")
        selected.append(chosen)
    return np.asarray(selected, dtype=np.int64)


def non_cpml_element_origins(
    height: int,
    width: int,
    *,
    element_intervals: int = 20,
    cpml_margin: int = 20,
) -> torch.Tensor:
    origins = cartesian_element_origins(
        height, width, element_intervals=element_intervals
    )
    size, margin = int(element_intervals), int(cpml_margin)
    keep = []
    for z0, x0 in origins.tolist():
        keep.append(
            x0 >= margin
            and x0 + size <= width - 1 - margin
            and z0 + size <= height - 1 - margin
        )
    retained = origins[torch.tensor(keep, dtype=torch.bool)]
    if retained.numel() == 0:
        raise ValueError("CPML exclusion removed every local element")
    return retained


def _patches(field: np.ndarray, origins: torch.Tensor, size: int) -> np.ndarray:
    return np.stack(
        [
            field[int(z0) : int(z0) + size + 1, int(x0) : int(x0) + size + 1]
            for z0, x0 in origins.tolist()
        ]
    )


def _decode_string(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _balanced_records(manifest: dict, count_per_family: int) -> list[dict]:
    selected = []
    for family in FAMILIES:
        rows = [row for row in manifest["records"] if row["family"] == family]
        if len(rows) < count_per_family:
            raise RuntimeError(f"not enough {family} records")
        selected.extend(rows[:count_per_family])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--records-per-family", type=int, default=8)
    parser.add_argument("--element-intervals", type=int, default=20)
    parser.add_argument("--trace-modes", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.identity_output.exists():
        raise FileExistsError("refusing to overwrite Transfer DG local data")
    prereg = json.loads(args.preregistration.read_text())
    manifest = json.loads(args.manifest.read_text())
    if _sha256(Path(__file__)) != prereg["bindings"]["builder_sha256"]:
        raise RuntimeError("builder binding drift")
    if _sha256(args.manifest) != prereg["bindings"]["manifest_sha256"]:
        raise RuntimeError("manifest binding drift")
    frozen = prereg["representation"]
    if (
        args.records_per_family != prereg["selection"]["records_per_family"]
        or args.element_intervals != frozen["element_intervals"]
        or args.trace_modes != frozen["trace_modes_per_edge"]
        or str(args.output) != prereg["output"]
        or str(args.identity_output) != prereg["identity_output"]
    ):
        raise RuntimeError("builder arguments drift from preregistration")
    if (
        manifest.get("split") != "train"
        or manifest.get("validation_opened")
        or manifest.get("test_id_opened")
    ):
        raise RuntimeError("local element data must come from sealed train records")
    records = _balanced_records(manifest, args.records_per_family)
    source_path = Path(manifest["source_h5"])
    if source_path.stat().st_size != int(manifest["source_h5_byte_count"]):
        raise RuntimeError("source HDF5 byte count drift")

    feature_rows, trace_rows, flux_rows = [], [], []
    frequency_rows, scale_rows, origin_rows = [], [], []
    record_rows, family_rows, source_index_rows = [], [], []
    partial = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    try:
        with h5py.File(source_path, "r", swmr=True) as source:
            time_s = np.asarray(source["time_s"][:], dtype=np.float64)
            if not np.allclose(np.diff(time_s), np.diff(time_s)[0]):
                raise RuntimeError("source time grid is not uniform")
            dt_s = float(np.diff(time_s)[0])
            frequencies = np.fft.rfftfreq(time_s.size, d=dt_s)
            height, width = source["wavefield"].shape[-2:]
            dx_m = float(np.median(np.diff(source["x_m"][:])))
            dz_m = float(np.median(np.diff(source["z_m"][:])))
            origins = non_cpml_element_origins(
                height,
                width,
                element_intervals=args.element_intervals,
                cpml_margin=20,
            )
            basis = orthonormal_cosine_trace_basis(
                args.element_intervals + 1, args.trace_modes
            )
            taper = final_cosine_taper(time_s.size)
            z_coordinate = np.broadcast_to(
                np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None],
                (height, width),
            )

            for record_number, row in enumerate(records):
                index = int(row["source_index"])
                if _decode_string(source["split"][index]) != "train":
                    raise RuntimeError("selected source record is not train")
                if _decode_string(source["sample_id"][index]) != row["sample_id"]:
                    raise RuntimeError("source/manifest sample order drift")
                velocity = np.asarray(source["velocity_mps"][index], dtype=np.float32)
                source_map = np.asarray(source["source_map"][index], dtype=np.float32)
                pressure = np.asarray(source["wavefield"][index], dtype=np.float32)
                wavelet = np.asarray(source["source_wavelet"][index], dtype=np.float32)
                f0 = float(source["source_f0_hz"][index])
                if not np.isclose(f0, float(row["source_f0_hz"])):
                    raise RuntimeError("source frequency drift")
                bins = select_frequency_bins(frequencies, f0)
                spectrum = np.fft.rfft(
                    pressure * taper[:, None, None], axis=0, norm="ortho"
                )[bins].astype(np.complex64)
                source_spectrum = np.fft.rfft(
                    wavelet * taper, axis=0, norm="ortho"
                ).astype(np.complex64)
                source_norm = max(float(np.abs(source_spectrum).max()), 1.0e-12)

                real = torch.from_numpy(spectrum.real.copy())
                imaginary = torch.from_numpy(spectrum.imag.copy())
                dx_real, dz_real = first_derivatives8(
                    real, dx_m=dx_m, dz_m=dz_m, boundary="free_surface"
                )
                dx_imag, dz_imag = first_derivatives8(
                    imaginary, dx_m=dx_m, dz_m=dz_m, boundary="free_surface"
                )
                pressure_trace = torch.stack(
                    (
                        extract_element_traces(
                            real, origins, element_intervals=args.element_intervals
                        ),
                        extract_element_traces(
                            imaginary, origins, element_intervals=args.element_intervals
                        ),
                    ),
                    dim=2,
                )
                normal_flux = torch.stack(
                    (
                        extract_element_normal_flux_traces(
                            dx_real,
                            dz_real,
                            origins,
                            element_intervals=args.element_intervals,
                        ),
                        extract_element_normal_flux_traces(
                            dx_imag,
                            dz_imag,
                            origins,
                            element_intervals=args.element_intervals,
                        ),
                    ),
                    dim=2,
                )
                trace_coeff = project_trace_modes(pressure_trace, basis)
                flux_coeff = project_trace_modes(normal_flux, basis)
                field_rms = torch.sqrt(
                    real.square().mean((-2, -1)) + imaginary.square().mean((-2, -1))
                )
                trace_rms = torch.sqrt(pressure_trace.square().mean((-3, -2, -1)))
                scale = torch.maximum(trace_rms, 0.05 * field_rms[:, None]).clamp_min(1.0e-12)
                trace_coeff = trace_coeff / scale[:, :, None, None, None]
                element_extent_m = args.element_intervals * 0.5 * (dx_m + dz_m)
                flux_coeff = element_extent_m * flux_coeff / scale[:, :, None, None, None]

                log_velocity = np.log(np.maximum(velocity, 1.0))
                grad_z, grad_x = np.gradient(log_velocity)
                interface = np.clip(
                    20.0 * np.sqrt(grad_x**2 + grad_z**2) / 4.0, 0.0, 1.0
                ).astype(np.float32)
                feature = np.stack(
                    (
                        np.clip((velocity - 4500.0) / 2500.0, -2.0, 2.0),
                        interface,
                        source_map,
                        z_coordinate,
                    ),
                    axis=0,
                )
                element_features = np.stack(
                    [_patches(channel, origins, args.element_intervals) for channel in feature],
                    axis=1,
                ).astype(np.float16)

                for local_frequency, bin_index in enumerate(bins.tolist()):
                    count = origins.shape[0]
                    source_value = source_spectrum[bin_index] / source_norm
                    freq_feature = np.asarray(
                        [
                            frequencies[bin_index] / 40.0,
                            f0 / 30.0,
                            source_value.real,
                            source_value.imag,
                        ],
                        dtype=np.float32,
                    )
                    feature_rows.append(element_features)
                    trace_rows.append(trace_coeff[local_frequency].numpy().astype(np.float32))
                    flux_rows.append(flux_coeff[local_frequency].numpy().astype(np.float32))
                    frequency_rows.append(np.broadcast_to(freq_feature, (count, 4)).copy())
                    scale_rows.append(scale[local_frequency].numpy().astype(np.float32))
                    origin_rows.append(origins.numpy().astype(np.int16))
                    record_rows.extend([row["sample_id"]] * count)
                    family_rows.extend([row["family"]] * count)
                    source_index_rows.extend([index] * count)
                print(
                    json.dumps(
                        {
                            "event": "local_element_record",
                            "record": record_number + 1,
                            "of": len(records),
                            "sample_id": row["sample_id"],
                        }
                    ),
                    flush=True,
                )

            features_array = np.concatenate(feature_rows)
            trace_array = np.concatenate(trace_rows)
            flux_array = np.concatenate(flux_rows)
            frequency_array = np.concatenate(frequency_rows)
            scale_array = np.concatenate(scale_rows)
            origins_array = np.concatenate(origin_rows)
            with h5py.File(partial, "w") as output:
                output.attrs["schema"] = "transfer_dg_local_frequency_elements_v1"
                output.attrs["split"] = "train"
                output.attrs["manifest_sha256"] = _sha256(args.manifest)
                output.attrs["source_config_sha256"] = source.attrs["config_sha256"]
                output.attrs["source_manifest_sha256"] = source.attrs["manifest_sha256"]
                output.attrs["element_intervals"] = args.element_intervals
                output.attrs["trace_modes"] = args.trace_modes
                output.attrs["cpml_margin"] = 20
                output.attrs["dt_s"] = dt_s
                output.attrs["dx_m"] = dx_m
                output.attrs["dz_m"] = dz_m
                output.attrs["fft_terminal_taper_fraction"] = 0.10
                output.attrs["flux_nondimensionalization"] = "element_extent_m * normal_derivative / pressure_scale"
                output.create_dataset("element_features", data=features_array, compression="lzf")
                output.create_dataset("frequency_features", data=frequency_array, compression="lzf")
                output.create_dataset("pressure_trace_coeff", data=trace_array, compression="lzf")
                output.create_dataset("normal_flux_coeff", data=flux_array, compression="lzf")
                output.create_dataset("pressure_scale", data=scale_array, compression="lzf")
                output.create_dataset("element_origin_zx", data=origins_array, compression="lzf")
                output.create_dataset("source_index", data=np.asarray(source_index_rows, dtype=np.int32))
                text_dtype = h5py.string_dtype("utf-8")
                output.create_dataset("sample_id", data=np.asarray(record_rows, dtype=object), dtype=text_dtype)
                output.create_dataset("family", data=np.asarray(family_rows, dtype=object), dtype=text_dtype)
        os.replace(partial, args.output)
        identity = {
            "schema": "transfer_dg_local_frequency_dataset_identity_v1",
            "dataset": str(args.output),
            "dataset_sha256": _sha256(args.output),
            "builder_sha256": _sha256(Path(__file__)),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "records_per_family": args.records_per_family,
            "record_count": len(records),
            "family_counts": {family: args.records_per_family for family in FAMILIES},
            "sample_count": int(features_array.shape[0]),
            "elements_per_record_frequency": int(origins.shape[0]),
            "frequency_bins_per_record": len(FREQUENCY_RATIOS),
            "future_train_truth_used_for_offline_supervision": True,
            "selection_used_future_truth": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.identity_output)
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
