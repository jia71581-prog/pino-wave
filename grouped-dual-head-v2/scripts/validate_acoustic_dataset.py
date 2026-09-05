#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fno_acoustic.data_generation.hdf5_writer import validate_v3_hdf5_schema
from fno_acoustic.data_generation.config import load_config
from fno_acoustic.data_generation.hdf5_lwc84 import (
    SCHEMA_VERSION as LWC84_SCHEMA_VERSION,
    validate_lwc84_dataset_vds,
    validate_lwc84_hybrid_vds,
    validate_lwc84_shard,
)
from fno_acoustic.data_generation.quality import sha256_array
from fno_acoustic.data_generation.split_manifest import CATEGORY_IDS, FREQUENCY_ROLE_IDS, SPLIT_IDS, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v3 acoustic dataset shards or final HDF5.")
    parser.add_argument("--dataset", "--input", dest="dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--readback-check", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", "--output-json", dest="output", default=None)
    return parser.parse_args()


def _reverse(mapping: dict[str, int]) -> dict[int, str]:
    return {int(value): key for key, value in mapping.items()}


def _scan_numeric_dataset(dataset) -> dict[str, float | int | bool]:
    nan_count = 0
    inf_count = 0
    min_value = np.inf
    max_value = -np.inf
    max_abs = 0.0
    for i in range(int(dataset.shape[0])):
        arr = np.asarray(dataset[i])
        nan_count += int(np.isnan(arr).sum())
        inf_count += int(np.isinf(arr).sum())
        finite = arr[np.isfinite(arr)]
        if finite.size:
            min_value = min(min_value, float(np.min(finite)))
            max_value = max(max_value, float(np.max(finite)))
            max_abs = max(max_abs, float(np.max(np.abs(finite))))
    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": 0.0 if np.isposinf(min_value) else float(min_value),
        "max": 0.0 if np.isneginf(max_value) else float(max_value),
        "max_abs": max_abs,
        "passed": bool(nan_count == 0 and inf_count == 0),
    }


def _free_surface_gate(wavefield) -> dict[str, float | bool]:
    max_abs = 0.0
    for i in range(int(wavefield.shape[0])):
        top = np.asarray(wavefield[i, 0, :, :])
        max_abs = max(max_abs, float(np.max(np.abs(top))) if top.size else 0.0)
    return {"max_abs_top_surface": max_abs, "passed": bool(max_abs <= 1.0e-7)}


def _cfl_gate(h5: h5py.File) -> dict[str, float | bool | int]:
    q = h5["/quality"]
    cmax = np.asarray(q["c_max_mps"], dtype=np.float64)
    stored = np.asarray(q["cfl_axis"], dtype=np.float64)
    dt = np.asarray(q["truth_internal_dt_s"], dtype=np.float64)
    n_substeps = np.asarray(q["truth_internal_n_substeps"], dtype=np.float64)
    dx = float(h5.attrs["dx_m"])
    dz = float(h5.attrs["dz_m"])
    dt_out = float(h5.attrs["dt_out_s"])
    actual = cmax * dt / min(dx, dz)
    alignment = np.abs(dt * n_substeps - dt_out)
    max_alignment = float(np.max(alignment)) if alignment.size else 0.0
    max_actual = float(np.max(actual)) if actual.size else 0.0
    max_stored = float(np.max(stored)) if stored.size else 0.0
    min_substeps = int(np.min(n_substeps)) if n_substeps.size else 0
    passed = bool(
        np.all(n_substeps >= 1)
        and np.all(stored <= 0.3000000001)
        and np.all(actual <= stored + 1.0e-6)
        and max_alignment <= 1.0e-12
    )
    return {
        "max_actual_cfl_axis": max_actual,
        "max_stored_cfl_axis": max_stored,
        "max_dt_alignment_error_s": max_alignment,
        "min_truth_internal_n_substeps": min_substeps,
        "passed": passed,
    }


def _time_axis_gate(h5: h5py.File) -> dict[str, float | int | bool]:
    t_s = np.asarray(h5["/coordinates/t_s"], dtype=np.float64)
    nt_out = int(h5.attrs["nt_out"])
    dt_out = float(h5.attrs["dt_out_s"])
    expected = np.arange(nt_out, dtype=np.float64) * dt_out
    max_error = float(np.max(np.abs(t_s - expected))) if t_s.size else 0.0
    t_end = float(t_s[-1]) if t_s.size else 0.0
    return {
        "nt_out": int(t_s.size),
        "dt_out_s": dt_out,
        "t_end_s": t_end,
        "max_axis_error_s": max_error,
        "passed": bool(t_s.size == nt_out and max_error <= 1.0e-12),
    }


def _source_reconstruction_gate(h5: h5py.File) -> dict[str, float | bool]:
    dx = float(h5.attrs["dx_m"])
    dz = float(h5.attrs["dz_m"])
    source_xy = np.asarray(h5["/samples/source_xy_m"], dtype=np.float64)
    indices = np.asarray(h5["/samples/source_indices"], dtype=np.int64)
    weights = np.asarray(h5["/samples/source_weights"], dtype=np.float64)
    source_map = h5["/samples/source_map"]
    max_weight_sum_error = 0.0
    max_xy_error_m = 0.0
    max_map_error = 0.0
    for i in range(source_xy.shape[0]):
        sample_indices = indices[i]
        sample_weights = weights[i]
        max_weight_sum_error = max(max_weight_sum_error, float(abs(np.sum(sample_weights) - 1.0)))
        z_idx = sample_indices[:, 0].astype(np.float64)
        x_idx = sample_indices[:, 1].astype(np.float64)
        x_recon = float(np.sum(sample_weights * ((x_idx + 0.5) * dx)))
        z_recon = float(np.sum(sample_weights * ((z_idx + 0.5) * dz)))
        max_xy_error_m = max(max_xy_error_m, float(np.max(np.abs(np.asarray([x_recon, z_recon]) - source_xy[i]))))
        expected = np.zeros(source_map.shape[1:], dtype=np.float32)
        for (z_cell, x_cell), weight in zip(sample_indices, sample_weights):
            expected[int(z_cell), int(x_cell)] += np.float32(weight)
        stored = np.asarray(source_map[i], dtype=np.float32)
        max_map_error = max(max_map_error, float(np.max(np.abs(stored - expected))))
    passed = bool(max_weight_sum_error <= 1.0e-12 and max_xy_error_m <= 1.0e-9 and max_map_error <= 1.0e-7)
    return {
        "max_weight_sum_error": max_weight_sum_error,
        "max_xy_reconstruction_error_m": max_xy_error_m,
        "max_source_map_error": max_map_error,
        "passed": passed,
    }


def _frequency_coverage(h5: h5py.File) -> dict[str, object]:
    f0 = np.asarray(h5["/samples/f0_hz"], dtype=np.float64)
    return {
        "has_10hz": bool(np.any(np.isclose(f0, 10.0, atol=1.0e-6))),
        "has_25hz": bool(np.any(np.isclose(f0, 25.0, atol=1.0e-6))),
        "min_hz": float(np.min(f0)) if f0.size else None,
        "max_hz": float(np.max(f0)) if f0.size else None,
        "anchor_10hz_count": int(np.isclose(f0, 10.0, atol=1.0e-6).sum()),
        "anchor_25hz_count": int(np.isclose(f0, 25.0, atol=1.0e-6).sum()),
    }


def _id_counts(h5: h5py.File) -> dict[str, dict[str, int]]:
    mappings = {
        "category_counts": ("/samples/category_id", _reverse(CATEGORY_IDS)),
        "split_counts": ("/samples/split_id", _reverse(SPLIT_IDS)),
        "frequency_role_counts": ("/samples/frequency_role", _reverse(FREQUENCY_ROLE_IDS)),
    }
    out: dict[str, dict[str, int]] = {}
    for name, (path, reverse) in mappings.items():
        values = np.asarray(h5[path], dtype=np.int64)
        out[name] = {reverse.get(int(value), str(int(value))): int((values == value).sum()) for value in sorted(set(values.tolist()))}
    return out


def _readback_sha_gate(h5: h5py.File) -> dict[str, int | bool]:
    if "/sample_sha256" not in h5:
        return {"checked": 0, "mismatch_count": 0, "passed": False}
    mismatch = 0
    stored = h5["/sample_sha256"]
    wavefield = h5["/samples/wavefield"]
    for i in range(int(wavefield.shape[0])):
        expected = stored[i].decode("ascii") if isinstance(stored[i], bytes) else str(stored[i])
        if sha256_array(np.asarray(wavefield[i], dtype=np.float32)) != expected:
            mismatch += 1
    return {"checked": int(wavefield.shape[0]), "mismatch_count": mismatch, "passed": bool(mismatch == 0)}


def validate_dataset_file(path: Path, *, strict: bool, expected_samples: int | None = None, readback_check: bool = False) -> dict:
    with h5py.File(path, "r") as h5:
        schema = validate_v3_hdf5_schema(h5, expected_n=expected_samples, strict=strict)
        wavefield_stats = _scan_numeric_dataset(h5["/samples/wavefield"])
        velocity_stats = _scan_numeric_dataset(h5["/samples/velocity_mps"])
        source_stats = _scan_numeric_dataset(h5["/samples/source_map"])
        completed = np.asarray(h5["/completed_mask"], dtype=bool) if "/completed_mask" in h5 else np.zeros(schema["sample_count"], dtype=bool)
        gates = {
            "finite": {
                "passed": bool(wavefield_stats["passed"] and velocity_stats["passed"] and source_stats["passed"]),
                "wavefield": wavefield_stats,
                "velocity": velocity_stats,
                "source_map": source_stats,
            },
            "nonzero_wavefield": {"max_abs_wavefield": wavefield_stats["max_abs"], "passed": bool(float(wavefield_stats["max_abs"]) > 0.0)},
            "completed_mask": {"completed_count": int(completed.sum()), "passed": bool(completed.size == schema["sample_count"] and completed.all())},
            "time_axis": _time_axis_gate(h5),
            "cfl": _cfl_gate(h5),
            "source_reconstruction": _source_reconstruction_gate(h5),
            "free_surface": _free_surface_gate(h5["/samples/wavefield"]),
            "quality_passed": {"passed_count": int(np.asarray(h5["/quality/passed"], dtype=bool).sum()), "passed": bool(np.asarray(h5["/quality/passed"], dtype=bool).all())},
        }
        if readback_check:
            gates["readback_sha256"] = _readback_sha_gate(h5)
        payload = schema | {
            "dataset": str(path),
            "gates": gates,
            "frequency_coverage": _frequency_coverage(h5),
            **_id_counts(h5),
        }
        payload["strict_passed"] = bool(all(gate.get("passed", False) for gate in gates.values()))
        return payload


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _json_shape(value: object) -> list[int] | None:
    try:
        parsed = json.loads(_text(value))
        return [int(item) for item in parsed]
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _lwc84_config_protocol_gate(h5: h5py.File, config: dict) -> dict[str, object]:
    """Bind validation to the requested forward-model protocol, not just the file schema."""
    grid = config["grid"]
    saved = config["storage_grid"]
    time = config["time"]
    expected_time = np.arange(int(time["nt_out"]), dtype=np.float64) * float(time["dt_out_s"])
    actual_time = np.asarray(h5["time_s"], dtype=np.float64)
    expected_dt = float(time["dt_used_s"])
    actual_sample_dt = np.asarray(h5["dt_used_s"], dtype=np.float64)
    expected_solver_shape = [int(grid["nz"]), int(grid["nx"])]
    expected_saved_shape = [int(saved["nz"]), int(saved["nx"])]

    checks = {
        "config_sha256": _text(h5.attrs.get("config_sha256", ""))
        == str(config["config_sha256"]),
        "forward_protocol": _text(h5.attrs.get("forward_protocol", ""))
        == str(config["schema_version"]),
        "lwc_time_order_4_space_order_8": _text(h5.attrs.get("lwc_version", ""))
        == "LWC-84: fourth-order time, eighth-order centred space",
        "ricker_source": _text(h5.attrs.get("source_formula", "")).startswith("Ricker "),
        "top_reflecting_free_surface": "p(z=0)=0"
        in _text(h5.attrs.get("free_surface", "")),
        "left_right_bottom_cpml": "left,right,bottom; no top CPML"
        in _text(h5.attrs.get("cpml", "")),
        "solver_grid_401x401": expected_solver_shape == [401, 401]
        and _json_shape(h5.attrs.get("solver_grid_shape", "")) == expected_solver_shape,
        "saved_grid_201x201": expected_saved_shape == [201, 201]
        and _json_shape(h5.attrs.get("saved_grid_shape", "")) == expected_saved_shape
        and list(h5["wavefield"].shape[-2:]) == expected_saved_shape,
        "solver_extent_2km_x_2km": np.isclose(float(grid["lx_m"]), 2000.0, rtol=0.0, atol=1.0e-12)
        and np.isclose(float(grid["lz_m"]), 2000.0, rtol=0.0, atol=1.0e-12),
        "solver_spacing_5m": np.isclose(float(h5.attrs.get("solver_dx_m", np.nan)), float(grid["dx_m"]), rtol=0.0, atol=1.0e-12)
        and np.isclose(float(h5.attrs.get("solver_dz_m", np.nan)), float(grid["dz_m"]), rtol=0.0, atol=1.0e-12),
        "saved_spacing_10m": np.isclose(float(h5.attrs.get("saved_dx_m", np.nan)), float(saved["dx_m"]), rtol=0.0, atol=1.0e-12)
        and np.isclose(float(h5.attrs.get("saved_dz_m", np.nan)), float(saved["dz_m"]), rtol=0.0, atol=1.0e-12),
        "dt_requested": np.isclose(float(h5.attrs.get("dt_requested_s", np.nan)), float(time["dt_requested_s"]), rtol=0.0, atol=1.0e-15),
        "dt_used_exact": np.isclose(float(h5.attrs.get("dt_used_s", np.nan)), expected_dt, rtol=0.0, atol=1.0e-15)
        and bool(np.allclose(actual_sample_dt, expected_dt, rtol=0.0, atol=1.0e-15)),
        "snapshot_stride": int(h5.attrs.get("snapshot_stride", -1))
        == int(time["snapshot_stride"]),
        "output_alignment": np.isclose(
            expected_dt * int(time["snapshot_stride"]),
            float(time["dt_out_s"]),
            rtol=0.0,
            atol=1.0e-15,
        ),
        "time_axis": actual_time.shape == expected_time.shape
        and bool(np.allclose(actual_time, expected_time, rtol=0.0, atol=1.0e-12)),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "expected_dt_s": expected_dt,
        "observed_dt_s": float(h5.attrs.get("dt_used_s", np.nan)),
        "expected_snapshot_stride": int(time["snapshot_stride"]),
        "observed_snapshot_stride": int(h5.attrs.get("snapshot_stride", -1)),
        "expected_solver_grid": expected_solver_shape,
        "expected_saved_grid": expected_saved_shape,
    }


def _validate_file(
    path: Path,
    strict: bool,
    expected_samples: int | None,
    readback_check: bool,
    config: dict | None = None,
) -> dict:
    with h5py.File(path, "r") as h5:
        schema_version = h5.attrs.get("schema_version", "")
        if isinstance(schema_version, bytes):
            schema_version = schema_version.decode("utf-8")
        is_dataset_vds = bool(
            schema_version == LWC84_SCHEMA_VERSION
            and "wavefield" in h5
            and h5["wavefield"].is_virtual
            and h5.attrs.get("split", "") == "all"
        )
        is_hybrid_vds = bool(is_dataset_vds and "vds_source_datasets" in h5.attrs)
    if schema_version == LWC84_SCHEMA_VERSION:
        if is_hybrid_vds:
            summary = validate_lwc84_hybrid_vds(
                path, strict=strict, expected_n=expected_samples
            )
        elif is_dataset_vds:
            summary = validate_lwc84_dataset_vds(
                path, strict=strict, expected_n=expected_samples
            )
        else:
            summary = validate_lwc84_shard(
                path, strict=strict, expected_n=expected_samples, require_sidecar=strict
            )
        with h5py.File(path, "r") as h5:
            wavefield = h5["wavefield"]
            maximum = 0.0
            top_maximum = 0.0
            for index in range(int(wavefield.shape[0])):
                sample = np.asarray(wavefield[index])
                maximum = max(maximum, float(np.max(np.abs(sample))))
                top_maximum = max(top_maximum, float(np.max(np.abs(sample[:, 0, :]))))
            cfl = np.asarray(h5["cfl_2d"], dtype=np.float64)
            qmax = np.asarray(h5["lwc_qmax"], dtype=np.float64)
            time_s = np.asarray(h5["time_s"], dtype=np.float64)
            source_sum_error = float(
                np.max(np.abs(np.asarray(h5["source_map"]).sum(axis=(1, 2), dtype=np.float64) - 1.0))
            )
        gates = {
            "nonzero_wavefield": {"max_abs": maximum, "passed": maximum > 0.0},
            "free_surface": {"max_abs_top": top_maximum, "passed": top_maximum <= 1.0e-7},
            "cfl": {
                "maximum_cfl_2d": float(cfl.max(initial=0.0)),
                "maximum_lwc_q": float(qmax.max(initial=0.0)),
                "passed": bool(np.all(cfl <= 0.45 + 1.0e-12) and np.all(qmax <= 9.6 + 1.0e-12)),
            },
            "time_axis": {
                "count": int(time_s.size),
                "strictly_increasing": bool(time_s.size == 1 or np.all(np.diff(time_s) > 0.0)),
                "passed": bool(time_s.size == 1 or np.all(np.diff(time_s) > 0.0)),
            },
            "source_map": {"max_sum_error": source_sum_error, "passed": source_sum_error <= 2.0e-6},
        }
        if config is not None:
            with h5py.File(path, "r") as h5:
                gates["forward_protocol"] = _lwc84_config_protocol_gate(h5, config)
        return summary | {"gates": gates, "strict_passed": bool(all(gate["passed"] for gate in gates.values()))}
    return validate_dataset_file(path, strict=strict, expected_samples=expected_samples, readback_check=readback_check)


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    config = load_config(args.config) if args.config else None
    if dataset.is_dir():
        files = sorted(dataset.rglob("*.h5"))
        results = [
            _validate_file(path, args.strict, None, args.readback_check, config)
            | {"path": str(path)}
            for path in files
        ]
        payload = {"dataset": str(dataset), "file_count": len(files), "files": results, "sample_count": sum(r["sample_count"] for r in results)}
    elif dataset.suffix == ".json":
        rows = read_manifest(dataset)
        payload = {"dataset": str(dataset), "manifest_rows": len(rows)}
    else:
        payload = _validate_file(
            dataset,
            args.strict,
            args.expected_samples,
            args.readback_check,
            config,
        ) | {"dataset": str(dataset)}
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict:
        if "strict_passed" in payload:
            return 0 if payload["strict_passed"] else 2
        if "files" in payload:
            return 0 if all(file.get("strict_passed", False) for file in payload["files"]) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
