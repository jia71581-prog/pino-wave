#!/usr/bin/env python3
"""Build R40 DCT-compressed frequency-residual caches from train-only records."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.fft import dctn

import build_r25_coarse_residual_cache as coarse_builder
import build_r40_frequency_manifest as manifest_builder
import train_r25_coarse_residual_operator as r25
import train_r39_hfs_tail_finetune as r39


SCHEMA = "r40_frequency_residual_cache_v2"


def complex_channels(value: np.ndarray) -> np.ndarray:
    return np.stack((value.real, value.imag), axis=1)


def dct_complex(value: np.ndarray, retained: int) -> np.ndarray:
    real = dctn(value.real.astype(np.float32), type=2, norm="ortho", axes=(-2, -1))
    imag = dctn(value.imag.astype(np.float32), type=2, norm="ortho", axes=(-2, -1))
    return (
        real[..., :retained, :retained]
        + 1j * imag[..., :retained, :retained]
    ).astype(np.complex64)


def spatial_static_dct(
    value: np.ndarray, retained: int
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = dctn(
        value.astype(np.float32), type=2, norm="ortho", axes=(-2, -1)
    )[..., :retained, :retained]
    scale = np.max(np.abs(coefficients), axis=(-2, -1), keepdims=True)
    safe_scale = np.maximum(scale, 1.0e-6)
    return (
        (coefficients / safe_scale).astype(np.float32),
        safe_scale[:, 0, 0].astype(np.float32),
    )


def rfft_weights(count: int) -> np.ndarray:
    weights = np.full(count // 2 + 1, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if count % 2 == 0:
        weights[-1] = 1.0
    return weights


@torch.inference_mode()
def base_prediction(
    model,
    coarse: np.ndarray,
    static: np.ndarray,
    *,
    time_s: np.ndarray,
    f0: float,
    t0: float,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    static_tensor = torch.from_numpy(static.astype(np.float32)).to(device)
    blocks: list[np.ndarray] = []
    for start in range(0, len(time_s), int(batch_size)):
        stop = min(start + int(batch_size), len(time_s))
        coarse_tensor = torch.from_numpy(coarse[start:stop].astype(np.float32)).to(device)
        block = stop - start
        time_tensor = torch.from_numpy(time_s[start:stop].astype(np.float32)).to(device)
        f0_tensor = torch.full((block,), float(f0), device=device)
        t0_tensor = torch.full((block,), float(t0), device=device)
        features = r25.make_dynamic_features(
            coarse_tensor,
            static_tensor[None].expand(block, -1, -1, -1),
            time_s=time_tensor,
            source_f0_hz=f0_tensor,
            source_t0_s=t0_tensor,
        )
        active = (time_tensor >= t0_tensor).float()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            correction = model(features, active=active)
        blocks.append((coarse_tensor + correction.float()).cpu().numpy())
    return np.concatenate(blocks, axis=0)


def create_cache(args: argparse.Namespace) -> dict:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != manifest_builder.SCHEMA:
        raise RuntimeError("unexpected R40 manifest schema")
    verification = dict(manifest)
    selection_sha = str(verification.pop("selection_sha256"))
    if manifest_builder.canonical_sha(verification) != selection_sha:
        raise RuntimeError("R40 manifest digest mismatch")
    if bool(manifest.get("validation_opened")) or bool(manifest.get("test_id_opened")):
        raise RuntimeError("R40 manifest evidence boundary violated")

    source_h5 = Path(str(manifest["source_h5"])).resolve()
    if source_h5.stat().st_size != int(manifest["source_h5_byte_count"]):
        raise RuntimeError("source HDF5 byte count changed")
    with h5py.File(source_h5, "r", swmr=True) as source_handle:
        if str(source_handle.attrs.get("manifest_sha256", "")) != str(
            manifest["source_manifest_sha256"]
        ):
            raise RuntimeError("source manifest identity changed")
        if str(source_handle.attrs.get("config_sha256", "")) != str(
            manifest["source_config_sha256"]
        ):
            raise RuntimeError("source config identity changed")

    checkpoint_path = Path(str(manifest["base_checkpoint"])).resolve()
    if coarse_builder.sha256_file(checkpoint_path) != str(
        manifest["base_checkpoint_sha256"]
    ):
        raise RuntimeError("R40 base checkpoint hash changed")

    subset_key = f"{args.subset}_records"
    all_records = coarse_builder.rows_to_records(manifest[subset_key])
    records = all_records[int(args.shard_index) :: int(args.shard_count)]
    if not records:
        raise RuntimeError("empty R40 shard")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    temporary.unlink(missing_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R40 cache generation requires CUDA")
    torch.cuda.set_device(device)
    time_s, x_m, z_m = coarse_builder._read_axes(source_h5)
    stored_dt_s, dx_m, dz_m = coarse_builder._validate_axes(
        time_s,
        x_m,
        z_m,
        internal_dt_s=coarse_builder.INTERNAL_DT_S,
    )
    if len(time_s) != coarse_builder.STORED_TIME_COUNT:
        raise RuntimeError("unexpected stored time count")
    frequencies = np.fft.rfftfreq(len(time_s), d=stored_dt_s)
    policy = manifest["frequency_policy"]
    frequency_indices = np.flatnonzero(
        (frequencies >= float(policy["minimum_hz"]))
        & (frequencies <= float(policy["maximum_hz"]))
    ).astype(np.int64)
    if len(frequency_indices) < 2:
        raise RuntimeError("R40 selected no useful frequencies")
    selected_frequencies = frequencies[frequency_indices].astype(np.float32)
    retained = int(manifest["spatial_compression"]["retained_shape"][0])
    if manifest["spatial_compression"]["retained_shape"] != [retained, retained]:
        raise RuntimeError("R40 requires square DCT retention")

    coarse_builder._warm_up_cuda(
        dx_m=dx_m,
        dz_m=dz_m,
        internal_dt_s=coarse_builder.INTERNAL_DT_S,
        npml=coarse_builder.NPML,
        c_ref_mps=coarse_builder.C_REF_MPS,
        device=device,
    )
    solver = coarse_builder.LWC84CPMLSolver(
        grid=coarse_builder.AcousticGrid(
            nx=coarse_builder.GRID_SIZE,
            nz=coarse_builder.GRID_SIZE,
            dx_m=dx_m,
            dz_m=dz_m,
            lx_m=float(x_m[-1] - x_m[0]),
            lz_m=float(z_m[-1] - z_m[0]),
            centering="node",
        ),
        boundaries=coarse_builder.BoundaryConfig(npml=coarse_builder.NPML),
        dt_s=coarse_builder.INTERNAL_DT_S,
        output_times_s=time_s,
        c_ref_mps=coarse_builder.C_REF_MPS,
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model = r39.HFSTailSpectralResidualUNet(
        base_width=int(config.get("base_width", 32)),
        correction_cap=float(config.get("correction_cap", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    text_dtype = h5py.string_dtype(encoding="utf-8")
    count = len(records)
    frequency_count = len(frequency_indices)
    weights = rfft_weights(len(time_s))
    started = time.perf_counter()
    record_errors: list[float] = []
    try:
        with h5py.File(temporary, "x") as cache:
            cache.attrs["schema"] = SCHEMA
            cache.attrs["status"] = "building"
            cache.attrs["selection_sha256"] = selection_sha
            cache.attrs["manifest_sha256"] = coarse_builder.sha256_file(manifest_path)
            cache.attrs["subset"] = args.subset
            cache.attrs["shard_index"] = int(args.shard_index)
            cache.attrs["shard_count"] = int(args.shard_count)
            cache.attrs["base_checkpoint_sha256"] = str(
                manifest["base_checkpoint_sha256"]
            )
            cache.attrs["truth_policy"] = "train_supervision_or_opened_development_only"
            cache.attrs["stored_dt_s"] = float(stored_dt_s)
            cache.create_dataset("frequency_indices", data=frequency_indices)
            cache.create_dataset("frequency_hz", data=selected_frequencies)
            cache.create_dataset(
                "source_index",
                data=np.asarray([record.source_index for record in records], dtype=np.int64),
            )
            cache.create_dataset(
                "sample_id",
                data=np.asarray([record.sample_id for record in records], dtype=object),
                dtype=text_dtype,
            )
            cache.create_dataset(
                "group_id",
                data=np.asarray([record.group_id for record in records], dtype=object),
                dtype=text_dtype,
            )
            cache.create_dataset(
                "family",
                data=np.asarray([record.medium_type for record in records], dtype=object),
                dtype=text_dtype,
            )
            f0_ds = cache.create_dataset("source_f0_hz", shape=(count,), dtype=np.float32)
            t0_ds = cache.create_dataset("source_t0_s", shape=(count,), dtype=np.float32)
            field_scale_ds = cache.create_dataset("field_scale", shape=(count,), dtype=np.float32)
            frequency_scale_ds = cache.create_dataset(
                "frequency_scale", shape=(count, frequency_count), dtype=np.float32
            )
            static_ds = cache.create_dataset(
                "static_dct_norm",
                shape=(count, 7, retained, retained),
                dtype=np.float16,
                chunks=(1, 7, retained, retained),
                compression="lzf",
                shuffle=True,
            )
            static_scale_ds = cache.create_dataset(
                "static_dct_scale", shape=(count, 7), dtype=np.float32
            )
            base_ds = cache.create_dataset(
                "base_dct_norm",
                shape=(count, frequency_count, 2, retained, retained),
                dtype=np.float16,
                chunks=(1, 1, 2, retained, retained),
                compression="lzf",
                shuffle=True,
            )
            residual_ds = cache.create_dataset(
                "residual_dct_norm",
                shape=(count, frequency_count, 2, retained, retained),
                dtype=np.float16,
                chunks=(1, 1, 2, retained, retained),
                compression="lzf",
                shuffle=True,
            )
            target_total_ds = cache.create_dataset(
                "target_square_total", shape=(count,), dtype=np.float64
            )
            base_unselected_ds = cache.create_dataset(
                "base_error_square_unselected", shape=(count,), dtype=np.float64
            )
            if args.subset == "holdout":
                base_full_ds = cache.create_dataset(
                    "base_spectrum_selected",
                    shape=(
                        count,
                        frequency_count,
                        2,
                        coarse_builder.GRID_SIZE,
                        coarse_builder.GRID_SIZE,
                    ),
                    dtype=np.float32,
                    chunks=(1, 1, 2, coarse_builder.GRID_SIZE, coarse_builder.GRID_SIZE),
                    compression="lzf",
                    shuffle=True,
                )
                truth_full_ds = cache.create_dataset(
                    "truth_spectrum_selected",
                    shape=(
                        count,
                        frequency_count,
                        2,
                        coarse_builder.GRID_SIZE,
                        coarse_builder.GRID_SIZE,
                    ),
                    dtype=np.float32,
                    chunks=(1, 1, 2, coarse_builder.GRID_SIZE, coarse_builder.GRID_SIZE),
                    compression="lzf",
                    shuffle=True,
                )

            solver_batch = int(args.solver_batch_size)
            for batch_start in range(0, count, solver_batch):
                batch_records = records[batch_start : batch_start + solver_batch]
                velocity, source = coarse_builder._read_generation_inputs(
                    source_h5, batch_records
                )
                torch.cuda.synchronize(device)
                result = solver.simulate(
                    velocity,
                    source_x_m=source["source_x_m"],
                    source_z_m=source["source_z_m"],
                    source_f0_hz=source["source_f0_hz"],
                    source_t0_s=source["source_t0_s"],
                    source_amplitude=source["source_amplitude"],
                )
                torch.cuda.synchronize(device)
                coarse_builder._validate_solver_batch(
                    result,
                    velocity,
                    records=batch_records,
                    stored_time_count=coarse_builder.STORED_TIME_COUNT,
                )
                coarse_batch = np.asarray(result.wavefield, dtype=np.float32)
                source_indices = [record.source_index for record in batch_records]
                with h5py.File(source_h5, "r", swmr=True) as source_handle:
                    truth_batch = np.stack(
                        [
                            np.asarray(source_handle["wavefield"][index], dtype=np.float32)
                            for index in source_indices
                        ],
                        axis=0,
                    )

                for local, record in enumerate(batch_records):
                    row = batch_start + local
                    coarse_time = coarse_batch[local]
                    truth_time = truth_batch[local]
                    field_scale = max(float(np.max(np.abs(coarse_time))), 1.0e-12)
                    coarse_norm = coarse_time / field_scale
                    truth_norm = truth_time / field_scale
                    static = coarse_builder.static_features(
                        velocity[local],
                        x_m=x_m,
                        z_m=z_m,
                        source_x_m=float(source["source_x_m"][local]),
                        source_z_m=float(source["source_z_m"][local]),
                    ).astype(np.float32)
                    f0 = float(source["source_f0_hz"][local])
                    t0 = float(source["source_t0_s"][local])
                    predicted = base_prediction(
                        model,
                        coarse_norm,
                        static,
                        time_s=time_s,
                        f0=f0,
                        t0=t0,
                        device=device,
                        batch_size=int(args.model_batch_size),
                        amp=bool(args.amp),
                    )
                    base_f = np.fft.rfft(predicted, axis=0, norm="ortho").astype(
                        np.complex64
                    )
                    truth_f = np.fft.rfft(truth_norm, axis=0, norm="ortho").astype(
                        np.complex64
                    )
                    base_selected = base_f[frequency_indices]
                    truth_selected = truth_f[frequency_indices]
                    base_coeff = dct_complex(base_selected, retained)
                    residual_coeff = dct_complex(
                        truth_selected - base_selected, retained
                    )
                    frequency_scale = np.max(
                        np.abs(base_coeff), axis=(-2, -1)
                    ).astype(np.float32)
                    frequency_scale = np.maximum(frequency_scale, 1.0e-6)
                    base_coeff /= frequency_scale[:, None, None]
                    residual_coeff /= frequency_scale[:, None, None]

                    error_by_frequency = np.sum(
                        np.abs(base_f - truth_f) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                    target_by_frequency = np.sum(
                        np.abs(truth_f) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )
                    selected_mask = np.zeros(len(frequencies), dtype=bool)
                    selected_mask[frequency_indices] = True
                    total_error = float((weights * error_by_frequency).sum())
                    total_target = float((weights * target_by_frequency).sum())
                    record_error = math.sqrt(total_error / max(total_target, 1.0e-30))
                    record_errors.append(record_error)

                    f0_ds[row] = f0
                    t0_ds[row] = t0
                    field_scale_ds[row] = field_scale
                    frequency_scale_ds[row] = frequency_scale
                    static_dct_norm, static_dct_scale = spatial_static_dct(
                        static, retained
                    )
                    static_ds[row] = static_dct_norm.astype(np.float16)
                    static_scale_ds[row] = static_dct_scale
                    base_ds[row] = complex_channels(base_coeff).astype(np.float16)
                    residual_ds[row] = complex_channels(residual_coeff).astype(np.float16)
                    target_total_ds[row] = total_target
                    base_unselected_ds[row] = float(
                        (weights[~selected_mask] * error_by_frequency[~selected_mask]).sum()
                    )
                    if args.subset == "holdout":
                        base_full_ds[row] = complex_channels(base_selected).astype(np.float32)
                        truth_full_ds[row] = complex_channels(truth_selected).astype(np.float32)
                    print(
                        json.dumps(
                            {
                                "event": "cached_record",
                                "subset": args.subset,
                                "shard": int(args.shard_index),
                                "row": row,
                                "count": count,
                                "sample_id": record.sample_id,
                                "family": record.medium_type,
                                "base_rel_l2": record_error,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    cache.flush()
                del result, coarse_batch, truth_batch
                torch.cuda.empty_cache()

            cache.attrs["base_record_rel_l2_mean"] = float(np.mean(record_errors))
            cache.attrs["base_record_rel_l2_max"] = float(np.max(record_errors))
            cache.attrs["status"] = "complete"
            cache.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema": "r40_frequency_residual_cache_summary_v2",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": coarse_builder.sha256_file(output),
        "selection_sha256": selection_sha,
        "manifest_sha256": coarse_builder.sha256_file(manifest_path),
        "base_checkpoint_sha256": str(manifest["base_checkpoint_sha256"]),
        "subset": args.subset,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "record_count": count,
        "frequency_count": frequency_count,
        "frequency_min_hz": float(selected_frequencies.min()),
        "frequency_max_hz": float(selected_frequencies.max()),
        "dct_retained": retained,
        "base_record_rel_l2_mean": float(np.mean(record_errors)),
        "base_record_rel_l2_max": float(np.max(record_errors)),
        "elapsed_seconds": time.perf_counter() - started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    coarse_builder.atomic_json(summary, output.with_suffix(output.suffix + ".summary.json"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subset", choices=("fit", "holdout"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--solver-batch-size", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index")
    create_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
