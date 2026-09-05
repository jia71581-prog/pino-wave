#!/usr/bin/env python3
"""Materialize leakage-safe 401-frame WFP validation predictions on one GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.eikonal import (  # noqa: E402
    debiased_grid_eikonal_travel_time,
)
from saved_time_phase_operator_v4.exterior_cpml import (  # noqa: E402
    ExteriorCPMLContract,
    extend_velocity_to_exterior,
)
from saved_time_phase_operator_v4.phase_carrier import (  # noqa: E402
    rotate_complex_pairs,
    travel_phase_carrier,
)
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator  # noqa: E402


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")
FREQUENCY_COUNT = 64
FRAME_COUNT = 401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cpml_features(cache_path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(cache_path, "r", swmr=True) as cache:
        if cache.attrs.get("status") != "complete":
            raise RuntimeError("training cache is not complete")
        physical_scale = float(cache.attrs["physical_scale"])
        profiles = {
            key: np.asarray(cache["profiles"][key]) for key in cache["profiles"]
        }
    sigma_max = max(
        float(np.max(profiles["sigma_x"])),
        float(np.max(profiles["sigma_z"])),
        1.0,
    )
    alpha_max = max(
        float(np.max(profiles["alpha_x"])),
        float(np.max(profiles["alpha_z"])),
        1.0,
    )
    features = np.stack(
        (
            profiles["sigma_x"] / sigma_max,
            profiles["sigma_z"] / sigma_max,
            (profiles["kappa_x"] - 1.0) / 2.0,
            (profiles["kappa_z"] - 1.0) / 2.0,
            profiles["alpha_x"] / alpha_max,
            profiles["alpha_z"] / alpha_max,
            profiles["active_x"].astype(np.float32),
            profiles["active_z"].astype(np.float32),
        ),
        axis=0,
    ).astype(np.float32)
    if features.shape != (8, 221, 241) or physical_scale != 1.0e-8:
        raise RuntimeError("CPML feature or physical-scale contract drift")
    return features, physical_scale


def build_medium(
    velocity: np.ndarray,
    family: str,
    profile_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    background = (
        np.asarray(velocity, dtype=np.float32)
        if family == "uniform"
        else gaussian_filter(velocity, sigma=2.0, mode="nearest").astype(np.float32)
    )
    exterior = np.asarray(
        extend_velocity_to_exterior(background, ExteriorCPMLContract()),
        dtype=np.float32,
    )
    log_velocity = np.log(np.maximum(exterior, 1.0))
    grad_z, grad_x = np.gradient(log_velocity)
    physical_mask = np.zeros((221, 241), dtype=np.float32)
    physical_mask[:201, 20:221] = 1.0
    medium = np.concatenate(
        (
            np.stack(
                (
                    (exterior - 4500.0) / 2500.0,
                    20.0 * grad_x,
                    20.0 * grad_z,
                ),
                axis=0,
            ),
            profile_features,
            physical_mask[None],
        ),
        axis=0,
    ).astype(np.float32)
    return background, medium


def source_batch(
    source_map: np.ndarray,
    source_wavelet: np.ndarray,
    source_parameters: tuple[float, float, float, float],
    frequencies: np.ndarray,
    frequency_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wavelet_fft = np.fft.rfft(source_wavelet, norm="ortho")
    wavelet_fft /= max(float(np.max(np.abs(wavelet_fft))), 1.0e-12)
    sx, sz, f0, t0 = source_parameters
    x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
    z = np.arange(221, dtype=np.float32) * 10.0
    xx, zz = np.meshgrid(x, z)
    extended_source = np.zeros((221, 241), dtype=np.float32)
    extended_source[:201, 20:221] = source_map
    rows, scalars = [], []
    for frequency in frequencies:
        wave = wavelet_fft[int(frequency)]
        rows.append(
            np.stack(
                (
                    extended_source,
                    extended_source * float(wave.real),
                    extended_source * float(wave.imag),
                    np.clip((xx - sx) / 2000.0, -1.2, 1.2),
                    np.clip((zz - sz) / 2000.0, -0.2, 1.2),
                ),
                axis=0,
            ).astype(np.float32)
        )
        scalars.append(
            (
                float(frequency_hz[int(frequency)]) / 200.0,
                f0 / 30.0,
                t0 / 0.2,
                float(wave.real),
                float(wave.imag),
            )
        )
    return np.stack(rows), np.asarray(scalars, dtype=np.float32)


@torch.inference_mode()
def predict_record(
    model: BackgroundFrequencyOperator,
    device: torch.device,
    medium: np.ndarray,
    source_map: np.ndarray,
    source_wavelet: np.ndarray,
    source_parameters: tuple[float, float, float, float],
    physical_travel: np.ndarray,
    frequency_hz: np.ndarray,
    physical_scale: float,
    frequency_batch: int,
) -> np.ndarray:
    coefficients = torch.zeros(
        (201, 201, 201), dtype=torch.complex64, device=device
    )
    medium_tensor = torch.from_numpy(medium)[None].to(device)
    travel_tensor = torch.from_numpy(physical_travel).to(device)
    for start in range(0, FREQUENCY_COUNT, frequency_batch):
        frequencies = np.arange(
            start, min(start + frequency_batch, FREQUENCY_COUNT), dtype=np.int64
        )
        source, scalars = source_batch(
            source_map,
            source_wavelet,
            source_parameters,
            frequencies,
            frequency_hz,
        )
        count = len(frequencies)
        physical, _ = model(
            medium_tensor.expand(count, -1, -1, -1),
            torch.from_numpy(source).to(device),
            torch.from_numpy(scalars).to(device),
        )
        frequencies_tensor = torch.from_numpy(frequency_hz[frequencies]).to(
            device=device, dtype=torch.float32
        )
        physical = rotate_complex_pairs(
            physical,
            travel_phase_carrier(
                travel_tensor[None].expand(count, -1, -1), frequencies_tensor
            ),
        )
        coefficients[frequencies] = torch.complex(physical[:, 0], physical[:, 1])
    prediction = torch.fft.irfft(
        coefficients, n=FRAME_COUNT, dim=0, norm="ortho"
    ).float()
    prediction.mul_(physical_scale)
    prediction[:, 0, :] = 0.0
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("non-finite materialized prediction")
    return prediction.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--frequency-batch", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker index outside worker count")
    if args.worker_count != 4:
        raise RuntimeError("frozen validation requires four workers")

    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if manifest.get("schema") not in {
        "transfer_dg_wfp_validation600_public_manifest_v1",
        "transfer_dg_wfp_validation30_public_manifest_v1",
    }:
        raise RuntimeError("unexpected validation manifest")
    if manifest.get("validation_future_truth_opened") or manifest.get("test_id_opened"):
        raise RuntimeError("public validation manifest violates split boundary")
    if sha256(args.manifest) != bindings["validation_manifest_sha256"]:
        raise RuntimeError("validation manifest binding drift")
    if sha256(args.checkpoint) != bindings["checkpoint_sha256"]:
        raise RuntimeError("checkpoint binding drift")
    if sha256(Path(__file__)) != bindings["predictor_sha256"]:
        raise RuntimeError("predictor binding drift")
    if args.frequency_batch != int(prereg["prediction"]["frequency_batch"]):
        raise RuntimeError("frequency batch differs from preregistration")

    rows = manifest["records"][args.worker_index :: args.worker_count]
    if args.max_records:
        rows = rows[: args.max_records]
    if not rows:
        raise RuntimeError("empty validation prediction shard")
    source_path = Path(manifest["source_h5"])
    profile_features, physical_scale = cpml_features(args.training_cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = checkpoint["identity"]
    if int(identity["width"]) != 64 or int(identity["rank"]) != 32 or int(identity["depth"]) != 6:
        raise RuntimeError("checkpoint is not the selected high-capacity model")
    device = torch.device("cuda")
    model = BackgroundFrequencyOperator(
        medium_channels=12,
        source_channels=5,
        width=64,
        rank=32,
        depth=6,
        arm="wfp",
        radii=(1, 2, 3, 4, 5, 6),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    started = time.time()
    record_runtimes = []
    top_maximum = 0.0
    try:
        with h5py.File(source_path, "r", swmr=True) as source, h5py.File(partial, "x") as output:
            time_s = np.asarray(source["time_s"], dtype=np.float64)
            if len(time_s) != FRAME_COUNT or not np.allclose(np.diff(time_s), 0.0025):
                raise RuntimeError("validation time-axis contract drift")
            frequency_hz = np.fft.rfftfreq(FRAME_COUNT, d=0.0025)[:FREQUENCY_COUNT].astype(np.float32)
            output.attrs["schema"] = "transfer_dg_wfp_validation401_prediction_shard_v1"
            output.attrs["status"] = "building"
            output.attrs["worker_index"] = args.worker_index
            output.attrs["worker_count"] = args.worker_count
            output.attrs["frame_count"] = FRAME_COUNT
            output.attrs["frequency_count"] = FREQUENCY_COUNT
            output.attrs["checkpoint_sha256"] = bindings["checkpoint_sha256"]
            output.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
            output.attrs["model_input_wavefield_frames"] = 0
            output.attrs["validation_future_truth_read"] = False
            output.attrs["test_id_opened"] = False
            text_dtype = h5py.string_dtype("utf-8")
            output.create_dataset("sample_id", shape=(len(rows),), dtype=text_dtype)
            output.create_dataset("family", shape=(len(rows),), dtype=text_dtype)
            output.create_dataset("source_index", shape=(len(rows),), dtype=np.int64)
            output.create_dataset("time_s", data=time_s)
            prediction_ds = output.create_dataset(
                "prediction",
                shape=(len(rows), FRAME_COUNT, 201, 201),
                dtype=np.float32,
                chunks=(1, 4, 201, 201),
                compression="lzf",
            )
            for local, row in enumerate(rows):
                record_started = time.perf_counter()
                index = int(row["source_index"])
                family = str(row["family"])
                velocity = np.asarray(source["velocity_mps"][index], dtype=np.float32)
                background, medium = build_medium(velocity, family, profile_features)
                sx = float(source["source_x_m"][index])
                sz = float(source["source_z_m"][index])
                f0 = float(source["source_f0_hz"][index])
                t0 = float(source["source_t0_s"][index])
                physical_travel = debiased_grid_eikonal_travel_time(
                    background,
                    source_coordinates_m=[(sz, sx)],
                    dx_m=10.0,
                    dz_m=10.0,
                )[0].astype(np.float32)
                prediction = predict_record(
                    model,
                    device,
                    medium,
                    np.asarray(source["source_map"][index], dtype=np.float32),
                    np.asarray(source["source_wavelet"][index], dtype=np.float32),
                    (sx, sz, f0, t0),
                    physical_travel,
                    frequency_hz,
                    physical_scale,
                    args.frequency_batch,
                )
                if prediction.shape != (FRAME_COUNT, 201, 201):
                    raise RuntimeError("materialized prediction shape drift")
                top_maximum = max(top_maximum, float(np.max(np.abs(prediction[:, 0]))))
                output["sample_id"][local] = row["sample_id"]
                output["family"][local] = family
                output["source_index"][local] = index
                prediction_ds[local] = prediction
                output.flush()
                elapsed = time.perf_counter() - record_started
                record_runtimes.append(elapsed)
                print(json.dumps({
                    "event": "prediction_record",
                    "worker": args.worker_index,
                    "record": local + 1,
                    "of": len(rows),
                    "sample_id": row["sample_id"],
                    "runtime_s": elapsed,
                    "elapsed_s": time.time() - started,
                }, sort_keys=True), flush=True)
            output.attrs["top_pressure_max_abs"] = top_maximum
            output.attrs["status"] = "complete"
            output.flush()
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)

    summary = {
        "schema": "transfer_dg_wfp_validation401_prediction_summary_v1",
        "status": "complete",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "record_count": len(rows),
        "frame_count": FRAME_COUNT,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "manifest_selection_sha256": manifest["selection_sha256"],
        "runtime_mean_s": float(np.mean(record_runtimes)),
        "runtime_p95_s": float(np.quantile(record_runtimes, 0.95)),
        "top_pressure_max_abs": top_maximum,
        "model_input_wavefield_frames": 0,
        "predictions_serialized": True,
        "validation_future_truth_read": False,
        "test_id_opened": False,
        "smoke": bool(args.max_records),
        "elapsed_s": time.time() - started,
    }
    atomic_json(summary, summary_path)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
