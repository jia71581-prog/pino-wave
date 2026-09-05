#!/usr/bin/env python3
"""Diagnose temporal-frequency and spatial-wavenumber errors on opened dev records."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

import train_r25_coarse_residual_operator as r25
import train_r39_hfs_tail_finetune as r39


@torch.inference_mode()
def predict_record(model, handle, local_index: int, *, device, batch_size: int, amp: bool):
    static = torch.from_numpy(
        np.asarray(handle["static_features"][local_index], dtype=np.float32)
    ).to(device)
    f0 = float(handle["source_f0_hz"][local_index])
    t0 = float(handle["source_t0_s"][local_index])
    time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
    coarse_ds = handle["coarse_norm"][local_index]
    truth = np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
    prediction_blocks: list[np.ndarray] = []
    coarse_blocks: list[np.ndarray] = []
    for start in range(0, len(time_s), int(batch_size)):
        stop = min(start + int(batch_size), len(time_s))
        coarse = torch.from_numpy(
            np.asarray(coarse_ds[start:stop], dtype=np.float32)
        ).to(device)
        block = stop - start
        time_block = torch.from_numpy(time_s[start:stop].astype(np.float32)).to(device)
        f0_tensor = torch.full((block,), f0, device=device)
        t0_tensor = torch.full((block,), t0, device=device)
        features = r25.make_dynamic_features(
            coarse,
            static[None].expand(block, -1, -1, -1),
            time_s=time_block,
            source_f0_hz=f0_tensor,
            source_t0_s=t0_tensor,
        )
        active = (time_block >= t0_tensor).float()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            correction = model(features, active=active)
        prediction_blocks.append((coarse + correction.float()).cpu().numpy())
        coarse_blocks.append(coarse.cpu().numpy())
    return (
        np.concatenate(prediction_blocks, axis=0),
        np.concatenate(coarse_blocks, axis=0),
        truth,
        time_s,
        f0,
        t0,
    )


def rfft_weights(count: int) -> np.ndarray:
    frequencies = count // 2 + 1
    weights = np.full(frequencies, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if count % 2 == 0:
        weights[-1] = 1.0
    return weights


def spectral_diagnostic(prediction, coarse, truth, time_s, f0):
    sample_count = int(prediction.shape[0])
    dt = float(np.median(np.diff(time_s)))
    frequencies = np.fft.rfftfreq(sample_count, d=dt)
    weights = rfft_weights(sample_count)
    pred_f = np.fft.rfft(prediction, axis=0, norm="ortho")
    coarse_f = np.fft.rfft(coarse, axis=0, norm="ortho")
    truth_f = np.fft.rfft(truth, axis=0, norm="ortho")
    error_f = pred_f - truth_f
    coarse_error_f = coarse_f - truth_f
    error_square = np.sum(np.abs(error_f) ** 2, axis=(1, 2), dtype=np.float64)
    coarse_error_square = np.sum(
        np.abs(coarse_error_f) ** 2, axis=(1, 2), dtype=np.float64
    )
    target_square = np.sum(np.abs(truth_f) ** 2, axis=(1, 2), dtype=np.float64)
    weighted_error = weights * error_square
    weighted_coarse_error = weights * coarse_error_square
    weighted_target = weights * target_square

    numerator = np.sum(np.conj(pred_f) * truth_f, axis=(1, 2), dtype=np.complex128)
    denominator = np.sum(np.abs(pred_f) ** 2, axis=(1, 2), dtype=np.float64)
    alpha = numerator / np.maximum(denominator, 1.0e-30)
    oracle_error = np.sum(
        np.abs(alpha[:, None, None] * pred_f - truth_f) ** 2,
        axis=(1, 2),
        dtype=np.float64,
    )

    total_error = float(weighted_error.sum())
    total_target = float(weighted_target.sum())
    top = np.argsort(weighted_error)[::-1][:16]
    top_bins = [
        {
            "index": int(index),
            "frequency_hz": float(frequencies[index]),
            "error_fraction": float(weighted_error[index] / max(total_error, 1.0e-30)),
            "target_fraction": float(weighted_target[index] / max(total_target, 1.0e-30)),
            "relative_l2_in_bin": float(
                math.sqrt(error_square[index] / max(target_square[index], 1.0e-30))
            ),
            "oracle_gain_amplitude": float(abs(alpha[index])),
            "oracle_gain_phase_rad": float(np.angle(alpha[index])),
        }
        for index in top
    ]
    bands = []
    for low, high in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 101)]:
        mask = (frequencies >= low) & (frequencies < high)
        band_error = float(weighted_error[mask].sum())
        band_target = float(weighted_target[mask].sum())
        bands.append(
            {
                "low_hz": low,
                "high_hz": high,
                "error_fraction": band_error / max(total_error, 1.0e-30),
                "target_fraction": band_target / max(total_target, 1.0e-30),
                "relative_l2": math.sqrt(band_error / max(band_target, 1.0e-30)),
            }
        )

    spatial_error = np.zeros((prediction.shape[1], prediction.shape[2] // 2 + 1))
    for start in range(0, sample_count, 16):
        block = prediction[start : start + 16] - truth[start : start + 16]
        transformed = np.fft.rfft2(block, axes=(1, 2), norm="ortho")
        spatial_error += np.sum(np.abs(transformed) ** 2, axis=0, dtype=np.float64)
    spatial_error *= rfft_weights(prediction.shape[2])[None, :]
    ky = np.fft.fftfreq(prediction.shape[1])[:, None] / 0.5
    kx = np.fft.rfftfreq(prediction.shape[2])[None, :] / 0.5
    radius = np.sqrt(ky * ky + kx * kx)
    radial_bands = []
    total_spatial = float(spatial_error.sum())
    for low, high in zip(np.linspace(0.0, 1.5, 11)[:-1], np.linspace(0.0, 1.5, 11)[1:]):
        mask = (radius >= low) & (radius < high)
        radial_bands.append(
            {
                "low_normalized_nyquist": float(low),
                "high_normalized_nyquist": float(high),
                "error_fraction": float(spatial_error[mask].sum() / max(total_spatial, 1.0e-30)),
            }
        )

    return {
        "stored_dt_s": dt,
        "source_f0_hz": float(f0),
        "candidate_rel_l2": math.sqrt(total_error / max(total_target, 1.0e-30)),
        "coarse_rel_l2": math.sqrt(
            float(weighted_coarse_error.sum()) / max(total_target, 1.0e-30)
        ),
        "per_frequency_complex_scalar_oracle_rel_l2": math.sqrt(
            float((weights * oracle_error).sum()) / max(total_target, 1.0e-30)
        ),
        "temporal_frequency_bands": bands,
        "top_error_frequency_bins": top_bins,
        "spatial_wavenumber_error_bands": radial_bands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    collection = r25.CacheCollection(args.cache, expected_subset="holdout")
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    config = checkpoint.get("model_config", {})
    model = r39.HFSTailSpectralResidualUNet(
        base_width=int(config.get("base_width", 32)),
        correction_cap=float(config.get("correction_cap", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    wanted = set(args.sample_id)
    records = {}
    for file_index, local_index in collection.records:
        handle = collection.handles[file_index]
        sample_id = str(handle["sample_id"].asstr()[local_index])
        if sample_id not in wanted:
            continue
        prediction, coarse, truth, time_s, f0, t0 = predict_record(
            model,
            handle,
            local_index,
            device=device,
            batch_size=int(args.batch_size),
            amp=bool(args.amp),
        )
        diagnostic = spectral_diagnostic(prediction, coarse, truth, time_s, f0)
        diagnostic.update(
            {
                "family": str(handle["family"].asstr()[local_index]),
                "source_t0_s": t0,
            }
        )
        records[sample_id] = diagnostic
    missing = sorted(wanted - set(records))
    payload = {
        "schema": "r40_frequency_residual_diagnostic_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "selection_sha256": collection.selection_sha256,
        "records": records,
        "missing_sample_ids": missing,
        "truth_use": "opened development diagnostic only; not a deployment input",
        "validation_opened": False,
        "test_id_opened": False,
    }
    args.output.expanduser().resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    collection.close()
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
