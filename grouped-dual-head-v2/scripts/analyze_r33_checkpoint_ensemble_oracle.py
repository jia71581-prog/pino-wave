#!/usr/bin/env python3
"""R33 train-holdout diagnostic: sweep an ensemble of two residual checkpoints.

The script never trains on or opens validation/test_id data.  It reduces each
already-opened train-holdout record to the quadratic error coefficients of

    prediction(w) = prediction_b + w * (prediction_a - prediction_b)

and then evaluates a preregistered global weight grid.  Per-record oracle
weights are reported only as a diversity diagnostic and are not a deployable
result.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
R26_PATH = SCRIPT_PATH.with_name("train_r26_tail_spectral_pilot.py")
SPEC = importlib.util.spec_from_file_location("r33_r26_components", R26_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R26 components: {R26_PATH}")
r26 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r26
SPEC.loader.exec_module(r26)
r25 = r26.r25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(
    path: Path, device: torch.device, *, model_kind: str
) -> torch.nn.Module:
    if str(model_kind) == "zero":
        class ZeroCorrection(torch.nn.Module):
            def forward(self, features, *, active=None):
                return torch.zeros(
                    (features.shape[0], features.shape[-2], features.shape[-1]),
                    device=features.device,
                    dtype=features.dtype,
                )

        return ZeroCorrection().to(device).eval()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model_class = {
        "spectral": r26.TailSpectralResidualUNet,
        "local": r25.CoarseResidualUNet,
    }[str(model_kind)]
    model = model_class(
        base_width=int(config.get("base_width", 32)),
        correction_cap=float(config.get("correction_cap", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def summarize(
    rows: Sequence[Mapping[str, Any]], weights: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for weight in weights:
        values = []
        for row in rows:
            square = (
                float(row["quadratic_a"]) * float(weight) ** 2
                + 2.0 * float(row["quadratic_b"]) * float(weight)
                + float(row["quadratic_c"])
            )
            values.append(
                math.sqrt(max(square, 0.0) / max(float(row["target_square"]), 1.0e-30))
            )
        array = np.asarray(values, dtype=np.float64)
        curves.append(
            {
                "weight_a": float(weight),
                "mean": float(array.mean()),
                "max": float(array.max()),
                "median": float(np.median(array)),
                "passed": bool(array.mean() <= 0.05 and array.max() <= 0.05),
            }
        )
    min_sum = min(curves, key=lambda item: (item["mean"] + item["max"], item["max"]))
    min_max = min(curves, key=lambda item: (item["max"], item["mean"]))
    return curves, dict(min_sum), dict(min_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-a-model", choices=("spectral", "local", "zero"), default="spectral")
    parser.add_argument("--checkpoint-b-model", choices=("spectral", "local", "zero"), default="spectral")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-min", type=float, default=-1.0)
    parser.add_argument("--weight-max", type=float, default=2.0)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R33 diagnostic requires CUDA")
    checkpoint_a = args.checkpoint_a.expanduser().resolve()
    checkpoint_b = args.checkpoint_b.expanduser().resolve()
    model_a = load_model(checkpoint_a, device, model_kind=args.checkpoint_a_model)
    model_b = load_model(checkpoint_b, device, model_kind=args.checkpoint_b_model)
    collection = r25.CacheCollection(args.holdout_cache, expected_subset="holdout")
    rows: list[dict[str, Any]] = []
    try:
        for file_index, local_index in collection.records:
            handle = collection.handles[file_index]
            sample_id = str(handle["sample_id"].asstr()[local_index])
            family = str(handle["family"].asstr()[local_index])
            static = torch.from_numpy(
                np.asarray(handle["static_features"][local_index], dtype=np.float32)
            ).to(device)
            f0 = float(handle["source_f0_hz"][local_index])
            t0 = float(handle["source_t0_s"][local_index])
            times = np.asarray(handle["time_s"][:], dtype=np.float32)
            coarse_all = handle["coarse_norm"][local_index]
            truth_all = handle["truth_norm"][local_index]
            quadratic_a = 0.0
            quadratic_b = 0.0
            quadratic_c = 0.0
            target_square = 0.0
            parent_square = 0.0
            for start in range(0, len(times), int(args.batch_size)):
                stop = min(start + int(args.batch_size), len(times))
                coarse = torch.from_numpy(
                    np.asarray(coarse_all[start:stop], dtype=np.float32)
                ).to(device)
                truth = torch.from_numpy(
                    np.asarray(truth_all[start:stop], dtype=np.float32)
                ).to(device)
                block = stop - start
                time_tensor = torch.from_numpy(times[start:stop]).to(device)
                f0_tensor = torch.full((block,), f0, device=device)
                t0_tensor = torch.full((block,), t0, device=device)
                features = r25.make_dynamic_features(
                    coarse,
                    static[None].expand(block, -1, -1, -1),
                    time_s=time_tensor,
                    source_f0_hz=f0_tensor,
                    source_t0_s=t0_tensor,
                )
                active = (time_tensor >= t0_tensor).float()
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if args.amp
                    else nullcontext()
                )
                with context:
                    correction_a = model_a(features, active=active).float()
                    correction_b = model_b(features, active=active).float()
                prediction_a = coarse + correction_a
                prediction_b = coarse + correction_b
                delta = (prediction_a - prediction_b).double()
                residual_b = (prediction_b - truth).double()
                quadratic_a += float(delta.square().sum())
                quadratic_b += float((residual_b * delta).sum())
                quadratic_c += float(residual_b.square().sum())
                target_square += float(truth.double().square().sum())
                parent_square += float((coarse.double() - truth.double()).square().sum())
            if quadratic_a > 0.0:
                oracle_weight = float(
                    np.clip(
                        -quadratic_b / quadratic_a,
                        float(args.weight_min),
                        float(args.weight_max),
                    )
                )
            else:
                oracle_weight = 0.0
            oracle_square = (
                quadratic_a * oracle_weight**2
                + 2.0 * quadratic_b * oracle_weight
                + quadratic_c
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "family": family,
                    "source_f0_hz": f0,
                    "quadratic_a": quadratic_a,
                    "quadratic_b": quadratic_b,
                    "quadratic_c": quadratic_c,
                    "target_square": target_square,
                    "parent_rel_l2": math.sqrt(parent_square / max(target_square, 1.0e-30)),
                    "checkpoint_a_rel_l2": math.sqrt(
                        max(quadratic_a + 2.0 * quadratic_b + quadratic_c, 0.0)
                        / max(target_square, 1.0e-30)
                    ),
                    "checkpoint_b_rel_l2": math.sqrt(
                        quadratic_c / max(target_square, 1.0e-30)
                    ),
                    "oracle_weight_a": oracle_weight,
                    "oracle_rel_l2": math.sqrt(
                        max(oracle_square, 0.0) / max(target_square, 1.0e-30)
                    ),
                }
            )
    finally:
        collection.close()

    count = int(round((float(args.weight_max) - float(args.weight_min)) / float(args.weight_step))) + 1
    weights = np.linspace(float(args.weight_min), float(args.weight_max), count)
    curves, best_sum, best_minmax = summarize(rows, weights)
    a_values = np.asarray([row["checkpoint_a_rel_l2"] for row in rows], dtype=np.float64)
    b_values = np.asarray([row["checkpoint_b_rel_l2"] for row in rows], dtype=np.float64)
    oracle_values = np.asarray([row["oracle_rel_l2"] for row in rows], dtype=np.float64)
    payload = {
        "schema": "r33_checkpoint_ensemble_train_holdout_oracle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "R28_already_opened_train_holdout_diagnostic_only",
        "selection_sha256": collection.selection_sha256,
        "checkpoint_a": str(checkpoint_a),
        "checkpoint_a_model": str(args.checkpoint_a_model),
        "checkpoint_a_sha256": sha256_file(checkpoint_a),
        "checkpoint_b": str(checkpoint_b),
        "checkpoint_b_model": str(args.checkpoint_b_model),
        "checkpoint_b_sha256": sha256_file(checkpoint_b),
        "weight_definition": "prediction_b + weight_a * (prediction_a - prediction_b)",
        "grid": {
            "min": float(args.weight_min),
            "max": float(args.weight_max),
            "step": float(args.weight_step),
            "count": int(len(weights)),
        },
        "checkpoint_a_metrics": {
            "mean": float(a_values.mean()), "max": float(a_values.max())
        },
        "checkpoint_b_metrics": {
            "mean": float(b_values.mean()), "max": float(b_values.max())
        },
        "best_mean_plus_max": best_sum,
        "best_minimax": best_minmax,
        "per_record_oracle": {
            "mean": float(oracle_values.mean()),
            "max": float(oracle_values.max()),
            "deployable": False,
        },
        "curves": curves,
        "records": rows,
        "absolute_goal": {
            "mean_lte_0p05": bool(best_minmax["mean"] <= 0.05),
            "max_lte_0p05": bool(best_minmax["max"] <= 0.05),
            "passed": bool(best_minmax["mean"] <= 0.05 and best_minmax["max"] <= 0.05),
        },
        "validation_opened": False,
        "test_id_opened": False,
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint_a_metrics": payload["checkpoint_a_metrics"],
        "checkpoint_b_metrics": payload["checkpoint_b_metrics"],
        "best_mean_plus_max": best_sum,
        "best_minimax": best_minmax,
        "per_record_oracle": payload["per_record_oracle"],
        "absolute_goal": payload["absolute_goal"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
