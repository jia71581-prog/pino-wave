#!/usr/bin/env python3
"""Diagnose R28 tail errors without opening validation or test data.

The analysis uses only the already-opened R28 train holdout.  It reports
record-relative errors for even/odd stored time indices and can evaluate the
arithmetic mean of multiple R28 checkpoints.  It is diagnostic evidence only;
it is not an independent accuracy result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
R28_PATH = SCRIPT_PATH.with_name("train_r28_expanded_tail_spectral.py")
SPEC = importlib.util.spec_from_file_location("r28_diagnostic_components", R28_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import R28 components: {R28_PATH}")
r28 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r28
SPEC.loader.exec_module(r28)
r26 = r28.r26
r25 = r28.r25


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model = r26.TailSpectralResidualUNet(
        base_width=int(config.get("base_width", 32)),
        correction_cap=float(config.get("correction_cap", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidate = np.asarray([row["candidate_rel_l2"] for row in rows], dtype=np.float64)
    parent = np.asarray([row["parent_rel_l2"] for row in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "candidate_mean": float(candidate.mean()),
        "candidate_max": float(candidate.max()),
        "candidate_median": float(np.median(candidate)),
        "parent_mean": float(parent.mean()),
        "parent_max": float(parent.max()),
        "mean_relative_improvement": float(1.0 - candidate.mean() / parent.mean()),
        "max_relative_improvement": float(1.0 - candidate.max() / parent.max()),
    }


@torch.inference_mode()
def evaluate_models(
    models: Sequence[torch.nn.Module],
    collection: Any,
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in r25.FAMILIES
    }
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

        candidate_frame_error: list[float] = []
        parent_frame_error: list[float] = []
        target_frame_square: list[float] = []
        disagreement_frame_square: list[float] = []
        for start in range(0, len(times), int(batch_size)):
            stop = min(start + int(batch_size), len(times))
            coarse = torch.from_numpy(
                np.asarray(coarse_all[start:stop], dtype=np.float32)
            ).to(device)
            truth = torch.from_numpy(
                np.asarray(truth_all[start:stop], dtype=np.float32)
            ).to(device)
            block = stop - start
            features = r25.make_dynamic_features(
                coarse,
                static[None].expand(block, -1, -1, -1),
                time_s=torch.from_numpy(times[start:stop]).to(device),
                source_f0_hz=torch.full((block,), f0, device=device),
                source_t0_s=torch.full((block,), t0, device=device),
            )
            active = torch.from_numpy(times[start:stop]).to(device) >= t0
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp
                else nullcontext()
            )
            corrections: list[torch.Tensor] = []
            with context:
                for model in models:
                    corrections.append(model(features, active=active.float()).float())
            stacked = torch.stack(corrections, dim=0)
            prediction = coarse + stacked.mean(dim=0)
            candidate_frame_error.extend(
                (prediction.double() - truth.double())
                .square()
                .sum(dim=(1, 2))
                .cpu()
                .tolist()
            )
            parent_frame_error.extend(
                (coarse.double() - truth.double())
                .square()
                .sum(dim=(1, 2))
                .cpu()
                .tolist()
            )
            target_frame_square.extend(
                truth.double().square().sum(dim=(1, 2)).cpu().tolist()
            )
            if len(models) > 1:
                centered = stacked.double() - stacked.double().mean(dim=0, keepdim=True)
                disagreement_frame_square.extend(
                    centered.square().mean(dim=0).sum(dim=(1, 2)).cpu().tolist()
                )
            else:
                disagreement_frame_square.extend([0.0] * block)

        candidate_error = np.asarray(candidate_frame_error, dtype=np.float64)
        parent_error = np.asarray(parent_frame_error, dtype=np.float64)
        target_square = np.asarray(target_frame_square, dtype=np.float64)
        disagreement = np.asarray(disagreement_frame_square, dtype=np.float64)
        indices = np.arange(len(times))

        def relative(mask: np.ndarray, numerator: np.ndarray) -> float:
            return float(
                math.sqrt(
                    float(numerator[mask].sum())
                    / max(float(target_square[mask].sum()), 1.0e-30)
                )
            )

        total_target = max(float(target_square.sum()), 1.0e-30)
        ranked = np.argsort(candidate_error)[::-1][:8]
        row = {
            "sample_id": sample_id,
            "family": family,
            "candidate_rel_l2": relative(indices >= 0, candidate_error),
            "parent_rel_l2": relative(indices >= 0, parent_error),
            "even_time_rel_l2": relative(indices % 2 == 0, candidate_error),
            "odd_time_rel_l2": relative(indices % 2 == 1, candidate_error),
            "first_half_rel_l2": relative(indices < len(times) // 2, candidate_error),
            "second_half_rel_l2": relative(indices >= len(times) // 2, candidate_error),
            "ensemble_disagreement_rel_l2": float(
                math.sqrt(float(disagreement.sum()) / total_target)
            ),
            "top_error_frames": [
                {
                    "frame_index": int(frame),
                    "time_s": float(times[frame]),
                    "record_error_fraction": float(candidate_error[frame] / max(candidate_error.sum(), 1.0e-30)),
                    "frame_error_over_record_target": float(candidate_error[frame] / total_target),
                }
                for frame in ranked
            ],
        }
        rows.append(row)
        per_family[family].append(row)

    aggregate = summarize(rows)
    return {
        "aggregate": aggregate,
        "per_family": {family: summarize(values) for family, values in per_family.items()},
        "absolute_goal": {
            "mean_lte_0p05": bool(aggregate["candidate_mean"] <= 0.05),
            "max_lte_0p05": bool(aggregate["candidate_max"] <= 0.05),
            "passed": bool(
                aggregate["candidate_mean"] <= 0.05
                and aggregate["candidate_max"] <= 0.05
            ),
        },
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    collection = r25.CacheCollection(args.holdout_cache, expected_subset="holdout")
    models: list[torch.nn.Module] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for path in args.checkpoint:
        model, checkpoint = load_model(path.expanduser().resolve(), device)
        models.append(model)
        checkpoint_rows.append(
            {
                "path": str(path.expanduser().resolve()),
                "epoch": int(checkpoint.get("epoch", -1)),
                "global_step": int(checkpoint.get("global_step", -1)),
                "sha256": r25.sha256_file(path.expanduser().resolve()),
            }
        )
    result = evaluate_models(
        models,
        collection,
        device=device,
        batch_size=int(args.batch_size),
        amp=bool(args.amp),
    )
    result.update(
        {
            "schema": "r28_tail_parity_ensemble_diagnostic_v1",
            "role": "already_opened_train_holdout_diagnostic_only",
            "checkpoints": checkpoint_rows,
            "checkpoint_aggregation": "arithmetic_mean_of_corrections",
            "validation_opened": False,
            "test_id_opened": False,
            "script_sha256": r25.sha256_file(SCRIPT_PATH),
        }
    )
    r25.atomic_json(result, args.output.expanduser().resolve())
    worst = max(result["records"], key=lambda row: row["candidate_rel_l2"])
    print(
        json.dumps(
            {
                "event": "R28_TAIL_DIAGNOSTIC_COMPLETE",
                "aggregate": result["aggregate"],
                "worst_record": worst,
                "absolute_goal": result["absolute_goal"],
                "validation_opened": False,
                "test_id_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
