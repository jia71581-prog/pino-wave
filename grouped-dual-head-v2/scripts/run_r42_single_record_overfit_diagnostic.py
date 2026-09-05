#!/usr/bin/env python3
"""Train-only R42 diagnostic on one cached record.

The diagnostic deliberately memorizes all retained frequency/DCT residuals of
one fit record.  It does not read the opened development holdout, R29B, final
validation, or test-ID data.  Its purpose is to distinguish an optimization
failure (initialization/objective) from a cross-record generalization failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def load_r40_module(path: Path):
    spec = importlib.util.spec_from_file_location("r40_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import R40 training module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_record_rel_l2(
    prediction: torch.Tensor,
    residual: torch.Tensor,
    frequency_scale: torch.Tensor,
    frequency_weight: torch.Tensor,
    *,
    irreducible_error_square: float,
    target_square_total: float,
) -> float:
    error = prediction.float() - residual.float()
    error_square = error.square().sum(dim=(1, 2, 3))
    selected = torch.sum(
        frequency_weight.float() * frequency_scale.float().square() * error_square
    )
    total = float(irreducible_error_square) + float(selected.detach().cpu())
    return math.sqrt(total / max(float(target_square_total), 1.0e-30))


@torch.inference_mode()
def predict_all(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    blocks = []
    for start in range(0, features.shape[0], int(batch_size)):
        blocks.append(model(features[start : start + int(batch_size)]).float().cpu())
    return torch.cat(blocks, dim=0)


def initialize_output(model: torch.nn.Module, mode: str) -> None:
    head = model.head[-1]
    if mode == "zero":
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    elif mode == "small_random":
        torch.nn.init.normal_(head.weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(head.bias)
    else:
        raise ValueError(mode)


def train_variant(
    r40,
    *,
    name: str,
    initialization: str,
    objective: str,
    features: torch.Tensor,
    residual: torch.Tensor,
    frequency_scale: torch.Tensor,
    target_square_total: torch.Tensor,
    frequency_weight: torch.Tensor,
    family_weight: torch.Tensor,
    irreducible_error_square: float,
    target_square_value: float,
    width: int,
    modes: int,
    blocks: int,
    correction_cap: float,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = features.device
    model = r40.FrequencyResidualFNO(
        width=width,
        modes=modes,
        blocks=blocks,
        correction_cap=correction_cap,
    ).to(device)
    initialize_output(model, initialization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 991)
    frequency_count = int(features.shape[0])
    evaluation_steps = {0, 1, 5, 10, 25, 50, 100, 200, int(steps)}
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_loss = None
    last_gradient_norm = None

    for step in range(0, int(steps) + 1):
        if step in evaluation_steps:
            prediction = predict_all(model, features, batch_size=batch_size)
            rel_l2 = exact_record_rel_l2(
                prediction,
                residual.cpu(),
                frequency_scale.cpu(),
                frequency_weight.cpu(),
                irreducible_error_square=irreducible_error_square,
                target_square_total=target_square_value,
            )
            coefficient_recovery = 1.0 - float(
                (prediction - residual.cpu()).square().sum()
                / residual.cpu().square().sum().clamp_min(1.0e-12)
            )
            trajectory.append(
                {
                    "step": step,
                    "record_rel_l2": rel_l2,
                    "coefficient_energy_recovery": coefficient_recovery,
                    "last_loss": last_loss,
                    "last_gradient_norm": last_gradient_norm,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "single_record_evaluation",
                        "variant": name,
                        **trajectory[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step == int(steps):
            break

        indices = torch.randperm(frequency_count, generator=generator)[
            : min(int(batch_size), frequency_count)
        ].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features.index_select(0, indices))
        target = residual.index_select(0, indices)
        if objective == "physical":
            loss, _ = r40.frequency_objective(
                prediction,
                target,
                frequency_scale=frequency_scale.index_select(0, indices),
                target_square_total=target_square_total.index_select(0, indices),
                frequency_weight=frequency_weight.index_select(0, indices),
                family_weight=family_weight.index_select(0, indices),
                frequency_count=frequency_count,
                tail_weight=1.0,
                hinge_weight=2.0,
                shape_weight=0.002,
            )
        elif objective == "coefficient_mse":
            loss = F.mse_loss(prediction.float(), target.float())
        elif objective == "coefficient_relative":
            error_square = (prediction.float() - target.float()).square().sum()
            target_square = target.float().square().sum().clamp_min(1.0e-8)
            loss = error_square / target_square
        else:
            raise ValueError(objective)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss for {name} at step {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        last_gradient_norm = float(gradient_norm.detach().cpu())

    final = trajectory[-1]
    del optimizer, model
    torch.cuda.empty_cache()
    return {
        "name": name,
        "initialization": initialization,
        "objective": objective,
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "trajectory": trajectory,
        "final_record_rel_l2": final["record_rel_l2"],
        "final_coefficient_energy_recovery": final["coefficient_energy_recovery"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r40-script", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--local-row", type=int, required=True)
    parser.add_argument("--expected-sample-id", required=True)
    parser.add_argument("--expected-parent-rel-l2", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--correction-cap", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=420828)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    r40 = load_r40_module(args.r40_script.expanduser().resolve())

    cache_path = args.fit_cache.expanduser().resolve()
    with h5py.File(cache_path, "r", swmr=True) as handle:
        if str(handle.attrs.get("schema", "")) != r40.CACHE_SCHEMA:
            raise RuntimeError("unexpected cache schema")
        if str(handle.attrs.get("subset", "")) != "fit":
            raise RuntimeError("diagnostic requires a train-only fit cache")
        row = int(args.local_row)
        sample_id = str(handle["sample_id"].asstr()[row])
        if sample_id != args.expected_sample_id:
            raise RuntimeError(
                f"sample mismatch at row {row}: {sample_id} != {args.expected_sample_id}"
            )
        family = str(handle["family"].asstr()[row])
        base = torch.from_numpy(
            np.asarray(handle["base_dct_norm"][row], dtype=np.float32)
        ).to(device)
        residual = torch.from_numpy(
            np.asarray(handle["residual_dct_norm"][row], dtype=np.float32)
        ).to(device)
        static_norm_single = torch.from_numpy(
            np.asarray(handle["static_dct_norm"][row], dtype=np.float32)
        ).to(device)
        static_scale_single = torch.from_numpy(
            np.asarray(handle["static_dct_scale"][row], dtype=np.float32)
        ).to(device)
        frequencies = torch.from_numpy(
            np.asarray(handle["frequency_hz"], dtype=np.float32)
        ).to(device)
        frequency_scale = torch.from_numpy(
            np.asarray(handle["frequency_scale"][row], dtype=np.float32)
        ).to(device)
        f0 = float(handle["source_f0_hz"][row])
        t0 = float(handle["source_t0_s"][row])
        target_square_value = float(handle["target_square_total"][row])
        temporal_unselected_error_square = float(
            handle["base_error_square_unselected"][row]
        )
        stored_dt_s = float(handle.attrs["stored_dt_s"])
        frequency_indices = np.asarray(handle["frequency_indices"], dtype=np.int64)

    frequency_count = int(base.shape[0])
    static_norm = static_norm_single[None].expand(frequency_count, -1, -1, -1)
    static_scale = static_scale_single[None].expand(frequency_count, -1)
    f0_tensor = torch.full((frequency_count,), f0, device=device)
    t0_tensor = torch.full((frequency_count,), t0, device=device)
    features = r40.make_features(
        base,
        static_norm,
        static_scale,
        frequency_hz=frequencies,
        frequency_scale=frequency_scale,
        source_f0_hz=f0_tensor,
        source_t0_s=t0_tensor,
    ).contiguous()
    frequency_weight = torch.from_numpy(
        r40.rfft_weights(r40.TIME_COUNT)[frequency_indices].astype(np.float32)
    ).to(device)
    target_square_total = torch.full(
        (frequency_count,), target_square_value, device=device
    )
    family_weight = torch.full(
        (frequency_count,), float(r40.FAMILY_WEIGHTS[family]), device=device
    )

    zero = torch.zeros_like(residual).cpu()
    retained_parent_error_square = float(
        torch.sum(
            frequency_weight.cpu()
            * frequency_scale.cpu().square()
            * residual.cpu().square().sum(dim=(1, 2, 3))
        )
    )
    base_total_error_square = (
        float(args.expected_parent_rel_l2) ** 2 * target_square_value
    )
    irreducible_error_square = base_total_error_square - retained_parent_error_square
    if irreducible_error_square < -1.0e-6 * max(base_total_error_square, 1.0):
        raise RuntimeError(
            "retained residual energy exceeds the fit-log total base error: "
            f"retained={retained_parent_error_square}, total={base_total_error_square}"
        )
    irreducible_error_square = max(irreducible_error_square, 0.0)
    if temporal_unselected_error_square > irreducible_error_square * (1.0 + 1.0e-5):
        raise RuntimeError(
            "temporal-unselected error exceeds total irreducible error: "
            f"temporal={temporal_unselected_error_square}, "
            f"irreducible={irreducible_error_square}"
        )
    parent_rel_l2 = exact_record_rel_l2(
        zero,
        residual.cpu(),
        frequency_scale.cpu(),
        frequency_weight.cpu(),
        irreducible_error_square=irreducible_error_square,
        target_square_total=target_square_value,
    )
    if abs(parent_rel_l2 - float(args.expected_parent_rel_l2)) > 1.0e-7:
        raise RuntimeError(
            f"parent metric reconstruction mismatch: {parent_rel_l2} != "
            f"{args.expected_parent_rel_l2}"
        )
    representation_oracle_rel_l2 = math.sqrt(
        irreducible_error_square / max(target_square_value, 1.0e-30)
    )
    variants = [
        ("zero_physical", "zero", "physical"),
        ("random_physical", "small_random", "physical"),
        ("random_mse", "small_random", "coefficient_mse"),
        ("random_relative", "small_random", "coefficient_relative"),
    ]
    results = []
    for offset, (name, initialization, objective) in enumerate(variants):
        results.append(
            train_variant(
                r40,
                name=name,
                initialization=initialization,
                objective=objective,
                features=features,
                residual=residual,
                frequency_scale=frequency_scale,
                target_square_total=target_square_total,
                frequency_weight=frequency_weight,
                family_weight=family_weight,
                irreducible_error_square=irreducible_error_square,
                target_square_value=target_square_value,
                width=int(args.width),
                modes=int(args.modes),
                blocks=int(args.blocks),
                correction_cap=float(args.correction_cap),
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                learning_rate=float(args.learning_rate),
                seed=int(args.seed) + offset,
            )
        )

    payload = {
        "schema": "r42_single_train_record_overfit_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "separate optimization failure from cross-record generalization failure",
        "sample": {
            "cache": str(cache_path),
            "local_row": int(args.local_row),
            "sample_id": sample_id,
            "family": family,
            "frequency_count": frequency_count,
            "stored_dt_s": stored_dt_s,
            "parent_record_rel_l2": parent_rel_l2,
            "representation_oracle_rel_l2": representation_oracle_rel_l2,
            "base_total_error_square": base_total_error_square,
            "retained_parent_error_square": retained_parent_error_square,
            "irreducible_error_square": irreducible_error_square,
            "temporal_unselected_error_square": temporal_unselected_error_square,
        },
        "architecture": {
            "width": int(args.width),
            "modes": int(args.modes),
            "blocks": int(args.blocks),
            "correction_cap": float(args.correction_cap),
        },
        "data_boundary": {
            "fit_truth_used": True,
            "opened_development_holdout_used": False,
            "r29b_opened": False,
            "final_validation_opened": False,
            "test_id_opened": False,
            "paper_modified": False,
        },
        "variants": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"event": "diagnostic_complete", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
