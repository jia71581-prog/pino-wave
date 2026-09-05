#!/usr/bin/env python3
"""Train a local passive DtN element with Transfer DG-HCAIS sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.hcais import (
    effective_sample_size,
    hcais_distribution,
    inverse_probability_weights,
    regularized_leverage_scores,
    within_stratum_percentile,
)
from saved_time_phase_operator_v4.transfer_dg_elements import (
    TransferDGLocalElementOperator,
)
from scripts.train_transfer_dg_local_dtn import (
    FAMILIES,
    LocalElementData,
    evaluate,
    robust_flux_relative,
)


FREQUENCY_RATIOS = np.asarray((0.50, 0.75, 1.00, 1.25), dtype=np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def build_hcais_strata(
    families: np.ndarray,
    frequency_features: np.ndarray,
    element_origins: np.ndarray,
    interface_strength: np.ndarray,
) -> np.ndarray:
    family = np.asarray(families).astype(str)
    frequency = np.asarray(frequency_features, dtype=np.float64)
    origins = np.asarray(element_origins)
    interface = np.asarray(interface_strength, dtype=np.float64).reshape(-1)
    count = len(family)
    if frequency.shape != (count, 4) or origins.shape != (count, 2) or interface.size != count:
        raise ValueError("HCAIS stratum features do not align")
    family_code = np.asarray([FAMILIES.index(value) for value in family])
    ratio = (40.0 * frequency[:, 0]) / np.maximum(30.0 * frequency[:, 1], 1.0e-8)
    ratio_code = np.argmin(np.abs(ratio[:, None] - FREQUENCY_RATIOS[None]), axis=1)
    element_type = np.zeros(count, dtype=np.int64)
    element_type[interface > 0.05] = 1
    element_type[origins[:, 0] == 0] = 2
    raw = family_code * 12 + ratio_code * 3 + element_type
    _, compact = np.unique(raw, return_inverse=True)
    return compact.astype(np.int64)


def combined_difficulty(
    current_error: np.ndarray,
    previous_error: np.ndarray,
    high_mode_ratio: np.ndarray,
    interface_strength: np.ndarray,
    strata: np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_error, dtype=np.float64)
    previous = np.asarray(previous_error, dtype=np.float64)
    slow = current / np.maximum(previous, 1.0e-8)
    score = (
        0.50 * within_stratum_percentile(current, strata)
        + 0.20 * within_stratum_percentile(high_mode_ratio, strata)
        + 0.20 * within_stratum_percentile(interface_strength, strata)
        + 0.10 * within_stratum_percentile(slow, strata)
    )
    return np.clip(score, 0.0, 1.0)


@torch.inference_mode()
def fit_errors(
    model: TransferDGLocalElementOperator,
    data: LocalElementData,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    rows = []
    for lo in range(0, len(indices), batch_size):
        selected = indices[lo : lo + batch_size]
        features, frequency, trace, flux = data.batch(selected, device)
        prediction = model(features, frequency, trace)
        rows.append(robust_flux_relative(prediction, flux).cpu().numpy())
    return np.concatenate(rows).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if prereg["method"] != "Transfer DG" or prereg["stage"] != "local_DtN_HCAIS_v1":
            raise RuntimeError("wrong HCAIS preregistration")
        checks = {
            Path(__file__): bindings["trainer_sha256"],
            ROOT / "saved_time_phase_operator_v4/hcais.py": bindings["hcais_sha256"],
            ROOT / "saved_time_phase_operator_v4/transfer_dg_elements.py": bindings[
                "element_module_sha256"
            ],
            ROOT / "scripts/train_transfer_dg_local_dtn.py": bindings[
                "baseline_trainer_sha256"
            ],
            args.dataset: bindings["dataset_sha256"],
        }
        for path, expected in checks.items():
            if _sha256(path) != expected:
                raise RuntimeError(f"binding drift: {path}")
        frozen = prereg["hyperparameters"]
        if (
            args.epochs != frozen["epochs"]
            or args.batch_size != frozen["batch_size"]
            or args.learning_rate != frozen["learning_rate"]
        ):
            raise RuntimeError("HCAIS hyperparameter drift")

        data = LocalElementData(args.dataset)
        with h5py.File(args.dataset, "r", swmr=True) as handle:
            origins_all = handle["element_origin_zx"][:]
        fit = data.fit_indices
        interface = data.features[fit, 1].amax(dim=(-2, -1)).numpy().astype(np.float64)
        strata = build_hcais_strata(
            data.families[fit],
            data.frequency[fit].numpy(),
            origins_all[fit],
            interface,
        )
        trace = data.trace[fit].numpy().astype(np.float64)
        high_energy = np.square(trace[..., 4:]).sum(axis=(1, 2, 3))
        total_energy = np.square(trace).sum(axis=(1, 2, 3))
        high_ratio = high_energy / np.maximum(total_energy, 1.0e-12)
        normalized_trace = trace.reshape(len(trace), -1)
        free_surface = (origins_all[fit, 0] == 0).astype(np.float64)[:, None]
        leverage_features = np.concatenate(
            (
                normalized_trace,
                data.frequency[fit].numpy().astype(np.float64),
                interface[:, None],
                free_surface,
            ),
            axis=1,
        )
        leverage = regularized_leverage_scores(leverage_features, ridge=1.0e-3)

        device = torch.device("cuda")
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed + 809)
        model = TransferDGLocalElementOperator(
            input_channels=4,
            trace_modes=8,
            context_dim=64,
            dtn_rank=8,
            frequency_dim=4,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
        identity = {
            "schema": "transfer_dg_local_dtn_hcais_lane_identity_v1",
            "method": "Transfer DG",
            "stage": "local_DtN_HCAIS_v1",
            "seed": args.seed,
            "dataset": str(args.dataset),
            "dataset_sha256": _sha256(args.dataset),
            "fit_source_groups_per_family": 6,
            "holdout_source_groups_per_family": 2,
            "coverage_floor": 0.40,
            "difficulty_fraction_at_floor": 0.40,
            "leverage_fraction_at_floor": 0.20,
            "temperature_anneal_epochs": 24,
            "score_refresh_epochs": 10,
            "ess_p5_floor_fraction": 0.50,
            "importance_correction": "unclipped Horvitz-Thompson p_uniform/q",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")

        zero_rows = []
        for lo in range(0, len(data.holdout_indices), args.batch_size):
            selected = data.holdout_indices[lo : lo + args.batch_size]
            _, _, _, flux = data.batch(selected, device)
            zero_rows.append(robust_flux_relative(torch.zeros_like(flux), flux).cpu())
        zero_baseline = float(torch.cat(zero_rows).mean())
        best = {"epoch": 0, "aggregate": float("inf"), "metrics": None}
        previous_error = np.ones(len(fit), dtype=np.float64)
        ema_error = previous_error.copy()
        difficulty = np.ones(len(fit), dtype=np.float64)
        coverage_mix = 0.40
        cumulative_sampling_s = 0.0
        ess_p5_history, maximum_weight_history = [], []
        started = time.time()

        for epoch in range(1, args.epochs + 1):
            sampler_started = time.perf_counter()
            refreshed = epoch == 1 or (epoch - 1) % 10 == 0
            if refreshed:
                current_error = fit_errors(
                    model, data, fit, device, args.batch_size
                )
                if epoch == 1:
                    ema_error = current_error.copy()
                    previous_error = current_error.copy()
                else:
                    old_ema = ema_error.copy()
                    ema_error = 0.8 * ema_error + 0.2 * current_error
                    previous_error = old_ema
                difficulty = combined_difficulty(
                    ema_error, previous_error, high_ratio, interface, strata
                )
            temperature = min(1.0, max(0.0, (epoch - 1) / 24.0))
            distribution = hcais_distribution(
                difficulty,
                leverage,
                strata,
                coverage_mix=coverage_mix,
                temperature_power=temperature,
            )
            sampled_local = rng.choice(
                len(fit), size=len(fit), replace=True, p=distribution
            )
            sampled_global = fit[sampled_local]
            weights = inverse_probability_weights(sampled_local, distribution)
            sampling_s = time.perf_counter() - sampler_started
            cumulative_sampling_s += sampling_s

            model.train()
            train_rows, batch_ess, batch_weight_max = [], [], []
            for lo in range(0, len(sampled_global), args.batch_size):
                hi = min(lo + args.batch_size, len(sampled_global))
                selected = sampled_global[lo:hi]
                local_weights = weights[lo:hi]
                features, frequency, pressure_trace, flux = data.batch(selected, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features, frequency, pressure_trace)
                relative = robust_flux_relative(prediction, flux)
                weight_tensor = torch.from_numpy(local_weights).to(
                    device=device, dtype=relative.dtype
                )
                loss = (weight_tensor * relative).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite HCAIS local DtN loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_rows.append(float(loss.detach()))
                batch_ess.append(effective_sample_size(local_weights) / len(local_weights))
                batch_weight_max.append(float(np.max(local_weights)))
            scheduler.step()
            ess_p5 = float(np.quantile(batch_ess, 0.05))
            ess_p5_history.append(ess_p5)
            maximum_weight_history.append(max(batch_weight_max))
            if ess_p5 < 0.50:
                coverage_mix = min(0.80, coverage_mix + 0.10)
            elif ess_p5 > 0.70 and coverage_mix > 0.40:
                coverage_mix = max(0.40, coverage_mix - 0.05)

            metrics = evaluate(
                model,
                data,
                data.holdout_indices,
                device,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - started
            entropy = float(
                -np.sum(distribution * np.log(distribution)) / np.log(len(distribution))
            )
            event = {
                "event": "epoch",
                "epoch": epoch,
                "seed": args.seed,
                "train_loss": float(np.mean(train_rows)),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(elapsed, 1),
                "zero_flux_baseline": zero_baseline,
                "holdout": metrics,
                "sampler": {
                    "score_refreshed": refreshed,
                    "temperature_power": temperature,
                    "coverage_mix_next": coverage_mix,
                    "probability_min": float(distribution.min()),
                    "probability_max": float(distribution.max()),
                    "normalized_entropy": entropy,
                    "ess_mean_fraction": float(np.mean(batch_ess)),
                    "ess_p5_fraction": ess_p5,
                    "ess_min_fraction": float(np.min(batch_ess)),
                    "maximum_importance_weight": max(batch_weight_max),
                    "sampling_s": sampling_s,
                    "cumulative_sampling_overhead_fraction": cumulative_sampling_s
                    / max(elapsed, 1.0e-12),
                },
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            if metrics["aggregate"] < best["aggregate"]:
                best = {"epoch": epoch, "aggregate": metrics["aggregate"], "metrics": metrics}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "identity": identity,
                        "metrics": best,
                    },
                    args.output_dir / "best.pt",
                )
                _atomic_json(best, args.output_dir / "best.json")

        elapsed = time.time() - started
        passed = bool(
            best["aggregate"] < zero_baseline
            and best["metrics"]["dissipation_minimum"] >= -1.0e-6
            and min(ess_p5_history) >= 0.50
            and cumulative_sampling_s / max(elapsed, 1.0e-12) <= 0.10
        )
        _atomic_json(
            {
                "status": "passed" if passed else "rejected",
                "seed": args.seed,
                "best_epoch": best["epoch"],
                "best_aggregate": best["aggregate"],
                "zero_flux_baseline": zero_baseline,
                "relative_gain": (zero_baseline - best["aggregate"])
                / max(zero_baseline, 1.0e-16),
                "best_metrics": best["metrics"],
                "sampler": {
                    "minimum_epoch_ess_p5_fraction": min(ess_p5_history),
                    "maximum_importance_weight": max(maximum_weight_history),
                    "sampling_overhead_s": cumulative_sampling_s,
                    "sampling_overhead_fraction": cumulative_sampling_s
                    / max(elapsed, 1.0e-12),
                    "final_coverage_mix": coverage_mix,
                },
                "elapsed_s": round(elapsed, 1),
                "validation_opened": False,
                "test_id_opened": False,
            },
            terminal,
        )
        return 0
    except Exception as error:
        import traceback

        _atomic_json(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
            terminal,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
