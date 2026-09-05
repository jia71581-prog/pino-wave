#!/usr/bin/env python3
"""Train one reusable passive local DtN neural element on train-only data."""
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

from saved_time_phase_operator_v4.transfer_dg_elements import (
    TransferDGLocalElementOperator,
)


FAMILIES = ("uniform", "layered", "marmousi")


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


def source_group_split(sample_ids: np.ndarray, families: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use six fit and two holdout source records per family, never sample rows."""
    sample = np.asarray(sample_ids).astype(str)
    family = np.asarray(families).astype(str)
    fit_records, holdout_records = set(), set()
    for name in FAMILIES:
        ordered = []
        for value in sample[family == name]:
            if value not in ordered:
                ordered.append(value)
        if len(ordered) != 8:
            raise RuntimeError(f"expected eight source groups for {name}")
        fit_records.update(ordered[:6])
        holdout_records.update(ordered[6:])
    fit = np.asarray([value in fit_records for value in sample], dtype=bool)
    holdout = np.asarray([value in holdout_records for value in sample], dtype=bool)
    if bool(np.any(fit & holdout)) or not bool(np.all(fit | holdout)):
        raise RuntimeError("source-group split is not a partition")
    return fit, holdout


def robust_flux_relative(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    denominator_floor: float = 10.0,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("local flux tensors must match [B,2,4,modes]")
    numerator = (prediction.double() - target.double()).flatten(1).norm(dim=1)
    denominator = target.double().flatten(1).norm(dim=1)
    return numerator / denominator.clamp_min(float(denominator_floor))


class LocalElementData:
    def __init__(self, path: Path) -> None:
        with h5py.File(path, "r", swmr=True) as handle:
            if handle.attrs.get("schema", "") != "transfer_dg_local_frequency_elements_v1":
                raise RuntimeError("unexpected local element dataset schema")
            if handle.attrs.get("split", "") != "train":
                raise RuntimeError("local DtN trainer accepts train data only")
            self.features = torch.from_numpy(handle["element_features"][:].astype(np.float32))
            self.frequency = torch.from_numpy(handle["frequency_features"][:].astype(np.float32))
            self.trace = torch.from_numpy(handle["pressure_trace_coeff"][:].astype(np.float32))
            self.flux = torch.from_numpy(handle["normal_flux_coeff"][:].astype(np.float32))
            self.sample_ids = handle["sample_id"][:].astype(str)
            self.families = handle["family"][:].astype(str)
        signal = self.trace.double().flatten(1).norm(dim=1).numpy() >= 1.0e-6
        fit, holdout = source_group_split(self.sample_ids, self.families)
        self.fit_indices = np.flatnonzero(fit & signal)
        self.holdout_indices = np.flatnonzero(holdout & signal)
        self.excluded_near_zero = int(np.sum(~signal))

    def batch(self, indices: np.ndarray, device: torch.device):
        index = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        return (
            self.features[index].to(device),
            self.frequency[index].to(device),
            self.trace[index].to(device),
            self.flux[index].to(device),
        )


@torch.inference_mode()
def evaluate(
    model: TransferDGLocalElementOperator,
    data: LocalElementData,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
) -> dict:
    model.eval()
    metric_rows, family_rows, dissipation_rows = [], [], []
    for lo in range(0, len(indices), batch_size):
        selected = indices[lo : lo + batch_size]
        features, frequency, trace, flux = data.batch(selected, device)
        prediction = model(features, frequency, trace)
        metric_rows.append(robust_flux_relative(prediction, flux).cpu())
        context = model.encode_context(features, frequency)
        dissipation_rows.append(
            model.dtn.dissipation_quadratic(context, trace.flatten(2)).cpu()
        )
        family_rows.extend(data.families[selected].tolist())
    metric = torch.cat(metric_rows)
    dissipation = torch.cat(dissipation_rows)
    family_array = np.asarray(family_rows)
    return {
        "aggregate": float(metric.mean()),
        "maximum": float(metric.max()),
        "per_family": {
            family: float(metric[torch.from_numpy(family_array == family)].mean())
            for family in FAMILIES
        },
        "dissipation_minimum": float(dissipation.min()),
        "sample_count": int(metric.numel()),
    }


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
        if prereg["method"] != "Transfer DG" or prereg["stage"] != "local_passive_DtN_v1":
            raise RuntimeError("wrong local DtN preregistration")
        checks = {
            Path(__file__): bindings["trainer_sha256"],
            ROOT / "saved_time_phase_operator_v4/transfer_dg_elements.py": bindings[
                "element_module_sha256"
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
            raise RuntimeError("local DtN hyperparameters drift")

        data = LocalElementData(args.dataset)
        device = torch.device("cuda")
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed + 701)
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
            "schema": "transfer_dg_local_dtn_lane_identity_v1",
            "method": "Transfer DG",
            "stage": "local_passive_DtN_v1",
            "seed": args.seed,
            "dataset": str(args.dataset),
            "dataset_sha256": _sha256(args.dataset),
            "trace_modes": 8,
            "context_dim": 64,
            "dtn_rank": 8,
            "fit_source_groups_per_family": 6,
            "holdout_source_groups_per_family": 2,
            "near_zero_trace_threshold": 1.0e-6,
            "flux_denominator_floor": 10.0,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        zero_metrics = []
        for lo in range(0, len(data.holdout_indices), args.batch_size):
            selected = data.holdout_indices[lo : lo + args.batch_size]
            _, _, _, flux = data.batch(selected, device)
            zero_metrics.append(robust_flux_relative(torch.zeros_like(flux), flux).cpu())
        zero_baseline = float(torch.cat(zero_metrics).mean())
        best = {"epoch": 0, "aggregate": float("inf"), "metrics": None}
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(data.fit_indices)
            train_rows = []
            for lo in range(0, len(order), args.batch_size):
                selected = order[lo : lo + args.batch_size]
                features, frequency, trace, flux = data.batch(selected, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features, frequency, trace)
                loss = robust_flux_relative(prediction, flux).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite local DtN loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_rows.append(float(loss.detach()))
            scheduler.step()
            metrics = evaluate(
                model,
                data,
                data.holdout_indices,
                device,
                batch_size=args.batch_size,
            )
            event = {
                "event": "epoch",
                "epoch": epoch,
                "seed": args.seed,
                "train_loss": float(np.mean(train_rows)),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                "zero_flux_baseline": zero_baseline,
                "holdout": metrics,
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
        passed = bool(
            best["aggregate"] < zero_baseline
            and best["metrics"]["dissipation_minimum"] >= -1.0e-6
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
                "fit_sample_count": int(len(data.fit_indices)),
                "holdout_sample_count": int(len(data.holdout_indices)),
                "excluded_near_zero": data.excluded_near_zero,
                "elapsed_s": round(time.time() - started, 1),
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
