#!/usr/bin/env python3
"""Train one PA-CORA P1 multiwindow lane with V9-compatible settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.b2_v5_components import (
    b2_v5_loss,
    per_record_relative_l2,
    temporal_band_relative_l2,
)
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


FAMILIES = ("uniform", "layered", "marmousi")
TIME_SLICES = {"early": (0, 19), "middle": (19, 38), "late": (38, 56)}


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


class MultiWindowCache:
    def __init__(self, cache_paths: list[Path], manifests: list[dict], *, key: str):
        if len(cache_paths) != len(manifests) or not cache_paths:
            raise ValueError("cache paths and manifests must be nonempty and aligned")
        self.handles = []
        self.mapping: list[tuple[int, int]] = []
        families = []
        self.key = str(key)
        for file_index, (path, manifest) in enumerate(zip(cache_paths, manifests)):
            handle = h5py.File(path, "r", swmr=True)
            self.handles.append(handle)
            if handle.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError(f"unexpected cache schema: {path}")
            if handle.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError(f"cache/manifest drift: {path}")
            if self.key not in handle:
                raise RuntimeError(f"conditioning dataset absent: {self.key}")
            count = int(handle["base_seq"].shape[0])
            if count != len(manifest["records"]):
                raise RuntimeError("cache record count differs from manifest")
            self.mapping.extend((file_index, local) for local in range(count))
            families.extend(handle["family"][:].astype(str).tolist())
        self.families = np.asarray(families, dtype=str)

    def __len__(self) -> int:
        return len(self.mapping)

    def batch(self, indices) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = [self.mapping[int(index)] for index in indices]
        base = np.stack(
            [self.handles[file]["base_seq"][local] for file, local in rows]
        ).astype(np.float32)
        target = np.stack(
            [self.handles[file]["target"][local] for file, local in rows]
        ).astype(np.float32)
        conditioning = np.stack(
            [self.handles[file][self.key][local] for file, local in rows]
        ).astype(np.float32)
        return (
            torch.from_numpy(base)[:, :, None],
            torch.from_numpy(target)[:, :, None],
            torch.from_numpy(conditioning),
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.close()
        self.handles = []


def _load_holdout(path: Path, manifest: dict, *, key: str) -> dict:
    with h5py.File(path, "r", swmr=True) as cache:
        if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
            raise RuntimeError("holdout cache is not B2-v5 causal")
        if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
            raise RuntimeError("holdout cache/manifest drift")
        return {
            "base": torch.from_numpy(cache["base_seq"][:].astype(np.float32))[:, :, None],
            "target": torch.from_numpy(cache["target"][:].astype(np.float32))[:, :, None],
            "cond": torch.from_numpy(cache[key][:].astype(np.float32)),
            "families": cache["family"][:].astype(str),
        }


def _baseline(holdout: dict) -> tuple[dict, torch.Tensor]:
    relative = per_record_relative_l2(
        holdout["base"][:, 8:].double(), holdout["target"][:, 8:].double()
    ).cpu()
    return {
        "aggregate": float(relative.mean()),
        "maximum": float(relative.max()),
        "per_family": {
            family: float(relative[torch.from_numpy(holdout["families"] == family)].mean())
            for family in FAMILIES
        },
    }, relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", type=Path, action="append", required=True)
    parser.add_argument("--fit-cache", type=Path, action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=23)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--micro-records", type=int, default=4)
    parser.add_argument("--maximum-epoch1-s", type=float, default=900.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    if len(args.fit_manifest) != 4 or len(args.fit_cache) != 4:
        raise ValueError("PA-CORA P1 requires exactly four fit-window slots")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    fit = None
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if _sha256(Path(__file__)) != bindings["trainer_sha256"]:
            raise RuntimeError("trainer binding drift")
        manifests = [json.loads(path.read_text()) for path in args.fit_manifest]
        holdout_manifest = json.loads(args.holdout_manifest.read_text())
        for path, manifest in zip(args.fit_manifest, manifests):
            if _sha256(path) != bindings["fit_manifest_sha256"][str(path)]:
                raise RuntimeError(f"fit manifest binding drift: {path}")
            if manifest.get("split") != "train" or manifest.get("validation_opened") or manifest.get("test_id_opened"):
                raise RuntimeError("fit manifest violates the train-only boundary")
        for path in args.fit_cache:
            if _sha256(path) != bindings["fit_cache_sha256"][str(path)]:
                raise RuntimeError(f"fit cache binding drift: {path}")
        if _sha256(args.holdout_manifest) != bindings["holdout_manifest_sha256"]:
            raise RuntimeError("holdout manifest binding drift")
        if _sha256(args.holdout_cache) != bindings["holdout_cache_sha256"]:
            raise RuntimeError("holdout cache binding drift")
        fit_groups = {row["group_id"] for manifest in manifests for row in manifest["records"]}
        holdout_groups = {row["group_id"] for row in holdout_manifest["records"]}
        if fit_groups & holdout_groups:
            raise RuntimeError("fit/holdout group leakage")
        conditioning_key = "cond"
        fit = MultiWindowCache(args.fit_cache, manifests, key=conditioning_key)
        holdout = _load_holdout(args.holdout_cache, holdout_manifest, key=conditioning_key)
        holdout_initial = holdout["target"][:, :8, 0]
        baseline, baseline_relative = _baseline(holdout)
        device = torch.device("cuda")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = FrameConditionedPropagator(
            state_channels=8,
            cond_channels=7,
            width=64,
            spectral_rank=32,
            modes=24,
            depth=4,
            gate_init=1.0,
            activation_checkpointing=True,
        ).to(device)
        torch.nn.init.zeros_(model.decoder[-1].weight)
        torch.nn.init.zeros_(model.decoder[-1].bias)
        model.gate.requires_grad_(False)
        steps_per_epoch = (len(fit) + args.micro_records - 1) // args.micro_records
        total_steps = args.epochs * steps_per_epoch
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=1.0e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.01
        )

        def evaluate() -> dict:
            model.eval()
            relative_rows = []
            correction_rows = []
            temporal_rows = {name: [] for name in TIME_SLICES}
            frequency_rows = []
            with torch.inference_mode():
                for lo in range(0, len(holdout["families"]), args.micro_records):
                    hi = min(lo + args.micro_records, len(holdout["families"]))
                    anchor = holdout["base"][lo:hi].to(device)
                    target = holdout["target"][lo:hi].to(device)
                    prediction = model.forward_anchored(
                        anchor,
                        holdout["cond"][lo:hi].to(device),
                        initial_state=holdout_initial[lo:hi].to(device),
                    )
                    future = prediction[:, 8:]
                    target_future = target[:, 8:]
                    anchor_future = anchor[:, 8:]
                    relative_rows.append(
                        per_record_relative_l2(future.double(), target_future.double()).cpu()
                    )
                    correction_rows.append(
                        per_record_relative_l2(future.double(), anchor_future.double()).cpu()
                    )
                    for name, (start, stop) in TIME_SLICES.items():
                        temporal_rows[name].append(
                            per_record_relative_l2(
                                future[:, start:stop].double(),
                                target_future[:, start:stop].double(),
                            ).cpu()
                        )
                    frequency_rows.append(
                        temporal_band_relative_l2(
                            future,
                            target_future,
                            energy_floor_fraction=0.005,
                        ).cpu()
                    )
            relative = torch.cat(relative_rows)
            frequency = torch.cat(frequency_rows)
            return {
                "aggregate": float(relative.mean()),
                "maximum": float(relative.max()),
                "nonworse": int((relative.double() <= baseline_relative).sum()),
                "correction_energy": float(torch.cat(correction_rows).mean()),
                "per_family": {
                    family: float(relative[torch.from_numpy(holdout["families"] == family)].mean())
                    for family in FAMILIES
                },
                "temporal": {
                    name: float(torch.cat(values).mean())
                    for name, values in temporal_rows.items()
                },
                "frequency": {
                    name: float(frequency[:, index].mean())
                    for index, name in enumerate(("low", "middle", "high"))
                },
            }

        identity = {
            "schema": "pa_cora_p1_multiwindow_lane_identity_v1",
            "method": "PA-CORA",
            "stage": "P1_multiwindow_only",
            "seed": args.seed,
            "epochs": args.epochs,
            "micro_records": args.micro_records,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "record_exposures": total_steps * args.micro_records,
            "conditioning_key": conditioning_key,
            "baseline": baseline,
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "trainer_sha256": _sha256(Path(__file__)),
            "fit_manifest_sha256": {str(path): _sha256(path) for path in args.fit_manifest},
            "fit_cache_sha256": {str(path): _sha256(path) for path in args.fit_cache},
            "holdout_manifest_sha256": _sha256(args.holdout_manifest),
            "holdout_cache_sha256": _sha256(args.holdout_cache),
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        best = None
        order_rng = np.random.default_rng(args.seed + 101)
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = order_rng.permutation(len(fit))
            losses = []
            components = []
            for lo in range(0, len(order), args.micro_records):
                indices = order[lo : lo + args.micro_records]
                anchor, target, conditioning = fit.batch(indices)
                optimizer.zero_grad(set_to_none=True)
                anchor = anchor.to(device)
                target = target.to(device)
                prediction = model.forward_anchored(
                    anchor,
                    conditioning.to(device),
                    initial_state=target[:, :8, 0],
                )
                loss, row = b2_v5_loss(
                    prediction[:, 8:],
                    target[:, 8:],
                    anchor[:, 8:],
                    arm="spectral",
                    spectral_weight=0.1,
                    spectral_energy_floor_fraction=0.005,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                components.append({key: float(value) for key, value in row.items()})
            metrics = evaluate()
            event = {
                "event": "epoch",
                "epoch": epoch,
                "seed": args.seed,
                "train_loss": float(np.mean(losses)),
                "train_components": {
                    key: float(np.mean([row[key] for row in components]))
                    for key in components[0]
                },
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                **metrics,
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            if epoch == 1 and event["elapsed_s"] > args.maximum_epoch1_s:
                raise RuntimeError("epoch-1 wall-time budget exceeded")
            if best is None or metrics["aggregate"] < best["aggregate"]:
                best = {"epoch": epoch, **metrics}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "identity": identity,
                        "metrics": metrics,
                    },
                    args.output_dir / "best.pt",
                )
                _atomic_json(best, args.output_dir / "best.json")
        _atomic_json(
            {
                "status": "complete",
                "best": best,
                "baseline": baseline,
                "elapsed_s": round(time.time() - started, 1),
            },
            terminal,
        )
        fit.close()
        return 0
    except Exception as error:
        import traceback

        if fit is not None:
            fit.close()
        _atomic_json(
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
            terminal,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
