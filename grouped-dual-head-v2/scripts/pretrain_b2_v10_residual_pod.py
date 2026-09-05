#!/usr/bin/env python3
"""Fit B2-v10 family residual modes offline from train truth only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


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


def _fit_pod_memory_bounded(
    residuals: torch.Tensor,
    rank: int,
    *,
    feature_chunk: int = 131072,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gram POD without materializing an N-by-field float64 matrix."""
    if residuals.ndim != 5 or not 0 < rank <= residuals.shape[0]:
        raise ValueError("invalid residual tensor or rank")
    matrix = residuals.reshape(residuals.shape[0], -1).float()
    gram = torch.zeros(matrix.shape[0], matrix.shape[0], dtype=torch.float64)
    for start in range(0, matrix.shape[1], feature_chunk):
        stop = min(start + feature_chunk, matrix.shape[1])
        block = matrix[:, start:stop].double()
        gram.add_(block @ block.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    values = eigenvalues[order].clamp_min(1.0e-16)
    projection = eigenvectors[:, order].T.float() / values.sqrt().float()[:, None]
    modes = torch.empty(rank, matrix.shape[1], dtype=torch.float32)
    for start in range(0, matrix.shape[1], feature_chunk):
        stop = min(start + feature_chunk, matrix.shape[1])
        modes[:, start:stop] = projection @ matrix[:, start:stop]
    return modes.reshape(rank, *residuals.shape[1:]), values.float()


def _load_parent(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, str]:
    identity = checkpoint["identity"]
    conditioning_key = str(identity["conditioning_key"])
    model = FrameConditionedPropagator(
        state_channels=8,
        cond_channels=int(identity["cond_channels"]),
        width=int(identity["width"]) if "width" in identity else 64,
        spectral_rank=int(identity["spectral_rank"])
        if "spectral_rank" in identity
        else 32,
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, conditioning_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        for path, key in (
            (Path(__file__), "pretrain_sha256"),
            (args.checkpoint, "checkpoint_sha256"),
            (args.cache, "fit_cache_sha256"),
            (args.manifest, "fit_manifest_sha256"),
        ):
            if _sha256(path) != bindings[key]:
                raise RuntimeError(f"binding drift: {path}")
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("split") != "train":
            raise RuntimeError("POD fitting is restricted to train records")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        if manifest.get("future_truth_opened_for_window_selection"):
            raise RuntimeError("future-derived window selection is forbidden")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        device = torch.device("cuda")
        model, conditioning_key = _load_parent(checkpoint, device)
        identity = {
            "schema": "b2_v10_offline_pod_identity_v1",
            "scope": "offline_train_truth_only",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "cache": str(args.cache),
            "cache_sha256": _sha256(args.cache),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "pretrain_sha256": _sha256(Path(__file__)),
            "rank": args.rank,
            "conditioning_key": conditioning_key,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        bundle = {
            "schema": "b2_v10_residual_pod_bundle_v1",
            "rank": args.rank,
            "parent_checkpoint_sha256": identity["checkpoint_sha256"],
            "fit_manifest_sha256": identity["manifest_sha256"],
            "families": {},
        }
        report = {
            "schema": "b2_v10_offline_pod_report_v1",
            "scope": "train_only_oracle_capacity_diagnostic",
            "families": {},
            "validation_opened": False,
            "test_id_opened": False,
        }
        with h5py.File(args.cache, "r", swmr=True) as cache:
            if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
                raise RuntimeError("fit cache is not B2-v5 causal")
            if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
                raise RuntimeError("fit cache/manifest selection drift")
            if conditioning_key not in cache:
                raise RuntimeError(f"conditioning dataset absent: {conditioning_key}")
            for family in FAMILIES:
                indices = [
                    index
                    for index, row in enumerate(manifest["records"])
                    if row["family"] == family
                ]
                if len(indices) < args.rank:
                    raise RuntimeError(f"insufficient {family} records for rank {args.rank}")
                residual_rows = []
                target_future_norms = []
                for index in indices:
                    base = torch.from_numpy(
                        cache["base_seq"][index : index + 1].astype(np.float32)
                    )[:, :, None].to(device)
                    target = torch.from_numpy(
                        cache["target"][index : index + 1].astype(np.float32)
                    )[:, :, None].to(device)
                    conditioning = torch.from_numpy(
                        cache[conditioning_key][index : index + 1].astype(np.float32)
                    ).to(device)
                    with torch.inference_mode():
                        parent = model.forward_anchored(
                            base,
                            conditioning,
                            initial_state=target[:, :8, 0],
                        )
                    residual = (target - parent).cpu().float()
                    residual_rows.append(residual)
                    target_future_norms.append(float(target[:, 8:].double().norm().cpu()))
                    del base, target, conditioning, parent, residual
                residuals = torch.cat(residual_rows, dim=0)
                modes, eigenvalues = _fit_pod_memory_bounded(residuals, args.rank)
                coefficients = residuals.reshape(len(indices), -1) @ modes.reshape(
                    args.rank, -1
                ).T
                parent_errors = []
                projected_errors = []
                for row_index in range(len(indices)):
                    residual_future = residuals[row_index, 8:]
                    reconstructed = torch.einsum(
                        "r,rtczx->tczx", coefficients[row_index], modes[:, 8:]
                    )
                    denominator = max(target_future_norms[row_index], 1.0e-16)
                    parent_errors.append(float(residual_future.double().norm()) / denominator)
                    projected_errors.append(
                        float((residual_future - reconstructed).double().norm()) / denominator
                    )
                parent_values = np.asarray(parent_errors, dtype=np.float64)
                projected_values = np.asarray(projected_errors, dtype=np.float64)
                gains = (parent_values - projected_values) / np.maximum(
                    parent_values, 1.0e-16
                )
                bundle["families"][family] = {
                    "modes": modes.half(),
                    "eigenvalues": eigenvalues,
                    "record_count": len(indices),
                }
                report["families"][family] = {
                    "records": len(indices),
                    "mean_parent_relative_l2": float(parent_values.mean()),
                    "mean_projected_relative_l2": float(projected_values.mean()),
                    "mean_oracle_gain": float(gains.mean()),
                    "minimum_oracle_gain": float(gains.min()),
                }
                del residual_rows, residuals, modes, coefficients
        torch.save(bundle, args.output_dir / "pod_bundle.pt")
        report["bundle_sha256"] = _sha256(args.output_dir / "pod_bundle.pt")
        minimum_gain = float(prereg["gates"]["minimum_mean_oracle_gain_per_family"])
        report["passed"] = all(
            report["families"][family]["mean_oracle_gain"] > minimum_gain
            for family in FAMILIES
        )
        _atomic_json(report, args.output_dir / "report.json")
        _atomic_json(
            {
                "status": "passed" if report["passed"] else "rejected",
                "report": str(args.output_dir / "report.json"),
                "bundle": str(args.output_dir / "pod_bundle.pt"),
                "bundle_sha256": report["bundle_sha256"],
            },
            terminal,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
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
