#!/usr/bin/env python3
"""Offline meta-training of B2-v11 local causal residual adaptation."""
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
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from saved_time_phase_operator_v4.instance_adaptation.b2_v10_prefix_assimilation import (
    source_cycle_prefix_count,
)
from saved_time_phase_operator_v4.instance_adaptation.b2_v11_local_meta import (
    LocalMetaAdaptConfig,
    LocalResidualMetaOperator,
    adapt_local_meta,
    apply_temporal_polynomial,
    fit_temporal_polynomial,
    prefix_residual_context,
    query_features,
)


FAMILIES = ("uniform", "layered", "marmousi")


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


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        (prediction.float() - target.float()).flatten(1).norm(dim=1)
        / target.float().flatten(1).norm(dim=1).clamp_min(1.0e-8)
    )


def _query_indices(start: int, total: int, count: int) -> torch.Tensor:
    available = total - int(start)
    requested = min(int(count), available)
    if requested <= 0:
        raise ValueError("future query set is empty")
    values = np.rint(np.linspace(start, total - 1, requested)).astype(np.int64)
    values = np.unique(values)
    return torch.from_numpy(values)


def meta_episode_loss(
    model: torch.nn.Module,
    parent: torch.Tensor,
    conditioning: torch.Tensor,
    target: torch.Tensor,
    *,
    observed_count: int,
    query_indices_tensor: torch.Tensor,
    time_s: torch.Tensor,
    source_f0_hz: torch.Tensor,
    source_t0_s: torch.Tensor,
    anchor_frames: int = 8,
    validation_tail_frames: int = 3,
    ridge_fraction: float = 1.0e-3,
    trust_ratio: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One differentiable support/inner-fit/future-query meta episode."""
    count = int(observed_count)
    support_count = count - int(validation_tail_frames)
    if support_count < int(anchor_frames) + 2:
        raise ValueError("meta episode has insufficient support frames")
    observed = target[:, :count]
    support_context = prefix_residual_context(
        parent, observed[:, :support_count], anchor_frames=anchor_frames
    )
    validation_indices = torch.arange(support_count, count, device=parent.device)
    validation_features = query_features(
        parent,
        conditioning,
        support_context,
        validation_indices,
        context_count=torch.tensor([support_count], device=parent.device),
        time_s=time_s,
        source_f0_hz=source_f0_hz,
        source_t0_s=source_t0_s,
    )
    validation_direction = model(validation_features).reshape(
        1, len(validation_indices), 1, *parent.shape[-2:]
    )
    validation_target = target[:, support_count:count] - parent[:, support_count:count]
    axis = torch.as_tensor(time_s, dtype=parent.dtype, device=parent.device)
    coefficients = fit_temporal_polynomial(
        validation_direction,
        validation_target,
        axis[validation_indices],
        axis,
        ridge_fraction=ridge_fraction,
    )
    full_context = prefix_residual_context(parent, observed, anchor_frames=anchor_frames)
    query = torch.as_tensor(query_indices_tensor, dtype=torch.long, device=parent.device)
    future_features = query_features(
        parent,
        conditioning,
        full_context,
        query,
        context_count=torch.tensor([count], device=parent.device),
        time_s=axis,
        source_f0_hz=source_f0_hz,
        source_t0_s=source_t0_s,
    )
    future_direction = model(future_features).reshape(
        1, len(query), 1, *parent.shape[-2:]
    )
    correction = apply_temporal_polynomial(
        future_direction, coefficients, axis[query], axis
    )
    parent_query = parent[:, query]
    target_query = target[:, query]
    ratio = correction.norm() / parent_query.norm().clamp_min(1.0e-8)
    scale = torch.minimum(
        torch.ones_like(ratio),
        torch.as_tensor(trust_ratio, device=ratio.device, dtype=ratio.dtype)
        / ratio.detach().clamp_min(1.0e-8),
    )
    correction = correction * scale
    candidate = parent_query + correction
    candidate_relative = _relative_l2(candidate, target_query).mean()
    parent_relative = _relative_l2(parent_query, target_query).mean().detach()
    return candidate_relative, {
        "candidate_relative_l2": float(candidate_relative.detach()),
        "parent_relative_l2": float(parent_relative),
        "correction_ratio": float(
            correction.detach().norm() / parent_query.detach().norm().clamp_min(1.0e-8)
        ),
    }


class _CachePair:
    def __init__(self, source_path: Path, parent_path: Path, manifest: dict):
        self.source = h5py.File(source_path, "r", swmr=True)
        self.parent = h5py.File(parent_path, "r", swmr=True)
        if self.source.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
            raise RuntimeError("unexpected source cache schema")
        if self.parent.attrs.get("schema", "") != "b2_v11_parent_prediction_cache_v1":
            raise RuntimeError("unexpected parent cache schema")
        selection = manifest["selection_sha256"]
        if self.source.attrs["manifest_selection_sha256"] != selection:
            raise RuntimeError("source cache/manifest drift")
        if self.parent.attrs["manifest_selection_sha256"] != selection:
            raise RuntimeError("parent cache/manifest drift")
        source_ids = self.source["sample_id"][:].astype(str).tolist()
        parent_ids = self.parent["sample_id"][:].astype(str).tolist()
        manifest_ids = [row["sample_id"] for row in manifest["records"]]
        if source_ids != parent_ids or source_ids != manifest_ids:
            raise RuntimeError("cache sample order drift")
        self.conditioning_key = str(self.parent.attrs["conditioning_key"])
        self.manifest = manifest

    def close(self) -> None:
        self.source.close()
        self.parent.close()


def _record_tensors(
    pair: _CachePair,
    index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = torch.from_numpy(
        pair.parent["parent"][index : index + 1].astype(np.float32)
    )[:, :, None].to(device)
    target = torch.from_numpy(
        pair.source["target"][index : index + 1].astype(np.float32)
    )[:, :, None].to(device)
    conditioning = torch.from_numpy(
        pair.source[pair.conditioning_key][index : index + 1].astype(np.float32)
    ).to(device)
    return parent, target, conditioning


def _observed_count(row: dict, global_time_s: torch.Tensor, total: int) -> int:
    start = int(row["window_start"])
    window = global_time_s[start : start + total]
    return source_cycle_prefix_count(
        window,
        source_t0_s=float(row["source_t0_s"]),
        source_f0_hz=float(row["source_f0_hz"]),
        after_peak_cycles=0.5,
        minimum_frames=13,
        maximum_frames=total - 1,
    )


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    pair: _CachePair,
    global_time_s: torch.Tensor,
    device: torch.device,
    config: LocalMetaAdaptConfig,
) -> dict:
    model.eval()
    rows = []
    for index, row in enumerate(pair.manifest["records"]):
        total = int(pair.parent["parent"].shape[1])
        count = _observed_count(row, global_time_s, total)
        parent = torch.from_numpy(
            pair.parent["parent"][index : index + 1].astype(np.float32)
        )[:, :, None].to(device)
        conditioning = torch.from_numpy(
            pair.source[pair.conditioning_key][index : index + 1].astype(np.float32)
        ).to(device)
        observed = torch.from_numpy(
            pair.source["target"][index : index + 1, :count].astype(np.float32)
        )[:, :, None].to(device)
        start = int(row["window_start"])
        axis = global_time_s[start : start + total].to(device)
        candidate, adaptation = adapt_local_meta(
            model,
            parent,
            conditioning,
            observed,
            time_s=axis,
            source_f0_hz=torch.tensor([row["source_f0_hz"]], device=device),
            source_t0_s=torch.tensor([row["source_t0_s"]], device=device),
            config=config,
        )
        future = torch.from_numpy(
            pair.source["target"][index : index + 1, count:].astype(np.float32)
        )[:, :, None].to(device)
        parent_relative = float(_relative_l2(parent[:, count:], future)[0])
        adapted_relative = float(_relative_l2(candidate[:, count:], future)[0])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "family": row["family"],
                "observed_count": count,
                "parent": parent_relative,
                "adapted": adapted_relative,
                "adaptation": adaptation,
            }
        )
    parent_values = np.asarray([row["parent"] for row in rows], np.float64)
    adapted_values = np.asarray([row["adapted"] for row in rows], np.float64)
    return {
        "parent_aggregate": float(parent_values.mean()),
        "adapted_aggregate": float(adapted_values.mean()),
        "relative_gain": float(
            (parent_values.mean() - adapted_values.mean())
            / max(parent_values.mean(), 1.0e-16)
        ),
        "nonworse": int((adapted_values <= parent_values).sum()),
        "accepted_online": int(sum(row["adaptation"]["accepted"] for row in rows)),
        "per_family": {
            family: {
                "parent": float(np.mean([row["parent"] for row in rows if row["family"] == family])),
                "adapted": float(np.mean([row["adapted"] for row in rows if row["family"] == family])),
            }
            for family in FAMILIES
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fit_source_cache",
        "fit_parent_cache",
        "fit_manifest",
        "holdout_source_cache",
        "holdout_parent_cache",
        "holdout_manifest",
        "preregistration",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--future-queries", type=int, default=4)
    parser.add_argument("--validation-tail-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=372)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        for path, key in (
            (Path(__file__), "trainer_sha256"),
            (args.fit_source_cache, "fit_source_cache_sha256"),
            (args.fit_parent_cache, "fit_parent_cache_sha256"),
            (args.fit_manifest, "fit_manifest_sha256"),
            (args.holdout_source_cache, "holdout_source_cache_sha256"),
            (args.holdout_parent_cache, "holdout_parent_cache_sha256"),
            (args.holdout_manifest, "holdout_manifest_sha256"),
        ):
            if _sha256(path) != bindings[key]:
                raise RuntimeError(f"binding drift: {path}")
        fit_manifest = json.loads(args.fit_manifest.read_text())
        holdout_manifest = json.loads(args.holdout_manifest.read_text())
        fit_groups = {row["group_id"] for row in fit_manifest["records"]}
        holdout_groups = {row["group_id"] for row in holdout_manifest["records"]}
        if fit_groups & holdout_groups:
            raise RuntimeError("fit/holdout group leakage")
        for manifest in (fit_manifest, holdout_manifest):
            if manifest.get("split") != "train":
                raise RuntimeError("B2-v11 development is train-only")
            if manifest.get("validation_opened") or manifest.get("test_id_opened"):
                raise RuntimeError("sealed split flag is open")
        with h5py.File(fit_manifest["source_h5"], "r", swmr=True) as source:
            global_time_s = torch.from_numpy(np.asarray(source["time_s"][:], np.float32))
        fit = _CachePair(args.fit_source_cache, args.fit_parent_cache, fit_manifest)
        holdout = _CachePair(
            args.holdout_source_cache, args.holdout_parent_cache, holdout_manifest
        )
        device = torch.device("cuda")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = LocalResidualMetaOperator(
            physics_channels=20,
            base_width=args.base_width,
            correction_cap=0.25,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
        )
        adapt_config = LocalMetaAdaptConfig(
            anchor_frames=8,
            validation_tail_frames=args.validation_tail_frames,
            ridge_fraction=1.0e-3,
            trust_ratio=0.05,
            minimum_observed_gain=0.0,
        )
        identity = {
            "schema": "b2_v11_local_meta_training_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "trainer_sha256": _sha256(Path(__file__)),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "base_width": args.base_width,
            "future_queries": args.future_queries,
            "validation_tail_frames": args.validation_tail_frames,
            "seed": args.seed,
            "online_trainable_parameters": 3,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        initial = _evaluate(model, holdout, global_time_s, device, adapt_config)
        _atomic_json(initial, args.output_dir / "initial_holdout.json")
        best = {"epoch": 0, **initial}
        torch.save(
            {"model_state": model.state_dict(), "epoch": 0, "identity": identity, "metrics": initial},
            args.output_dir / "best.pt",
        )
        _atomic_json(
            {key: value for key, value in best.items() if key != "rows"},
            args.output_dir / "best.json",
        )
        rng = np.random.default_rng(args.seed + 101)
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            episode_gains = []
            for index in rng.permutation(len(fit_manifest["records"])):
                row = fit_manifest["records"][int(index)]
                parent, target, conditioning = _record_tensors(fit, int(index), device)
                total = int(parent.shape[1])
                count = _observed_count(row, global_time_s, total)
                query = _query_indices(count, total, args.future_queries).to(device)
                start = int(row["window_start"])
                axis = global_time_s[start : start + total].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, report = meta_episode_loss(
                    model,
                    parent,
                    conditioning,
                    target,
                    observed_count=count,
                    query_indices_tensor=query,
                    time_s=axis,
                    source_f0_hz=torch.tensor([row["source_f0_hz"]], device=device),
                    source_t0_s=torch.tensor([row["source_t0_s"]], device=device),
                    anchor_frames=8,
                    validation_tail_frames=args.validation_tail_frames,
                    ridge_fraction=1.0e-3,
                    trust_ratio=0.05,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite B2-v11 meta loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                episode_gains.append(
                    (report["parent_relative_l2"] - report["candidate_relative_l2"])
                    / max(report["parent_relative_l2"], 1.0e-16)
                )
            holdout_metrics = _evaluate(
                model, holdout, global_time_s, device, adapt_config
            )
            event = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_query_relative_gain": float(np.mean(episode_gains)),
                "elapsed_s": time.time() - started,
                **{key: value for key, value in holdout_metrics.items() if key != "rows"},
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            if holdout_metrics["adapted_aggregate"] < best["adapted_aggregate"]:
                best = {"epoch": epoch, **holdout_metrics}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "identity": identity,
                        "metrics": holdout_metrics,
                    },
                    args.output_dir / "best.pt",
                )
                _atomic_json(
                    {key: value for key, value in best.items() if key != "rows"},
                    args.output_dir / "best.json",
                )
        passed = best["adapted_aggregate"] < best["parent_aggregate"]
        _atomic_json(
            {
                "status": "passed" if passed else "rejected",
                "best_epoch": best["epoch"],
                "best_parent_aggregate": best["parent_aggregate"],
                "best_adapted_aggregate": best["adapted_aggregate"],
                "best_relative_gain": best["relative_gain"],
                "elapsed_s": time.time() - started,
            },
            terminal,
        )
        fit.close()
        holdout.close()
        return 0
    except Exception as error:
        import traceback

        _atomic_json(
            {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()},
            terminal,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
