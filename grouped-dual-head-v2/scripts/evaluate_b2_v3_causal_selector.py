#!/usr/bin/env python3
"""Test a prefix-only selector over anchor and two frozen B2 checkpoints."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator, build_cache

FAMILIES = ("uniform", "layered", "marmousi")
TEMPORAL_BANDS = {"early": (0, 19), "middle": (19, 38), "late": (38, 56)}
FREQUENCY_BANDS = {"low": (0, 5), "middle": (5, 12), "high": (12, 29)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _relative_l2(prediction, target, dims):
    numerator = (prediction - target).abs().square().sum(dims).sqrt()
    denominator = target.abs().square().sum(dims).clamp_min(1.0e-16).sqrt()
    return numerator / denominator


def choose_by_prefix(prefix_errors: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return option names and per-record argmin using prefix errors only."""
    names = list(prefix_errors)
    matrix = np.stack([prefix_errors[name] for name in names], axis=0)
    return names, np.argmin(matrix, axis=0)


def judge_selector(summary: dict, gate: dict) -> dict:
    aggregate_pass = (
        summary["aggregate"]
        <= summary["baseline_aggregate"] - float(gate["minimum_absolute_gain_vs_anchor"])
    )
    family_pass = all(
        summary["per_family"][name]["candidate"]
        < summary["per_family"][name]["baseline"]
        for name in FAMILIES
    )
    nonworse_pass = summary["nonworse_count"] >= int(gate["minimum_nonworse_records"])
    return {
        "aggregate_pass": aggregate_pass,
        "family_pass": family_pass,
        "nonworse_pass": nonworse_pass,
        "passed": aggregate_pass and family_pass and nonworse_pass,
    }


def _evaluate_option(checkpoint_path: Path | None, cache, device) -> dict:
    model = None
    name = "anchor"
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        identity = checkpoint["identity"]
        name = checkpoint_path.parent.name
        model = FrameConditionedPropagator(
            state_channels=int(identity["ic_frames"]),
            cond_channels=int(cache["cond"].shape[1]),
            width=int(identity["width"]),
            spectral_rank=int(identity["spectral_rank"]),
            modes=24,
            depth=4,
            gate_init=1.0,
            activation_checkpointing=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

    rows = []
    sample_ids = cache["sample_id"][:].astype(str)
    families = cache["family"][:].astype(str)
    for lo in range(0, len(sample_ids), 2):
        hi = min(lo + 2, len(sample_ids))
        base = torch.from_numpy(cache["base_seq"][lo:hi].astype(np.float32))[:, :, None].to(device)
        target = torch.from_numpy(cache["target"][lo:hi].astype(np.float32))[:, :, None].to(device)
        if model is None:
            prediction = base
        else:
            cond = torch.from_numpy(cache["cond"][lo:hi].astype(np.float32)).to(device)
            initial = target[:, :8, 0]
            with torch.no_grad():
                prediction = model.forward_anchored(base, cond, initial_state=initial)
        pred_prefix = prediction[:, :8, 0].double()
        truth_prefix = target[:, :8, 0].double()
        pred_future = prediction[:, 8:, 0].double()
        truth_future = target[:, 8:, 0].double()
        prefix = _relative_l2(pred_prefix, truth_prefix, (1, 2, 3))
        future = _relative_l2(pred_future, truth_future, (1, 2, 3))
        temporal = {
            band: _relative_l2(
                pred_future[:, start:stop], truth_future[:, start:stop], (1, 2, 3)
            )
            for band, (start, stop) in TEMPORAL_BANDS.items()
        }
        pred_fft = torch.fft.rfft(pred_future, dim=1)
        truth_fft = torch.fft.rfft(truth_future, dim=1)
        frequency = {
            band: _relative_l2(
                pred_fft[:, start:stop], truth_fft[:, start:stop], (1, 2, 3)
            )
            for band, (start, stop) in FREQUENCY_BANDS.items()
        }
        for offset in range(hi - lo):
            rows.append({
                "sample_id": sample_ids[lo + offset],
                "family": families[lo + offset],
                "prefix": float(prefix[offset]),
                "future": float(future[offset]),
                "temporal": {
                    band: float(value[offset]) for band, value in temporal.items()
                },
                "frequency": {
                    band: float(value[offset]) for band, value in frequency.items()
                },
            })
    return {"name": name, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--build-cache", action="store_true")
    args = parser.parse_args()
    if len(args.checkpoint) != 2:
        raise SystemExit("exactly two --checkpoint values are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal_path = args.output_dir / "terminal.json"
    try:
        manifest = json.loads(args.manifest.read_text())
        prereg = json.loads(args.preregistration.read_text())
        if manifest.get("split") != "train":
            raise RuntimeError("selector manifest must be train-only")
        if manifest.get("validation_opened") or manifest.get("test_id_opened"):
            raise RuntimeError("sealed split flag is open")
        bindings = prereg["bindings"]
        if _sha256(Path(__file__)) != bindings["selector_sha256"]:
            raise RuntimeError("selector code binding drift")
        if _sha256(args.manifest) != bindings["manifest_sha256"]:
            raise RuntimeError("manifest binding drift")
        expected = bindings["checkpoint_sha256"]
        for checkpoint in args.checkpoint:
            if _sha256(checkpoint) != expected[str(checkpoint)]:
                raise RuntimeError(f"checkpoint binding drift: {checkpoint}")
        if args.cache.exists():
            raise FileExistsError(f"refusing to reuse cache: {args.cache}")
        if not args.build_cache:
            raise RuntimeError("cache is absent; pass --build-cache")
        device = torch.device("cuda")
        build_cache(manifest, device, cache_path=args.cache, data_split="train")
        with h5py.File(args.cache, "r", swmr=True) as cache:
            options = [_evaluate_option(None, cache, device)]
            options.extend(_evaluate_option(path, cache, device) for path in args.checkpoint)
        prefix_errors = {
            option["name"]: np.asarray([row["prefix"] for row in option["rows"]])
            for option in options
        }
        names, selected_indices = choose_by_prefix(prefix_errors)
        option_by_name = {option["name"]: option for option in options}
        anchor_rows = option_by_name["anchor"]["rows"]
        selected_rows = []
        for index, selected_index in enumerate(selected_indices):
            option_name = names[int(selected_index)]
            row = option_by_name[option_name]["rows"][index]
            selected_rows.append({
                **row,
                "selected_option": option_name,
                "anchor_future": anchor_rows[index]["future"],
                "future_delta": row["future"] - anchor_rows[index]["future"],
            })
        summary = {
            "schema": "b2_v3_causal_selector_metrics_v1",
            "record_count": len(selected_rows),
            "aggregate": float(np.mean([row["future"] for row in selected_rows])),
            "baseline_aggregate": float(np.mean([row["anchor_future"] for row in selected_rows])),
            "maximum": float(max(row["future"] for row in selected_rows)),
            "nonworse_count": sum(row["future_delta"] <= 0.0 for row in selected_rows),
            "selection_counts": {
                name: sum(row["selected_option"] == name for row in selected_rows)
                for name in names
            },
            "per_family": {},
            "temporal": {},
            "frequency": {},
            "options_unselected": {
                option["name"]: {
                    "prefix": float(np.mean([row["prefix"] for row in option["rows"]])),
                    "future": float(np.mean([row["future"] for row in option["rows"]])),
                }
                for option in options
            },
            "records": selected_rows,
        }
        for family in FAMILIES:
            rows = [row for row in selected_rows if row["family"] == family]
            summary["per_family"][family] = {
                "candidate": float(np.mean([row["future"] for row in rows])),
                "baseline": float(np.mean([row["anchor_future"] for row in rows])),
                "nonworse": sum(row["future_delta"] <= 0.0 for row in rows),
                "n": len(rows),
            }
        for band in TEMPORAL_BANDS:
            summary["temporal"][band] = float(
                np.mean([row["temporal"][band] for row in selected_rows])
            )
        for band in FREQUENCY_BANDS:
            summary["frequency"][band] = float(
                np.mean([row["frequency"][band] for row in selected_rows])
            )
        summary["judgement"] = judge_selector(summary, prereg["gate"])
        _atomic_json(summary, args.output_dir / "metrics.json")
        identity = {
            "schema": "b2_v3_causal_selector_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "cache": str(args.cache),
            "cache_sha256": _sha256(args.cache),
            "selector_sha256": _sha256(Path(__file__)),
            "checkpoint_sha256": expected,
            "selection_inputs": "first eight true frames only",
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        _atomic_json({
            "status": "passed" if summary["judgement"]["passed"] else "rejected",
            "judgement": summary["judgement"],
            "metrics": str(args.output_dir / "metrics.json"),
            "run_identity": str(args.output_dir / "run_identity.json"),
        }, terminal_path)
        print(json.dumps({"terminal": str(terminal_path), **json.loads(terminal_path.read_text())}, indent=2))
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, terminal_path)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
