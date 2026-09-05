#!/usr/bin/env python3
"""Calibrate a train-only coefficient-norm abstention gate for PTLSA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v5_instance_adaptation import write_report
from scripts.run_pretrained_temporal_subspace_adaptation import PTLSA_SCHEMA
from scripts.train_v5_residual_meta import _sha256


RISK_SCHEMA = "pretrained_temporal_subspace_norm_risk_calibration_v1"


def calibrate(
    *,
    run_dirs: tuple[str | Path, ...],
    output: str | Path,
    family: str = "layered",
    minimum_nonworse_fraction: float = 0.95,
    minimum_acceptance_fraction: float = 0.20,
) -> dict[str, object]:
    normalized_dirs = tuple(Path(value).expanduser().resolve() for value in run_dirs)
    if not normalized_dirs:
        raise ValueError("at least one train-only PTLSA run is required")
    if not 0.0 <= float(minimum_nonworse_fraction) <= 1.0:
        raise ValueError("minimum nonworse fraction must lie in [0,1]")
    if not 0.0 < float(minimum_acceptance_fraction) <= 1.0:
        raise ValueError("minimum acceptance fraction must lie in (0,1]")

    rows: list[dict[str, object]] = []
    parent_identity: dict[str, object] | None = None
    method_identity: dict[str, object] | None = None
    seen: set[str] = set()
    for run_dir in normalized_dirs:
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        if summary.get("schema") != PTLSA_SCHEMA or summary.get("selection_split") != "train":
            raise ValueError(f"risk calibration requires a train-only PTLSA run: {summary_path}")
        shard_parent = dict(summary["parent"])
        shard_method = dict(summary.get("method") or {})
        if parent_identity is None:
            parent_identity = shard_parent
        elif shard_parent != parent_identity:
            raise ValueError("risk calibration parent identities do not match")
        if shard_method:
            comparable_method = {
                key: value
                for key, value in shard_method.items()
                if key not in {"config_sha256"}
            }
            if method_identity is None:
                method_identity = comparable_method
            elif comparable_method != method_identity:
                raise ValueError("risk calibration method identities do not match")
        for raw_report in summary["records"]:
            report = dict(raw_report)
            if str(report["medium_type"]) != str(family):
                continue
            sample_id = str(report["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate risk calibration sample: {sample_id}")
            seen.add(sample_id)
            adaptation = dict(report["adaptation"])
            if bool(adaptation.get("future_truth_used", True)):
                raise ValueError(f"future truth was used online for {sample_id}")
            if float(adaptation.get("observed_probe_weight", 0.0)) != 100.0:
                raise ValueError("risk calibration requires the frozen weight-100 probe")
            artifact = run_dir / sample_id / "adaptation.pt"
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            solver_coefficients = torch.as_tensor(
                payload["adaptation"].get(
                    "solver_coefficients", payload["adaptation"]["coefficients"]
                )
            ).double()
            parent_error = float(report["parent_future_fullfield_relative_l2"])
            adapted_error = float(report["future_fullfield_relative_l2"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "coefficient_l2_norm": float(solver_coefficients.norm()),
                    "parent_error": parent_error,
                    "adapted_error": adapted_error,
                    "relative_improvement": (
                        (parent_error - adapted_error) / max(parent_error, 1.0e-12)
                    ),
                    "artifact_sha256": _sha256(artifact),
                }
            )
    if not rows:
        raise ValueError("risk calibration selection is empty")

    candidates: list[dict[str, object]] = []
    for threshold in sorted({float(row["coefficient_l2_norm"]) for row in rows}):
        accepted = [float(row["coefficient_l2_norm"]) <= threshold for row in rows]
        gated_improvements = [
            float(row["relative_improvement"]) if keep else 0.0
            for row, keep in zip(rows, accepted)
        ]
        gated_errors = [
            float(row["adapted_error"]) if keep else float(row["parent_error"])
            for row, keep in zip(rows, accepted)
        ]
        acceptance_fraction = sum(accepted) / len(rows)
        nonworse_fraction = sum(value >= -1.0e-9 for value in gated_improvements) / len(rows)
        mean_improvement = sum(gated_improvements) / len(rows)
        parent_mean = sum(float(row["parent_error"]) for row in rows) / len(rows)
        adapted_mean = sum(gated_errors) / len(rows)
        candidates.append(
            {
                "threshold": threshold,
                "accepted_count": sum(accepted),
                "acceptance_fraction": acceptance_fraction,
                "nonworse_fraction": nonworse_fraction,
                "mean_per_record_relative_improvement": mean_improvement,
                "ratio_of_mean_improvement": (
                    (parent_mean - adapted_mean) / max(parent_mean, 1.0e-12)
                ),
                "eligible": (
                    acceptance_fraction >= float(minimum_acceptance_fraction)
                    and nonworse_fraction >= float(minimum_nonworse_fraction)
                    and mean_improvement > 0.0
                ),
            }
        )
    eligible = [value for value in candidates if bool(value["eligible"])]
    if not eligible:
        raise RuntimeError("no coefficient-norm threshold satisfies the calibration gate")
    selected = max(
        eligible,
        key=lambda value: (
            float(value["mean_per_record_relative_improvement"]),
            float(value["nonworse_fraction"]),
            float(value["acceptance_fraction"]),
        ),
    )
    checkpoint = {
        "schema": RISK_SCHEMA,
        "future_truth_scope": "train_split_after_adaptation_seal_only",
        "family": str(family),
        "record_count": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "source_run_dirs": [str(value) for value in normalized_dirs],
        "parent": parent_identity,
        "base_method": method_identity,
        "coefficient_statistic": "solver_coefficients_l2_norm",
        "maximum_coefficient_l2_norm": float(selected["threshold"]),
        "selection_constraints": {
            "minimum_nonworse_fraction": float(minimum_nonworse_fraction),
            "minimum_acceptance_fraction": float(minimum_acceptance_fraction),
        },
        "calibration_metrics": selected,
        "records": rows,
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite a PTLSA risk checkpoint")
    write_report(checkpoint, output_path)
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", default="layered")
    parser.add_argument("--minimum-nonworse-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-acceptance-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    calibrate(
        run_dirs=tuple(args.run_dir),
        output=args.output,
        family=args.family,
        minimum_nonworse_fraction=args.minimum_nonworse_fraction,
        minimum_acceptance_fraction=args.minimum_acceptance_fraction,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
