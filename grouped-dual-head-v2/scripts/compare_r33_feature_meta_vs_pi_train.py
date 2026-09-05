#!/usr/bin/env python3
"""Same-record all-401 train-only comparison of r33 and full-data PI-DeepONet.

The r33 predictions are already sealed and scored before this script runs.  This
script evaluates only the frozen PI-DeepONet checkpoint on those exact records
and future windows, then combines the independently bound squared-error terms.
It is a development comparison, not validation or test evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from patch_deeponet_baseline.model import PatchDeepONet, PatchDeepONetConfig
from patch_deeponet_baseline.training import dense_training_pair
from saved_time_phase_operator_v4.data import ExactStoredTimeBatchDataset
from saved_time_phase_operator_v4.full_support import FullSupportStepSpec
from scripts.train_grouped_v3_pilot import _to_device, load_normalizer
from scripts.train_saved_time_v4_full_support import _load_context


R33_SUMMARY_SHA256 = "ea42d7370ac42533c58ae7cf86e554c7a0ead50f0e69cf0300565dbca01b0f97"
PI_CHECKPOINT_SHA256 = "e2d6d7a9ec5481268bae2f41cff2400ee78b93a3a9216b2c338db572d405c4c1"
R33_ADAPTER_SHA256 = "77b8866f8aaba20d66bf5c2e0d9c0a1b93d39ca95dda1edf62795183cea42162"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _schedule(indices: tuple[int, ...]) -> tuple[FullSupportStepSpec, ...]:
    return tuple(
        FullSupportStepSpec(
            step=1_330_000 + offset,
            epoch=0,
            record_indices=(int(index),),
            appearance_indices=(0,),
        )
        for offset, index in enumerate(indices)
    )


def _relative(error: float, truth: float) -> float:
    return float(math.sqrt(float(error) / max(float(truth), 1.0e-30)))


def _aggregate(rows: list[dict[str, object]], error_key: str) -> dict[str, object]:
    def one(selected: list[dict[str, object]]) -> dict[str, float]:
        error = sum(float(row[error_key]) for row in selected)
        truth = sum(float(row["future_truth_squared_norm"]) for row in selected)
        return {
            "relative_l2": _relative(error, truth),
            "squared_error": error,
            "truth_squared_norm": truth,
        }

    families = sorted({str(row["family"]) for row in rows})
    return {
        "global": one(rows),
        "by_family": {
            family: one([row for row in rows if row["family"] == family])
            for family in families
        },
        "mean_record_relative_l2": float(
            sum(
                _relative(row[error_key], row["future_truth_squared_norm"])
                for row in rows
            )
            / len(rows)
        ),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, object]:
    if _sha256(args.r33_summary) != R33_SUMMARY_SHA256:
        raise ValueError("r33 summary binding changed")
    if _sha256(args.pi_checkpoint) != PI_CHECKPOINT_SHA256:
        raise ValueError("PI-DeepONet checkpoint binding changed")

    r33 = json.loads(args.r33_summary.read_text())
    if (
        r33.get("selection_split") != "train"
        or r33.get("refinement_mode") != "residual_trust_gate"
        or r33.get("adapter_checkpoint_sha256") != R33_ADAPTER_SHA256
        or len(r33.get("records", ())) != 6
        or not bool(r33.get("future_truth_opened_only_after_seal"))
    ):
        raise ValueError("r33 is not the registered sealed train-only result")

    r5b_identity = json.loads(args.r5b_identity.read_text())
    base, manifest, _ = _load_context(dict(r5b_identity["config"]))
    train_rows = tuple(row for row in manifest.records if row.split == "train")
    index_by_id = {row.sample_id: index for index, row in enumerate(train_rows)}
    r33_by_id = {str(row["sample_id"]): row for row in r33["records"]}
    sample_ids = tuple(str(row["sample_id"]) for row in r33["records"])
    if len(r33_by_id) != 6 or any(value not in index_by_id for value in sample_ids):
        raise ValueError("r33 sample binding is invalid")
    selected_indices = tuple(index_by_id[value] for value in sample_ids)

    pi_config = yaml.safe_load(args.pi_config.read_text())
    if pi_config.get("schema") != "pi_deeponet_training_config_v1":
        raise ValueError("unexpected PI-DeepONet training config")
    dataset = ExactStoredTimeBatchDataset(
        base.data.source_h5,
        manifest,
        split="train",
        schedule=_schedule(selected_indices),
        query_points=1,
        seed=int(pi_config["seed"]),
        time_policy="all_saved",
        frames_per_record=len(manifest.time_s),
        travel_time_h5=pi_config["travel_time_h5"],
    )
    device = torch.device(args.device)
    normalizer = load_normalizer(base, manifest.digest)
    pi = PatchDeepONet(
        PatchDeepONetConfig(**dict(pi_config.get("model", {})))
    ).to(device)
    payload = torch.load(args.pi_checkpoint, map_location=device, weights_only=True)
    if (
        payload.get("schema") != "patch_deeponet_checkpoint_v1"
        or payload.get("manifest_digest") != manifest.digest
    ):
        raise ValueError("PI-DeepONet checkpoint contract changed")
    pi.load_state_dict(payload["model_state"], strict=True)
    pi.eval()

    execution = dict(pi_config["model_execution"])
    lead_cycles = float(pi_config["loss"]["hard_causality_lead_cycles"])
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for position, batch in enumerate(dataset):
            sample_id = str(batch.sample_id[0])
            expected = sample_ids[position]
            if sample_id != expected:
                raise RuntimeError("all-saved comparison order changed")
            prediction_n, _, time_indices = dense_training_pair(
                pi,
                batch,
                normalizer,
                device,
                time_block=int(execution["time_block"]),
                query_chunk=int(execution["query_chunk"]),
                hard_causality_lead_cycles=lead_cycles,
            )
            expected_indices = torch.arange(len(manifest.time_s), dtype=torch.long)
            if not torch.equal(time_indices[0].cpu(), expected_indices):
                raise RuntimeError("PI comparison did not materialize all stored times")
            tensors = _to_device(batch, device)
            prediction = normalizer.decode_pressure(
                prediction_n, tensors["source_parameters"][:, 4]
            )
            truth = tensors["dense_target_physical"]
            observed = tuple(int(value) for value in r33_by_id[sample_id]["observed_indices"])
            future = slice(observed[1] + 1, len(manifest.time_s))
            prediction64 = prediction[:, future].double()
            truth64 = truth[:, future].double()
            pi_error = float((prediction64 - truth64).square().sum())
            truth_energy = float(truth64.square().sum())

            hybrid = r33_by_id[sample_id]["hybrid"]
            r33_truth = float(hybrid["future_truth_squared_norm"])
            if not math.isclose(truth_energy, r33_truth, rel_tol=2.0e-6, abs_tol=1.0e-30):
                raise RuntimeError("PI and r33 future truth energies differ")
            row = {
                "sample_id": sample_id,
                "family": str(r33_by_id[sample_id]["medium_type"]),
                "source_index": int(train_rows[selected_indices[position]].source_index),
                "observed_indices": observed,
                "future_frame_count": len(manifest.time_s) - observed[1] - 1,
                "future_truth_squared_norm": r33_truth,
                "r5b_parent_squared_error": float(hybrid["future_parent_squared_error"]),
                "our_instance_adapted_squared_error": float(
                    hybrid["future_adapted_squared_error"]
                ),
                "pi_deeponet_squared_error": pi_error,
            }
            for name in (
                "r5b_parent",
                "our_instance_adapted",
                "pi_deeponet",
            ):
                row[f"{name}_relative_l2"] = _relative(
                    row[f"{name}_squared_error"], r33_truth
                )
            rows.append(row)
            print(
                f"[{position + 1}/6] {sample_id} "
                f"ours={row['our_instance_adapted_relative_l2']:.6g} "
                f"pi={row['pi_deeponet_relative_l2']:.6g}",
                flush=True,
            )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()

    metrics = {
        name: _aggregate(rows, f"{name}_squared_error")
        for name in ("r5b_parent", "our_instance_adapted", "pi_deeponet")
    }
    ours = float(metrics["our_instance_adapted"]["global"]["relative_l2"])
    parent = float(metrics["r5b_parent"]["global"]["relative_l2"])
    pi_value = float(metrics["pi_deeponet"]["global"]["relative_l2"])
    output = {
        "schema": "r33_feature_meta_vs_pi_train_all401_v1",
        "status": "complete_train_only_comparison",
        "role": "development_evidence_not_validation_or_test_evidence",
        "method_identity": {
            "ours": "frozen_r5b_plus_parent_residual_conditioned_instance_finetuning",
            "baseline": "full_training_set_pi_deeponet",
            "physical_wavefield_propagator": False,
            "pde_loss_weight": 0.0,
        },
        "protocol": {
            "split": "train",
            "record_count": len(rows),
            "family_count": dict(defaultdict(int)),
            "frames": "all saved frames strictly after the second guarded onset observation",
            "same_records": True,
            "same_future_time_indices": True,
            "physical_pressure_float64_energy_metric": True,
            "validation_opened": False,
            "test_id_opened": False,
        },
        "metrics": metrics,
        "comparison": {
            "our_relative_improvement_from_r5b_parent": (parent - ours) / parent,
            "our_relative_improvement_over_pi": (pi_value - ours) / pi_value,
            "pi_error_over_our_error": pi_value / ours,
            "our_better_than_pi_on_this_train_panel": ours < pi_value,
        },
        "records": rows,
        "bindings": {
            "r33_summary": str(args.r33_summary.resolve()),
            "r33_summary_sha256": R33_SUMMARY_SHA256,
            "r33_adapter_checkpoint_sha256": R33_ADAPTER_SHA256,
            "pi_checkpoint": str(args.pi_checkpoint.resolve()),
            "pi_checkpoint_sha256": PI_CHECKPOINT_SHA256,
            "pi_checkpoint_epoch": int(payload["epoch"]),
            "pi_checkpoint_global_step": int(payload["global_step"]),
            "pi_training_records": 2240,
            "pi_training_epochs": 100,
            "r5b_identity_sha256": _sha256(args.r5b_identity),
            "pi_config_sha256": _sha256(args.pi_config),
            "script_sha256": _sha256(Path(__file__)),
            "manifest_digest": manifest.digest,
        },
        "elapsed_s": float(time.perf_counter() - started),
    }
    family_count: dict[str, int] = defaultdict(int)
    for row in rows:
        family_count[str(row["family"])] += 1
    output["protocol"]["family_count"] = dict(family_count)
    _atomic_json(output, args.output)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r33-summary", type=Path, required=True)
    parser.add_argument("--r5b-identity", type=Path, required=True)
    parser.add_argument("--pi-config", type=Path, required=True)
    parser.add_argument("--pi-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
