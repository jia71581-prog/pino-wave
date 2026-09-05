#!/usr/bin/env python3
"""R34 diagnostic: DRP-0.45 coarse input and R28 on selected train tails.

Only records listed in the train-only R28 manifest may be selected.  The DRP
wavefield is sealed before cached train truth is read.  Validation and test_id
are never selected or opened.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH = SCRIPT_PATH.with_name("build_r25_coarse_residual_cache.py")
R26_PATH = SCRIPT_PATH.with_name("train_r26_tail_spectral_pilot.py")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = import_file("r34_cache_components", BASE_PATH)
r26 = import_file("r34_r26_components", R26_PATH)
r25 = r26.r25


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    difference = prediction.astype(np.float64) - truth.astype(np.float64)
    target = truth.astype(np.float64)
    return math.sqrt(float(np.square(difference).sum()) / max(float(np.square(target).sum()), 1.0e-30))


def load_r28(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    model = r26.TailSpectralResidualUNet(
        base_width=int(config.get("base_width", 32)),
        correction_cap=float(config.get("correction_cap", 0.25)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def apply_model(
    model: torch.nn.Module,
    coarse_norm: np.ndarray,
    static: np.ndarray,
    times: np.ndarray,
    *,
    f0: float,
    t0: float,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    corrections: list[np.ndarray] = []
    static_tensor = torch.from_numpy(np.asarray(static, dtype=np.float32)).to(device)
    for start in range(0, len(times), int(batch_size)):
        stop = min(start + int(batch_size), len(times))
        coarse = torch.from_numpy(np.asarray(coarse_norm[start:stop], dtype=np.float32)).to(device)
        time_tensor = torch.from_numpy(np.asarray(times[start:stop], dtype=np.float32)).to(device)
        block = stop - start
        f0_tensor = torch.full((block,), float(f0), device=device)
        t0_tensor = torch.full((block,), float(t0), device=device)
        features = r25.make_dynamic_features(
            coarse,
            static_tensor[None].expand(block, -1, -1, -1),
            time_s=time_tensor,
            source_f0_hz=f0_tensor,
            source_t0_s=t0_tensor,
        )
        active = (time_tensor >= t0_tensor).float()
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if amp
            else nullcontext()
        )
        with context:
            correction = model(features, active=active).float()
        corrections.append(correction.cpu().numpy())
    return np.asarray(coarse_norm, dtype=np.float32) + np.concatenate(corrections, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--drp-fraction", type=float, default=0.45)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R34 requires CUDA")
    torch.cuda.set_device(device)

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != base.SCHEMA_MANIFEST:
        raise RuntimeError("unexpected R28 manifest schema")
    digest_payload = dict(manifest)
    selection_sha256 = str(digest_payload.pop("selection_sha256"))
    if base.canonical_json_sha256(digest_payload) != selection_sha256:
        raise RuntimeError("R28 manifest digest mismatch")
    all_rows = list(manifest["fit_records"]) + list(manifest["holdout_records"])
    by_id = {str(row["sample_id"]): row for row in all_rows}
    if any(sample_id not in by_id for sample_id in args.sample_id):
        raise RuntimeError("R34 sample is outside the train-only R28 manifest")
    selected_rows = [by_id[sample_id] for sample_id in args.sample_id]
    if any(str(row["split"]) != "train" for row in selected_rows):
        raise RuntimeError("R34 selected a non-train record")
    records = base.rows_to_records(selected_rows)

    holdout = r25.CacheCollection(args.holdout_cache, expected_subset="holdout")
    if str(holdout.selection_sha256) != selection_sha256:
        raise RuntimeError("holdout cache selection mismatch")
    cache_by_id: dict[str, tuple[int, int]] = {}
    for file_index, handle in enumerate(holdout.handles):
        for local_index, sample_id in enumerate(handle["sample_id"].asstr()[:]):
            cache_by_id[str(sample_id)] = (file_index, local_index)
    if any(sample_id not in cache_by_id for sample_id in args.sample_id):
        raise RuntimeError("R34 diagnostic is restricted to the already-opened R28 holdout")

    source_h5 = Path(str(manifest["source_h5"])).resolve()
    times, x_m, z_m = base._read_axes(source_h5)
    _, dx_m, dz_m = base._validate_axes(times, x_m, z_m, internal_dt_s=base.INTERNAL_DT_S)
    base._warm_up_cuda(
        dx_m=dx_m,
        dz_m=dz_m,
        internal_dt_s=base.INTERNAL_DT_S,
        npml=base.NPML,
        c_ref_mps=base.C_REF_MPS,
        device=device,
    )
    solver = base.LWC84CPMLSolver(
        grid=base.AcousticGrid(
            nx=base.GRID_SIZE,
            nz=base.GRID_SIZE,
            dx_m=dx_m,
            dz_m=dz_m,
            lx_m=float(x_m[-1] - x_m[0]),
            lz_m=float(z_m[-1] - z_m[0]),
            centering="node",
        ),
        boundaries=base.BoundaryConfig(npml=base.NPML),
        dt_s=base.INTERNAL_DT_S,
        output_times_s=times,
        c_ref_mps=base.C_REF_MPS,
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
        drp_max_nyquist_fraction=float(args.drp_fraction),
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model = load_r28(checkpoint_path, device)

    rows: list[dict] = []
    started = time.perf_counter()
    try:
        for record in records:
            velocity, source = base._read_generation_inputs(source_h5, [record])
            torch.cuda.synchronize(device)
            solve_started = time.perf_counter()
            result = solver.simulate(
                velocity,
                source_x_m=source["source_x_m"],
                source_z_m=source["source_z_m"],
                source_f0_hz=source["source_f0_hz"],
                source_t0_s=source["source_t0_s"],
                source_amplitude=source["source_amplitude"],
            )
            torch.cuda.synchronize(device)
            solve_seconds = time.perf_counter() - solve_started
            drp_physical = np.asarray(result.wavefield[0], dtype=np.float32).copy()
            sealed_before_truth_ns = time.monotonic_ns()

            file_index, local_index = cache_by_id[record.sample_id]
            handle = holdout.handles[file_index]
            scale = float(handle["field_scale"][local_index])
            standard = np.asarray(handle["coarse_norm"][local_index], dtype=np.float32)
            truth = np.asarray(handle["truth_norm"][local_index], dtype=np.float32)
            static = np.asarray(handle["static_features"][local_index], dtype=np.float32)
            if time.monotonic_ns() <= sealed_before_truth_ns:
                raise RuntimeError("predict-before-truth audit failed")
            drp = drp_physical / max(scale, 1.0e-12)
            f0 = float(handle["source_f0_hz"][local_index])
            t0 = float(handle["source_t0_s"][local_index])
            r28_standard = apply_model(
                model, standard, static, times, f0=f0, t0=t0, device=device,
                batch_size=int(args.batch_size), amp=bool(args.amp)
            )
            r28_drp = apply_model(
                model, drp, static, times, f0=f0, t0=t0, device=device,
                batch_size=int(args.batch_size), amp=bool(args.amp)
            )
            delta = (r28_drp.astype(np.float64) - r28_standard.astype(np.float64)).reshape(-1)
            residual = (r28_standard.astype(np.float64) - truth.astype(np.float64)).reshape(-1)
            denominator = float(np.dot(delta, delta))
            oracle_weight = float(np.clip(-float(np.dot(residual, delta)) / max(denominator, 1.0e-30), -2.0, 3.0))
            oracle = r28_standard + oracle_weight * (r28_drp - r28_standard)
            row = {
                "sample_id": record.sample_id,
                "family": record.medium_type,
                "group_id": record.group_id,
                "source_index": int(record.source_index),
                "source_f0_hz": f0,
                "standard_coarse_rel_l2": relative_l2(standard, truth),
                "drp_coarse_rel_l2": relative_l2(drp, truth),
                "r28_standard_rel_l2": relative_l2(r28_standard, truth),
                "r28_drp_rel_l2": relative_l2(r28_drp, truth),
                "r28_pair_oracle_weight_drp": oracle_weight,
                "r28_pair_oracle_rel_l2": relative_l2(oracle, truth),
                "solver_seconds": solve_seconds,
            }
            rows.append(row)
            print(json.dumps({"event": "R34_RECORD", **row}, sort_keys=True), flush=True)
            del result, drp_physical, drp, r28_standard, r28_drp, oracle
            torch.cuda.empty_cache()
    finally:
        holdout.close()

    payload = {
        "schema": "r34_drp45_r28_train_holdout_tail_diagnostic_v1",
        "role": "selected_R28_already_opened_train_holdout_diagnostic_only",
        "drp_max_nyquist_fraction": float(args.drp_fraction),
        "grid": [base.GRID_SIZE, base.GRID_SIZE],
        "spacing_m": [dx_m, dz_m],
        "internal_dt_s": base.INTERNAL_DT_S,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": base.sha256_file(checkpoint_path),
        "records": rows,
        "aggregate": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "max": float(np.max([row[key] for row in rows])),
            }
            for key in (
                "standard_coarse_rel_l2", "drp_coarse_rel_l2",
                "r28_standard_rel_l2", "r28_drp_rel_l2", "r28_pair_oracle_rel_l2"
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
        "validation_opened": False,
        "test_id_opened": False,
        "manifest_sha256": base.sha256_file(manifest_path),
        "selection_sha256": selection_sha256,
        "script_sha256": base.sha256_file(SCRIPT_PATH),
    }
    base.atomic_json(payload, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
