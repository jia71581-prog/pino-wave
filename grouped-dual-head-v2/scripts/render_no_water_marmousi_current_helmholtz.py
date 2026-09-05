#!/usr/bin/env python3
"""Render the current audited Helmholtz checkpoint on a derived no-water crop."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from fno_acoustic.data_generation.model_marmousi import _load_velocity
from grouped_ufno_mionet_v3.data.index import build_manifest
from saved_time_phase_operator_v4.losses import apply_hard_causality, source_causality_onset_s
from saved_time_phase_operator_v4.probe import ProbeVariant
from scripts.diagnose_capacity_ladder_overfit import build_base_config, build_probe_config
from scripts.diagnose_helmholtz_g3_heldout import warmstart_full_helmholtz
from scripts.render_current_helmholtz_three_family import (
    _plot_receiver_gather,
    _plot_receiver_waveforms,
    _plot_velocity_model,
    _plot_wavefields,
    _relative_l2,
    _sha256,
    _snapshot_indices,
)
from scripts.train_grouped_v3_pilot import load_normalizer
from scripts.train_saved_time_v4_probe import _model


EXPECTED_SCHEMAS = {
    "current_helmholtz_no_water_marmousi_visualization_v1",
    "current_helmholtz_no_water_marmousi_visualization_v2",
}


def _content_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-h5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snapshot-count", type=int, default=6)
    parser.add_argument("--time-block", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    sample_h5 = args.sample_h5.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    identity = json.loads((run_dir / "run_identity.json").read_text())
    terminal = json.loads((run_dir / "terminal.json").read_text())
    if terminal.get("status") != "complete":
        raise ValueError("selected Helmholtz run is not complete")
    checkpoint = Path(str(terminal["best_checkpoint"])).expanduser().resolve()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_payload.get("manifest_digest") != identity["manifest_digest"]:
        raise ValueError("checkpoint and run identity manifest digests disagree")
    if checkpoint_payload.get("config_digest") != identity["run_digest"]:
        raise ValueError("checkpoint and run identity config digests disagree")
    checkpoint_metric = float(
        checkpoint_payload.get("metrics", {}).get("triplet_aggregate_relative_l2", np.nan)
    )
    if not np.isclose(
        checkpoint_metric,
        float(terminal["best_fixed_heldout_relative_l2"]),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError("checkpoint no longer reproduces its terminal-bound metric")
    checkpoint_global_step = int(checkpoint_payload.get("global_step", -1))
    del checkpoint_payload

    with h5py.File(sample_h5, "r", swmr=True) as handle:
        if str(handle.attrs.get("schema", "")) not in EXPECTED_SCHEMAS:
            raise ValueError("custom sample schema is not recognized")
        if str(handle.attrs.get("status", "")) != "complete":
            raise ValueError("custom sample is incomplete")
        sample_id = str(handle.attrs["sample_id"])
        group_id = str(handle.attrs["group_id"])
        sample_content_sha256 = str(handle.attrs["content_sha256"])
        selection_basis = str(handle.attrs["selection_basis"])
        source_model_name = str(handle.attrs.get("source_model_name", "Marmousi-derived"))
        source_velocity_file = str(handle.attrs["source_velocity_file"])
        source_velocity_sha256 = str(handle.attrs["source_velocity_sha256"])
        source_provenance_json = str(handle.attrs.get("source_provenance_json", ""))
        source_provenance_sha256 = str(handle.attrs.get("source_provenance_sha256", ""))
        crop_x0_m = float(handle.attrs["crop_x0_m"])
        crop_z0_m = float(handle.attrs["crop_z0_m"])
        water_fraction = float(handle.attrs["water_fraction_le_1500_5"])
        time_s = np.asarray(handle["time_s"][:], dtype=np.float32)
        x_m = np.asarray(handle["x_m"][:], dtype=np.float32)
        z_m = np.asarray(handle["z_m"][:], dtype=np.float32)
        source_values = np.asarray(handle["source_parameters"][:], dtype=np.float32)
        source_map = np.asarray(handle["source_map"][:], dtype=np.float32)
        velocity_mps = np.asarray(handle["velocity_mps"][:], dtype=np.float32)
        travel_time_s = np.asarray(handle["travel_time_s"][:], dtype=np.float32)
        target = np.asarray(handle["wavefield_target"][:], dtype=np.float32)
        background = np.asarray(handle["background_pbg"][:], dtype=np.float32)
    observed_content_sha256 = _content_sha256(
        velocity_mps,
        target,
        background,
        travel_time_s,
        source_map,
        source_values,
    )
    if observed_content_sha256 != sample_content_sha256:
        raise ValueError("custom sample content hash mismatch")
    source_velocity_path = Path(source_velocity_file).expanduser().resolve()
    if _sha256(source_velocity_path) != source_velocity_sha256:
        raise ValueError("registered Marmousi source hash mismatch")
    source_velocity, source_velocity_loader = _load_velocity(source_velocity_path)
    source_velocity = np.squeeze(np.asarray(source_velocity, dtype=np.float32))
    if source_velocity.ndim != 2 or not np.isfinite(source_velocity).all():
        raise ValueError("registered Marmousi source is not a finite 2D velocity model")
    source_velocity_color_limits_mps = (
        float(source_velocity.min()),
        float(source_velocity.max()),
    )
    if source_provenance_json:
        provenance_path = Path(source_provenance_json).expanduser().resolve()
        if _sha256(provenance_path) != source_provenance_sha256:
            raise ValueError("Marmousi source provenance hash mismatch")
        source_provenance = json.loads(provenance_path.read_text())
    else:
        source_provenance = None
    if water_fraction != 0.0 or np.any(velocity_mps <= 1500.5):
        raise ValueError("custom Marmousi sample is not water-free")
    if float(np.mean(velocity_mps[-20:])) <= float(np.mean(velocity_mps[:20])):
        raise ValueError("custom Marmousi sample has an invalid depth trend")

    device = torch.device(args.device)
    torch.manual_seed(int(identity["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(identity["seed"]))
    base = build_base_config(int(identity["width"]))
    base = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            normalization_json=str(args.normalization_json.expanduser().resolve()),
        ),
    )
    manifest = build_manifest(base.data.source_h5)
    if manifest.digest != identity["manifest_digest"]:
        raise ValueError("source manifest no longer matches the completed run")
    normalizer = load_normalizer(base, manifest.digest)
    variant = ProbeVariant(
        depth=int(identity["dense_depth"]),
        use_local_phase=True,
        spectral_rank=int(identity["dense_spectral_rank"]),
        modes=int(identity["dense_modes"]),
        temporal_basis_rank=0,
        family_expert_rank=0,
        local_field=True,
        local_field_residual=False,
        local_field_helmholtz_synthesis=True,
        local_field_helmholtz_synthesis_frequencies=int(identity["helmholtz_frequencies"]),
        local_field_helmholtz_synthesis_wkb_phase=bool(identity["helmholtz_wkb_phase"]),
        local_field_helmholtz_synthesis_rank=int(identity["helmholtz_rank"]),
        local_field_helmholtz_synthesis_late_rank=int(identity.get("helmholtz_late_rank", 0)),
        local_field_helmholtz_synthesis_late_frequencies=int(
            identity.get("helmholtz_late_frequencies", 0)
        ),
        local_field_helmholtz_spectral_bypass=bool(
            identity.get("helmholtz_spectral_bypass", False)
        ),
        local_field_helmholtz_spectral_bypass_per_branch=bool(
            identity.get("helmholtz_spectral_bypass_per_branch", False)
        ),
        local_field_helmholtz_background_conditioning=bool(
            identity.get("helmholtz_background_conditioning", False)
        ),
        local_field_helmholtz_background_sigma_cells=float(
            identity.get("helmholtz_background_sigma_cells", 2.0)
        ),
        local_field_helmholtz_background_global_propagator=bool(
            identity.get("helmholtz_background_global_propagator", False)
        ),
        local_field_helmholtz_background_propagation_modes=int(
            identity.get("helmholtz_background_propagation_modes", 48)
        ),
    )
    model = _model(base, manifest, variant).to(device)
    transfer = warmstart_full_helmholtz(model, checkpoint)
    if int(transfer["transferred_tensor_count"]) != len(model.state_dict()):
        raise ValueError("checkpoint did not populate every model tensor")
    model.eval()
    config = build_probe_config(
        dense_lr=float(identity["dense_learning_rate"]),
        backbone_lr=float(identity["backbone_learning_rate"]),
        temporal_lr=1.0e-5,
        seed=int(identity["seed"]),
        travel_time_h5="",
        family_gradient_weights=identity.get("family_gradient_weights"),
        per_frame_frame=bool(identity.get("per_frame_frame", False)),
        frame_energy_floor_fraction=float(identity.get("frame_energy_floor_fraction", 0.0)),
        late_frame_gain=float(identity.get("late_frame_gain", 0.0)),
        late_frame_start_fraction=float(identity.get("late_frame_start_fraction", 0.4)),
    )

    velocity_tensor = torch.from_numpy(velocity_mps)[None, None].to(device)
    source_tensor = torch.from_numpy(source_values)[None].to(device)
    source_map_tensor = torch.from_numpy(source_map)[None, None].to(device)
    requested_time_tensor = torch.from_numpy(time_s)[None].to(device)
    travel_tensor = torch.from_numpy(travel_time_s)[None].to(device)
    x_tensor = torch.from_numpy(x_m).to(device)
    z_tensor = torch.from_numpy(z_m).to(device)
    prepared = model.prepare_sources(
        model.encode_medium(velocity_tensor, normalizer),
        source_tensor,
        source_map_tensor,
        normalizer,
        record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
    )
    dense_grid = model.prepare_dense_grid(
        prepared,
        x_m=x_tensor,
        z_m=z_tensor,
        travel_time_s=travel_tensor,
    )
    prediction_norm, background_reference_norm = model.dense_normalized_with_coarse(
        prepared,
        requested_time_tensor,
        dense_grid=dense_grid,
        time_block=int(args.time_block),
    )
    background_tensor = torch.from_numpy(background)[None].to(device)
    background_norm = normalizer.encode_pressure(background_tensor, source_tensor[:, 4])
    prediction_norm = prediction_norm + background_norm
    background_reference_norm = background_reference_norm + background_norm
    if bool(config["loss"].get("hard_causality", False)):
        onset = source_causality_onset_s(
            source_tensor,
            lead_cycles=float(config["loss"].get("hard_causality_lead_cycles", 0.0)),
        )
        prediction_norm = apply_hard_causality(
            prediction_norm, requested_time_tensor, onset
        )
        background_reference_norm = apply_hard_causality(
            background_reference_norm, requested_time_tensor, onset
        )
    prediction = (
        normalizer.decode_pressure(prediction_norm.float(), source_tensor[:, 4])
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    background_reference = (
        normalizer.decode_pressure(background_reference_norm.float(), source_tensor[:, 4])
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    snap_idx = _snapshot_indices(time_s, float(source_values[3]), int(args.snapshot_count))
    velocity_path = output_dir / "marmousi_true_velocity_model.png"
    wavefield_path = output_dir / "marmousi_wavefield_snapshots.png"
    receiver_path = output_dir / "marmousi_receiver_waveforms.png"
    gather_path = output_dir / "marmousi_receiver_gather.png"
    plotted_data_path = output_dir / "marmousi_plotted_data.npz"
    _plot_velocity_model(
        velocity_mps,
        x_m,
        z_m,
        float(source_values[0]),
        float(source_values[1]),
        f"{source_model_name} no-water complex crop",
        sample_id,
        group_id,
        velocity_path,
        color_limits_mps=source_velocity_color_limits_mps,
    )
    frame_errors = _plot_wavefields(
        target,
        prediction,
        time_s,
        snap_idx,
        x_m,
        z_m,
        float(source_values[0]),
        float(source_values[1]),
        f"{source_model_name} no-water complex crop",
        wavefield_path,
    )
    receiver_z, receiver_x, trace_errors = _plot_receiver_waveforms(
        target,
        prediction,
        time_s,
        x_m,
        z_m,
        f"{source_model_name} no-water complex crop",
        receiver_path,
    )
    _plot_receiver_gather(
        target,
        prediction,
        time_s,
        x_m,
        z_m,
        receiver_z,
        float(source_values[0]),
        f"{source_model_name} no-water complex crop",
        gather_path,
    )
    np.savez_compressed(
        plotted_data_path,
        time_s=time_s,
        snapshot_indices=snap_idx,
        snapshot_target=target[snap_idx],
        snapshot_prediction=prediction[snap_idx],
        receiver_x_indices=receiver_x,
        receiver_z_index=np.asarray(receiver_z),
        receiver_target=target[:, receiver_z, receiver_x],
        receiver_prediction=prediction[:, receiver_z, receiver_x],
        x_m=x_m,
        z_m=z_m,
        velocity_mps=velocity_mps,
        source_parameters=source_values,
    )
    full_rel = _relative_l2(prediction, target)
    background_rel = _relative_l2(background_reference, target)
    correction_ratio = float(
        np.linalg.norm((prediction - background_reference).astype(np.float64))
        / max(np.linalg.norm(background_reference.astype(np.float64)), 1.0e-30)
    )
    report = {
        "status": "complete",
        "protocol": "derived_verified_no_water_marmousi_all_401_stored_times",
        "terminal_metric_comparable": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_global_step": checkpoint_global_step,
        "run_digest": identity["run_digest"],
        "manifest_digest": manifest.digest,
        "sample_h5": str(sample_h5),
        "sample_h5_sha256": _sha256(sample_h5),
        "sample_content_sha256": sample_content_sha256,
        "sample_id": sample_id,
        "group_id": group_id,
        "source_velocity_file": source_velocity_file,
        "source_velocity_sha256": source_velocity_sha256,
        "source_model_name": source_model_name,
        "source_provenance_json": source_provenance_json,
        "source_provenance_sha256": source_provenance_sha256,
        "source_provenance": source_provenance,
        "crop_x0_m": crop_x0_m,
        "crop_z0_m": crop_z0_m,
        "selection_basis": selection_basis,
        "water_fraction_le_1500_5": water_fraction,
        "velocity_range_mps": [float(velocity_mps.min()), float(velocity_mps.max())],
        "velocity_color_scale_mps": list(source_velocity_color_limits_mps),
        "velocity_color_scale_basis": "verified_source_global_range",
        "source_velocity_loader": source_velocity_loader,
        "top_200m_mean_velocity_mps": float(np.mean(velocity_mps[:20])),
        "bottom_200m_mean_velocity_mps": float(np.mean(velocity_mps[-20:])),
        "source_parameters": [float(value) for value in source_values],
        "full_wavefield_relative_l2": full_rel,
        "background_only_relative_l2": background_rel,
        "neural_correction_to_background_l2_ratio": correction_ratio,
        "snapshot_indices": [int(value) for value in snap_idx],
        "snapshot_times_s": [float(value) for value in time_s[snap_idx]],
        "snapshot_frame_relative_l2": frame_errors,
        "receiver_z_m": float(z_m[receiver_z]),
        "receiver_x_m": [float(x_m[index]) for index in receiver_x],
        "receiver_trace_relative_l2": trace_errors,
        "receiver_mean_relative_l2": float(np.mean(trace_errors)),
        "outputs": {
            "true_velocity_model": str(velocity_path),
            "wavefield_snapshots": str(wavefield_path),
            "receiver_waveforms": str(receiver_path),
            "receiver_gather": str(gather_path),
            "plotted_data": str(plotted_data_path),
        },
    }
    report_path = output_dir / "render_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "sample_id": sample_id,
                "full_relative_l2": full_rel,
                "receiver_mean_relative_l2": float(np.mean(trace_errors)),
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
