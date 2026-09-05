#!/usr/bin/env python3
"""Fail-closed self-test for the packaged v22 radial-guard candidate."""
from __future__ import annotations

import inspect
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
SPEC = ROOT / "results/r16_dscp_v22_guarded_candidate_spec_20260827.json"
OUT = ROOT / "results/r16_dscp_v22_guarded_candidate"
BASIS = ROOT / "results/r16_dscp_v1/basis_rank16.pt"
CHECKPOINT = ROOT / "results/r16_dscp_v18/B_data_hinge/best.pt"

from saved_time_phase_operator_v4.band_adapter import registered_high_band_mask  # noqa: E402
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (  # noqa: E402
    predictor_parameter_count,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_wide128_radial_guard import (  # noqa: E402
    R16DSCPWide128RadialGuard,
    WIDE128_PARAMETERS,
)
from scripts.train_r16_dscp_v16 import atomic_json, sha256_file, utc_now  # noqa: E402


def main() -> int:
    if not SPEC.exists():
        raise RuntimeError(f"missing frozen spec: {SPEC}")
    spec = json.loads(SPEC.read_text())
    bindings = spec["bindings"]
    observed = {
        "script_sha256": sha256_file(Path(__file__)),
        "module_sha256": sha256_file(
            ROOT
            / "saved_time_phase_operator_v4/instance_adaptation/"
            "r16_dscp_wide128_radial_guard.py"
        ),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "basis_sha256": sha256_file(BASIS),
    }
    for name, value in observed.items():
        if value != bindings[name]:
            raise RuntimeError(f"binding mismatch: {name}")
    if OUT.exists():
        raise RuntimeError(f"self-test output already exists: {OUT}")

    started = time.monotonic()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    artifact = torch.load(BASIS, map_location="cpu", weights_only=False)
    model, payload = R16DSCPWide128RadialGuard.from_v18_checkpoint(
        artifact["basis"], artifact["coefficient_scales"], CHECKPOINT
    )
    model = model.to(device).float().eval()
    head_state_exact = all(
        torch.equal(model.head.state_dict()[name].cpu(), value.cpu())
        for name, value in payload["model_state"].items()
    )
    if not head_state_exact:
        raise RuntimeError("v18 checkpoint did not load exactly")
    if predictor_parameter_count(model) != WIDE128_PARAMETERS:
        raise RuntimeError("guarded candidate parameter count mismatch")

    forbidden = (
        "truth",
        "target",
        "family",
        "medium_type",
        "split",
        "sample",
        "group",
        "oracle",
    )
    forbidden_signature_parameters = {}
    for method in (
        R16DSCPWide128RadialGuard.predict_coefficients,
        R16DSCPWide128RadialGuard.forward,
    ):
        hits = [
            name
            for name in inspect.signature(method).parameters
            if any(token in name.lower() for token in forbidden)
        ]
        forbidden_signature_parameters[method.__name__] = hits
        if hits:
            raise RuntimeError(f"deployment signature contains forbidden parameters: {hits}")

    generator = torch.Generator(device=device).manual_seed(372)
    velocity = torch.full((1, 1, 201, 201), 2000.0, device=device)
    source_map = torch.zeros_like(velocity)
    source_map[0, 0, 20, 100] = 1.0
    travel = torch.ones_like(velocity)
    x_m = torch.linspace(0.0, 2000.0, 201, device=device)
    z_m = torch.linspace(0.0, 2000.0, 201, device=device)
    parent = torch.randn(
        (1, 401, 201, 201), generator=generator, device=device, dtype=torch.float32
    ) * 1.0e-8
    k0 = torch.tensor([20], device=device)
    k1 = torch.tensor([21], device=device)
    observed0 = parent[:, 20, None].clone()
    observed1 = parent[:, 21, None].clone()
    with torch.no_grad():
        coefficients, decisions, conditions = model.predict_coefficients(
            velocity,
            source_map,
            travel,
            x_m,
            z_m,
            observed0,
            observed1,
            parent,
            k0,
            k1,
        )
        spectrum = torch.fft.rfft2(coefficients.float(), norm="ortho")
        high = registered_high_band_mask(201, 201, spectrum.device)
        high_energy_fraction = float(
            spectrum[..., high].abs().square().sum()
            / spectrum.abs().square().sum().clamp_min(1.0e-30)
        )
        corrected = model(
            velocity,
            source_map,
            travel,
            x_m,
            z_m,
            observed0,
            observed1,
            parent,
            k0,
            k1,
        )
        correction = corrected - parent
    causal_max = float(correction[:, :22].abs().max())
    top_row_max = float(correction[:, :, 0].abs().max())
    future_energy = float(correction[:, 22:].double().square().sum())
    gates = {
        "checkpoint_state_exact": {"value": head_state_exact, "passed": head_state_exact},
        "parameter_count": {
            "value": predictor_parameter_count(model),
            "expected": WIDE128_PARAMETERS,
            "passed": predictor_parameter_count(model) == WIDE128_PARAMETERS,
        },
        "deployment_signature_label_free": {
            "value": forbidden_signature_parameters,
            "passed": not any(forbidden_signature_parameters.values()),
        },
        "coefficient_radial_high_energy": {
            "value": high_energy_fraction,
            "maximum": 1.0e-10,
            "passed": high_energy_fraction <= 1.0e-10,
        },
        "causal_through_k1": {
            "value": causal_max,
            "maximum": 0.0,
            "passed": causal_max == 0.0,
        },
        "top_boundary_zero": {
            "value": top_row_max,
            "maximum": 2.0e-12,
            "passed": top_row_max <= 2.0e-12,
        },
        "future_correction_nonzero": {
            "value": future_energy,
            "passed": math.isfinite(future_energy) and future_energy > 0.0,
        },
        "finite": {
            "value": bool(torch.isfinite(corrected).all()),
            "passed": bool(torch.isfinite(corrected).all()),
        },
    }
    passed = all(value["passed"] for value in gates.values())
    terminal = {
        "schema": "r16_dscp_v22_guarded_candidate_selftest_v1",
        "status": "passed" if passed else "failed",
        "candidate": "r16_dscp_v22_wide128_radial_guard",
        "bindings": {**observed, "spec_sha256": sha256_file(SPEC)},
        "gates": gates,
        "route": {
            "name": decisions[0].name,
            "condition": float(conditions[0]),
            "abstain": bool(decisions[0].abstain),
        },
        "truth_reads": 0,
        "checkpoint_writes": 0,
        "resources": {
            "wall_s": time.monotonic() - started,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "completed_utc": utc_now(),
    }
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(terminal, OUT / "selftest.json")
    if not passed:
        raise RuntimeError("guarded candidate self-test gate failed")
    print(json.dumps(terminal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
