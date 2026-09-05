from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hashlib
import json

import h5py
import pytest
import torch
import yaml

from saved_time_phase_operator_v4.instance_adaptation.defect_correction import (
    CausalErrorBasisGenerator,
    DefectCorrectionWeights,
    FactorizedCausalBasis,
    causal_observation_probe_basis,
    causal_smoothstep_envelope,
    meta_defect_correction_loss,
    sampled_basis_defect_design,
    sampled_basis_defect_design_sparse,
    solve_causal_defect_correction,
    solve_weighted_ridge,
)
from saved_time_phase_operator_v4.instance_adaptation.cpadc_contract import (
    cpadc_implementation_digests,
    validate_dataset_cpml_contract,
)
from saved_time_phase_operator_v4.instance_adaptation.forced_defect import (
    SavedGridCPMLConfig,
    _validate_uniform_time_spacing,
    lwc84_discrete_defect,
    lwc84_source_terms,
    lwc84_step,
)
from saved_time_phase_operator_v4.instance_adaptation.losses import (
    build_fixed_physics_points,
    sample_fixed_physics_residual,
)
from scripts.run_causal_defect_adaptation import (
    _load_basis,
    _promotion_gate,
    build_complete_evaluation_manifest,
    resolve_basis_training_per_family,
)
from scripts.run_v5_instance_adaptation import validate_external_evaluation_contract
from scripts.train_causal_defect_basis import (
    _checkpoint_payload,
    _cpadc_adamw_parameter_groups,
    _cpadc_learning_rate_for_epoch,
    _cpml_physics_window,
    _first_order_projected_coefficients,
    _weighted_cpml_defect_loss,
)
from scripts.calibrate_causal_defect_risk import (
    select_family_strength_thresholds,
    select_strength_threshold,
)
from scripts.train_v5_residual_meta import _sha256


def _forced_case(*, dtype=torch.float64):
    torch.manual_seed(31)
    records, times, height, width = 1, 9, 25, 25
    dt, dx, dz = 1.0e-4, 10.0, 10.0
    velocity = torch.full((records, height, width), 900.0, dtype=dtype)
    source_parameters = torch.tensor(
        [[120.0, 120.0, 18.0, 2.0e-4, 2.5]], dtype=dtype
    )
    source_map = torch.zeros((records, 1, height, width), dtype=dtype)
    source_map[:, :, height // 2, width // 2] = 1.0
    time_s = torch.arange(times, dtype=dtype) * dt
    source, source_tt = lwc84_source_terms(
        source_parameters,
        source_map,
        time_s,
        dx_m=dx,
        dz_m=dz,
        dtype=dtype,
    )
    previous = torch.zeros((records, height, width), dtype=dtype)
    current = torch.zeros_like(previous)
    frames = [previous, current]
    for center in range(1, times - 1):
        following = lwc84_step(
            previous,
            current,
            velocity,
            source=source[:, center],
            source_tt=source_tt[:, center],
            dt_s=dt,
            dx_m=dx,
            dz_m=dz,
        )
        frames.append(following)
        previous, current = current, following
    return {
        "field": torch.stack(frames, dim=1),
        "velocity": velocity[:, None],
        "source_parameters": source_parameters,
        "source_map": source_map,
        "time_s": time_s,
        "dt": dt,
        "dx": dx,
        "dz": dz,
    }


def test_forced_defect_matches_the_complete_lwc84_recurrence():
    case = _forced_case()
    forced = lwc84_discrete_defect(
        case["field"],
        case["velocity"],
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        source_parameters=case["source_parameters"],
        source_map=case["source_map"],
        time_s=case["time_s"],
        normalize=False,
    )
    homogeneous = lwc84_discrete_defect(
        case["field"],
        case["velocity"],
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        normalize=False,
    )
    assert forced.shape[-2:] == (case["field"].shape[-2] - 16, case["field"].shape[-1] - 16)
    assert float(forced.abs().max()) < 1.0e-9
    assert float(homogeneous.abs().max()) > 1.0e-4


def test_forced_defect_respects_pressure_normalization_and_affine_linearity():
    case = _forced_case()
    physical_scale = torch.tensor([7.25], dtype=case["field"].dtype)
    normalized = case["field"] / physical_scale[:, None, None, None]
    normalized_defect = lwc84_discrete_defect(
        normalized,
        case["velocity"],
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        source_parameters=case["source_parameters"],
        source_map=case["source_map"],
        time_s=case["time_s"],
        field_scale_pa=physical_scale,
        normalize=False,
    )
    assert float(normalized_defect.abs().max()) < 1.0e-9

    correction = torch.randn_like(case["field"]) * 1.0e-7
    forced_corrected = lwc84_discrete_defect(
        case["field"] + correction,
        case["velocity"],
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        source_parameters=case["source_parameters"],
        source_map=case["source_map"],
        time_s=case["time_s"],
        normalize=False,
    )
    correction_defect = lwc84_discrete_defect(
        correction,
        case["velocity"],
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        normalize=False,
    )
    torch.testing.assert_close(
        forced_corrected,
        correction_defect,
        atol=2.0e-9,
        rtol=2.0e-9,
    )


def test_cpml_defect_is_finite_linear_and_uses_full_physical_grid():
    torch.manual_seed(912)
    records, times, height, width = 1, 7, 25, 27
    field = torch.randn(records, times, height, width, dtype=torch.float64) * 1.0e-7
    field[:, :, 0] = 0.0
    correction = torch.randn_like(field) * 1.0e-8
    correction[:, :, 0] = 0.0
    velocity = torch.full((records, 1, height, width), 1500.0, dtype=torch.float64)
    cpml = SavedGridCPMLConfig(npml=8)
    first = lwc84_discrete_defect(
        field,
        velocity,
        dt=0.0025,
        dx=10.0,
        dz=10.0,
        observed_indices=(0, 1),
        normalize=False,
        cpml_config=cpml,
    )
    combined = lwc84_discrete_defect(
        field + correction,
        velocity,
        dt=0.0025,
        dx=10.0,
        dz=10.0,
        observed_indices=(0, 1),
        normalize=False,
        cpml_config=cpml,
    )
    correction_only = lwc84_discrete_defect(
        correction,
        velocity,
        dt=0.0025,
        dx=10.0,
        dz=10.0,
        observed_indices=(0, 1),
        normalize=False,
        cpml_config=cpml,
    )
    assert first.shape == (records, times - 3, height, width)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(
        combined - first, correction_only, rtol=2.0e-8, atol=2.0e-8
    )


def test_cpml_defect_rejects_saved_internal_time_mismatch():
    field = torch.zeros(1, 5, 25, 25)
    velocity = torch.full((1, 1, 25, 25), 1500.0)
    with pytest.raises(ValueError, match="must equal dt"):
        lwc84_discrete_defect(
            field,
            velocity,
            dt=0.0025,
            dx=10.0,
            dz=10.0,
            observed_indices=(0, 1),
            cpml_config=SavedGridCPMLConfig(
                npml=8,
                internal_dt_s=1.0e-4,
                internal_substeps_per_saved_frame=20,
            ),
        )


def test_saved_time_spacing_accepts_float32_roundoff_but_rejects_real_jitter():
    dt = 0.0025
    axis = (torch.arange(401, dtype=torch.float32) * dt).unsqueeze(0)
    deviation = float(((axis[:, 1:] - axis[:, :-1]) - dt).abs().max())
    assert deviation > dt * 1.0e-5
    _validate_uniform_time_spacing(axis, dt=dt)

    jittered = axis.clone()
    jittered[:, 200] += 1.0e-3
    with pytest.raises(ValueError, match="uniformly spaced"):
        _validate_uniform_time_spacing(jittered, dt=dt)


def test_cpml_physics_window_covers_low_frequency_peak_and_side_arrival():
    time_s = torch.arange(401, dtype=torch.float64) * 0.0025
    source = torch.tensor([[1000.0, 50.0, 8.0, 0.1875, 1.0]])
    velocity = torch.full((1, 1, 201, 201), 5000.0)
    start, stop = _cpml_physics_window(
        time_s,
        source,
        velocity,
        (0, 1),
        dx_m=10.0,
        residual_count=64,
        prearrival_frames=8,
    )
    expected_arrival = 0.1875 + 1000.0 / 5000.0
    arrival_index = int(torch.searchsorted(time_s, expected_arrival))
    assert start == arrival_index - 8
    assert stop - start - 1 == 64
    assert float(time_s[start]) < expected_arrival < float(time_s[stop - 1])


def test_cpml_boundary_weighted_loss_emphasizes_three_absorbing_sides():
    interior = torch.zeros(1, 3, 25, 25)
    boundary = torch.zeros_like(interior)
    interior[..., 12, 12] = 1.0
    boundary[..., 12, 0] = 1.0
    plain_interior = _weighted_cpml_defect_loss(
        interior, boundary_band_cells=4, boundary_weight=1.0
    )
    plain_boundary = _weighted_cpml_defect_loss(
        boundary, boundary_band_cells=4, boundary_weight=1.0
    )
    weighted_interior = _weighted_cpml_defect_loss(
        interior, boundary_band_cells=4, boundary_weight=4.0
    )
    weighted_boundary = _weighted_cpml_defect_loss(
        boundary, boundary_band_cells=4, boundary_weight=4.0
    )
    torch.testing.assert_close(plain_boundary, plain_interior)
    assert float(weighted_boundary) > float(weighted_interior)


def test_cpml_gradient_prefix_is_explicitly_truncated():
    torch.manual_seed(914)
    field = torch.randn(1, 9, 25, 25, dtype=torch.float64, requires_grad=True)
    velocity = torch.full((1, 1, 25, 25), 1500.0, dtype=torch.float64)
    defect = lwc84_discrete_defect(
        field,
        velocity,
        dt=0.0025,
        dx=10.0,
        dz=10.0,
        observed_indices=(-1, 4),
        normalize=False,
        cpml_config=SavedGridCPMLConfig(npml=8),
        cpml_gradient_start_index=3,
    )
    defect.square().mean().backward()
    assert field.grad is not None and torch.isfinite(field.grad).all()
    assert torch.count_nonzero(field.grad[:, :3]) == 0
    assert torch.count_nonzero(field.grad[:, 3:]) > 0


_REGISTERED_MARMOUSI_DATASET = Path(
    "/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)


@pytest.mark.skipif(
    not _REGISTERED_MARMOUSI_DATASET.is_file(),
    reason="registered Marmousi LWC84 dataset is not mounted",
)
def test_cpadc_cpml_is_bound_to_exact_teacher_numerics():
    config = yaml.safe_load(
        Path(
            "configs/saved_time_v5/"
            "causal_defect_basis_marmousi1_4m_v2_target10_rank32.yaml"
        ).read_text()
    )
    cpml = dict(config["pde_cpml"])
    cpml.pop("enabled")
    contract = validate_dataset_cpml_contract(
        _REGISTERED_MARMOUSI_DATASET,
        saved_grid_cpml=cpml,
        saved_dt_s=float(config["dt_s"]),
        saved_dx_m=float(config["dx_m"]),
        saved_dz_m=float(config["dz_m"]),
    )
    assert contract["fine_cpml_cells"] == 40
    assert contract["saved_cpml_cells"] == 20
    assert contract["cpml_physical_thickness_m"] == 200.0
    assert contract["internal_substeps_per_saved_frame"] == 20
    assert contract["absorbing_boundaries"] == ["left", "right", "bottom"]
    assert contract["top_boundary"] == "free_surface_dirichlet"
    assert contract["exact_fine_state_reconstruction"] is False
    assert contract["exact_fine_source_restriction"] is False
    assert contract["saved_source_contract"] == (
        "direct_saved_grid_bilinear_unit_mass_reinjection"
    )

    invalid = dict(cpml)
    invalid["npml"] = 19
    with pytest.raises(ValueError, match="physical x thickness"):
        validate_dataset_cpml_contract(
            _REGISTERED_MARMOUSI_DATASET,
            saved_grid_cpml=invalid,
            saved_dt_s=float(config["dt_s"]),
            saved_dx_m=float(config["dx_m"]),
            saved_dz_m=float(config["dz_m"]),
        )


def test_cpadc_implementation_identity_covers_training_and_deployment():
    digests = cpadc_implementation_digests(Path(__file__).resolve().parents[2])
    assert "scripts/train_causal_defect_basis.py" in digests
    assert "scripts/run_causal_defect_adaptation.py" in digests
    assert "src/fno_acoustic/data_generation/cpml.py" in digests
    assert all(len(value) == 64 for value in digests.values())


def test_generator_and_cpu_solve_accept_full_grid_cpml_defect():
    torch.manual_seed(1912)
    records, times, height, width = 1, 9, 25, 25
    dt = 0.0025
    parent = torch.randn(records, times, height, width) * 1.0e-4
    parent[:, :, 0] = 0.0
    velocity = torch.full((records, 1, height, width), 1500.0)
    observed = parent[:, :2].clone()
    source_parameters = torch.tensor([[120.0, 80.0, 18.0, 0.05, 1.0]])
    source_map = torch.zeros(records, 1, height, width)
    source_map[:, :, 8, 12] = 1.0
    time_s = torch.arange(times, dtype=parent.dtype) * dt
    cpml = SavedGridCPMLConfig(npml=8)
    parent_defect, scale = lwc84_discrete_defect(
        parent,
        velocity,
        dt=dt,
        dx=10.0,
        dz=10.0,
        observed_indices=(0, 1),
        source_parameters=source_parameters,
        source_map=source_map,
        time_s=time_s,
        normalize=False,
        return_scale=True,
        cpml_config=cpml,
    )
    generator = CausalErrorBasisGenerator(
        rank=2, phase_rank=0, width=8, ramp_steps=3
    )
    basis = generator(
        parent,
        velocity,
        observed,
        source_parameters,
        time_s,
        (0, 1),
        parent_defect=parent_defect,
        defect_scale=scale,
    )
    points = build_fixed_physics_points(times, (0, 1), count=32, seed=28)
    result = solve_causal_defect_correction(
        basis,
        parent,
        velocity,
        source_parameters,
        source_map,
        time_s,
        (0, 1),
        points,
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-3
        ),
        dt=dt,
        dx=10.0,
        dz=10.0,
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=1.0,
        rank_chunk_size=2,
        cpml_config=cpml,
        solve_device="cpu",
    )
    assert result.field.shape == parent.shape
    assert result.coefficient_solve_device == "cpu"
    assert result.coefficient_objective_device == "cpu"
    assert result.correction_materialization_device == "cpu"
    assert torch.isfinite(result.coefficients).all()


def _sealed_report(
    family,
    parent,
    adapted,
    *,
    elapsed=0.2,
    total_elapsed=None,
    synchronized=False,
    accepted=True,
):
    return {
        "medium_type": family,
        "parent_future_fullfield_relative_l2": parent,
        "future_fullfield_relative_l2": adapted,
        "future_truth_squared_norm": 1.0,
        "future_parent_squared_error": parent**2,
        "future_adapted_squared_error": adapted**2,
        "all_saved_time_indices": 401,
        "adaptation": {
            "adaptation_elapsed_s": elapsed,
            "total_inference_elapsed_s": (
                elapsed if total_elapsed is None else total_elapsed
            ),
            "cuda_synchronized_timing": synchronized,
            "future_truth_used": False,
            "accepted": accepted,
            "coefficient_finetune_cpu_only": True,
            "coefficient_finetune_device": "cpu",
            "coefficient_objective_device": "cpu",
            "correction_materialization_device": "cpu",
            "synthetic_bridge_device": "cpu",
            "instance_trainable_path_cpu_only": True,
        },
    }


def test_promotion_gate_requires_full_protocol_and_family_safe_improvement():
    reports = [
        _sealed_report("uniform", 0.20, 0.18),
        _sealed_report("layered", 0.30, 0.27),
        _sealed_report("marmousi", 0.60, 0.54),
    ]
    gate = _promotion_gate(
        reports,
        {
            "minimum_blind_mean_improvement": 0.05,
            "minimum_nonworse_fraction": 1.0,
            "maximum_runtime_s": 1.0,
            "require_every_family_nonworse": True,
        },
        all_validation=True,
    )
    assert gate["passed"] is True
    assert gate["checks"]["family_safe"] is True
    sampled_gate = _promotion_gate(reports, {}, all_validation=False)
    assert sampled_gate["passed"] is False
    assert sampled_gate["checks"]["full_validation_protocol"] is False


def test_promotion_gate_rejects_relative_gain_above_absolute_accuracy_target():
    reports = [
        _sealed_report("uniform", 0.20, 0.18),
        _sealed_report("layered", 0.30, 0.27),
        _sealed_report("marmousi", 0.60, 0.54),
    ]
    gate = _promotion_gate(
        reports,
        {
            "minimum_blind_mean_improvement": 0.05,
            "minimum_nonworse_fraction": 1.0,
            "maximum_runtime_s": 1.0,
            "require_every_family_nonworse": True,
            "maximum_mean_adapted_future_relative_l2": 0.10,
            "maximum_family_mean_adapted_future_relative_l2": 0.10,
        },
        all_validation=True,
    )
    assert gate["mean_relative_improvement"] > 0.05
    assert gate["passed"] is False
    assert gate["checks"]["absolute_mean_accuracy"] is False
    assert gate["checks"]["absolute_family_accuracy"] is False


def test_promotion_gate_accepts_same_protocol_absolute_accuracy_target():
    reports = [
        _sealed_report("uniform", 0.12, 0.08),
        _sealed_report("layered", 0.14, 0.09),
        _sealed_report("marmousi", 0.16, 0.095),
    ]
    gate = _promotion_gate(
        reports,
        {
            "minimum_blind_mean_improvement": 0.05,
            "minimum_nonworse_fraction": 1.0,
            "maximum_runtime_s": 1.0,
            "require_every_family_nonworse": True,
            "maximum_mean_adapted_future_relative_l2": 0.10,
            "maximum_family_mean_adapted_future_relative_l2": 0.10,
        },
        all_validation=True,
    )
    assert gate["passed"] is True
    assert gate["mean_adapted_future_relative_l2"] < 0.10
    assert gate["checks"]["absolute_mean_accuracy"] is True
    assert gate["checks"]["absolute_family_accuracy"] is True


def test_promotion_gate_requires_synchronized_p95_end_to_end_ten_x_speedup():
    reports = [
        _sealed_report(
            "uniform", 0.12, 0.08, elapsed=0.20, total_elapsed=0.80,
            synchronized=True,
        ),
        _sealed_report(
            "layered", 0.14, 0.09, elapsed=0.25, total_elapsed=0.90,
            synchronized=True,
        ),
        _sealed_report(
            "marmousi", 0.16, 0.095, elapsed=0.30, total_elapsed=1.01,
            synchronized=True,
        ),
    ]
    settings = {
        "minimum_blind_mean_improvement": 0.05,
        "minimum_nonworse_fraction": 1.0,
        "maximum_runtime_s": 1.0,
        "maximum_p95_adaptation_runtime_s": 0.75,
        "require_every_family_nonworse": True,
        "maximum_mean_adapted_future_relative_l2": 0.10,
        "maximum_family_mean_adapted_future_relative_l2": 0.10,
        "minimum_end_to_end_speedup_vs_traditional": 10.0,
        "traditional_solver_reference_runtime_s": 10.0,
    }
    gate = _promotion_gate(reports, settings, all_validation=True)
    assert gate["checks"]["end_to_end_speedup_mean"] is True
    assert gate["checks"]["end_to_end_speedup_p95"] is False
    assert gate["p95_end_to_end_speedup_vs_traditional"] < 10.0
    assert gate["passed"] is False

    reports[-1]["adaptation"]["total_inference_elapsed_s"] = 0.99
    gate = _promotion_gate(reports, settings, all_validation=True)
    assert gate["checks"]["end_to_end_speedup_p95"] is True
    assert gate["passed"] is True

    reports[-1]["adaptation"]["cuda_synchronized_timing"] = False
    gate = _promotion_gate(reports, settings, all_validation=True)
    assert gate["checks"]["runtime_measurement_synchronized"] is False
    assert gate["passed"] is False


@pytest.mark.parametrize(
    "key",
    (
        "maximum_mean_adapted_future_relative_l2",
        "maximum_family_mean_adapted_future_relative_l2",
    ),
)
def test_promotion_gate_rejects_invalid_absolute_accuracy_target(key):
    reports = [_sealed_report("uniform", 0.12, 0.08)]
    with pytest.raises(ValueError, match="must be positive"):
        _promotion_gate(reports, {key: float("nan")}, all_validation=True)


def test_cpadc_adamw_excludes_bias_and_norm_vectors_from_decay():
    module = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )
    groups = _cpadc_adamw_parameter_groups(
        module.named_parameters(), weight_decay=1.0e-5
    )
    by_name = {str(group["group_name"]): group for group in groups}
    assert by_name["decay"]["weight_decay"] == pytest.approx(1.0e-5)
    assert by_name["no_decay"]["weight_decay"] == 0.0
    assert len(by_name["decay"]["params"]) == 1
    assert len(by_name["no_decay"]["params"]) == 3


def test_cpadc_adamw_scales_only_the_trust_controller_learning_rate():
    generator = CausalErrorBasisGenerator(rank=8, phase_rank=2, width=16)
    groups = _cpadc_adamw_parameter_groups(
        generator.named_parameters(),
        weight_decay=1.0e-6,
        learning_rate=5.0e-6,
        trust_learning_rate_scale=100.0,
    )
    by_name = {str(group["group_name"]): group for group in groups}
    assert by_name["decay"]["lr"] == pytest.approx(5.0e-6)
    assert by_name["no_decay"]["lr"] == pytest.approx(5.0e-6)
    assert by_name["trust_decay"]["lr"] == pytest.approx(5.0e-4)
    assert by_name["trust_no_decay"]["lr"] == pytest.approx(5.0e-4)


def test_cpadc_learning_rate_warms_then_cosine_decays():
    values = [
        _cpadc_learning_rate_for_epoch(
            epoch,
            total_epochs=8,
            maximum_learning_rate=1.0e-4,
            minimum_learning_rate=1.0e-5,
            warmup_epochs=2,
        )
        for epoch in range(1, 9)
    ]
    assert values[0] < values[1]
    assert values[1] == pytest.approx(1.0e-4)
    assert values[2] == pytest.approx(1.0e-4)
    assert values[-1] == pytest.approx(1.0e-5)
    assert all(left >= right for left, right in zip(values[2:], values[3:]))


def test_cpadc_one_epoch_feasibility_schedule_requires_zero_warmup():
    assert _cpadc_learning_rate_for_epoch(
        1,
        total_epochs=1,
        maximum_learning_rate=5.0e-5,
        minimum_learning_rate=5.0e-6,
        warmup_epochs=0,
    ) == pytest.approx(5.0e-5)
    with pytest.raises(ValueError, match="warmup must be shorter"):
        _cpadc_learning_rate_for_epoch(
            1,
            total_epochs=1,
            maximum_learning_rate=5.0e-5,
            minimum_learning_rate=5.0e-6,
            warmup_epochs=1,
        )


def test_complete_cpadc_evaluation_manifest_supports_frozen_test_id():
    from types import SimpleNamespace

    records = tuple(
        SimpleNamespace(
            sample_id=f"{split}-{family}", split=split, medium_type=family
        )
        for split in ("validation", "test_id")
        for family in ("uniform", "layered", "marmousi")
    )
    manifest = SimpleNamespace(records=records)
    selected = build_complete_evaluation_manifest(manifest, split="test_id")
    assert tuple(record.sample_id for record in selected) == (
        "test_id-uniform",
        "test_id-layered",
        "test_id-marmousi",
    )
    with pytest.raises(ValueError, match="validation or test_id"):
        build_complete_evaluation_manifest(manifest, split="train")


def test_trainonly_evaluation_uses_effective_basis_episode_override():
    config = {"cpadc": {"per_family": 64}}
    assert resolve_basis_training_per_family(config, None) == 64
    assert resolve_basis_training_per_family(config, 4) == 4
    with pytest.raises(ValueError, match="must be positive"):
        resolve_basis_training_per_family(config, 0)


def test_external_evaluation_contract_is_hash_bound_and_disjoint(tmp_path):
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record(sample_id, group_id, sample_hash):
        return SimpleNamespace(
            sample_id=sample_id,
            group_id=group_id,
            sample_sha256=sample_hash,
        )

    candidate = tmp_path / "candidate.h5"
    candidate_shard = tmp_path / "candidate-shard.h5"
    candidate_shard.write_bytes(b"registered-after-generation")
    candidate_sidecar = tmp_path / "candidate-shard.h5.sha256"
    candidate_sidecar.write_text(sha256(candidate_shard) + "\n")
    with h5py.File(candidate, "w") as handle:
        handle.attrs["vds_source_shards"] = json.dumps(
            [str(candidate_shard.resolve())]
        )
    training = SimpleNamespace(
        digest="train-digest",
        source_path=str(tmp_path / "training.h5"),
        time_s=(0.0, 0.0025),
        x_m=(0.0, 10.0),
        z_m=(0.0, 10.0),
        records=(record("train-a", "train-g", "a" * 64),),
    )
    evaluation = SimpleNamespace(
        digest="validation-digest",
        source_path=str(candidate),
        time_s=training.time_s,
        x_m=training.x_m,
        z_m=training.z_m,
        records=(record("validation-a", "validation-g", "b" * 64),),
    )
    audit_path = tmp_path / "sample_hash_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "intersection_count": 0,
                "future_truth_opened_by_evaluator": False,
                "candidate_dataset": str(candidate.resolve()),
            }
        )
    )
    pretruth_path = tmp_path / "pretruth.json"
    pretruth_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "intersection_counts": {
                    "sample_id": 0,
                    "group_id": 0,
                    "semantic_group_sha256": 0,
                    "problem_sha256": 0,
                },
            }
        )
    )
    internal_path = tmp_path / "internal.json"
    internal_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "passed": True,
                "pairwise_intersection_counts": {
                    "validation__vs__test_id": {
                        "sample_id": 0,
                        "group_id": 0,
                        "semantic_group_sha256": 0,
                        "problem_sha256": 0,
                    }
                },
            }
        )
    )
    travel_path = tmp_path / "travel.h5"
    with h5py.File(travel_path, "w") as handle:
        handle.attrs["source_h5"] = str(candidate.resolve())
        handle.attrs["content_sha256"] = "c" * 64
        handle.attrs["schema"] = "source_family_adaptive_travel_v1"
    config = {
        "travel_time_h5": str(travel_path),
        "external_evaluation_contract": {
            "role": "frozen_validation",
            "training_manifest_digest": training.digest,
            "evaluation_manifest_digest": evaluation.digest,
            "evaluation_only": True,
            "training_forbidden": True,
            "evaluation_dataset": str(candidate.resolve()),
            "evaluation_vds_shards": [
                {
                    "path": str(candidate_shard.resolve()),
                    "byte_count": candidate_shard.stat().st_size,
                    "sha256": sha256(candidate_shard),
                    "sidecar": str(candidate_sidecar.resolve()),
                    "sidecar_sha256": sha256(candidate_sidecar),
                }
            ],
            "postgeneration_sample_sha256_audit": str(audit_path),
            "postgeneration_sample_sha256_audit_sha256": sha256(audit_path),
            "pretruth_historical_overlap_audit": str(pretruth_path),
            "pretruth_historical_overlap_audit_sha256": sha256(pretruth_path),
            "internal_split_overlap_audit": str(internal_path),
            "internal_split_overlap_audit_sha256": sha256(internal_path),
            "travel_time_h5": str(travel_path),
            "travel_time_h5_sha256": sha256(travel_path),
            "travel_time_content_sha256": "c" * 64,
            "travel_time_schema": "source_family_adaptive_travel_v1",
        }
    }

    report = validate_external_evaluation_contract(config, training, evaluation)

    assert report["external"] is True
    assert report["role"] == "frozen_validation"

    evaluation.records = (record("train-a", "validation-g", "b" * 64),)
    with pytest.raises(ValueError, match="sample_id overlaps"):
        validate_external_evaluation_contract(config, training, evaluation)

    evaluation.records = (record("validation-a", "validation-g", "b" * 64),)
    audit_path.write_text(audit_path.read_text() + "\n")
    with pytest.raises(ValueError, match="sample-hash audit hash binding"):
        validate_external_evaluation_contract(config, training, evaluation)

    audit_path.write_text(audit_path.read_text().rstrip() + "\n")
    config["external_evaluation_contract"][
        "postgeneration_sample_sha256_audit_sha256"
    ] = sha256(audit_path)
    candidate_shard.write_bytes(b"x" * candidate_shard.stat().st_size)
    with pytest.raises(ValueError, match="shard content binding"):
        validate_external_evaluation_contract(config, training, evaluation)


def test_basis_checkpoint_is_bound_to_parent_manifest_and_defect_contract(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"immutable-parent")
    generator = CausalErrorBasisGenerator(rank=4, phase_rank=1, width=8)
    payload = {
        "schema": "causal_physics_aligned_defect_correction_v1",
        "schema_version": 4,
        "basis_state": generator.state_dict(),
        "basis_rank": 4,
        "phase_rank": 1,
        "basis_width": 8,
        "causal_ramp_steps": 4,
        "manifest_digest": "manifest-test",
        "parent_checkpoint": str(parent.resolve()),
        "parent_checkpoint_sha256": _sha256(parent),
        "defect_contract": {
            "name": "source_consistent_effective_saved_grid_lwc84_v1",
            "spatial_halo_cells": 8,
            "exact_fine_generator_residual": False,
        },
        "online_solve_contract": {
            "name": "ridge_direction_learned_energy_ball_projection_v1",
        },
    }
    checkpoint = tmp_path / "basis.pt"
    torch.save(payload, checkpoint)
    loaded, info = _load_basis(
        checkpoint,
        parent_checkpoint=parent,
        manifest_digest="manifest-test",
        device=torch.device("cpu"),
    )
    assert loaded.rank == 4
    assert info["defect_contract"]["spatial_halo_cells"] == 8
    payload["schema_version"] = 5
    payload["risk_calibration"] = {
        "future_truth_scope": "disjoint_train_split_only",
        "record_count": 12,
    }
    payload["online_solve_contract"] = {
        "name": "ridge_direction_family_calibrated_strength_abstention_v1",
        "minimum_unconstrained_correction_ratio_by_family": {
            "uniform": 1.0,
            "layered": 0.5,
            "marmousi": 2.0,
        },
    }
    torch.save(payload, checkpoint)
    _, family_info = _load_basis(
        checkpoint,
        parent_checkpoint=parent,
        manifest_digest="manifest-test",
        device=torch.device("cpu"),
    )
    assert family_info[
        "minimum_unconstrained_correction_ratio_by_family"
    ] == pytest.approx({"uniform": 1.0, "layered": 0.5, "marmousi": 2.0})
    payload["defect_contract"] = {}
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="defect contract"):
        _load_basis(
            checkpoint,
            parent_checkpoint=parent,
            manifest_digest="manifest-test",
            device=torch.device("cpu"),
        )


def test_schema10_checkpoint_requires_dataset_and_code_identities(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"schema10-parent")
    generator = CausalErrorBasisGenerator(rank=4, phase_rank=1, width=8)
    cpml = SavedGridCPMLConfig(npml=8)
    config = {
        "pde_cpml": cpml.as_dict(),
        "online_inner_weights": {
            "defect": 0.0,
            "observed": 1.0,
            "bridge": 0.5,
        },
        "outer_weights": {
            "physics": 0.05,
            "physics_time_count": 64,
            "physics_window_strategy": (
                "nearest_side_cpml_peak_arrival_causal_prefix_v1"
            ),
            "physics_prearrival_frames": 8,
            "physics_boundary_band_cells": 12,
            "physics_boundary_weight": 4.0,
        },
        "cpadc": {"optimizer": "muon_adamw"},
    }
    numerics = {
        "schema": "cpadc_dataset_numerical_contract_v1",
        "config": "frozen",
    }
    implementation = {"forced_defect.py": "a" * 64}
    payload = _checkpoint_payload(
        generator,
        config=config,
        manifest_digest="schema10-manifest",
        parent_checkpoint=parent,
        parent_checkpoint_sha256=_sha256(parent),
        dataset_numerical_contract=numerics,
        implementation_digests=implementation,
        epoch=1,
        event={"status": "unit"},
    )
    checkpoint = tmp_path / "basis-schema10.pt"
    torch.save(payload, checkpoint)
    loaded, info = _load_basis(
        checkpoint,
        parent_checkpoint=parent,
        manifest_digest="schema10-manifest",
        device=torch.device("cpu"),
        dataset_numerical_contract=numerics,
        implementation_digests=implementation,
    )
    assert loaded.rank == 4
    assert info["schema_version"] == 10
    assert info["dataset_numerical_contract"] == numerics
    assert info["implementation_digests"] == implementation
    assert info["online_solve_contract"]["name"] == (
        "cpu_causal_observation_bridge_ridge_v5"
    )

    with pytest.raises(ValueError, match="implementation digest"):
        _load_basis(
            checkpoint,
            parent_checkpoint=parent,
            manifest_digest="schema10-manifest",
            device=torch.device("cpu"),
            dataset_numerical_contract=numerics,
            implementation_digests={"forced_defect.py": "b" * 64},
        )


def test_schema10_sparse_online_defect_contract_is_explicit(tmp_path):
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"schema10-sparse-parent")
    generator = CausalErrorBasisGenerator(rank=4, phase_rank=1, width=8)
    cpml = SavedGridCPMLConfig(npml=8)
    config = {
        "pde_cpml": cpml.as_dict(),
        "online_inner_weights": {
            "defect": 1.0,
            "observed": 0.25,
            "bridge": 0.5,
        },
        "online_defect_design": "sparse_interior",
        "online_defect_time_order": 2,
        "online_physics_point_count": 512,
        "online_physics_sampling": "rad",
        "outer_weights": {
            "physics": 0.05,
            "physics_time_count": 64,
            "physics_window_strategy": (
                "nearest_side_cpml_peak_arrival_causal_prefix_v1"
            ),
            "physics_prearrival_frames": 8,
            "physics_boundary_band_cells": 12,
            "physics_boundary_weight": 4.0,
        },
        "cpadc": {"optimizer": "muon_adamw"},
    }
    numerics = {"schema": "cpadc_dataset_numerical_contract_v1"}
    implementation = {"forced_defect.py": "c" * 64}
    payload = _checkpoint_payload(
        generator,
        config=config,
        manifest_digest="schema10-sparse-manifest",
        parent_checkpoint=parent,
        parent_checkpoint_sha256=_sha256(parent),
        dataset_numerical_contract=numerics,
        implementation_digests=implementation,
        epoch=1,
        event={"status": "unit"},
    )
    contract = payload["online_solve_contract"]
    assert contract["name"] == (
        "cpu_causal_sparse_defect_observation_bridge_ridge_v6"
    )
    assert contract["online_defect_design"] == "sparse_interior"
    assert contract["online_defect_time_order"] == 2
    assert contract["online_defect_test_function_width"] == 1
    assert contract["online_defect_test_function_normalization"] == (
        "discrete_l2_unit"
    )
    assert contract["online_physics_point_count"] == 512
    assert contract["online_physics_sampling"] == "rad"
    assert contract["online_rad_k"] == 1.0
    assert contract["online_rad_c"] == 1.0
    assert contract["online_rad_time_tilt"] == 1.5
    assert contract["online_physics_seed"] == 372
    assert contract["online_prior_weight"] == 1.0e-4
    assert contract["online_spatial_stride"] == 4
    checkpoint = tmp_path / "basis-schema10-sparse.pt"
    torch.save(payload, checkpoint)
    loaded, info = _load_basis(
        checkpoint,
        parent_checkpoint=parent,
        manifest_digest="schema10-sparse-manifest",
        device=torch.device("cpu"),
        dataset_numerical_contract=numerics,
        implementation_digests=implementation,
    )
    assert loaded.rank == 4
    assert info["online_solve_contract"] == contract

    weak_config = dict(config)
    weak_config["online_defect_test_function_width"] = 3
    weak_payload = _checkpoint_payload(
        generator,
        config=weak_config,
        manifest_digest="schema10-weak-sparse-manifest",
        parent_checkpoint=parent,
        parent_checkpoint_sha256=_sha256(parent),
        dataset_numerical_contract=numerics,
        implementation_digests=implementation,
        epoch=1,
        event={"status": "unit"},
    )
    weak_contract = weak_payload["online_solve_contract"]
    assert weak_contract["name"] == (
        "cpu_causal_weak_sparse_defect_observation_bridge_ridge_v7"
    )
    assert weak_contract["online_defect_test_function_width"] == 3
    assert weak_contract["online_defect_test_function_normalization"] == (
        "discrete_l2_unit"
    )
    weak_checkpoint = tmp_path / "basis-schema10-weak-sparse.pt"
    torch.save(weak_payload, weak_checkpoint)
    weak_loaded, weak_info = _load_basis(
        weak_checkpoint,
        parent_checkpoint=parent,
        manifest_digest="schema10-weak-sparse-manifest",
        device=torch.device("cpu"),
        dataset_numerical_contract=numerics,
        implementation_digests=implementation,
    )
    assert weak_loaded.rank == 4
    assert weak_info["online_solve_contract"] == weak_contract


def _factorized_basis(case, *, rank=4, phase_rank=2):
    field = case["field"].float()
    records, times, height, width = field.shape
    envelope = causal_smoothstep_envelope(times, (0, 1), ramp_steps=3)
    phase = torch.zeros_like(field)
    phase[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
    return FactorizedCausalBasis(
        spatial_modes=torch.randn(records, rank, height, width),
        temporal_modes=torch.randn(records, rank, times),
        causal_envelope=envelope[None],
        field_scale=torch.full((records,), 1.0e-5),
        phase_reference=phase / 1.0e-5,
        trust_fraction=torch.ones(records),
        phase_rank=phase_rank,
    )


def test_factorized_basis_is_causal_and_combine_matches_materialization():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case)
    coefficients = torch.randn(1, basis.rank)
    materialized = basis.materialize()
    combined = torch.einsum("rk,rkthw->rthw", coefficients, materialized)
    torch.testing.assert_close(basis.combine(coefficients), combined)
    assert torch.count_nonzero(materialized[:, :, :2]) == 0


def test_observation_probe_exposes_design_modes_without_breaking_causal_output():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case)
    probe = causal_observation_probe_basis(basis)
    coefficients = torch.ones(1, basis.rank)
    assert torch.count_nonzero(basis.materialize(time_indices=(0, 1))) == 0
    assert torch.count_nonzero(probe.materialize(time_indices=(0, 1))) > 0
    assert torch.count_nonzero(basis.combine(coefficients)[:, :2]) == 0


def test_basis_generator_uses_only_causal_inputs_and_has_phase_modes():
    case = _forced_case(dtype=torch.float32)
    field = case["field"]
    defect, scale = lwc84_discrete_defect(
        field,
        case["velocity"],
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        source_parameters=case["source_parameters"],
        source_map=case["source_map"],
        time_s=case["time_s"],
        normalize=False,
        return_scale=True,
    )
    generator = CausalErrorBasisGenerator(rank=6, phase_rank=2, width=8)
    basis = generator(
        field,
        case["velocity"],
        field[:, :2],
        case["source_parameters"],
        case["time_s"],
        (0, 1),
        parent_defect=defect,
        defect_scale=scale,
    )
    assert basis.rank == 6 and basis.phase_rank == 2
    assert basis.trust_fraction.shape == (1,)
    assert bool(((basis.trust_fraction > 0.0) & (basis.trust_fraction <= 1.0)).all())
    assert torch.count_nonzero(basis.materialize()[:, :, :2]) == 0
    assert all(parameter.requires_grad for parameter in generator.parameters())


def test_chunked_basis_defect_design_matches_direct_materialization():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case, rank=3, phase_rank=1)
    points = build_fixed_physics_points(
        basis.time_count, (0, 1), count=23, seed=41
    )
    chunked = sampled_basis_defect_design(
        basis,
        case["velocity"],
        points,
        (0, 1),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        rank_chunk_size=2,
    )
    materialized = basis.materialize().reshape(
        basis.rank, basis.time_count, *basis.spatial_shape
    )
    direct_defect = lwc84_discrete_defect(
        materialized,
        case["velocity"].expand(basis.rank, -1, -1, -1),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        normalize=False,
    )
    direct = sample_fixed_physics_residual(
        direct_defect, points, (0, 1)
    ).transpose(0, 1)[None]
    torch.testing.assert_close(chunked, direct, atol=1.0e-4, rtol=1.0e-5)


def test_sparse_basis_defect_design_matches_second_order_materialization():
    case = _forced_case(dtype=torch.float64)
    basis = _factorized_basis(case, rank=3, phase_rank=1)
    points = build_fixed_physics_points(
        basis.time_count, (0, 1), count=23, seed=141
    )
    sparse = sampled_basis_defect_design_sparse(
        basis,
        case["velocity"],
        points,
        (0, 1),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
    )
    materialized = basis.materialize().reshape(
        basis.rank, basis.time_count, *basis.spatial_shape
    )
    direct_defect = lwc84_discrete_defect(
        materialized,
        case["velocity"].expand(basis.rank, -1, -1, -1),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        observed_indices=(0, 1),
        time_order=2,
        normalize=False,
    )
    direct = sample_fixed_physics_residual(
        direct_defect, points, (0, 1)
    ).transpose(0, 1)[None]
    torch.testing.assert_close(sparse, direct, atol=1.0e-8, rtol=1.0e-8)


def test_small_convex_solve_recovers_an_error_inside_the_basis():
    case = _forced_case(dtype=torch.float32)
    truth = case["field"]
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    true_coefficients = torch.tensor([[0.13, -0.08, 0.04]])
    parent = truth - basis.combine(true_coefficients)
    points = build_fixed_physics_points(
        truth.shape[1], (0, 1), count=128, seed=73
    )
    result = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        observed_wavefield=truth[:, :2],
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-10
        ),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=10.0,
        rank_chunk_size=2,
    )
    assert bool(result.accepted[0])
    assert result.future_truth_used is False
    assert result.coefficient_solve_device == "cpu"
    torch.testing.assert_close(
        result.coefficients,
        true_coefficients,
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    assert float((result.field - truth).norm() / truth.norm().clamp_min(1.0e-8)) < 1.0e-2


def test_sparse_physics_solve_recovers_an_error_inside_the_basis():
    case = _forced_case(dtype=torch.float64)
    truth = case["field"]
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    true_coefficients = torch.tensor(
        [[0.13, -0.08, 0.04]], dtype=truth.dtype
    )
    parent = truth - basis.combine(true_coefficients)
    points = build_fixed_physics_points(
        truth.shape[1], (0, 1), count=128, seed=173
    )
    result = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-10
        ),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        time_order=2,
        defect_design="sparse_interior",
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=10.0,
    )
    assert bool(result.accepted[0])
    torch.testing.assert_close(
        result.coefficients,
        true_coefficients,
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    assert float((result.field - truth).norm() / truth.norm().clamp_min(1.0e-8)) < 1.0e-2


def test_unit_width_defect_test_function_exactly_preserves_the_legacy_solve():
    case = _forced_case(dtype=torch.float64)
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    parent = case["field"] - basis.combine(
        torch.tensor([[0.13, -0.08, 0.04]], dtype=case["field"].dtype)
    )
    points = build_fixed_physics_points(
        case["field"].shape[1], (0, 1), count=64, seed=271
    )
    common = dict(
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-10
        ),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        time_order=2,
        defect_design="sparse_interior",
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=10.0,
    )
    legacy = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        **common,
    )
    explicit = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        defect_test_function_width=1,
        **common,
    )
    torch.testing.assert_close(explicit.coefficients, legacy.coefficients, rtol=0, atol=0)
    torch.testing.assert_close(explicit.field, legacy.field, rtol=0, atol=0)
    torch.testing.assert_close(
        explicit.objective_after, legacy.objective_after, rtol=0, atol=0
    )


def test_local_weak_defect_solve_recovers_an_error_without_future_truth():
    case = _forced_case(dtype=torch.float64)
    truth = case["field"]
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    true_coefficients = torch.tensor(
        [[0.13, -0.08, 0.04]], dtype=truth.dtype
    )
    parent = truth - basis.combine(true_coefficients)
    points = build_fixed_physics_points(
        truth.shape[1], (0, 1), count=128, seed=277
    )
    result = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-10
        ),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        time_order=2,
        defect_design="sparse_interior",
        defect_test_function_width=3,
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=10.0,
    )
    assert bool(result.accepted[0])
    assert result.future_truth_used is False
    assert result.design_row_count == len(points)
    torch.testing.assert_close(
        result.coefficients,
        true_coefficients,
        atol=2.0e-3,
        rtol=2.0e-3,
    )


def test_defect_test_function_rejects_even_width():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    points = build_fixed_physics_points(
        basis.time_count, (0, 1), count=8, seed=281
    )
    with pytest.raises(ValueError, match="positive odd integer"):
        solve_causal_defect_correction(
            basis,
            case["field"],
            case["velocity"],
            case["source_parameters"],
            case["source_map"],
            case["time_s"],
            (0, 1),
            points,
            weights=DefectCorrectionWeights(
                defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-4
            ),
            dt=case["dt"],
            dx=case["dx"],
            dz=case["dz"],
            time_order=2,
            defect_design="sparse_interior",
            defect_test_function_width=2,
        )


def test_sparse_physics_solve_rejects_cpml_or_fourth_order_contract():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    points = build_fixed_physics_points(
        basis.time_count, (0, 1), count=8, seed=179
    )
    with pytest.raises(ValueError, match="requires time_order=2 without CPML"):
        solve_causal_defect_correction(
            basis,
            case["field"],
            case["velocity"],
            case["source_parameters"],
            case["source_map"],
            case["time_s"],
            (0, 1),
            points,
            weights=DefectCorrectionWeights(
                defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-4
            ),
            dt=case["dt"],
            dx=case["dx"],
            dz=case["dz"],
            time_order=4,
            defect_design="sparse_interior",
        )


def test_convex_direction_is_projected_into_the_correction_energy_ball():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    parent = case["field"] - basis.combine(torch.tensor([[2.0, -1.5, 1.0]]))
    points = build_fixed_physics_points(
        case["field"].shape[1], (0, 1), count=128, seed=79
    )
    result = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        weights=DefectCorrectionWeights(defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-10),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=0.05,
    )
    assert bool(result.accepted[0])
    assert float(result.unconstrained_correction_ratio[0]) > 0.05
    assert 0.0 < float(result.projection_scale[0]) < 1.0
    assert float(result.correction_ratio[0]) <= 0.05 * (1.0 + 1.0e-5)
    assert float(result.objective_after[0]) <= float(result.objective_before[0])


def test_first_order_projection_keeps_ridge_direction_but_trains_trust_radius():
    case = _forced_case(dtype=torch.float32)
    fixed = _factorized_basis(case, rank=3, phase_rank=0)
    trust_logit = torch.tensor([-1.0], requires_grad=True)
    live = FactorizedCausalBasis(
        spatial_modes=fixed.spatial_modes,
        temporal_modes=fixed.temporal_modes,
        causal_envelope=fixed.causal_envelope,
        field_scale=fixed.field_scale,
        phase_reference=fixed.phase_reference,
        trust_fraction=torch.sigmoid(trust_logit),
        phase_rank=fixed.phase_rank,
    )
    parent = case["field"] - fixed.combine(torch.tensor([[2.0, -1.5, 1.0]]))
    points = build_fixed_physics_points(
        case["field"].shape[1], (0, 1), count=128, seed=97
    )
    hard_limit = 0.2
    with torch.no_grad():
        result = solve_causal_defect_correction(
            live.detached(),
            parent,
            case["velocity"],
            case["source_parameters"],
            case["source_map"],
            case["time_s"],
            (0, 1),
            points,
            weights=DefectCorrectionWeights(
                defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-10
            ),
            dt=case["dt"],
            dx=case["dx"],
            dz=case["dz"],
            minimum_relative_improvement=0.0,
            maximum_condition_number=1.0e12,
            maximum_correction_ratio=hard_limit,
        )
    coefficients = _first_order_projected_coefficients(
        live, result, maximum_correction_ratio=hard_limit
    )
    torch.testing.assert_close(coefficients.detach(), result.coefficients)
    live.combine(coefficients).square().mean().backward()
    assert trust_logit.grad is not None
    assert float(trust_logit.grad.abs()) > 0.0


def test_calibrated_strength_floor_exactly_abstains_to_the_parent():
    case = _forced_case(dtype=torch.float32)
    basis = _factorized_basis(case, rank=3, phase_rank=0)
    parent = case["field"] - basis.combine(torch.tensor([[0.2, -0.1, 0.05]]))
    points = build_fixed_physics_points(
        case["field"].shape[1], (0, 1), count=64, seed=89
    )
    reference = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-8
        ),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=0.1,
    )
    abstained = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        weights=DefectCorrectionWeights(
            defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-8
        ),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=0.1,
        minimum_unconstrained_correction_ratio=float(
            reference.unconstrained_correction_ratio[0]
        )
        + 1.0,
    )
    assert not bool(abstained.accepted[0])
    assert abstained.rollback_reasons == ("calibrated_causal_strength_abstention",)
    torch.testing.assert_close(abstained.field, parent)
    assert float(abstained.correction_ratio[0]) == 0.0


def test_train_only_strength_calibration_selects_a_safe_beneficial_floor():
    reports = []
    for family in ("uniform", "layered", "marmousi"):
        for index, (strength, adapted) in enumerate(
            ((2.0, 0.8), (1.5, 0.9), (0.5, 1.2), (0.4, 1.1))
        ):
            reports.append(
                {
                    "sample_id": f"{family}-{index}",
                    "medium_type": family,
                    "parent_future_fullfield_relative_l2": 1.0,
                    "future_fullfield_relative_l2": adapted,
                    "adaptation": {
                        "accepted": True,
                        "unconstrained_correction_ratio": strength,
                    },
                }
            )
    selected = select_strength_threshold(
        reports,
        minimum_nonworse_fraction=1.0,
        minimum_family_nonworse_fraction=1.0,
        minimum_mean_improvement=0.01,
    )
    assert float(selected["strength_floor"]) == pytest.approx(1.5)
    assert float(selected["nonworse_fraction"]) == 1.0
    assert float(selected["mean_relative_improvement"]) > 0.0


def test_train_only_family_strength_calibration_uses_per_family_floors():
    patterns = {
        "uniform": ((2.0, 0.8), (1.5, 0.9), (0.5, 1.2), (0.4, 1.1)),
        "layered": ((2.0, 0.9), (1.5, 0.9), (0.5, 0.8), (0.4, 1.1)),
        "marmousi": ((2.0, 0.95), (1.5, 1.1), (0.5, 1.2), (0.4, 1.3)),
    }
    reports = []
    for family, rows in patterns.items():
        for index, (strength, adapted) in enumerate(rows):
            reports.append(
                {
                    "sample_id": f"{family}-{index}",
                    "medium_type": family,
                    "parent_future_fullfield_relative_l2": 1.0,
                    "future_fullfield_relative_l2": adapted,
                    "adaptation": {
                        "accepted": True,
                        "unconstrained_correction_ratio": strength,
                    },
                }
            )
    selected = select_family_strength_thresholds(
        reports,
        minimum_nonworse_fraction=0.75,
        minimum_family_nonworse_fraction=0.75,
        minimum_mean_improvement=0.01,
    )
    floors = selected["strength_floor_by_family"]
    assert floors == pytest.approx(
        {"uniform": 1.5, "layered": 0.5, "marmousi": 2.0}
    )
    assert selected["selection_strategy"].startswith("family_specific")
    assert float(selected["nonworse_fraction"]) >= 0.75


def test_weighted_ridge_is_differentiable_and_rejects_negative_weights():
    torch.manual_seed(9)
    design = torch.randn(1, 32, 5, requires_grad=True)
    expected = torch.randn(1, 5)
    target = torch.einsum("rmk,rk->rm", design.detach(), expected)
    result = solve_weighted_ridge(design, target, prior_precision=1.0e-6)
    loss = (result.coefficients - expected).square().mean()
    loss.backward()
    assert design.grad is not None and torch.isfinite(design.grad).all()
    with pytest.raises(ValueError, match="nonnegative"):
        solve_weighted_ridge(
            design.detach(), target, row_weights=-torch.ones(32)
        )


def test_meta_loss_is_explicitly_future_only():
    target = torch.randn(1, 7, 17, 17)
    candidate = target.clone()
    candidate[:, :2] += 100.0
    terms = meta_defect_correction_loss(
        candidate,
        target,
        torch.zeros(1, 3),
        future_start_index=2,
    )
    assert float(terms["total"]) == pytest.approx(0.0, abs=1.0e-8)


def test_outer_loss_differentiates_through_the_online_convex_solve():
    case = _forced_case(dtype=torch.float32)
    original = _factorized_basis(case, rank=3, phase_rank=0)
    spatial = original.spatial_modes.detach().requires_grad_(True)
    temporal = original.temporal_modes.detach().requires_grad_(True)
    basis = FactorizedCausalBasis(
        spatial_modes=spatial,
        temporal_modes=temporal,
        causal_envelope=original.causal_envelope,
        field_scale=original.field_scale,
        phase_reference=original.phase_reference,
        trust_fraction=torch.tensor([0.5], requires_grad=True),
        phase_rank=0,
    )
    parent = case["field"] - original.combine(torch.tensor([[0.10, -0.06, 0.03]]))
    points = build_fixed_physics_points(case["field"].shape[1], (0, 1), count=64, seed=83)
    result = solve_causal_defect_correction(
        basis,
        parent,
        case["velocity"],
        case["source_parameters"],
        case["source_map"],
        case["time_s"],
        (0, 1),
        points,
        weights=DefectCorrectionWeights(defect=1.0, observed=0.0, bridge=0.0, prior=1.0e-6),
        dt=case["dt"],
        dx=case["dx"],
        dz=case["dz"],
        minimum_relative_improvement=0.0,
        maximum_condition_number=1.0e12,
        maximum_correction_ratio=0.2,
    )
    loss = (result.field - case["field"]).square().mean()
    loss.backward()
    assert spatial.grad is not None and torch.isfinite(spatial.grad).all()
    assert temporal.grad is not None and torch.isfinite(temporal.grad).all()
    assert basis.trust_fraction.grad is not None
    assert torch.isfinite(basis.trust_fraction.grad).all()
