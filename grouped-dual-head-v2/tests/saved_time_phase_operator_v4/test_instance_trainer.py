from __future__ import annotations

import torch

from saved_time_phase_operator_v4.instance_adaptation.adapters import OnsetAdaptedV5
from saved_time_phase_operator_v4.instance_adaptation.contracts import SnapshotAccessAudit
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetRecord
from saved_time_phase_operator_v4.instance_adaptation.trainer import (
    adapt_instance,
    build_supervised_chonknoris_residual_function,
    high_residual_spatial_indices,
    linearize_supervised_chonknoris_batch,
    project_residual_gate_to_observations,
)
from saved_time_phase_operator_v4.instance_adaptation.losses import PhysicsSampling
from saved_time_phase_operator_v4.instance_adaptation.chonknoris import (
    ReducedChonknorisConfig,
    _linearize,
)


class DummyParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4_000_000,), 2.0))

    def predict_wavefield(self, batch, times, height, width):
        # A spatially varying field so the residual's parent-relative output
        # scale (std of the parent field) is nonzero and a corrupted residual
        # can actually perturb the observed frames.
        base = self.weight[0]
        ramp = torch.linspace(-1.0, 1.0, height * width).reshape(1, 1, height, width)
        return base + ramp.expand(batch, times, height, width).clone()


def _record() -> GuardedOnsetRecord:
    observed_indices = (2, 3)
    return GuardedOnsetRecord(
        velocity_mps=torch.full((1, 17, 17), 1500.0),
        source_parameters=torch.tensor([80.0, 80.0, 12.0, 0.01, 1.0]),
        source_map=torch.zeros(1, 17, 17),
        time_s=torch.arange(8, dtype=torch.float32) * 0.0025,
        x_m=torch.arange(17, dtype=torch.float32),
        z_m=torch.arange(17, dtype=torch.float32),
        sample_id="sample",
        group_id="medium",
        medium_type="uniform",
        source_index=0,
        observed_indices=observed_indices,
        observed_wavefield=torch.zeros(2, 17, 17),
        input_digest="digest",
        audit=SnapshotAccessAudit(observed_indices),
    )


def _model_and_parent():
    parent = DummyParent()
    wrapper = OnsetAdaptedV5(parent, latent_dim=8, lora_rank=2)
    record = _record()
    parent_field = parent.predict_wavefield(1, len(record.time_s), 17, 17).detach()
    return wrapper, record, parent_field


def test_trainer_updates_only_adapter_and_records_access():
    model, record, parent = _model_and_parent()
    result = adapt_instance(model, record, parent, adam_steps=1, lbfgs_steps=0)
    assert result.accessed_true_indices == record.observed_indices
    assert result.trainable_parameter_count < result.total_parameter_count / 100


def test_nonfinite_or_bad_physics_candidate_rolls_back(monkeypatch):
    model, record, parent = _model_and_parent()
    monkeypatch.setattr(
        "saved_time_phase_operator_v4.instance_adaptation.trainer.instance_loss_terms",
        lambda **kwargs: {"total": torch.tensor(float("nan"), requires_grad=True)},
    )
    result = adapt_instance(model, record, parent, adam_steps=1, lbfgs_steps=0)
    assert result.accepted is False
    assert result.rollback_reason == "nonfinite_loss"


def test_limited_lbfgs_uses_fixed_points():
    model, record, parent = _model_and_parent()
    result = adapt_instance(model, record, parent, adam_steps=1, lbfgs_steps=1)
    assert result.lbfgs_closure_calls >= 1
    assert result.future_truth_used is False


def test_bad_pretrained_residual_rolls_back_to_parent_not_pretrained_state():
    model, record, parent = _model_and_parent()
    with torch.no_grad():
        model.residual.output.bias.fill_(1.0)
    result = adapt_instance(model, record, parent, adam_steps=0, lbfgs_steps=0)
    assert result.accepted is False
    restored = model.raw_wavefield(
        parent,
        record.velocity_mps.unsqueeze(0),
        record.source_parameters.unsqueeze(0),
        record.observed_wavefield.unsqueeze(0),
        record.time_s,
    )
    assert torch.allclose(restored, parent)


def test_deployment_mode_freezes_meta_network_and_trains_only_latent():
    model, _, _ = _model_and_parent()
    model.set_deployment_mode(True)
    active = model.active_parameters()
    assert set(active) == {model.latent_delta, model.residual_gate}
    assert model.latent_delta.requires_grad and model.residual_gate.requires_grad
    assert all(not p.requires_grad for p in model.conditioner.parameters())
    assert all(not p.requires_grad for p in model.residual.parameters())


def test_deployment_lora_optimizes_only_latent_delta():
    model, record, parent = _model_and_parent()
    conditioner_before = [p.detach().clone() for p in model.conditioner.parameters()]
    residual_before = [p.detach().clone() for p in model.residual.parameters()]
    result = adapt_instance(
        model, record, parent, adam_steps=2, lbfgs_steps=0, deployment_lora=True
    )
    # Only the per-instance deployment parameters (latent shift + residual gate).
    assert result.trainable_parameter_count == (
        model.latent_delta.numel() + model.residual_gate.numel()
    )
    # The frozen meta network is untouched regardless of accept/rollback.
    for before, after in zip(conditioner_before, model.conditioner.parameters()):
        assert torch.equal(before, after)
    for before, after in zip(residual_before, model.residual.parameters()):
        assert torch.equal(before, after)
    assert result.future_truth_used is False


def test_deployment_lora_rollback_restores_zero_latent_delta():
    model, record, parent = _model_and_parent()
    # A harmful pre-set latent/gate should be rolled back to the frozen parent
    # (latent_delta = 0 and residual_gate = 0) when the gates reject it.
    with torch.no_grad():
        model.latent_delta.fill_(5.0)
        model.residual_gate.fill_(3.0)
    result = adapt_instance(
        model, record, parent, adam_steps=0, lbfgs_steps=0, deployment_lora=True
    )
    if not result.accepted:
        assert torch.count_nonzero(model.latent_delta) == 0
        assert float(model.residual_gate) == 0.0


def test_deployment_pde_weight_enables_self_normalized_physics_term(monkeypatch):
    # The self-normalized LWC-84 PDE residual is scale-invariant, so it is valid
    # in the normalized deployment space.  deployment_pde_weight>0 must flow into
    # the loss weights the closure actually uses (not silently dropped to 0).
    import saved_time_phase_operator_v4.instance_adaptation.trainer as trainer_mod

    seen = []
    original = trainer_mod.instance_loss_terms

    def _spy(*args, **kwargs):
        seen.append(float(kwargs["weights"].pde))
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "instance_loss_terms", _spy)
    model, record, parent = _model_and_parent()
    adapt_instance(
        model, record, parent, adam_steps=1, lbfgs_steps=0,
        deployment_lora=True, deployment_pde_weight=0.25,
    )
    assert seen, "closure never ran"
    assert all(w == 0.25 for w in seen)

    # Default (0.0) keeps the prior observed-only deployment behavior.
    seen.clear()
    model, record, parent = _model_and_parent()
    adapt_instance(
        model, record, parent, adam_steps=1, lbfgs_steps=0, deployment_lora=True
    )
    assert seen and all(w == 0.0 for w in seen)


def test_rad_sampling_resamples_during_adam_and_freezes_for_lbfgs(monkeypatch):
    # RAD must (a) run end-to-end through adapt_instance, (b) resample points
    # during the Adam phase, and (c) stop resampling once LBFGS begins.
    import saved_time_phase_operator_v4.instance_adaptation.trainer as trainer_mod

    calls = {"rad": 0}
    original = trainer_mod.build_rad_physics_points

    def _spy(*args, **kwargs):
        calls["rad"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "build_rad_physics_points", _spy)
    model, record, parent = _model_and_parent()
    sampling = PhysicsSampling(method="rad", k=1.0, c=1.0, time_tilt=1.5, resample_every=2)
    result = adapt_instance(
        model, record, parent,
        adam_steps=6, lbfgs_steps=3, deployment_lora=True,
        deployment_pde_weight=0.1, physics_sampling=sampling,
    )
    # Adam phase: 6 closures with resample_every=2 → 3 RAD draws (calls 0,2,4).
    # LBFGS closures must not add more (points frozen).
    assert calls["rad"] == 3
    # Causal + safety contracts untouched by the sampling change.
    assert result.future_truth_used is False
    assert result.accessed_true_indices == record.observed_indices


def test_uniform_default_never_calls_rad(monkeypatch):
    import saved_time_phase_operator_v4.instance_adaptation.trainer as trainer_mod

    called = {"rad": 0}
    monkeypatch.setattr(
        trainer_mod, "build_rad_physics_points",
        lambda *a, **k: called.__setitem__("rad", called["rad"] + 1),
    )
    model, record, parent = _model_and_parent()
    adapt_instance(
        model, record, parent, adam_steps=3, lbfgs_steps=2, deployment_lora=True,
    )
    assert called["rad"] == 0


def test_reduced_chonknoris_records_contractive_causal_iterations():
    model, record, parent = _model_and_parent()
    # Make the residual gate observable in the reduced state.  The normal
    # zero-initialized head is deliberately state-independent before meta-training.
    with torch.no_grad():
        model.residual.output.bias.fill_(0.1)
    result = adapt_instance(
        model,
        record,
        parent,
        deployment_lora=True,
        optimizer_name="chonknoris",
        deployment_pde_weight=0.1,
        chonknoris_config=ReducedChonknorisConfig(
            iterations=2,
            residual_pool_size=2,
            physics_point_count=8,
        ),
    )
    assert result.optimizer_name == "chonknoris"
    assert result.future_truth_used is False
    assert result.accessed_true_indices == record.observed_indices
    assert result.trainable_parameter_count == model.latent_delta.numel() + 1
    assert all(value < 1.0 for value in result.contraction_history)


def test_vectorized_supervised_linearization_matches_individual_results():
    model, record, parent = _model_and_parent()
    parent_batch = torch.cat((parent, parent + 0.05), dim=0)
    velocity = record.velocity_mps.unsqueeze(0).expand(2, -1, -1, -1).clone()
    source = record.source_parameters.unsqueeze(0).expand(2, -1).clone()
    observed = record.observed_wavefield.unsqueeze(0).expand(2, -1, -1, -1).clone()
    times = record.time_s.unsqueeze(0).expand(2, -1).clone()
    spatial_ramp = torch.arange(17 * 17, dtype=parent.dtype).reshape(1, 1, 17, 17)
    spatial_ramp = spatial_ramp.expand_as(parent) / float(17 * 17)
    target = parent_batch + torch.stack(
        (0.1 * spatial_ramp, -0.2 * spatial_ramp), dim=0
    ).squeeze(1)
    state = model.deployment_state().detach()
    for sampling in ("average_pool", "topk"):
        residuals, jacobians, _ = linearize_supervised_chonknoris_batch(
            model,
            parent_batch,
            velocity,
            source,
            observed,
            times,
            target,
            pool_size=2,
            residual_sampling=sampling,
        )
        for index in range(2):
            residual_fn = build_supervised_chonknoris_residual_function(
                model,
                parent_batch[index : index + 1],
                velocity[index : index + 1],
                source[index : index + 1],
                observed[index : index + 1],
                times[index],
                target[index : index + 1],
                pool_size=2,
                residual_sampling=sampling,
            )
            expected_residual, expected_jacobian = _linearize(residual_fn, state)
            assert torch.equal(residuals[index], expected_residual)
            assert torch.equal(jacobians[index], expected_jacobian)


def test_high_residual_spatial_indices_keep_each_frames_hotspots():
    residual = torch.zeros(2, 3, 4)
    residual[0, 1, 2] = -9.0
    residual[0, 0, 0] = 4.0
    residual[1, 2, 3] = 8.0
    residual[1, 1, 1] = -5.0
    selected = high_residual_spatial_indices(residual, points_per_frame=2)
    assert selected.tolist() == [[6, 0], [11, 5]]


def test_causal_gate_projection_minimizes_observed_error_without_future_truth():
    class ScalarGateModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_gate = torch.nn.Parameter(torch.tensor(1.0))
            self.latent = torch.nn.Parameter(torch.zeros(2))

        def deployment_state(self):
            return torch.cat((self.latent, self.residual_gate.reshape(1)))

        def raw_wavefield_from_state(
            self, parent, velocity, source, observed, time_s, state
        ):
            del velocity, source, observed, time_s
            correction = torch.ones_like(parent) * 2.0
            return parent + state[-1] * correction

    model = ScalarGateModel()
    parent = torch.ones(1, 4, 2, 2)
    observed = torch.full((1, 2, 2, 2), 0.5)
    projected = project_residual_gate_to_observations(
        model,
        parent,
        torch.empty(1),
        torch.empty(1),
        observed,
        torch.arange(4),
        (1, 2),
    )
    assert abs(projected + 0.25) < 1.0e-7
    candidate = model.raw_wavefield_from_state(
        parent,
        torch.empty(1),
        torch.empty(1),
        observed,
        torch.arange(4),
        model.deployment_state(),
    )
    assert torch.equal(candidate[:, [1, 2]], observed)
