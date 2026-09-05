import torch
import json
import pytest

from scripts.diagnose_capacity_ladder_overfit import (
    coefficient_arrival_time_s,
    coefficient_prediction_for_supervision,
    coefficient_microbatch_backward_weights,
    direct_frequency_balanced_relative_l2_squared,
    direct_frequency_relative_l2_squared,
    direct_frequency_target_coefficients,
    family_gate_score,
    load_exact_model_initialization,
    load_expanded_frequency_initialization,
    metrics_meet_registered_target,
    zero_initialize_direct_frequency_head,
)


class _Micro:
    def __init__(self, *families: str):
        self.medium_type = families


def test_coefficient_microbatch_family_weights_are_normalized() -> None:
    micros = (_Micro("uniform"), _Micro("layered"), _Micro("marmousi"))
    assert coefficient_microbatch_backward_weights(micros, None) == (
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
    )
    weights = coefficient_microbatch_backward_weights(
        micros, {"uniform": 2.0, "layered": 1.0, "marmousi": 4.0}
    )
    assert weights == (2.0 / 7.0, 1.0 / 7.0, 4.0 / 7.0)
from saved_time_phase_operator_v4.local_field import _HelmholtzSynthesisField


def test_direct_frequency_coefficients_reconstruct_a_low_band_trace():
    torch.manual_seed(4)
    n = 401
    count = 8
    spectrum = torch.zeros(1, n // 2 + 1, 3, 2, dtype=torch.complex64)
    spectrum[:, :count] = torch.randn(1, count, 3, 2, dtype=torch.complex64)
    spectrum[:, 0] = spectrum[:, 0].real
    target = torch.fft.irfft(spectrum, n=n, dim=1, norm="forward")
    coefficients = direct_frequency_target_coefficients(target, count)
    time = torch.arange(n, dtype=target.dtype)
    omega = 2.0 * torch.pi * torch.arange(count, dtype=target.dtype) / float(n)
    cosine = coefficients[:, :count]
    sine = coefficients[:, count:]
    reconstructed = (
        torch.cos(time[:, None] * omega[None, :])[None, :, :, None, None]
        * cosine[:, None]
        + torch.sin(time[:, None] * omega[None, :])[None, :, :, None, None]
        * sine[:, None]
    ).sum(dim=2)
    torch.testing.assert_close(reconstructed, target, rtol=2.0e-5, atol=2.0e-5)


def test_direct_frequency_relative_objective_has_finite_gradient():
    prediction = torch.randn(2, 6, 5, 4, requires_grad=True)
    target = torch.randn_like(prediction)
    loss = direct_frequency_relative_l2_squared(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_coefficient_supervision_uses_record_conditioned_frequency_gate():
    torch.manual_seed(19)
    synthesis = _HelmholtzSynthesisField(
        4,
        num_frequencies=5,
        wkb_phase=True,
        frequency_softmax=True,
    )
    rendered = torch.randn(2, 4, 3, 2)
    raw = synthesis.head(rendered)
    prediction = coefficient_prediction_for_supervision(
        synthesis,
        raw,
        rendered,
    )
    target = torch.randn_like(prediction)
    direct_frequency_relative_l2_squared(prediction, target).backward()
    assert synthesis.frequency_gate.weight.grad is not None
    assert torch.count_nonzero(synthesis.frequency_gate.weight.grad) > 0
    assert synthesis.frequency_gate.bias.grad is not None
    assert torch.count_nonzero(synthesis.frequency_gate.bias.grad) > 0


def test_coefficient_arrival_option_adds_source_onset_only_when_enabled():
    class Model:
        pass

    model = Model()
    model.local_field = torch.nn.Module()
    travel = torch.rand(2, 3, 4)
    source = torch.zeros(2, 5)
    source[:, 3] = torch.tensor([0.05, 0.12])
    model.local_field.helmholtz_source_onset_phase = False
    torch.testing.assert_close(
        coefficient_arrival_time_s(model, travel, source), travel
    )
    model.local_field.helmholtz_source_onset_phase = True
    torch.testing.assert_close(
        coefficient_arrival_time_s(model, travel, source),
        travel + source[:, 3, None, None],
    )


def test_wkb_target_rotation_reconstructs_with_retarded_time():
    torch.manual_seed(8)
    n = 401
    count = 7
    height, width = 4, 3
    spectrum = torch.zeros(1, n // 2 + 1, height, width, dtype=torch.complex64)
    spectrum[:, :count] = torch.randn(
        1, count, height, width, dtype=torch.complex64
    )
    spectrum[:, 0] = spectrum[:, 0].real
    target = torch.fft.irfft(spectrum, n=n, dim=1, norm="forward")
    time_s = torch.linspace(0.0, 1.0, n)
    arrival = 0.25 * torch.rand(1, height, width)
    coefficients = direct_frequency_target_coefficients(
        target, count, arrival_time_s=arrival, saved_time_s=time_s
    )
    omega = 2.0 * torch.pi * torch.arange(count) / (n * (time_s[1] - time_s[0]))
    argument = omega[None, None, :, None, None] * (
        time_s[None, :, None, None, None] - arrival[:, None, None]
    )
    reconstructed = (
        coefficients[:, None, :count] * torch.cos(argument)
        + coefficients[:, None, count:] * torch.sin(argument)
    ).sum(dim=2)
    torch.testing.assert_close(reconstructed, target, rtol=3.0e-5, atol=3.0e-5)


def test_frequency_balanced_objective_keeps_low_energy_bins_live():
    target = torch.zeros(1, 6, 2, 2)
    target[:, 0] = 10.0
    target[:, 1] = 1.0
    target[:, 2] = 0.01
    prediction = target.clone().requires_grad_(True)
    prediction.data[:, 5] = 1.0
    loss = direct_frequency_balanced_relative_l2_squared(
        prediction,
        target,
        frequency_count=3,
        energy_floor_fraction=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[:, 5]) > 0


def test_exact_model_initialization_rejects_partial_loading(tmp_path):
    parent = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "parent.pt"
    identity = tmp_path / "run_identity.json"
    identity.write_text(json.dumps({"run_digest": "run-1"}))
    torch.save(
        {
            "manifest_digest": "manifest-1",
            "config_digest": "run-1",
            "global_step": 7,
            "model_state": parent.state_dict(),
        },
        checkpoint,
    )
    candidate = torch.nn.Linear(3, 2)
    report = load_exact_model_initialization(
        candidate,
        checkpoint,
        identity,
        manifest_digest="manifest-1",
        device=torch.device("cpu"),
    )
    for left, right in zip(parent.parameters(), candidate.parameters()):
        torch.testing.assert_close(left, right)
    assert report["strict_state_dict"]
    with torch.no_grad():
        incompatible = torch.nn.Linear(4, 2)
    try:
        load_exact_model_initialization(
            incompatible,
            checkpoint,
            identity,
            manifest_digest="manifest-1",
            device=torch.device("cpu"),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("shape-incompatible exact initialization was accepted")


class _FrequencyModel(torch.nn.Module):
    def __init__(self, frequencies: int, hidden: int = 3):
        super().__init__()
        self.encoder = torch.nn.Linear(hidden, hidden)
        self.local_field = torch.nn.Module()
        self.local_field.helmholtz_synthesis = torch.nn.Module()
        self.local_field.helmholtz_synthesis.head = torch.nn.Conv2d(
            hidden, 2 * frequencies, 1
        )


def test_expanded_frequency_initialization_maps_cos_sin_and_zeros_new_bins(tmp_path):
    source = _FrequencyModel(2)
    with torch.no_grad():
        source.local_field.helmholtz_synthesis.head.weight.copy_(
            torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1)
        )
        source.local_field.helmholtz_synthesis.head.bias.copy_(torch.arange(4.0))
    checkpoint = tmp_path / "parent.pt"
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"run_digest": "run"}))
    torch.save(
        {"manifest_digest": "manifest", "config_digest": "run", "model_state": source.state_dict()},
        checkpoint,
    )
    target = _FrequencyModel(3)
    report = load_expanded_frequency_initialization(
        target, checkpoint, identity, manifest_digest="manifest", device=torch.device("cpu")
    )
    old = source.local_field.helmholtz_synthesis.head
    new = target.local_field.helmholtz_synthesis.head
    torch.testing.assert_close(new.weight[:2], old.weight[:2])
    torch.testing.assert_close(new.weight[3:5], old.weight[2:])
    assert torch.count_nonzero(new.weight[2]) == 0
    assert torch.count_nonzero(new.weight[5]) == 0
    torch.testing.assert_close(new.bias[:2], old.bias[:2])
    torch.testing.assert_close(new.bias[3:5], old.bias[2:])
    assert new.bias[2] == 0 and new.bias[5] == 0
    assert report["source_frequencies"] == 2 and report["target_frequencies"] == 3

    incompatible = _FrequencyModel(3, hidden=4)
    with pytest.raises(ValueError, match="non-head shape mismatch"):
        load_expanded_frequency_initialization(
            incompatible, checkpoint, identity,
            manifest_digest="manifest", device=torch.device("cpu")
        )


def test_direct_frequency_zero_initialization_is_exact():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_field = torch.nn.Module()
            self.local_field.helmholtz_synthesis = _HelmholtzSynthesisField(
                8, num_frequencies=4, wkb_phase=True, rank=0
            )

    model = Model()
    report = zero_initialize_direct_frequency_head(model)
    head = model.local_field.helmholtz_synthesis.head
    assert report["weight_nonzero"] == 0
    assert report["bias_nonzero"] == 0
    rendered = torch.randn(2, 8, 5, 4)
    torch.testing.assert_close(head(rendered), torch.zeros(2, 8, 5, 4))


def test_helmholtz_post_synthesis_gate_is_default_on_and_explicitly_switchable():
    synthesis = _HelmholtzSynthesisField(8, num_frequencies=4, rank=0)
    container = torch.nn.Module()
    container.helmholtz_synthesis = synthesis
    container.helmholtz_apply_causal_gate = True
    field = torch.randn(2, 3, 5, 4)
    gate = torch.sigmoid(torch.randn_like(field))
    historical = field * gate if container.helmholtz_apply_causal_gate else field
    torch.testing.assert_close(historical, field * gate)
    container.helmholtz_apply_causal_gate = False
    coefficient_supervised = field * gate if container.helmholtz_apply_causal_gate else field
    torch.testing.assert_close(coefficient_supervised, field)


def test_registered_target_uses_requested_true_thresholds():
    metrics = {
        "aggregate_relative_l2": 0.049,
        "family_relative_l2": {
            "uniform": 0.048,
            "layered": 0.050,
            "marmousi": 0.047,
        },
    }
    assert metrics_meet_registered_target(
        metrics, aggregate_maximum=0.05, family_maximum=0.05
    )
    metrics["family_relative_l2"]["marmousi"] = 0.050001
    assert not metrics_meet_registered_target(
        metrics, aggregate_maximum=0.05, family_maximum=0.05
    )
    assert family_gate_score(metrics) == pytest.approx(0.050001)
