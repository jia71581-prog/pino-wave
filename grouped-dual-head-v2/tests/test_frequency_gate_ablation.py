import pytest
import torch

from saved_time_phase_operator_v4.local_field import _HelmholtzSynthesisField
from scripts.evaluate_frequency_gate_ablation import (
    relative_improvement,
    set_frequency_gate_identity,
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_field = torch.nn.Module()
        self.local_field.helmholtz_synthesis = _HelmholtzSynthesisField(
            4,
            num_frequencies=6,
            frequency_softmax=True,
        )


def test_frequency_gate_identity_ablation_changes_only_gate_parameters():
    model = _Model()
    synthesis = model.local_field.helmholtz_synthesis
    with torch.no_grad():
        synthesis.frequency_gate.weight.normal_()
        synthesis.frequency_gate.bias.normal_()
    head_before = synthesis.head.weight.detach().clone()
    report = set_frequency_gate_identity(model)
    assert report["before"]["weight_nonzero"] > 0
    assert report["before"]["bias_nonzero"] > 0
    assert report["after"] == {"weight_nonzero": 0, "bias_nonzero": 0}
    torch.testing.assert_close(synthesis.head.weight, head_before)


def test_relative_improvement_direction_and_guard():
    assert relative_improvement(0.05, 0.04) == pytest.approx(0.2)
    assert relative_improvement(0.04, 0.05) == pytest.approx(-0.25)
    with pytest.raises(ValueError):
        relative_improvement(0.0, 0.0)
