import pytest
import torch

from fno_acoustic import train_elastic_vti as training


def test_component_balanced_relative_l2_equalizes_component_amplitude() -> None:
    target = torch.ones(1, 2, 2, 3, 2)
    target[..., 1] *= 10.0
    pred = target.clone()
    pred[..., 0] += 0.5
    pred[..., 1] += 1.0

    loss, components = training._component_balanced_relative_l2(
        pred,
        target,
        component_weights=[1.0, 1.0],
        eps=1.0e-8,
    )

    assert loss.item() == pytest.approx(0.3, abs=1.0e-6)
    assert components.tolist() == pytest.approx([0.5, 0.1], abs=1.0e-6)


@pytest.mark.parametrize(
    "weights",
    [[1.0], [1.0, 0.0], [1.0, float("nan")]],
)
def test_component_balanced_relative_l2_rejects_invalid_weights(weights) -> None:
    values = torch.ones(1, 2, 2, 3, 2)

    with pytest.raises(ValueError):
        training._component_balanced_relative_l2(
            values,
            values,
            component_weights=weights,
            eps=1.0e-8,
        )


def test_elastic_supervised_loss_without_component_weights_matches_legacy() -> None:
    torch.manual_seed(7)
    pred = torch.randn(2, 3, 4, 5, 2)
    target = torch.randn_like(pred)
    config = {
        "relative_l2_weight": 1.0,
        "mse_weight": 0.05,
        "eps": 1.0e-8,
    }

    actual, _ = training._elastic_supervised_loss(pred, target, config)
    expected, _ = training.combined_loss(pred, target, **config)

    assert torch.allclose(actual, expected)


def test_elastic_supervised_loss_reports_component_diagnostics() -> None:
    target = torch.ones(1, 2, 2, 3, 2)
    pred = target.clone()
    pred[..., 0] += 0.25
    pred[..., 1] += 0.5

    _, parts = training._elastic_supervised_loss(
        pred,
        target,
        {
            "relative_l2_weight": 1.0,
            "mse_weight": 0.0,
            "component_relative_l2_weights": [1.0, 1.0],
            "eps": 1.0e-8,
        },
    )

    assert parts["relative_l2_component_0"] == pytest.approx(0.25)
    assert parts["relative_l2_component_1"] == pytest.approx(0.5)
    assert parts["relative_l2"] == pytest.approx(0.375)


def test_spatial_importance_loss_matches_full_component_loss_when_all_pixels_selected() -> None:
    torch.manual_seed(9)
    pred = torch.randn(1, 2, 3, 4, 2)
    target = torch.randn_like(pred)
    indices = torch.arange(6)
    probabilities = torch.full((6,), 1.0 / 6.0)
    config = {
        "relative_l2_weight": 1.0,
        "mse_weight": 0.0,
        "component_relative_l2_weights": [1.0, 1.0],
        "eps": 1.0e-8,
    }

    sampled, _ = training._spatial_importance_loss(
        pred,
        target,
        flat_indices=indices,
        flat_probabilities=probabilities,
        full_spatial_count=6,
        loss_config=config,
        reweight=True,
    )
    full, _ = training._elastic_supervised_loss(pred, target, config)

    assert torch.allclose(sampled, full, atol=1.0e-6)


def test_spatial_residual_map_supports_elastic_components() -> None:
    values = torch.zeros(2, 3, 4, 5, 2)

    result = training._spatial_residual_map(values + 1.0, values)

    assert result.shape == (3, 4)
    assert torch.allclose(result, torch.ones_like(result))
