#!/usr/bin/env python3
"""Small CPU contract test for R40 features, identity, objective, and gradients."""

import torch

import train_r40_frequency_residual_operator as r40


def main() -> int:
    torch.manual_seed(40)
    batch = 3
    retained = 24
    base = torch.randn(batch, 2, retained, retained)
    residual = 0.1 * torch.randn(batch, 2, retained, retained)
    static = torch.randn(batch, 7, retained, retained)
    static_scale = torch.rand(batch, 7) * 100.0 + 1.0e-6
    frequency = torch.tensor([10.0, 30.0, 60.0])
    f0 = torch.tensor([12.0, 20.0, 28.0])
    t0 = torch.tensor([0.05, 0.06, 0.07])
    features = r40.make_features(
        base,
        static,
        static_scale,
        frequency_hz=frequency,
        frequency_scale=torch.ones(batch),
        source_f0_hz=f0,
        source_t0_s=t0,
    )
    assert features.shape == (batch, r40.INPUT_CHANNELS, retained, retained)
    model = r40.FrequencyResidualFNO(
        width=8, modes=4, blocks=2, correction_cap=1.5
    )
    prediction = model(features)
    assert prediction.shape == residual.shape
    assert torch.equal(prediction, torch.zeros_like(prediction))
    loss, components = r40.frequency_objective(
        prediction,
        residual,
        frequency_scale=torch.ones(batch),
        target_square_total=torch.full((batch,), 1.0e4),
        frequency_weight=torch.full((batch,), 2.0),
        family_weight=torch.tensor([1.0, 1.25, 1.75]),
        frequency_count=75,
        tail_weight=1.0,
        hinge_weight=2.0,
        shape_weight=0.05,
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()
    assert float(model.head[-1].weight.grad.abs().sum()) > 0.0
    print(
        {
            "features": tuple(features.shape),
            "prediction_identity_max_abs": float(prediction.detach().abs().max()),
            "loss": float(loss.detach()),
            "head_gradient_l1": float(model.head[-1].weight.grad.abs().sum()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
