#!/usr/bin/env python3
"""One full-size BF16 R40 training step for CUDA memory/runtime validation."""

from contextlib import nullcontext

import torch

import train_r40_frequency_residual_operator as r40


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(40)
    batch = 16
    retained = 96
    base = torch.randn(batch, 2, retained, retained, device=device)
    residual = 0.08 * torch.randn(batch, 2, retained, retained, device=device)
    static = torch.randn(batch, 7, retained, retained, device=device)
    static_scale = torch.rand(batch, 7, device=device) * 120.0 + 1.0e-6
    frequency = torch.linspace(8.0, 75.0, batch, device=device)
    f0 = torch.linspace(10.0, 28.0, batch, device=device)
    t0 = torch.full((batch,), 0.06, device=device)
    model = r40.FrequencyResidualFNO(
        width=48, modes=16, blocks=4, correction_cap=1.5
    ).to(device)
    features = r40.make_features(
        base,
        static,
        static_scale,
        frequency_hz=frequency,
        frequency_scale=torch.ones(batch, device=device),
        source_f0_hz=f0,
        source_t0_s=t0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    optimizer.zero_grad(set_to_none=True)
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_bf16_supported()
        else nullcontext()
    )
    with context:
        prediction = model(features)
        loss, _ = r40.frequency_objective(
            prediction.float(),
            residual,
            frequency_scale=torch.ones(batch, device=device),
            target_square_total=torch.full((batch,), 1.0e5, device=device),
            frequency_weight=torch.full((batch,), 2.0, device=device),
            family_weight=torch.ones(batch, device=device),
            frequency_count=75,
            tail_weight=1.0,
            hinge_weight=2.0,
            shape_weight=0.05,
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    print(
        {
            "loss": float(loss.detach()),
            "parameter_count": r40.parameter_count(model),
            "max_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "bf16": torch.cuda.is_bf16_supported(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
