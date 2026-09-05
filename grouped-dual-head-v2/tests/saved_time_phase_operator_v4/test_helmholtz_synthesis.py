"""Contract tests for the temporal-frequency (Helmholtz) coarse-field form (H1).

The launch-critical claims pinned here:
  * ``_HelmholtzSynthesisField`` reproduces the exact analytic inverse transform
    ``p(t) = sum_j a_j cos(w_j t) + b_j sin(w_j t)`` on a fixed rfft grid;
  * the frequency bank matches numpy.fft.rfftfreq for the stored-time axis;
  * frame synthesis is invariant to the frame-chunk (a memory knob, not a math knob);
  * PER-FRAME QUERY INVARIANCE at the WHOLE-MODEL level: a single-frame (count==1)
    query of frame i is bit-identical to that frame inside a multi-frame batch --
    the hard contract every coarse-field form must satisfy;
  * the coefficient head + phase anchor receive real gradient (the field is trainable).
"""
from __future__ import annotations

import copy
import math
import types

import numpy as np
import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _BackgroundBornConditioner,
    _HelmholtzSynthesisField,
)
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


# --- module-level: analytic inverse transform + frequency bank ----------------------


def test_frequency_bank_matches_numpy_rfftfreq():
    mod = _HelmholtzSynthesisField(8, num_frequencies=16)
    n_saved = 401
    dt = 0.0025
    values = torch.arange(n_saved, dtype=torch.float64) * dt
    omega = mod.frequency_bank(values)
    ref = 2.0 * math.pi * torch.from_numpy(np.fft.rfftfreq(n_saved, d=dt)[:16])
    assert omega.shape == (16,)
    assert torch.allclose(omega.double(), ref, atol=1e-8)


def test_synthesis_reproduces_analytic_inverse_transform():
    """The forward must be EXACTLY p(t) = sum_j a_j cos(w_j t) + b_j sin(w_j t)
    in the non-WKB (raw-coefficient) mode."""
    torch.manual_seed(0)
    nf, R, H, W = 12, 2, 6, 5
    mod = _HelmholtzSynthesisField(4, num_frequencies=nf, wkb_phase=False)
    rendered = torch.randn(R, 4, H, W)
    arrival = torch.rand(R, H, W) * 0.2 + 0.05
    n_saved = 401
    values = torch.arange(n_saved, dtype=torch.float64) * 0.0025
    time_s = torch.tensor([[0.10, 0.55, 0.90], [0.20, 0.33, 0.77]])
    out = mod(rendered, arrival, time_s, values, domain_t_s=1.0)
    assert out.shape == (R, 3, H, W)
    # reference: build coeffs the same way the module does, then sum the basis.
    coeffs = mod.head(rendered)
    a = coeffs[:, :nf]
    b = coeffs[:, nf:]
    omega = mod.frequency_bank(values).to(rendered.dtype)
    ref = torch.zeros_like(out)
    for r in range(R):
        for ti, t in enumerate(time_s[r]):
            acc = torch.zeros(H, W)
            for j in range(nf):
                acc = acc + a[r, j] * math.cos(float(omega[j]) * float(t))
                acc = acc + b[r, j] * math.sin(float(omega[j]) * float(t))
            ref[r, ti] = acc
    assert torch.allclose(out, ref, atol=1e-5)


def test_wkb_synthesis_uses_retarded_time():
    """WKB mode must be EXACTLY p(t) = sum_j a_j cos(w_j (t - tau)) + b_j sin(w_j (t - tau)),
    with tau the eikonal travel time -- the fast oscillation supplied analytically."""
    torch.manual_seed(0)
    nf, R, H, W = 10, 2, 6, 5
    mod = _HelmholtzSynthesisField(4, num_frequencies=nf, wkb_phase=True)
    rendered = torch.randn(R, 4, H, W)
    arrival = torch.rand(R, H, W) * 0.2 + 0.05
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    time_s = torch.tensor([[0.10, 0.55, 0.90], [0.20, 0.33, 0.77]])
    out = mod(rendered, arrival, time_s, values, domain_t_s=1.0)
    coeffs = mod.head(rendered)
    a = coeffs[:, :nf]
    b = coeffs[:, nf:]
    omega = mod.frequency_bank(values).to(rendered.dtype)
    ref = torch.zeros_like(out)
    for r in range(R):
        for ti, t in enumerate(time_s[r]):
            acc = torch.zeros(H, W)
            for j in range(nf):
                arg = float(omega[j]) * (float(t) - arrival[r])
                acc = acc + a[r, j] * torch.cos(arg) + b[r, j] * torch.sin(arg)
            ref[r, ti] = acc
    assert torch.allclose(out, ref, atol=1e-5)


def test_wkb_wavefront_frequencies_align_at_arrival():
    """At the wavefront (t == tau) every frequency's cos term is 1 -> the amplitude
    envelopes add COHERENTLY (a sharp pulse), which is the whole point of retarded time.
    With b==0 and a==1 the field at t==tau must equal nf exactly."""
    nf, R, H, W = 8, 1, 4, 4
    mod = _HelmholtzSynthesisField(4, num_frequencies=nf, wkb_phase=True)
    with torch.no_grad():
        mod.head.weight.zero_()
        mod.head.bias.zero_()
        mod.head.bias[:nf] = 1.0  # a_j = 1, b_j = 0
    rendered = torch.zeros(R, 4, H, W)
    arrival = torch.full((R, H, W), 0.25)
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    time_s = torch.full((R, 1), 0.25)  # t == tau everywhere
    out = mod(rendered, arrival, time_s, values, domain_t_s=1.0)
    assert torch.allclose(out, torch.full_like(out, float(nf)), atol=1e-4)


def test_frequency_softmax_is_identity_then_record_conditioned_and_trainable():
    torch.manual_seed(17)
    nf, records, height, width = 6, 2, 4, 3
    mod = _HelmholtzSynthesisField(
        4,
        num_frequencies=nf,
        wkb_phase=False,
        frequency_softmax=True,
    )
    rendered = torch.randn(records, 4, height, width)
    raw = mod.head(rendered)

    initial = mod.apply_frequency_gate_to_coefficients(raw, rendered)
    torch.testing.assert_close(initial, raw)

    with torch.no_grad():
        mod.frequency_gate.weight[0, 0] = 0.75
        mod.frequency_gate.weight[1, 0] = -0.50
        rendered[1, 0].mul_(-1.0)
    weights = mod.frequency_gate_weights(rendered)
    assert not torch.allclose(weights[0], weights[1])

    gated = mod.apply_frequency_gate_to_coefficients(mod.head(rendered), rendered)
    target = torch.randn_like(gated)
    (gated - target).square().mean().backward()
    assert mod.frequency_gate.weight.grad is not None
    assert torch.count_nonzero(mod.frequency_gate.weight.grad) > 0
    assert mod.frequency_gate.bias.grad is not None
    assert torch.count_nonzero(mod.frequency_gate.bias.grad) > 0


def test_synthesis_frame_chunk_invariance():
    """Frame chunk is a memory knob; the synthesized frames must not depend on it."""
    torch.manual_seed(1)
    mod_a = _HelmholtzSynthesisField(4, num_frequencies=10, frame_chunk=64, wkb_phase=True)
    mod_b = _HelmholtzSynthesisField(4, num_frequencies=10, frame_chunk=1, wkb_phase=True)
    mod_b.load_state_dict(mod_a.state_dict())
    rendered = torch.randn(2, 4, 6, 5)
    arrival = torch.rand(2, 6, 5) * 0.2 + 0.05
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    time_s = torch.rand(2, 7)
    out_a = mod_a(rendered, arrival, time_s, values, domain_t_s=1.0)
    out_b = mod_b(rendered, arrival, time_s, values, domain_t_s=1.0)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_background_born_conditioner_is_exact_warmstart_noop_with_gradient():
    torch.manual_seed(5)
    nf, height, width = 8, 9, 7
    conditioner = _BackgroundBornConditioner(
        12, num_frequencies=nf, background_sigma_cells=2.0
    )
    background = torch.randn(2, 33, height, width)
    velocity = torch.full((2, height, width), 1800.0)
    velocity[0, 3:6] = 2300.0
    velocity[1, :, width // 2 :] = 2600.0
    mapping = torch.tensor([0, 1])
    arrival = torch.rand(2, height, width) * 0.3
    omega = torch.arange(nf, dtype=torch.float32) * (2.0 * math.pi)
    out = conditioner(background, velocity, mapping, arrival, omega)
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    final_weight_grad = conditioner.input_projection[-1].weight.grad
    assert final_weight_grad is not None and final_weight_grad.abs().max() > 0.0


def test_global_background_born_propagator_is_warmstart_noop_with_gradient():
    torch.manual_seed(13)
    nf, height, width = 6, 9, 9
    conditioner = _BackgroundBornConditioner(
        8,
        num_frequencies=nf,
        background_sigma_cells=2.0,
        global_propagation=True,
        propagation_modes=4,
    )
    background = torch.randn(1, 17, height, width)
    velocity = torch.full((1, height, width), 1800.0)
    velocity[:, 3:6, 2:7] = 2400.0
    arrival = torch.rand(1, height, width) * 0.3
    omega = torch.arange(nf, dtype=torch.float32) * (2.0 * math.pi)

    out = conditioner(
        background,
        velocity,
        torch.zeros(1, dtype=torch.long),
        arrival,
        omega,
    )
    assert torch.count_nonzero(out) == 0
    out.sum().backward()
    assert conditioner.output_projection is not None
    output_grad = conditioner.output_projection.weight.grad
    assert output_grad is not None and output_grad.abs().max() > 0.0


def test_global_background_born_coupled_path_is_zero_init_and_receives_gradient():
    torch.manual_seed(131)
    nf, height, width = 6, 9, 9
    conditioner = _BackgroundBornConditioner(
        8,
        num_frequencies=nf,
        global_propagation=True,
        propagation_modes=4,
        direct_frequency_output=True,
        direct_frequency_count=4,
    )
    assert conditioner.output_projection is not None
    with torch.no_grad():
        torch.nn.init.normal_(conditioner.output_projection.weight, std=0.05)
        conditioner.output_projection.bias.zero_()

    background = torch.randn(1, 17, height, width)
    velocity = torch.full((1, height, width), 1800.0)
    velocity[:, 3:6, 2:7] = 2400.0
    arrival = torch.rand(1, height, width) * 0.3
    omega = torch.arange(nf, dtype=torch.float32) * (2.0 * math.pi)

    with torch.no_grad():
        baseline = conditioner(
            background,
            velocity,
            torch.zeros(1, dtype=torch.long),
            arrival,
            omega,
        )
        conditioner.propagation_coupled_gates.fill_(0.25)
        coupled = conditioner(
            background,
            velocity,
            torch.zeros(1, dtype=torch.long),
            arrival,
            omega,
        )
        conditioner.propagation_coupled_gates.zero_()
    assert not torch.allclose(coupled, baseline)

    output = conditioner(
        background,
        velocity,
        torch.zeros(1, dtype=torch.long),
        arrival,
        omega,
    )
    output.square().mean().backward()
    gate_grad = conditioner.propagation_coupled_gates.grad
    assert gate_grad is not None and gate_grad.abs().max() > 0.0


def test_direct_spectral_experts_are_exact_zero_init_and_kernels_learn_immediately():
    torch.manual_seed(137)
    nf, height, width = 6, 9, 9
    conditioner = _BackgroundBornConditioner(
        8,
        num_frequencies=nf,
        global_propagation=True,
        propagation_modes=4,
        direct_frequency_output=True,
        direct_frequency_count=4,
        direct_spectral_experts=2,
    )
    expert = conditioner.direct_spectral_propagator
    assert expert is not None
    assert torch.count_nonzero(expert.kernel) == 0
    assert torch.count_nonzero(expert.contrast_expert_gate.weight) == 0

    background = torch.randn(1, 17, height, width)
    velocity = torch.full((1, height, width), 1800.0)
    velocity[:, 2:7, 3:6] = 2400.0
    arrival = torch.rand(1, height, width) * 0.3
    omega = torch.arange(nf, dtype=torch.float32) * (2.0 * math.pi)
    mapping = torch.zeros(1, dtype=torch.long)

    initial = conditioner(background, velocity, mapping, arrival, omega)
    assert torch.count_nonzero(initial) == 0
    target = torch.randn_like(initial)
    (initial - target).square().mean().backward()
    assert expert.kernel.grad is not None
    assert torch.isfinite(expert.kernel.grad).all()
    assert torch.count_nonzero(expert.kernel.grad) > 0

    conditioner.zero_grad(set_to_none=True)
    with torch.no_grad():
        expert.kernel.normal_(std=1.0e-3)
    opened = conditioner(background, velocity, mapping, arrival, omega)
    assert torch.count_nonzero(opened) > 0
    opened.square().mean().backward()
    assert expert.expert_gate.weight.grad is not None
    assert torch.count_nonzero(expert.expert_gate.weight.grad) > 0
    assert expert.contrast_expert_gate.weight.grad is not None
    assert torch.count_nonzero(expert.contrast_expert_gate.weight.grad) > 0


def test_global_background_born_propagator_can_fit_a_nonlocal_scattering_field():
    torch.manual_seed(41)
    nf, height, width = 6, 9, 9
    student = _BackgroundBornConditioner(
        8,
        num_frequencies=nf,
        global_propagation=True,
        propagation_modes=4,
    )
    teacher = copy.deepcopy(student)
    assert teacher.output_projection is not None
    with torch.no_grad():
        torch.nn.init.normal_(teacher.output_projection.weight, std=0.08)
        torch.nn.init.normal_(teacher.output_projection.bias, std=0.02)

    background = torch.randn(1, 17, height, width)
    velocity = torch.full((1, height, width), 1800.0)
    velocity[:, 3:6, 2:7] = 2400.0
    mapping = torch.zeros(1, dtype=torch.long)
    arrival = torch.rand(1, height, width) * 0.3
    omega = torch.arange(nf, dtype=torch.float32) * (2.0 * math.pi)
    with torch.no_grad():
        target = teacher(background, velocity, mapping, arrival, omega)

    # The target has energy away from the compact contrast patch: the spectral path
    # has genuinely propagated the Born source instead of merely relabelling it locally.
    outside_contrast = torch.ones(height, width, dtype=torch.bool)
    outside_contrast[3:6, 2:7] = False
    assert target[..., outside_contrast].square().mean() > 1.0e-5

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    assert student.output_projection is not None
    for parameter in student.output_projection.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(student.output_projection.parameters(), lr=0.05)
    with torch.no_grad():
        initial_loss = (
            student(background, velocity, mapping, arrival, omega) - target
        ).square().mean()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            student(background, velocity, mapping, arrival, omega) - target
        ).square().mean()
        loss.backward()
        optimizer.step()
    assert loss < 0.02 * initial_loss


def test_direct_background_frequency_output_uses_absolute_time_basis():
    conditioner = _BackgroundBornConditioner(
        8,
        num_frequencies=4,
        global_propagation=True,
        propagation_modes=4,
        direct_frequency_output=True,
    )
    coefficients = torch.randn(2, 8, 5, 4)
    times = torch.tensor([[0.1, 0.35], [0.2, 0.7]])
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    out = conditioner.synthesize_direct_frequency_output(
        coefficients, times, values, frame_chunk=1
    )

    dt = float(values[1] - values[0])
    omega = 2.0 * math.pi * torch.arange(4) / (len(values) * dt)
    a, b = coefficients[:, :4], coefficients[:, 4:]
    ref = torch.zeros_like(out)
    for record in range(2):
        for frame, time in enumerate(times[record]):
            for frequency in range(4):
                phase = omega[frequency] * time
                ref[record, frame] += (
                    a[record, frequency] * torch.cos(phase)
                    + b[record, frequency] * torch.sin(phase)
                )
    assert torch.allclose(out, ref, atol=1e-5)


def test_direct_background_frequency_count_can_be_lower_than_parent_bank():
    conditioner = _BackgroundBornConditioner(
        8,
        num_frequencies=8,
        global_propagation=True,
        propagation_modes=4,
        direct_frequency_output=True,
        direct_frequency_count=4,
    )
    assert conditioner.num_frequencies == 4
    assert conditioner.output_projection.out_channels == 8
    output = conditioner(
        torch.randn(1, 17, 7, 7),
        torch.where(
            torch.arange(7)[None, :, None] < 3,
            torch.tensor(1800.0),
            torch.tensor(2300.0),
        ).expand(1, 7, 7),
        torch.zeros(1, dtype=torch.long),
        torch.rand(1, 7, 7),
        torch.arange(8, dtype=torch.float32),
    )
    assert output.shape == (1, 8, 7, 7)


def test_direct_background_frequency_head_is_zero_init_with_gradient():
    gen = LocalPropagationFieldGenerator(
        width=8,
        pyramid_levels=2,
        saved_time_count=401,
        domain_t_s=1.0,
        domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2),
        activation_checkpointing=False,
        helmholtz_synthesis=True,
        helmholtz_synthesis_frequencies=4,
        helmholtz_synthesis_rank=2,
        helmholtz_background_conditioning=True,
        helmholtz_background_global_propagator=True,
        helmholtz_background_propagation_modes=4,
        helmholtz_background_direct_frequency_head=True,
        helmholtz_background_direct_frequencies=4,
    )
    inputs = _inputs(gen)
    velocity = torch.full((2, 16, 16), 1800.0)
    velocity[0, 4:12, 5:11] = 2400.0
    velocity[1, :, 8:] = 2300.0
    inputs["velocity_mps"] = velocity
    inputs["background_normalized"] = torch.randn(2, 401, 16, 16)
    out = gen(**inputs)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    conditioner = gen.helmholtz_background_conditioner
    assert conditioner is not None and conditioner.output_projection is not None
    grad = conditioner.output_projection.weight.grad
    assert grad is not None and grad.abs().max() > 0.0


def test_direct_background_frequency_field_bypasses_first_arrival_gate():
    gen = LocalPropagationFieldGenerator(
        width=8,
        pyramid_levels=2,
        saved_time_count=401,
        domain_t_s=1.0,
        domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2),
        activation_checkpointing=False,
        helmholtz_synthesis=True,
        helmholtz_synthesis_frequencies=4,
        helmholtz_synthesis_rank=2,
        helmholtz_background_conditioning=True,
        helmholtz_background_global_propagator=True,
        helmholtz_background_propagation_modes=4,
        helmholtz_background_direct_frequency_head=True,
        helmholtz_background_direct_frequencies=4,
    ).eval()
    inputs = _inputs(gen)
    velocity = torch.full((2, 16, 16), 1800.0)
    velocity[:, 8:] = 2300.0
    inputs["velocity_mps"] = velocity
    inputs["background_normalized"] = torch.randn(2, 401, 16, 16)
    conditioner = gen.helmholtz_background_conditioner
    assert conditioner is not None

    def synthesize_constant(self, coefficients, time_s, saved_time_values, *, frame_chunk=64):
        return coefficients.new_ones(
            coefficients.shape[0], time_s.shape[1], coefficients.shape[2], coefficients.shape[3]
        )

    def synthesize_zero(self, coefficients, time_s, saved_time_values, *, frame_chunk=64):
        return coefficients.new_zeros(
            coefficients.shape[0], time_s.shape[1], coefficients.shape[2], coefficients.shape[3]
        )

    with torch.no_grad():
        conditioner.synthesize_direct_frequency_output = types.MethodType(
            synthesize_zero, conditioner
        )
        without_direct = gen(**inputs)
        conditioner.synthesize_direct_frequency_output = types.MethodType(
            synthesize_constant, conditioner
        )
        with_direct = gen(**inputs)

    assert torch.allclose(with_direct - without_direct, torch.ones_like(with_direct), atol=1.0e-6)


def test_background_born_conditioner_uniform_medium_has_zero_physical_source():
    conditioner = _BackgroundBornConditioner(8, num_frequencies=6)
    velocity = torch.full((2, 11, 9), 2000.0)
    contrast = conditioner.slowness_contrast(velocity)
    assert torch.allclose(contrast, torch.zeros_like(contrast), atol=1e-7)
    contrast_channel = conditioner.slowness_contrast(velocity[:, None])
    assert torch.equal(contrast, contrast_channel)
    with torch.no_grad():
        torch.nn.init.normal_(conditioner.input_projection[-1].weight, std=0.1)
        torch.nn.init.normal_(conditioner.input_projection[-1].bias, std=0.1)
    out = conditioner(
        torch.randn(2, 17, 11, 9),
        velocity,
        torch.arange(2),
        torch.rand(2, 11, 9),
        torch.arange(6, dtype=torch.float32),
    )
    assert torch.count_nonzero(out) == 0


# --- whole-model helpers (mirror the real generator fixtures) -----------------------


def _helmholtz_generator(
    *, nf=16, width=8, pyramid_levels=2, source_onset_phase=False,
    source_relative_coordinates=False
):
    return LocalPropagationFieldGenerator(
        width=width, pyramid_levels=pyramid_levels, saved_time_count=401,
        domain_t_s=1.0, domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2), residual=False,
        activation_checkpointing=False,
        helmholtz_synthesis=True, helmholtz_synthesis_frequencies=nf,
        helmholtz_synthesis_source_onset_phase=source_onset_phase,
        helmholtz_source_relative_coordinates=source_relative_coordinates,
    )


def _inputs(gen, *, records=2, count=3, medium_count=2, height=16, grid_w=16, seed=0):
    torch.manual_seed(seed)
    base = gen.width
    pyramid = [torch.randn(medium_count, base, height, grid_w)]
    h, w = height, grid_w
    for _ in range(1, gen.pyramid_levels):
        h = max(1, h // 2); w = max(1, w // 2)
        pyramid.append(torch.randn(medium_count, base, h, w))
    medium = MediumEncoding(pyramid=tuple(pyramid), tokens=None, token_positions=None, rank=None)
    source = SourceEncoding(
        hidden=torch.randn(records, base), rank=None,
        map_field=torch.randn(records, base, height, grid_w), local_medium=None,
    )
    points = height * grid_w
    travel = RayTravelTime(
        seconds=torch.rand(records, points) * 0.3 + 0.05,
        distance_m=torch.rand(records, points) * 1000.0 + 1.0,
        path_velocity_mps=torch.full((records, points), 1500.0),
        endpoint_velocity_mps=torch.full((records, points), 1500.0),
        mean_slowness_s_per_m=torch.full((records, points), 1.0 / 1500.0),
    )
    source_parameters = torch.zeros(records, 5)
    source_parameters[:, 2] = 15.0
    source_parameters[:, 3] = 0.05
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    time_s = torch.tensor([[0.10, 0.55, 0.90], [0.20, 0.33, 0.77]])[:records, :count]
    return dict(
        velocity_mps=torch.randn(records, 1, height, grid_w),
        medium=medium, source=source, source_parameters=source_parameters,
        record_to_medium=torch.arange(records) % medium_count,
        time_s=time_s, travel=travel,
        saved_time_indices=torch.randint(0, 401, (records, count)),
        saved_time_values=values,
    )


def test_source_onset_phase_is_exact_analytic_time_shift():
    plain = _helmholtz_generator(source_onset_phase=False)
    onset_aligned = _helmholtz_generator(source_onset_phase=True)
    onset_aligned.load_state_dict(plain.state_dict())
    plain.helmholtz_apply_causal_gate = False
    onset_aligned.helmholtz_apply_causal_gate = False
    inputs = _inputs(plain)
    onset = inputs["source_parameters"][:, 3]
    shifted_inputs = dict(inputs)
    shifted_inputs["time_s"] = inputs["time_s"] - onset[:, None]
    with torch.no_grad():
        shifted = onset_aligned(**inputs)
        reference = plain(**shifted_inputs)
    torch.testing.assert_close(shifted, reference, rtol=2.0e-5, atol=2.0e-5)


def test_source_relative_coordinates_are_exact_zero_init_and_receive_gradient():
    plain = _helmholtz_generator(source_relative_coordinates=False)
    candidate = _helmholtz_generator(source_relative_coordinates=True)
    loaded = candidate.load_state_dict(plain.state_dict(), strict=False)
    assert sorted(loaded.missing_keys) == [
        "helmholtz_source_relative_projection.bias",
        "helmholtz_source_relative_projection.weight",
    ]
    inputs = _inputs(plain)
    inputs["source_parameters"][:, 0] = torch.tensor([400.0, 1600.0])
    inputs["source_parameters"][:, 1] = torch.tensor([200.0, 1200.0])
    with torch.no_grad():
        reference = plain(**inputs)
        initial = candidate(**inputs)
    torch.testing.assert_close(initial, reference)

    candidate.zero_grad(set_to_none=True)
    candidate(**inputs).square().mean().backward()
    projection = candidate.helmholtz_source_relative_projection
    assert projection is not None
    assert projection.weight.grad is not None
    assert torch.count_nonzero(projection.weight.grad) > 0


def test_whole_model_shape_and_finite():
    gen = _helmholtz_generator()
    gen.eval()
    with torch.no_grad():
        out = gen(**_inputs(gen))
    assert out.shape == (2, 3, 16, 16)
    assert torch.isfinite(out).all()


def test_whole_model_per_frame_query_invariance():
    """HARD CONTRACT: a single-frame (count==1) query of frame i is bit-identical to
    that frame inside the multi-frame batch.  Helmholtz synthesis satisfies this BY
    CONSTRUCTION -- a_j/b_j depend only on the record, and each frame uses only its own
    scalar t through the analytic basis, with no per-frame FiLM / saved-index embedding.
    """
    gen = _helmholtz_generator()
    gen.eval()
    batch = _inputs(gen, count=3)
    with torch.no_grad():
        out_batch = gen(**batch)
        for i in range(batch["time_s"].shape[1]):
            single = dict(batch)
            single["time_s"] = batch["time_s"][:, i : i + 1]
            single["saved_time_indices"] = batch["saved_time_indices"][:, i : i + 1]
            out_single = gen(**single)
            assert torch.allclose(out_single[:, 0], out_batch[:, i], atol=1e-6), (
                f"frame {i} single-query differs from batched query"
            )


def test_whole_model_head_and_phase_anchor_receive_gradient():
    """The Helmholtz coefficient head and the travel-time phase anchor must both train
    (this IS the coarse field, not a gated residual, so there is no warm-start no-op to
    protect -- gradient must flow from the first step)."""
    gen = _helmholtz_generator()
    gen.train()
    out = gen(**_inputs(gen))
    out.square().mean().backward()
    head_w = gen.helmholtz_synthesis.head.weight.grad
    anchor_w = gen.helmholtz_synthesis.phase_anchor.weight.grad
    assert head_w is not None and head_w.abs().max() > 0.0
    assert anchor_w is not None and anchor_w.abs().max() > 0.0


def test_low_rank_factorization_reduces_dof_and_reproduces_basis_mixing():
    """rank>0: a_j(x)=sum_r cos_mix[j,r] B_r(x), b_j(x)=sum_r sin_mix[j,r] B_r(x).
    Per-pixel DOF is R (the basis fields), NOT 2*nf -- the fix for the temporal
    underdetermination the G2 probe measured."""
    torch.manual_seed(0)
    nf, R, W_, H, WW = 20, 6, 4, 6, 5
    mod = _HelmholtzSynthesisField(W_, num_frequencies=nf, wkb_phase=True, rank=R)
    # far fewer mixing params than an independent 2*nf head would need spatially
    assert mod.cos_mix.shape == (nf, R) and mod.sin_mix.shape == (nf, R)
    assert not hasattr(mod, "head")
    rendered = torch.randn(2, W_, H, WW)
    arrival = torch.rand(2, H, WW) * 0.2 + 0.05
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    time_s = torch.tensor([[0.10, 0.55], [0.20, 0.77]])
    out = mod(rendered, arrival, time_s, values, domain_t_s=1.0)
    # reference from basis + per-record conditioned mixing
    basis = mod.basis_head(rendered).reshape(2, R, H * WW)
    pooled = rendered.mean(dim=(-2, -1))
    delta = mod.mix_condition(pooled).reshape(2, 2, nf, R)
    cos_mix = mod.cos_mix[None] + delta[:, 0]
    sin_mix = mod.sin_mix[None] + delta[:, 1]
    a = torch.einsum("bjr,brs->bjs", cos_mix, basis)
    b = torch.einsum("bjr,brs->bjs", sin_mix, basis)
    omega = mod.frequency_bank(values).to(rendered.dtype)
    ref = torch.zeros_like(out)
    for r in range(2):
        for ti, t in enumerate(time_s[r]):
            acc = torch.zeros(H * WW)
            for j in range(nf):
                arg = float(omega[j]) * (float(t) - arrival[r].reshape(-1))
                acc = acc + a[r, j] * torch.cos(arg) + b[r, j] * torch.sin(arg)
            ref[r, ti] = acc.reshape(H, WW)
    assert torch.allclose(out, ref, atol=1e-5)


def test_late_rank_record_gate_preserves_zero_scattering_family():
    torch.manual_seed(73)
    mod = _HelmholtzSynthesisField(
        4,
        num_frequencies=8,
        wkb_phase=True,
        rank=2,
        late_rank=3,
        late_frequencies=4,
    )
    baseline = copy.deepcopy(mod)
    with torch.no_grad():
        mod.late_cos_mix.normal_(std=0.2)
        mod.late_sin_mix.normal_(std=0.2)
    rendered = torch.randn(2, 4, 6, 5)
    arrival = torch.rand(2, 6, 5) * 0.2
    values = torch.arange(401, dtype=torch.float64) * 0.0025
    times = torch.tensor([[0.2, 0.6], [0.2, 0.6]])

    gated = mod(
        rendered,
        arrival,
        times,
        values,
        domain_t_s=1.0,
        late_record_gate=torch.tensor([0.0, 1.0]),
    )
    without_late = baseline(
        rendered,
        arrival,
        times,
        values,
        domain_t_s=1.0,
    )
    assert torch.allclose(gated[0], without_late[0], atol=1e-6)
    assert not torch.allclose(gated[1], without_late[1], atol=1e-6)


def test_low_rank_whole_model_query_invariance():
    """Query invariance must hold in low-rank mode too."""
    gen = LocalPropagationFieldGenerator(
        width=8, pyramid_levels=2, saved_time_count=401,
        domain_t_s=1.0, domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2), residual=False,
        activation_checkpointing=False,
        helmholtz_synthesis=True, helmholtz_synthesis_frequencies=16,
        helmholtz_synthesis_rank=4,
    )
    gen.eval()
    batch = _inputs(gen, count=3)
    with torch.no_grad():
        out_batch = gen(**batch)
        for i in range(batch["time_s"].shape[1]):
            single = dict(batch)
            single["time_s"] = batch["time_s"][:, i : i + 1]
            single["saved_time_indices"] = batch["saved_time_indices"][:, i : i + 1]
            out_single = gen(**single)
            assert torch.allclose(out_single[:, 0], out_batch[:, i], atol=1e-6)


def test_background_conditioned_whole_model_query_invariance():
    """The conditioner always uses the same complete P_bg axis, never the queried set."""

    gen = LocalPropagationFieldGenerator(
        width=8,
        pyramid_levels=2,
        saved_time_count=401,
        domain_t_s=1.0,
        domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2),
        residual=False,
        activation_checkpointing=False,
        helmholtz_synthesis=True,
        helmholtz_synthesis_frequencies=16,
        helmholtz_synthesis_rank=4,
        helmholtz_background_conditioning=True,
    )
    with torch.no_grad():
        torch.nn.init.normal_(
            gen.helmholtz_background_conditioner.input_projection[-1].weight,
            std=0.01,
        )
    gen.eval()
    batch = _inputs(gen, count=3)
    height, width = batch["medium"].pyramid[0].shape[-2:]
    velocity = torch.full((2, height, width), 1800.0)
    velocity[0, 5:10] = 2300.0
    velocity[1, :, width // 2 :] = 2500.0
    batch["velocity_mps"] = velocity
    batch["background_normalized"] = torch.randn(2, 401, height, width)
    with torch.no_grad():
        out_batch = gen(**batch)
        for i in range(batch["time_s"].shape[1]):
            single = dict(batch)
            single["time_s"] = batch["time_s"][:, i : i + 1]
            single["saved_time_indices"] = batch["saved_time_indices"][:, i : i + 1]
            out_single = gen(**single)
            assert torch.allclose(out_single[:, 0], out_batch[:, i], atol=1e-6)


def test_background_conditioned_generator_requires_complete_background():
    gen = LocalPropagationFieldGenerator(
        width=8,
        pyramid_levels=2,
        saved_time_count=401,
        domain_t_s=1.0,
        domain_diagonal_m=2828.0,
        channel_multipliers=(1, 1, 2, 2),
        activation_checkpointing=False,
        helmholtz_synthesis=True,
        helmholtz_synthesis_frequencies=16,
        helmholtz_background_conditioning=True,
    )
    inputs = _inputs(gen)
    inputs["velocity_mps"] = torch.full((2, 16, 16), 1800.0)
    try:
        gen(**inputs)
    except ValueError as exc:
        assert "complete normalized P_bg" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing P_bg conditioning error")


def test_requires_saved_time_values():
    gen = _helmholtz_generator()
    inputs = _inputs(gen)
    inputs["saved_time_values"] = None
    try:
        gen(**inputs)
    except ValueError as exc:
        assert "saved_time_values" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when saved_time_values is None")
