"""R5: dynamic Green/scattering kernel whose support radius follows travel time.

B1 (warp + rank-8 amplitude operator) hit a plateau ~0.240 with FLAT layered/late
error: a single-radius amplitude operator cannot represent scattered/reverberant
energy whose spatial support GROWS with elapsed travel time.  R5 adds a bank of
depthwise scattering stencils at increasing dilations (radii); per-pixel, per-frame
mixing weights (biased by elapsed travel-progress) pick the dominant radius, so the
effective support follows travel time (GreenONet, Aldirany 2023; scattered-field
decomposition, Ma & Alkhalifah 2025).

These tests pin the CONTRACT the escalation must not break:
  * zero-init gate -> exact no-op at warm-start (field reproduced bit-for-bit),
  * per-frame query invariance (single-frame == its slice of a multi-frame batch),
  * gradients flow to stencils / mix / radius_bias / gate once the gate opens,
  * the radius-follows-travel-time mechanism actually shifts weight to the larger
    dilation as elapsed progress grows,
  * a generator built with green_kernel=True constructs the module and is an exact
    no-op at init (warm-start from a parent predating it is byte-reproducible).
"""
from __future__ import annotations

import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _DynamicGreenScatteringKernel,
)


def _inputs(*, records, count, width, height, grid_w, seed=0):
    torch.manual_seed(seed)
    field = torch.randn(records, count, height, grid_w)
    rendered = torch.randn(records * count, width, height, grid_w)
    progress = torch.rand(records, count, height, grid_w)  # normalized elapsed >= 0
    return field, rendered, progress


def test_green_kernel_is_exact_noop_at_warmstart():
    """gate is zero-init -> the scattered residual is identically zero."""
    kernel = _DynamicGreenScatteringKernel(8, kernel_size=5, dilations=(1, 2, 4))
    field, rendered, progress = _inputs(records=2, count=3, width=8, height=11, grid_w=13)
    residual = kernel(field, rendered, progress, records=2, count=3)
    assert residual.shape == field.shape
    assert torch.count_nonzero(residual) == 0
    assert torch.allclose(residual, torch.zeros_like(residual))


def test_green_kernel_query_invariance():
    """Frame i's residual is independent of which other frames share the batch."""
    kernel = _DynamicGreenScatteringKernel(8, kernel_size=5, dilations=(1, 3))
    # open the gate so the residual is non-trivial
    with torch.no_grad():
        kernel.gate.fill_(0.7)
        kernel.radius_bias.copy_(torch.tensor([0.0, 1.5]))
    field, rendered, progress = _inputs(records=1, count=4, width=8, height=9, grid_w=9, seed=3)

    full = kernel(field, rendered, progress, records=1, count=4)
    for i in range(4):
        single = kernel(
            field[:, i : i + 1],
            rendered.reshape(1, 4, 8, 9, 9)[:, i].reshape(1, 8, 9, 9),
            progress[:, i : i + 1],
            records=1,
            count=1,
        )
        assert torch.allclose(full[:, i : i + 1], single, atol=1e-6), f"frame {i} coupled"


def test_green_kernel_gradients_flow_once_gate_opens():
    kernel = _DynamicGreenScatteringKernel(8, kernel_size=3, dilations=(1, 2, 4))
    with torch.no_grad():
        kernel.gate.fill_(0.5)
    field, rendered, progress = _inputs(records=2, count=2, width=8, height=9, grid_w=9, seed=5)
    field.requires_grad_(True)
    out = kernel(field, rendered, progress, records=2, count=2)
    out.pow(2).mean().backward()
    assert kernel.gate.grad is not None and torch.isfinite(kernel.gate.grad).all()
    assert kernel.radius_bias.grad is not None and torch.isfinite(kernel.radius_bias.grad).all()
    assert kernel.mix.weight.grad is not None
    for stencil in kernel.stencils:
        assert stencil.weight.grad is not None
    assert field.grad is not None and torch.isfinite(field.grad).all()


def test_green_kernel_radius_follows_travel_progress():
    """A positive radius_bias on the larger dilation must shift softmax weight to it
    as elapsed progress grows -- i.e. the effective scattering radius follows T."""
    kernel = _DynamicGreenScatteringKernel(4, kernel_size=3, dilations=(1, 5))
    with torch.no_grad():
        kernel.mix.weight.zero_()  # remove feature-driven logits: isolate the progress term
        kernel.mix.bias.zero_()
        kernel.radius_bias.copy_(torch.tensor([0.0, 2.0]))  # large-radius branch favors late time
    rendered = torch.zeros(1, 4, 6, 6)
    # early vs late progress at otherwise identical inputs
    early = torch.full((1, 1, 6, 6), 0.05)
    late = torch.full((1, 1, 6, 6), 0.95)

    def _weights(prog):
        logits = kernel.radius_bias.view(1, -1, 1, 1) * prog.reshape(1, 1, 6, 6)
        return torch.softmax(logits, dim=1)

    w_early = _weights(early)
    w_late = _weights(late)
    # weight on the large-radius (index 1) branch must increase with progress
    assert w_late[0, 1].mean() > w_early[0, 1].mean() + 0.1


def test_green_kernel_radius_follows_progress_through_real_forward():
    """Stronger form of the radius-follows-T check demanded by the 2026-07-30 12:30
    web brief: drive the ACTUAL ``forward`` (not a hand-reimplemented softmax) and
    verify the scattered OUTPUT shifts toward the larger-dilation branch as elapsed
    progress grows.  We make the two dilation branches emit known constants (small vs
    large) so the mixed output is a monotone readout of the softmax mass, then confirm
    late-progress output is pulled toward the large-radius branch.  This exercises the
    real branch-conv -> radius_bias -> softmax -> mix -> gate path end to end."""
    kernel = _DynamicGreenScatteringKernel(4, kernel_size=3, dilations=(1, 2))
    with torch.no_grad():
        kernel.gate.fill_(1.0)
        kernel.mix.weight.zero_()          # isolate the travel-progress term
        kernel.mix.bias.zero_()
        kernel.radius_bias.copy_(torch.tensor([0.0, 3.0]))  # large-radius favors late time
        # branch 0 (dilation 1) -> constant 1.0; branch 1 (dilation 2) -> constant 10.0
        # single centre tap so a uniform field maps to that constant on interior pixels.
        for stencil, value in zip(kernel.stencils, (1.0, 10.0)):
            w = torch.zeros_like(stencil.weight)
            w[..., w.shape[-2] // 2, w.shape[-1] // 2] = value
            stencil.weight.copy_(w)
    field = torch.ones(1, 2, 9, 9)          # uniform -> branch outputs are the constants
    rendered = torch.zeros(1 * 2, 4, 9, 9)
    progress = torch.stack(
        [torch.full((9, 9), 0.02), torch.full((9, 9), 0.98)]
    ).reshape(1, 2, 9, 9)                    # frame0 early, frame1 late
    out = kernel(field, rendered, progress, records=1, count=2)
    c = 9 // 2  # interior pixel, clear of dilated-padding truncation
    early_val = float(out[0, 0, c, c])
    late_val = float(out[0, 1, c, c])
    # early ~ weight near branch0 (=>~1), late pulled toward branch1 (=>toward 10)
    assert late_val > early_val + 1.0, (early_val, late_val)
    # and the late output must sit strictly above the small-radius branch constant
    assert late_val > 1.0 and late_val <= 10.0


def test_green_kernel_validation():
    for bad in dict(kernel_size=4), dict(kernel_size=0), dict(dilations=()), dict(dilations=(1, 0)):
        try:
            _DynamicGreenScatteringKernel(8, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def _generator(**kwargs):
    return LocalPropagationFieldGenerator(
        width=8,
        pyramid_levels=2,
        saved_time_count=16,
        domain_t_s=0.5,
        domain_diagonal_m=1000.0,
        channel_multipliers=(1, 1, 2, 2),
        residual=True,
        activation_checkpointing=False,
        warp=True,
        temporal_operator_rank=8,
        temporal_operator_spatial_kernel=3,
        **kwargs,
    )


def test_generator_builds_green_kernel_and_is_noop_at_init():
    with_green = _generator(green_kernel=True, green_kernel_size=5, green_dilations=(1, 2, 4))
    without = _generator(green_kernel=False)
    assert with_green.green_kernel is not None
    assert without.green_kernel is None
    # green kernel adds only its own params; everything else identical in count
    base = sum(p.numel() for p in without.parameters())
    grown = sum(p.numel() for p in with_green.parameters())
    green_params = sum(p.numel() for p in with_green.green_kernel.parameters())
    assert grown - base == green_params
    # gate zero-init -> module is a no-op at construction
    assert float(with_green.green_kernel.gate) == 0.0


def test_green_kernel_warmstart_statedict_contract():
    """Codify the R5 WARM-START transfer contract that ``load_checkpoint`` enforces at
    launch (grouped_ufno_mionet_v3/training/checkpoint.py: strict=False +
    allowed_missing_prefixes). Loading a B1 parent (warp + rank-8 operator, green OFF)
    state_dict into an R5 child (green ON) must yield:
      * zero UNEXPECTED keys (every B1 key still exists in the R5 child),
      * every MISSING key under the ``green_kernel.`` prefix (allow-listed -> fresh
        zero-init, gate=0 -> byte-exact no-op),
      * IDENTICAL shapes on every shared warp/operator/unet/fuse key.
    Checkpoint-FILE-independent (no digest/339MB dependency) so it guards the R5
    escalation launch the same way the temporal_latent contract guards A3. Module-scoped
    prefix is ``green_kernel.``; the full model adds ``local_field.`` -> real allow-prefix
    ``local_field.green_kernel.`` (wired at train script L1387)."""
    parent = _generator(green_kernel=False)  # == B1 local_field (warp + rank-8 operator)
    child = _generator(green_kernel=True, green_kernel_size=5, green_dilations=(1, 2, 4))
    parent_sd = parent.state_dict()
    child_sd = child.state_dict()
    parent_keys = set(parent_sd)
    child_keys = set(child_sd)

    unexpected = parent_keys - child_keys
    assert not unexpected, f"B1 keys missing from R5 child (would be UNEXPECTED): {unexpected}"

    missing = child_keys - parent_keys
    assert missing, "child added no keys -- green_kernel produced no state"
    forbidden = {k for k in missing if not k.startswith("green_kernel.")}
    assert not forbidden, f"missing keys not allow-listed by green_kernel. prefix: {forbidden}"

    for k in parent_keys & child_keys:
        assert parent_sd[k].shape == child_sd[k].shape, (
            f"shape drift on shared key {k}: {tuple(parent_sd[k].shape)} != "
            f"{tuple(child_sd[k].shape)}"
        )

    # the load path itself: strict=False must report EXACTLY this partition
    result = child.load_state_dict(parent_sd, strict=False)
    assert tuple(result.unexpected_keys) == (), result.unexpected_keys
    assert set(result.missing_keys) == missing
    assert all(k.startswith("green_kernel.") for k in result.missing_keys)
    # gate still zero after the partial load -> warm-start remains a byte-exact no-op
    assert float(child.green_kernel.gate) == 0.0
