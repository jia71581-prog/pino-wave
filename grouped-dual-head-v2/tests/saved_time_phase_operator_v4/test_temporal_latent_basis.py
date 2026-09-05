"""A3: query-invariant continuous temporal LATENT BASIS (class-A coarse-field lever).

The aggregate ceiling bound (ARCH_CANDIDATES ~12:30) showed the aggregate floor is
gated by the UNIFORM coarse-field limit (~0.126) that class-B scattering correctors
(warp / temporal_operator / green_kernel) provably cannot touch: uniform media have
no scattered energy for them to fix.  A3 is the class-A response -- it enriches the
COARSE field itself with a low-rank continuous temporal basis so a fixed (x,z) can
encode multi-arrival / late-time structure a single U-Net render cannot.

These tests pin the CONTRACT the escalation must not break:
  * zero-init gate -> exact no-op at warm-start (residual is identically zero),
  * per-frame query invariance BY CONSTRUCTION (a frame's residual depends only on
    its own scalar time + the record's time-INDEPENDENT conditioning, never on which
    other saved times share the batch -- the A1 blocker-1 the design resolves),
  * anchors are query-independent: the same conditioning yields the same anchor bank
    regardless of the time set (no queried-index-axis mixing),
  * gradients flow to anchors / time_trunk / gate once the gate opens,
  * a generator built with temporal_latent_basis=True constructs the module, adds
    ONLY its own params, and is an exact no-op at init (warm-start byte-reproducible).
"""
from __future__ import annotations

import math

import pytest

import torch

from saved_time_phase_operator_v4.local_field import (
    LocalPropagationFieldGenerator,
    _ContinuousTemporalLatentBasis,
)
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


def _inputs(*, records, count, width, height, grid_w, seed=0):
    torch.manual_seed(seed)
    conditioning = torch.randn(records, width, height, grid_w)
    time_s = torch.rand(records, count) * 0.4  # within a 0.5 s domain
    # source_parameters columns: [.., .., frequency, onset, ..]; module reads [2],[3].
    source_parameters = torch.zeros(records, 5)
    source_parameters[:, 2] = 15.0 + torch.rand(records) * 10.0  # frequency (Hz)
    source_parameters[:, 3] = torch.rand(records) * 0.05         # onset (s)
    return conditioning, time_s, source_parameters


def test_temporal_latent_is_exact_noop_at_warmstart():
    """gate is zero-init -> the temporal-basis residual is identically zero."""
    basis = _ContinuousTemporalLatentBasis(8, rank=8, harmonics=4)
    conditioning, time_s, source = _inputs(records=2, count=3, width=8, height=11, grid_w=13)
    residual = basis(conditioning, time_s, source, domain_t_s=0.5)
    assert residual.shape == (2, 3, 11, 13)
    assert torch.count_nonzero(residual) == 0
    assert torch.allclose(residual, torch.zeros_like(residual))


def test_temporal_latent_query_invariance():
    """Frame i's residual is independent of which other frames share the batch."""
    basis = _ContinuousTemporalLatentBasis(8, rank=6, harmonics=3)
    with torch.no_grad():
        basis.gate.fill_(0.8)  # open the gate so the residual is non-trivial
    conditioning, time_s, source = _inputs(records=1, count=5, width=8, height=9, grid_w=9, seed=3)

    full = basis(conditioning, time_s, source, domain_t_s=0.5)
    for i in range(5):
        single = basis(
            conditioning,
            time_s[:, i : i + 1],
            source,
            domain_t_s=0.5,
        )
        assert torch.allclose(full[:, i : i + 1], single, atol=1e-6), f"frame {i} coupled"


def test_temporal_latent_permutation_equivariance_over_time():
    """Reordering the queried times permutes the output identically -- there is no
    ordering-dependent state (a stronger form of query invariance)."""
    basis = _ContinuousTemporalLatentBasis(8, rank=4, harmonics=2)
    with torch.no_grad():
        basis.gate.fill_(0.5)
    conditioning, time_s, source = _inputs(records=2, count=4, width=8, height=7, grid_w=7, seed=9)
    perm = torch.tensor([3, 0, 2, 1])

    out = basis(conditioning, time_s, source, domain_t_s=0.5)
    out_perm = basis(conditioning, time_s[:, perm], source, domain_t_s=0.5)
    assert torch.allclose(out_perm, out[:, perm], atol=1e-6)


def test_temporal_latent_anchors_are_query_independent():
    """The anchor bank L_m(x,z) is built from conditioning ALONE -- identical for any
    time set (it carries no queried-index-axis information)."""
    basis = _ContinuousTemporalLatentBasis(8, rank=5, harmonics=2)
    conditioning, _, _ = _inputs(records=2, count=1, width=8, height=9, grid_w=9, seed=7)
    anchors_a = basis.anchor_projection(conditioning)
    anchors_b = basis.anchor_projection(conditioning)
    assert anchors_a.shape == (2, 5, 9, 9)
    assert torch.allclose(anchors_a, anchors_b)


def test_temporal_latent_gradients_flow_once_gate_opens():
    basis = _ContinuousTemporalLatentBasis(8, rank=6, harmonics=3)
    with torch.no_grad():
        basis.gate.fill_(0.5)
    conditioning, time_s, source = _inputs(records=2, count=2, width=8, height=9, grid_w=9, seed=5)
    conditioning.requires_grad_(True)
    out = basis(conditioning, time_s, source, domain_t_s=0.5)
    out.pow(2).mean().backward()
    assert basis.gate.grad is not None and torch.isfinite(basis.gate.grad).all()
    for name, param in basis.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
    assert conditioning.grad is not None and torch.isfinite(conditioning.grad).all()


def test_temporal_latent_scales_with_rank_normalisation():
    """Residual magnitude stays O(1) in rank thanks to the 1/sqrt(M) scaling -- the
    einsum sums M anchor contributions, so without normalisation it would grow ~M."""
    conditioning, time_s, source = _inputs(records=1, count=3, width=8, height=8, grid_w=8, seed=2)
    magnitudes = {}
    for rank in (4, 16, 64):
        basis = _ContinuousTemporalLatentBasis(8, rank=rank, harmonics=3)
        with torch.no_grad():
            basis.gate.fill_(1.0)
            # deterministic unit anchors + unit coefficients isolate the M-scaling
            for p in basis.anchor_projection.parameters():
                p.zero_()
            basis.anchor_projection[-1].bias.fill_(1.0)  # anchors == 1 everywhere
        out = basis(conditioning, time_s, source, domain_t_s=0.5)
        magnitudes[rank] = float(out.abs().mean())
    # 16x more anchors must NOT inflate the residual by ~16x (would if unnormalised)
    assert magnitudes[64] < 3.0 * magnitudes[4]


def test_temporal_latent_validation():
    for bad in dict(rank=0), dict(harmonics=0):
        try:
            _ContinuousTemporalLatentBasis(8, **{"rank": 8, "harmonics": 4, **bad})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    try:
        _ContinuousTemporalLatentBasis(0, rank=8, harmonics=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for width=0")


def test_temporal_latent_negative_gate_init_rejected():
    for bad in (-0.01, -1.0):
        try:
            _ContinuousTemporalLatentBasis(8, rank=8, harmonics=4, gate_init=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for gate_init={bad}")


def test_temporal_latent_gate_init_breaks_cold_start_deadlock():
    """gate_init > 0 makes the basis contribute at load AND delivers real gradient to
    the anchor/time trunks -- breaking the zero-init deadlock (with gate == 0 on a
    frozen parent the trunks get dL/d(trunk) = gate * ... == 0 and never align)."""
    basis = _ContinuousTemporalLatentBasis(8, rank=6, harmonics=3, gate_init=0.05)
    assert float(basis.gate) == pytest.approx(0.05)
    conditioning, time_s, source = _inputs(records=2, count=3, width=8, height=9, grid_w=9, seed=5)
    residual = basis(conditioning, time_s, source, domain_t_s=0.5)
    assert residual.abs().max() > 1e-6            # contributes at load
    residual.square().mean().backward()
    first_conv = basis.anchor_projection[0]
    assert first_conv.weight.grad is not None and first_conv.weight.grad.abs().max() > 0.0
    assert basis.gate.grad is not None and basis.gate.grad.abs().max() > 0.0


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
        **kwargs,
    )


def test_generator_builds_temporal_latent_and_is_noop_at_init():
    with_latent = _generator(
        temporal_latent_basis=True, temporal_latent_rank=8, temporal_latent_harmonics=4
    )
    without = _generator(temporal_latent_basis=False)
    assert with_latent.temporal_latent is not None
    assert without.temporal_latent is None
    # A3 adds ONLY its own params; everything else identical in count (single factor)
    base = sum(p.numel() for p in without.parameters())
    grown = sum(p.numel() for p in with_latent.parameters())
    latent_params = sum(p.numel() for p in with_latent.temporal_latent.parameters())
    assert grown - base == latent_params
    # gate zero-init -> module is a no-op at construction (warm-start byte-exact)
    assert float(with_latent.temporal_latent.gate) == 0.0


def test_generator_latent_params_are_prefixed_for_lr_routing():
    """The staged optimizer routes by name prefix; every A3 param must live under
    ``temporal_latent.`` so it lands in its own LR group (fresh adapter warmup)."""
    gen = _generator(temporal_latent_basis=True, temporal_latent_rank=4)
    latent_names = {n for n, _ in gen.named_parameters() if ".temporal_latent." in f".{n}"}
    assert latent_names, "no temporal_latent params found"
    for name, _ in gen.named_parameters():
        param = dict(gen.named_parameters())[name]
        if any(param is p for p in gen.temporal_latent.parameters()):
            assert name.startswith("temporal_latent."), name


def test_temporal_latent_query_invariance_at_launch_rank():
    """Query invariance is claimed BY CONSTRUCTION, but the evidence-favored launch
    config is rank=32 / harmonics=4 (SVD rank diagnostic 2026-07-30: late-time fields
    need 17-33 modes), not the rank<=6 exercised elsewhere. Pin the contract at the
    exact rank that will hit the GPU so an escalation launch cannot silently break it."""
    basis = _ContinuousTemporalLatentBasis(8, rank=32, harmonics=4)
    with torch.no_grad():
        basis.gate.fill_(0.8)
    conditioning, time_s, source = _inputs(records=1, count=5, width=8, height=9, grid_w=9, seed=7)
    full = basis(conditioning, time_s, source, domain_t_s=0.5)
    for i in range(5):
        single = basis(conditioning, time_s[:, i : i + 1], source, domain_t_s=0.5)
        assert torch.allclose(full[:, i : i + 1], single, atol=1e-6), f"frame {i} coupled"


def test_temporal_latent_warmstart_statedict_contract():
    """Codify the WARM-START transfer contract that ``load_checkpoint`` enforces at
    launch (grouped_ufno_mionet_v3/training/checkpoint.py: strict=False +
    allowed_missing_prefixes). Loading a WARP-ONLY parent state_dict into a
    warp+temporal_latent child must yield:
      * zero UNEXPECTED keys (every parent key still exists in the child),
      * every MISSING key under the ``temporal_latent.`` prefix (allow-listed ->
        fresh zero-init, gate=0 -> byte-exact no-op),
      * IDENTICAL shapes on every shared key (no silent fan-in drift like the
        multiscale_fuse mismatch caught during the manual dry-run).
    This is checkpoint-FILE-independent (no digest/339MB dependency) so it guards
    every future escalation warm-start, not just the one launch verified by hand.
    The module-scoped prefix is ``temporal_latent.``; the full model adds
    ``local_field.`` so the real allow-prefix is ``local_field.temporal_latent.``."""
    # Mirror the REAL launch: warp-on parent -> warp+temporal_latent child (single factor).
    # rank/harmonics match the evidence-favored launch config; pyramid_levels=2 for speed
    # (the key-diff invariant is pyramid-independent).
    common = dict(warp=True, warp_max_shift_cells=8.0)
    parent = _generator(temporal_latent_basis=False, **common)
    child = _generator(
        temporal_latent_basis=True,
        temporal_latent_rank=32,
        temporal_latent_harmonics=4,
        **common,
    )
    parent_sd = parent.state_dict()
    child_sd = child.state_dict()
    parent_keys = set(parent_sd)
    child_keys = set(child_sd)

    # (1) no unexpected keys: nothing in the parent checkpoint is absent from the child
    unexpected = parent_keys - child_keys
    assert not unexpected, f"parent keys missing from child (would be UNEXPECTED): {unexpected}"

    # (2) every added (missing-at-load) key is under the allow-listed temporal_latent prefix
    missing = child_keys - parent_keys
    assert missing, "child added no keys -- temporal_latent produced no state"
    forbidden = {k for k in missing if not k.startswith("temporal_latent.")}
    assert not forbidden, f"missing keys not allow-listed by temporal_latent. prefix: {forbidden}"

    # (3) exact shape match on every shared key -- this is what strict=False silently skips
    #     and load_state_dict would ERROR on; assert it explicitly.
    for k in parent_keys & child_keys:
        assert parent_sd[k].shape == child_sd[k].shape, (
            f"shape drift on shared key {k}: {tuple(parent_sd[k].shape)} != "
            f"{tuple(child_sd[k].shape)}"
        )

    # the load path itself: strict=False must report EXACTLY this partition
    result = child.load_state_dict(parent_sd, strict=False)
    assert tuple(result.unexpected_keys) == (), result.unexpected_keys
    assert set(result.missing_keys) == missing
    assert all(k.startswith("temporal_latent.") for k in result.missing_keys)
    # gate still zero after the partial load -> warm-start remains a byte-exact no-op
    assert float(child.temporal_latent.gate) == 0.0


def test_launch_config_temporal_latent_param_counts_are_pinned():
    """Lock the EXACT param cost asserted in the A3 configs + NEXT_ACTION so the
    single-factor / warm-start claims stay honest. Full width=128 generator (matches the
    real model; allocation only, no forward). rank8=12,505, rank32=51,937 (<1% of the
    ~5.58M base), gate=0.0 -> warm-start byte-exact no-op."""
    def _full(**kw):
        return LocalPropagationFieldGenerator(
            width=128, pyramid_levels=2, saved_time_count=401, domain_t_s=0.5,
            domain_diagonal_m=1000.0, channel_multipliers=(1, 1, 2, 2),
            residual=True, activation_checkpointing=False, **kw,
        )
    base = _full(temporal_latent_basis=False)
    n_base = sum(p.numel() for p in base.parameters())
    for rank, expected in ((8, 12_505), (32, 51_937)):
        gen = _full(temporal_latent_basis=True, temporal_latent_rank=rank, temporal_latent_harmonics=4)
        n_latent = sum(p.numel() for p in gen.temporal_latent.parameters())
        assert n_latent == expected, f"rank={rank}: {n_latent} != {expected}"
        # single factor: the ONLY new params are the temporal_latent bank
        assert sum(p.numel() for p in gen.parameters()) - n_base == n_latent
        assert float(gen.temporal_latent.gate) == 0.0
        assert n_latent < 0.01 * n_base  # <1% of base -> capacity lever, not a rewrite


# --- A3-WARM launch de-risk (R8+1): the module-level deadlock-break test above proves the
# isolated basis learns; these prove it end-to-end THROUGH the real forward (render -> warp ->
# temporal_latent), at the exact launch gate (0.03) with warp ON, and quantify the warm-start
# perturbation the config comment claims is "~1e-3, negligible" (integrity: verify the number). ---

def _a3_forward_inputs(gen, *, records, count, medium_count, height, grid_w, seed=0):
    """Shape-valid encodings for LocalPropagationFieldGenerator.forward (mirrors the warp test)."""
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
    return dict(
        velocity_mps=torch.randn(records, 1, height, grid_w),
        medium=medium, source=source, source_parameters=source_parameters,
        record_to_medium=torch.arange(records) % medium_count,
        time_s=(torch.rand(records, count) * 0.4 + 0.1).sort(dim=1).values,
        travel=travel,
        saved_time_indices=torch.randint(0, 401, (records, count)),
    )


def _a3warm_generator(*, gate_init):
    """The A3-warm launch config in miniature: warp ON + temporal_latent rank32 (single factor
    vs running A3 is ONLY gate_init 0.0 -> 0.03)."""
    return LocalPropagationFieldGenerator(
        width=8, pyramid_levels=2, saved_time_count=401, domain_t_s=1.0,
        domain_diagonal_m=2828.0, residual=True, activation_checkpointing=False,
        warp=True, warp_max_shift_cells=8.0,
        temporal_latent_basis=True, temporal_latent_rank=32, temporal_latent_harmonics=4,
        adapter_gate_init=gate_init,
    )


def test_full_forward_a3_gate0_is_exact_noop_vs_disabled():
    """Within the REAL forward (warp ON), the gate=0 temporal_latent adds nothing -> output
    equals the identical model with temporal_latent disabled (the running-A3 warm-start path)."""
    gen = _a3warm_generator(gate_init=0.0)
    gen.eval()
    inputs = _a3_forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16)
    with torch.no_grad():
        out = gen(**inputs)
        saved = gen.temporal_latent
        gen.temporal_latent = None            # emulate the warp-only parent forward
        out_disabled = gen(**inputs)
        gen.temporal_latent = saved
    assert out.shape == (2, 3, 16, 16)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, out_disabled, atol=1e-6)


def test_full_forward_a3warm_gate003_is_live_and_perturbation_is_small():
    """At the launch gate (0.03), temporal_latent changes the real forward output (it is wired
    LIVE past the warp), AND the warm-start perturbation is small.

    NOTE on the metric: a FRESH residual generator's output head is zero-init, so with
    temporal_latent DISABLED it emits an identically-zero field (||out_base||==0 here) -- a
    RELATIVE perturbation is therefore undefined under random init.  The meaningful, honest
    quantity is the ABSOLUTE magnitude of the gated residual (max|out_warm|).  The real launch
    loads the TRAINED warp_r1 parent whose field is O(1), so this absolute residual (~1e-3 scale)
    IS the launch perturbation, and ~1e-3 / O(1) ~ sub-percent -- matching the config comment."""
    gen = _a3warm_generator(gate_init=0.03)
    gen.eval()
    inputs = _a3_forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16)
    with torch.no_grad():
        out_warm = gen(**inputs)
        saved = gen.temporal_latent
        gen.temporal_latent = None
        out_base = gen(**inputs)
        gen.temporal_latent = saved
    assert torch.isfinite(out_warm).all()
    # LIVE: opening the gate to 0.03 must change the output (not bypassed by the warp)
    assert not torch.allclose(out_warm, out_base, atol=1e-6)
    abs_max = float(out_warm.abs().max())
    print(f"\n[A3-warm] gate=0.03 absolute gated-residual: max={abs_max:.4e} "
          f"mean={float(out_warm.abs().mean()):.4e} (||out_base||={float(out_base.norm()):.2e})")
    # gate=0.03 * random-init anchors on an O(1) field -> ~1e-3-scale residual: small but nonzero,
    # recoverable by the learned gate.  Guard against a silent explosion (bad init/normalisation).
    assert 1e-6 < abs_max < 0.05, f"gated residual max={abs_max:.4e} outside the expected 1e-3 band"


def test_full_forward_a3warm_gradient_reaches_latent_trunks_through_warp():
    """End-to-end deadlock-break: with gate=0.03, backprop through the FULL pipeline
    (render -> warp -> temporal_latent) delivers real gradient to BOTH the anchor_projection
    and the time_trunk. This is what the running A3 (gate=0) never gets."""
    gen = _a3warm_generator(gate_init=0.03)
    gen.train()
    inputs = _a3_forward_inputs(gen, records=2, count=3, medium_count=2, height=16, grid_w=16, seed=1)
    out = gen(**inputs)
    out.square().mean().backward()
    anchor_w = gen.temporal_latent.anchor_projection[0].weight
    # time_trunk is nn.Sequential(Linear, GELU, Linear); grab the first Linear weight
    time_trunk_w = next(p for n, p in gen.temporal_latent.time_trunk.named_parameters()
                        if n.endswith("weight"))
    gate_g = gen.temporal_latent.gate.grad
    assert anchor_w.grad is not None and anchor_w.grad.abs().max() > 0.0, "anchor_projection got no grad"
    assert time_trunk_w.grad is not None and time_trunk_w.grad.abs().max() > 0.0, "time_trunk got no grad"
    assert gate_g is not None and gate_g.abs().max() > 0.0, "gate got no grad"
    assert torch.isfinite(anchor_w.grad).all() and torch.isfinite(time_trunk_w.grad).all()
