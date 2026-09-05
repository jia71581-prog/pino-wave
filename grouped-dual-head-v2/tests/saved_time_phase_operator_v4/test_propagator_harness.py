"""CPU contract tests for the B2 harness (propagator_harness), GPU-free.

Pins the properties any B2 trainer relies on:
  1. eval metric == the fixed protocol metric coarse_lwc84.complete_field_relative_l2
     (per-record joint space-time relative L2, mean over records) -- to <1e-9.
     This is the anti-cheat guarantee: B2 is scored on the SAME protocol as warp/opt16.
  2. gate==0 warm-start: rollout == base_seq bit-exact => reported aggregate ==
     the frozen warp baseline aggregate (B2 improves a real baseline, never a fake one).
  3. gate>0 is live and the sequence loss backprops to ALL propagator params.
  4. joint vs per_frame_uniform weighting both finite/differentiable and genuinely differ.
  5. causal prefix invariance carries through the objective (frame k uses only base_0..k).
"""
import torch

from saved_time_phase_operator_v4.coarse_lwc84 import complete_field_relative_l2
from saved_time_phase_operator_v4.propagator import CausalSemigroupPropagator
from saved_time_phase_operator_v4.propagator_harness import (
    AnchoredSequenceObjective,
    aggregate_relative_l2,
    per_frame_relative_l2,
    sequence_loss,
)


def _make(gate_init=0.03, **kw):
    torch.manual_seed(0)
    return CausalSemigroupPropagator(
        state_channels=2, cond_channels=6, width=32, spectral_rank=16,
        modes=8, depth=3, gate_init=gate_init, activation_checkpointing=False, **kw
    )


def _seqs(b=3, k=6, z=24, x=24):
    torch.manual_seed(2)
    base = torch.randn(b, k, 1, z, x)
    target = base + 0.3 * torch.randn(b, k, 1, z, x)  # correlated -> realistic ~O(0.3) rel L2
    cond = torch.randn(b, 6, z, x)
    return base, cond, target


def test_eval_metric_equals_protocol_metric():
    # aggregate_relative_l2 must reproduce complete_field_relative_l2 per record, mean over records.
    base, _, target = _seqs()
    ours = aggregate_relative_l2(base, target)
    per_record = [
        complete_field_relative_l2(base[b, :, 0], target[b, :, 0])
        for b in range(base.shape[0])
    ]
    protocol = sum(per_record) / len(per_record)
    assert abs(ours - protocol) < 1e-9, (ours, protocol)


def test_gate0_warmstart_reports_exact_base_aggregate():
    m = _make(gate_init=0.0).eval()
    base, cond, target = _seqs()
    obj = AnchoredSequenceObjective(m)
    with torch.no_grad():
        out = obj.step(base, cond, target)
    # gate=0 => rollout is base_seq bit-exact => reported aggregate == base aggregate exactly.
    assert torch.equal(out["prediction"], base)
    assert out["aggregate_relative_l2"] == out["base_aggregate_relative_l2"]
    assert out["aggregate_relative_l2"] == aggregate_relative_l2(base, target)


def test_gate_live_loss_backprops_all_params():
    m = _make(gate_init=0.03).train()
    base, cond, target = _seqs()
    obj = AnchoredSequenceObjective(m, weighting="joint")
    out = obj.step(base, cond, target)
    assert torch.isfinite(out["loss"]).all()
    out["loss"].backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_weightings_differ_and_are_finite():
    base, _, target = _seqs()
    pred = base + 0.1 * torch.randn_like(base)
    pred.requires_grad_(True)
    lj = sequence_loss(pred, target, weighting="joint")
    lu = sequence_loss(pred, target, weighting="per_frame_uniform")
    assert torch.isfinite(lj) and torch.isfinite(lu)
    assert abs(float(lj) - float(lu)) > 1e-6, "weightings must genuinely differ"
    lu.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_per_frame_gives_lowenergy_frames_weight():
    # a low-energy late frame with large relative error is nearly invisible to the
    # joint aggregate but dominant under per_frame_uniform -> proves the lever exists.
    b, k, z, x = 1, 4, 8, 8
    target = torch.zeros(b, k, 1, z, x)
    target[:, :3] = 1.0            # frames 0..2 high energy
    target[:, 3] = 0.01            # frame 3 low energy (late)
    pred = target.clone()
    pred[:, 3] = -0.01             # 200% relative error on the low-energy frame only
    joint = float(sequence_loss(pred, target, weighting="joint"))
    uni = float(sequence_loss(pred, target, weighting="per_frame_uniform"))
    assert joint < 0.02, joint          # invisible to energy-weighted aggregate
    assert uni > 0.4, uni               # dominant under uniform per-frame weighting


def test_objective_causal_prefix_invariance():
    m = _make(gate_init=0.5).eval()
    base, cond, target = _seqs(k=8)
    obj = AnchoredSequenceObjective(m)
    with torch.no_grad():
        full = obj.rollout(base, cond)
        prefix = obj.rollout(base[:, :5].contiguous(), cond)
    assert torch.allclose(full[:, :5], prefix, atol=1e-6, rtol=0)
    # and the metric restricted to the prefix matches
    a_full_prefix = aggregate_relative_l2(full[:, :5], target[:, :5])
    a_prefix = aggregate_relative_l2(prefix, target[:, :5])
    assert abs(a_full_prefix - a_prefix) < 1e-9
