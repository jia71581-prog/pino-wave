from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from grouped_ufno_mionet_v3.data.index import V3DataManifest, V3RecordIndex
from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from saved_time_phase_operator_v4.instance_adaptation.feature_modulation import (
    EarlyFeatureModulator,
)
from scripts.train_v5_onset_conditioner import build_onset_episodes
from scripts.train_v5_feature_meta import (
    build_balanced_meta_episodes,
    metric_aligned_energy_weights,
)
from scripts import evaluate_v5_feature_meta as feature_evaluator


def _manifest():
    records = tuple(
        V3RecordIndex(i, f"sample-{i}", f"medium-{i}", "sha", "train", i, family)
        for i, family in enumerate(("uniform", "layered", "marmousi"))
    )
    return V3DataManifest(
        schema="test", source_path="/tmp/test.h5", source_file_sha256="sha",
        source_manifest_sha256="", source_config_sha256="", allowed_medium_types=("uniform", "layered", "marmousi"),
        excluded_medium_types=(), counts_before={}, counts_after={"train": 3}, indices_by_split={"train": (0, 1, 2)},
        records=records, time_s=(0.0, 0.1), x_m=(0.0,), z_m=(0.0,), digest="digest",
    )


def test_conditioner_episode_uses_only_train_group_ids():
    episodes = build_onset_episodes(_manifest(), split="train", seed=17)
    assert all(episode.split == "train" for episode in episodes)
    assert len({episode.group_id for episode in episodes}) == len(episodes)


def test_conditioner_episode_order_is_reproducible():
    left = build_onset_episodes(_manifest(), split="train", seed=17)
    right = build_onset_episodes(_manifest(), split="train", seed=17)
    assert tuple(item.sample_id for item in left) == tuple(item.sample_id for item in right)


def test_balanced_meta_episodes_respect_background_cache_support():
    manifest = _manifest()
    allowed = {"sample-0", "sample-1", "sample-2"}
    episodes = build_balanced_meta_episodes(
        manifest,
        split="train",
        per_family=1,
        seed=17,
        allowed_sample_ids=allowed,
    )
    assert {episode.sample_id for episode in episodes} == allowed


def test_streaming_energy_weights_match_joint_relative_l2_gradient():
    target = torch.randn(2, 7, 3, 4)
    direct_prediction = torch.randn_like(target, requires_grad=True)
    direct = (
        (direct_prediction - target).flatten(1).norm(dim=-1)
        / target.flatten(1).norm(dim=-1).clamp_min(1.0e-8)
    ).mean()
    direct.backward()

    streamed_prediction = direct_prediction.detach().clone().requires_grad_(True)
    with torch.no_grad():
        error_energy = (
            (streamed_prediction - target).square().flatten(1).sum(dim=-1)
        )
        target_energy = target.square().flatten(1).sum(dim=-1)
        value, weights = metric_aligned_energy_weights(
            error_energy, target_energy
        )
    for start in range(0, target.shape[1], 3):
        error = streamed_prediction[:, start : start + 3] - target[:, start : start + 3]
        piece_energy = error.square().flatten(1).sum(dim=-1)
        (weights * piece_energy).sum().backward()

    torch.testing.assert_close(value, direct.detach())
    torch.testing.assert_close(
        streamed_prediction.grad, direct_prediction.grad, atol=1.0e-6, rtol=1.0e-5
    )


def test_onset_only_refinement_uses_two_true_frames_without_bridge(monkeypatch):
    class Audit:
        def __init__(self):
            self.reads = []

        def read(self, indices):
            self.reads.append(tuple(indices))

        def payload(self):
            return {"reads": self.reads}

    audit = Audit()
    record = SimpleNamespace(
        audit=audit,
        observed_indices=(1, 2),
        time_s=torch.arange(4, dtype=torch.float32),
        observed_wavefield=torch.ones(2, 1, 1),
    )
    adapter = SimpleNamespace(
        conditioner=SimpleNamespace(latent_dim=1),
        adapter_parameters=lambda: (),
    )

    def prediction(
        _adapter, _normalizer, _record, _device, *, time_s, latent_delta=None, **_kwargs
    ):
        value = torch.zeros(1, len(time_s), 1, 1)
        if latent_delta is not None:
            value = value + latent_delta.reshape(1, 1, 1, 1)
        return value

    monkeypatch.setattr(feature_evaluator, "_predict_feature", prediction)
    delta, info = feature_evaluator._refine_latent_onset_only(
        adapter,
        None,
        record,
        torch.device("cpu"),
        steps=2,
        learning_rate=0.1,
    )
    assert audit.reads == [(1, 2)]
    assert info["mode"] == "onset_only_no_wavefield_propagator"
    assert info["physical_wavefield_propagator"] is False
    assert info["synthetic_bridge"] is False
    assert float(delta.norm()) > 0.0


def test_zero_modulation_gate_is_exact_parent_feature_identity():
    modulator = EarlyFeatureModulator(
        latent_dim=2,
        pyramid_widths=(2,),
        token_width=3,
        rank_width=1,
        max_scale=0.1,
        max_bias=0.05,
    )
    with torch.no_grad():
        modulator.affine.weight.fill_(0.2)
        modulator.affine.bias.fill_(0.1)
    medium = MediumEncoding(
        pyramid=(torch.randn(1, 2, 4, 4),),
        tokens=torch.randn(1, 5, 3),
        token_positions=torch.randn(1, 5, 2),
        rank=torch.randn(1, 1),
    )
    output = modulator(
        medium,
        torch.randn(1, 2),
        torch.zeros(1, dtype=torch.long),
        modulation_gate=torch.zeros(1),
    )
    torch.testing.assert_close(output.pyramid[0], medium.pyramid[0])
    torch.testing.assert_close(output.tokens, medium.tokens)
    torch.testing.assert_close(output.rank, medium.rank)


def test_residual_trust_gate_starts_from_parent_and_uses_no_propagator(monkeypatch):
    class Audit:
        def __init__(self):
            self.reads = []

        def read(self, indices):
            self.reads.append(tuple(indices))

        def payload(self):
            return {"reads": self.reads}

    audit = Audit()
    record = SimpleNamespace(
        audit=audit,
        observed_indices=(1, 2),
        time_s=torch.arange(4, dtype=torch.float32),
        observed_wavefield=torch.ones(2, 1, 1),
    )
    adapter = SimpleNamespace(
        conditioner=SimpleNamespace(latent_dim=1),
        adapter_parameters=lambda: (),
    )

    def prediction(
        _adapter,
        _normalizer,
        _record,
        _device,
        *,
        time_s,
        modulation_gate=None,
        **_kwargs,
    ):
        gate = torch.zeros(1) if modulation_gate is None else modulation_gate
        return gate.reshape(1, 1, 1, 1).expand(1, len(time_s), 1, 1)

    monkeypatch.setattr(feature_evaluator, "_predict_feature", prediction)
    delta, gate, info = feature_evaluator._refine_residual_trust_gate(
        adapter,
        None,
        record,
        torch.device("cpu"),
        steps=4,
        learning_rate=0.1,
        conditioner_wavefield=torch.ones(1, 2, 1, 1),
    )
    assert audit.reads == [(1, 2)]
    assert info["mode"] == "parent_residual_zero_origin_trust_gate"
    assert info["physical_wavefield_propagator"] is False
    assert info["synthetic_bridge"] is False
    assert info["candidate_observed_loss"] < info["baseline_observed_loss"]
    assert float(gate) > 0.0
    assert float(delta.norm()) == 0.0


def test_future_energy_aggregation_is_global_and_family_resolved():
    reports = []
    for family, truth, parent_error, adapted_error in (
        ("uniform", 4.0, 1.0, 0.25),
        ("layered", 9.0, 4.0, 1.0),
        ("marmousi", 16.0, 9.0, 4.0),
    ):
        reports.append(
            {
                "medium_type": family,
                "hybrid": {
                    "future_truth_squared_norm": truth,
                    "future_parent_squared_error": parent_error,
                    "future_adapted_squared_error": adapted_error,
                },
            }
        )
    result = feature_evaluator.aggregate_future_energy(reports, "hybrid")
    assert result["global"]["parent_relative_l2"] == (14.0 / 29.0) ** 0.5
    assert result["global"]["adapted_relative_l2"] == (5.25 / 29.0) ** 0.5
    assert set(result["by_family"]) == {"uniform", "layered", "marmousi"}
