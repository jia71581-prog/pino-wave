from __future__ import annotations

import numpy as np
import torch

from scripts.b2_v5_components import (
    append_source_conditioning,
    b2_v5_loss,
    causal_window_start,
    temporal_band_relative_l2,
)
from scripts.build_b2_v5_causal_manifest import stratified_unique_group_sample
from scripts.train_b2_v5_pilot import conditioning_key_for_arm


def test_causal_window_uses_source_parameters_and_matches_half_cycle_rule():
    time_s = np.arange(401, dtype=np.float64) * 0.0025
    start = causal_window_start(
        time_s, source_t0_s=0.10, source_f0_hz=20.0, k_frames=64
    )
    assert start == 30


def test_causal_window_clamps_to_stored_horizon():
    time_s = np.arange(20, dtype=np.float64) * 0.1
    assert causal_window_start(
        time_s, source_t0_s=-1.0, source_f0_hz=10.0, k_frames=8
    ) == 0
    assert causal_window_start(
        time_s, source_t0_s=99.0, source_f0_hz=10.0, k_frames=8
    ) == 12


def test_source_conditioning_appends_normalized_constant_maps():
    base = np.zeros((2, 7, 4, 5), dtype=np.float32)
    result = append_source_conditioning(base, [10.0, 30.0], [0.05, 0.15])
    assert result.shape == (2, 9, 4, 5)
    assert np.all(result[0, 7] == -1.0)
    assert np.all(result[1, 7] == 1.0)
    assert np.all(result[0, 8] == -1.0)
    assert np.all(result[1, 8] == 1.0)


def test_nonworse_hinge_penalizes_only_anchor_regression():
    target = torch.ones(2, 4, 1, 3, 3)
    anchor = target.clone()
    prediction = target.clone()
    prediction[1] += 0.2
    loss, parts = b2_v5_loss(
        prediction, target, anchor, arm="nonworse_hinge", nonworse_weight=1.0
    )
    assert loss > parts["relative_l2"]
    assert parts["nonworse_hinge"] > 0.0


def test_spectral_loss_sees_high_frequency_error_with_energy_floor():
    time = torch.arange(56, dtype=torch.float32)
    target = torch.sin(2.0 * torch.pi * time / 28.0)[None, :, None, None, None]
    target = target.expand(1, 56, 1, 4, 4).clone()
    prediction = target + 0.05 * torch.sin(2.0 * torch.pi * time / 2.0)[
        None, :, None, None, None
    ]
    bands = temporal_band_relative_l2(prediction, target)
    assert bands.shape == (1, 3)
    assert bands[0, 2] > bands[0, 0]


def test_stratified_sampler_is_deterministic_and_group_unique():
    candidates = np.arange(24)
    groups = np.asarray([f"g{i // 2}" for i in candidates])
    features = np.stack(
        (candidates, candidates[::-1], candidates % 5, candidates % 7), axis=1
    )
    first = stratified_unique_group_sample(
        candidates, groups, features, count=8, rng=np.random.default_rng(9)
    )
    second = stratified_unique_group_sample(
        candidates, groups, features, count=8, rng=np.random.default_rng(9)
    )
    assert first == second
    assert len({groups[index] for index in first}) == 8


def test_only_source_conditioning_arm_changes_condition_tensor():
    assert conditioning_key_for_arm("source_cond") == "cond_source"
    assert conditioning_key_for_arm("physics_cond") == "cond_physics"
    for arm in ("control", "nonworse_hinge", "spectral"):
        assert conditioning_key_for_arm(arm) == "cond"
