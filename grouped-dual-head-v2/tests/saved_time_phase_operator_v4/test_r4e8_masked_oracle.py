"""Hand-computable tests for the r4e8 masked-oracle metric conventions.

Every expected value below is written as an explicit closed form so a reader can
check it by hand.  The near-zero-truth-frame case is the reason the mask exists:
under convention (i) a frame whose truth energy has decayed to 1e-8 while the
error is O(1) contributes a ratio of 20000 and single-handedly destroys the frame
mean; under convention (ii) that frame is dropped and the mean is exactly 0.5.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation import r16_dscp_training_v2 as harness
from scripts import probe_r4e8_masked_oracle as probe


# ---------------------------------------------------------------------------
# the hand case
#
#   truth energies E   = [4.0, 1.0, 1e-8, 9.0]      max = 9.0
#   squared errors err2= [1.0, 0.25, 4.0, 2.25]
#   tau = 1e-3  ->  floor = 1e-3 * 9.0 = 9e-3 = 0.009
#   truth mask         = [1, 1, 0, 1]               (1e-8 < 0.009)
#
#   (i)   ratios = sqrt(err2/E) = [0.5, 0.5, 20000.0, 0.5]
#         mean   = (0.5 + 0.5 + 20000.0 + 0.5) / 4 = 20001.5 / 4 = 5000.375
#   (ii)  kept frames 0,1,3, denominators max(E, 0.009) = [4.0, 1.0, 9.0]
#         ratios = [0.5, 0.5, 0.5]  ->  mean = 0.5
#   (iii) sqrt(sum err2 / sum E) = sqrt(7.5 / 14.00000001)
# ---------------------------------------------------------------------------

HAND_ENERGY = [4.0, 1.0, 1.0e-8, 9.0]
HAND_ERROR_SQUARED = [1.0, 0.25, 4.0, 2.25]
HAND_FLOOR = 1.0e-3 * 9.0
HAND_CONVENTION_I = 20001.5 / 4.0
HAND_CONVENTION_II = 0.5
HAND_CONVENTION_III = math.sqrt(7.5 / 14.00000001)


def _energy():
    return torch.tensor(HAND_ENERGY, dtype=torch.float64)


def _error():
    return torch.tensor(HAND_ERROR_SQUARED, dtype=torch.float64)


def test_frame_energy_matches_hand_sum_of_squares():
    field = torch.tensor(
        [[[3.0, 4.0]], [[1.0, 0.0]], [[0.0, 0.0]]], dtype=torch.float32
    )
    energy = probe.frame_energy(field)
    assert energy.dtype == torch.float64
    assert energy.tolist() == pytest.approx([25.0, 1.0, 0.0])


def test_energy_mask_drops_only_the_near_zero_truth_frame():
    keep, floor = probe.energy_mask(_energy(), tau=1.0e-3)
    assert floor == pytest.approx(HAND_FLOOR)
    assert [int(v) for v in keep.tolist()] == [1, 1, 0, 1]


def test_energy_mask_keeps_a_frame_exactly_at_the_floor():
    energy = torch.tensor([1.0, 1.0e-3], dtype=torch.float64)
    keep, floor = probe.energy_mask(energy, tau=1.0e-3)
    assert floor == pytest.approx(1.0e-3)
    assert [int(v) for v in keep.tolist()] == [1, 1]


def test_energy_mask_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        probe.energy_mask(torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        probe.energy_mask(torch.tensor([1.0, -1.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        probe.energy_mask(_energy(), tau=0.0)
    with pytest.raises(ValueError):
        probe.energy_mask(_energy(), tau=1.0)


def test_three_conventions_on_the_hand_case():
    result = probe.convention_metrics(_error(), _energy(), tau=1.0e-3)
    metrics = result["metrics"]
    assert metrics["unmasked_per_frame_unsquared_mean"] == pytest.approx(
        HAND_CONVENTION_I, rel=1.0e-12
    )
    assert metrics["masked_energy_floored_per_frame"] == pytest.approx(
        HAND_CONVENTION_II, rel=1.0e-12
    )
    assert metrics["global_energy_rel_l2"] == pytest.approx(
        HAND_CONVENTION_III, rel=1.0e-12
    )


def test_near_zero_truth_frame_is_the_whole_point_of_the_mask():
    result = probe.convention_metrics(_error(), _energy(), tau=1.0e-3)
    metrics = result["metrics"]
    diagnostics = result["diagnostics"]
    # convention (i) is destroyed by one decayed frame
    assert metrics["unmasked_per_frame_unsquared_mean"] > 5000.0
    # convention (ii) is unaffected and exactly the sane value
    assert metrics["masked_energy_floored_per_frame"] == pytest.approx(0.5)
    # the mask dropped exactly one frame out of four
    assert diagnostics["frame_count"] == 4
    assert diagnostics["masked_frame_count"] == 3
    assert diagnostics["dropped_frame_count"] == 1
    assert diagnostics["energy_floor"] == pytest.approx(HAND_FLOOR)
    # the floor alone bounds the pathology but does not remove it:
    # sqrt(4.0/0.009) = 21.0818..., mean = (0.5+0.5+sqrt(4/0.009)+0.5)/4
    expected_floored = (0.5 + 0.5 + math.sqrt(4.0 / HAND_FLOOR) + 0.5) / 4.0
    assert diagnostics["floored_unmasked_per_frame"] == pytest.approx(
        expected_floored, rel=1.0e-12
    )
    assert diagnostics["floored_unmasked_per_frame"] < 6.0


def test_convention_ii_ignores_error_on_dropped_frames():
    """Blowing up the error on the decayed frame must not move convention (ii)."""
    worse = list(HAND_ERROR_SQUARED)
    worse[2] = 1.0e6
    base = probe.convention_metrics(_error(), _energy())["metrics"]
    moved = probe.convention_metrics(
        torch.tensor(worse, dtype=torch.float64), _energy()
    )["metrics"]
    assert moved["masked_energy_floored_per_frame"] == pytest.approx(
        base["masked_energy_floored_per_frame"]
    )
    assert moved["unmasked_per_frame_unsquared_mean"] > base[
        "unmasked_per_frame_unsquared_mean"
    ]


def test_convention_metrics_requires_a_retained_frame():
    with pytest.raises(FloatingPointError):
        probe.convention_metrics(
            torch.tensor([float("nan"), 1.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError):
        probe.convention_metrics(
            torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )


def test_gain_is_the_v14_relative_reduction_form():
    parent = {"a": 0.8, "b": 2.0}
    corrected = {"a": 0.4, "b": 3.0}
    gains = probe.convention_gains(parent, corrected)
    assert gains["a"] == pytest.approx(0.5)          # (0.8-0.4)/0.8
    assert gains["b"] == pytest.approx(-0.5)         # (2.0-3.0)/2.0


def test_mask_agreement_rate_is_hand_countable():
    truth = torch.tensor([True, True, False, True])
    parent = torch.tensor([True, True, True, True])
    assert probe.mask_agreement(truth, parent) == pytest.approx(0.75)
    assert probe.mask_agreement(truth, truth) == pytest.approx(1.0)
    assert probe.mask_agreement(truth, ~truth) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        probe.mask_agreement(truth, torch.tensor([True, False]))


def test_parent_energy_mask_can_disagree_with_truth_energy_mask():
    """The deployment-causal proxy check must be able to fail, not just pass."""
    truth_energy = torch.tensor([9.0, 1.0, 1.0e-9, 1.0e-9], dtype=torch.float64)
    parent_energy = torch.tensor([9.0, 1.0, 5.0, 5.0], dtype=torch.float64)
    truth_keep, _ = probe.energy_mask(truth_energy)
    parent_keep, _ = probe.energy_mask(parent_energy)
    assert [int(v) for v in truth_keep.tolist()] == [1, 1, 0, 0]
    assert [int(v) for v in parent_keep.tolist()] == [1, 1, 1, 1]
    assert probe.mask_agreement(truth_keep, parent_keep) == pytest.approx(0.5)
    assert probe.mask_agreement(truth_keep, parent_keep) < probe.MASK_AGREEMENT_MINIMUM


def test_weighted_coefficient_map_recovers_an_exact_low_rank_residual():
    torch.manual_seed(0)
    time_count, rank, points = 40, 3, 7
    design = torch.linalg.qr(torch.randn(time_count, rank, dtype=torch.float64))[0]
    true_coefficients = torch.randn(rank, points, dtype=torch.float64)
    residual = design @ true_coefficients
    weights = torch.rand(time_count, dtype=torch.float64) + 0.5
    mapping = probe.weighted_coefficient_map(design, weights)
    assert torch.allclose(mapping @ residual, true_coefficients, atol=1.0e-10)


def test_weighted_coefficient_map_rejects_bad_weights():
    design = torch.eye(4, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        probe.weighted_coefficient_map(design, torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        probe.weighted_coefficient_map(design, torch.ones(3, dtype=torch.float64))


def test_weighted_map_agrees_with_the_frozen_harness_weighted_ls_path():
    """probe.weighted_coefficient_map must equal weighted_coefficient_target at rank 16.

    The spec requires the oracle coefficients come from the existing weighted-LS
    path.  weighted_coefficient_target solves the normal equations; the probe uses
    the pinv-of-rooted-design form that produced the published r4e7 constants.
    They must agree.
    """
    torch.manual_seed(1)
    height = width = 3
    k1 = 12
    start = k1 + 1
    basis = torch.linalg.qr(torch.randn(401, harness.RANK, dtype=torch.float64))[0]
    parent = torch.randn(401, height, width, dtype=torch.float32)
    truth = parent + 0.25 * torch.randn(401, height, width, dtype=torch.float32)

    reference = harness.weighted_coefficient_target(
        basis.float(), parent, truth, k1=k1
    ).double().reshape(harness.RANK, -1)

    truth_energy = probe.frame_energy(truth[start:])
    weights = truth_energy.clamp_min(probe.EPS_SQUARED).reciprocal()
    mapping = probe.weighted_coefficient_map(basis[start:], weights)
    residual = (truth[start:] - parent[start:]).reshape(401 - start, -1).double()
    mine = mapping @ residual

    assert mine.shape == reference.shape
    assert torch.allclose(mine, reference, rtol=1.0e-6, atol=1.0e-8)


def test_masked_oracle_record_on_an_exactly_correctable_synthetic_record():
    """A residual that lies exactly in the basis span must drive every gain to 1."""
    torch.manual_seed(2)
    height = width = 4
    k1 = 20
    start = k1 + 1
    basis = torch.linalg.qr(torch.randn(401, 32, dtype=torch.float64))[0]
    truth = torch.randn(401, height, width, dtype=torch.float32)
    coefficients = torch.randn(8, height * width, dtype=torch.float64)
    residual = (basis[:, :8] @ coefficients).reshape(401, height, width)
    parent = (truth.double() - residual).float()

    scored = probe.masked_oracle_record(
        parent, truth, basis, k1=k1, ranks=(8, 16, 32)
    )
    assert scored["future_start_index"] == start
    assert scored["future_frame_count"] == 401 - start
    for rank in ("8", "16", "32"):
        item = scored["ranks"][rank]
        assert item["finite"] is True
        assert item["raw_residual_energy_capture"] == pytest.approx(1.0, abs=1.0e-8)
        for convention, value in item["gain"].items():
            assert value == pytest.approx(1.0, abs=1.0e-6), convention
    assert scored["mask"]["agreement_rate"] == pytest.approx(1.0)


def test_masked_oracle_record_reports_all_three_conventions_and_the_masks():
    torch.manual_seed(3)
    height = width = 4
    k1 = 20
    basis = torch.linalg.qr(torch.randn(401, 32, dtype=torch.float64))[0]
    truth = torch.randn(401, height, width, dtype=torch.float32)
    parent = truth + 0.5 * torch.randn(401, height, width, dtype=torch.float32)
    scored = probe.masked_oracle_record(parent, truth, basis, k1=k1, ranks=(16,))
    expected = {
        "unmasked_per_frame_unsquared_mean",
        "masked_energy_floored_per_frame",
        "global_energy_rel_l2",
    }
    assert set(scored["parent_metrics"]) == expected
    assert set(scored["ranks"]["16"]["corrected_metrics"]) == expected
    assert set(scored["ranks"]["16"]["gain"]) == expected
    assert "correction_energy_ratio" in scored["ranks"]["16"]
    assert set(scored["profiles"]) == {
        "truth_energy",
        "truth_energy_normalized",
        "parent_energy",
        "parent_energy_normalized",
        "truth_mask",
        "parent_mask",
    }
    assert max(scored["profiles"]["truth_energy_normalized"]) == pytest.approx(1.0)
    assert scored["mask"]["tau"] == probe.TAU
    assert 0.0 <= scored["mask"]["agreement_rate"] <= 1.0


def test_masked_oracle_record_rejects_shape_and_range_errors():
    basis = torch.eye(401, 32, dtype=torch.float64)
    truth = torch.randn(401, 2, 2, dtype=torch.float32)
    with pytest.raises(ValueError):
        probe.masked_oracle_record(truth[:400], truth, basis, k1=10, ranks=(8,))
    with pytest.raises(ValueError):
        probe.masked_oracle_record(truth, truth.clone(), basis, k1=400, ranks=(8,))
    with pytest.raises(ValueError):
        probe.masked_oracle_record(
            truth, truth + 1.0, basis, k1=10, ranks=(64,)
        )


# ---------------------------------------------------------------------------
# contract guards
# ---------------------------------------------------------------------------


def test_spec_binding_is_the_frozen_hash():
    binding = probe.verify_spec()
    assert binding["sha256"] == probe.SPEC_SHA256


def test_frozen_v14_bindings_are_unchanged():
    report = probe.verify_v14_bindings()
    assert report["count"] == 16
    assert report["all_unchanged"] is True
    assert all(item["unchanged"] for item in report["bindings"].values())


def test_writes_outside_the_authorized_result_directory_are_refused():
    allowed = probe.RESULT_DIR / "terminal.json"
    assert probe.assert_write_allowed(allowed) == allowed.resolve()
    for forbidden in (
        PROJECT_ROOT / "results/r16_dscp_v14/design_preflight.json",
        PROJECT_ROOT / "results/r16_dscp_v1/basis_rank16.pt",
        PROJECT_ROOT / "configs/r16_dscp_v14.yaml",
        PROJECT_ROOT / "results/anything_else.json",
    ):
        with pytest.raises(probe.ProbeContractError):
            probe.assert_write_allowed(forbidden)


def test_truth_scope_allowlist_covers_only_train_probe_records():
    assert "train_uniform_00321" in probe.ALLOWED_TRUTH_SAMPLE_IDS
    assert "train_layered_00564" in probe.ALLOWED_TRUTH_SAMPLE_IDS
    assert "train_marmousi_00385" in probe.ALLOWED_TRUTH_SAMPLE_IDS
    for sample_id in probe.ALLOWED_TRUTH_SAMPLE_IDS:
        assert sample_id.startswith("train_")
    assert not any(
        "valid" in sample_id or "test" in sample_id
        for sample_id in probe.ALLOWED_TRUTH_SAMPLE_IDS
    )


def test_panel_is_exactly_the_v14_smoke_panel():
    assert probe.PANEL_IDS == (
        ("uniform", "train_uniform_00321"),
        ("layered", "train_layered_00564"),
        ("marmousi", "train_marmousi_00385"),
    )


def test_v14_transplanted_constants_match_the_engine_source():
    engine = (
        PROJECT_ROOT
        / "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v3.py"
    ).read_text(encoding="utf8")
    for family, item in probe.V14_TRANSPLANTED_ORACLE.items():
        assert repr(item["value"]).lstrip("0") in engine, family
        assert item["measured_on"] != dict(probe.PANEL_IDS)[family]


def test_decision_rule_follows_the_pre_stated_spec_rule():
    def family(gain16, gain32, agreement, sample_id="s"):
        def ranks(value):
            return {
                "gain": {"masked_energy_floored_per_frame": value},
                "masked_matched_fit": {
                    "gain": {"masked_energy_floored_per_frame": value}
                },
            }

        return {
            "sample_id": sample_id,
            "mask": {"agreement_rate": agreement},
            "ranks": {"8": ranks(0.05), "16": ranks(gain16), "32": ranks(gain32)},
        }

    passing = {
        "uniform": family(0.30, 0.33, 1.0),
        "layered": family(0.40, 0.42, 0.98),
        "marmousi": family(0.25, 0.28, 0.95),
    }
    verdict = probe.apply_decision_rule(passing)
    assert verdict["all_families_rank16_meet_minimum"] is True
    assert verdict["all_families_rank32_marginal_below_maximum"] is True
    assert verdict["all_families_mask_agreement_pass"] is True
    assert verdict["rank_verdict"].startswith("rank_is_not_the_bottleneck")
    assert verdict["mask_verdict"].startswith("parent_energy_mask_is_a_valid")
    assert verdict["uniform_single_threshold_permitted"] is False

    # a family below 0.20 forbids a single uniform threshold and reports per family
    short = dict(passing)
    short["uniform"] = family(0.10, 0.12, 1.0)
    verdict = probe.apply_decision_rule(short)
    assert verdict["all_families_rank16_meet_minimum"] is False
    assert verdict["rank_verdict"].startswith("rank16_misses_the_minimum")
    assert verdict["families"]["uniform"]["masked_gain_rank16"] == pytest.approx(0.10)

    # rank 32 still adding >= 0.05 means rank is still a bottleneck
    deep = dict(passing)
    deep["layered"] = family(0.40, 0.60, 0.98)
    verdict = probe.apply_decision_rule(deep)
    assert verdict["all_families_rank16_meet_minimum"] is True
    assert verdict["all_families_rank32_marginal_below_maximum"] is False
    assert verdict["rank_verdict"].startswith("rank16_clears_the_minimum_everywhere_but")

    # a mask below 0.90 invalidates the proxy
    weak = dict(passing)
    weak["marmousi"] = family(0.25, 0.28, 0.80)
    verdict = probe.apply_decision_rule(weak)
    assert verdict["all_families_mask_agreement_pass"] is False
    assert verdict["mask_verdict"].startswith("parent_energy_mask_is_not_a_valid")


def test_self_check_targets_are_the_published_values_and_records():
    uniform = probe.SELF_CHECK["uniform_rank16_unmasked_per_frame_reduction"]
    layered = probe.SELF_CHECK["layered_rank16_raw_residual_energy_capture"]
    assert uniform["sample_id"] == "train_uniform_00002"
    assert uniform["published_value"] == pytest.approx(-1.639005450056983)
    assert layered["sample_id"] == "train_layered_01032"
    assert layered["published_value"] == pytest.approx(0.6430370502360689)
    for spec in (uniform, layered):
        published = PROJECT_ROOT / str(spec["published_file"])
        assert published.exists()
        assert spec["absolute_tolerance"] > 0.0


# ---------------------------------------------------------------------------
# the convention (ii) matched arm
# ---------------------------------------------------------------------------


def test_masked_matched_arm_is_reported_for_every_rank():
    torch.manual_seed(5)
    height = width = 4
    basis = torch.linalg.qr(torch.randn(401, 32, dtype=torch.float64))[0]
    truth = torch.randn(401, height, width, dtype=torch.float32)
    parent = truth + 0.5 * torch.randn(401, height, width, dtype=torch.float32)
    scored = probe.masked_oracle_record(parent, truth, basis, k1=20, ranks=(8, 16, 32))
    for rank in ("8", "16", "32"):
        arm = scored["ranks"][rank]["masked_matched_fit"]
        assert set(arm["gain"]) == {
            "unmasked_per_frame_unsquared_mean",
            "masked_energy_floored_per_frame",
            "global_energy_rel_l2",
        }
        assert arm["fit_frame_count"] == scored["mask"]["truth_masked_frame_count"]
        assert math.isfinite(arm["correction_energy_ratio"])


def test_masked_matched_arm_beats_the_published_arm_on_convention_ii():
    """The (ii)-matched least squares must dominate the (i)-weighted fit on (ii).

    This is the whole reason the second arm exists: the published estimator
    optimizes a different objective, so it can and does score worse on (ii).
    """
    torch.manual_seed(6)
    height = width = 5
    k1 = 20
    start = k1 + 1
    basis = torch.linalg.qr(torch.randn(401, 32, dtype=torch.float64))[0]
    # decaying truth energy so masked and unmasked conventions genuinely differ
    decay = torch.exp(-torch.arange(401, dtype=torch.float32) / 40.0)
    truth = decay[:, None, None] * torch.randn(401, height, width, dtype=torch.float32)
    parent = truth + 0.2 * torch.randn(401, height, width, dtype=torch.float32)
    scored = probe.masked_oracle_record(parent, truth, basis, k1=k1, ranks=(16,))
    item = scored["ranks"]["16"]
    published = item["gain"]["masked_energy_floored_per_frame"]
    matched = item["masked_matched_fit"]["gain"]["masked_energy_floored_per_frame"]
    assert matched >= published - 1.0e-12
    # and the mask must actually be dropping decayed frames in this setup
    assert scored["parent_metric_diagnostics"]["dropped_frame_count"] > 0
    assert scored["future_start_index"] == start


def test_masked_matched_arm_also_reaches_unit_gain_when_the_residual_is_in_span():
    torch.manual_seed(7)
    height = width = 4
    basis = torch.linalg.qr(torch.randn(401, 32, dtype=torch.float64))[0]
    truth = torch.randn(401, height, width, dtype=torch.float32)
    coefficients = torch.randn(8, height * width, dtype=torch.float64)
    residual = (basis[:, :8] @ coefficients).reshape(401, height, width)
    parent = (truth.double() - residual).float()
    scored = probe.masked_oracle_record(parent, truth, basis, k1=20, ranks=(8,))
    arm = scored["ranks"]["8"]["masked_matched_fit"]
    assert arm["raw_residual_energy_capture"] == pytest.approx(1.0, abs=1.0e-8)
    for convention, value in arm["gain"].items():
        assert value == pytest.approx(1.0, abs=1.0e-6), convention


def test_decision_rule_can_read_either_fit_arm():
    def family(published16, matched16, agreement):
        def ranks(pub, mat):
            return {
                "gain": {"masked_energy_floored_per_frame": pub},
                "masked_matched_fit": {
                    "gain": {"masked_energy_floored_per_frame": mat}
                },
            }

        return {
            "sample_id": "s",
            "mask": {"agreement_rate": agreement},
            "ranks": {
                "8": ranks(0.01, 0.05),
                "16": ranks(published16, matched16),
                "32": ranks(published16 + 0.01, matched16 + 0.01),
            },
        }

    # published arm fails the 0.20 minimum, matched arm clears it
    panel = {
        "uniform": family(-0.23, 0.31, 0.99),
        "layered": family(-0.10, 0.44, 0.99),
        "marmousi": family(0.05, 0.28, 0.99),
    }
    published = probe.apply_decision_rule(panel, arm="published_weighted")
    matched = probe.apply_decision_rule(panel, arm="masked_matched")
    assert published["fit_arm"] == "published_weighted"
    assert matched["fit_arm"] == "masked_matched"
    assert published["all_families_rank16_meet_minimum"] is False
    assert matched["all_families_rank16_meet_minimum"] is True
    assert matched["rank_verdict"].startswith("rank_is_not_the_bottleneck")
    assert published["thresholds"] == matched["thresholds"]
    with pytest.raises(ValueError):
        probe.apply_decision_rule(panel, arm="nonsense")


def test_fit_arms_are_exactly_the_two_named_arms():
    assert probe.FIT_ARMS == ("published_weighted", "masked_matched")
