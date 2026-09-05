"""Hand-computable tests for the r4e9 mask-confined oracle probe.

Gate 2 of the frozen spec requires a synthetic case, checkable by hand, on which
confinement (a) leaves the in-mask error completely unchanged and (b) drives the
out-of-mask error to exactly the parent value.  ``test_confinement_leaves_...``
and ``test_confinement_restores_...`` below are that case; every expected number
is written as a closed form.

The rest of the file pins the binding hashes, the error-energy split algebra, the
blow-up attribution, and each of the three pre-stated decision branches.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from scripts import probe_r4e8_masked_oracle as r4e8
from scripts import probe_r4e9_mask_confined as probe


# ---------------------------------------------------------------------------
# The hand case for gate 2.
#
# Four future frames, one spatial point, so every energy is just a square.
#
#   parent field p = [3.0, 4.0, 0.0, 0.0]     (parent frame energies 9,16,0,0)
#   tau = 1e-3, parent peak energy = 16.0  ->  parent floor = 0.016
#   parent keep mask m^parent = [1, 1, 0, 0]
#
# The correction the oracle wants to apply is c = [1.0, 2.0, 5.0, 7.0] and the
# parent residual is r = [2.0, 3.0, 1.0, 1.0].
#
#   unconfined error_t = (r_t - c_t)^2 = [1.0, 1.0, 16.0, 36.0]
#   confined   error_t = (r_t - m_t*c_t)^2 = [1.0, 1.0, 1.0, 1.0]
#   parent     error_t = r_t^2            = [4.0, 9.0, 1.0, 1.0]
#
# In-mask (frames 0,1):  confined == unconfined  -> [1.0, 1.0], sum 2.0
# Out-of-mask (2,3):     confined == parent      -> [1.0, 1.0], sum 2.0
#                        unconfined              -> [16.0, 36.0], sum 52.0
#
# So the unconfined arm's total error energy is 2.0 + 52.0 = 54.0 against the
# parent's 4+9+1+1 = 15.0, an increase of 39.0, of which the outside-mask part is
# 52.0 - 2.0 = 50.0 and the inside part is 2.0 - 13.0 = -11.0.  The outside share
# of the increase is 50/39, above 1 because confinement also helps inside.
# ---------------------------------------------------------------------------

HAND_PARENT_FIELD = [3.0, 4.0, 0.0, 0.0]
HAND_RESIDUAL = [2.0, 3.0, 1.0, 1.0]
HAND_CORRECTION = [1.0, 2.0, 5.0, 7.0]
HAND_PARENT_MASK = [1, 1, 0, 0]

HAND_PARENT_ERROR = [4.0, 9.0, 1.0, 1.0]
HAND_UNCONFINED_ERROR = [1.0, 1.0, 16.0, 36.0]
HAND_CONFINED_ERROR = [1.0, 1.0, 1.0, 1.0]

HAND_PARENT_INSIDE = 4.0 + 9.0
HAND_PARENT_OUTSIDE = 1.0 + 1.0
HAND_UNCONFINED_INSIDE = 1.0 + 1.0
HAND_UNCONFINED_OUTSIDE = 16.0 + 36.0
HAND_CONFINED_INSIDE = 1.0 + 1.0
HAND_CONFINED_OUTSIDE = 1.0 + 1.0


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def _parent_mask():
    return torch.tensor(HAND_PARENT_MASK, dtype=torch.bool)


# ---------------------------------------------------------------------------
# bindings
# ---------------------------------------------------------------------------


def test_spec_sha256_matches_the_authorized_binding():
    binding = probe.verify_bindings()
    assert binding["spec"]["sha256"] == probe.SPEC_SHA256
    assert binding["parent_spec"]["sha256"] == probe.PARENT_SPEC_SHA256
    assert binding["reused_code_path"]["sha256"] == probe.PARENT_SCRIPT_SHA256


def test_reused_r4e8_script_is_the_exact_frozen_revision():
    digest = probe.parent_runtime.sha256_file(probe.PARENT_SCRIPT_PATH)
    assert digest == "3373b5838e49da81e5a8e2a3bada3e236aa7855d4bcdcd149cd4e1233900b485"


def test_probe_reuses_the_r4e8_metric_functions_rather_than_reimplementing_them():
    # the conventions and the mask must come from the untouched r4e8 module
    assert probe.TAU is r4e8.TAU
    assert probe.RANKS is r4e8.RANKS
    assert probe.PANEL_IDS is r4e8.PANEL_IDS
    assert probe.EPS_SQUARED == r4e8.EPS_SQUARED


def test_write_scope_refuses_paths_outside_the_r4e9_result_directory():
    allowed = probe.assert_write_allowed(probe.RESULT_DIR / "terminal.json")
    assert allowed.parent == probe.RESULT_DIR.resolve()
    for forbidden in (
        PROJECT_ROOT / "results/r4e8_masked_oracle_probe_20260826/terminal.json",
        PROJECT_ROOT / "results/r16_dscp_v14/design_preflight.json",
        PROJECT_ROOT / "scripts/probe_r4e8_masked_oracle.py",
    ):
        with pytest.raises(probe.ProbeContractError):
            probe.assert_write_allowed(forbidden)


# ---------------------------------------------------------------------------
# gate 2: the hand-computable confinement case
# ---------------------------------------------------------------------------


def test_hand_case_masks_and_errors_are_what_the_comment_claims():
    """Guard the arithmetic the rest of gate 2 leans on."""
    parent_energy = r4e8.frame_energy(_tensor(HAND_PARENT_FIELD)[:, None])
    assert parent_energy.tolist() == pytest.approx([9.0, 16.0, 0.0, 0.0])
    keep, floor = r4e8.energy_mask(parent_energy, tau=1.0e-3)
    assert floor == pytest.approx(1.0e-3 * 16.0)
    assert [int(v) for v in keep.tolist()] == HAND_PARENT_MASK

    residual = _tensor(HAND_RESIDUAL)
    correction = _tensor(HAND_CORRECTION)
    gate = keep.to(dtype=torch.float64)

    assert (residual**2).tolist() == pytest.approx(HAND_PARENT_ERROR)
    assert ((residual - correction) ** 2).tolist() == pytest.approx(
        HAND_UNCONFINED_ERROR
    )
    assert ((residual - gate * correction) ** 2).tolist() == pytest.approx(
        HAND_CONFINED_ERROR
    )


def test_confinement_leaves_in_mask_error_completely_unchanged():
    """Gate 2 (a): inside the mask, confined error == unconfined error, exactly."""
    keep = _parent_mask()
    unconfined = _tensor(HAND_UNCONFINED_ERROR)
    confined = _tensor(HAND_CONFINED_ERROR)

    assert confined[keep].tolist() == unconfined[keep].tolist()
    assert float((confined[keep] - unconfined[keep]).abs().max()) == 0.0

    unconfined_split = probe.error_energy_split(unconfined, keep)
    confined_split = probe.error_energy_split(confined, keep)
    assert confined_split["inside_mask"] == unconfined_split["inside_mask"]
    assert confined_split["inside_mask"] == pytest.approx(HAND_CONFINED_INSIDE)
    assert unconfined_split["inside_mask"] == pytest.approx(HAND_UNCONFINED_INSIDE)


def test_confinement_restores_out_of_mask_error_to_the_parent_value_exactly():
    """Gate 2 (b): outside the mask, confined error == parent error, exactly."""
    keep = _parent_mask()
    outside = ~keep
    parent = _tensor(HAND_PARENT_ERROR)
    confined = _tensor(HAND_CONFINED_ERROR)
    unconfined = _tensor(HAND_UNCONFINED_ERROR)

    assert confined[outside].tolist() == parent[outside].tolist()
    assert float((confined[outside] - parent[outside]).abs().max()) == 0.0
    # and the unconfined arm is the one that broke it
    assert float((unconfined[outside] - parent[outside]).abs().max()) > 0.0

    parent_split = probe.error_energy_split(parent, keep)
    confined_split = probe.error_energy_split(confined, keep)
    assert confined_split["outside_mask"] == parent_split["outside_mask"]
    assert confined_split["outside_mask"] == pytest.approx(HAND_PARENT_OUTSIDE)


def test_confinement_invariants_hold_through_the_full_correction_pipeline():
    """The same two invariants, computed the way the probe computes them."""
    parent_energy = r4e8.frame_energy(_tensor(HAND_PARENT_FIELD)[:, None])
    keep, _floor = r4e8.energy_mask(parent_energy, tau=1.0e-3)
    gate = keep.to(dtype=torch.float64)

    residual = _tensor(HAND_RESIDUAL)
    correction = _tensor(HAND_CORRECTION)

    parent_error = residual.square()
    unconfined_error = (residual - correction).square()
    confined_error = (residual - gate * correction).square()

    assert float(
        (confined_error[keep] - unconfined_error[keep]).abs().max()
    ) == 0.0
    assert float(
        (confined_error[~keep] - parent_error[~keep]).abs().max()
    ) == 0.0
    # confinement is exactly zero outside, not merely small
    assert (gate * correction)[~keep].tolist() == [0.0, 0.0]


def test_confined_correction_energy_is_the_in_mask_part_only():
    keep = _parent_mask()
    gate = keep.to(dtype=torch.float64)
    correction = _tensor(HAND_CORRECTION)
    confined = gate * correction
    assert float(confined.square().sum()) == pytest.approx(1.0 + 4.0)
    assert float(correction.square().sum()) == pytest.approx(1.0 + 4.0 + 25.0 + 49.0)


def test_chunked_and_whole_row_reductions_can_differ_in_the_last_ulps():
    """Why the probe gates on the chunk-matched parent error, not the whole-row one.

    Confinement makes the applied correction identically zero outside the mask,
    so the confined error there is the parent error computed by the *same*
    arithmetic.  Summing the same row in two different association orders is a
    different arithmetic and may legitimately disagree in the last ulps, so the
    probe asserts exactness against the chunk-matched accumulation and merely
    reports the whole-row difference.
    """
    torch.manual_seed(0)
    row = torch.randn(8192, dtype=torch.float64) * 1.0e-8
    whole_row = float(row.square().sum())
    chunked = 0.0
    for lo in range(0, row.numel(), 4096):
        chunked += float(row[lo : lo + 4096].square().sum())
    # equal to rounding, but not necessarily bit-identical
    assert chunked == pytest.approx(whole_row, rel=1.0e-12)

    # the exact claim confinement actually makes: a zero gate zeroes the
    # correction identically, for any values whatsoever
    gate = torch.zeros(row.shape, dtype=torch.float64)
    assert float((gate * row).abs().max()) == 0.0
    residual = torch.randn(8192, dtype=torch.float64)
    assert ((residual - gate * row) ** 2).tolist() == (residual**2).tolist()


# ---------------------------------------------------------------------------
# error energy split
# ---------------------------------------------------------------------------


def test_error_energy_split_matches_the_hand_sums():
    keep = _parent_mask()
    split = probe.error_energy_split(_tensor(HAND_PARENT_ERROR), keep)
    assert split["inside_mask"] == pytest.approx(HAND_PARENT_INSIDE)
    assert split["outside_mask"] == pytest.approx(HAND_PARENT_OUTSIDE)
    assert split["total"] == pytest.approx(15.0)
    assert split["inside_frame_count"] == 2
    assert split["outside_frame_count"] == 2
    assert split["outside_share_of_total"] == pytest.approx(2.0 / 15.0)


def test_error_energy_split_parts_sum_to_the_total_error_energy():
    keep = _parent_mask()
    for values in (HAND_PARENT_ERROR, HAND_UNCONFINED_ERROR, HAND_CONFINED_ERROR):
        error = _tensor(values)
        split = probe.error_energy_split(error, keep)
        assert split["inside_mask"] + split["outside_mask"] == pytest.approx(
            float(error.sum())
        )


def test_error_energy_split_rejects_a_non_finite_error_vector():
    with pytest.raises(FloatingPointError):
        probe.error_energy_split(
            torch.tensor([1.0, float("nan"), 2.0, 3.0], dtype=torch.float64),
            _parent_mask(),
        )


def test_error_energy_split_rejects_a_shape_mismatch():
    with pytest.raises(ValueError):
        probe.error_energy_split(_tensor([1.0, 2.0]), _parent_mask())


def test_split_is_a_direct_measurement_not_an_elimination():
    """The outside part is read off the dropped frames, never inferred as a residue."""
    keep = _parent_mask()
    unconfined = _tensor(HAND_UNCONFINED_ERROR)
    split = probe.error_energy_split(unconfined, keep)
    direct = float(unconfined[~keep].sum())
    assert split["outside_mask"] == direct
    assert split["outside_mask"] == pytest.approx(HAND_UNCONFINED_OUTSIDE)


# ---------------------------------------------------------------------------
# blow-up attribution
# ---------------------------------------------------------------------------


def test_blowup_attribution_locates_the_hand_case_increase_outside_the_mask():
    keep = _parent_mask()
    parent_split = probe.error_energy_split(_tensor(HAND_PARENT_ERROR), keep)
    unconfined_split = probe.error_energy_split(_tensor(HAND_UNCONFINED_ERROR), keep)
    attribution = probe.blowup_attribution(parent_split, unconfined_split)

    assert attribution["delta_inside_mask"] == pytest.approx(2.0 - 13.0)
    assert attribution["delta_outside_mask"] == pytest.approx(52.0 - 2.0)
    assert attribution["delta_total"] == pytest.approx(54.0 - 15.0)
    assert attribution["total_error_energy_increased_vs_parent"] is True
    assert attribution["outside_share_of_increase"] == pytest.approx(50.0 / 39.0)
    assert (
        attribution["outside_share_of_increase"]
        >= probe.OUTSIDE_MASK_BLOWUP_SHARE_MINIMUM
    )


def test_blowup_attribution_reports_no_share_when_there_was_no_increase():
    keep = _parent_mask()
    parent_split = probe.error_energy_split(_tensor(HAND_PARENT_ERROR), keep)
    confined_split = probe.error_energy_split(_tensor(HAND_CONFINED_ERROR), keep)
    attribution = probe.blowup_attribution(parent_split, confined_split)
    assert attribution["delta_total"] == pytest.approx(4.0 - 15.0)
    assert attribution["total_error_energy_increased_vs_parent"] is False
    assert attribution["outside_share_of_increase"] is None


def test_confined_variant_never_changes_the_out_of_mask_delta():
    keep = _parent_mask()
    parent_split = probe.error_energy_split(_tensor(HAND_PARENT_ERROR), keep)
    confined_split = probe.error_energy_split(_tensor(HAND_CONFINED_ERROR), keep)
    attribution = probe.blowup_attribution(parent_split, confined_split)
    assert attribution["delta_outside_mask"] == 0.0


# ---------------------------------------------------------------------------
# reproduction gate against the stored r4e8 artifact
# ---------------------------------------------------------------------------


def _reference():
    return probe.load_r4e8_reference()


def test_r4e8_reference_artifact_loads_all_nine_points():
    reference = _reference()
    assert reference["candidate"] == "r4e8_masked_oracle_probe_20260826"
    assert sorted(reference["families"]) == ["layered", "marmousi", "uniform"]
    for family in reference["families"].values():
        assert sorted(family["ranks"]) == ["16", "32", "8"]
        for rank_item in family["ranks"].values():
            assert sorted(rank_item["gain"]) == sorted(probe.CONVENTIONS)


def _panel_from_reference(reference, *, perturb=None):
    """Build a synthetic panel whose unconfined arm equals the r4e8 reference."""
    panel = {}
    for family, item in reference["families"].items():
        ranks = {}
        for rank in probe.RANKS:
            gain = dict(item["ranks"][str(rank)]["gain"])
            if perturb is not None:
                gain = perturb(family, rank, gain)
            ranks[str(rank)] = {
                "variants": {
                    "unconfined": {"gain": gain},
                    "confined": {"gain": dict(gain)},
                },
                "blowup_attribution": {
                    "unconfined": {
                        "delta_total": 1.0,
                        "total_error_energy_increased_vs_parent": True,
                        "outside_share_of_increase": 1.0,
                    }
                },
            }
        panel[family] = {"sample_id": item["sample_id"], "ranks": ranks}
    return panel


def test_reproduction_gate_passes_when_every_point_is_bitwise_identical():
    reference = _reference()
    report = probe.compare_against_r4e8(_panel_from_reference(reference), reference)
    assert report["point_count"] == 9
    assert report["all_bitwise_identical"] is True
    assert report["max_absolute_difference_over_all_points"] == 0.0


def test_reproduction_gate_fails_on_a_one_ulp_drift_at_a_single_point():
    reference = _reference()

    def perturb(family, rank, gain):
        if family == "layered" and rank == 32:
            gain = dict(gain)
            gain["global_energy_rel_l2"] = math.nextafter(
                gain["global_energy_rel_l2"], math.inf
            )
        return gain

    report = probe.compare_against_r4e8(
        _panel_from_reference(reference, perturb=perturb), reference
    )
    assert report["all_bitwise_identical"] is False
    assert report["points"]["layered_rank32"]["bitwise_identical"] is False
    assert report["points"]["uniform_rank16"]["bitwise_identical"] is True


def test_reproduction_gate_rejects_a_record_swap():
    reference = _reference()
    panel = _panel_from_reference(reference)
    panel["uniform"]["sample_id"] = "train_uniform_00102"
    with pytest.raises(probe.ProbeContractError):
        probe.compare_against_r4e8(panel, reference)


# ---------------------------------------------------------------------------
# the pre-stated decision rule
# ---------------------------------------------------------------------------


def _decision_panel(entries):
    """entries: family -> rank -> (ii_unconfined, ii_confined, iii_unconfined, iii_confined, outside_share)"""
    panel = {}
    for family, ranks in entries.items():
        built = {}
        for rank, values in ranks.items():
            ii_un, ii_con, iii_un, iii_con, share = values
            increased = iii_un < 0.0
            built[str(rank)] = {
                "variants": {
                    "unconfined": {
                        "gain": {
                            "unmasked_per_frame_unsquared_mean": 0.0,
                            "masked_energy_floored_per_frame": ii_un,
                            "global_energy_rel_l2": iii_un,
                        }
                    },
                    "confined": {
                        "gain": {
                            "unmasked_per_frame_unsquared_mean": 0.0,
                            "masked_energy_floored_per_frame": ii_con,
                            "global_energy_rel_l2": iii_con,
                        }
                    },
                },
                "blowup_attribution": {
                    "unconfined": {
                        "delta_total": 1.0 if increased else -1.0,
                        "total_error_energy_increased_vs_parent": increased,
                        "outside_share_of_increase": share if increased else None,
                    }
                },
            }
        panel[family] = {
            "sample_id": f"train_{family}_00000",
            "mask": {"agreement_rate": 0.99},
            "ranks": built,
        }
    return panel


def _uniform_entries(values):
    return {family: {rank: values for rank in probe.RANKS} for family in probe.FAMILIES}


def test_branch_a_when_all_three_families_keep_ii_and_reach_nonnegative_iii():
    panel = _decision_panel(_uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9)))
    decision = probe.apply_decision_rule(panel)
    assert decision["branch"].startswith("a_")
    assert decision["hypothesis_H"]["confirmed"] is True
    assert decision["families_routed_to_abstention"] == []
    assert decision["collapsed_to_single_threshold"] is False


def test_branch_a_tolerance_is_exactly_two_hundredths_and_inclusive():
    exactly = _decision_panel(_uniform_entries((0.30, 0.28, -0.5, 0.0, 0.9)))
    assert probe.apply_decision_rule(exactly)["branch"].startswith("a_")
    just_under = _decision_panel(_uniform_entries((0.30, 0.279, -0.5, 0.0, 0.9)))
    assert not probe.apply_decision_rule(just_under)["branch"].startswith("a_")


def test_branch_b_when_one_family_still_has_negative_iii_at_every_rank():
    entries = _uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9))
    entries["uniform"] = {rank: (0.30, 0.29, -30.0, -2.0, 0.99) for rank in probe.RANKS}
    decision = probe.apply_decision_rule(_decision_panel(entries))
    assert decision["branch"].startswith("b_")
    assert decision["hypothesis_H"]["confirmed"] is True
    assert decision["families_failing_convention_iii_at_rank16"] == ["uniform"]
    assert decision["families_routed_to_abstention"] == ["uniform"]
    assert (
        decision["families"]["uniform"][
            "confined_convention_iii_negative_at_every_measured_rank"
        ]
        is True
    )


def test_branch_c_when_the_outside_mask_split_cannot_explain_the_blowup():
    entries = _uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9))
    entries["layered"] = {rank: (0.30, 0.29, -0.5, 0.05, 0.10) for rank in probe.RANKS}
    decision = probe.apply_decision_rule(_decision_panel(entries))
    assert decision["branch"].startswith("c_")
    assert decision["hypothesis_H"]["refuted"] is True
    assert decision["hypothesis_H"]["blowup_located_outside_mask_by_family"][
        "layered"
    ] is False
    assert "do_not_freeze_v15" in decision["branch"]


def test_branch_c_takes_precedence_over_branch_a():
    """A refuted H must stop the probe even if the numbers otherwise look good."""
    entries = _uniform_entries((0.30, 0.29, -0.5, 0.90, 0.01))
    decision = probe.apply_decision_rule(_decision_panel(entries))
    assert decision["branch"].startswith("c_")


def test_branch_b_requires_its_own_precondition_of_a_negative_iii():
    """The measured r4e9 outcome: (iii) clears everywhere but (ii) loses too much.

    Branch (b) is only available when some family is still negative under
    convention (iii).  When every family clears (iii) and a family nonetheless
    breaks the frozen 0.02 tolerance on (ii), the frozen rule covers nothing and
    the probe must escalate rather than silently reporting (b).
    """
    entries = _uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9))
    # layered: clears (iii) but loses 0.05 of (ii), which is over the tolerance
    entries["layered"] = {rank: (0.386, 0.336, 0.077, 0.190, None) for rank in probe.RANKS}
    decision = probe.apply_decision_rule(_decision_panel(entries))
    assert decision["branch"].startswith("unclassified_")
    assert decision["families_failing_convention_iii_at_rank16"] == []
    assert decision["families_routed_to_abstention"] == []
    assert "layered" in decision["verdict"]
    assert "must not be frozen" in decision["verdict"]


def test_branch_b_verdict_always_names_at_least_one_family():
    """Guard against the empty-list verdict string that motivated the fix."""
    entries = _uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9))
    entries["uniform"] = {rank: (0.30, 0.29, -30.0, -2.0, 0.99) for rank in probe.RANKS}
    decision = probe.apply_decision_rule(_decision_panel(entries))
    assert decision["branch"].startswith("b_")
    assert decision["families_failing_convention_iii_at_rank16"]
    assert "for ;" not in decision["verdict"]
    for family in decision["families_failing_convention_iii_at_rank16"]:
        assert family in decision["verdict"]


def test_decision_reports_every_family_and_every_rank_separately():
    decision = probe.apply_decision_rule(
        _decision_panel(_uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9)))
    )
    assert sorted(decision["families"]) == ["layered", "marmousi", "uniform"]
    for family in decision["families"].values():
        assert sorted(family["ranks"]) == ["16", "32", "8"]
        for rank_item in family["ranks"].values():
            assert sorted(rank_item["unconfined_gain"]) == sorted(probe.CONVENTIONS)
            assert sorted(rank_item["confined_gain"]) == sorted(probe.CONVENTIONS)


def test_decision_rank_is_sixteen_as_the_spec_states():
    assert probe.DECISION_RANK == 16
    assert probe.CONFINED_CONVENTION_II_TOLERANCE == 0.02
    assert probe.CONFINED_CONVENTION_III_MINIMUM == 0.0


def test_families_without_a_blowup_are_not_counted_against_hypothesis_h():
    entries = _uniform_entries((0.30, 0.29, -0.5, 0.05, 0.9))
    # marmousi never blew up in r4e8; a positive (iii) means nothing to explain
    entries["marmousi"] = {rank: (0.12, 0.12, 0.13, 0.13, None) for rank in probe.RANKS}
    decision = probe.apply_decision_rule(_decision_panel(entries))
    assert "marmousi" in decision["hypothesis_H"][
        "families_without_blowup_nothing_to_explain"
    ]
    assert decision["hypothesis_H"]["refuted"] is False
    assert decision["branch"].startswith("a_")


# ---------------------------------------------------------------------------
# sealed-data and no-write attestations
# ---------------------------------------------------------------------------


def test_probe_never_names_validation_or_test_id_datasets():
    source = Path(probe.__file__).read_text(encoding="utf8")
    for token in ("validation", "test_id"):
        for line in source.splitlines():
            if token in line:
                # only attestation lines asserting they were NOT opened are allowed
                assert "False" in line or "not_" in line or "never" in line.lower(), line


def test_authorized_truth_ids_are_train_only():
    for sample_id in r4e8.ALLOWED_TRUTH_SAMPLE_IDS:
        assert sample_id.startswith("train_")


def test_panel_records_are_the_three_the_spec_names():
    assert probe.PANEL_IDS == (
        ("uniform", "train_uniform_00321"),
        ("layered", "train_layered_00564"),
        ("marmousi", "train_marmousi_00385"),
    )


def test_result_dir_is_the_only_authorized_write_root():
    assert probe.ALLOWED_WRITE_ROOTS == (probe.RESULT_DIR,)
    assert probe.RESULT_DIR.name == "r4e9_mask_confined_oracle_20260826"


def test_r4e8_artifacts_are_read_only_inputs():
    assert probe.R4E8_TERMINAL_PATH.exists()
    with pytest.raises(probe.ProbeContractError):
        probe.assert_write_allowed(probe.R4E8_TERMINAL_PATH)


def test_stored_r4e8_reference_still_hashes_to_what_this_run_read():
    reference = probe.load_r4e8_reference()
    again = probe.parent_runtime.sha256_file(probe.R4E8_TERMINAL_PATH)
    assert reference["sha256"] == again
    payload = json.loads(probe.R4E8_TERMINAL_PATH.read_text(encoding="utf8"))
    assert payload["training"]["checkpoints_written"] == 0
