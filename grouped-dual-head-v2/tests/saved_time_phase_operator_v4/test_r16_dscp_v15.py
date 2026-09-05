"""V15 verification gates: mask confinement, per-record smoke, oracle, units, VRAM.

Gate numbering follows the frozen v15 task list:

1. the confined correction is exactly zero outside the mask and the output there
   is bit-identical to the parent; inside the mask it equals the unconfined one;
2. the mask is derived from parent frame energy only, never from truth;
3. smoke loss reduction is per record, and the v14 cross-record formula is shown
   to report a near-perfect reduction on a stream that learned almost nothing;
4. the oracle upper bound is recomputed on the scored record, and the three
   voided v14 constants appear nowhere in v15 non-test code;
5. the coefficient-energy term is dimensionless;
6. peak VRAM is recorded and its gate cannot pass on an unmeasured or zero peak.
"""
from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import smoke_gates as v14_smoke_gates
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v15 import (
    CANDIDATE,
    LOSS_REDUCTION_MIN,
    ORACLE_GAIN_FRACTION,
    OracleProvenanceRefusal,
    SmokeAccountingRefusal,
    SmokePromotionRefusal,
    V15ProductionBackend,
    VOIDED_ORACLE_PROVENANCE,
    VramMeasurementRefusal,
    acceptance_convention_gain,
    oracle_gain_gate,
    record_oracle_bound,
    refuse_promotion_from_smoke,
    smoke_gates_v15,
    v14_cross_record_loss_reduction,
    validate_oracle_provenance,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v3 import (
    ACCEPTANCE_CONVENTION,
    TAU,
    ConfinementLeak,
    CudaPeakVram,
    PerRecordLossLedger,
    TRAIN_VRAM_LIMIT_BYTES,
    V15ContractError,
    apply_confined_correction,
    bitwise_identical,
    coefficient_energy_definition,
    confine_correction,
    confinement_invariants,
    convention_metrics,
    dimensionless_coefficient_energy,
    frame_energy,
    full_time_keep_mask,
    loss_specification,
    mask_provenance,
    masked_confined_loss,
    parent_energy_keep_mask,
    vram_gate,
)
import scripts.train_r16_dscp_v15 as cli

ROOT = Path(cli.ROOT)
V15_NON_TEST_FILES = (cli.SCRIPT, cli.ENGINE, cli.HARNESS, cli.CONFIG)
#: the three v14 oracle constants, void; kept here only as search needles
VOIDED_CONSTANT_NEEDLES = (
    "0.2224565778",
    "0.373345452",
    "0.2978927157",
    ".2224565778",
    ".373345452",
    ".2978927157",
)
TIME, HEIGHT, WIDTH, RANK = 24, 5, 6, 4
DROPPED = (10, 11, 12, 13)


def _parent_with_quiet_frames(*, quiet=DROPPED, value: float = 0.0) -> torch.Tensor:
    torch.manual_seed(11)
    parent = torch.randn(TIME, HEIGHT, WIDTH)
    for index in quiet:
        parent[index] = value
    return parent


def _keep(parent: torch.Tensor) -> torch.Tensor:
    keep, _floor = parent_energy_keep_mask(parent, tau=TAU)
    return keep


def _correction() -> torch.Tensor:
    torch.manual_seed(23)
    return torch.randn(TIME, HEIGHT, WIDTH)


class _MaskStubBackend(V15ProductionBackend):
    """Only the attributes the mask and VRAM paths need; no data, no optimizer."""

    def __init__(self, bases: torch.Tensor, *, tau: float = TAU, device: str = "cpu"):
        self.candidate = SimpleNamespace(bases=bases)
        self.tau = float(tau)
        self.device = torch.device(device)
        self.vram = CudaPeakVram(self.device).reset()
        self.peak_bytes = 0
        self.mask_events = []


class _LossStubBackend(_MaskStubBackend):
    def __init__(self, bases: torch.Tensor, coefficient: torch.Tensor, **kwargs):
        super().__init__(bases, **kwargs)
        self._fixed = coefficient

    def _coefficients(self, features, route_index, abstain, condition):
        return self._fixed


def _bases() -> torch.Tensor:
    torch.manual_seed(5)
    return torch.linalg.qr(torch.randn(TIME, RANK))[0][None]


def _prepared(parent: torch.Tensor, k1: int) -> SimpleNamespace:
    return SimpleNamespace(
        features=torch.zeros(1, 1),
        route_index=0,
        condition=1.0,
        args=(None,) * 7 + (parent,),
        public=SimpleNamespace(observed_indices=(0, k1)),
    )


# ---------------------------------------------------------------------------
# gate 1: exact confinement, bit-identical outside the mask
# ---------------------------------------------------------------------------


def test_gate1_confined_correction_is_identically_zero_outside_the_mask():
    parent = _parent_with_quiet_frames()
    keep = _keep(parent)
    correction = _correction()
    confined = confine_correction(correction, keep)
    assert keep.tolist() == [index not in DROPPED for index in range(TIME)]
    assert float(confined[~keep].abs().max()) == 0.0
    assert torch.equal(confined[~keep], torch.zeros_like(confined[~keep]))
    assert torch.equal(confined[keep], correction[keep])


def test_gate1_output_is_bitwise_identical_to_parent_outside_the_mask():
    parent = _parent_with_quiet_frames(value=-0.0)
    keep = _keep(parent)
    correction = _correction()
    corrected = apply_confined_correction(parent, correction, keep)
    assert bitwise_identical(corrected[~keep], parent[~keep])
    assert bitwise_identical(corrected, torch.where(keep[:, None, None], parent + confine_correction(correction, keep), parent))
    assert torch.equal(corrected[keep], (parent + correction)[keep])
    naive = parent[~keep] + confine_correction(correction, keep)[~keep]
    assert torch.equal(naive, parent[~keep])
    assert not bitwise_identical(naive, parent[~keep])
    invariants = confinement_invariants(parent, correction, keep)
    assert invariants["correction_identically_zero_outside_mask"]
    assert invariants["max_abs_applied_correction_outside_mask"] == 0.0
    assert invariants["output_bitwise_identical_to_parent_outside_mask"]
    assert invariants["dropped_frame_count"] == len(DROPPED)


def test_gate1_confinement_is_enforced_by_construction_not_by_tolerance():
    parent = _parent_with_quiet_frames()
    keep = _keep(parent)
    with pytest.raises(ValueError):
        confine_correction(_correction(), keep[:-1])
    with pytest.raises(ValueError):
        confine_correction(_correction(), keep.float())
    with pytest.raises(ValueError):
        apply_confined_correction(parent, _correction()[:-1], keep)
    assert "ConfinementLeak" in inspect.getsource(confine_correction)
    assert issubclass(ConfinementLeak, V15ContractError)


def test_gate1_engine_materialization_is_confined_and_keeps_parent_bits():
    parent = _parent_with_quiet_frames(value=-0.0)[None]
    backend = _MaskStubBackend(_bases())
    coefficient = torch.randn(1, RANK, HEIGHT, WIDTH)
    corrected = backend._materialize(parent, coefficient, 0, 3)
    keep, _floor = full_time_keep_mask(parent[0], k1=3, tau=TAU)
    assert not bool(keep[:4].any())
    assert bitwise_identical(corrected[0][~keep], parent[0][~keep])
    assert float((corrected[0] - parent[0])[~keep].abs().max()) == 0.0
    assert backend.mask_events and backend.mask_events[-1]["confined"]
    assert backend.mask_ledger()["unconfined_materializations"] == 0
    assert backend.mask_ledger()["driven_by"] == "parent_frame_energy"


def test_gate1_inside_mask_values_equal_the_unconfined_correction():
    parent = _parent_with_quiet_frames()[None]
    backend = _MaskStubBackend(_bases())
    coefficient = torch.randn(1, RANK, HEIGHT, WIDTH)
    keep, _floor = full_time_keep_mask(parent[0], k1=3, tau=TAU)
    unconfined = torch.einsum("tr,rhw->thw", backend.candidate.bases[0], coefficient[0])
    from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import c1_causal_mask

    unconfined = unconfined * c1_causal_mask(TIME, 3)[:, None, None]
    unconfined = torch.cat((torch.zeros_like(unconfined[:, :1]), unconfined[:, 1:]), dim=1)
    corrected = backend._materialize(parent, coefficient, 0, 3)
    assert torch.equal(corrected[0][keep], (parent[0] + unconfined)[keep])
    assert float((corrected[0] - parent[0])[~keep].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# gate 2: the mask is parent-energy only
# ---------------------------------------------------------------------------


def test_gate2_mask_uses_parent_energy_not_truth_energy():
    parent = _parent_with_quiet_frames(quiet=(10, 11, 12, 13))
    truth = _parent_with_quiet_frames(quiet=(2, 3, 4, 5))
    parent_keep, _pf = parent_energy_keep_mask(parent, tau=TAU)
    truth_keep, _tf = parent_energy_keep_mask(truth, tau=TAU)
    assert parent_keep.tolist() != truth_keep.tolist()
    assert [i for i, keep in enumerate(parent_keep.tolist()) if not keep] == [10, 11, 12, 13]
    assert [i for i, keep in enumerate(truth_keep.tolist()) if not keep] == [2, 3, 4, 5]
    correction = _correction()
    confined = confine_correction(correction, parent_keep)
    assert float(confined[[10, 11, 12, 13]].abs().max()) == 0.0
    assert float(confined[[2, 3, 4, 5]].abs().max()) > 0.0


def test_gate2_mask_functions_have_no_truth_parameter():
    for function in (parent_energy_keep_mask, full_time_keep_mask):
        names = set(inspect.signature(function).parameters)
        assert not {name for name in names if "truth" in name}
    assert "truth" not in set(inspect.signature(V15ProductionBackend.keep_mask).parameters)
    provenance = mask_provenance()
    assert provenance["driven_by"] == "parent_frame_energy"
    assert provenance["uses_truth_frame_energy"] is False
    assert provenance["deployment_computable"] is True
    assert provenance["replaces_c1_causal_ramp"] is False
    assert provenance["layered_on_top_of_c1_causal_ramp"] is True


def test_gate2_engine_drops_the_parent_quiet_frames_even_when_truth_is_loud():
    parent = _parent_with_quiet_frames(quiet=(10, 11, 12, 13))[None]
    truth = _parent_with_quiet_frames(quiet=(2, 3, 4, 5))[None]
    backend = _MaskStubBackend(_bases())
    corrected = backend._materialize(parent, torch.randn(1, RANK, HEIGHT, WIDTH), 0, 3)
    delta = (corrected - parent)[0]
    assert float(delta[[10, 11, 12, 13]].abs().max()) == 0.0
    assert float(delta[[8, 9, 14, 15]].abs().max()) > 0.0
    truth_energy = frame_energy(truth[0])
    assert float(truth_energy[[10, 11, 12, 13]].max()) > 0.0


def test_gate2_full_time_mask_is_zero_through_k1():
    parent = _parent_with_quiet_frames()
    keep, floor = full_time_keep_mask(parent, k1=6, tau=TAU)
    assert not bool(keep[:7].any()) and bool(keep[7:].any()) and floor > 0.0


# ---------------------------------------------------------------------------
# gate 3: per-record smoke loss accounting
# ---------------------------------------------------------------------------

_RECORDS = ("uniform", "layered", "marmousi")


def _round_robin_stream() -> list[tuple[str, float]]:
    """192 updates over three records; only marmousi falls, and only at the end."""
    stream: list[tuple[str, float]] = []
    for update in range(192):
        record = _RECORDS[update % 3]
        cycle = update // 3
        if record == "uniform":
            value = 1767.36 - 0.05 * cycle
        elif record == "layered":
            value = 900.0 - 0.02 * cycle
        else:
            value = 3.0 if cycle < 63 else 0.5876
        stream.append((record, value))
    return stream


def _score_rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"train_{family}_000{index}",
            "family": family,
            "mean_frame_rel_l2": 0.30,
            "parent_mean_frame_rel_l2": 0.31,
            "aggregate_rel_l2": 0.30,
            "parent_rel_l2": 0.31,
            "nonworse": True,
            "finite": True,
            "ledger_digest": f"digest_{index}",
        }
        for index, family in enumerate(_RECORDS)
    ]


def test_gate3_v14_cross_record_reduction_is_near_perfect_on_a_stream_that_barely_moved():
    stream = _round_robin_stream()
    losses = [value for _record, value in stream]
    assert len(losses) == 192 and len(losses) % 3 == 0
    assert stream[0][0] == "uniform" and stream[-1][0] == "marmousi"
    reduction = v14_cross_record_loss_reduction(losses)
    assert reduction > 0.99
    resources = {
        "wall_s": 10.0,
        "peak_bytes": 0,
        "checkpoint_bytes": 1024,
        "space_passed": True,
    }
    gates, metrics = v14_smoke_gates(losses[0], losses[-1], _score_rows(), resources)
    assert gates["loss"]["value"] > 0.99 and gates["loss"]["passed"]
    assert gates["vram"]["passed"] and gates["vram"]["value"] == 0
    ledger = PerRecordLossLedger()
    for record, value in stream:
        ledger.observe(record, value)
    per_record = ledger.per_record()
    assert set(per_record) == set(_RECORDS)
    assert per_record["uniform"]["loss_reduction"] < 0.01
    assert per_record["layered"]["loss_reduction"] < 0.01
    v15_gate = ledger.gate(threshold=LOSS_REDUCTION_MIN)
    assert v15_gate["passed"] is False
    assert sorted(v15_gate["failing_records"]) == ["layered", "uniform"]
    assert v15_gate["computed_per_record"] and not v15_gate["cross_record_first_last_used"]
    assert v15_gate["is_learning_or_convergence_evidence"] is False


def test_gate3_ledger_cannot_express_a_cross_record_first_and_last():
    ledger = PerRecordLossLedger()
    for record, value in _round_robin_stream():
        ledger.observe(record, value)
    assert not [
        name
        for name in dir(ledger)
        if name in {"losses", "flat", "all_losses", "ordered_losses"}
    ]
    reductions = ledger.reductions()
    assert set(reductions) == set(_RECORDS)
    assert all(0.0 <= value <= 1.0 for value in reductions.values())


def test_gate3_every_record_must_clear_the_inherited_threshold_alone():
    ledger = PerRecordLossLedger()
    for record, first, last in (("a", 10.0, 1.0), ("b", 10.0, 1.0), ("c", 10.0, 9.9)):
        ledger.observe(record, first)
        ledger.observe(record, last)
    gate = ledger.gate(threshold=LOSS_REDUCTION_MIN)
    assert gate["threshold"] == 0.80 and gate["failing_records"] == ["c"] and not gate["passed"]
    ledger.observe("d", 1.0)
    assert ledger.gate(threshold=LOSS_REDUCTION_MIN)["records_with_fewer_than_two_observations"] == ["d"]


def test_gate3_smoke_gates_v15_refuses_non_per_record_accounting():
    with pytest.raises(SmokeAccountingRefusal):
        smoke_gates_v15(
            ledger=[1.0, 0.1],
            score_records=_score_rows(),
            resources={"vram": {"device_type": "cuda", "measured": True, "peak_reserved_bytes": 1}},
            oracle_by_record={},
        )
    with pytest.raises(SmokeAccountingRefusal):
        v14_cross_record_loss_reduction([1.0])


def test_gate3_smoke_evidence_cannot_promote():
    with pytest.raises(SmokePromotionRefusal):
        refuse_promotion_from_smoke("long")
    with pytest.raises(SmokePromotionRefusal):
        cli.validate_stage_authorization(
            {
                "candidate": CANDIDATE,
                "stage": "pilot",
                "preregistration_sha256": cli.FROZEN_BINDINGS["preregistration"][1],
                "lead_authorized": True,
                "acknowledges_post_hoc_rule_revision": True,
                "acknowledges_inherited_vetoes": True,
                "promoted_from": "smoke",
            },
            "pilot",
        )


# ---------------------------------------------------------------------------
# gate 4: the oracle is recomputed on the scored record
# ---------------------------------------------------------------------------


def _oracle_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = _parent_with_quiet_frames()
    torch.manual_seed(seed)
    truth = parent + 0.2 * torch.randn_like(parent)
    basis = torch.linalg.qr(torch.randn(TIME, RANK, dtype=torch.float64))[0]
    return parent, truth, basis


def test_gate4_oracle_bound_is_recomputed_on_the_record_it_scores():
    first = record_oracle_bound(
        sample_id="train_uniform_00321",
        family="uniform",
        **dict(zip(("parent_future", "truth_future", "basis_future"), _oracle_inputs(1))),
        spatial_chunk=7,
    )
    second = record_oracle_bound(
        sample_id="train_marmousi_00385",
        family="marmousi",
        **dict(zip(("parent_future", "truth_future", "basis_future"), _oracle_inputs(2))),
        spatial_chunk=7,
    )
    for payload in (first, second):
        assert payload["recomputed_on_the_scored_record"] is True
        assert payload["transplanted_constants_used"] is False
        assert payload["synthetic_records_used"] is False
        assert payload["is_achieved_result"] is False
        assert payload["acceptance_convention"] == ACCEPTANCE_CONVENTION
        assert math.isfinite(payload["acceptance_convention_gain"])
        assert payload["confinement_invariants"]["max_abs_applied_correction_outside_mask"] == 0.0
    assert first["acceptance_convention_gain"] != second["acceptance_convention_gain"]
    with pytest.raises(OracleProvenanceRefusal):
        validate_oracle_provenance("train_uniform_00321", second)
    with pytest.raises(OracleProvenanceRefusal):
        validate_oracle_provenance(
            "train_uniform_00321", {**first, "transplanted_constants_used": True}
        )


def test_gate4_oracle_gate_is_per_record_and_refuses_a_missing_bound():
    rows = _score_rows()
    bounds = {}
    for index, row in enumerate(rows):
        bounds[row["sample_id"]] = record_oracle_bound(
            sample_id=row["sample_id"],
            family=row["family"],
            **dict(zip(("parent_future", "truth_future", "basis_future"), _oracle_inputs(index + 3))),
            spatial_chunk=7,
        )
    gate = oracle_gain_gate(rows, bounds, fraction=ORACLE_GAIN_FRACTION)
    assert gate["recomputed_per_record"] and not gate["transplanted_constants_used"]
    assert set(gate["per_record"]) == {row["sample_id"] for row in rows}
    for sample_id, entry in gate["per_record"].items():
        assert entry["convention"] == ACCEPTANCE_CONVENTION
        assert entry["oracle_upper_bound_gain"] == bounds[sample_id]["acceptance_convention_gain"]
        assert entry["achieved_gain"] == pytest.approx(
            acceptance_convention_gain(next(r for r in rows if r["sample_id"] == sample_id))
        )
    with pytest.raises(OracleProvenanceRefusal):
        oracle_gain_gate(rows, {}, fraction=ORACLE_GAIN_FRACTION)
    swapped = {rows[0]["sample_id"]: bounds[rows[1]["sample_id"]]}
    with pytest.raises(OracleProvenanceRefusal):
        oracle_gain_gate(rows[:1], swapped, fraction=ORACLE_GAIN_FRACTION)


def test_gate4_a_nonpositive_oracle_bound_cannot_be_cleared():
    row = _score_rows()[0]
    bound = record_oracle_bound(
        sample_id=row["sample_id"],
        family=row["family"],
        **dict(zip(("parent_future", "truth_future", "basis_future"), _oracle_inputs(9))),
        spatial_chunk=7,
    )
    gate = oracle_gain_gate([row], {row["sample_id"]: {**bound, "acceptance_convention_gain": 0.0}})
    assert gate["passed"] is False
    assert gate["per_record"][row["sample_id"]]["oracle_bound_positive"] is False


def test_gate4_voided_v14_constants_appear_nowhere_in_v15_code():
    for path in V15_NON_TEST_FILES:
        text = Path(path).read_text()
        for needle in VOIDED_CONSTANT_NEEDLES:
            assert needle not in text, f"{needle} found in {path}"
    assert VOIDED_ORACLE_PROVENANCE["status"] == "void"
    assert VOIDED_ORACLE_PROVENANCE["numeric_values_present_in_v15_code"] is False
    assert "synthetic_marmousi" in VOIDED_ORACLE_PROVENANCE["v14_source_records"]
    frozen = json.loads(Path(cli.PREREG).read_text())
    listed = frozen["v14_bad_gates_that_must_be_replaced"]["oracle_gain_constants"]["voided_constants"]
    assert len(listed) == 3
    in_v14_gate = [
        float(match) for match in re.findall(r"\.\d{10,}", inspect.getsource(v14_smoke_gates))
    ]
    assert len(in_v14_gate) == 3
    for value in listed:
        rendered = f"{value}"
        assert not any(rendered in Path(path).read_text() for path in V15_NON_TEST_FILES)
        assert any(abs(found - float(value)) < 5.0e-10 for found in in_v14_gate)
    for found in in_v14_gate:
        assert not any(repr(found) in Path(path).read_text() for path in V15_NON_TEST_FILES)


# ---------------------------------------------------------------------------
# gate 5: the coefficient energy term is dimensionless
# ---------------------------------------------------------------------------


def test_gate5_coefficient_energy_is_invariant_under_a_common_rescaling():
    torch.manual_seed(31)
    parent = _parent_with_quiet_frames().double()[None]
    keep = _keep(parent[0])
    coefficients = torch.randn(1, RANK, HEIGHT, WIDTH, dtype=torch.float64)
    base = float(dimensionless_coefficient_energy(coefficients, parent, keep))
    for scale in (1.0e-3, 7.3, 1.0e3):
        scaled = float(dimensionless_coefficient_energy(coefficients * scale, parent * scale, keep))
        assert scaled == pytest.approx(base, rel=1.0e-12)
        v14_term = float((coefficients * scale).square().mean())
        assert v14_term == pytest.approx(float(coefficients.square().mean()) * scale**2, rel=1.0e-12)
        assert not math.isclose(v14_term, float(coefficients.square().mean()), rel_tol=1.0e-6)


def test_gate5_masked_loss_coefficient_term_survives_a_common_rescaling():
    torch.manual_seed(37)
    parent = _parent_with_quiet_frames()[None]
    truth = parent + 0.1 * torch.randn_like(parent)
    keep = _keep(parent[0])
    coefficients = torch.randn(1, RANK, HEIGHT, WIDTH)
    first = masked_confined_loss(parent, truth, coefficients, parent, keep)
    scale = 250.0
    second = masked_confined_loss(
        parent * scale, truth * scale, coefficients * scale, parent * scale, keep
    )
    assert float(second["normalized_coefficient_energy"]) == pytest.approx(
        float(first["normalized_coefficient_energy"]), rel=1.0e-4
    )
    assert float(second["frame_relative_l2_squared"]) == pytest.approx(
        float(first["frame_relative_l2_squared"]), rel=1.0e-4
    )


def test_gate5_recorded_definition_and_rationale_are_complete():
    definition = coefficient_energy_definition()
    assert definition["v14_units"] == "amplitude_squared"
    assert definition["v15_units"] == "dimensionless"
    assert definition["normalizer"] == "mean_squared_parent_amplitude_on_retained_future_frames"
    assert definition["normalizer_uses_truth"] is False
    assert definition["weight_unchanged_from_v14"] == 1.0e-4
    assert definition["alternative_considered"] == "residual_energy_normalization"
    assert len(definition["why_parent_energy_was_chosen"]) >= 3
    specification = loss_specification()
    assert specification["weights_unchanged_from_v14"] is True
    assert specification["pde_residual_used"] is False
    assert specification["deployment_time_truth_supervision"] is False
    assert specification["acceptance_convention"] == ACCEPTANCE_CONVENTION
    assert specification["is_learning_or_convergence_evidence"] is False
    evidence = cli.static_evidence()
    assert evidence["coefficient_energy_dimensional_fix"] == definition
    assert evidence["acceptance"]["convention"] == ACCEPTANCE_CONVENTION
    assert evidence["acceptance"]["validation_test_rel_l2_max"] == 0.05
    assert evidence["acceptance"]["modified"] is False
    assert evidence["sealed_future_wavefields_opened"] is False


def test_gate5_loss_is_restricted_to_the_parent_mask_frames():
    torch.manual_seed(41)
    parent = _parent_with_quiet_frames()[None]
    truth = parent + 0.1 * torch.randn_like(parent)
    keep = _keep(parent[0])
    coefficients = torch.zeros(1, RANK, HEIGHT, WIDTH)
    reference = masked_confined_loss(parent, truth, coefficients, parent, keep)
    perturbed = truth.clone()
    perturbed[:, list(DROPPED)] += 1.0
    changed = masked_confined_loss(parent, perturbed, coefficients, parent, keep)
    assert float(changed["frame_relative_l2_squared"]) == pytest.approx(
        float(reference["frame_relative_l2_squared"]), rel=1.0e-6
    )


# ---------------------------------------------------------------------------
# gate 6: peak VRAM is measured and its gate is non-trivial
# ---------------------------------------------------------------------------


def test_gate6_unmeasured_or_zero_peak_cannot_pass_the_vram_gate():
    cpu = CudaPeakVram(torch.device("cpu")).reset()
    cpu.sample()
    payload = cpu.payload()
    assert payload["measured"] is False and payload["peak_reserved_bytes"] == 0
    gate = vram_gate(payload)
    assert gate["passed"] is False and gate["nontrivial"] is False
    assert gate["threshold"] == TRAIN_VRAM_LIMIT_BYTES
    zero_cuda = vram_gate(
        {"device_type": "cuda", "measured": False, "peak_reserved_bytes": 0}
    )
    assert zero_cuda["passed"] is False
    over = vram_gate(
        {
            "device_type": "cuda",
            "measured": True,
            "peak_reserved_bytes": TRAIN_VRAM_LIMIT_BYTES + 1,
        }
    )
    assert over["passed"] is False and over["nontrivial"] is True
    inside = vram_gate(
        {"device_type": "cuda", "measured": True, "peak_reserved_bytes": 6941 * 1024**2}
    )
    assert inside["passed"] is True and inside["nontrivial"] is True


def test_gate6_smoke_gates_refuse_resources_without_a_vram_measurement():
    ledger = PerRecordLossLedger()
    ledger.observe("a", 10.0)
    ledger.observe("a", 1.0)
    with pytest.raises(VramMeasurementRefusal):
        smoke_gates_v15(
            ledger=ledger,
            score_records=_score_rows(),
            resources={"space_passed": True},
            oracle_by_record={},
        )


def test_gate6_every_cached_path_samples_the_peak():
    for name in (
        "preload",
        "cached_prepare",
        "update_cached",
        "measure_cached",
        "backward_cached",
        "score_cached",
        "update",
        "score",
        "resources",
        "_forward_loss",
        "_materialize",
    ):
        assert name in V15ProductionBackend.__dict__, name
        assert "_sample_vram" in inspect.getsource(V15ProductionBackend.__dict__[name])
    source = inspect.getsource(CudaPeakVram.sample)
    assert "max_memory_reserved" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required to measure a real peak")
def test_gate6_real_cuda_peak_is_nonzero_and_the_gate_uses_it():
    device = torch.device("cuda:0")
    warm = torch.zeros(64, 64, device=device)
    monitor = CudaPeakVram(device).reset()
    assert monitor.sample() > 0
    small = torch.zeros(256, 256, device=device)
    small.add_(1.0)
    monitor.sample()
    payload = monitor.payload()
    assert payload["measured"] is True
    assert payload["peak_reserved_bytes"] > 0
    assert payload["peak_allocated_bytes"] > 0
    assert payload["source"] == "torch.cuda.max_memory_reserved"
    gate = vram_gate(payload)
    assert gate["nontrivial"] and gate["passed"] and gate["value"] == payload["peak_reserved_bytes"]
    del small, warm
    torch.cuda.empty_cache()


def test_gate6_backend_peak_tracks_the_monitor():
    backend = _MaskStubBackend(_bases())
    assert backend.peak_bytes == 0
    backend.vram.peak_reserved_bytes = 4096
    assert backend._sample_vram() == 4096 and backend.peak_bytes == 4096
    payload = backend.vram_payload()
    assert payload["backend_peak_bytes"] == 4096
    assert "update_cached" in payload["sampled_paths"]
    assert payload["source"] == "torch.cuda.max_memory_reserved"


# ---------------------------------------------------------------------------
# forward-loss wiring, bindings, authorization, scope statements
# ---------------------------------------------------------------------------


def test_forward_loss_uses_the_masked_confined_loss_and_the_parent_mask():
    parent = _parent_with_quiet_frames()[None]
    truth_future = parent[:, 4:] + 0.05 * torch.randn_like(parent[:, 4:])
    coefficient = torch.randn(1, RANK, HEIGHT, WIDTH) * 0.01
    backend = _LossStubBackend(_bases(), coefficient)
    losses, returned = backend._forward_loss(_prepared(parent, 3), truth_future)
    assert returned is coefficient
    assert set(losses) == {
        "total",
        "frame_relative_l2_squared",
        "late_third_relative_l2_squared",
        "temporal_difference",
        "normalized_coefficient_energy",
    }
    assert torch.isfinite(losses["total"])
    assert backend.mask_events[-1]["dropped_frames"] >= len(DROPPED)


def test_convention_metrics_expose_all_three_and_name_the_acceptance_one():
    error = torch.tensor([1.0, 2.0, 3.0, 4.0])
    energy = torch.tensor([10.0, 20.0, 30.0, 40.0])
    metrics = convention_metrics(error, energy)
    assert set(metrics) == {
        "unmasked_per_frame_unsquared_mean",
        "masked_energy_floored_per_frame",
        "global_energy_rel_l2",
    }
    assert metrics["global_energy_rel_l2"] == pytest.approx(math.sqrt(10.0 / 100.0))
    assert ACCEPTANCE_CONVENTION == "global_energy_rel_l2"


def test_frozen_bindings_are_verified_and_drift_refuses(tmp_path):
    report = cli.verify_bindings(require_parent=Path(cli.PARENT_PATH).exists())
    assert report["passed"] and not report["drift"]
    assert report["checked"]["preregistration"]["status"] == "match"
    assert report["checked"]["r4e9_probe_reference"]["status"] == "match"
    assert report["checked"]["r4e9_terminal_reference"]["status"] == "match"
    drifted = tmp_path / "prereg.json"
    drifted.write_text("{}")
    original = cli.FROZEN_BINDINGS["preregistration"]
    cli.FROZEN_BINDINGS["preregistration"] = (drifted, original[1])
    try:
        with pytest.raises(cli.V15BindingRefusal):
            cli.verify_bindings(require_parent=False)
    finally:
        cli.FROZEN_BINDINGS["preregistration"] = original


def test_parent_checkpoint_is_read_only_for_this_candidate():
    audit = cli.parent_write_audit()
    assert audit["status"] in {"read_only_unchanged", "measurement_gap"}
    if audit["reachable"]:
        assert audit["sha256_matches_frozen"] is True and audit["writes"] == 0
    else:
        assert audit["writes"] is None
    assert audit["opened_for_write_by_v15"] is False


def test_stage_authorization_is_required_and_complete(tmp_path):
    with pytest.raises(cli.StageAuthorizationRequired):
        cli.load_stage_authorization(tmp_path / "absent.json", "smoke")
    complete = {
        "candidate": CANDIDATE,
        "stage": "smoke",
        "preregistration_sha256": cli.FROZEN_BINDINGS["preregistration"][1],
        "lead_authorized": True,
        "acknowledges_post_hoc_rule_revision": True,
        "acknowledges_inherited_vetoes": True,
    }
    assert cli.validate_stage_authorization(complete, "smoke")["stage"] == "smoke"
    for key in ("lead_authorized", "acknowledges_post_hoc_rule_revision", "acknowledges_inherited_vetoes"):
        with pytest.raises(cli.StageAuthorizationRequired):
            cli.validate_stage_authorization({**complete, key: False}, "smoke")
    with pytest.raises(cli.StageAuthorizationRequired):
        cli.validate_stage_authorization(complete, "pilot")
    with pytest.raises(cli.StageAuthorizationRequired):
        cli.validate_stage_authorization({**complete, "preregistration_sha256": "0" * 64}, "smoke")
    with pytest.raises(cli.StageAuthorizationRequired):
        cli.validate_stage_authorization({k: v for k, v in complete.items() if k != "stage"}, "smoke")


def test_sealed_splits_stay_sealed():
    for split in ("validation", "test_id"):
        with pytest.raises(cli.SealedSplitRefusal):
            cli.sealed_split_guard(split)
    cli.sealed_split_guard("train")
    sealed_authorization = {
        "candidate": CANDIDATE,
        "stage": "validation-once",
        "preregistration_sha256": cli.FROZEN_BINDINGS["preregistration"][1],
        "lead_authorized": True,
        "acknowledges_post_hoc_rule_revision": True,
        "acknowledges_inherited_vetoes": True,
    }
    with pytest.raises(cli.SealedSplitRefusal):
        cli.validate_stage_authorization(sealed_authorization, "validation-once")
    assert cli.validate_stage_authorization(
        {**sealed_authorization, "sealed_chain_verified": True}, "validation-once"
    )


def _valid_smoke_authorization_payload(**overrides) -> dict[str, object]:
    payload = {
        "candidate": CANDIDATE,
        "stage": "smoke",
        "preregistration_sha256": cli.FROZEN_BINDINGS["preregistration"][1],
        "lead_authorized": True,
        "acknowledges_post_hoc_rule_revision": True,
        "acknowledges_inherited_vetoes": True,
    }
    payload.update(overrides)
    return payload


def _write_authorization(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_smoke_handler_refuses_without_authorization_and_without_a_factory(tmp_path):
    # (a) no authorization file at all -> refuse
    args = SimpleNamespace(authorization=str(tmp_path / "missing.json"))
    with pytest.raises(cli.StageAuthorizationRequired):
        cli.smoke_handler(args)

    path = _write_authorization(
        tmp_path / "smoke.json", _valid_smoke_authorization_payload()
    )
    valid = SimpleNamespace(authorization=str(path))

    # (b) valid authorization but an explicit factory=None (implementation-only
    #     caller) -> still refuse, and nothing is constructed
    with pytest.raises(cli.StageAuthorizationRequired):
        cli.smoke_handler(valid, backend_factory=None)

    # (c) valid authorization and a factory -> allowed through to the runner
    seen: dict[str, object] = {}

    def factory(authorization):
        seen["authorization"] = dict(authorization)
        return {"backend": "stub", "records": [0]}

    def runner(**bundle):
        seen["bundle"] = bundle
        return {"status": "stub_ran"}

    assert cli.smoke_handler(valid, backend_factory=factory, smoke_runner=runner) == {
        "status": "stub_ran"
    }
    assert seen["authorization"]["stage"] == "smoke"
    assert seen["authorization"]["authorization_sha256"] == cli.sha256_file(path)
    assert seen["bundle"] == {"backend": "stub", "records": [0]}

    # the two acknowledgement bits are load-bearing: dropping either one refuses
    for missing in (
        "acknowledges_post_hoc_rule_revision",
        "acknowledges_inherited_vetoes",
    ):
        incomplete = _write_authorization(
            tmp_path / f"missing_{missing}.json",
            {
                key: value
                for key, value in _valid_smoke_authorization_payload().items()
                if key != missing
            },
        )
        with pytest.raises(cli.StageAuthorizationRequired):
            cli.smoke_handler(
                SimpleNamespace(authorization=str(incomplete)), backend_factory=factory
            )
        falsified = _write_authorization(
            tmp_path / f"false_{missing}.json",
            _valid_smoke_authorization_payload(**{missing: False}),
        )
        with pytest.raises(cli.StageAuthorizationRequired):
            cli.smoke_handler(
                SimpleNamespace(authorization=str(falsified)), backend_factory=factory
            )
    assert seen["bundle"] == {"backend": "stub", "records": [0]}


def test_smoke_handler_defaults_to_the_production_factory():
    """A bare CLI call reaches the real factory, not a refusal and not a stub."""
    assert cli.HANDLERS["smoke"] is cli.smoke_handler
    signature = inspect.signature(cli.smoke_handler)
    assert signature.parameters["backend_factory"].default is cli.PRODUCTION_FACTORY
    assert cli.PRODUCTION_FACTORY is not None
    source = inspect.getsource(cli.smoke_handler)
    assert "build_v15_smoke_bundle" in source


def test_production_factory_supplies_every_run_v15_smoke_argument():
    required = {
        name
        for name, parameter in inspect.signature(cli.run_v15_smoke).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    assert required == {
        "backend",
        "records",
        "quick_gate",
        "checkpoint",
        "resource_snapshot",
        "terminal",
        "oracle",
        "lineage",
        "effective",
        "parent",
    }
    source = inspect.getsource(cli.build_v15_smoke_bundle)
    for name in required | {"record_key"}:
        assert f'"{name}"' in source


def test_production_factory_binds_train_only_and_rechecks_bindings():
    source = inspect.getsource(cli.build_v15_smoke_bundle)
    assert "verify_bindings(require_parent=True)" in source
    assert "sealed_split_guard" in source
    assert 'backend.split != "train"' in source
    access = cli.build_smoke_access_authorization(
        {"authorization_path": "/tmp/a.json", "authorization_sha256": "abc"},
        cli.build_smoke_lineage(cli.v15_run_digest()),
    )
    assert access["schema"] == cli.AUTH_SCHEMA
    assert access["status"] == "authorized"
    assert access["split"] == "train"
    assert access["sealed_splits_authorized"] == []
    assert access["mode"] == "smoke"
    assert access["parent_sha256"] == cli.PARENT_SHA256
    assert access["gates_sha256"] is not None
    assert access["authorization_digest"] == cli.canonical_sha(
        {k: v for k, v in access.items() if k != "authorization_digest"}
    )
    # the synthesized access authorization satisfies the v3 truth-read contract
    from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import (
        validate_authorization,
    )

    validate_authorization(access, cli.build_smoke_lineage(cli.v15_run_digest()))
    assert "test_only_production_backend" not in access


def test_lead_rulings_are_recorded_for_the_terminal():
    assert cli.SMOKE_RECORD_DECISION["sample_ids"] == (
        "train_uniform_00321",
        "train_layered_00564",
        "train_marmousi_00385",
    )
    assert cli.SMOKE_RECORD_DECISION["same_as_r4e9_probe_records"] is True
    assert cli.SMOKE_RECORD_DECISION["produces_promotable_conclusion"] is False
    assert "veto (d)" in cli.SMOKE_RECORD_DECISION["rationale"]
    assert cli.SMOKE_ORACLE_DECISION["mask_source"] == "parent_frame_energy_only"
    assert cli.SMOKE_ORACLE_DECISION["truth_split_read"] == "train"
    assert cli.SMOKE_ORACLE_DECISION["sealed_splits_read"] == []
    assert cli.SMOKE_ORACLE_DECISION["is_achieved_result"] is False
    assert cli.SMOKE_ORACLE_DECISION["recomputed_on_the_scored_record"] is True
    assert cli.SMOKE_ORACLE_DECISION["transplanted_constants_used"] is False
    assert cli.SMOKE_ORACLE_DECISION["deviation_from_r4e9_probe_retained"] is True
    panels = json.loads(Path(cli.PANELS).read_text())["records"]
    assert [row["sample_id"] for row in panels if row.get("role") == "smoke"] == list(
        cli.SMOKE_RECORD_DECISION["sample_ids"]
    )


def test_smoke_terminal_writer_records_rulings_and_refuses_overwrite(tmp_path):
    writer = cli.build_smoke_terminal_writer(
        run=tmp_path,
        stage_authorization={
            "authorization_path": "/tmp/smoke.json",
            "authorization_sha256": "abc",
            "lead_authorized": True,
            "acknowledges_post_hoc_rule_revision": True,
            "acknowledges_inherited_vetoes": True,
        },
        access_authorization={"authorization_digest": "digest"},
        records={"indices": [0, 1, 2], "resolved_sample_ids": ["a", "b", "c"]},
    )
    writer({"status": "fail_gate", "gates": {"loss": {"passed": False}}})
    payload = json.loads((tmp_path / "terminal.json").read_text())
    # the writer annotates; it never rewrites a status, a gate or a threshold
    assert payload["status"] == "fail_gate"
    assert payload["gates"] == {"loss": {"passed": False}}
    assert payload["smoke_record_decision"]["sample_ids"] == list(
        cli.SMOKE_RECORD_DECISION["sample_ids"]
    )
    assert payload["smoke_record_decision"]["indices"] == [0, 1, 2]
    assert payload["oracle_decision"]["is_achieved_result"] is False
    assert payload["oracle_decision"]["recomputed_on_the_scored_record"] is True
    assert payload["oracle_decision"]["transplanted_constants_used"] is False
    assert payload["parent_write_audit"]["writes"] == 0
    assert payload["parent_write_audit"]["opened_for_write_by_v15"] is False
    assert payload["acceptance_convention_modified"] is False
    assert payload["sealed_future_wavefields_opened"] is False
    assert payload["thresholds"]["absolute_rel_l2_max"] == 0.05
    with pytest.raises(FileExistsError):
        writer({"status": "passed"})


def test_production_oracle_reads_the_scored_record_only():
    torch.manual_seed(5)
    frames = TIME
    k1 = 1
    parent = _parent_with_quiet_frames()
    truth = parent + 0.05 * torch.randn(frames, HEIGHT, WIDTH)
    bases = torch.randn(2, frames, RANK)

    entry = SimpleNamespace(
        public=SimpleNamespace(observed_indices=(0, k1)),
        parent=parent,
        truth=truth[k1 + 1 :],
        route_index=0,
    )
    key = SimpleNamespace(sample_id="train_uniform_00321")

    class _Cache:
        def get(self, requested_key, ledger):
            assert requested_key is key
            return entry

    backend = SimpleNamespace(
        index_keys={7: key},
        cache=_Cache(),
        split="train",
        tau=TAU,
        candidate=SimpleNamespace(bases=bases),
        family_by_sample={"train_uniform_00321": "uniform"},
    )
    oracle = cli.build_smoke_oracle(backend)
    payload = oracle(7, {"sample_id": "train_uniform_00321", "family": "uniform"})
    validate_oracle_provenance("train_uniform_00321", payload)
    assert payload["is_achieved_result"] is False
    assert payload["recomputed_on_the_scored_record"] is True
    assert payload["transplanted_constants_used"] is False
    assert payload["synthetic_records_used"] is False
    assert "deviation_from_r4e9_probe" in payload
    assert payload["decision"]["mask_source"] == "parent_frame_energy_only"

    # an abstaining record gets no bound; no other record's bound stands in
    backend.index_keys[8] = key
    entry.route_index = -1
    with pytest.raises(cli.V15BindingRefusal):
        oracle(8, {"sample_id": "train_uniform_00321", "family": "uniform"})


def test_all_frozen_commands_parse_and_every_mode_has_a_handler():
    config = yaml.safe_load(cli.CONFIG.read_text())
    parsed = {name: cli.build_parser().parse_args(cli.command_argv(command)).mode for name, command in config["commands"].items()}
    assert set(parsed.values()) == set(cli.MODES)
    assert set(cli.HANDLERS) == set(cli.MODES)
    assert all(callable(handler) for handler in cli.HANDLERS.values())
    assert config["candidate"] == CANDIDATE
    assert config["preregistration_sha256"] == cli.FROZEN_BINDINGS["preregistration"][1]
    assert config["scientific_design"]["mask"]["uses_truth_frame_energy"] is False
    assert config["scientific_design"]["mask"]["tau"] == TAU
    assert config["scientific_design"]["correction"]["is_physics_or_pde_residual"] is False
    assert config["scientific_design"]["correction"]["pde_residual_terms"] == 0
    assert config["smoke"]["loss_reduction_computed_per_record"] is True
    assert config["smoke"]["may_promote_to_pilot_or_long"] is False
    assert config["vram"]["unmeasured_or_zero_peak_passes_gate"] is False
    assert config["oracle"]["recomputed_on_the_scored_record"] is True
    assert config["oracle"]["v14_constants_status"] == "void"
    assert config["gates"]["validation_test_rel_l2_max"] == 0.05
    assert config["acceptance"]["modified"] is False
    assert len(config["inherited_vetoes"]) == 6


def test_inherited_thresholds_match_the_frozen_preregistration():
    frozen = json.loads(Path(cli.PREREG).read_text())
    assert frozen["candidate"] == CANDIDATE
    assert frozen["acceptance"]["validation_test_rel_l2_max"] == 0.05
    assert frozen["acceptance"]["modified"] is False
    assert frozen["objective"]["mask"]["tau"] == TAU
    assert frozen["objective"]["mask"]["uses_truth_frame_energy"] is False
    assert cli.THRESHOLDS["absolute_rel_l2_max"] == 0.05
    assert cli.THRESHOLDS["smoke_loss_reduction_min"] == LOSS_REDUCTION_MIN == 0.80
    assert cli.THRESHOLDS["smoke_oracle_gain_fraction"] == ORACLE_GAIN_FRACTION == 0.5
    assert len(cli.INHERITED_VETOES) == 6
