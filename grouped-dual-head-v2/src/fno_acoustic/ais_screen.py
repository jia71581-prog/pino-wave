"""Pure deterministic decisions for the registered AIS-MQFNO screen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType


CANDIDATE_IDS = tuple(f"N{index}" for index in range(8))
CATEGORIES = ("uniform", "layered", "marmousi")
GATE_O_METRICS = frozenset(
    {
        "relative_l2",
        "relative_l2_q4",
        "prediction_target_norm_ratio",
        "prediction_target_pearson",
    }
)
HALVING_METRICS = frozenset(
    {
        "relative_l2",
        "relative_l2_q4",
        "prediction_target_norm_ratio",
        "zero_relative_l2",
        "zero_relative_l2_q4",
    }
)
LOWER_GUARDS = (
    "receiver_relative_l2",
    "receiver_relative_l2_q1",
    "receiver_relative_l2_q2",
    "receiver_relative_l2_q3",
    "receiver_relative_l2_q4",
    "arrival_mae_s",
    "arrival_miss_rate",
    "receiver_lag_abs_s",
    "receiver_phase_error",
    "energy_log_ratio",
    "komega_relative_l2",
    "komega_high",
    "komega_relative_l2_q4",
    "komega_high_q4",
)
HIGHER_GUARDS = ("receiver_xcorr_peak", "receiver_phase_coherence")
FINAL_CANDIDATE_METRICS = frozenset(
    {
        "relative_l2",
        "relative_l2_q4",
        "prediction_target_norm_ratio",
        "prediction_target_pearson",
        *LOWER_GUARDS,
        *HIGHER_GUARDS,
    }
)
FINAL_BASELINE_METRICS = frozenset(
    {"relative_l2", "relative_l2_q4", *LOWER_GUARDS, *HIGHER_GUARDS}
)


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
    return value.lower()


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _freeze_category_metrics(
    value: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(CATEGORIES):
        raise ValueError(f"category_metrics categories must equal {CATEGORIES}")
    frozen: dict[str, Mapping[str, float]] = {}
    for category in CATEGORIES:
        metrics = value[category]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"category {category} metrics must be a mapping")
        copied: dict[str, float] = {}
        for name, metric in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be nonempty strings")
            copied[name] = _finite_real(metric, f"{category}.{name}")
        frozen[category] = MappingProxyType(copied)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class CandidateGateRecord:
    candidate_id: str
    update: int
    checkpoint_sha256: str
    category_metrics: Mapping[str, Mapping[str, float]]
    finite: bool
    parent_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_id not in CANDIDATE_IDS:
            raise ValueError(f"candidate_id must be one of {CANDIDATE_IDS}")
        if isinstance(self.update, bool) or not isinstance(self.update, int) or self.update < 1:
            raise ValueError("update must be a positive integer")
        if not isinstance(self.finite, bool):
            raise ValueError("finite must be a boolean")
        object.__setattr__(self, "checkpoint_sha256", _sha256(self.checkpoint_sha256))
        if self.parent_checkpoint_sha256 is not None:
            object.__setattr__(
                self,
                "parent_checkpoint_sha256",
                _sha256(self.parent_checkpoint_sha256),
            )
        object.__setattr__(
            self, "category_metrics", _freeze_category_metrics(self.category_metrics)
        )


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    passed: bool
    reasons: tuple[str, ...]
    score: float | None = None


_CANONICAL_REASONS = {
    ("O", "advance"): frozenset({"representable_candidates_advanced"}),
    ("O", "stop"): frozenset(
        {"n0_not_representable", "fewer_than_four_representable_candidates"}
    ),
    ("H1", "advance"): frozenset({"4_candidates_advanced"}),
    ("H1", "stop"): frozenset({"fewer_than_four_eligible_candidates"}),
    ("H2", "advance"): frozenset({"2_candidates_advanced"}),
    ("H2", "stop"): frozenset({"fewer_than_two_eligible_candidates"}),
    ("H3", "promote"): frozenset({"recipe_promoted"}),
    ("H3", "stop"): frozenset({"no_candidate_satisfies_final_accuracy_gate"}),
}


def _build_screen_decision_type():

    @dataclass(frozen=True, init=False)
    class ScreenDecision:
        gate: str
        status: str
        reason: str
        advanced: tuple[str, ...]
        rejected: Mapping[str, tuple[str, ...]]
        checkpoint_sha256s: Mapping[str, str]
        promoted_recipe: str | None

        def __init__(
            self,
            gate: object = None,
            status: object = None,
            reason: object = None,
            advanced: object = None,
            rejected: object = None,
            checkpoint_sha256s: object = None,
            promoted_recipe: object = None,
        ) -> None:
            if not isinstance(gate, str) or not isinstance(status, str):
                raise ValueError("decision gate and status must be strings")
            allowed_reasons = _CANONICAL_REASONS.get((gate, status))
            if not isinstance(reason, str) or allowed_reasons is None or reason not in allowed_reasons:
                raise ValueError("decision requires a canonical reason for its gate and status")
            if (
                isinstance(advanced, (str, bytes))
                or not isinstance(advanced, Sequence)
                or any(not isinstance(candidate_id, str) for candidate_id in advanced)
            ):
                raise ValueError("decision advanced candidates must be a sequence of strings")
            advanced_tuple = tuple(advanced)
            if (
                len(set(advanced_tuple)) != len(advanced_tuple)
                or any(candidate_id not in CANDIDATE_IDS for candidate_id in advanced_tuple)
            ):
                raise ValueError("decision advanced candidates are invalid")
            if not isinstance(checkpoint_sha256s, Mapping) or not checkpoint_sha256s:
                raise ValueError("decision checkpoint mapping must be nonempty")
            canonical_hashes: dict[str, str] = {}
            for candidate_id, checkpoint_sha256 in checkpoint_sha256s.items():
                if not isinstance(candidate_id, str) or candidate_id not in CANDIDATE_IDS:
                    raise ValueError("decision checkpoint mapping has an invalid candidate")
                canonical_hashes[candidate_id] = _sha256(checkpoint_sha256)
            if not isinstance(rejected, Mapping):
                raise ValueError("decision rejected reasons must be a mapping")
            frozen_rejected: dict[str, tuple[str, ...]] = {}
            for candidate_id, reasons in rejected.items():
                if not isinstance(candidate_id, str) or candidate_id not in CANDIDATE_IDS:
                    raise ValueError("decision rejected mapping has an invalid candidate")
                if (
                    isinstance(reasons, (str, bytes))
                    or not isinstance(reasons, Sequence)
                    or not reasons
                    or any(not isinstance(item, str) or not item for item in reasons)
                ):
                    raise ValueError("decision rejected reasons must be nonempty strings")
                frozen_rejected[candidate_id] = tuple(reasons)
            checkpoint_ids = set(canonical_hashes)
            advanced_ids = set(advanced_tuple)
            if not advanced_ids.issubset(checkpoint_ids):
                raise ValueError("decision checkpoint mapping is incomplete")
            if set(frozen_rejected) != checkpoint_ids - advanced_ids:
                raise ValueError("decision rejected partition is inconsistent")
            if gate == "O":
                if checkpoint_ids != set(CANDIDATE_IDS):
                    raise ValueError("Gate O decision requires exactly N0-N7 checkpoints")
                if status == "advance" and (not 4 <= len(advanced_tuple) <= 8 or "N0" not in advanced_tuple):
                    raise ValueError("Gate O advance decision has invalid candidates")
                if status == "stop" and advanced_tuple:
                    raise ValueError("Gate O stop decision cannot advance candidates")
            elif gate == "H1":
                if not 4 <= len(checkpoint_ids) <= 8:
                    raise ValueError("H1 decision requires four to eight checkpoints")
                if (status == "advance" and len(advanced_tuple) != 4) or (
                    status == "stop" and advanced_tuple
                ):
                    raise ValueError("H1 decision has an invalid advanced set")
            elif gate == "H2":
                if len(checkpoint_ids) != 4:
                    raise ValueError("H2 decision requires exactly four checkpoints")
                if (status == "advance" and len(advanced_tuple) != 2) or (
                    status == "stop" and advanced_tuple
                ):
                    raise ValueError("H2 decision has an invalid advanced set")
            elif gate == "H3":
                if len(checkpoint_ids) != 2:
                    raise ValueError("H3 decision requires exactly two checkpoints")
                if status == "promote":
                    if len(advanced_tuple) != 1 or promoted_recipe != advanced_tuple[0]:
                        raise ValueError("H3 promotion must identify its sole advanced recipe")
                elif advanced_tuple or promoted_recipe is not None:
                    raise ValueError("H3 stop decision cannot promote a recipe")
            if gate != "H3" and promoted_recipe is not None:
                raise ValueError("only H3 may contain a promoted recipe")
            object.__setattr__(self, "gate", gate)
            object.__setattr__(self, "status", status)
            object.__setattr__(self, "reason", reason)
            object.__setattr__(self, "advanced", advanced_tuple)
            object.__setattr__(self, "rejected", MappingProxyType(frozen_rejected))
            object.__setattr__(
                self, "checkpoint_sha256s", MappingProxyType(canonical_hashes)
            )
            object.__setattr__(self, "promoted_recipe", promoted_recipe)

        def to_dict(self) -> dict[str, object]:
            return {
                "gate": self.gate,
                "status": self.status,
                "reason": self.reason,
                "advanced": list(self.advanced),
                "rejected": {
                    candidate_id: list(reasons)
                    for candidate_id, reasons in self.rejected.items()
                },
                "checkpoint_sha256s": dict(self.checkpoint_sha256s),
                "promoted_recipe": self.promoted_recipe,
            }

    return ScreenDecision


ScreenDecision = _build_screen_decision_type()
del _build_screen_decision_type


def _validate_metric_names(record: CandidateGateRecord, expected: frozenset[str]) -> None:
    for category in CATEGORIES:
        if set(record.category_metrics[category]) != set(expected):
            raise ValueError(
                f"{record.candidate_id} {category} metrics must equal {sorted(expected)}"
            )


def _records(
    records: Sequence[CandidateGateRecord], expected_update: int
) -> tuple[CandidateGateRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("gate records must be a sequence of CandidateGateRecord values")
    if not records:
        raise ValueError("gate records must be a nonempty sequence")
    normalized = tuple(records)
    if any(not isinstance(record, CandidateGateRecord) for record in normalized):
        raise TypeError("gate records must contain CandidateGateRecord values")
    candidate_ids = [record.candidate_id for record in normalized]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("gate records require unique candidate IDs")
    for record in normalized:
        if record.update != expected_update:
            raise ValueError(f"gate requires exact update {expected_update} last checkpoint")
    return normalized


def _require_no_parents(
    records: Sequence[CandidateGateRecord], gate: str
) -> None:
    if any(record.parent_checkpoint_sha256 is not None for record in records):
        if gate == "H1":
            raise ValueError("H1 restart records must not bind a parent checkpoint")
        raise ValueError(f"{gate} records must not bind a parent checkpoint")


def _validate_parent_lineage(
    records: Sequence[CandidateGateRecord], prior: ScreenDecision
) -> None:
    for record in records:
        if record.parent_checkpoint_sha256 != prior.checkpoint_sha256s[record.candidate_id]:
            raise ValueError(
                f"{record.candidate_id} parent checkpoint SHA does not match the prior last checkpoint"
            )


def gate_o_candidate(record: CandidateGateRecord) -> CandidateAssessment:
    if not isinstance(record, CandidateGateRecord):
        raise TypeError("record must be a CandidateGateRecord")
    if record.update != 400:
        raise ValueError("Gate O requires exact update 400 last checkpoint")
    _require_no_parents((record,), "Gate O")
    _validate_metric_names(record, GATE_O_METRICS)
    reasons: list[str] = []
    if not record.finite:
        reasons.append("nonfinite_state")
    for category in CATEGORIES:
        metrics = record.category_metrics[category]
        if metrics["relative_l2"] < 0 or metrics["relative_l2_q4"] < 0:
            raise ValueError(f"{category} Gate O relative L2 metrics must be nonnegative")
        if metrics["relative_l2"] > 0.35:
            reasons.append(f"{category}:relative_l2_above_0.35")
        if metrics["relative_l2_q4"] > 0.50:
            reasons.append(f"{category}:relative_l2_q4_above_0.50")
        norm_ratio = metrics["prediction_target_norm_ratio"]
        if norm_ratio < 0:
            raise ValueError(f"{category} Gate O norm ratio must be nonnegative")
        if not 0.50 <= norm_ratio <= 1.50:
            reasons.append(f"{category}:norm_ratio_outside_[0.5,1.5]")
        pearson = metrics["prediction_target_pearson"]
        if not -1.0 <= pearson <= 1.0:
            raise ValueError(f"{category} Gate O Pearson must lie in [-1,1]")
        if pearson < 0.80:
            reasons.append(f"{category}:pearson_below_0.80")
    return CandidateAssessment(record.candidate_id, not reasons, tuple(reasons))


def gate_o(records: Sequence[CandidateGateRecord]) -> ScreenDecision:
    normalized = _records(records, 400)
    if {record.candidate_id for record in normalized} != set(CANDIDATE_IDS):
        raise ValueError("Gate O requires exactly N0-N7 candidate records")
    _require_no_parents(normalized, "Gate O")
    assessments = {
        record.candidate_id: gate_o_candidate(record) for record in normalized
    }
    passed = tuple(sorted(candidate_id for candidate_id, item in assessments.items() if item.passed))
    if "N0" not in passed:
        status, reason, advanced = "stop", "n0_not_representable", ()
        stopped_reason = "gate_stopped_n0_not_representable"
    elif len(passed) < 4:
        status, reason, advanced = (
            "stop",
            "fewer_than_four_representable_candidates",
            (),
        )
        stopped_reason = "gate_stopped_insufficient_representable"
    else:
        status, reason, advanced = "advance", "representable_candidates_advanced", passed
        stopped_reason = None
    rejected = {
        candidate_id: (
            item.reasons
            if not item.passed
            else (stopped_reason,)
        )
        for candidate_id, item in sorted(assessments.items())
        if candidate_id not in advanced
    }
    return ScreenDecision(
        gate="O",
        status=status,
        reason=reason,
        advanced=advanced,
        rejected=rejected,
        checkpoint_sha256s={
            record.candidate_id: record.checkpoint_sha256 for record in normalized
        },
    )


def ranking_score(categories: Mapping[str, Mapping[str, float]]) -> float:
    """Return the immutable registered whole/Q4 worst-category score."""

    if not isinstance(categories, Mapping) or set(categories) != set(CATEGORIES):
        raise ValueError(f"ranking categories must equal {CATEGORIES}")
    for category in CATEGORIES:
        metrics = categories[category]
        if not isinstance(metrics, Mapping):
            raise ValueError("ranking category metrics must be mappings")
        for name in ("relative_l2", "relative_l2_q4"):
            value = _finite_real(metrics.get(name), f"{category}.{name}")
            if value < 0:
                raise ValueError(f"{category}.{name} must be nonnegative")
    return max(categories[category]["relative_l2"] for category in CATEGORIES) + 0.5 * max(
        categories[category]["relative_l2_q4"] for category in CATEGORIES
    )


def _halving_candidate(record: CandidateGateRecord) -> CandidateAssessment:
    _validate_metric_names(record, HALVING_METRICS)
    reasons: list[str] = []
    if not record.finite:
        reasons.append("nonfinite_state")
    for category in CATEGORIES:
        metrics = record.category_metrics[category]
        zero = metrics["zero_relative_l2"]
        zero_q4 = metrics["zero_relative_l2_q4"]
        if zero <= 0:
            raise ValueError(f"{category}.zero_relative_l2 must be positive")
        if zero_q4 <= 0:
            raise ValueError(f"{category}.zero_relative_l2_q4 must be positive")
        if metrics["relative_l2"] < 0 or metrics["relative_l2_q4"] < 0:
            raise ValueError(f"{category} relative L2 metrics must be nonnegative")
        norm_ratio = metrics["prediction_target_norm_ratio"]
        if norm_ratio < 0:
            raise ValueError(f"{category} norm ratio must be nonnegative")
        if not 0.25 <= norm_ratio <= 2.0:
            reasons.append(f"{category}:norm_ratio_outside_[0.25,2.0]")
        if metrics["relative_l2"] > 1.05 * zero:
            reasons.append(f"{category}:relative_l2_above_1.05_zero")
        if metrics["relative_l2_q4"] > 1.05 * zero_q4:
            reasons.append(f"{category}:relative_l2_q4_above_1.05_zero_q4")
    return CandidateAssessment(
        record.candidate_id,
        not reasons,
        tuple(reasons),
        ranking_score(record.category_metrics),
    )


def _halving_decision(
    records: Sequence[CandidateGateRecord],
    *,
    gate: str,
    update: int,
    survivor_count: int,
) -> ScreenDecision:
    normalized = _records(records, update)
    assessments = {
        record.candidate_id: _halving_candidate(record) for record in normalized
    }
    eligible = sorted(
        (item for item in assessments.values() if item.passed),
        key=lambda item: (item.score, item.candidate_id),
    )
    if len(eligible) < survivor_count:
        status = "stop"
        reason = f"fewer_than_{'four' if survivor_count == 4 else 'two'}_eligible_candidates"
        advanced: tuple[str, ...] = ()
        rejected = {
            candidate_id: (
                item.reasons
                if not item.passed
                else ("gate_stopped_insufficient_eligible",)
            )
            for candidate_id, item in sorted(assessments.items())
        }
    else:
        status = "advance"
        reason = f"{survivor_count}_candidates_advanced"
        advanced = tuple(item.candidate_id for item in eligible[:survivor_count])
        rejected = {
            candidate_id: (
                item.reasons
                if not item.passed
                else ("not_selected_by_ranking",)
            )
            for candidate_id, item in sorted(assessments.items())
            if candidate_id not in advanced
        }
    return ScreenDecision(
        gate=gate,
        status=status,
        reason=reason,
        advanced=advanced,
        rejected=rejected,
        checkpoint_sha256s={
            record.candidate_id: record.checkpoint_sha256 for record in normalized
        },
    )


def gate_h1(
    records: Sequence[CandidateGateRecord],
    gate_o_records: Sequence[CandidateGateRecord],
) -> ScreenDecision:
    prior = gate_o(gate_o_records)
    if prior.status != "advance":
        raise ValueError("recomputed Gate O decision must advance before H1")
    normalized = _records(records, 600)
    if {record.candidate_id for record in normalized} != set(prior.advanced):
        raise ValueError("H1 records must equal the Gate O advanced candidates")
    _require_no_parents(normalized, "H1")
    return _halving_decision(normalized, gate="H1", update=600, survivor_count=4)


def gate_h2(
    records: Sequence[CandidateGateRecord],
    h1_records: Sequence[CandidateGateRecord],
    gate_o_records: Sequence[CandidateGateRecord],
) -> ScreenDecision:
    prior = gate_h1(h1_records, gate_o_records)
    if prior.status != "advance":
        raise ValueError("recomputed Gate H1 decision must advance before H2")
    normalized = _records(records, 1500)
    if {record.candidate_id for record in normalized} != set(prior.advanced):
        raise ValueError("H2 records must equal the H1 advanced candidates")
    _validate_parent_lineage(normalized, prior)
    return _halving_decision(
        normalized, gate="H2", update=1500, survivor_count=2
    )


def _baseline_metrics(
    value: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(CATEGORIES):
        raise ValueError(f"baseline categories must equal {CATEGORIES}")
    result: dict[str, Mapping[str, float]] = {}
    for category in CATEGORIES:
        metrics = value[category]
        if not isinstance(metrics, Mapping) or set(metrics) != set(FINAL_BASELINE_METRICS):
            raise ValueError(
                f"baseline {category} metrics must equal {sorted(FINAL_BASELINE_METRICS)}"
            )
        copied = {
            name: _finite_real(metric, f"baseline.{category}.{name}")
            for name, metric in metrics.items()
        }
        for name in ("relative_l2", "relative_l2_q4", *LOWER_GUARDS):
            if copied[name] < 0:
                raise ValueError(f"baseline {category}.{name} must be nonnegative")
        if not -1.0 <= copied["receiver_xcorr_peak"] <= 1.0:
            raise ValueError(
                f"baseline {category}.receiver_xcorr_peak must lie in range [-1,1]"
            )
        if not 0.0 <= copied["receiver_phase_coherence"] <= 1.0:
            raise ValueError(
                f"baseline {category}.receiver_phase_coherence must lie in range [0,1]"
            )
        result[category] = MappingProxyType(copied)
    return MappingProxyType(result)


def _final_candidate(
    record: CandidateGateRecord,
    baseline: Mapping[str, Mapping[str, float]],
) -> CandidateAssessment:
    _validate_metric_names(record, FINAL_CANDIDATE_METRICS)
    reasons: list[str] = []
    if not record.finite:
        reasons.append("nonfinite_state")
    for category in CATEGORIES:
        candidate = record.category_metrics[category]
        reference = baseline[category]
        for name in ("relative_l2", "relative_l2_q4", *LOWER_GUARDS):
            if candidate[name] < 0:
                raise ValueError(f"{category}.{name} must be nonnegative")
        if not -1.0 <= candidate["receiver_xcorr_peak"] <= 1.0:
            raise ValueError(
                f"{category}.receiver_xcorr_peak must lie in range [-1,1]"
            )
        if not 0.0 <= candidate["receiver_phase_coherence"] <= 1.0:
            raise ValueError(
                f"{category}.receiver_phase_coherence must lie in range [0,1]"
            )
        if candidate["relative_l2"] > 0.70 * reference["relative_l2"]:
            reasons.append(f"{category}:relative_l2_fails_30_percent_improvement")
        if candidate["relative_l2_q4"] > 0.70 * reference["relative_l2_q4"]:
            reasons.append(f"{category}:relative_l2_q4_fails_30_percent_improvement")
        for name in LOWER_GUARDS:
            tolerance = max(0.05 * reference[name], 1e-6)
            if candidate[name] > reference[name] + tolerance:
                reasons.append(f"{category}:{name}_regresses_over_5_percent")
        for name in HIGHER_GUARDS:
            tolerance = max(0.05 * abs(reference[name]), 1e-6)
            if candidate[name] < reference[name] - tolerance:
                reasons.append(f"{category}:{name}_regresses_over_5_percent")
        norm_ratio = candidate["prediction_target_norm_ratio"]
        if norm_ratio < 0:
            raise ValueError(f"{category} norm ratio must be nonnegative")
        if not 0.50 <= norm_ratio <= 1.50:
            reasons.append(f"{category}:norm_ratio_outside_[0.5,1.5]")
        pearson = candidate["prediction_target_pearson"]
        if not -1.0 <= pearson <= 1.0:
            raise ValueError(f"{category} prediction-target Pearson must lie in [-1,1]")
        if pearson < 0.80:
            reasons.append(f"{category}:pearson_below_0.80")
    return CandidateAssessment(
        record.candidate_id,
        not reasons,
        tuple(reasons),
        ranking_score(record.category_metrics),
    )


def gate_h3(
    records: Sequence[CandidateGateRecord],
    h2_records: Sequence[CandidateGateRecord],
    h1_records: Sequence[CandidateGateRecord],
    gate_o_records: Sequence[CandidateGateRecord],
    baseline_metrics: Mapping[str, Mapping[str, float]],
    *,
    ranking_records: Sequence[CandidateGateRecord] | None = None,
) -> ScreenDecision:
    prior = gate_h2(h2_records, h1_records, gate_o_records)
    if prior.status != "advance":
        raise ValueError("recomputed Gate H2 decision must advance before H3")
    normalized = _records(records, 3000)
    if {record.candidate_id for record in normalized} != set(prior.advanced):
        raise ValueError("H3 records must equal the H2 advanced candidates")
    _validate_parent_lineage(normalized, prior)
    baseline = _baseline_metrics(baseline_metrics)
    native_assessments = {
        record.candidate_id: _final_candidate(record, baseline)
        for record in normalized
    }
    if ranking_records is None:
        assessments = native_assessments
    else:
        ranking = _records(ranking_records, 3000)
        if {record.candidate_id for record in ranking} != set(prior.advanced):
            raise ValueError("H3 screen ranking records must equal the H2 finalists")
        _validate_parent_lineage(ranking, prior)
        screen_assessments = {
            record.candidate_id: _halving_candidate(record) for record in ranking
        }
        assessments = {}
        for record in ranking:
            native = native_assessments[record.candidate_id]
            screen = screen_assessments[record.candidate_id]
            reasons = native.reasons + tuple(
                f"screen64:{reason}" for reason in screen.reasons
            )
            assessments[record.candidate_id] = CandidateAssessment(
                record.candidate_id,
                native.passed and screen.passed,
                reasons,
                screen.score,
            )
    eligible = sorted(
        (item for item in assessments.values() if item.passed),
        key=lambda item: (item.score, item.candidate_id),
    )
    if eligible:
        winner = eligible[0].candidate_id
        status, reason, advanced, promoted = "promote", "recipe_promoted", (winner,), winner
        rejected = {
            candidate_id: (
                item.reasons
                if not item.passed
                else ("not_selected_by_ranking",)
            )
            for candidate_id, item in sorted(assessments.items())
            if candidate_id != winner
        }
    else:
        status = "stop"
        reason = "no_candidate_satisfies_final_accuracy_gate"
        advanced = ()
        promoted = None
        rejected = {
            candidate_id: item.reasons
            for candidate_id, item in sorted(assessments.items())
        }
    return ScreenDecision(
        gate="H3",
        status=status,
        reason=reason,
        advanced=advanced,
        rejected=rejected,
        checkpoint_sha256s={
            record.candidate_id: record.checkpoint_sha256 for record in normalized
        },
        promoted_recipe=promoted,
    )


__all__ = [
    "CANDIDATE_IDS",
    "CATEGORIES",
    "HIGHER_GUARDS",
    "LOWER_GUARDS",
    "CandidateAssessment",
    "CandidateGateRecord",
    "ScreenDecision",
    "gate_h1",
    "gate_h2",
    "gate_h3",
    "gate_o",
    "gate_o_candidate",
    "ranking_score",
]
