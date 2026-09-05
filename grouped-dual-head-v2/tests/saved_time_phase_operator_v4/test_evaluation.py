import pytest

from saved_time_phase_operator_v4.evaluation import validate_evaluation_identity


@pytest.fixture
def bound_identity():
    return {
        "checkpoint_sha256": "a" * 64,
        "manifest_digest": "b" * 64,
        "time_axis_sha256": "c" * 64,
        "record_census": {"validation": 480},
        "model_config_digest": "d" * 64,
    }


def test_evaluation_accepts_exact_binding(bound_identity):
    validate_evaluation_identity(dict(bound_identity), bound_identity)


def test_evaluation_rejects_a_different_time_axis(bound_identity):
    altered = dict(bound_identity)
    altered["time_axis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="time axis"):
        validate_evaluation_identity(altered, bound_identity)


def test_evaluation_rejects_a_different_census(bound_identity):
    altered = dict(bound_identity)
    altered["record_census"] = {"validation": 479}
    with pytest.raises(ValueError, match="record census"):
        validate_evaluation_identity(altered, bound_identity)
