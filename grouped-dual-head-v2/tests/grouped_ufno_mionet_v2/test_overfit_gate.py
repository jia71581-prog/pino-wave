from scripts.overfit_grouped_v2 import evaluate_overfit_gate


def test_gate_requires_both_heads_to_beat_zero_and_have_gradients():
    result = evaluate_overfit_gate(query_rel_l2=.08, dense_rel_l2=.07,
                                   zero_query_rel_l2=1., zero_dense_rel_l2=1.,
                                   missing_gradients=[])
    assert result.passed
    assert not evaluate_overfit_gate(.08, 1.2, 1., 1., []).passed
    assert not evaluate_overfit_gate(.08, .07, 1., 1., ["source_encoder.map_proj.weight"]).passed


def test_gate_rejects_nonfinite_metrics():
    assert not evaluate_overfit_gate(float("nan"), .05, 1., 1., []).passed
