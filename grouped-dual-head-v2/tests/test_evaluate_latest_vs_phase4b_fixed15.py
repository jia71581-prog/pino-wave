from scripts.evaluate_latest_vs_phase4b_fixed15 import replacement_decision


def _metrics(full, receiver):
    return {
        family: {
            "full_wavefield_relative_l2": full[index],
            "receiver_mean_relative_l2": receiver[index],
        }
        for index, family in enumerate(("uniform", "layered", "marmousi"))
    }


def test_replacement_gate_requires_every_family_to_improve():
    historical = _metrics((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
    latest = _metrics((0.09, 0.19, 0.29), (0.1, 0.19, 0.29))

    result = replacement_decision(latest, historical)

    assert result["replacement_gate_passed"] is True


def test_one_family_regression_blocks_global_replacement():
    historical = _metrics((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
    latest = _metrics((0.09, 0.19, 0.31), (0.09, 0.19, 0.29))

    result = replacement_decision(latest, historical)

    assert result["replacement_gate_passed"] is False
    assert result["families"]["marmousi"]["full_wavefield_improved"] is False


def test_receiver_regression_blocks_replacement_even_if_full_field_improves():
    historical = _metrics((0.1, 0.2, 0.3), (0.1, 0.2, 0.3))
    latest = _metrics((0.09, 0.19, 0.29), (0.09, 0.21, 0.29))

    result = replacement_decision(latest, historical)

    assert result["replacement_gate_passed"] is False
    assert result["families"]["layered"]["receiver_improved_or_equal"] is False
