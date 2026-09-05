from __future__ import annotations

from types import SimpleNamespace

from grouped_ufno_mionet_v3.data.curriculum import build_curriculum_schedule


def _manifest():
    records = []
    for index in range(24):
        records.append(
            SimpleNamespace(
                split="train",
                medium_type="uniform",
                group_id=f"uniform-{index}",
                sample_id=f"u-{index}",
            )
        )
    for group in range(6):
        for source in range(4):
            records.append(
                SimpleNamespace(
                    split="train",
                    medium_type="layered",
                    group_id=f"layered-{group}",
                    sample_id=f"l-{group}-{source}",
                )
            )
    for group in range(6):
        for source in range(5):
            records.append(
                SimpleNamespace(
                    split="train",
                    medium_type="marmousi",
                    group_id=f"marmousi-{group}",
                    sample_id=f"m-{group}-{source}",
                )
            )
    return SimpleNamespace(records=tuple(records))


def test_uniform_stage_uses_three_unique_four_record_microbatches():
    schedule = build_curriculum_schedule(
        _manifest(), split="train", stage="uniform", optimizer_steps=2, seed=29
    )
    for step in schedule:
        assert [len(micro.record_indices) for micro in step.microbatches] == [4, 4, 4]
        assert all(micro.family == "uniform" and not micro.replay for micro in step.microbatches)
        indices = [index for micro in step.microbatches for index in micro.record_indices]
        assert len(indices) == len(set(indices)) == 12


def test_layered_stage_uses_three_complete_groups_and_periodic_uniform_replay():
    schedule = build_curriculum_schedule(
        _manifest(), split="train", stage="layered", optimizer_steps=4, seed=29
    )
    for step in schedule[:3]:
        assert len(step.microbatches) == 1
        assert step.microbatches[0].family == "layered"
        assert len(step.microbatches[0].record_indices) == 12
        assert not step.microbatches[0].replay
    replay = schedule[3]
    assert [micro.family for micro in replay.microbatches] == ["uniform"] * 3
    assert all(micro.replay for micro in replay.microbatches)


def test_marmousi_stage_uses_two_slices_and_replays_both_learned_families():
    schedule = build_curriculum_schedule(
        _manifest(), split="train", stage="marmousi", optimizer_steps=5, seed=29
    )
    for step in schedule[:3]:
        assert len(step.microbatches) == 1
        micro = step.microbatches[0]
        assert micro.family == "marmousi"
        assert len(micro.record_indices) == 10
        assert micro.loss_scale == 1.2
    assert [micro.family for micro in schedule[3].microbatches] == ["uniform"] * 3
    assert all(micro.replay for micro in schedule[3].microbatches)
    assert [micro.family for micro in schedule[4].microbatches] == ["layered"]
    assert schedule[4].microbatches[0].replay


def test_curriculum_schedule_is_deterministic_and_rejects_unknown_stage():
    first = build_curriculum_schedule(
        _manifest(), split="train", stage="marmousi", optimizer_steps=7, seed=29
    )
    second = build_curriculum_schedule(
        _manifest(), split="train", stage="marmousi", optimizer_steps=7, seed=29
    )
    assert first == second
    try:
        build_curriculum_schedule(
            _manifest(), split="train", stage="anomaly", optimizer_steps=1, seed=29
        )
    except ValueError as error:
        assert "stage" in str(error)
    else:
        raise AssertionError("unknown curriculum stage was accepted")
