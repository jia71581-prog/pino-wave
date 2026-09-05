from types import SimpleNamespace

import pytest

from scripts.diagnose_r20_source_medium_generalization import (
    build_source_medium_panels,
)


def _record(sample_id, family, group):
    return SimpleNamespace(
        split="train",
        sample_id=sample_id,
        medium_type=family,
        group_id=group,
    )


def test_group_panels_isolate_same_medium_sources_then_new_medium():
    records = (
        _record("l0", "layered", "lg0"),
        _record("l1", "layered", "lg0"),
        _record("l2", "layered", "lg0"),
        _record("l3", "layered", "lg1"),
        _record("l4", "layered", "lg1"),
        _record("m0", "marmousi", "mg0"),
        _record("m1", "marmousi", "mg0"),
        _record("m2", "marmousi", "mg0"),
        _record("m3", "marmousi", "mg1"),
        _record("m4", "marmousi", "mg1"),
    )
    panels = build_source_medium_panels(records, ("l0", "m0"))
    assert panels["anchor"] == (0, 5)
    assert panels["same_medium_new_source"] == (1, 2, 6, 7)
    assert panels["new_medium_new_source"] == (3, 4, 8, 9)
    assert not set(panels["anchor"]) & set(panels["same_medium_new_source"])
    assert not set(panels["new_medium_new_source"]) & set(
        panels["anchor"] + panels["same_medium_new_source"]
    )


def test_group_panels_require_both_supported_families():
    records = (
        _record("l0", "layered", "lg0"),
        _record("l1", "layered", "lg0"),
        _record("l2", "layered", "lg1"),
    )
    with pytest.raises(ValueError, match="one layered and one marmousi"):
        build_source_medium_panels(records, ("l0",))
