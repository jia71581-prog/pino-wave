import json

import pytest

from scripts.train_grouped_v2 import validate_launch_gate


def test_production_refuses_missing_or_failed_overfit_gate(tmp_path):
    with pytest.raises(RuntimeError, match="overfit gate"):
        validate_launch_gate(tmp_path / "missing.json")
    failed = tmp_path / "failed.json"; failed.write_text('{"passed": false}')
    with pytest.raises(RuntimeError, match="overfit gate"):
        validate_launch_gate(failed)


def test_production_requires_pilot_authorization(tmp_path):
    overfit = tmp_path / "overfit.json"; overfit.write_text('{"passed": true}')
    with pytest.raises(RuntimeError, match="pilot"):
        validate_launch_gate(overfit, require_production_authorized=True)
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"passed": True, "production_authorized": True}))
    assert validate_launch_gate(pilot, require_production_authorized=True)["production_authorized"]
