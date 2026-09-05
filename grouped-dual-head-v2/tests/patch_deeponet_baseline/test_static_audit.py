import json
from pathlib import Path

from scripts.audit_patch_deeponet_baseline import audit


PROTOCOL = Path(
    "paper/tgrs_helmholtz_operator/patch_deeponet_confirmatory_protocol_20260813.json"
)


def test_patch_deeponet_protocol_is_position_only_and_static_audit_passes() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf8"))
    position = protocol["position_generalization"]
    assert position["fixed_source_frequency_hz"] == 19.0
    assert position["source_frequency_generalization_in_scope"] is False
    assert position["frequency_sweep_permitted"] is False
    report = audit(PROTOCOL)
    assert report["status"] == "pass"
    assert report["parameter_match"]["within_tolerance"] is True
    assert report["forbidden_input_tokens_found"] == []
