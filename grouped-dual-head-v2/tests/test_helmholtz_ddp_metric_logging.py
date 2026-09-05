from __future__ import annotations

import hashlib
import json

import pytest
import torch

from scripts import diagnose_helmholtz_g3_heldout as heldout


def test_component_logging_preserves_single_process_values() -> None:
    result = heldout._distributed_average_components(
        {"total": 2.0, "delta": 3.0},
        {"enabled": False, "world_size": 1},
        torch.device("cpu"),
    )

    assert result == {"delta": 3.0, "total": 2.0}


def test_component_logging_averages_all_ddp_ranks(monkeypatch) -> None:
    def fake_all_reduce(values: torch.Tensor, *, op) -> None:
        assert op == heldout.dist.ReduceOp.SUM
        values.copy_(torch.tensor([8.0, 12.0], dtype=values.dtype))

    monkeypatch.setattr(heldout.dist, "all_reduce", fake_all_reduce)
    result = heldout._distributed_average_components(
        {"total": 99.0, "delta": 98.0},
        {"enabled": True, "world_size": 4},
        torch.device("cpu"),
    )

    assert result == {"delta": 2.0, "total": 3.0}


def test_replacement_gate_binds_source_and_manifest(tmp_path) -> None:
    source = tmp_path / "mixed.h5"
    source.write_bytes(b"audited mixed source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "sample_count": 4003,
                "replacement_sample_count": 700,
                "unchanged_nontrain_sample_count": 1203,
                "bindings": {
                    "hybrid_dataset_sha256": source_sha,
                    "hybrid_manifest_sha256": "manifest-digest",
                },
                "protocol": {
                    "replacement_scope": "train/Marmousi only",
                    "coordinate_arrays_identical": True,
                    "internal_dt_mixed": True,
                },
            }
        )
    )

    audit = heldout.audit_source_replacement_gate(
        gate,
        source_h5=source,
        manifest_digest="internal-manifest-digest",
        source_manifest_sha256="manifest-digest",
        sample_count=4003,
    )

    assert audit["replacement_sample_count"] == 700
    assert audit["source_sha256"] == source_sha
    assert audit["replacement_scope"] == "train/Marmousi only"


def test_replacement_gate_rejects_wrong_source_hash(tmp_path) -> None:
    source = tmp_path / "mixed.h5"
    source.write_bytes(b"different source")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "sample_count": 4003,
                "bindings": {
                    "hybrid_dataset_sha256": "0" * 64,
                    "hybrid_manifest_sha256": "manifest-digest",
                },
                "protocol": {
                    "replacement_scope": "train/Marmousi only",
                    "coordinate_arrays_identical": True,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="dataset SHA256"):
        heldout.audit_source_replacement_gate(
            gate,
            source_h5=source,
            manifest_digest="internal-manifest-digest",
            source_manifest_sha256="manifest-digest",
            sample_count=4003,
        )
