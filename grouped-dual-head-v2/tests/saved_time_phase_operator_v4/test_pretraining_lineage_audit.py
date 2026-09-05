import json

from scripts.audit_saved_time_pretraining_lineage import (
    audit_lineage,
    config_leakage_findings,
)


def test_config_audit_defaults_enabled_epoch_control_to_validation():
    findings = config_leakage_findings(
        {"epoch_validation_control": {"enabled": True}}
    )

    assert findings == [
        {
            "code": "epoch_control_uses_nontrain_truth",
            "evaluation_split": "validation",
        }
    ]


def test_config_audit_accepts_train_bound_control_and_adaptive_errors():
    findings = config_leakage_findings(
        {
            "epoch_validation_control": {
                "enabled": True,
                "evaluation_split": "train",
            },
            "adaptive_sampling": {
                "enabled": True,
                "evidence_split": "train",
                "time_bin_errors": [0.2, 0.3, 0.4],
                "family_errors": {"uniform": 0.1},
            },
        }
    )

    assert findings == []


def test_config_audit_rejects_mutable_and_unbound_best_parent_paths():
    assert config_leakage_findings(
        {"parent_checkpoint": "/run/latest.pt"}
    ) == [{"code": "mutable_parent_checkpoint_path"}]
    assert config_leakage_findings(
        {"parent_checkpoint": "/run/best.pt"}
    ) == [
        {
            "code": "parent_best_checkpoint_lacks_train_selection_binding",
            "selection_split": None,
        }
    ]
    assert config_leakage_findings(
        {
            "parent_checkpoint": "/run/best.pt",
            "parent_checkpoint_selection_split": "train",
        }
    ) == []


def test_lineage_audit_is_bounded_and_follows_only_declared_parent(tmp_path):
    parent = tmp_path / "parent.json"
    child = tmp_path / "child.json"
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"checkpoint")
    parent.write_text(
        json.dumps(
            {
                "run_digest": "parent-run",
                "manifest_digest": "manifest",
                "config": {},
            }
        )
    )
    child.write_text(
        json.dumps(
            {
                "run_digest": "child-run",
                "manifest_digest": "manifest",
                "config": {
                    "parent_checkpoint": "parent.pt",
                    "parent_checkpoint_identity": "parent.json",
                    "adaptive_sampling": {
                        "enabled": True,
                        "family_errors": {"marmousi": 0.5},
                    },
                },
            }
        )
    )

    report = audit_lineage(child, max_depth=3)

    assert report["node_count"] == 2
    assert report["terminal"] == "no_parent_identity"
    assert report["finding_count"] == 1
    assert not report["promotion_safe_from_config_evidence"]
    assert report["nodes"][0]["parent_checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "exists": True,
        "size_bytes": len(b"checkpoint"),
    }


def test_lineage_audit_marks_a_missing_parent_checkpoint(tmp_path):
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "config": {"parent_checkpoint": "missing_epoch.pt"},
                "run_digest": "run",
            }
        )
    )

    report = audit_lineage(identity)

    assert report["nodes"][0]["findings"] == [
        {"code": "missing_parent_checkpoint"}
    ]
