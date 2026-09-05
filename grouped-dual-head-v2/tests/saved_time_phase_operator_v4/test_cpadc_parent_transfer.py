from pathlib import Path

import torch

from scripts.rebind_cpadc_trainonly_transfer import rebind_trainonly_transfer, sha256_file


def test_rebind_cpadc_preserves_basis_and_removes_old_calibration(tmp_path: Path):
    source = tmp_path / "source.pt"
    parent = tmp_path / "parent.pt"
    output = tmp_path / "transfer.pt"
    basis = {"weight": torch.randn(3, 4)}
    torch.save(
        {
            "schema": "causal_physics_aligned_defect_correction_v1",
            "schema_version": 5,
            "basis_rank": 16,
            "phase_rank": 4,
            "basis_state": basis,
            "parent_checkpoint": "/old/parent.pt",
            "parent_checkpoint_sha256": "a" * 64,
            "online_solve_contract": {
                "name": "ridge_direction_family_calibrated_strength_abstention_v1",
                "minimum_unconstrained_correction_ratio_by_family": {
                    "uniform": 1.0,
                    "layered": 2.0,
                    "marmousi": 3.0,
                },
            },
            "risk_calibration": {"future_truth_scope": "disjoint_train_split_only"},
        },
        source,
    )
    torch.save({"model_state": {}}, parent)

    report = rebind_trainonly_transfer(source, output, parent_checkpoint=parent)
    transferred = torch.load(output, map_location="cpu", weights_only=False)

    torch.testing.assert_close(
        transferred["basis_state"]["weight"], basis["weight"]
    )
    assert transferred["parent_checkpoint"] == str(parent.resolve())
    assert transferred["parent_checkpoint_sha256"] == sha256_file(parent)
    assert (
        transferred["online_solve_contract"]["name"]
        == "ridge_direction_learned_energy_ball_projection_v1"
    )
    assert "minimum_unconstrained_correction_ratio_by_family" not in transferred[
        "online_solve_contract"
    ]
    assert transferred["risk_calibration"] == {}
    assert report["status"] == "uncalibrated_trainonly_transfer"

