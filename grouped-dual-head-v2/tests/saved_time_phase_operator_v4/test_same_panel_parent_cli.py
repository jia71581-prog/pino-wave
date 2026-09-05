from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_same_panel_evaluator_accepts_explicit_bound_checkpoint_and_seed() -> None:
    script = Path("scripts/evaluate_saved_time_family_expert_parent.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--checkpoint-identity" in result.stdout
    assert "--validation-seed" in result.stdout
