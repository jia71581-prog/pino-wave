from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_readout_lbfgs_cli_registers_bound_checkpoint_and_panel_controls() -> None:
    script = Path("scripts/refine_saved_time_band_adapter_readout_lbfgs.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--checkpoint-identity",
        "--validation-seed",
        "--train-macros",
        "--microbatch-records",
        "--steps",
    ):
        assert option in result.stdout
