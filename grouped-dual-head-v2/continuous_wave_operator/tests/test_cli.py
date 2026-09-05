from __future__ import annotations

import subprocess
import sys


def test_training_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "continuous_wave_operator.scripts.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout and "--output" in result.stdout
    assert "--data-workers" in result.stdout
    assert "--prefetch-batches" in result.stdout
    assert "--time-frames" in result.stdout and "--points-per-frame" in result.stdout
    assert "--validate-every" in result.stdout and "--checkpoint-every" in result.stdout
