#!/usr/bin/env python3
"""v23b wrapper: stable, spatially isolated coarse LWC-84 validation audit."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_coarse_lwc84_dispersion_validation_streaming as audit


WRAPPER = Path(__file__).resolve()
IMPLEMENTATION = Path(audit.__file__).resolve()
PREFLIGHT = ROOT / "scripts/preflight_v23b_stability.py"
ORIGINAL_LOAD = audit._load_and_verify_preregistration

# The implementation is reused without altering the failed, frozen v23 script.
# Rebinding __file__ makes the existing integrity check bind this v23b wrapper.
audit.__file__ = str(WRAPPER)
audit.PREREG = (
    ROOT
    / "results/r16_dscp_v23b_coarse_lwc84_dispersion_preregistration_20260827.json"
)
audit.OUT = ROOT / "results/r16_dscp_v23b_coarse_lwc84_dispersion_validation"
audit.INTERNAL_DT_S = 1.25e-4


def _load_and_verify_v23b() -> dict[str, Any]:
    payload = ORIGINAL_LOAD()
    bindings = payload["sha256_bindings"]
    extra_paths = {
        "implementation_script": IMPLEMENTATION,
        "stability_preflight": PREFLIGHT,
    }
    for name, path in extra_paths.items():
        observed = audit._sha256_file(path)
        if observed != str(bindings.get(name, "")):
            raise RuntimeError(f"sha256 binding drift: {name}")
        payload["observed_sha256_bindings"][name] = observed
    return payload


audit._load_and_verify_preregistration = _load_and_verify_v23b


if __name__ == "__main__":
    worker = os.environ.get("WORKER")
    audit.main()
