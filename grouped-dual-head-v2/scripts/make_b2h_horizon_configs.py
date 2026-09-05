#!/usr/bin/env python3
"""Generate the B2-H training-horizon diagnostic sweep configs.

Only ``data.rollout_steps`` varies across the four configs; every other field
(model, optimizer, warmstart, seed, records, epochs) is held identical so the
sole independent variable is the training rollout horizon.  Each trained
checkpoint is later judged by the *fixed* 401-step strict free rollout, so the
sweep answers: does a longer training horizon convert the tiny learned residual
from rollout-destabilizing into neutral/stabilizing?
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "configs" / "b2h_paired_control_r1.yaml"
OUT_DIR = REPO / "configs" / "b2h_horizon_sweep"
RUN_ROOT = "/root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/1/pretraining/b2h/horizon_sweep"

HORIZONS = (8, 16, 32, 64)
EPOCHS = 20
SAMPLES_PER_EPOCH = 48

# validation fixed_starts must satisfy: history_steps <= start <= T - rollout - 2.
# With T=401 and rollout<=64 the deepest valid start is 401-64-2 = 335.
VALIDATION_STARTS = [40, 100, 200, 300]


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for horizon in HORIZONS:
        config = copy.deepcopy(base)
        config["epochs"] = EPOCHS
        config["run_dir"] = f"{RUN_ROOT}/h{horizon:02d}/run"
        config["data"]["rollout_steps"] = int(horizon)
        config["data"]["samples_per_epoch"] = SAMPLES_PER_EPOCH
        config["data"]["validation_starts"] = list(VALIDATION_STARTS)
        # Identical warmstart / seed / model across the sweep (inherited from base).
        path = OUT_DIR / f"h{horizon:02d}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        written.append(path)
        max_start = 401 - horizon - 2
        assert all(s <= max_start for s in VALIDATION_STARTS), horizon
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
