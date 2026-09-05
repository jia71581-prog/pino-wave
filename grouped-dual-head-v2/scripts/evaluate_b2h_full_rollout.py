#!/usr/bin/env python3
"""Leak-free full-time rollout audit for B2-H.

Only the first two stored frames initialize the physical state.  Every later
frame is autoregressive; no teacher frame resets the rollout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from saved_time_phase_operator_v4.b2h import PhysicalResidualPropagator
from saved_time_phase_operator_v4.b2h_training import SequenceWindowDataset
from saved_time_phase_operator_v4.streaming_metrics import (
    ExactWavefieldMetricAccumulator,
)


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@torch.no_grad()
def _evaluate(
    model: PhysicalResidualPropagator,
    *,
    h5,
    records: list[int],
    learned: bool,
    device: torch.device,
) -> dict[str, object]:
    saved_gate = model.gate.detach().clone()
    if not learned:
        model.gate.zero_()
    accumulator = ExactWavefieldMetricAccumulator(
        energy_floor_fraction=0.01,
        require_unique=True,
        stored_time_count=int(h5["wavefield"].shape[1]),
    )
    for record in records:
            wavefield = torch.from_numpy(
                np.asarray(h5["wavefield"][record], dtype=np.float32)
            ).to(device)
            velocity = torch.from_numpy(
                np.asarray(h5["velocity_mps"][record], dtype=np.float32)
            )[None, None].to(device)
            source_map = torch.from_numpy(
                np.asarray(h5["source_map"][record], dtype=np.float32)
            )[None, None].to(device)
            source_series = torch.from_numpy(
                np.asarray(h5["source_wavelet"][record, 1:-1], dtype=np.float32)
            )[None].to(device)
            source_parameters = torch.tensor(
                [[
                    h5["source_f0_hz"][record],
                    h5["source_t0_s"][record],
                    h5["source_amplitude"][record],
                ]],
                dtype=torch.float32,
                device=device,
            )
            predicted_tail = model(
                wavefield[0:1][None],
                wavefield[1:2][None],
                velocity,
                source_map,
                source_series,
                source_parameters=source_parameters,
                initial_time_s=torch.tensor(
                    [h5["time_s"][1]], dtype=torch.float32, device=device
                ),
            )[0, :, 0]
            if not torch.isfinite(predicted_tail).all():
                raise FloatingPointError(
                    f"non-finite full rollout for record {record}"
                )
            prediction = torch.cat((wavefield[:2], predicted_tail), dim=0)[None]
            target = wavefield[None]
            accumulator.update(
                prediction,
                target,
                families=[_decode(h5["medium_type"][record])],
                group_ids=[_decode(h5["group_id"][record])],
                sample_ids=[_decode(h5["sample_id"][record])],
                time_indices=torch.arange(
                    wavefield.shape[0], dtype=torch.long
                )[None],
                source_onset_indices=[
                    int(
                        round(
                            float(h5["source_t0_s"][record])
                            / float(h5.attrs["dt_output_s"])
                        )
                    )
                ],
            )
    model.gate.copy_(saved_gate)
    return accumulator.finalize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records-per-family", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if int(config["model"].get("memory_steps", 0)) != 0:
        raise ValueError("strict initial-pair audit currently applies to B2-H only")
    device = torch.device("cuda")
    model = PhysicalResidualPropagator(**config["model"]).to(device).eval()
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    selector = SequenceWindowDataset(
        config["dataset"],
        split="validation",
        rollout_steps=1,
        seed=int(config["seed"]) + 17,
        records_per_family=int(args.records_per_family),
        fixed_starts=(1,),
    )
    records = list(selector.records)
    with h5py.File(config["dataset"], "r") as h5:
        learned = _evaluate(
            model, h5=h5, records=records, learned=True, device=device
        )
        baseline = _evaluate(
            model, h5=h5, records=records, learned=False, device=device
        )
    report = {
        "scope": "strict_full_rollout_small_panel",
        "strict_goal_eligible": False,
        "teacher_resets_after_initial_pair": 0,
        "record_indices": records,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "learned": learned,
        "physical_baseline": baseline,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
