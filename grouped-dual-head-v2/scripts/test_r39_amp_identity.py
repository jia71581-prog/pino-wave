#!/usr/bin/env python3
"""CUDA BF16 identity regression for the R39 HFS insertion."""

import json

import torch

import train_r39_hfs_tail_finetune as r39


def main() -> int:
    device = torch.device("cuda:0")
    torch.manual_seed(390828)
    base = r39.r29a.r26.TailSpectralResidualUNet(
        base_width=32, correction_cap=0.25
    ).to(device).eval()
    hfs = r39.HFSTailSpectralResidualUNet(
        base_width=32, correction_cap=0.25
    ).to(device).eval()
    hfs.load_state_dict(base.state_dict(), strict=True)
    value = torch.randn(
        1, r39.r29a.r25.INPUT_CHANNELS, 65, 65, device=device
    )
    active = torch.ones(1, device=device)
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        expected = base(value, active=active)
        actual = hfs(value, active=active)
    maximum = float((expected - actual).abs().max())
    print(
        json.dumps(
            {
                "max_abs_diff": maximum,
                "finite": bool(torch.isfinite(actual).all()),
                "output_dtype": str(actual.dtype),
            },
            sort_keys=True,
        )
    )
    return int(maximum != 0.0 or not bool(torch.isfinite(actual).all()))


if __name__ == "__main__":
    raise SystemExit(main())
