#!/usr/bin/env python3
"""R52 arm A: R39 HFS fine-tuning with a 10x base learning rate.

Identical to train_r39_hfs_tail_finetune except the base learning rate is
passed as 1e-5 by the launcher and HFS_LR_MULTIPLIER drops to 10 so the HFS
parameters stay at their audited 1e-4 rate.  Data, loss, DDP, and evaluation
paths are unchanged; sealed holdouts remain unopened.
"""

from __future__ import annotations

from pathlib import Path

import train_r39_hfs_tail_finetune as r39

SCRIPT_PATH = Path(__file__).resolve()


def main() -> int:
    r39.HFS_LR_MULTIPLIER = 10.0
    r39.SCRIPT_PATH = SCRIPT_PATH
    return r39.main()


if __name__ == "__main__":
    raise SystemExit(main())
